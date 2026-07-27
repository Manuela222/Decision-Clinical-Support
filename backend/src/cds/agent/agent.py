"""Phase 13: the agentic recommender.

Receives only the ClinicalState (+ whatever evidence tool calls retrieve
during the loop) — never raw DataFrames or a PatientTimeline dump; the only
thing resembling raw data is `MCPToolContext.patient_timeline`, which stays
server-side and is only ever touched through the `get_evidence_citations`
tool for specific already-surfaced IDs (see `cds.mcp_tools.context`).

SAFETY DESIGN DECISION, worth flagging explicitly: `hypertension_compatible`
/`hypertension_reasoning` on every recommended `MedicationRecommendation` are
NEVER taken from the LLM's own text. The orchestrator deterministically
calls `check_medication_compatibility` (Phase 12, tool 2) for every
candidate class the model proposes — reusing the result if the model
already called that tool during the loop, forcing the call itself
(logged as an explicit `HYPERTENSION_COMPATIBILITY_CHECK` reasoning step)
if it didn't. This guarantees the spec's "must explicitly reason about
hypertension compatibility for every candidate medication class, and this
reasoning step must appear in the trace" requirement holds regardless of
what the model actually decided to do — a prompt asking nicely is not
enough for a safety-critical field.

No hidden chain-of-thought: reasoning_trace only ever contains tool calls,
their (JSON-serializable) results, and the model's own final-answer content
— never a free-form "thinking" blob.
"""
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..medications import default_dictionary
from ..mcp_tools import tools as mcp_tools
from ..mcp_tools.context import MCPToolContext
from ..safety import check_recommendation_safety
from ..schemas import (
    ClinicalState,
    MedicationAction,
    MedicationRecommendation,
    ReasoningStep,
    ReasoningStepType,
    RecommendationMethod,
    RecommendationResult,
)
from .dispatch import dispatch_tool_call
from .providers import LLMProvider
from .tool_schemas import build_openai_tool_schemas

# Raised from an initial 10 after live testing against the real OpenAI API (not just the
# scripted mock) showed a real model can legitimately need more turns than a mock ever does
# -- e.g. one tool call per turn instead of batching, or extra exploratory calls for
# thin-evidence patients.
MAX_TOOL_CALLS = 25

SYSTEM_PROMPT = """You are a clinical decision support assistant. Given a patient's \
ClinicalState (JSON), recommend discharge medication classes for their admission \
condition, taking their chronic hypertension into account as a safety constraint \
-- hypertension itself is never something to recommend treatment for here.

You may call the provided tools as needed to gather evidence (similar past patients, \
medication compatibility, lab abnormalities, drug interactions, evidence citations) \
before answering. Do not recommend a medication class you have no basis for.

Every "medication_class" you output -- in tool calls and in your final answer -- MUST be \
copied EXACTLY, character-for-character, from this fixed list. Never invent a class name, \
pluralize one, abbreviate one, or use a synonym that is not in this list:
{vocabulary}

When ready, respond with ONLY a JSON object, no other text, of the form:
{{"recommendations": [
  {{"medication_class": "...", "action": "start|continue|stop|adjust|avoid", \
"rationale": "...", "confidence": 0.0-1.0, "evidence_ids": ["..."]}}
]}}

`rationale` must be a short, clinician-facing sentence stating the conclusion and \
why -- never your internal step-by-step reasoning."""


class AgentError(Exception):
    """Raised when the agent's final answer can't be parsed/used, or it
    exceeds the tool-call budget without producing one."""


def _known_medication_classes() -> List[str]:
    return sorted(set(default_dictionary().classes.values()))


def _build_initial_messages(clinical_state: ClinicalState, known_classes: List[str]) -> List[Dict[str, Any]]:
    system_prompt = SYSTEM_PROMPT.format(vocabulary=", ".join(known_classes))
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": clinical_state.model_dump_json()},
    ]


def _parse_final_answer(content: str) -> List[Dict[str, Any]]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise AgentError(f"Agent's final answer was not valid JSON ({e}): {content!r}") from e
    if not isinstance(data, dict) or not isinstance(data.get("recommendations"), list):
        raise AgentError(f"Agent's final answer JSON is missing a 'recommendations' list: {data!r}")
    return data["recommendations"]


def recommend_agent(
    clinical_state: ClinicalState,
    ctx: MCPToolContext,
    llm_provider: LLMProvider,
    model_version: str = "agent-v1",
    max_tool_calls: int = MAX_TOOL_CALLS,
    generated_at: Optional[datetime] = None,
) -> RecommendationResult:
    known_classes = _known_medication_classes()
    known_classes_set = set(known_classes)
    tool_schemas = build_openai_tool_schemas()
    messages = _build_initial_messages(clinical_state, known_classes)

    reasoning_steps: List[ReasoningStep] = []
    step_index = 0
    # medication_class -> MedicationCompatibilityResult.model_dump(mode="json"),
    # captured from any check_medication_compatibility calls made during the loop
    compatibility_checked: Dict[str, Dict[str, Any]] = {}

    n_tool_calls = 0
    raw_recommendations: List[Dict[str, Any]] = []

    while True:
        turn = llm_provider.get_next_turn(messages, tool_schemas)

        if not turn.tool_calls:
            if not turn.content:
                raise AgentError("Agent returned neither tool calls nor content — cannot produce a recommendation.")
            raw_recommendations = _parse_final_answer(turn.content)
            reasoning_steps.append(
                ReasoningStep(
                    step_index=step_index,
                    step_type=ReasoningStepType.CONCLUSION,
                    description=f"Proposed {len(raw_recommendations)} candidate medication class(es).",
                    timestamp=datetime.now(timezone.utc),
                )
            )
            break

        n_tool_calls += len(turn.tool_calls)
        if n_tool_calls > max_tool_calls:
            raise AgentError(f"Agent exceeded max_tool_calls={max_tool_calls} without producing a final answer.")

        messages.append(
            {
                "role": "assistant",
                "content": turn.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                    for tc in turn.tool_calls
                ],
            }
        )

        for tc in turn.tool_calls:
            try:
                result = dispatch_tool_call(ctx, tc.name, tc.arguments)
                tool_error = None
            except Exception as e:  # noqa: BLE001 - surfaced to the model as a tool error, not raised
                result = {"error": str(e)}
                tool_error = str(e)

            if tc.name == "check_medication_compatibility" and tool_error is None:
                compatibility_checked[tc.arguments.get("medication_class")] = result

            step_index += 1
            reasoning_steps.append(
                ReasoningStep(
                    step_index=step_index,
                    step_type=(
                        ReasoningStepType.RETRIEVAL
                        if tc.name == "search_similar_patient_profiles"
                        else ReasoningStepType.TOOL_CALL
                    ),
                    description=f"Called {tc.name}({tc.arguments})"
                    + (f" -> error: {tool_error}" if tool_error else ""),
                    tool_name=tc.name,
                    tool_input=dict(tc.arguments),
                    tool_output=result if isinstance(result, dict) else {"result": result},
                    timestamp=datetime.now(timezone.utc),
                )
            )

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})

    recommendations: List[MedicationRecommendation] = []
    for raw in raw_recommendations:
        medication_class = raw.get("medication_class")
        if not medication_class:
            continue

        if medication_class not in known_classes_set:
            # Safety backstop: check_hypertension_compatibility only recognizes exact
            # vocabulary strings, so a hallucinated/malformed class name (e.g. a plural
            # or synonym not in the fixed dictionary) would otherwise fall through its
            # unsafe-class check and be reported as "compatible" by default. Reject it
            # outright rather than surface an unvalidated recommendation.
            step_index += 1
            reasoning_steps.append(
                ReasoningStep(
                    step_index=step_index,
                    step_type=ReasoningStepType.REASONING,
                    description=(
                        f"Rejected candidate medication_class '{medication_class}': not part of the "
                        "fixed medication-class vocabulary, cannot be safety-checked or recommended."
                    ),
                    timestamp=datetime.now(timezone.utc),
                )
            )
            continue

        if medication_class not in compatibility_checked:
            compat_result = mcp_tools.check_medication_compatibility(ctx, medication_class=medication_class)
            compat_dump = compat_result.model_dump(mode="json")
            compatibility_checked[medication_class] = compat_dump
            step_index += 1
            reasoning_steps.append(
                ReasoningStep(
                    step_index=step_index,
                    step_type=ReasoningStepType.HYPERTENSION_COMPATIBILITY_CHECK,
                    description=(
                        f"Hypertension-compatibility check for '{medication_class}' (run automatically — the "
                        f"agent did not call this tool for this class itself): {compat_dump['hypertension_reasoning']}"
                    ),
                    tool_name="check_medication_compatibility",
                    tool_input={"medication_class": medication_class},
                    tool_output=compat_dump,
                    timestamp=datetime.now(timezone.utc),
                )
            )
        else:
            compat_dump = compatibility_checked[medication_class]
            step_index += 1
            reasoning_steps.append(
                ReasoningStep(
                    step_index=step_index,
                    step_type=ReasoningStepType.HYPERTENSION_COMPATIBILITY_CHECK,
                    description=(
                        f"Hypertension-compatibility check for '{medication_class}': "
                        f"{compat_dump['hypertension_reasoning']}"
                    ),
                    tool_name="check_medication_compatibility",
                    tool_input={"medication_class": medication_class},
                    tool_output=compat_dump,
                    timestamp=datetime.now(timezone.utc),
                )
            )

        recommendations.append(
            MedicationRecommendation(
                medication_class=medication_class,
                action=MedicationAction(raw.get("action", "start")),
                rationale=raw.get("rationale", ""),
                evidence_ids=list(raw.get("evidence_ids", [])),
                confidence=float(raw.get("confidence", 0.5)),
                hypertension_compatible=compat_dump["hypertension_compatible"],
                hypertension_reasoning=compat_dump["hypertension_reasoning"],
            )
        )

    safety_warnings = check_recommendation_safety(clinical_state, recommendations)

    return RecommendationResult(
        subject_id=clinical_state.subject_id,
        hadm_id=clinical_state.hadm_id,
        method=RecommendationMethod.AGENT,
        recommended_medications=recommendations,
        safety_warnings=safety_warnings,
        reasoning_trace=reasoning_steps,
        model_version=model_version,
        generated_at=generated_at or datetime.now(timezone.utc),
    )
