"""Phase 13: agentic recommender tests, using MockLLMProvider (scripted,
deterministic turns) -- no live OpenAI calls anywhere in this suite."""
import json
from datetime import datetime, timezone

import pytest

from cds.agent import AgentError, LLMTurn, MockLLMProvider, ToolCallRequest, recommend_agent
from cds.mcp_tools import MCPToolContext
from cds.retrieval import build_patient_profile_index
from cds.schemas import (
    ClinicalState,
    Gender,
    HypertensionStatus,
    LabValueSummary,
    MedicationMention,
    PatientTimeline,
    RecommendationMethod,
    ReasoningStepType,
    SafetyCategory,
)


def _clinical_state(idx=999, reason="Cardiac arrhythmias", recent_labs=None, current_medications=None):
    return ClinicalState(
        subject_id=idx,
        hadm_id=1000 + idx,
        age=65,
        gender=Gender.F,
        admission_reason=reason,
        hypertension_status=HypertensionStatus.CONFIRMED_CHRONIC,
        hypertension_evidence=[],
        recent_blood_pressure_labs=[],
        current_antihypertensive_medications=[],
        active_conditions=[reason, "Hypertension"],
        chronic_conditions=[reason, "Hypertension"],
        recent_labs=recent_labs or [],
        current_medications=current_medications or [],
        prior_medications=[],
        relevant_note_sections=[],
        missing_information=[],
        generated_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def ctx():
    train_states = [_clinical_state(1, "Cardiac arrhythmias"), _clinical_state(2, "Cardiac arrhythmias")]
    train_labels = [["antiarrhythmic"], ["antiarrhythmic", "anticoagulant"]]
    index = build_patient_profile_index(train_states, train_labels)
    timeline = PatientTimeline(subject_id=999, hadm_id=1999, events=[], generated_at=datetime.now(timezone.utc))
    return MCPToolContext(clinical_state=_clinical_state(), patient_timeline=timeline, patient_profile_index=index)


def _final_answer_turn(recommendations):
    return LLMTurn(content=json.dumps({"recommendations": recommendations}), tool_calls=[])


# --- basic happy paths -------------------------------------------------------

def test_agent_direct_final_answer_forces_compatibility_check(ctx):
    provider = MockLLMProvider(
        [_final_answer_turn([{"medication_class": "antiarrhythmic", "action": "start", "rationale": "Rate control.", "confidence": 0.8}])]
    )
    result = recommend_agent(ctx.clinical_state, ctx, provider)

    assert result.method == RecommendationMethod.AGENT
    assert result.subject_id == ctx.clinical_state.subject_id
    assert result.hadm_id == ctx.clinical_state.hadm_id
    assert len(result.recommended_medications) == 1
    rec = result.recommended_medications[0]
    assert rec.medication_class == "antiarrhythmic"
    assert rec.hypertension_reasoning  # must be populated

    compat_steps = [s for s in result.reasoning_trace if s.step_type == ReasoningStepType.HYPERTENSION_COMPATIBILITY_CHECK]
    assert len(compat_steps) == 1
    assert "automatically" in compat_steps[0].description


def test_agent_tool_call_then_final_answer(ctx):
    provider = MockLLMProvider(
        [
            LLMTurn(
                content=None,
                tool_calls=[ToolCallRequest(id="call_1", name="search_similar_patient_profiles", arguments={"top_k": 2})],
            ),
            _final_answer_turn(
                [{"medication_class": "antiarrhythmic", "action": "continue", "rationale": "Similar patients received this.", "confidence": 0.7}]
            ),
        ]
    )
    result = recommend_agent(ctx.clinical_state, ctx, provider)

    retrieval_steps = [s for s in result.reasoning_trace if s.step_type == ReasoningStepType.RETRIEVAL]
    assert len(retrieval_steps) == 1
    assert retrieval_steps[0].tool_name == "search_similar_patient_profiles"
    assert retrieval_steps[0].tool_output is not None


def test_agent_reuses_llm_provided_compatibility_check_not_forced(ctx):
    provider = MockLLMProvider(
        [
            LLMTurn(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call_1", name="check_medication_compatibility", arguments={"medication_class": "nsaid"})
                ],
            ),
            _final_answer_turn(
                [{"medication_class": "nsaid", "action": "start", "rationale": "Pain control.", "confidence": 0.6}]
            ),
        ]
    )
    result = recommend_agent(ctx.clinical_state, ctx, provider)

    compat_steps = [s for s in result.reasoning_trace if s.step_type == ReasoningStepType.HYPERTENSION_COMPATIBILITY_CHECK]
    assert len(compat_steps) == 1
    assert "automatically" not in compat_steps[0].description  # the agent already checked it itself

    rec = result.recommended_medications[0]
    assert rec.hypertension_compatible is False  # nsaid is unsafe for confirmed_chronic hypertension


def test_agent_flags_hypertension_unsafe_in_safety_warnings(ctx):
    provider = MockLLMProvider(
        [_final_answer_turn([{"medication_class": "nsaid", "action": "start", "rationale": "Pain control.", "confidence": 0.6}])]
    )
    result = recommend_agent(ctx.clinical_state, ctx, provider)
    assert any(w.category == SafetyCategory.HYPERTENSION_UNSAFE for w in result.safety_warnings)


# --- error handling -----------------------------------------------------------

def test_agent_invalid_json_final_answer_raises():
    provider = MockLLMProvider([LLMTurn(content="not json at all", tool_calls=[])])
    ctx_local = MCPToolContext(
        clinical_state=_clinical_state(),
        patient_timeline=PatientTimeline(subject_id=999, hadm_id=1999, events=[], generated_at=datetime.now(timezone.utc)),
        patient_profile_index=build_patient_profile_index([_clinical_state(1, "Cardiac arrhythmias")], [["antiarrhythmic"]]),
    )
    with pytest.raises(AgentError, match="not valid JSON"):
        recommend_agent(_clinical_state(), ctx_local, provider)


def test_agent_missing_recommendations_key_raises(ctx):
    provider = MockLLMProvider([LLMTurn(content=json.dumps({"foo": "bar"}), tool_calls=[])])
    with pytest.raises(AgentError, match="recommendations"):
        recommend_agent(ctx.clinical_state, ctx, provider)


def test_agent_no_tool_calls_and_no_content_raises(ctx):
    provider = MockLLMProvider([LLMTurn(content=None, tool_calls=[])])
    with pytest.raises(AgentError, match="cannot produce a recommendation"):
        recommend_agent(ctx.clinical_state, ctx, provider)


def test_agent_exceeds_max_tool_calls_raises(ctx):
    # 3 tool-call turns of 5 calls each = 15 total, budget is 10
    tool_call_turn = LLMTurn(
        content=None,
        tool_calls=[
            ToolCallRequest(id=f"call_{i}", name="search_similar_patient_profiles", arguments={"top_k": 1})
            for i in range(5)
        ],
    )
    provider = MockLLMProvider([tool_call_turn, tool_call_turn, tool_call_turn])
    with pytest.raises(AgentError, match="max_tool_calls"):
        recommend_agent(ctx.clinical_state, ctx, provider, max_tool_calls=10)


def test_agent_unknown_tool_call_recorded_as_error_not_crash(ctx):
    provider = MockLLMProvider(
        [
            LLMTurn(content=None, tool_calls=[ToolCallRequest(id="call_1", name="not_a_real_tool", arguments={})]),
            _final_answer_turn([{"medication_class": "statin", "action": "start", "rationale": "Lipid control.", "confidence": 0.5}]),
        ]
    )
    result = recommend_agent(ctx.clinical_state, ctx, provider)
    tool_steps = [s for s in result.reasoning_trace if s.tool_name == "not_a_real_tool"]
    assert len(tool_steps) == 1
    assert "error" in tool_steps[0].tool_output
    # the agent still produced a usable final result despite the bad tool call
    assert len(result.recommended_medications) == 1


# --- no hidden chain-of-thought ------------------------------------------------

def test_reasoning_trace_has_no_hidden_fields_beyond_schema(ctx):
    provider = MockLLMProvider(
        [_final_answer_turn([{"medication_class": "statin", "action": "start", "rationale": "Lipid control.", "confidence": 0.5}])]
    )
    result = recommend_agent(ctx.clinical_state, ctx, provider)
    # every step round-trips through JSON cleanly -- i.e. it's fully serializable, UI-displayable (XAI requirement)
    for step in result.reasoning_trace:
        json.loads(step.model_dump_json())


# --- defensive handling of real live-model failure modes (Section 4.3) --------

def test_agent_rejects_medication_class_outside_fixed_vocabulary(ctx):
    provider = MockLLMProvider(
        [
            _final_answer_turn(
                [
                    {"medication_class": "statin", "action": "start", "rationale": "Lipid control.", "confidence": 0.5},
                    {"medication_class": "made-up-class", "action": "start", "rationale": "Hallucinated.", "confidence": 0.9},
                ]
            )
        ]
    )
    result = recommend_agent(ctx.clinical_state, ctx, provider)

    classes = [r.medication_class for r in result.recommended_medications]
    assert classes == ["statin"]  # the hallucinated one was dropped, not silently marked "compatible"
    rejected_steps = [s for s in result.reasoning_trace if "made-up-class" in s.description and "not part of the fixed" in s.description]
    assert len(rejected_steps) == 1


def test_agent_strips_markdown_code_fence_from_final_answer(ctx):
    # Observed live: some models wrap the JSON answer in a ```json ... ``` fence
    # despite the system prompt asking for "ONLY a JSON object, no other text".
    fenced_content = "```json\n" + json.dumps(
        {"recommendations": [{"medication_class": "statin", "action": "start", "rationale": "Lipid control.", "confidence": 0.5}]}
    ) + "\n```"
    provider = MockLLMProvider([LLMTurn(content=fenced_content, tool_calls=[])])

    result = recommend_agent(ctx.clinical_state, ctx, provider)

    assert [r.medication_class for r in result.recommended_medications] == ["statin"]


def test_agent_rejects_invalid_action_without_crashing_whole_admission(ctx):
    # Observed live: the model can propose an action outside the fixed enum
    # (e.g. "consider") despite the system prompt listing exactly five values.
    provider = MockLLMProvider(
        [
            _final_answer_turn(
                [
                    {"medication_class": "statin", "action": "start", "rationale": "Lipid control.", "confidence": 0.5},
                    {"medication_class": "nsaid", "action": "consider", "rationale": "Maybe.", "confidence": 0.4},
                ]
            )
        ]
    )
    result = recommend_agent(ctx.clinical_state, ctx, provider)

    classes = [r.medication_class for r in result.recommended_medications]
    assert classes == ["statin"]  # the invalid-action one was dropped, the rest of the admission still succeeded
    rejected_steps = [s for s in result.reasoning_trace if "nsaid" in s.description and "not one of start/continue/stop/adjust/avoid" in s.description]
    assert len(rejected_steps) == 1
