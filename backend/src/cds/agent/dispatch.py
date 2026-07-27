"""Dispatch a named tool call (as the LLM requests it) to the matching
Phase 12 tool function, and reduce the result to JSON-serializable data
for the conversation."""
from typing import Any, Dict

from ..mcp_tools import tools as mcp_tools
from ..mcp_tools.context import MCPToolContext

_TOOL_FUNCTIONS = {
    "search_similar_patient_profiles": mcp_tools.search_similar_patient_profiles,
    "check_medication_compatibility": mcp_tools.check_medication_compatibility,
    "lookup_lab_abnormalities": mcp_tools.lookup_lab_abnormalities,
    "check_drug_interactions": mcp_tools.check_drug_interactions,
    "get_evidence_citations": mcp_tools.get_evidence_citations,
}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def dispatch_tool_call(ctx: MCPToolContext, name: str, arguments: Dict[str, Any]) -> Any:
    if name not in _TOOL_FUNCTIONS:
        raise ValueError(f"Unknown tool '{name}' — the agent may only call the fixed Phase 12 tool list.")
    result = _TOOL_FUNCTIONS[name](ctx, **arguments)
    return _to_jsonable(result)
