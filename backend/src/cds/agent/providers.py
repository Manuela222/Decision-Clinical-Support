"""LLM provider abstraction for the agentic recommender (Phase 13).

`OpenAIProvider` is the real, OpenAI-API-backed implementation — the model
is configurable via the `OPENAI_MODEL` environment variable (default
`DEFAULT_OPENAI_MODEL`), and the API key is read from `OPENAI_API_KEY` by
the OpenAI SDK itself; it is never hardcoded or passed as a literal here.
The `openai` package is imported lazily inside `__init__` so the rest of
this module (and `MockLLMProvider`) stays importable/testable without it.

`OpenAIProvider` is NOT exercised by the test suite — there is no live
network access or API key in this environment. `MockLLMProvider` provides
scripted, deterministic turns so `cds.agent.agent`'s orchestration logic
(the tool-calling loop, the forced hypertension-compatibility check, the
reasoning trace) is fully tested without ever calling out to OpenAI.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMTurn:
    content: Optional[str]
    tool_calls: List[ToolCallRequest] = field(default_factory=list)


class LLMProvider(Protocol):
    def get_next_turn(self, messages: List[Dict[str, Any]], tool_schemas: List[Dict[str, Any]]) -> LLMTurn: ...


class OpenAIProvider:
    def __init__(self, model: Optional[str] = None):
        import openai  # lazy: keep this module importable without `openai` installed for mock-only usage

        self.model = model or os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        self._client = openai.OpenAI()  # reads OPENAI_API_KEY from the environment

    def get_next_turn(self, messages: List[Dict[str, Any]], tool_schemas: List[Dict[str, Any]]) -> LLMTurn:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tool_schemas or None,
        )
        message = response.choices[0].message
        tool_calls = [
            ToolCallRequest(id=tc.id, name=tc.function.name, arguments=json.loads(tc.function.arguments))
            for tc in (message.tool_calls or [])
        ]
        return LLMTurn(content=message.content, tool_calls=tool_calls)


class MockLLMProvider:
    """Deterministic, scripted provider for tests. Pass a fixed sequence of
    `LLMTurn`s; each call to `get_next_turn` returns the next one in order,
    ignoring the actual messages/tool_schemas passed in (these tests exercise
    the agent's orchestration logic, not language behavior)."""

    def __init__(self, scripted_turns: List[LLMTurn]):
        self._turns = list(scripted_turns)
        self._index = 0

    def get_next_turn(self, messages: List[Dict[str, Any]], tool_schemas: List[Dict[str, Any]]) -> LLMTurn:
        if self._index >= len(self._turns):
            raise AssertionError("MockLLMProvider ran out of scripted turns — the agent looped more than expected.")
        turn = self._turns[self._index]
        self._index += 1
        return turn
