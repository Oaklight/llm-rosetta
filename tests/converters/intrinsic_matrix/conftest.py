"""Shared fixtures and tables for the intrinsic-handling characterization matrix.

Phase 1 of the intrinsic refactor: these tests pin *current* behaviour per
converter × kind × path.  Nothing here changes production code.

Two axes describe the matrix:

* **converter** — the five dialects that carry IR intrinsic parts:
  ``chat_completions``, ``messages``, ``open_responses``,
  ``google_generate``, ``google_interactions``.
* **kind** — an intrinsic kind in the converter's *own* vocabulary (it has a
  native provider form), a *foreign* kind (a sibling dialect's vocabulary),
  or an *unknown* kind no dialect renders.

Every outcome is classified as one of:

``native``
    written/read in the dialect's own intrinsic (server-tool) form.
``degraded``
    written as an ordinary client function call/result with a name.
``dropped``
    not emitted at all.
``deferred``
    not emitted at the event that carried it, but written later (google_generate
    flushes intrinsic calls at the finish event).
``held``
    intentionally empty: the payload rides a later event (open_responses moves
    a server call's arguments onto its ``output_item.done``).
``leaked``
    the opening event was dropped but a later event for the same call is still
    emitted (an inconsistent half-write).
"""

from __future__ import annotations

import json
from typing import Any
from collections.abc import Callable

import pytest

from llm_rosetta.converters.base.tools.intrinsic import (
    make_intrinsic_tool_call,
    make_intrinsic_tool_definition,
    make_intrinsic_tool_result,
)
from llm_rosetta.converters.chat_completions import ChatCompletionsConverter
from llm_rosetta.converters.google_generate import GoogleGenerateConverter
from llm_rosetta.converters.google_interactions import GoogleInteractionsConverter
from llm_rosetta.converters.messages import MessagesConverter
from llm_rosetta.converters.open_responses import OpenAIResponsesConverter

# ---------------------------------------------------------------------------
# Converter registry
# ---------------------------------------------------------------------------

CONVERTERS: dict[str, Callable[[], Any]] = {
    "chat_completions": ChatCompletionsConverter,
    "messages": MessagesConverter,
    "open_responses": OpenAIResponsesConverter,
    "google_generate": GoogleGenerateConverter,
    "google_interactions": GoogleInteractionsConverter,
}

#: The intrinsic kinds each dialect renders natively (its "own vocabulary").
OWN_KINDS: dict[str, tuple[str, ...]] = {
    "chat_completions": (),
    "messages": ("web_search", "code_execution"),
    "open_responses": ("web_search", "code_interpreter"),
    "google_generate": ("code_execution", "google_search"),
    "google_interactions": ("google_search", "code_execution"),
}

#: Kinds that are valid in a *sibling* dialect but foreign to this one.
FOREIGN_KIND: dict[str, str] = {
    "chat_completions": "web_search",
    "messages": "google_search",
    "open_responses": "google_search",
    "google_generate": "web_search",
    "google_interactions": "web_search",
}

#: A kind no dialect knows.
UNKNOWN_KIND = "special_search"

#: A per-kind ``tool_input`` that satisfies the dialect's native-replay
#: requirements (open_responses needs ``container_id`` for a code interpreter).
KIND_INPUT: dict[str, dict[str, Any]] = {
    "web_search": {"queries": ["mars rover"]},
    "code_execution": {"code": "print(42)", "language": "PYTHON"},
    "code_interpreter": {"code": "print(42)", "container_id": "cntr_1"},
    "google_search": {"queries": ["mars rover"]},
    UNKNOWN_KIND: {"query": "mars rover"},
}


def kind_input(kind: str) -> dict[str, Any]:
    return dict(KIND_INPUT.get(kind, {"query": "x"}))


# ---------------------------------------------------------------------------
# IR builders
# ---------------------------------------------------------------------------


def ir_history(kind: str, *, with_result: bool = True, tool_input: dict | None = None):
    """An IR request whose assistant turn carries one intrinsic call."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {
            "role": "assistant",
            "content": [
                make_intrinsic_tool_call(
                    tool_call_id="s1",
                    tool_name=kind,
                    tool_input=tool_input
                    if tool_input is not None
                    else kind_input(kind),
                    intrinsic_kind=kind,
                )
            ],
        },
    ]
    if with_result:
        messages.append(
            {
                "role": "tool",
                "content": [
                    make_intrinsic_tool_result(
                        tool_call_id="s1", result="RESULT", intrinsic_kind=kind
                    )
                ],
            }
        )
    return {"model": "m", "messages": messages}


def ir_response(kind: str, *, with_result: bool = True, tool_input: dict | None = None):
    """An IR response whose assistant turn carries one intrinsic call/result."""
    content: list[dict[str, Any]] = [
        make_intrinsic_tool_call(
            tool_call_id="c1",
            tool_name=kind,
            tool_input=tool_input if tool_input is not None else kind_input(kind),
            intrinsic_kind=kind,
        )
    ]
    if with_result:
        content.append(
            make_intrinsic_tool_result(
                tool_call_id="c1", result="RESULT", intrinsic_kind=kind
            )
        )
    content.append({"type": "text", "text": "done"})
    return {
        "id": "r1",
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": {"reason": "tool_calls"},
                "message": {"role": "assistant", "content": content},
            }
        ],
    }


def intrinsic_definition(kind: str) -> dict[str, Any]:
    return make_intrinsic_tool_definition(kind)


# ---------------------------------------------------------------------------
# Outcome observers — one per path, one generic vocabulary
# ---------------------------------------------------------------------------


def definition_outcome(converter_name: str, kind: str) -> str:
    """Classify ``ir_tool_definition_to_p`` for an intrinsic definition."""
    out = CONVERTERS[converter_name]().tool_ops.ir_tool_definition_to_p(
        intrinsic_definition(kind)
    )
    return "native" if out else "dropped"


_NATIVE_CALL_MARKERS: dict[str, tuple[str, ...]] = {
    "chat_completions": (),
    "messages": ('"server_tool_use"',),
    "open_responses": (
        '"web_search_call"',
        '"code_interpreter_call"',
        '"file_search_call"',
        '"shell_call"',
        '"computer_call"',
    ),
    "google_generate": ('"executableCode"',),
    "google_interactions": (
        '"google_search_call"',
        '"code_execution_call"',
        '"google_maps_call"',
        '"file_search_call"',
        '"retrieval_call"',
        '"url_context_call"',
    ),
}

_DEGRADED_CALL_MARKERS: dict[str, tuple[str, ...]] = {
    "chat_completions": ('"tool_calls"',),
    "messages": (),
    "open_responses": ('"function_call"',),
    "google_generate": ('"functionCall"',),
    "google_interactions": ('"function_call"',),
}


def _classify_call(converter_name: str, payload: dict[str, Any]) -> str:
    text = json.dumps(payload, default=str)
    if any(m in text for m in _NATIVE_CALL_MARKERS[converter_name]):
        return "native"
    if any(m in text for m in _DEGRADED_CALL_MARKERS[converter_name]):
        return "degraded"
    return "dropped"


def history_call_outcome(converter_name: str, kind: str) -> str:
    """Classify the history-leg write of one intrinsic call."""
    body, _ = CONVERTERS[converter_name]().request_to_provider(ir_history(kind))
    return _classify_call(converter_name, body)


def _result_written(payload: dict[str, Any]) -> bool:
    """Whether the intrinsic result's payload (``"RESULT"``) reached the output."""
    return '"RESULT"' in json.dumps(payload, default=str)


def history_result_outcome(converter_name: str, kind: str) -> str:
    """Classify the history-leg write of one intrinsic result.

    ``degraded`` marks the dialects whose history result is an ordinary client
    result (chat's ``role:"tool"`` message, messages' plain ``tool_result``),
    where the kind is lost.
    """
    body, _ = CONVERTERS[converter_name]().request_to_provider(ir_history(kind))
    text = json.dumps(body, default=str)
    if not _result_written(body):
        return "dropped"
    if converter_name in ("chat_completions", "messages"):
        return "degraded"
    if converter_name == "open_responses":
        # A degraded call pairs with a client function_call_output; a native
        # server call accepts no output item, so its result is dropped.
        return "degraded" if '"function_call_output"' in text else "native"
    return "native"


def response_call_outcome(converter_name: str, kind: str) -> str:
    """Classify the non-streaming response write of one intrinsic call."""
    out = CONVERTERS[converter_name]().response_to_provider(ir_response(kind))
    return _classify_call(converter_name, out)


def response_result_outcome(converter_name: str, kind: str) -> str:
    """Classify the non-streaming response write of one intrinsic result."""
    out = CONVERTERS[converter_name]().response_to_provider(ir_response(kind))
    if not _result_written(out):
        return "dropped"
    if converter_name == "chat_completions":
        return "degraded"
    return "native"


# ---------------------------------------------------------------------------
# Parametrisation helper
# ---------------------------------------------------------------------------


def matrix_cases(table: dict[str, dict[str, str]]):
    """Flatten ``{converter: {kind: expected}}`` into pytest parameters."""
    return [
        pytest.param(converter, kind, expected, id=f"{converter}-{kind}")
        for converter, kinds in table.items()
        for kind, expected in kinds.items()
    ]
