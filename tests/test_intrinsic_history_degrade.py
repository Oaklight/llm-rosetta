"""Tests for ``degrade_intrinsic_history`` — the intrinsic history pass.

An intrinsic (provider-executed) call in message history is written in the
shape the *target* can replay.  When the target has no part for the kind, the
pass rewrites the call and every result answering it into an ordinary function
pair.  These are the pass's own unit tests; the pipeline tests in
``tests/converters/intrinsic_matrix/test_pipeline_intrinsic.py`` and
``tests/test_intrinsic_pipeline_*.py`` cover it end to end.

Replaces the tests of the removed deprecated ``strip_intrinsic_tools``: the
assertions that are still meaningful (a no-op where nothing is intrinsic, and
the rewrite of intrinsic history) are retargeted here.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from llm_rosetta.capabilities import degrade_intrinsic_history
from llm_rosetta.converters.chat_completions import ChatCompletionsConverter
from llm_rosetta.converters.google_generate import GoogleGenerateConverter
from llm_rosetta.converters.messages import MessagesConverter
from llm_rosetta.converters.open_responses import OpenAIResponsesConverter


def _chat() -> Any:
    return ChatCompletionsConverter().tool_ops


def _messages() -> Any:
    return MessagesConverter().tool_ops


def _intrinsic_call(
    kind: str, call_id: str = "s1", tool_input: dict | None = None
) -> dict[str, Any]:
    return {
        "type": "tool_call",
        "tool_call_id": call_id,
        "tool_name": kind,
        "tool_input": tool_input if tool_input is not None else {"queries": ["q"]},
        "tool_type": "intrinsic",
        "intrinsic_kind": kind,
    }


def _intrinsic_result(kind: str, call_id: str = "s1") -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_call_id": call_id,
        "result": "R",
        "tool_type": "intrinsic",
        "intrinsic_kind": kind,
    }


def _request(
    call: dict[str, Any],
    *,
    result_kind: str = "",
    result_in_tool_message: bool = True,
    tools: list[Any] | None = None,
) -> dict[str, Any]:
    """A request whose assistant turn carries *call*, answered by a result."""
    if not result_kind:
        result_kind = call.get("intrinsic_kind", "")
    request: dict[str, Any] = {
        "model": "m",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [call]},
        ],
    }
    result = _intrinsic_result(result_kind)
    if result_in_tool_message:
        request["messages"].append({"role": "tool", "content": [result]})
    else:
        request["messages"][1]["content"].append(result)
    if tools is not None:
        request["tools"] = tools
    return request


def _text(request: dict[str, Any]) -> str:
    return json.dumps(request, default=str)


# ---------------------------------------------------------------------------
# No-ops
# ---------------------------------------------------------------------------


def test_noop_without_intrinsic_history():
    """A request with no intrinsic parts is returned unchanged (same object)."""
    request = {
        "tools": [
            {"type": "function", "name": "get_weather", "parameters": {}},
        ],
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call_id": "c1",
                        "tool_name": "get_weather",
                        "tool_input": {},
                        "tool_type": "function",
                    }
                ],
            },
            {"role": "tool", "content": [_intrinsic_result("web_search", "c1")]},
        ],
    }
    # The result is intrinsic but its call is not, so nothing is paired.
    result = degrade_intrinsic_history(request, _chat())
    assert result is request


def test_noop_when_the_target_renders_the_kind():
    """A kind the target renders natively is left exactly as it was."""
    request = _request(_intrinsic_call("web_search"))
    snapshot = copy.deepcopy(request)
    result = degrade_intrinsic_history(request, _messages())
    assert result is request
    assert result == snapshot


def test_empty_messages_noop():
    request: dict[str, Any] = {"model": "m", "messages": []}
    assert degrade_intrinsic_history(request, _chat()) is request


# ---------------------------------------------------------------------------
# The degrade
# ---------------------------------------------------------------------------


def test_degrades_a_call_and_its_tool_message_result_pair():
    request = _request(_intrinsic_call("web_search"))
    warnings: list[str] = []

    out = degrade_intrinsic_history(request, _chat(), warnings=warnings)

    call = out["messages"][1]["content"][0]
    result = out["messages"][2]["content"][0]
    for part in (call, result):
        assert part["tool_type"] == "function"
        assert "intrinsic_kind" not in part
        assert part["provider_metadata"]["_downgraded_from"] == {
            "type": "intrinsic",
            "kind": "web_search",
        }
    assert call["tool_name"] == "web_search"
    assert call["tool_call_id"] == "s1"
    assert call["tool_input"] == {"queries": ["q"]}
    assert result["tool_call_id"] == "s1"
    assert result["result"] == "R"
    assert "tool_name" not in result  # a result carries no name of its own
    assert len(warnings) == 1
    assert "web_search" in warnings[0]


def test_degrades_a_result_inline_in_the_assistant_turn():
    request = _request(_intrinsic_call("web_search"), result_in_tool_message=False)
    out = degrade_intrinsic_history(request, _chat())
    content = out["messages"][1]["content"]
    assert [p["tool_type"] for p in content] == ["function", "function"]


def test_degrades_an_unknown_kind():
    request = _request(_intrinsic_call("special_search"))
    out = degrade_intrinsic_history(request, _chat())
    assert out["messages"][1]["content"][0]["tool_name"] == "special_search"


def test_degrades_a_kindless_part():
    """A missing kind is unknown, and unknown is not renderable."""
    call = _intrinsic_call("web_search")
    del call["intrinsic_kind"]
    request = _request(call)
    out = degrade_intrinsic_history(request, _chat())
    call_out = out["messages"][1]["content"][0]
    assert call_out["tool_type"] == "function"
    assert call_out["tool_name"] == "intrinsic_tool"
    assert call_out["provider_metadata"]["_downgraded_from"]["kind"] == ""


def test_legacy_metadata_kind_is_removed_with_the_marker():
    """A pre-#999 ``provider_metadata.intrinsic_kind`` does not survive."""
    call = _intrinsic_call("web_search")
    call.pop("intrinsic_kind")
    call["provider_metadata"] = {"intrinsic_kind": "web_search"}
    request = _request(call)
    out = degrade_intrinsic_history(request, _chat())
    metadata = out["messages"][1]["content"][0]["provider_metadata"]
    assert metadata["_downgraded_from"]["kind"] == "web_search"
    assert "intrinsic_kind" not in metadata


def test_only_the_unrenderable_pair_is_touched():
    """A renderable call in the same request keeps its intrinsic identity."""
    renderable = _intrinsic_call("web_search", "keep")
    unrenderable = _intrinsic_call("google_search", "drop")
    request: dict[str, Any] = {
        "model": "m",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    renderable,
                    _intrinsic_result("web_search", "keep"),
                    unrenderable,
                ],
            },
        ],
    }
    out = degrade_intrinsic_history(request, _messages())
    content = out["messages"][0]["content"]
    assert content[0]["tool_type"] == "intrinsic"
    assert content[0]["tool_name"] == "web_search"
    assert content[2]["tool_type"] == "function"
    assert content[2]["tool_name"] == "google_search"


def test_does_not_mutate_the_original():
    request = _request(_intrinsic_call("web_search"))
    snapshot = copy.deepcopy(request)
    degrade_intrinsic_history(request, _chat())
    assert request == snapshot


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------


def test_two_calls_of_one_kind_share_a_name():
    request: dict[str, Any] = {
        "model": "m",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    _intrinsic_call("web_search", "a"),
                    _intrinsic_call("web_search", "b"),
                ],
            }
        ],
    }
    out = degrade_intrinsic_history(request, _chat())
    names = [p["tool_name"] for p in out["messages"][0]["content"]]
    assert names == ["web_search", "web_search"]


def test_two_kinds_that_collide_on_a_name_are_qualified():
    """``a b`` and ``a_b`` both sanitise to ``a_b``; the second is qualified."""
    request: dict[str, Any] = {
        "model": "m",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    _intrinsic_call("a b", "a"),
                    _intrinsic_call("a_b", "b"),
                ],
            }
        ],
    }
    out = degrade_intrinsic_history(request, _chat())
    names = [p["tool_name"] for p in out["messages"][0]["content"]]
    assert names == ["a_b", "a_b_intrinsic"]
    assert len(set(names)) == 2


def test_degraded_name_avoids_a_declared_client_function():
    """A client function named ``web_search`` keeps its name to itself."""
    request = _request(
        _intrinsic_call("web_search"),
        tools=[
            {
                "type": "function",
                "name": "web_search",
                "description": "the client's own",
                "parameters": {"type": "object"},
            }
        ],
    )
    out = degrade_intrinsic_history(request, _chat())
    assert out["messages"][1]["content"][0]["tool_name"] == "web_search_intrinsic"


def test_degraded_name_is_not_taken_from_the_intrinsic_definition():
    """An intrinsic *definition* is the same identity, so it does not block."""
    request = _request(
        _intrinsic_call("web_search"),
        tools=[{"type": "intrinsic", "name": "web_search", "parameters": {}}],
    )
    out = degrade_intrinsic_history(request, _chat())
    assert out["messages"][1]["content"][0]["tool_name"] == "web_search"


def test_degraded_name_is_name_safe():
    request = _request(_intrinsic_call("mcp://server/tool name"))
    out = degrade_intrinsic_history(request, _chat())
    name = out["messages"][1]["content"][0]["tool_name"]
    assert name.replace("_", "").replace("-", "").isalnum()


# ---------------------------------------------------------------------------
# Per-target renderability
# ---------------------------------------------------------------------------


#: Per-kind ``tool_input`` a history replay needs to be renderable at all
#: (open_responses requires ``container_id`` on a code interpreter).
_RENDERABLE_INPUT: dict[str, dict[str, Any]] = {
    "code_execution": {"code": "print(1)", "language": "PYTHON"},
    "code_interpreter": {"code": "print(1)", "container_id": "cntr_1"},
    "google_search": {"queries": ["q"]},
    "web_search": {"queries": ["q"]},
}


@pytest.mark.parametrize(
    "converter,kinds_renderable,kinds_degraded",
    [
        (MessagesConverter, ("web_search", "code_execution"), ("google_search",)),
        (OpenAIResponsesConverter, ("web_search", "code_interpreter"), ()),
        (GoogleGenerateConverter, ("code_execution",), ("web_search",)),
        (ChatCompletionsConverter, (), ("web_search",)),
    ],
)
def test_renderability_decides(
    converter: Any, kinds_renderable: tuple[str, ...], kinds_degraded: tuple[str, ...]
):
    tool_ops = converter().tool_ops
    for kind in kinds_renderable:
        request = _request(_intrinsic_call(kind, tool_input=_RENDERABLE_INPUT[kind]))
        assert degrade_intrinsic_history(request, tool_ops) is request, kind
    for kind in kinds_degraded:
        request = _request(_intrinsic_call(kind, tool_input=_RENDERABLE_INPUT[kind]))
        out = degrade_intrinsic_history(request, tool_ops)
        assert out["messages"][1]["content"][0]["tool_type"] == "function", kind


def test_open_responses_degrades_a_code_interpreter_without_a_container():
    """The payload decides: no ``container_id`` is a replay the API rejects."""
    tool_ops = OpenAIResponsesConverter().tool_ops
    request = _request(_intrinsic_call("code_interpreter", tool_input={"code": "x"}))
    out = degrade_intrinsic_history(request, tool_ops)
    assert out["messages"][1]["content"][0]["tool_type"] == "function"


# ---------------------------------------------------------------------------
# Anonymous calls, the result requirement, and the reasons reported
# ---------------------------------------------------------------------------


def test_two_anonymous_calls_keep_their_own_kind():
    """Ids are optional, so two id-less calls must not collapse into one.

    Keyed on the id alone, both would take the last kind seen — the first
    call's kind and name would be lost, and the client would see two different
    server calls as the same function.
    """
    request: dict[str, Any] = {
        "model": "m",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    _intrinsic_call("google_search", ""),
                    _intrinsic_call("url_context", ""),
                ],
            }
        ],
    }
    warnings: list[str] = []
    out = degrade_intrinsic_history(request, _messages(), warnings=warnings)

    first, second = out["messages"][0]["content"]
    assert first["tool_name"] == "google_search"
    assert second["tool_name"] == "url_context"
    assert first["provider_metadata"]["_downgraded_from"]["kind"] == "google_search"
    assert second["provider_metadata"]["_downgraded_from"]["kind"] == "url_context"
    assert len(warnings) == 2


def test_an_anonymous_result_pairs_with_the_call_before_it():
    request: dict[str, Any] = {
        "model": "m",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    _intrinsic_call("google_search", ""),
                    _intrinsic_result("google_search", ""),
                ],
            }
        ],
    }
    out = degrade_intrinsic_history(request, _messages())

    call, result = out["messages"][0]["content"]
    assert call["tool_type"] == result["tool_type"] == "function"
    assert result["result"] == "R"


def test_a_call_the_target_needs_a_result_for_is_degraded():
    """Anthropic rejects a ``server_tool_use`` nothing answers."""
    tool_ops = _messages()
    answered = _request(_intrinsic_call("web_search", tool_input={"query": "q"}))
    assert degrade_intrinsic_history(answered, tool_ops) is answered

    unanswered = copy.deepcopy(answered)
    unanswered["messages"] = unanswered["messages"][:2]  # drop the result turn
    warnings: list[str] = []
    out = degrade_intrinsic_history(unanswered, tool_ops, warnings=warnings)

    assert out["messages"][1]["content"][0]["tool_type"] == "function"
    assert "needs a result" in warnings[0]


def test_the_warning_names_the_reason():
    """A reader must be able to tell the two failure modes apart."""
    unknown = _request(_intrinsic_call("special_search"))
    warnings: list[str] = []
    degrade_intrinsic_history(unknown, _messages(), warnings=warnings)
    assert "cannot write the 'special_search' call as it stands" in warnings[0]

    no_kind = _request(_intrinsic_call("web_search"))
    no_kind["messages"][1]["content"][0].pop("intrinsic_kind")
    warnings = []
    degrade_intrinsic_history(no_kind, _messages(), warnings=warnings)
    assert "carries no intrinsic kind" in warnings[0]


def test_an_untyped_degraded_result_becomes_json_text():
    """An ordinary tool result accepts a string or typed blocks, nothing else.

    A search result is a list of provider blocks with no ``type``; Anthropic
    and OpenAI reject those once the pair is degraded
    (``tool_result.content.0.type: Field required``), so the payload is carried
    as JSON text.
    """
    call = _intrinsic_call("google_search")
    request = _request(call)
    request["messages"][2]["content"][0]["result"] = [
        {"search_suggestions": "<div>x</div>"},
        "plain",
    ]
    out = degrade_intrinsic_history(request, _messages())

    result = out["messages"][2]["content"][0]["result"]
    assert result == json.dumps(
        [{"search_suggestions": "<div>x</div>"}, "plain"], ensure_ascii=False
    )


def test_a_typed_degraded_result_keeps_its_blocks():
    call = _intrinsic_call("google_search")
    request = _request(call)
    request["messages"][2]["content"][0]["result"] = [{"type": "text", "text": "t"}]
    out = degrade_intrinsic_history(request, _messages())

    assert out["messages"][2]["content"][0]["result"] == [{"type": "text", "text": "t"}]
