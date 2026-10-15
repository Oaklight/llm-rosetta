"""Intrinsic history through ``ConversionPipeline`` — phase 4 end-to-end pins.

The pipeline owns one decision for history: whether the *target* can replay an
intrinsic part at all (:func:`llm_rosetta.capabilities.degrade_intrinsic_history`,
which asks the target converter's ``intrinsic_renderable``).  A pair the target
cannot render is rewritten into an ordinary function call plus its result, with
one warning; a pair it can render is left alone.

These tests drive real client bodies through the pipeline in both directions
and assert the pair's shape, the shared id, the warning, and — for a target
that renders the kind natively — that the part survives untouched.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from llm_rosetta.pipeline import ConversionPipeline

# ---------------------------------------------------------------------------
# Client bodies
# ---------------------------------------------------------------------------


def _messages_body(*, server_pair: bool = True) -> dict[str, Any]:
    """An Anthropic body whose history carries one server-tool web search."""
    if server_pair:
        call: dict[str, Any] = {
            "type": "server_tool_use",
            "id": "srvtoolu_1",
            "name": "web_search",
            "input": {"query": "mars rover"},
        }
        result: dict[str, Any] = {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_1",
            "content": [{"type": "web_search_result", "url": "https://x"}],
        }
    else:
        call = {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "get_weather",
            "input": {"city": "x"},
        }
        result = {
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": "sunny",
        }
    return {
        "model": "claude-haiku-4-5",
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "search"}]},
            {"role": "assistant", "content": [call]},
            {"role": "user", "content": [result]},
        ],
    }


def _gi_body() -> dict[str, Any]:
    """A Google Interactions body whose history carries one google search."""
    return {
        "model": "gemini-2.0-flash",
        "input": [
            {"type": "user_input", "content": [{"type": "text", "text": "search"}]},
            {
                "type": "google_search_call",
                "id": "gi_1",
                "arguments": {"queries": ["mars rover"]},
            },
            {
                "type": "google_search_result",
                "call_id": "gi_1",
                "result": [{"rendered_content": "<div>x</div>"}],
            },
        ],
    }


def _or_body(with_container: bool = False) -> dict[str, Any]:
    """An OpenAI Responses body with a code interpreter call and its output."""
    item: dict[str, Any] = {
        "type": "code_interpreter_call",
        "id": "ci_1",
        "status": "completed",
        "code": "print(42)",
        "outputs": [{"type": "logs", "logs": "42"}],
    }
    if with_container:
        item["container_id"] = "cntr_1"
    return {
        "model": "gpt-5",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "run it"}],
            },
            item,
            {
                "type": "code_interpreter_call_output",
                "call_id": "ci_1",
                "output": [{"type": "logs", "logs": "42"}],
            },
        ],
    }


def _degradation_warnings(pipe: ConversionPipeline) -> list[str]:
    return [w for w in pipe.warnings if "degrade" in w or "cannot be replayed" in w]


# ---------------------------------------------------------------------------
# messages → (open_responses | google_interactions | chat_completions)
# ---------------------------------------------------------------------------


_CLIENT_PAIR_ITEMS = ("function_call", "function_call_output", "function_result")


def _names_and_ids(body: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Every client call/result name and id in a flat item list."""
    items = body.get("input") or []
    names: list[str] = []
    ids: list[str] = []
    for item in items:
        if item.get("type", "") in _CLIENT_PAIR_ITEMS:
            names.append(item.get("name", ""))
            ids.append(item.get("call_id") or item.get("id", ""))
    return names, ids


def test_messages_web_search_to_open_responses_stays_native():
    """``web_search`` is Responses' own server tool, so nothing is degraded."""
    pipe = ConversionPipeline("messages", "open_responses")
    out = pipe.convert_request(cast(Any, _messages_body()))

    types = [item["type"] for item in out["input"]]
    assert "web_search_call" in types
    assert "function_call" not in types
    assert _degradation_warnings(pipe) == []


def test_messages_web_search_to_google_interactions_degrades_the_pair():
    pipe = ConversionPipeline("messages", "google_interactions")
    out = pipe.convert_request(cast(Any, _messages_body()))

    names, ids = _names_and_ids(out)
    assert names == ["web_search", "web_search"]
    assert ids == ["srvtoolu_1", "srvtoolu_1"]
    assert _degradation_warnings(pipe)


def test_messages_web_search_to_chat_completions_degrades_the_pair():
    pipe = ConversionPipeline("messages", "chat_completions")
    out = pipe.convert_request(cast(Any, _messages_body()))

    calls = [tc for m in out["messages"] for tc in m.get("tool_calls", [])]
    results = [m for m in out["messages"] if m.get("role") == "tool"]
    assert [c["function"]["name"] for c in calls] == ["web_search"]
    assert [c["id"] for c in calls] == ["srvtoolu_1"]
    assert [m["tool_call_id"] for m in results] == ["srvtoolu_1"]
    assert _degradation_warnings(pipe)


def test_messages_web_search_round_trip_is_untouched():
    """A same-dialect route keeps the server pair exactly as the client sent it."""
    pipe = ConversionPipeline("messages", "messages")
    out = pipe.convert_request(cast(Any, _messages_body()))

    # The result is written back into the assistant turn that ran the tool,
    # where Anthropic requires it, even when the client sent it in a later turn.
    blocks = out["messages"][1]["content"]
    assert [b["type"] for b in blocks] == ["server_tool_use", "web_search_tool_result"]
    assert blocks[0]["name"] == "web_search"
    assert blocks[0]["id"] == "srvtoolu_1"
    assert blocks[1]["tool_use_id"] == "srvtoolu_1"
    assert _degradation_warnings(pipe) == []


# ---------------------------------------------------------------------------
# google_interactions → messages
# ---------------------------------------------------------------------------


def test_google_search_history_to_messages_degrades_the_pair():
    pipe = ConversionPipeline("google_interactions", "messages")
    out = pipe.convert_request(cast(Any, _gi_body()))

    blocks = [
        b for m in out["messages"] for b in m.get("content", []) if isinstance(b, dict)
    ]
    calls = [b for b in blocks if b["type"] in ("tool_use", "server_tool_use")]
    assert [c["type"] for c in calls] == ["tool_use"]
    assert calls[0]["name"] == "google_search"
    assert calls[0]["id"] == "gi_1"
    result = next(b for b in blocks if b["type"].endswith("tool_result"))
    assert result["tool_use_id"] == "gi_1"
    assert _degradation_warnings(pipe)


def test_google_search_round_trip_is_untouched():
    """The origin dialect replays its own server call natively."""
    pipe = ConversionPipeline("google_interactions", "google_interactions")
    out = pipe.convert_request(cast(Any, _gi_body()))

    assert "google_search_call" in json.dumps(out, default=str)
    assert _degradation_warnings(pipe) == []


# ---------------------------------------------------------------------------
# open_responses → open_responses
# ---------------------------------------------------------------------------


def test_code_interpreter_without_a_container_degrades_the_pair():
    """The IR cannot supply ``container_id``, so the replay would be rejected."""
    pipe = ConversionPipeline("open_responses", "open_responses")
    out = pipe.convert_request(cast(Any, _or_body()))

    names, ids = _names_and_ids(out)
    # The output item carries no name of its own; only the call is named.
    assert names == ["code_interpreter", ""]
    assert ids == ["ci_1", "ci_1"]
    assert _degradation_warnings(pipe)


def test_code_interpreter_with_a_container_replays_natively():
    pipe = ConversionPipeline("open_responses", "open_responses")
    out = pipe.convert_request(cast(Any, _or_body(with_container=True)))

    types = [item["type"] for item in out["input"]]
    assert "code_interpreter_call" in types
    assert "function_call" not in types
    assert _degradation_warnings(pipe) == []


# ---------------------------------------------------------------------------
# The decision is per target, not per source
# ---------------------------------------------------------------------------


#: The native and degraded server-call marker each dialect writes.
_NATIVE_MARKERS: dict[str, str] = {
    "messages": '"type": "server_tool_use"',
    "open_responses": '"type": "web_search_call"',
    "google_generate": '"executableCode"',
    "google_interactions": '"type": "google_search_call"',
    "chat_completions": '"type": "function"',
}
_DEGRADED_MARKERS: dict[str, str] = {
    "messages": '"type": "tool_use"',
    "open_responses": '"type": "function_call"',
    "google_generate": '"functionCall"',
    "google_interactions": '"type": "function_call"',
    "chat_completions": '"tool_calls"',
}


@pytest.mark.parametrize(
    "target,expect_native",
    [
        ("messages", True),
        ("open_responses", True),
        ("google_generate", False),
        ("google_interactions", False),
        ("chat_completions", False),
    ],
)
def test_one_messages_pair_decides_per_target(target: str, expect_native: bool):
    """The same history is native or degraded by what the target can replay."""
    pipe = ConversionPipeline("messages", target)
    out = pipe.convert_request(cast(Any, _messages_body()))
    text = json.dumps(out, default=str)

    if expect_native:
        marker = _NATIVE_MARKERS[target]
    else:
        marker = _DEGRADED_MARKERS[target]
    assert marker in text
    assert bool(_degradation_warnings(pipe)) is (not expect_native)
