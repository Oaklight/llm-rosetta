"""Intrinsic handling through ``ConversionPipeline`` (request, response, stream).

A few end-to-end pins for the three pairs the refactor touches directly, plus
the two cross-cutting behaviours that act on intrinsic parts before any
converter sees them: the orphaned-tool-call fixer and history-name respelling.

``ConversionPipeline(source, target)`` converts a *client* (source) body to an
*upstream* (target) body on the request leg, and upstream chunks back to client
chunks on the stream leg, so a stream processor reads target chunks and emits
source ones.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from llm_rosetta.capabilities.tool_names import (
    apply_upstream_tool_names,
    build_tool_name_map,
)
from llm_rosetta.converters.base.tools.intrinsic import make_intrinsic_tool_call
from llm_rosetta.pipeline import ConversionPipeline
from llm_rosetta.shims.provider_shim import ProviderShim, ToolsConfig

from .conftest import ir_history


def _stream(pipe: ConversionPipeline, chunks: list[dict[str, Any]]) -> list[Any]:
    proc = pipe.create_stream_processor()
    out: list[Any] = []
    for chunk in chunks:
        out.extend(proc.process_chunk(chunk))
    return out


def _messages_request() -> dict[str, Any]:
    return {
        "model": "m",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "hello"}],
    }


def _google_interactions_request() -> dict[str, Any]:
    return {"model": "m", "input": "hello"}


# google_generate warns while dropping an unrenderable kind; the tests
# that pin the warning use pytest.warns, which sets its own filter.
pytestmark = pytest.mark.filterwarnings("ignore:Unsupported intrinsic kind")


# ---------------------------------------------------------------------------
# Stream: messages <- open_responses (client messages, upstream open_responses)
# ---------------------------------------------------------------------------

_OR_CI_CHUNKS: list[dict[str, Any]] = [
    {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {
            "type": "code_interpreter_call",
            "id": "ci_1",
            "status": "in_progress",
        },
    },
    {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "type": "code_interpreter_call",
            "id": "ci_1",
            "status": "completed",
            "code": "print(42)",
            "container_id": "cntr_1",
            "outputs": [{"type": "logs", "logs": "42"}],
        },
    },
    {
        "type": "response.output_item.added",
        "output_index": 1,
        "item": {
            "type": "code_interpreter_call_output",
            "call_id": "ci_1",
            "output": "42",
        },
    },
]


def test_stream_messages_from_open_responses_drops_a_server_call_it_lacks():
    """Phase 4: the Anthropic client gets neither a fabricated server call nor
    a fabricated ``*_tool_result`` block for a kind Anthropic has no server
    tool for (``code_interpreter``).

    The upstream ran the tool; the client dialect cannot name it, and the
    response leg never hands a server call to the client as one it must run.
    """
    pipe = ConversionPipeline("messages", "open_responses")
    pipe.convert_request(_messages_request())
    chunks = _stream(pipe, _OR_CI_CHUNKS)

    blocks = [
        c["content_block"]["type"]
        for c in chunks
        if isinstance(c, dict) and c.get("type") == "content_block_start"
    ]
    assert "server_tool_use" not in blocks
    assert not [b for b in blocks if b.endswith("_tool_result")]


def test_stream_messages_from_google_generate_keeps_a_call_it_has():
    """A kind Anthropic *does* have a server tool for still passes through.

    Google's ``executableCode`` is canonical kind ``code_execution``, which is
    one of the two server tools Anthropic has.
    """
    pipe = ConversionPipeline("messages", "google_generate")
    pipe.convert_request(_messages_request())
    chunks = _stream(pipe, _GG_CODE_CHUNKS)

    blocks = [
        c["content_block"]["type"]
        for c in chunks
        if isinstance(c, dict) and c.get("type") == "content_block_start"
    ]
    assert "server_tool_use" in blocks
    assert "code_execution_tool_result" in blocks


# ---------------------------------------------------------------------------
# Stream: google_interactions <- google_generate
# ---------------------------------------------------------------------------

_GG_CODE_CHUNKS: list[dict[str, Any]] = [
    {
        "candidates": [
            {
                "index": 0,
                "content": {
                    "role": "model",
                    "parts": [
                        {"executableCode": {"code": "print(42)", "language": "PYTHON"}}
                    ],
                },
            }
        ]
    },
    {
        "candidates": [
            {
                "index": 0,
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "codeExecutionResult": {
                                "output": "42\n",
                                "outcome": "OUTCOME_OK",
                            }
                        }
                    ],
                },
            }
        ]
    },
]


def test_stream_google_interactions_from_google_generate_code_execution():
    pipe = ConversionPipeline("google_interactions", "google_generate")
    pipe.convert_request(_google_interactions_request())
    chunks = _stream(pipe, _GG_CODE_CHUNKS)

    steps = [
        c["step"]
        for c in chunks
        if isinstance(c, dict)
        and c.get("event_type") == "step.start"
        and isinstance(c.get("step"), dict)
    ]
    step_types = [s["type"] for s in steps]
    assert "code_execution_call" in step_types
    assert "code_execution_result" in step_types

    call = next(s for s in steps if s["type"] == "code_execution_call")
    assert call["id"]

    deltas = [
        c["delta"]
        for c in chunks
        if isinstance(c, dict)
        and c.get("event_type") == "step.delta"
        and isinstance(c.get("delta"), dict)
    ]
    code_call = next(d for d in deltas if d["type"] == "code_execution_call")
    assert code_call["arguments"]["code"] == "print(42)"


# ---------------------------------------------------------------------------
# Stream: messages <- chat_completions (client messages, upstream chat)
# ---------------------------------------------------------------------------

_CHAT_CALL_CHUNKS: list[dict[str, Any]] = [
    {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "web_search", "arguments": ""},
                        }
                    ],
                },
                "finish_reason": None,
            }
        ]
    },
    {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": '{"queries": ["q"]}'},
                        }
                    ]
                },
                "finish_reason": None,
            }
        ]
    },
    {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
]


def test_stream_messages_from_chat_degrades_the_server_call():
    """The upstream has no server tools, so the call arrives as a client call."""
    pipe = ConversionPipeline("messages", "chat_completions")
    pipe.convert_request(_messages_request())
    chunks = _stream(pipe, _CHAT_CALL_CHUNKS)

    call = next(
        c["content_block"]
        for c in chunks
        if isinstance(c, dict) and c.get("type") == "content_block_start"
    )
    assert call["type"] == "tool_use"  # a client call, not server_tool_use
    assert call["name"] == "web_search"


# ---------------------------------------------------------------------------
# Request leg: intrinsic history through the pipeline
# ---------------------------------------------------------------------------


def test_request_messages_to_open_responses_history():
    pipe = ConversionPipeline("messages", "open_responses")
    out = pipe.convert_request(cast(Any, _messages_history_request("web_search")))
    text = json.dumps(out, default=str)
    assert "web_search_call" in text


def test_request_google_interactions_to_google_generate_history():
    pipe = ConversionPipeline("google_interactions", "google_generate")
    out = pipe.convert_request(cast(Any, _gi_history_request("code_execution")))
    assert "executableCode" in json.dumps(out, default=str)


def test_request_messages_to_chat_history_degrades():
    pipe = ConversionPipeline("messages", "chat_completions")
    out = pipe.convert_request(cast(Any, _messages_history_request("web_search")))
    names = [
        tc["function"]["name"]
        for m in out["messages"]
        for tc in m.get("tool_calls", [])
    ]
    assert names == ["web_search"]
    assert out["messages"][-1]["role"] == "tool"


def _messages_history_request(kind: str) -> dict[str, Any]:
    """A messages-format request carrying one intrinsic call+result in history."""
    return {
        "model": "m",
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "server_tool_use",
                        "id": "s1",
                        "name": kind,
                        "input": {"queries": ["q"]},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "s1", "content": "R"}
                ],
            },
        ],
    }


def _gi_history_request(kind: str) -> dict[str, Any]:
    return {
        "model": "m",
        "input": [
            {"type": "user_input", "content": [{"type": "text", "text": "hello"}]},
            {"type": kind + "_call", "id": "s1", "arguments": {"code": "print(1)"}},
            {"type": kind + "_result", "call_id": "s1", "result": "1\n"},
        ],
    }


# ---------------------------------------------------------------------------
# Definition gating by the target shim
# ---------------------------------------------------------------------------


def _gg_shim(kinds: tuple[str, ...]) -> ProviderShim:
    return ProviderShim(
        name="gg-test", base="google_generate", tools=ToolsConfig(intrinsic_tools=kinds)
    )


def _gg_definition_request() -> dict[str, Any]:
    return {
        "model": "m",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [
            {"type": "code_execution_20250522", "name": "code_execution"},
            {"type": "web_search_20250305", "name": "web_search"},
        ],
    }


def _tool_names(body: dict[str, Any]) -> list[str]:
    return [
        (t.get("name") or next(iter(t)))
        for t in body.get("config", {}).get("tools", [])
    ] + [t.get("name") for t in body.get("tools", []) if isinstance(t, dict)]


def test_definition_gating_keeps_only_the_declared_kind():
    pipe = ConversionPipeline(
        "messages", "google_generate", target_shim=_gg_shim(("code_execution",))
    )
    out = pipe.convert_request(cast(Any, _gg_definition_request()))
    assert "code_execution" in json.dumps(out)
    assert "web_search" not in json.dumps(out)


def test_definition_gating_drops_all_without_a_declaration():
    pipe = ConversionPipeline("messages", "google_generate", target_shim=_gg_shim(()))
    out = pipe.convert_request(cast(Any, _gg_definition_request()))
    assert "code_execution" not in json.dumps(out)
    assert "web_search" not in json.dumps(out)


# ---------------------------------------------------------------------------
# Cross-cutting: the orphaned-tool-call fixer
# ---------------------------------------------------------------------------


def _orphan_ir():
    return {
        "model": "m",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {
                "role": "assistant",
                "content": [
                    make_intrinsic_tool_call(
                        tool_call_id="s1",
                        tool_name="web_search",
                        tool_input={"queries": ["q"]},
                        intrinsic_kind="web_search",
                    )
                ],
            },
        ],
    }


def test_orphan_fix_injects_a_placeholder_for_a_server_call():
    """CURRENT: a server call with no separate result gets a client placeholder."""
    from llm_rosetta.converters.chat_completions import ChatCompletionsConverter

    out, _ = ChatCompletionsConverter().request_to_provider(cast(Any, _orphan_ir()))
    roles = [m["role"] for m in out["messages"]]
    assert roles[-1] == "tool"
    assert out["messages"][-1]["content"] == "[No output available yet]"


def test_orphan_fix_placeholder_reaches_google_generate_as_function_response():
    from llm_rosetta.converters.google_generate import GoogleGenerateConverter

    out, _ = GoogleGenerateConverter().request_to_provider(cast(Any, _orphan_ir()))
    assert "functionResponse" in json.dumps(out)
    assert "[No output available yet]" in json.dumps(out)


@pytest.mark.xfail(
    strict=True,
    reason="intrinsic refactor phase 4: orphan fixer placeholders server calls",
)
def test_orphan_fix_should_not_placeholder_a_server_call():
    from llm_rosetta.converters.chat_completions import ChatCompletionsConverter

    out, _ = ChatCompletionsConverter().request_to_provider(cast(Any, _orphan_ir()))
    assert "[No output available yet]" not in json.dumps(out)


def test_orphan_fix_leaves_an_answered_intrinsic_call_alone():
    from llm_rosetta.converters.chat_completions import ChatCompletionsConverter

    ir = ir_history("web_search")
    out, _ = ChatCompletionsConverter().request_to_provider(cast(Any, ir))
    assert "[No output available yet]" not in json.dumps(out)


# ---------------------------------------------------------------------------
# Cross-cutting: history tool-name respelling
# ---------------------------------------------------------------------------


def test_respell_renames_an_intrinsic_history_call():
    """CURRENT: ``_respell_history_tool_calls`` renames intrinsic calls too."""
    ir = {
        "tools": [
            {
                "name": "agents_wait",
                "tool_group": "agents",
                "metadata": {"_original_name": "wait"},
            }
        ],
        "messages": [
            {
                "role": "assistant",
                "content": [
                    make_intrinsic_tool_call(
                        tool_call_id="s1",
                        tool_name="wait",
                        tool_input={},
                        intrinsic_kind="wait",
                    )
                ],
            }
        ],
    }
    apply_upstream_tool_names(ir, name_map=build_tool_name_map(ir), warnings=[])
    part = ir["messages"][0]["content"][0]
    assert part["tool_name"] == "agents_wait"
    assert part["intrinsic_kind"] == "wait"


@pytest.mark.xfail(
    strict=True, reason="intrinsic refactor phase 4: respell renames intrinsic calls"
)
def test_respell_should_leave_an_intrinsic_call_alone():
    ir = {
        "tools": [
            {
                "name": "agents_wait",
                "tool_group": "agents",
                "metadata": {"_original_name": "wait"},
            }
        ],
        "messages": [
            {
                "role": "assistant",
                "content": [
                    make_intrinsic_tool_call(
                        tool_call_id="s1",
                        tool_name="wait",
                        tool_input={},
                        intrinsic_kind="wait",
                    )
                ],
            }
        ],
    }
    apply_upstream_tool_names(ir, name_map=build_tool_name_map(ir), warnings=[])
    part = ir["messages"][0]["content"][0]
    assert part["tool_name"] == "wait"
