"""A hosted Google tool call is not a client tool call (#1227).

Google delivers a hosted (server-side) tool's call through the same
``functionCall`` part a client function uses, named after the tool
(``googleSearch``).  Before this, the reader hard-coded ``tool_type="function"``,
so a turn the provider had already completed read as ``finish_reason:
"tool_calls"`` and the orphan fixer fabricated a result for a call no client
could answer.

Classification now comes from what the request enabled upstream, recorded by
the request leg on the conversion context.  Only a response converted on its
own falls back to the name table, and that fallback is pinned here as the
documented limitation it is.

A promoted part also records the native ``functionCall`` it arrived as
(``provider_metadata["_native"]``).  That marker is what lets a Google target
write the call back in the same-format route; every other target, and a
``google_search`` from another dialect, still has no form for it and drops it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

from llm_rosetta.converters.base import StreamContext
from llm_rosetta.converters.llm.google_generate import GoogleGenerateConverter
from llm_rosetta.ir.tools.orphan_fix import (
    UNANSWERED_TOOL_CALL,
    fix_orphaned_tool_calls_ir,
)
from llm_rosetta.pipeline.chat import ConversionPipeline

#: The Google wire response an upstream sends when it ran ``googleSearch``
#: itself.  ``STOP`` is Google's normal completion reason; the parts decide
#: whether the client is owed anything.
GOOGLE_RESPONSE: dict[str, Any] = {
    "candidates": [
        {
            "content": {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "googleSearch", "args": {"q": "x"}}}
                ],
            },
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 1,
        "candidatesTokenCount": 2,
        "totalTokenCount": 3,
    },
}

#: The same logical response as Server-Sent Events: Google sends the call
#: whole in one chunk and finishes in a later one with no content.
GOOGLE_STREAM: list[dict[str, Any]] = [
    {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "googleSearch",
                                "args": {"q": "x"},
                            }
                        }
                    ],
                }
            }
        ]
    },
    {
        "candidates": [
            {"content": {"role": "model", "parts": []}, "finishReason": "STOP"}
        ],
        "usageMetadata": {
            "promptTokenCount": 1,
            "candidatesTokenCount": 2,
            "totalTokenCount": 3,
        },
    },
]


#: The Google request that enables the hosted search.  A same-format route
#: hands it back to a Google upstream unchanged.
GOOGLE_REQUEST: dict[str, Any] = {
    "model": "gemini-2.5-flash",
    "contents": [{"role": "user", "parts": [{"text": "search the web"}]}],
    "tools": [{"google_search": {}}],
}


def _client_request(tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """A Chat Completions client body, optionally declaring *tools*."""
    body: dict[str, Any] = {
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "search the web"}],
    }
    if tools:
        body["tools"] = tools
    return body


def _google_pipeline(tools: list[dict[str, Any]] | None = None) -> ConversionPipeline:
    """A Chat-client → Google-upstream pipeline whose request leg has run."""
    pipe = ConversionPipeline(
        "chat_completions",
        "google_generate",
        target_shim="google--google_generate",
    )
    pipe.convert_request(_client_request(tools))
    return pipe


def _hosted_search_tool() -> dict[str, Any]:
    """A Chat client asking for the provider-hosted search.

    The name is the canonical intrinsic kind, which is how a client format
    without server tools asks for one; the pipeline promotes it to an
    intrinsic definition before the request goes upstream.
    """
    return {
        "type": "function",
        "function": {
            "name": "google_search",
            "description": "search the web",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _client_function(name: str = "googleSearch") -> dict[str, Any]:
    """An ordinary client function that happens to share the hosted name."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "client-side search",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _ir_response(pipe: ConversionPipeline) -> dict[str, Any]:
    """Convert :data:`GOOGLE_RESPONSE` and return the IR it produced."""
    holder: dict[str, Any] = {}
    pipe.convert_response(cast(Any, GOOGLE_RESPONSE), on_ir_ready=holder.update)
    return holder


def _first_part(ir: Mapping[str, Any]) -> dict[str, Any]:
    return ir["choices"][0]["message"]["content"][0]


def _reason(ir: Mapping[str, Any]) -> str:
    return ir["choices"][0]["finish_reason"]["reason"]


def _stream_ir_events(pipe: ConversionPipeline) -> list[dict[str, Any]]:
    """Read :data:`GOOGLE_STREAM` through the pipeline, keeping IR events."""
    events: list[dict[str, Any]] = []
    processor = pipe.create_stream_processor(
        on_ir_event=lambda event: events.append(dict(event))
    )
    for chunk in GOOGLE_STREAM:
        processor.process_chunk(cast(Any, chunk))
    return events


# ---------------------------------------------------------------------------
# Request-aware classification
# ---------------------------------------------------------------------------


def test_a_request_that_enabled_hosted_search_reads_the_call_as_intrinsic():
    ir = _ir_response(_google_pipeline([_hosted_search_tool()]))

    part = _first_part(ir)
    assert part["tool_name"] == "googleSearch"
    assert part["tool_type"] == "intrinsic"
    assert part["intrinsic_kind"] == "google_search"


def test_a_completed_hosted_turn_is_not_a_tool_call_turn():
    ir = _ir_response(_google_pipeline([_hosted_search_tool()]))

    assert _reason(ir) == "stop"


def test_the_chat_target_is_never_asked_to_run_the_hosted_call():
    """The response leg drops a server call the client cannot answer (#986)."""
    out = _google_pipeline([_hosted_search_tool()]).convert_response(
        cast(Any, GOOGLE_RESPONSE)
    )

    assert out["choices"][0]["finish_reason"] == "stop"
    assert "googleSearch" not in json.dumps(out)


def test_the_orphan_fixer_leaves_a_hosted_call_alone():
    """No synthetic result beside a call the provider itself ran."""
    ir = _ir_response(_google_pipeline([_hosted_search_tool()]))
    history = [
        {"role": "user", "content": [{"type": "text", "text": "search the web"}]},
        ir["choices"][0]["message"],
    ]

    fixed = fix_orphaned_tool_calls_ir(cast(Any, history))

    assert fixed == history
    assert UNANSWERED_TOOL_CALL not in json.dumps(fixed)


def test_a_client_function_named_like_the_hosted_tool_stays_a_client_call():
    """The discrimination case: classification is not name-only."""
    ir = _ir_response(_google_pipeline([_client_function()]))

    part = _first_part(ir)
    assert part["tool_type"] == "function"
    assert "intrinsic_kind" not in part
    assert _reason(ir) == "tool_calls"


def test_a_client_function_named_like_the_hosted_tool_reaches_the_client():
    out = _google_pipeline([_client_function()]).convert_response(
        cast(Any, GOOGLE_RESPONSE)
    )

    assert out["choices"][0]["finish_reason"] == "tool_calls"
    call = out["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "googleSearch"


def test_a_request_with_no_hosted_tool_reads_the_call_as_a_client_call():
    """A recorded request that enabled nothing is not the fallback case."""
    ir = _ir_response(_google_pipeline())

    part = _first_part(ir)
    assert part["tool_type"] == "function"
    assert _reason(ir) == "tool_calls"


def test_a_declared_client_function_wins_over_the_hosted_tool_of_that_name():
    """Both enabled: the call may be either, so the answerable one wins."""
    ir = _ir_response(_google_pipeline([_hosted_search_tool(), _client_function()]))

    part = _first_part(ir)
    assert part["tool_type"] == "function"
    assert _reason(ir) == "tool_calls"


# ---------------------------------------------------------------------------
# Standalone responses (no request leg: documented name-table fallback)
# ---------------------------------------------------------------------------


def test_a_standalone_response_falls_back_to_the_name_table():
    ir = GoogleGenerateConverter().response_from_provider(cast(Any, GOOGLE_RESPONSE))

    part = _first_part(ir)
    assert part["tool_type"] == "intrinsic"
    assert part["intrinsic_kind"] == "google_search"
    assert _reason(ir) == "stop"


def test_a_standalone_stream_falls_back_to_the_name_table():
    context = StreamContext()
    events = [
        event
        for chunk in GOOGLE_STREAM
        for event in GoogleGenerateConverter().stream_response_from_provider(
            cast(Any, chunk), context=context
        )
    ]

    start = next(e for e in events if e["type"] == "tool_call_start")
    finish = next(e for e in events if e["type"] == "finish")
    assert start["tool_type"] == "intrinsic"
    assert start["intrinsic_kind"] == "google_search"
    assert finish["finish_reason"]["reason"] == "stop"


# ---------------------------------------------------------------------------
# The stream reader agrees with the JSON reader
# ---------------------------------------------------------------------------


def test_the_stream_reader_classifies_the_hosted_call_like_the_json_reader():
    ir = _ir_response(_google_pipeline([_hosted_search_tool()]))
    events = _stream_ir_events(_google_pipeline([_hosted_search_tool()]))

    start = next(e for e in events if e["type"] == "tool_call_start")
    delta = next(e for e in events if e["type"] == "tool_call_delta")
    finish = next(e for e in events if e["type"] == "finish")
    assert start["tool_type"] == _first_part(ir)["tool_type"] == "intrinsic"
    assert start["intrinsic_kind"] == "google_search"
    assert delta["intrinsic_kind"] == "google_search"
    assert finish["finish_reason"]["reason"] == _reason(ir) == "stop"


def test_the_stream_reader_discriminates_a_client_function_like_the_json_reader():
    ir = _ir_response(_google_pipeline([_client_function()]))
    events = _stream_ir_events(_google_pipeline([_client_function()]))

    start = next(e for e in events if e["type"] == "tool_call_start")
    finish = next(e for e in events if e["type"] == "finish")
    assert start.get("tool_type") != "intrinsic"
    assert "intrinsic_kind" not in start
    assert finish["finish_reason"]["reason"] == _reason(ir) == "tool_calls"


# ---------------------------------------------------------------------------
# A Google-origin hosted call round-trips to Google (same-format route)
# ---------------------------------------------------------------------------


def _google_same_format() -> ConversionPipeline:
    """A Google-client → Google-upstream pipeline whose request leg has run."""
    pipe = ConversionPipeline("google_generate", "google_generate")
    pipe.convert_request(dict(GOOGLE_REQUEST))
    return pipe


def _ir_response_message(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """A minimal IR response carrying *parts* in one assistant turn."""
    return {
        "id": "r1",
        "object": "response",
        "created": 1,
        "model": "gemini-2.5-flash",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": parts},
                "finish_reason": {"reason": "stop"},
            }
        ],
    }


def test_a_hosted_call_round_trips_to_a_google_response():
    """The same-format route returns the turn it received, not an empty one."""
    out = _google_same_format().convert_response(cast(Any, GOOGLE_RESPONSE))

    parts = out["candidates"][0]["content"]["parts"]
    assert len(parts) == 1
    func_call = parts[0]["functionCall"]
    assert func_call["name"] == "googleSearch"
    assert func_call["args"] == {"q": "x"}
    assert func_call["id"].startswith("call_")


def test_a_hosted_call_keeps_the_thought_signature_it_arrived_with():
    response = {
        **GOOGLE_RESPONSE,
        "candidates": [
            {
                **GOOGLE_RESPONSE["candidates"][0],
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "googleSearch",
                                "args": {"q": "x"},
                            },
                            "thoughtSignature": "sig-hosted-1",
                        }
                    ],
                },
            }
        ],
    }
    pipe = _google_same_format()
    ir: dict[str, Any] = {}
    out = pipe.convert_response(cast(Any, response), on_ir_ready=ir.update)

    part = ir["choices"][0]["message"]["content"][0]
    assert part["provider_metadata"]["google"]["thought_signature"] == "sig-hosted-1"
    assert out["candidates"][0]["content"]["parts"][0]["thoughtSignature"] == (
        "sig-hosted-1"
    )


def test_a_hosted_call_round_trips_through_a_google_stream():
    """SSE agrees with the JSON writer: the turn is not emptied there either."""
    pipe = _google_same_format()
    processor = pipe.create_stream_processor()
    chunks = [
        chunk
        for source in GOOGLE_STREAM
        for chunk in processor.process_chunk(cast(Any, source))
    ]

    parts = [
        part
        for chunk in chunks
        for candidate in chunk.get("candidates", [])
        for part in candidate.get("content", {}).get("parts", [])
    ]
    func_calls = [part["functionCall"] for part in parts if "functionCall" in part]
    assert len(func_calls) == 1
    assert func_calls[0]["name"] == "googleSearch"
    assert func_calls[0]["args"] == {"q": "x"}
    assert func_calls[0]["id"].startswith("call_")


def test_a_replayed_hosted_call_stays_native_in_google_history():
    """Master replayed it as a ``functionCall``; the origin marker keeps that.

    Without it the history pass degrades the call into a client function named
    ``google_search`` — one the client never declared — and the orphan fixer
    then invents a result for it.
    """
    ir: dict[str, Any] = {}
    _google_same_format().convert_response(
        cast(Any, GOOGLE_RESPONSE), on_ir_ready=ir.update
    )
    history_message = ir["choices"][0]["message"]

    replay = ConversionPipeline("google_generate", "google_generate")

    def inject_history(ir_request: dict[str, Any]) -> None:
        ir_request["messages"] = [
            {"role": "user", "content": [{"type": "text", "text": "search the web"}]},
            history_message,
        ]

    out = replay.convert_request(
        cast(Any, dict(GOOGLE_REQUEST)), on_ir_ready=inject_history
    )

    contents = out["contents"]
    assert [content["role"] for content in contents] == ["user", "model"]
    parts = contents[1]["parts"]
    assert [next(iter(part)) for part in parts] == ["functionCall"]
    assert parts[0]["functionCall"]["name"] == "googleSearch"
    assert parts[0]["functionCall"]["id"].startswith("call_")
    assert UNANSWERED_TOOL_CALL not in json.dumps(out)
    assert not replay.warnings


def test_a_google_search_from_another_origin_still_drops_on_google():
    """Only the dialect that produced the ``functionCall`` may write it back."""
    part: dict[str, Any] = {
        "type": "tool_call",
        "tool_call_id": "s1",
        "tool_name": "googleSearch",
        "tool_input": {"q": "x"},
        "tool_type": "intrinsic",
        "intrinsic_kind": "google_search",
        "provider_metadata": {
            "_native": {
                "base": "google_interactions",
                "part": {"functionCall": {"name": "googleSearch", "args": {"q": "x"}}},
            }
        },
    }

    out = GoogleGenerateConverter().response_to_provider(
        cast(Any, _ir_response_message([part]))
    )

    assert out["candidates"][0]["content"]["parts"] == []


def test_a_foreign_google_search_in_history_is_degraded_not_replayed():
    """The origin gate bites on the history leg too."""
    part: dict[str, Any] = {
        "type": "tool_call",
        "tool_call_id": "s1",
        "tool_name": "googleSearch",
        "tool_input": {"q": "x"},
        "tool_type": "intrinsic",
        "intrinsic_kind": "google_search",
        "provider_metadata": {
            "_native": {
                "base": "open_responses",
                "part": {"functionCall": {"name": "googleSearch", "args": {"q": "x"}}},
            }
        },
    }
    pipe = ConversionPipeline("google_generate", "google_generate")

    def inject_history(ir_request: dict[str, Any]) -> None:
        ir_request["messages"] = [
            {"role": "user", "content": [{"type": "text", "text": "search the web"}]},
            {"role": "assistant", "content": [part]},
        ]

    out = pipe.convert_request(
        cast(Any, dict(GOOGLE_REQUEST)), on_ir_ready=inject_history
    )

    parts = out["contents"][1]["parts"]
    assert parts[0]["functionCall"]["name"] == "google_search"


# ==================== The replay a real client sends ====================


def _replayed_body(tools: Any) -> dict[str, Any]:
    """The ordinary Gemini history shape: the client replays the model turn."""
    return {
        "model": "gemini-2.5-flash",
        "contents": [
            {"role": "user", "parts": [{"text": "search rust async"}]},
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "googleSearch", "args": {"q": "x"}}}
                ],
            },
            {"role": "user", "parts": [{"text": "thanks"}]},
        ],
        "tools": tools,
    }


def test_a_replayed_hosted_turn_is_read_as_a_server_call():
    """The history reader classifies from the tools the same body declares."""
    ir = GoogleGenerateConverter().request_from_provider(
        cast(Any, _replayed_body([{"googleSearch": {}}]))
    )

    message = cast(dict[str, Any], ir["messages"][1])
    part = cast(dict[str, Any], message["content"][0])
    assert part["tool_type"] == "intrinsic"
    assert part["intrinsic_kind"] == "google_search"


def test_a_replayed_hosted_turn_survives_the_real_round_trip():
    """No synthetic result is written into the history a client replayed.

    The earlier test injected the IR the *response* leg produced, which the
    ``_native`` marker makes look correct without the reader ever seeing the
    JSON a client sends.  This one goes through the wire, twice.
    """
    body = _replayed_body([{"googleSearch": {}}])

    forwarded = ConversionPipeline(
        "google_generate", "google_generate"
    ).convert_request(json.loads(json.dumps(body)))
    wire = json.dumps(forwarded, ensure_ascii=False)
    assert UNANSWERED_TOOL_CALL not in wire
    assert "googleSearch" in wire

    # And the forwarded body, read back, is stable.
    converter = GoogleGenerateConverter()
    again = converter.request_from_provider(cast(Any, forwarded))
    rebuilt, _ = converter.request_to_provider(again)
    assert UNANSWERED_TOOL_CALL not in json.dumps(rebuilt, ensure_ascii=False)


def test_a_declared_client_function_wins_over_the_hosted_table():
    body = _replayed_body(
        [{"functionDeclarations": [{"name": "googleSearch", "parameters": {}}]}]
    )

    ir = GoogleGenerateConverter().request_from_provider(cast(Any, body))

    message = cast(dict[str, Any], ir["messages"][1])
    part = cast(dict[str, Any], message["content"][0])
    assert part["tool_type"] == "function"


def test_a_body_without_the_hosted_tool_reads_it_as_a_client_call():
    body = _replayed_body(None)

    ir = GoogleGenerateConverter().request_from_provider(cast(Any, body))

    message = cast(dict[str, Any], ir["messages"][1])
    part = cast(dict[str, Any], message["content"][0])
    assert part["tool_type"] == "function"
