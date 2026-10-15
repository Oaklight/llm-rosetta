"""An assistant turn keeps its inline server-tool result (#965).

Anthropic executes server tools (web search, code execution) inside the
assistant turn and returns ``server_tool_use`` + ``*_tool_result`` blocks in
that same turn's content.  That is not the client-result shape the
``role: "tool"`` normalization exists for: the server produced the result in
the turn, so the turn owns it.

Before this, ``_p_message_to_ir`` split *any* message holding a
``tool_result``, assistant included, and handed a two-message list to
``choices[0].message`` -- which the IR response model rejects with
``ValidationError: Value at 'choices[0].message' does not match any variant of
SystemMessage | UserMessage | AssistantMessage | ToolMessage``.  The write path
then dropped the result block even when a hand-built IR did validate.

The reader now splits only ``role: "user"``; the writer renders an inline
server result through the same ``_native``-restore / IR-rebuild logic the
streaming path uses.  A converter that cannot render a server result drops it
without raising.
"""

import json
from typing import Any, cast

import pytest

from llm_rosetta.converters.chat_completions import ChatCompletionsConverter
from llm_rosetta.converters.google_generate import GoogleGenerateConverter
from llm_rosetta.converters.messages import MessagesConverter
from llm_rosetta.converters.messages.content_ops import MessagesContentOps
from llm_rosetta.converters.messages.message_ops import MessagesMessageOps
from llm_rosetta.converters.messages.tool_ops import MessagesToolOps
from llm_rosetta.converters.base.tools.orphan_fix import fix_orphaned_tool_calls_ir
from llm_rosetta.pipeline import ConversionPipeline
from llm_rosetta.converters.open_responses import OpenResponsesConverter
from llm_rosetta.types.ir import IRInputItem, Message
from llm_rosetta.types.ir.request import IRRequest

_SERVER_CALL: dict[str, Any] = {
    "type": "server_tool_use",
    "id": "srvtoolu_1",
    "name": "web_search",
    "input": {"query": "tokyo"},
}

_WEB_RESULT: dict[str, Any] = {
    "type": "web_search_tool_result",
    "tool_use_id": "srvtoolu_1",
    "content": [
        {
            "type": "web_search_result",
            "title": "Example",
            "url": "https://example.com",
            "encrypted_content": "enc",
        }
    ],
}

_TEXT: dict[str, Any] = {"type": "text", "text": "Tokyo is big."}


def _response(content: list[dict[str, Any]]) -> dict[str, Any]:
    """A minimal Anthropic response body wrapping ``content``."""
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "content": content,
    }


def _round_trip(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Provider content → IR response → provider content."""
    converter = MessagesConverter()
    ir = converter.response_from_provider(_response(content))
    out = converter.response_to_provider(ir)
    return cast(list[dict[str, Any]], out["content"])


class TestFourCaseTable:
    """The isolated cases from the issue, on the response path."""

    @pytest.mark.parametrize(
        "content",
        [
            [_TEXT],
            [_SERVER_CALL],
            [_WEB_RESULT],
            [_SERVER_CALL, _WEB_RESULT, _TEXT],
        ],
        ids=["text", "server_tool_use", "tool_result_alone", "full_turn"],
    )
    def test_read_then_write_does_not_raise_and_keeps_order(
        self, content: list[dict[str, Any]]
    ):
        out = _round_trip(content)

        assert [block["type"] for block in out] == [b["type"] for b in content]


class TestSameDialectRoundTrip:
    """A captured turn survives read → write block-for-block, in order."""

    def test_full_turn_is_reproduced_block_for_block(self):
        content = [_SERVER_CALL, _WEB_RESULT, _TEXT]

        out = _round_trip(content)

        # Same blocks, same order.
        assert [block["type"] for block in out] == [
            "server_tool_use",
            "web_search_tool_result",
            "text",
        ]
        # The result block is restored from the stored native payload, so the
        # server result is byte-for-byte what came in.
        assert out[1] == _WEB_RESULT
        assert out[0] == _SERVER_CALL
        # Text blocks pick up the response-spec ``citations: null``; the text
        # itself is unchanged.
        assert out[2] == {**_TEXT, "citations": None}

    def test_result_alone_stays_an_assistant_turn(self):
        """A lone ``*_tool_result`` is not split into a ``tool`` message."""
        converter = MessagesConverter()
        ir = converter.response_from_provider(_response([_WEB_RESULT]))

        message = cast(dict[str, Any], ir["choices"][0]["message"])
        assert message["role"] == "assistant"
        assert [p["type"] for p in message["content"]] == ["tool_result"]

    def test_message_writer_renders_inline_result(self):
        """The request-history writer reaches the same server-result writer."""
        message_ops = MessagesMessageOps(MessagesContentOps(), MessagesToolOps())
        result_block = MessagesToolOps.p_server_tool_result_to_ir(_WEB_RESULT)
        ir_messages = cast(
            list[Message],
            [
                {
                    "role": "assistant",
                    "content": [
                        MessagesToolOps.p_tool_call_to_ir(_SERVER_CALL),
                        result_block,
                        {"type": "text", "text": "Tokyo is big."},
                    ],
                }
            ],
        )

        converted, warnings = message_ops.ir_messages_to_p(ir_messages)

        assert len(converted) == 1
        assert [block["type"] for block in converted[0]["content"]] == [
            "server_tool_use",
            "web_search_tool_result",
            "text",
        ]
        assert warnings == []

    def test_message_writer_drops_a_client_result_inline_without_raising(self):
        """A function result cannot sit inline in an Anthropic assistant turn."""
        message_ops = MessagesMessageOps(MessagesContentOps(), MessagesToolOps())
        ir_messages = cast(
            list[Message],
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "here"},
                        {
                            "type": "tool_result",
                            "tool_call_id": "call_1",
                            "result": "42",
                        },
                    ],
                }
            ],
        )

        converted, warnings = message_ops.ir_messages_to_p(ir_messages)

        assert [block["type"] for block in converted[0]["content"]] == ["text"]
        assert any("assistant message" in w for w in warnings)


class TestCrossFormatDegrade:
    """A converter that cannot render a server result drops it, no raise."""

    @pytest.mark.parametrize(
        "converter_cls",
        [ChatCompletionsConverter, OpenResponsesConverter],
        ids=["chat_completions", "open_responses"],
    )
    def test_server_result_degrades_without_raising(self, converter_cls):
        content = [_SERVER_CALL, _WEB_RESULT, _TEXT]
        ir = MessagesConverter().response_from_provider(_response(content))

        assert converter_cls().response_to_provider(ir) is not None


class TestRequestSideRuleUnchanged:
    """A client tool_result in a user turn still normalizes to ``tool``."""

    def test_user_turn_splits_into_tool_message(self):
        converter = MessagesConverter()
        ir = converter.request_from_provider(
            {
                "model": "claude-haiku-4-5",
                "max_tokens": 16,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t1",
                                "content": "42",
                            },
                            {"type": "text", "text": "thanks"},
                        ],
                    }
                ],
            }
        )

        messages = cast(list[dict[str, Any]], ir["messages"])
        roles = [(m["role"], [p["type"] for p in m["content"]]) for m in messages]
        assert roles == [("tool", ["tool_result"]), ("user", ["text"])]

    def test_user_turn_with_only_results_collapses_to_tool(self):
        converter = MessagesConverter()
        ir = converter.request_from_provider(
            {
                "model": "claude-haiku-4-5",
                "max_tokens": 16,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t1",
                                "content": "42",
                            }
                        ],
                    }
                ],
            }
        )

        messages = cast(list[dict[str, Any]], ir["messages"])
        assert [m["role"] for m in messages] == ["tool"]


_PLACEHOLDER = "[No output available yet]"


def _request_messages() -> list[IRInputItem]:
    """The request-history shape: an unrelated user turn, then the server turn."""
    return cast(
        list[IRInputItem],
        [
            {"role": "user", "content": [{"type": "text", "text": "search"}]},
            {
                "role": "assistant",
                "content": [
                    MessagesToolOps.p_tool_call_to_ir(_SERVER_CALL),
                    MessagesToolOps.p_server_tool_result_to_ir(_WEB_RESULT),
                    {"type": "text", "text": "done"},
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "thanks"}]},
        ],
    )


class TestRequestLegDoesNotFabricateAResult:
    """An inline server result answers its call; no placeholder is injected.

    ``fix_orphaned_tool_calls_ir`` used to count an answer only when it sat in a
    ``role: "tool"`` message.  Keeping the server result inline in the assistant
    turn left the call looking unanswered, so the fixer injected a synthetic
    ``role: "tool"`` placeholder next to the real block.
    """

    def test_orphan_fixer_leaves_the_list_alone(self):
        out = fix_orphaned_tool_calls_ir(_request_messages())

        assert _PLACEHOLDER not in json.dumps(out)
        assert len(out) == 3

    @pytest.mark.parametrize(
        "converter_cls",
        [
            MessagesConverter,
            ChatCompletionsConverter,
            GoogleGenerateConverter,
            OpenResponsesConverter,
        ],
        ids=["messages", "chat_completions", "google_generate", "open_responses"],
    )
    def test_request_to_provider_injects_no_placeholder(self, converter_cls):
        out, _ = converter_cls().request_to_provider(
            cast(
                IRRequest,
                {"model": "claude-haiku-4-5", "messages": _request_messages()},
            )
        )

        assert _PLACEHOLDER not in json.dumps(out)

    def test_same_dialect_keeps_the_real_payload(self):
        out, _ = MessagesConverter().request_to_provider(
            cast(
                IRRequest,
                {"model": "claude-haiku-4-5", "messages": _request_messages()},
            )
        )

        blocks = out["messages"][1]["content"]
        assert [b["type"] for b in blocks] == [
            "server_tool_use",
            "web_search_tool_result",
            "text",
        ]
        assert blocks[1] == _WEB_RESULT

    def test_a_genuine_orphan_still_gets_a_placeholder(self):
        """The fix must not disarm the orphan injection entirely.

        A *client* call still owes the client a result; only a server call is
        answered by the provider that ran it.
        """
        messages = cast(
            list[IRInputItem],
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "toolu_1",
                            "tool_name": "get_weather",
                            "tool_input": {"city": "tokyo"},
                            "tool_type": "function",
                        }
                    ],
                }
            ],
        )

        out = fix_orphaned_tool_calls_ir(messages)

        assert _PLACEHOLDER in json.dumps(out)


class TestPipelineRequestLeg:
    """The production pipeline lands no placeholder in any target format.

    ``fix_orphaned_tool_calls_ir`` is called from every writer, so the fix in
    the shared ``_collect_ir_tool_ids`` has to hold for all of them, not just
    messages-to-messages.
    """

    def _source_body(self) -> dict[str, Any]:
        return {
            "model": "claude-haiku-4-5",
            "max_tokens": 32,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "search"}]},
                {
                    "role": "assistant",
                    "content": [
                        _SERVER_CALL,
                        _WEB_RESULT,
                        {"type": "text", "text": "done"},
                    ],
                },
                {"role": "user", "content": [{"type": "text", "text": "thanks"}]},
            ],
        }

    @pytest.mark.parametrize(
        "target",
        ["messages", "chat_completions", "google_generate", "open_responses"],
    )
    def test_pipeline_request_has_no_placeholder(self, target):
        pipeline = ConversionPipeline("messages", target)

        out = pipeline.convert_request(self._source_body())

        assert _PLACEHOLDER not in json.dumps(out)

    def test_pipeline_same_dialect_keeps_the_real_result(self):
        pipeline = ConversionPipeline("messages", "messages")

        out = pipeline.convert_request(self._source_body())

        blocks = out["messages"][1]["content"]
        assert [b["type"] for b in blocks] == [
            "server_tool_use",
            "web_search_tool_result",
            "text",
        ]
        assert blocks[1] == _WEB_RESULT


class TestClientResultInAssistantTurnStillNormalizes:
    """A *client* result inline in an assistant turn is still a tool message.

    The split keys on the result kind, not the role.  Only a server result
    stays inline; a client result is normalized to ``role: "tool"`` exactly as
    on base, so the payload reaches the model instead of being dropped and
    replaced by the orphan repair's placeholder.
    """

    _CALL: dict[str, Any] = {
        "type": "tool_use",
        "id": "tu_1",
        "name": "get_weather",
        "input": {"city": "tokyo"},
    }
    _CLIENT_RESULT: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": "tu_1",
        "content": "REAL-PAYLOAD",
    }

    def _source_body(self) -> dict[str, Any]:
        return {
            "model": "claude-haiku-4-5",
            "max_tokens": 32,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": [self._CALL, self._CLIENT_RESULT]},
                {"role": "user", "content": [{"type": "text", "text": "thanks"}]},
            ],
        }

    def test_reader_splits_the_client_result_out(self):
        ir = MessagesConverter().request_from_provider(self._source_body())

        messages = cast(list[dict[str, Any]], ir["messages"])
        roles = [(m["role"], [p["type"] for p in m["content"]]) for m in messages]
        assert roles == [
            ("user", ["text"]),
            ("assistant", ["tool_call"]),
            ("tool", ["tool_result"]),
            ("user", ["text"]),
        ]

    def test_writer_keeps_the_payload_and_injects_no_placeholder(self):
        converter = MessagesConverter()
        ir = converter.request_from_provider(self._source_body())

        out, _ = converter.request_to_provider(ir)

        rendered = json.dumps(out)
        assert "REAL-PAYLOAD" in rendered
        assert _PLACEHOLDER not in rendered
        # The assistant turn holds only the call; the result rides in a user
        # tool_result (Anthropic's client-result shape).
        blocks = out["messages"][1]["content"]
        assert [b["type"] for b in blocks] == ["tool_use"]
        assert out["messages"][2]["content"][0]["content"] == "REAL-PAYLOAD"

    @pytest.mark.parametrize(
        "target", ["messages", "chat_completions", "google_generate"]
    )
    def test_pipeline_keeps_the_payload(self, target):
        pipeline = ConversionPipeline("messages", target)

        rendered = json.dumps(pipeline.convert_request(self._source_body()))

        assert "REAL-PAYLOAD" in rendered
        assert _PLACEHOLDER not in rendered


class TestGenericStreamingResultFallback:
    """The generic ``tool_result`` fallback keeps ``is_error``/``cache_control``."""

    def test_fallback_preserves_the_generic_fields(self):
        converter = MessagesConverter()
        event = {
            "type": "tool_result",
            "tool_call_id": "toolu_1",
            "result": "boom",
            "is_error": True,
            "cache_hint": {"type": "ephemeral"},
        }

        block = converter._ir_tool_result_block(cast(Any, event))

        assert block == {
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": "boom",
            "is_error": True,
            "cache_control": {"type": "ephemeral"},
        }
