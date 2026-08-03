from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

from llm_rosetta.converters.anthropic.converter import AnthropicConverter
from llm_rosetta.converters.base import BaseConverter
from llm_rosetta.converters.base.context import ConversionContext, StreamContext
from llm_rosetta.pipeline import StreamProcessor
from llm_rosetta.converters.base.helpers.tool_orphan_fix import (
    fix_orphaned_tool_calls_ir,
)
from llm_rosetta.converters.openai_responses.converter import OpenAIResponsesConverter
from llm_rosetta.converters.openai_chat.converter import OpenAIChatConverter
from llm_rosetta.converters.google_genai.converter import GoogleGenAIConverter
from llm_rosetta.types.ir import (
    IRInputItem,
    IRResponse,
    ProviderPassthroughEvent,
    ProviderPassthroughItem,
    is_provider_passthrough_event,
    is_provider_passthrough_item,
)
from llm_rosetta.types.ir.validation import (
    validate_ir_request,
    validate_ir_response,
    validate_messages,
)


def _item(provider: str = "openai_responses", position: int = 1):
    return ProviderPassthroughItem(
        type="provider_passthrough_item",
        provider=provider,
        payload={"type": "vendor_event", "value": 1},
        position=position,
    )


class TestPassthroughTypes:
    def test_event_type_guard(self):
        event = ProviderPassthroughEvent(
            type="provider_passthrough",
            provider="anthropic",
            payload={"type": "ping"},
        )
        assert is_provider_passthrough_event(event)
        assert not is_provider_passthrough_event({"type": "text_delta", "text": "x"})

    def test_item_type_guard(self):
        item = _item()
        assert is_provider_passthrough_item(item)
        assert not is_provider_passthrough_item({"role": "user", "content": []})

    def test_request_validation_accepts_passthrough_item(self):
        request = validate_ir_request(
            {"model": "test", "messages": [_item(position=0)]}
        )
        item = cast(ProviderPassthroughItem, request["messages"][0])
        assert item["type"] == "provider_passthrough_item"

    def test_validate_messages_accepts_passthrough_item(self):
        messages = validate_messages([_item(position=0)])
        item = cast(ProviderPassthroughItem, messages[0])
        assert item["provider"] == "openai_responses"

    def test_response_validation_accepts_provider_passthrough_items(self):
        response = validate_ir_response(
            {
                "id": "resp_1",
                "object": "response",
                "created": 1,
                "model": "test",
                "choices": [],
                "provider_passthrough_items": [_item(position=0)],
            }
        )
        assert (
            response["provider_passthrough_items"][0]["provider"] == "openai_responses"
        )


class TestStreamPassthrough:
    def test_anthropic_ping_emits_passthrough(self):
        chunk = {"type": "ping"}
        events = AnthropicConverter().stream_response_from_provider(chunk)
        assert events == [
            {
                "type": "provider_passthrough",
                "provider": "anthropic",
                "payload": chunk,
            }
        ]
        passthrough = cast(ProviderPassthroughEvent, events[0])
        assert passthrough["payload"] is not chunk

    def test_responses_in_progress_dropped(self):
        """response.in_progress is dropped (re-synthesised on the to_provider side)."""
        chunk = {
            "type": "response.in_progress",
            "sequence_number": 3,
            "response": {"id": "resp_1", "status": "in_progress"},
        }
        events = OpenAIResponsesConverter().stream_response_from_provider(chunk)
        assert events == []

    def test_same_format_restores_copy(self):
        converter = AnthropicConverter()
        event = ProviderPassthroughEvent(
            type="provider_passthrough",
            provider="anthropic",
            payload={"type": "ping"},
        )
        restored = converter.stream_response_to_provider(event)
        assert restored == {"type": "ping"}
        assert restored is not event["payload"]

    def test_cross_format_drops_event(self):
        event = ProviderPassthroughEvent(
            type="provider_passthrough",
            provider="anthropic",
            payload={"type": "ping"},
        )
        ctx = StreamContext()
        assert OpenAIResponsesConverter().stream_response_to_provider(event, ctx) == {}
        upgraded = ctx.metadata.get("_responses_stream_ctx")
        assert upgraded is None or upgraded._sequence_number == 0

    def test_responses_same_format_rewrites_sequence_number(self):
        converter = OpenAIResponsesConverter()
        ctx = converter.create_stream_context()
        event = ProviderPassthroughEvent(
            type="provider_passthrough",
            provider="openai_responses",
            payload={"type": "response.in_progress", "sequence_number": 99},
        )
        restored = cast(
            dict[str, Any], converter.stream_response_to_provider(event, ctx)
        )
        assert restored["sequence_number"] == 1
        assert event["payload"]["sequence_number"] == 99


class TestNonStreamPassthroughHelpers:
    def test_message_ops_restore_same_provider(self):
        items, warnings = OpenAIResponsesConverter().message_ops.ir_messages_to_p(
            [_item(position=0)], target_provider="openai_responses"
        )
        assert items == [{"type": "vendor_event", "value": 1}]
        assert warnings == []

    def test_message_ops_drop_cross_provider_with_warning(self):
        messages, warnings = AnthropicConverter().message_ops.ir_messages_to_p(
            [_item(position=0)]
        )
        assert messages == []
        assert len(warnings) == 1

    def test_all_message_ops_restore_same_tag_and_drop_foreign(self):
        converters = [
            OpenAIResponsesConverter(),
            OpenAIChatConverter(),
            AnthropicConverter(),
            GoogleGenAIConverter(),
        ]
        for converter in converters:
            local = _item(provider=converter._CONVERTER_TAG, position=0)
            restored, warnings = converter.message_ops.ir_messages_to_p(
                [local], target_provider=converter._CONVERTER_TAG
            )
            assert restored == [{"type": "vendor_event", "value": 1}]
            assert warnings == []

            foreign = _item(provider="foreign", position=0)
            dropped, warnings = converter.message_ops.ir_messages_to_p(
                [foreign], target_provider=converter._CONVERTER_TAG
            )
            assert dropped == []
            assert len(warnings) == 1

    def test_same_position_merge_is_stable(self):
        from llm_rosetta.converters.base.passthrough import merge_provider_output_items

        first = _item(position=0)
        first["payload"] = {"type": "first"}
        second = _item(position=0)
        second["payload"] = {"type": "second"}
        merged, warnings = merge_provider_output_items(
            [{"type": "message"}],
            [first, second],
            target_provider="openai_responses",
        )
        assert [entry["type"] for entry in merged] == [
            "first",
            "second",
            "message",
        ]
        assert warnings == []

    def test_restore_same_provider_returns_copy(self):
        from llm_rosetta.converters.base.passthrough import (
            restore_provider_passthrough_item,
        )

        item = _item()
        payload, warning = restore_provider_passthrough_item(
            item, target_provider="openai_responses"
        )
        assert payload == item["payload"]
        assert payload is not item["payload"]
        assert warning is None

    def test_restore_cross_provider_warns_and_drops(self):
        from llm_rosetta.converters.base.passthrough import (
            restore_provider_passthrough_item,
        )

        payload, warning = restore_provider_passthrough_item(
            _item(), target_provider="anthropic"
        )
        assert payload is None
        assert warning is not None
        assert "openai_responses" in warning
        assert "anthropic" in warning

    def test_merge_output_items_by_position(self):
        from llm_rosetta.converters.base.passthrough import merge_provider_output_items

        portable = [
            {"type": "reasoning", "id": "r"},
            {"type": "function_call", "id": "f"},
        ]
        passthrough = [_item(position=1)]
        original = deepcopy(passthrough)
        merged, warnings = merge_provider_output_items(
            portable,
            passthrough,
            target_provider="openai_responses",
        )
        assert [entry["type"] for entry in merged] == [
            "reasoning",
            "vendor_event",
            "function_call",
        ]
        assert warnings == []
        assert passthrough == original

    def test_cross_provider_output_items_are_dropped(self):
        from llm_rosetta.converters.base.passthrough import merge_provider_output_items

        merged, warnings = merge_provider_output_items(
            [{"type": "message"}],
            [_item(position=0)],
            target_provider="anthropic",
        )
        assert merged == [{"type": "message"}]
        assert len(warnings) == 1

    def test_conversion_context_metadata_coexists(self):
        from llm_rosetta.converters.base.passthrough import merge_provider_output_items

        ctx = ConversionContext(options={"metadata_mode": "preserve"})
        ctx.store_response_extras({"billing": {"payer": "developer"}})
        ctx.store_output_items_meta([{"id": "msg_1"}])
        item = _item(position=0)

        merged, warnings = merge_provider_output_items(
            [{"type": "message"}],
            [item],
            target_provider="openai_responses",
        )

        assert [entry["type"] for entry in merged] == [
            "vendor_event",
            "message",
        ]
        assert warnings == []
        assert ctx.get_echo_fields() == {"billing": {"payer": "developer"}}
        assert ctx.get_output_items_meta() == [{"id": "msg_1"}]

    def test_orphan_fix_preserves_passthrough_item_position(self):
        passthrough = _item(position=1)
        messages: list[IRInputItem] = [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            passthrough,
        ]
        fixed = fix_orphaned_tool_calls_ir(messages)
        assert fixed == messages
        assert fixed[1] is passthrough

    def test_responses_nonstream_captures_and_restores_unknown_output_item(self):
        converter = OpenAIResponsesConverter()
        ctx = ConversionContext(options={"metadata_mode": "preserve"})
        provider_response = {
            "id": "resp_1",
            "object": "response",
            "created_at": 1,
            "model": "test",
            "status": "completed",
            "output": [
                {"type": "vendor_event", "id": "vendor_1", "value": 1},
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "hello",
                            "annotations": [],
                        }
                    ],
                },
            ],
        }
        ir_response = converter.response_from_provider(provider_response, context=ctx)
        assert ir_response["provider_passthrough_items"] == [
            {
                "type": "provider_passthrough_item",
                "provider": "openai_responses",
                "payload": provider_response["output"][0],
                "position": 0,
            }
        ]

        restored = converter.response_to_provider(ir_response, context=ctx)
        assert [item["type"] for item in restored["output"]] == [
            "vendor_event",
            "message",
        ]
        assert restored["output"][1]["id"] == "msg_1"

    def test_distinct_positions_merge_against_original_output_indices(self):
        from llm_rosetta.converters.base.passthrough import merge_provider_output_items

        first = _item(position=1)
        first["payload"] = {"type": "first"}
        second = _item(position=2)
        second["payload"] = {"type": "second"}
        merged, warnings = merge_provider_output_items(
            [{"type": "portable_0"}, {"type": "portable_3"}],
            [first, second],
            target_provider="openai_responses",
        )
        assert [item["type"] for item in merged] == [
            "portable_0",
            "first",
            "second",
            "portable_3",
        ]
        assert warnings == []

    def test_response_converters_restore_matching_passthrough_items(self):
        converters_and_keys: list[tuple[BaseConverter, str]] = [
            (OpenAIResponsesConverter(), "output"),
            (OpenAIChatConverter(), "choices"),
            (AnthropicConverter(), "content"),
            (GoogleGenAIConverter(), "candidates"),
        ]
        for converter, output_key in converters_and_keys:
            item = _item(provider=converter._CONVERTER_TAG, position=0)
            response: IRResponse = {
                "id": "resp_1",
                "object": "response",
                "created": 1,
                "model": "test",
                "choices": [],
                "provider_passthrough_items": [item],
            }
            restored = converter.response_to_provider(response)
            assert restored[output_key] == [{"type": "vendor_event", "value": 1}]

    def test_response_converter_drops_foreign_items_with_warning(self):
        ctx = ConversionContext()
        response: IRResponse = {
            "id": "resp_1",
            "object": "response",
            "created": 1,
            "model": "test",
            "choices": [],
            "provider_passthrough_items": [
                _item(provider="openai_responses", position=0)
            ],
        }
        restored = AnthropicConverter().response_to_provider(response, context=ctx)
        assert restored["content"] == []
        assert len(ctx.warnings) == 1

    def test_missing_position_appends(self):
        from llm_rosetta.converters.base.passthrough import merge_provider_output_items

        item = _item()
        item.pop("position")
        merged, warnings = merge_provider_output_items(
            [{"type": "message"}],
            [item],
            target_provider="openai_responses",
        )
        assert [entry["type"] for entry in merged] == ["message", "vendor_event"]
        assert warnings == []


class TestStreamProcessorPassthrough:
    def test_same_format_forwards_passthrough(self):
        """Anthropic ping events pass through same-format round-trip."""
        target = AnthropicConverter()
        source = AnthropicConverter()
        processor = StreamProcessor(
            target_converter=target,
            source_converter=source,
            from_ctx=target.create_stream_context(),
            to_ctx=source.create_stream_context(),
        )
        result = processor.process_chunk({"type": "ping"})
        assert result[0]["type"] == "ping"

    def test_cross_format_drops_passthrough(self):
        target = OpenAIResponsesConverter()
        source = AnthropicConverter()
        processor = StreamProcessor(
            target_converter=target,
            source_converter=source,
            from_ctx=target.create_stream_context(),
            to_ctx=source.create_stream_context(),
        )
        result = processor.process_chunk(
            {
                "type": "response.in_progress",
                "sequence_number": 8,
                "response": {"id": "resp_1", "status": "in_progress"},
            }
        )
        assert result == []
