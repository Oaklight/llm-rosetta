"""Open Responses spec extensions on top of the OpenAI Responses wire format.

Covers the extensions that distinguish ``open_responses`` from
``openai_responses`` (issue #493): the ``allowed_tools`` ``tool_choice``
variant, ``compaction`` items, raw reasoning ``content``, slug-prefixed
extension items, and the ``phase`` field on messages.
"""

from __future__ import annotations

from llm_rosetta import convert
from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.converters.openai_responses import (
    OpenAIResponsesConverter,
    OpenResponsesConverter,
)
from llm_rosetta.pipeline import ConversionPipeline


def _user(text: str = "hi") -> dict:
    return {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": text}],
    }


def _tools() -> list[dict]:
    return [
        {
            "type": "function",
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "type": "function",
            "name": "get_time",
            "parameters": {"type": "object", "properties": {}},
        },
    ]


_ALLOWED_TOOLS = {
    "type": "allowed_tools",
    "tools": [{"type": "function", "name": "get_weather"}],
    "mode": "auto",
}


class TestAllowedToolsToolChoice:
    """The spec puts ``allowed_tools`` inside ``tool_choice``."""

    def _body(self) -> dict:
        return {
            "model": "m",
            "input": [_user()],
            "tools": _tools(),
            "tool_choice": _ALLOWED_TOOLS,
        }

    def test_lossless_same_format_round_trip(self):
        out = convert(
            self._body(),
            "open_responses",
            source_provider="open_responses",
            baseline=False,
        )
        assert out["tool_choice"] == _ALLOWED_TOOLS
        # Not duplicated as a top-level field.
        assert "allowed_tools" not in out
        assert "_open_responses_allowed_tools" not in out

    def test_lossless_to_openai_profile(self):
        out = convert(
            self._body(),
            "openai_responses",
            source_provider="open_responses",
            baseline=False,
        )
        assert out["tool_choice"] == _ALLOWED_TOOLS

    def test_ir_stashes_object_and_maps_required_to_any(self):
        body = self._body()
        body["tool_choice"] = dict(_ALLOWED_TOOLS, mode="required")
        conv = OpenResponsesConverter()
        ir = conv.request_from_provider(body, context=ConversionContext())
        assert ir["tool_choice"]["mode"] == "any"
        assert (
            ir["provider_extensions"]["_open_responses_allowed_tools"]
            == (body["tool_choice"])
        )

    def test_degrades_across_format_keeping_mode(self):
        # The restriction has no IR equivalent; the mode degrades to the
        # selector and no tool_choice-shaped object leaks into the target.
        out = convert(
            self._body(),
            "anthropic",
            source_provider="open_responses",
            baseline=False,
        )
        assert out["tool_choice"] == {"type": "auto"}
        # The provider-bound object must not escape onto another dialect's wire.
        assert "allowed_tools" not in out
        assert "_open_responses_allowed_tools" not in out

    def test_no_leak_onto_openai_chat(self):
        out = convert(
            self._body(),
            "openai_chat",
            source_provider="open_responses",
            baseline=False,
        )
        assert out["tool_choice"] == "auto"
        assert "allowed_tools" not in out
        assert "_open_responses_allowed_tools" not in out


class TestCompaction:
    _COMPACTION = {"type": "compaction", "id": "cmp_1", "encrypted_content": "blob"}

    def _body(self) -> dict:
        return {
            "model": "m",
            "input": [_user("hi"), dict(self._COMPACTION), _user("again")],
        }

    def test_same_format_preserves_item_verbatim(self):
        out = convert(
            self._body(),
            "open_responses",
            source_provider="open_responses",
            baseline=False,
        )
        assert [i["type"] for i in out["input"]] == ["message", "compaction", "message"]
        assert out["input"][1] == self._COMPACTION

    def test_cross_format_drops_with_warning(self):
        pipe = ConversionPipeline("open_responses", "anthropic", baseline=False)
        pipe.convert_request(self._body())
        assert any("Dropped provider passthrough item" in w for w in pipe.warnings), (
            pipe.warnings
        )

    def test_openai_profile_is_a_different_tag(self):
        # ``open_responses`` and ``openai_responses`` are distinct dialects, so
        # the provider-bound blob is dropped rather than replayed.
        out = convert(
            self._body(),
            "openai_responses",
            source_provider="open_responses",
            baseline=False,
        )
        assert [i["type"] for i in out["input"]] == ["message", "message"]

    def test_response_leg_preserves_item(self):
        conv = OpenResponsesConverter()
        ctx = ConversionContext()
        provider_response = {
            "id": "resp_1",
            "object": "response",
            "created_at": 1,
            "model": "m",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "ok", "annotations": []}
                    ],
                },
                {"type": "compaction", "id": "cmp_9", "encrypted_content": "blob"},
            ],
        }
        ir = conv.response_from_provider(provider_response, context=ctx)
        out = conv.response_to_provider(ir, context=ctx)
        assert [i["type"] for i in out["output"]] == ["message", "compaction"]


class TestReasoningContent:
    def test_raw_content_round_trips(self):
        conv = OpenResponsesConverter()
        ctx = ConversionContext()
        provider_response = {
            "id": "resp_1",
            "object": "response",
            "created_at": 1,
            "model": "m",
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "status": "completed",
                    "content": [{"type": "reasoning_text", "text": "think"}],
                    "summary": [],
                }
            ],
        }
        ir = conv.response_from_provider(provider_response, context=ctx)
        out = conv.response_to_provider(ir, context=ctx)
        reasoning = [i for i in out["output"] if i["type"] == "reasoning"][0]
        assert reasoning["content"] == [{"type": "reasoning_text", "text": "think"}]


class TestSlugItems:
    def test_slug_prefixed_item_survives(self):
        item = {
            "type": "openai:web_search_call",
            "id": "ws_1",
            "status": "completed",
            "action": {"type": "search", "query": "q"},
        }
        body = {"model": "m", "input": [_user(), dict(item)]}
        out = convert(
            body, "open_responses", source_provider="open_responses", baseline=False
        )
        assert any(i["type"] == "openai:web_search_call" for i in out["input"])


class TestPhase:
    def test_phase_preserved_on_response_message(self):
        conv = OpenResponsesConverter()
        ctx = ConversionContext()
        provider_response = {
            "id": "resp_1",
            "object": "response",
            "created_at": 1,
            "model": "m",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "phase": "commentary",
                    "content": [
                        {"type": "output_text", "text": "checking", "annotations": []}
                    ],
                }
            ],
        }
        ir = conv.response_from_provider(provider_response, context=ctx)
        out = conv.response_to_provider(ir, context=ctx)
        msg = [i for i in out["output"] if i["type"] == "message"][0]
        assert msg["phase"] == "commentary"


class TestStoreDefault:
    """The spec is stateless; the OpenAI profile defaults to server storage."""

    _RESPONSE = {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "model": "m",
        "status": "completed",
        "output": [],
    }

    def _round_trip(self, converter):
        ctx = ConversionContext(options={"metadata_mode": "preserve"})
        ir = converter.response_from_provider(self._RESPONSE, context=ctx)
        return converter.response_to_provider(ir, context=ctx)

    def test_base_omits_store(self):
        assert "store" not in self._round_trip(OpenResponsesConverter())

    def test_openai_profile_injects_store_true(self):
        assert self._round_trip(OpenAIResponsesConverter())["store"] is True
