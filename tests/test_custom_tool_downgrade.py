"""Tests for custom tool downgrade/restore across non-supporting upstreams.

When a provider does not support custom tools (``supports_custom_tools=False``),
the pipeline downgrades them to functions on the request path and restores
them on the response path.
"""

import copy
import json
from typing import Any

import pytest

from llm_rosetta.capabilities import (
    enforce_custom_tools,
    get_custom_tool_names,
    restore_custom_tool_calls,
    unwrap_custom_tool_input,
)
from llm_rosetta.pipeline import ConversionPipeline
from llm_rosetta.shims.provider_shim import (
    ProviderShim,
    register_shim,
    unregister_shim,
)

PATCH_TEXT = "*** Begin Patch\n*** Add File: a.txt\n+hi\n*** End Patch"

CUSTOM_TOOL_IR = {
    "type": "custom",
    "name": "apply_patch",
    "description": "Apply a V4A patch.",
    "parameters": {},
    "metadata": {
        "format": {"type": "grammar", "syntax": "lark"},
    },
}

FUNCTION_TOOL_IR = {
    "type": "function",
    "name": "get_weather",
    "description": "Get weather.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
}


def _make_shim(supports: bool, name: str = "__test_custom_tool__") -> ProviderShim:
    return ProviderShim(
        name=name,
        base="openai_chat",
        supports_custom_tools=supports,
    )


# ---------------------------------------------------------------------------
# enforce_custom_tools
# ---------------------------------------------------------------------------


class TestEnforceCustomTools:
    def test_noop_when_shim_supports(self):
        shim = _make_shim(supports=True)
        ir = {"tools": [dict(CUSTOM_TOOL_IR)]}
        result = enforce_custom_tools(ir, shim=shim)
        assert result["tools"][0]["type"] == "custom"

    def test_noop_when_shim_is_none(self):
        ir = {"tools": [dict(CUSTOM_TOOL_IR)]}
        result = enforce_custom_tools(ir, shim=None)
        assert result["tools"][0]["type"] == "custom"

    def test_noop_when_no_custom_tools(self):
        shim = _make_shim(supports=False)
        ir = {"tools": [dict(FUNCTION_TOOL_IR)]}
        result = enforce_custom_tools(ir, shim=shim)
        assert result["tools"][0]["type"] == "function"
        assert "provider_type" not in (result["tools"][0].get("metadata") or {})

    def test_downgrades_custom_to_function(self):
        shim = _make_shim(supports=False)
        tool = dict(CUSTOM_TOOL_IR)
        tool["metadata"] = copy.deepcopy(CUSTOM_TOOL_IR.get("metadata", {}))
        ir = {"tools": [tool]}
        result = enforce_custom_tools(ir, shim=shim)

        t = result["tools"][0]
        assert t["type"] == "function"
        assert t["metadata"]["provider_type"] == "custom"
        assert t["parameters"]["properties"]["input"]["type"] == "string"
        assert t["parameters"]["required"] == ["input"]

    def test_preserves_existing_parameters(self):
        shim = _make_shim(supports=False)
        tool = dict(CUSTOM_TOOL_IR)
        tool["metadata"] = copy.deepcopy(CUSTOM_TOOL_IR.get("metadata", {}))
        custom_params = {"type": "object", "properties": {"code": {"type": "string"}}}
        tool["parameters"] = custom_params
        ir = {"tools": [tool]}
        result = enforce_custom_tools(ir, shim=shim)

        assert result["tools"][0]["parameters"] == custom_params

    def test_description_includes_format_hint(self):
        shim = _make_shim(supports=False)
        tool = dict(CUSTOM_TOOL_IR)
        tool["metadata"] = copy.deepcopy(CUSTOM_TOOL_IR.get("metadata", {}))
        ir = {"tools": [tool]}
        result = enforce_custom_tools(ir, shim=shim)

        assert (
            "[Output format: grammar, syntax: lark]"
            in result["tools"][0]["description"]
        )

    def test_downgrades_tool_choice(self):
        shim = _make_shim(supports=False)
        tool = dict(CUSTOM_TOOL_IR)
        tool["metadata"] = copy.deepcopy(CUSTOM_TOOL_IR.get("metadata", {}))
        ir = {
            "tools": [tool],
            "tool_choice": {
                "mode": "tool",
                "tool_name": "apply_patch",
                "tool_type": "custom",
            },
        }
        result = enforce_custom_tools(ir, shim=shim)

        assert "tool_type" not in result["tool_choice"]
        assert result["tool_choice"]["tool_name"] == "apply_patch"

    def test_mixed_tools_only_downgrades_custom(self):
        shim = _make_shim(supports=False)
        custom = dict(CUSTOM_TOOL_IR)
        custom["metadata"] = copy.deepcopy(CUSTOM_TOOL_IR.get("metadata", {}))
        func = dict(FUNCTION_TOOL_IR)
        ir = {"tools": [custom, func]}
        result = enforce_custom_tools(ir, shim=shim)

        assert result["tools"][0]["type"] == "function"
        assert result["tools"][0]["metadata"]["provider_type"] == "custom"
        assert result["tools"][1]["type"] == "function"
        assert "provider_type" not in (result["tools"][1].get("metadata") or {})


# ---------------------------------------------------------------------------
# get_custom_tool_names
# ---------------------------------------------------------------------------


class TestGetCustomToolNames:
    def test_returns_downgraded_names(self):
        ir = {
            "tools": [
                {"name": "apply_patch", "metadata": {"provider_type": "custom"}},
                {"name": "get_weather", "metadata": {}},
            ]
        }
        assert get_custom_tool_names(ir) == frozenset({"apply_patch"})

    def test_empty_when_no_custom(self):
        ir = {"tools": [{"name": "fn", "metadata": {}}]}
        assert get_custom_tool_names(ir) == frozenset()

    def test_empty_when_no_tools(self):
        assert get_custom_tool_names({}) == frozenset()


# ---------------------------------------------------------------------------
# restore_custom_tool_calls
# ---------------------------------------------------------------------------


class TestRestoreCustomToolCalls:
    def test_restores_tool_type(self):
        ir_response = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {
                                "type": "tool_call",
                                "tool_name": "apply_patch",
                                "tool_type": "function",
                                "tool_input": {"input": PATCH_TEXT},
                            }
                        ]
                    }
                }
            ]
        }
        restore_custom_tool_calls(
            ir_response, custom_tool_names=frozenset({"apply_patch"})
        )
        assert (
            ir_response["choices"][0]["message"]["content"][0]["tool_type"] == "custom"
        )

    def test_noop_when_empty_set(self):
        ir_response = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {
                                "type": "tool_call",
                                "tool_name": "apply_patch",
                                "tool_type": "function",
                            }
                        ]
                    }
                }
            ]
        }
        restore_custom_tool_calls(ir_response, custom_tool_names=frozenset())
        assert (
            ir_response["choices"][0]["message"]["content"][0]["tool_type"]
            == "function"
        )

    def test_only_restores_matching_names(self):
        ir_response = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {
                                "type": "tool_call",
                                "tool_name": "apply_patch",
                                "tool_type": "function",
                            },
                            {
                                "type": "tool_call",
                                "tool_name": "get_weather",
                                "tool_type": "function",
                            },
                        ]
                    }
                }
            ]
        }
        restore_custom_tool_calls(
            ir_response, custom_tool_names=frozenset({"apply_patch"})
        )
        assert (
            ir_response["choices"][0]["message"]["content"][0]["tool_type"] == "custom"
        )
        assert (
            ir_response["choices"][0]["message"]["content"][1]["tool_type"]
            == "function"
        )


# ---------------------------------------------------------------------------
# unwrap_custom_tool_input
# ---------------------------------------------------------------------------


class TestUnwrapCustomToolInput:
    def test_unwraps_input_key(self):
        assert unwrap_custom_tool_input(json.dumps({"input": PATCH_TEXT})) == PATCH_TEXT

    def test_passthrough_non_json(self):
        assert unwrap_custom_tool_input("not json") == "not json"

    def test_passthrough_multi_key_dict(self):
        raw = json.dumps({"input": "a", "other": "b"})
        assert unwrap_custom_tool_input(raw) == raw

    def test_passthrough_non_input_key(self):
        raw = json.dumps({"code": "print(1)"})
        assert unwrap_custom_tool_input(raw) == raw


# ---------------------------------------------------------------------------
# Pipeline integration: non-streaming
# ---------------------------------------------------------------------------


class TestPipelineNonStreamingRoundtrip:
    """Full request + response through pipeline with a non-supporting shim."""

    @pytest.fixture(autouse=True)
    def _register_shim(self):
        shim = _make_shim(supports=False, name="__test_ns_custom__")
        register_shim(shim)
        yield
        unregister_shim("__test_ns_custom__")

    def _request(self) -> dict[str, Any]:
        return {
            "model": "test-model",
            "tools": [
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "Apply a V4A patch.",
                    "format": {
                        "type": "grammar",
                        "grammar": {"definition": "start: /.+/s", "syntax": "lark"},
                    },
                }
            ],
            "tool_choice": {"type": "custom", "custom": {"name": "apply_patch"}},
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "create a.txt"}],
                }
            ],
        }

    def _chat_completion(self) -> dict[str, Any]:
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "apply_patch",
                                    "arguments": json.dumps({"input": PATCH_TEXT}),
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    def test_custom_tool_call_roundtrip(self):
        pipe = ConversionPipeline(
            "openai_responses", "openai_chat", shim="__test_ns_custom__"
        )
        pipe.convert_request(self._request())
        out = pipe.convert_response(self._chat_completion())

        items = out.get("output", [])
        custom_items = [i for i in items if i.get("type") == "custom_tool_call"]
        assert len(custom_items) == 1
        assert custom_items[0]["input"] == PATCH_TEXT
        assert custom_items[0]["name"] == "apply_patch"


# ---------------------------------------------------------------------------
# Pipeline integration: streaming
# ---------------------------------------------------------------------------


class TestPipelineStreamingRoundtrip:
    """Streaming request + response through pipeline with a non-supporting shim."""

    @pytest.fixture(autouse=True)
    def _register_shim(self):
        shim = _make_shim(supports=False, name="__test_stream_custom__")
        register_shim(shim)
        yield
        unregister_shim("__test_stream_custom__")

    def _request(self) -> dict[str, Any]:
        return {
            "model": "test-model",
            "tools": [
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "Apply a V4A patch.",
                    "format": {
                        "type": "grammar",
                        "grammar": {"definition": "start: /.+/s", "syntax": "lark"},
                    },
                }
            ],
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "create a.txt"}],
                }
            ],
        }

    def _chat_chunks(self) -> list[dict[str, Any]]:
        def chunk(delta: dict, finish: str | None = None) -> dict[str, Any]:
            return {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "test-model",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }

        args = json.dumps({"input": PATCH_TEXT})
        half = len(args) // 2
        return [
            chunk(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "apply_patch", "arguments": ""},
                        }
                    ],
                }
            ),
            chunk(
                {"tool_calls": [{"index": 0, "function": {"arguments": args[:half]}}]}
            ),
            chunk(
                {"tool_calls": [{"index": 0, "function": {"arguments": args[half:]}}]}
            ),
            chunk({}, finish="tool_calls"),
        ]

    def test_streaming_emits_custom_tool_call(self):
        pipe = ConversionPipeline(
            "openai_responses", "openai_chat", shim="__test_stream_custom__"
        )
        pipe.convert_request(self._request())
        proc = pipe.create_stream_processor()

        events: list[dict[str, Any]] = []
        for chunk in self._chat_chunks():
            events.extend(proc.process_chunk(chunk))

        done_items = [
            e["item"]
            for e in events
            if isinstance(e, dict) and e.get("type", "").endswith("output_item.done")
        ]
        custom_items = [d for d in done_items if d.get("type") == "custom_tool_call"]
        assert len(custom_items) == 1
        assert custom_items[0]["input"] == PATCH_TEXT

        event_types = {e.get("type") for e in events if isinstance(e, dict)}
        assert "response.custom_tool_call_input.done" in event_types
        assert "response.function_call_arguments.done" not in event_types


# ---------------------------------------------------------------------------
# ProviderShim field
# ---------------------------------------------------------------------------


class TestProviderShimCustomToolField:
    def test_default_false(self):
        shim = ProviderShim(name="test", base="openai_chat")
        assert shim.supports_custom_tools is False

    def test_explicit_true(self):
        shim = ProviderShim(name="test", base="openai_chat", supports_custom_tools=True)
        assert shim.supports_custom_tools is True

    def test_explicit_false(self):
        shim = ProviderShim(
            name="test", base="openai_chat", supports_custom_tools=False
        )
        assert shim.supports_custom_tools is False
