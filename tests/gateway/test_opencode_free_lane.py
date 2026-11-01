"""Tests for the OpenCode Zen free-lane shim and its helpers."""

from __future__ import annotations

import importlib
import re
import time

import pytest

from llm_rosetta_gateway.providers import build_provider_info
from llm_rosetta_gateway.transport.provider_info import opencode_identifier
from llm_rosetta.shims import get_shim

_TRANSFORMS = importlib.import_module(
    "llm_rosetta.shims.providers.opencode.llm.zen_free.transforms"
)


@pytest.fixture(autouse=True)
def _fresh_request_state():
    """Each test starts with no carried-over call state."""
    _TRANSFORMS._INJECTED.set(False)
    _TRANSFORMS._STATE.set(None)
    yield


class TestFreeLaneShim:
    def test_declares_the_client_shape(self):
        shim = get_shim("opencode_zen_free--chat_completions")
        assert shim is not None
        conn = shim.connection
        assert conn.keyless
        assert conn.stream_required
        assert conn.session_header == "x-opencode-session"
        assert conn.session_id_format == "opencode"
        assert conn.request_header == "x-opencode-request"
        assert conn.headers is not None
        assert conn.headers["x-opencode-client"] == "desktop"
        assert conn.headers["User-Agent"].startswith("opencode/")

    def test_provider_info_is_built_keyless(self):
        info = build_provider_info(
            "chat_completions", {}, shim_name="opencode_zen_free--chat_completions"
        )
        assert info.keyless
        assert info.stream_required
        assert info.ready  # keyless providers are always ready
        assert "x-opencode-client" in info.extra_headers


class TestOpenCodeIdentifier:
    def test_shape_and_stability(self):
        first = opencode_identifier("ses", "conversation-1")
        assert re.fullmatch(r"ses_[0-9a-f]{12}[0-9A-Za-z]{14}", first)
        assert first == opencode_identifier("ses", "conversation-1")
        assert first != opencode_identifier("ses", "conversation-2")

    def test_encoded_timestamp_is_recent(self):
        value = opencode_identifier("ses", "conversation-1")
        head = value[4:16]
        # The low three hex characters are fixed by the bitwise-NOT layout.
        assert head.endswith("ffe")
        payload = int(head, 16)
        ms36 = (~payload & ((1 << 48) - 1)) >> 12
        modulus = 1 << 36
        now_ms = int(time.time() * 1000)
        # The decoder sees 36 bits; the nearest matching instant must be
        # within the measured +/-30 day validity window.
        nearest = ms36 + round((now_ms - ms36) / modulus) * modulus
        assert abs(now_ms - nearest) < 30 * 24 * 3600 * 1000

    def test_request_identifiers_differ(self):
        info = build_provider_info(
            "chat_completions", {}, shim_name="opencode_zen_free--chat_completions"
        )
        first = info.request_headers()
        second = info.request_headers()
        assert first.keys() == {"x-opencode-request"}
        assert first != second
        assert re.fullmatch(
            r"msg_[0-9a-f]{12}[0-9A-Za-z]{14}", first["x-opencode-request"]
        )


class TestReadToolGate:
    def test_injects_a_read_tool_when_absent(self):
        body = _TRANSFORMS.ensure_read_tool(
            {
                "model": "m",
                "messages": [],
                "tools": [{"type": "function", "function": {"name": "bash"}}],
            }
        )
        names = [tool["function"]["name"] for tool in body["tools"]]
        assert names == ["bash", "read"]
        assert body["tools"][-1]["function"]["parameters"]["type"] == "object"
        # The declaration exists for the gate, not for the model: it must say
        # so, or the model treats it as a real file reader and calls it.
        description = body["tools"][-1]["function"]["description"].lower()
        assert "placeholder" in description
        assert "never call" in description

    def test_leaves_a_client_read_tool_alone(self):
        tools = [{"type": "function", "function": {"name": "read", "parameters": {}}}]
        body = _TRANSFORMS.ensure_read_tool({"tools": tools})
        assert body["tools"] == tools

    def test_injected_tool_calls_are_dropped(self):
        _TRANSFORMS.ensure_read_tool({"model": "m"})
        chunk = {
            "id": "chunk-drop-1",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {
                                    "name": "read",
                                    "arguments": '{"filePath"',
                                },
                            }
                        ]
                    },
                }
            ],
        }
        _TRANSFORMS.drop_injected_read_calls(chunk)
        assert chunk["choices"][0]["delta"]["tool_calls"] == []

        fragment = {
            "id": "chunk-drop-1",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": ":1}"}}]
                    },
                }
            ],
        }
        _TRANSFORMS.drop_injected_read_calls(fragment)
        assert fragment["choices"][0]["delta"]["tool_calls"] == []

    def test_client_owned_read_calls_survive(self):
        _TRANSFORMS.ensure_read_tool(
            {"tools": [{"type": "function", "function": {"name": "read"}}]}
        )
        chunk = {
            "id": "chunk-keep-1",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"name": "read", "arguments": "{}"},
                            }
                        ]
                    },
                }
            ],
        }
        _TRANSFORMS.drop_injected_read_calls(chunk)
        assert len(chunk["choices"][0]["delta"]["tool_calls"]) == 1

    def test_a_stream_that_only_called_the_decoy_finishes_with_stop(self):
        """The client must not wait for a result the gateway dropped."""
        _TRANSFORMS.ensure_read_tool({"model": "m"})
        call = {
            "id": "chunk-stop-1",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"name": "read", "arguments": "{}"},
                            }
                        ]
                    },
                }
            ],
        }
        _TRANSFORMS.drop_injected_read_calls(call)
        # A conformant finish chunk carries no `delta` at all.
        finish = {
            "id": "chunk-stop-1",
            "choices": [{"index": 0, "finish_reason": "tool_calls"}],
        }
        _TRANSFORMS.drop_injected_read_calls(finish)
        assert finish["choices"][0]["finish_reason"] == "stop"

    def test_a_kept_call_keeps_the_tool_finish(self):
        _TRANSFORMS.ensure_read_tool({"model": "m"})
        call = {
            "id": "chunk-mixed-1",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"name": "bash"}},
                            {"index": 1, "function": {"name": "read"}},
                        ]
                    },
                }
            ],
        }
        _TRANSFORMS.drop_injected_read_calls(call)
        assert [c["index"] for c in call["choices"][0]["delta"]["tool_calls"]] == [0]
        finish = {
            "id": "chunk-mixed-1",
            "choices": [{"index": 0, "finish_reason": "tool_calls"}],
        }
        _TRANSFORMS.drop_injected_read_calls(finish)
        assert finish["choices"][0]["finish_reason"] == "tool_calls"

    def test_each_choice_keeps_its_own_call_indexes(self):
        """Two choices number their tool calls independently."""
        _TRANSFORMS.ensure_read_tool({"model": "m"})
        call = {
            "id": "chunk-two-choices-1",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"name": "read"}},
                        ]
                    },
                },
                {
                    "index": 1,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"name": "bash"}},
                        ]
                    },
                },
            ],
        }
        _TRANSFORMS.drop_injected_read_calls(call)
        assert call["choices"][0]["delta"]["tool_calls"] == []
        assert [c["index"] for c in call["choices"][1]["delta"]["tool_calls"]] == [0]

        finish = {
            "id": "chunk-two-choices-1",
            "choices": [
                {"index": 0, "finish_reason": "tool_calls"},
                {"index": 1, "finish_reason": "tool_calls"},
            ],
        }
        _TRANSFORMS.drop_injected_read_calls(finish)
        assert finish["choices"][0]["finish_reason"] == "stop"
        assert finish["choices"][1]["finish_reason"] == "tool_calls"

    def test_a_finish_chunk_under_another_id_keeps_a_kept_call(self):
        """The state is the request's stream, not one upstream chunk id."""
        _TRANSFORMS.ensure_read_tool({"model": "m"})
        call = {
            "id": "chunk-kept-a",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"name": "bash"}}]
                    },
                }
            ],
        }
        _TRANSFORMS.drop_injected_read_calls(call)
        finish = {
            "id": "chunk-kept-b",
            "choices": [{"index": 0, "finish_reason": "tool_calls"}],
        }
        _TRANSFORMS.drop_injected_read_calls(finish)
        assert finish["choices"][0]["finish_reason"] == "tool_calls"

    def test_a_finish_chunk_under_another_id_still_stops_without_calls(self):
        _TRANSFORMS.ensure_read_tool({"model": "m"})
        call = {
            "id": "chunk-decoy-a",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"name": "read"}}]
                    },
                }
            ],
        }
        _TRANSFORMS.drop_injected_read_calls(call)
        finish = {
            "id": "chunk-decoy-b",
            "choices": [{"index": 0, "finish_reason": "tool_calls"}],
        }
        _TRANSFORMS.drop_injected_read_calls(finish)
        assert finish["choices"][0]["finish_reason"] == "stop"

    def test_call_state_does_not_leak_into_the_next_request(self):
        """A reused context must not carry the previous request's verdict."""
        # Request 1: the decoy only, at index 0.
        _TRANSFORMS.ensure_read_tool({"model": "m"})
        first = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"name": "read"}}]
                    },
                }
            ]
        }
        _TRANSFORMS.drop_injected_read_calls(first)
        # Request 2: a real call at the same index must survive.
        _TRANSFORMS.ensure_read_tool({"model": "m"})
        second = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"name": "bash"}}]
                    },
                }
            ]
        }
        _TRANSFORMS.drop_injected_read_calls(second)
        assert second["choices"][0]["delta"]["tool_calls"] == [
            {"index": 0, "function": {"name": "bash"}}
        ]
        finish = {"choices": [{"index": 0, "finish_reason": "tool_calls"}]}
        _TRANSFORMS.drop_injected_read_calls(finish)
        assert finish["choices"][0]["finish_reason"] == "tool_calls"

    def test_a_non_streaming_decoy_only_turn_finishes_with_stop(self):
        _TRANSFORMS.ensure_read_tool({"model": "m"})
        body = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {"name": "read", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
        _TRANSFORMS.drop_injected_read_calls(body)
        assert body["choices"][0]["message"]["tool_calls"] == []
        assert body["choices"][0]["finish_reason"] == "stop"

    def test_the_model_list_keeps_zero_cost_models(self):
        def priced(price_in, price_out):
            return {
                "cost": {
                    "text": {
                        "input": {"price": price_in},
                        "output": {"price": price_out},
                    }
                }
            }

        entries = [
            {"id": "big-pickle", **priced(0, 0)},
            {"id": "beam-free", **priced(0, 0)},
            {"id": "claude-opus-5", **priced(1, 5)},
            {"id": "half-free", **priced(0, 3)},
            # One stated side is not a free statement.
            {"id": "one-sided", "cost": {"text": {"input": {"price": 0}}}},
            # JSON numbers arrive as numbers; a boolean is not a price.
            # `False == 0`, so this is the value the guard exists for.
            {"id": "bool-price", **priced(False, 0)},
            # No cost block: the name the lane has used is the only signal.
            {"id": "legacy-free"},
            {"id": "legacy-paid"},
            {"id": 5, **priced(0, 0)},
            {"id": ""},
            {},
            "not-a-mapping",
        ]
        ids, upstream = _TRANSFORMS.model_list_transform(entries)
        assert ids == ["big-pickle", "beam-free", "legacy-free"]
        assert upstream == {}

    def test_the_shim_registers_the_model_filter(self):
        """The loader picks the hook up from the leaf transforms module."""
        from llm_rosetta.shims.providers import get_model_list_transform

        transform = get_model_list_transform("opencode_zen_free--chat_completions")
        assert transform is not None
        ids, upstream = transform([{"id": "beam-free"}, {"id": "claude-opus-5"}])
        assert ids == ["beam-free"]
        assert upstream == {}
