"""Tests for the OpenCode Zen free-lane shim and its helpers."""

from __future__ import annotations

import importlib
import re
import time

from llm_rosetta_gateway.providers import build_provider_info
from llm_rosetta_gateway.transport.provider_info import opencode_identifier
from llm_rosetta.shims import get_shim

_TRANSFORMS = importlib.import_module(
    "llm_rosetta.shims.providers.opencode.zen_free.transforms"
)


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
