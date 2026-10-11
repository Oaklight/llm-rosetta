"""Intrinsic tool *history* is mapped by the target converter.

There is no global degrade step: the target converter emits an intrinsic call
natively where it has a form for the kind, and falls back to a plain function
call (with the intrinsic's natural name) otherwise.  See #839.
"""

import json
from typing import cast

from llm_rosetta.converters.anthropic import AnthropicConverter
from llm_rosetta.converters.chat_completions import ChatCompletionsConverter
from llm_rosetta.converters.openai_responses import OpenAIResponsesConverter
from llm_rosetta.pipeline import ConversionPipeline
from llm_rosetta.types.ir import IRRequest


def _ir_with_intrinsic_history(kind: str = "web_search") -> IRRequest:
    return cast(
        IRRequest,
        {
            "model": "m",
            "max_tokens": 50,
            "max_output_tokens": 50,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Search Iceland"}],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "s1",
                            "tool_name": kind,
                            "tool_input": {"query": "Iceland"},
                            "tool_type": "intrinsic",
                            "provider_metadata": {"intrinsic_kind": kind},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_call_id": "s1",
                            "result": "~400k",
                            "tool_type": "intrinsic",
                            "provider_metadata": {"intrinsic_kind": kind},
                        }
                    ],
                },
            ],
        },
    )


class TestConverterMapsIntrinsicHistory:
    def test_responses_emits_native_web_search_call(self):
        body, _ = OpenAIResponsesConverter().request_to_provider(
            _ir_with_intrinsic_history()
        )
        assert "web_search_call" in json.dumps(body)

    def test_anthropic_emits_native_server_tool_use(self):
        body, _ = AnthropicConverter().request_to_provider(_ir_with_intrinsic_history())
        assert "server_tool_use" in json.dumps(body)

    def test_openai_chat_falls_back_to_natural_function_name(self):
        body, _ = ChatCompletionsConverter().request_to_provider(
            _ir_with_intrinsic_history()
        )
        names = [
            tc["function"]["name"]
            for m in body["messages"]
            for tc in m.get("tool_calls", [])
        ]
        assert "web_search" in names
        # No internal naming convention leaks to the wire.
        assert not any("_intrinsic" in n for n in names)


class TestPipelineKeepsIntrinsicHistory:
    """The pipeline must not degrade intrinsic history before the converter."""

    ANTHROPIC_REQ = {
        "model": "m",
        "max_tokens": 50,
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        "messages": [
            {"role": "user", "content": "Search Iceland"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "server_tool_use",
                        "id": "s1",
                        "name": "web_search",
                        "input": {"query": "Iceland"},
                    },
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "s1",
                        "content": [
                            {
                                "type": "web_search_result",
                                "url": "u",
                                "title": "t",
                                "encrypted_content": "e",
                            }
                        ],
                    },
                    {"type": "text", "text": "~400k"},
                ],
            },
            {"role": "user", "content": "more"},
        ],
    }

    def test_anthropic_to_responses_history_stays_native(self):
        from llm_rosetta.shims.providers import load_providers

        shims = {s.name: s for s in load_providers()}
        pipe = ConversionPipeline(
            "anthropic",
            "openai_responses",
            target_shim=shims["openai_responses"],
            upstream_model="gpt-5-nano",
        )
        body = pipe.convert_request(json.loads(json.dumps(self.ANTHROPIC_REQ)))
        s = json.dumps(body)
        assert "web_search_call" in s  # native, not degraded
        assert "_intrinsic" not in s


class TestHistoryIndependentOfShim:
    """Definitions are shim-gated; history is not (it is context).

    Documents/locks the deliberate split flagged in review: a target shim that
    drops the `web_search` *definition* still gets a native `web_search_call`
    in history rather than a silently mangled part.
    """

    IR = _ir_with_intrinsic_history("web_search")

    def test_definition_dropped_but_history_native(self):
        from llm_rosetta.shims.provider_shim import ProviderShim, ToolsConfig

        # A responses shim that declares NO intrinsic tools.
        bare = ProviderShim(name="x", base="openai_responses", tools=ToolsConfig())
        pipe = ConversionPipeline("anthropic", "openai_responses", target_shim=bare)
        # feed an anthropic provider request (the shim is the target's)
        req = {
            "model": "m",
            "max_tokens": 50,
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            "messages": [
                {"role": "user", "content": "Search Iceland"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "server_tool_use",
                            "id": "s1",
                            "name": "web_search",
                            "input": {"query": "Iceland"},
                        },
                        {
                            "type": "web_search_tool_result",
                            "tool_use_id": "s1",
                            "content": [
                                {
                                    "type": "web_search_result",
                                    "url": "u",
                                    "title": "t",
                                    "encrypted_content": "e",
                                }
                            ],
                        },
                        {"type": "text", "text": "~400k"},
                    ],
                },
                {"role": "user", "content": "more"},
            ],
        }
        body = pipe.convert_request(req)
        # Definition: gated out (shim declares nothing).
        assert not body.get("tools")
        # History: still mapped natively (shim-independent).
        assert "web_search_call" in json.dumps(body)
