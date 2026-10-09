"""Client-declared intrinsic tools: `intrinsic__<kind>` in OpenAI Chat format."""

import pytest

from llm_rosetta.converters.anthropic import AnthropicConverter
from llm_rosetta.converters.google_generate import GoogleGenerateConverter
from llm_rosetta.converters.google_interactions import GoogleInteractionsConverter
from llm_rosetta.converters.openai_chat import OpenAIChatConverter
from llm_rosetta.converters.openai_responses import OpenAIResponsesConverter
from llm_rosetta.converters.base.tools.intrinsic import (
    intrinsic_name_to_kind,
    make_intrinsic_tool_definition,
)
from llm_rosetta.pipeline import ConversionPipeline


class TestNameConvention:
    def test_kind_extracted(self):
        assert intrinsic_name_to_kind("intrinsic__web_search") == "web_search"
        assert intrinsic_name_to_kind("intrinsic__code_execution") == "code_execution"

    def test_non_intrinsic_name(self):
        assert intrinsic_name_to_kind("get_weather") is None
        assert intrinsic_name_to_kind("intrinsic") is None

    def test_make_definition(self):
        d = make_intrinsic_tool_definition("web_search")
        assert d["type"] == "intrinsic"
        assert d["metadata"]["intrinsic_kind"] == "web_search"
        assert d["name"] == "intrinsic__web_search"


class TestOpenAIChatSource:
    def test_declared_intrinsic_promoted(self):
        tool = {
            "type": "function",
            "function": {"name": "intrinsic__web_search", "parameters": {}},
        }
        ir = OpenAIChatConverter().tool_ops.p_tool_definition_to_ir(tool)
        assert ir["type"] == "intrinsic"
        assert ir["metadata"]["intrinsic_kind"] == "web_search"

    def test_plain_function_unaffected(self):
        tool = {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {}},
        }
        ir = OpenAIChatConverter().tool_ops.p_tool_definition_to_ir(tool)
        assert ir["type"] == "function"


class TestTargetEmission:
    def _ir(self, kind):
        return make_intrinsic_tool_definition(kind)

    def test_anthropic_web_search(self):
        out = AnthropicConverter().tool_ops.ir_tool_definition_to_p(
            self._ir("web_search")
        )
        assert out["type"] == "web_search_20250305"
        assert out["name"] == "web_search"

    def test_anthropic_unsupported_dropped(self):
        assert (
            AnthropicConverter().tool_ops.ir_tool_definition_to_p(
                self._ir("google_maps")
            )
            == {}
        )

    def test_openai_responses_web_search(self):
        out = OpenAIResponsesConverter().tool_ops.ir_tool_definition_to_p(
            self._ir("web_search")
        )
        assert out == {"type": "web_search"}

    def test_openai_responses_code_interpreter(self):
        out = OpenAIResponsesConverter().tool_ops.ir_tool_definition_to_p(
            self._ir("code_interpreter")
        )
        assert out["type"] == "code_interpreter"
        assert out["container"] == {"type": "auto"}

    def test_google_generate_code_execution(self):
        out = GoogleGenerateConverter().tool_ops.ir_tool_definition_to_p(
            self._ir("code_execution")
        )
        assert out == {"code_execution": {}}

    def test_google_interactions_google_search(self):
        out = GoogleInteractionsConverter().tool_ops.ir_tool_definition_to_p(
            self._ir("google_search")
        )
        assert out == {"type": "google_search"}

    def test_openai_chat_drops_intrinsic(self):
        assert (
            OpenAIChatConverter().tool_ops.ir_tool_definition_to_p(
                self._ir("web_search")
            )
            == {}
        )


CHAT_REQ = {
    "model": "gpt-4o-mini",
    "max_tokens": 50,
    "messages": [{"role": "user", "content": "hi"}],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "intrinsic__web_search",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ],
}


class TestPipelineEndToEnd:
    @pytest.mark.parametrize(
        "provider,expected",
        [
            ("anthropic", {"type": "web_search_20250305", "name": "web_search"}),
            ("openai_responses", {"type": "web_search"}),
        ],
    )
    def test_chat_request_produces_native_server_tool(self, provider, expected):
        pipe = ConversionPipeline("openai_chat", provider, upstream_model="m")
        body = pipe.convert_request(dict(CHAT_REQ))
        tools = body.get("tools", body.get("config", {}).get("tools"))
        assert any(t == expected for t in tools), tools

    def test_chat_target_drops_intrinsic(self):
        """openai_chat → openai_chat: no native equivalent, dropped."""
        pipe = ConversionPipeline("anthropic", "openai_chat", upstream_model="m")
        ir = {
            "model": "m",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [make_intrinsic_tool_definition("web_search")],
        }
        body = pipe.convert_request(ir)
        tools = body.get("tools")
        assert not tools or all(t.get("type") != "intrinsic" for t in tools)

    def test_unsupported_kind_dropped_not_malformed(self):
        """A kind the target lacks must be dropped, not emitted as {}."""
        pipe = ConversionPipeline("openai_chat", "anthropic", upstream_model="m")
        req = {
            "model": "m",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "intrinsic__google_maps", "parameters": {}},
                }
            ],
        }
        body = pipe.convert_request(req)
        tools = body.get("tools") or []
        assert tools == [] or all("google_maps" not in str(t) for t in tools)
