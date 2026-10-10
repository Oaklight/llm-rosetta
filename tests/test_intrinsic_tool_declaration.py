"""Shim-declared intrinsic tools: expression and cross-format translation.

A provider opts into intrinsic (server-hosted) tools by declaring them in its
shim (``ToolsConfig.intrinsic_tools``).  Native formats declare them natively;
OpenAI Chat, which has no native server tools, may declare them either by tool
name (``{"type":"function","function":{"name":"web_search"}}``) or explicitly
(``{"type":"intrinsic","name":"web_search"}``).
"""

from llm_rosetta.capabilities import resolve_intrinsic_tools
from llm_rosetta.converters.anthropic import AnthropicConverter
from llm_rosetta.converters.google_generate import GoogleGenerateConverter
from llm_rosetta.converters.google_interactions import GoogleInteractionsConverter
from llm_rosetta.converters.openai_chat import OpenAIChatConverter
from llm_rosetta.converters.openai_responses import OpenAIResponsesConverter
from llm_rosetta.converters.base.tools.intrinsic import (
    get_definition_kind,
    make_intrinsic_tool_definition,
)
from llm_rosetta.pipeline import ConversionPipeline
from llm_rosetta.shims.provider_shim import ProviderShim, ToolsConfig


def _shim(base, kinds):
    return ProviderShim(
        name="test", base=base, tools=ToolsConfig(intrinsic_tools=tuple(kinds))
    )


# ── Source recognition (native declarations → IR intrinsic) ─────────────


class TestSourceRecognition:
    def test_anthropic_web_search(self):
        ir = AnthropicConverter().tool_ops.p_tool_definition_to_ir(
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}
        )
        assert ir["type"] == "intrinsic"
        assert get_definition_kind(ir) == "web_search"

    def test_anthropic_code_execution(self):
        ir = AnthropicConverter().tool_ops.p_tool_definition_to_ir(
            {"type": "code_execution_20250522", "name": "code_execution"}
        )
        assert ir["type"] == "intrinsic"
        assert get_definition_kind(ir) == "code_execution"

    def test_anthropic_plain_function_unaffected(self):
        ir = AnthropicConverter().tool_ops.p_tool_definition_to_ir(
            {"name": "get_weather", "input_schema": {}}
        )
        assert ir["type"] == "function"

    def test_responses_web_search(self):
        ir = OpenAIResponsesConverter().tool_ops.p_tool_definition_to_ir(
            {"type": "web_search"}
        )
        assert ir["type"] == "intrinsic"
        assert get_definition_kind(ir) == "web_search"

    def test_responses_code_interpreter(self):
        ir = OpenAIResponsesConverter().tool_ops.p_tool_definition_to_ir(
            {"type": "code_interpreter"}
        )
        assert get_definition_kind(ir) == "code_interpreter"

    def test_google_code_execution(self):
        ir = GoogleGenerateConverter().tool_ops.p_tool_definition_to_ir(
            {"code_execution": {}}
        )
        assert ir["type"] == "intrinsic"
        assert get_definition_kind(ir) == "code_execution"

    def test_google_search(self):
        ir = GoogleGenerateConverter().tool_ops.p_tool_definition_to_ir(
            {"google_search": {}}
        )
        assert get_definition_kind(ir) == "google_search"

    def test_gi_google_search(self):
        ir = GoogleInteractionsConverter().tool_ops.p_tool_to_ir(
            {"type": "google_search"}
        )
        assert ir["type"] == "intrinsic"
        assert get_definition_kind(ir) == "google_search"

    def test_chat_explicit_intrinsic(self):
        ir = OpenAIChatConverter().tool_ops.p_tool_definition_to_ir(
            {"type": "intrinsic", "name": "web_search"}
        )
        assert ir["type"] == "intrinsic"
        assert get_definition_kind(ir) == "web_search"

    def test_chat_plain_function_not_promoted_at_source(self):
        ir = OpenAIChatConverter().tool_ops.p_tool_definition_to_ir(
            {"type": "function", "function": {"name": "web_search"}}
        )
        # Bare-name promotion happens later, driven by the target shim.
        assert ir["type"] == "function"


# ── Target emission (IR intrinsic → native) ─────────────────────────────


class TestTargetEmission:
    def _ir(self, kind):
        return make_intrinsic_tool_definition(kind)

    def test_anthropic(self):
        out = AnthropicConverter().tool_ops.ir_tool_definition_to_p(
            self._ir("web_search")
        )
        assert out == {"type": "web_search_20250305", "name": "web_search"}

    def test_anthropic_unsupported_dropped(self):
        assert (
            AnthropicConverter().tool_ops.ir_tool_definition_to_p(
                self._ir("google_maps")
            )
            == {}
        )

    def test_responses(self):
        out = OpenAIResponsesConverter().tool_ops.ir_tool_definition_to_p(
            self._ir("web_search")
        )
        assert out == {"type": "web_search"}

    def test_google(self):
        out = GoogleGenerateConverter().tool_ops.ir_tool_definition_to_p(
            self._ir("code_execution")
        )
        assert out == {"code_execution": {}}

    def test_gi(self):
        out = GoogleInteractionsConverter().tool_ops.ir_tool_definition_to_p(
            self._ir("google_search")
        )
        assert out == {"type": "google_search"}

    def test_chat_drops(self):
        assert (
            OpenAIChatConverter().tool_ops.ir_tool_definition_to_p(
                self._ir("web_search")
            )
            == {}
        )


# ── Shim gating (resolve_intrinsic_tools) ───────────────────────────────


class TestResolve:
    def test_drops_unsupported_kind(self):
        ir = {"tools": [make_intrinsic_tool_definition("web_search")]}
        out = resolve_intrinsic_tools(ir, shim=_shim("google", ["code_execution"]))
        assert out["tools"] == []

    def test_keeps_supported_kind(self):
        ir = {"tools": [make_intrinsic_tool_definition("web_search")]}
        out = resolve_intrinsic_tools(ir, shim=_shim("anthropic", ["web_search"]))
        assert len(out["tools"]) == 1
        assert get_definition_kind(out["tools"][0]) == "web_search"

    def test_no_shim_drops_intrinsic(self):
        ir = {"tools": [make_intrinsic_tool_definition("web_search")]}
        out = resolve_intrinsic_tools(ir, shim=None)
        assert out["tools"] == []

    def test_same_format_untouched(self):
        ir = {"tools": [make_intrinsic_tool_definition("web_search")]}
        out = resolve_intrinsic_tools(ir, shim=None, same_format=True)
        assert out is ir

    def test_promotes_bare_name_when_allowed(self):
        ir = {
            "tools": [
                {
                    "type": "function",
                    "name": "web_search",
                    "description": "",
                    "parameters": {},
                }
            ]
        }
        out = resolve_intrinsic_tools(
            ir, shim=_shim("anthropic", ["web_search"]), allow_name_promotion=True
        )
        assert get_definition_kind(out["tools"][0]) == "web_search"
        assert out["tools"][0]["type"] == "intrinsic"

    def test_no_promote_when_disallowed(self):
        ir = {"tools": [{"type": "function", "name": "web_search", "parameters": {}}]}
        out = resolve_intrinsic_tools(ir, shim=_shim("anthropic", ["web_search"]))
        assert out["tools"][0]["type"] == "function"

    def test_promote_only_supported_names(self):
        ir = {"tools": [{"type": "function", "name": "get_weather", "parameters": {}}]}
        out = resolve_intrinsic_tools(
            ir, shim=_shim("anthropic", ["web_search"]), allow_name_promotion=True
        )
        assert out["tools"][0]["type"] == "function"


# ── End-to-end pipeline ─────────────────────────────────────────────────


class TestPipeline:
    def test_anthropic_web_search_to_google_is_dropped(self):
        """Anthropic web_search → google: google has no web_search kind."""
        pipe = ConversionPipeline(
            "anthropic",
            "google",
            target_shim=_shim("google", ["code_execution", "google_search"]),
        )
        body = pipe.convert_request(
            {
                "model": "m",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            }
        )
        tools = body.get("tools") or body.get("config", {}).get("tools") or []
        assert all("web_search" not in str(t) for t in tools)

    def test_anthropic_code_execution_to_google_maps(self):
        """Anthropic code_execution → google emits google's code_execution."""
        pipe = ConversionPipeline(
            "anthropic",
            "google",
            target_shim=_shim("google", ["code_execution", "google_search"]),
        )
        body = pipe.convert_request(
            {
                "model": "m",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {"type": "code_execution_20250522", "name": "code_execution"}
                ],
            }
        )
        tools = body.get("tools") or body.get("config", {}).get("tools") or []
        assert {"code_execution": {}} in tools

    def test_chat_bare_name_promoted_via_target_shim(self):
        pipe = ConversionPipeline(
            "openai_chat", "anthropic", target_shim=_shim("anthropic", ["web_search"])
        )
        body = pipe.convert_request(
            {
                "model": "m",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "web_search", "parameters": {}},
                    }
                ],
            }
        )
        assert {"type": "web_search_20250305", "name": "web_search"} in body["tools"]

    def test_no_shim_no_intrinsic(self):
        """Without a shim the provider supports nothing intrinsic."""
        pipe = ConversionPipeline("openai_chat", "anthropic")
        body = pipe.convert_request(
            {
                "model": "m",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "web_search", "parameters": {}},
                    }
                ],
            }
        )
        tools = body.get("tools") or []
        # No promotion, no native server tool: the tool stays a plain
        # function named web_search (Anthropic form has no "type").
        assert tools and all(
            "web_search_20250305" not in str(t) and t.get("type") != "intrinsic"
            for t in tools
        )
