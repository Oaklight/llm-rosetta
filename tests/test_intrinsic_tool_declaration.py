"""Shim-declared intrinsic tools: expression and cross-format translation.

A provider opts into intrinsic (server-hosted) tools by declaring them in its
shim (``ToolsConfig.intrinsic_tools``).  Native formats declare them natively;
OpenAI Chat, which has no native server tools, may declare them either by tool
name (``{"type":"function","function":{"name":"web_search"}}``) or explicitly
(``{"type":"intrinsic","name":"web_search"}``).
"""

from llm_rosetta.capabilities import resolve_intrinsic_tools
from llm_rosetta.converters.messages import MessagesConverter
from llm_rosetta.converters.google_generate import GoogleGenerateConverter
from llm_rosetta.converters.google_interactions import GoogleInteractionsConverter
from llm_rosetta.converters.chat_completions import ChatCompletionsConverter
from llm_rosetta.converters.open_responses import OpenAIResponsesConverter
from llm_rosetta.converters.base.tools.intrinsic import (
    get_definition_kind,
    make_intrinsic_tool_definition,
)
from llm_rosetta.pipeline import ConversionPipeline
from llm_rosetta.shims import resolve_shim
from llm_rosetta.shims.provider_shim import ProviderShim, ToolsConfig


def _shim(base, kinds):
    return ProviderShim(
        name="test", base=base, tools=ToolsConfig(intrinsic_tools=tuple(kinds))
    )


# ── Source recognition (native declarations → IR intrinsic) ─────────────


class TestSourceRecognition:
    def test_anthropic_web_search(self):
        ir = MessagesConverter().tool_ops.p_tool_definition_to_ir(
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}
        )
        assert ir["type"] == "intrinsic"
        assert get_definition_kind(ir) == "web_search"

    def test_anthropic_code_execution(self):
        ir = MessagesConverter().tool_ops.p_tool_definition_to_ir(
            {"type": "code_execution_20250522", "name": "code_execution"}
        )
        assert ir["type"] == "intrinsic"
        assert get_definition_kind(ir) == "code_execution"

    def test_anthropic_plain_function_unaffected(self):
        ir = MessagesConverter().tool_ops.p_tool_definition_to_ir(
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
        ir = ChatCompletionsConverter().tool_ops.p_tool_definition_to_ir(
            {"type": "intrinsic", "name": "web_search"}
        )
        assert ir["type"] == "intrinsic"
        assert get_definition_kind(ir) == "web_search"

    def test_chat_plain_function_not_promoted_at_source(self):
        ir = ChatCompletionsConverter().tool_ops.p_tool_definition_to_ir(
            {"type": "function", "function": {"name": "web_search"}}
        )
        # Bare-name promotion happens later, driven by the target shim.
        assert ir["type"] == "function"


# ── Target emission (IR intrinsic → native) ─────────────────────────────


class TestTargetEmission:
    def _ir(self, kind):
        return make_intrinsic_tool_definition(kind)

    def test_anthropic(self):
        out = MessagesConverter().tool_ops.ir_tool_definition_to_p(
            self._ir("web_search")
        )
        assert out == {"type": "web_search_20250305", "name": "web_search"}

    def test_anthropic_unsupported_dropped(self):
        assert (
            MessagesConverter().tool_ops.ir_tool_definition_to_p(
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
            ChatCompletionsConverter().tool_ops.ir_tool_definition_to_p(
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
        out = resolve_intrinsic_tools(ir, shim=_shim("messages", ["web_search"]))
        assert len(out["tools"]) == 1
        assert get_definition_kind(out["tools"][0]) == "web_search"

    def test_no_shim_drops_intrinsic(self):
        ir = {"tools": [make_intrinsic_tool_definition("web_search")]}
        out = resolve_intrinsic_tools(ir, shim=None)
        assert out["tools"] == []

    def test_shim_name_resolves_here_too(self):
        """A shim *name* is resolved in the gate itself, like the siblings.

        Pins this branch on its own: the pipeline also hands over an object, so
        the end-to-end case would stay green if this half were deleted as
        looking unreachable.  ``enforce_custom_tools`` and ``enforce_reasoning``
        resolve names the same way.
        """
        ir = {"tools": [make_intrinsic_tool_definition("web_search")]}
        out = resolve_intrinsic_tools(ir, shim="anthropic--messages")
        assert len(out["tools"]) == 1
        assert get_definition_kind(out["tools"][0]) == "web_search"

    def test_unresolvable_shim_name_is_no_shim(self):
        ir = {"tools": [make_intrinsic_tool_definition("web_search")]}
        out = resolve_intrinsic_tools(ir, shim="not-a-registered-shim")
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
            ir, shim=_shim("messages", ["web_search"]), allow_name_promotion=True
        )
        assert get_definition_kind(out["tools"][0]) == "web_search"
        assert out["tools"][0]["type"] == "intrinsic"

    def test_no_promote_when_disallowed(self):
        ir = {"tools": [{"type": "function", "name": "web_search", "parameters": {}}]}
        out = resolve_intrinsic_tools(ir, shim=_shim("messages", ["web_search"]))
        assert out["tools"][0]["type"] == "function"

    def test_promote_only_supported_names(self):
        ir = {"tools": [{"type": "function", "name": "get_weather", "parameters": {}}]}
        out = resolve_intrinsic_tools(
            ir, shim=_shim("messages", ["web_search"]), allow_name_promotion=True
        )
        assert out["tools"][0]["type"] == "function"


# ── End-to-end pipeline ─────────────────────────────────────────────────


class TestPipeline:
    def test_anthropic_web_search_to_google_is_dropped(self):
        """Anthropic web_search → google: google has no web_search kind."""
        pipe = ConversionPipeline(
            "messages",
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
            "messages",
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
            "openai_chat", "messages", target_shim=_shim("messages", ["web_search"])
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
        pipe = ConversionPipeline("openai_chat", "messages")
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

    def test_shim_name_behaves_like_a_shim_object(self):
        """A shim passed by *name* resolves, not only a shim object.

        ``gateway/proxy.py`` builds the pipeline with ``target_shim=route.shim_name``
        — a name.  Reading shim attributes off that string found nothing, so every
        intrinsic definition was dropped on gateway requests; the library path,
        which passes an object, was unaffected.
        """
        body = {
            "model": "m",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "code_execution_20250522", "name": "code_execution"}],
        }
        by_name = ConversionPipeline(
            "messages", "google_generate", target_shim="google--google_generate"
        ).convert_request(dict(body))
        by_object = ConversionPipeline(
            "messages",
            "google_generate",
            target_shim=resolve_shim("google--google_generate"),
        ).convert_request(dict(body))
        named_tools = (
            by_name.get("tools") or by_name.get("config", {}).get("tools") or []
        )
        assert {"code_execution": {}} in named_tools
        assert by_name == by_object

    def test_shim_name_still_drops_an_undeclared_kind(self):
        """Resolving the name must not become a blanket keep."""
        pipe = ConversionPipeline(
            "messages", "google_generate", target_shim="google--google_generate"
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

    def test_promotion_follows_the_source_converter_flag(self, monkeypatch):
        """Promotion is read off the source converter, not the format name.

        Only Chat Completions lacks native server tools today, so the two
        derivations coincide on every shipped format; flipping the flag on the
        converter shows which one the pipeline actually consults.
        """
        monkeypatch.setattr(
            ChatCompletionsConverter, "_SUPPORTS_NATIVE_SERVER_TOOLS", True
        )
        pipe = ConversionPipeline(
            "openai_chat", "messages", target_shim=_shim("messages", ["web_search"])
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
        assert not any(
            t.get("type") == "web_search_20250305" for t in body.get("tools") or []
        )

    def test_shim_spelled_chat_source_promotes_too(self):
        """A chat-completions *shim* source promotes, not just the bare name.

        The old rule compared the literal canonical name, so
        ``deepseek--chat_completions`` did not promote while ``chat_completions``
        did — the same format, the same converter, and just as unable to name a
        server tool.  Only reachable when the target shim declares the kind.
        """
        pipe = ConversionPipeline(
            "deepseek--chat_completions",
            "messages",
            target_shim=_shim("messages", ["web_search"]),
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


class TestShimEmitConsistency:
    """Every kind a shim declares must be emittable by the target converter.

    Otherwise ``resolve_intrinsic_tools`` accepts a kind as supported and the
    target converter then silently drops it at emit time (review feedback on
    PR #878: `file_search` was declared for openai_responses but had no emit
    entry).
    """

    def _shim_kinds(self, name):
        from llm_rosetta.shims.providers import load_providers

        shims = {s.name: s for s in load_providers()}
        return set(shims[name].tools.intrinsic_tools) if name in shims else set()

    def test_anthropic(self):
        from llm_rosetta.converters.messages.tool_ops import _ANTHROPIC_INTRINSIC_TOOLS

        assert self._shim_kinds("anthropic--messages") <= set(
            _ANTHROPIC_INTRINSIC_TOOLS
        )

    def test_openai_responses(self):
        from llm_rosetta.converters.open_responses.tool_ops import (
            _RESPONSES_INTRINSIC_TOOLS,
        )

        assert self._shim_kinds("openai_responses") <= set(_RESPONSES_INTRINSIC_TOOLS)

    def test_google(self):
        from llm_rosetta.converters.google_generate.tool_ops import _INTRINSIC_TOOL_KEYS

        assert self._shim_kinds("google") <= set(_INTRINSIC_TOOL_KEYS.values())

    def test_google_interactions(self):
        from llm_rosetta.converters.google_interactions.tool_ops import (
            _GI_INTRINSIC_TOOLS,
        )

        assert self._shim_kinds("google_interactions") <= set(_GI_INTRINSIC_TOOLS)


class TestUnsupportedKindNotCachedAsEmpty:
    """A dropped (`{}`) tool must not resurface on a cache hit (PR #878 review).

    The tool-conversion cache stores the falsy result too, so the truthiness
    check has to run on the cache-hit branch as well.
    """

    def _calls(self, converter, tool, n=3):
        return [converter._get_cached_ir_tools_to_p([tool]) for _ in range(n)]

    def test_unsupported_intrinsic_stays_dropped(self):
        out = self._calls(
            MessagesConverter(), make_intrinsic_tool_definition("google_maps")
        )
        assert all(o == [] for o in out), out

    def test_chat_drops_intrinsic_every_time(self):
        out = self._calls(
            ChatCompletionsConverter(), make_intrinsic_tool_definition("web_search")
        )
        assert all(o == [] for o in out), out

    def test_supported_still_emits_every_time(self):
        out = self._calls(
            MessagesConverter(), make_intrinsic_tool_definition("web_search")
        )
        assert all(
            o == [{"type": "web_search_20250305", "name": "web_search"}] for o in out
        )


class TestSameFormatNativeRoundTrip:
    """A kind that needs config (file_search) round-trips same-provider via the
    preserved native payload, even though it has no synthetic emit entry."""

    def test_responses_file_search_roundtrip(self):
        conv = OpenAIResponsesConverter()
        tool = {"type": "file_search", "vector_store_ids": ["vs_123"]}
        ir = conv.tool_ops.p_tool_definition_to_ir(tool)
        assert ir["type"] == "intrinsic"
        out = conv.tool_ops.ir_tool_definition_to_p(ir)
        assert out == tool
