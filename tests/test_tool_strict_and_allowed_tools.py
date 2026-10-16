"""Cross-format coverage for tool ``strict`` and the ``allowed_tools`` restriction.

Both are shared concepts that used to be lost on the way through the IR:

- ``strict`` is stated by OpenAI Chat, OpenAI Responses and Anthropic Messages;
  only the Responses converter kept it, and only in ``metadata["strict"]``.
- An allow-list of tools is stated by OpenAI Chat and Responses, Google
  Interactions and Google generateContent, in four different shapes; the IR
  used to carry it as an opaque Responses-shaped extension, so every other
  target dropped it (Google could not even keep a second name).

These tests pin the IR fields that replaced both —
``ToolDefinition.strict`` and ``ToolChoice.allowed_tools`` — and the warning a
target without an allow-list is expected to raise.
"""

from __future__ import annotations

from typing import cast

import pytest

from llm_rosetta import convert
from llm_rosetta.capabilities.tool_names import (
    apply_upstream_tool_names,
    build_tool_name_map,
    qualify_tool_names,
)
from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.converters.chat_completions import ChatCompletionsConverter
from llm_rosetta.converters.google_interactions import GoogleInteractionsConverter
from llm_rosetta.converters.messages import MessagesConverter
from llm_rosetta.converters.open_responses import OpenAIResponsesConverter
from llm_rosetta.pipeline import ConversionPipeline
from llm_rosetta.types.ir import IRRequest

_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}


def _chat_tool(strict: bool | None = None) -> dict:
    func: dict = {"name": "get_weather", "description": "d", "parameters": _SCHEMA}
    if strict is not None:
        func["strict"] = strict
    return {"type": "function", "function": func}


def _chat_body(tools: list[dict], **extra) -> dict:
    return {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": tools,
        **extra,
    }


def _responses_tool(strict: bool | None = None) -> dict:
    tool: dict = {
        "type": "function",
        "name": "get_weather",
        "description": "d",
        "parameters": _SCHEMA,
    }
    if strict is not None:
        tool["strict"] = strict
    return tool


def _responses_body(tools: list[dict], **extra) -> dict:
    return {
        "model": "m",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hi"}],
            }
        ],
        "tools": tools,
        **extra,
    }


class TestStrictRoundTrip:
    """``strict`` survives every format that has the concept."""

    def test_chat_round_trip_keeps_false(self):
        out = convert(
            _chat_body([_chat_tool(strict=False)]),
            "chat_completions",
            source_provider="chat_completions",
            baseline=False,
        )
        assert out["tools"][0]["function"]["strict"] is False

    def test_chat_round_trip_keeps_true(self):
        out = convert(
            _chat_body([_chat_tool(strict=True)]),
            "chat_completions",
            source_provider="chat_completions",
            baseline=False,
        )
        assert out["tools"][0]["function"]["strict"] is True

    def test_chat_to_responses_is_not_downgraded(self):
        """The reported bug: ``strict: true`` used to reach Responses as false."""
        out = convert(
            _chat_body([_chat_tool(strict=True)]),
            "open_responses",
            source_provider="chat_completions",
            baseline=False,
        )
        assert out["tools"][0]["strict"] is True

    def test_chat_without_strict_stays_absent(self):
        out = convert(
            _chat_body([_chat_tool()]),
            "open_responses",
            source_provider="chat_completions",
            baseline=False,
        )
        # Responses defaults the flag; Chat did not state one.
        assert out["tools"][0]["strict"] is False
        back = convert(
            _chat_body([_chat_tool()]),
            "chat_completions",
            source_provider="chat_completions",
            baseline=False,
        )
        assert "strict" not in back["tools"][0]["function"]

    def test_responses_to_chat_keeps_true(self):
        out = convert(
            _responses_body([_responses_tool(strict=True)]),
            "chat_completions",
            source_provider="open_responses",
            baseline=False,
        )
        assert out["tools"][0]["function"]["strict"] is True

    def test_messages_round_trip(self):
        body = {
            "model": "claude-haiku-4-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "d",
                    "input_schema": _SCHEMA,
                    "strict": True,
                }
            ],
        }
        out = convert(
            body,
            "messages",
            source_provider="messages",
            baseline=False,
        )
        assert out["tools"][0]["strict"] is True

    def test_messages_to_chat_keeps_true(self):
        body = {
            "model": "claude-haiku-4-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "d",
                    "input_schema": _SCHEMA,
                    "strict": True,
                }
            ],
        }
        out = convert(
            body,
            "chat_completions",
            source_provider="messages",
            baseline=False,
        )
        assert out["tools"][0]["function"]["strict"] is True

    def test_google_generate_drops_it_without_error(self):
        """Google has no strict mode, so the flag simply does not travel."""
        out = convert(
            _chat_body([_chat_tool(strict=True)]),
            "google_generate",
            source_provider="chat_completions",
            baseline=False,
        )
        declarations = out["config"]["tools"][0]["function_declarations"]
        assert "strict" not in declarations[0]

    def test_metadata_strict_is_still_read(self):
        """IR built before the field existed names the flag in metadata."""
        conv = ChatCompletionsConverter()
        ir = cast(
            IRRequest,
            {
                "model": "m",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "hi"}]}
                ],
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "d",
                        "parameters": _SCHEMA,
                        "metadata": {"strict": True},
                    }
                ],
            },
        )
        out, _ = conv.request_to_provider(ir)
        assert out["tools"][0]["function"]["strict"] is True


class TestAllowedToolsAcrossFormats:
    """The restriction travels between the formats that can state it."""

    _CHAT_CHOICE = {
        "type": "allowed_tools",
        "mode": "auto",
        "tools": [
            {"type": "function", "function": {"name": "get_weather"}},
            {"type": "function", "function": {"name": "get_time"}},
        ],
    }

    def _two_tools_chat(self) -> dict:
        return _chat_body(
            [
                _chat_tool(),
                {
                    "type": "function",
                    "function": {"name": "get_time", "description": "d"},
                },
            ],
            tool_choice=self._CHAT_CHOICE,
        )

    def test_chat_reads_the_restriction_it_cannot_send(self):
        """A Chat-shaped restriction is read so another target can honour it."""
        conv = ChatCompletionsConverter()
        ir = conv.request_from_provider(self._two_tools_chat())
        assert ir["tool_choice"]["mode"] == "auto"
        assert ir["tool_choice"]["allowed_tools"] == [
            {"name": "get_weather"},
            {"name": "get_time"},
        ]

    def test_chat_to_responses_keeps_the_restriction(self):
        out = convert(
            self._two_tools_chat(),
            "open_responses",
            source_provider="chat_completions",
            baseline=False,
        )
        assert out["tool_choice"] == {
            "type": "allowed_tools",
            "mode": "auto",
            "tools": [
                {"type": "function", "name": "get_weather"},
                {"type": "function", "name": "get_time"},
            ],
        }

    def test_entry_type_follows_the_declaration(self):
        """A custom tool needs a custom-shaped entry, or the API 400s.

        Measured against ``api.openai.com``: a ``function``-shaped entry naming
        a custom tool is rejected with "Tool choice 'function' not found in
        'tools' parameter", and the reverse likewise.  The entry type is read
        off the declaration rather than stored in the IR.
        """
        body = _responses_body(
            [
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "d",
                    "format": {"type": "text"},
                }
            ],
            tool_choice={
                "type": "allowed_tools",
                "mode": "auto",
                "tools": [{"type": "custom", "name": "apply_patch"}],
            },
        )
        conv = OpenAIResponsesConverter()
        ir = conv.request_from_provider(body)
        out, _ = conv.request_to_provider(ir)
        assert out["tool_choice"]["tools"] == [
            {"type": "custom", "name": "apply_patch"}
        ]

    def test_chat_target_warns_and_widens(self):
        conv = ChatCompletionsConverter()
        ir = OpenAIResponsesConverter().request_from_provider(
            _responses_body(
                [_responses_tool(), _responses_tool()],
                tool_choice={
                    "type": "allowed_tools",
                    "mode": "auto",
                    "tools": [{"type": "function", "name": "get_weather"}],
                },
            )
        )
        ctx = ConversionContext()
        out, _ = conv.request_to_provider(ir, context=ctx)
        assert out["tool_choice"] == "auto"
        assert any("allowed_tools" in w for w in ctx.warnings), ctx.warnings

    def test_messages_target_warns_and_widens(self):
        conv = MessagesConverter()
        ir = OpenAIResponsesConverter().request_from_provider(
            _responses_body(
                [_responses_tool()],
                tool_choice={
                    "type": "allowed_tools",
                    "mode": "auto",
                    "tools": [{"type": "function", "name": "get_weather"}],
                },
            )
        )
        ctx = ConversionContext()
        out, _ = conv.request_to_provider(ir, context=ctx)
        assert out["tool_choice"] == {"type": "auto"}
        assert any("allowed_tools" in w for w in ctx.warnings), ctx.warnings

    def test_names_are_respelled_for_the_upstream(self):
        """A qualified upstream name reaches the wire, not the client's.

        ``mode: "required"`` — a mandatory choice — is the shape
        generateContent can state a name list under; see
        ``TestAllowedToolsToGoogle.test_google_generate_keeps_the_mode_and_warns``
        for the other one.
        """
        pipe = ConversionPipeline("open_responses", "google_generate", baseline=False)
        body = _responses_body(
            [_responses_tool()],
            tool_choice={
                "type": "allowed_tools",
                "mode": "required",
                "tools": [{"type": "function", "name": "get_weather"}],
            },
        )
        out = pipe.convert_request(body)
        assert out["tool_config"]["function_calling_config"][
            "allowed_function_names"
        ] == ["get_weather"]

    def test_undeclared_name_warns(self):
        """A restriction naming no declared tool cannot be honoured."""
        pipe = ConversionPipeline("open_responses", "google_generate", baseline=False)
        body = _responses_body(
            [_responses_tool()],
            tool_choice={
                "type": "allowed_tools",
                "mode": "auto",
                "tools": [{"type": "function", "name": "nope"}],
            },
        )
        pipe.convert_request(body)
        assert any("nope" in w for w in pipe.warnings), pipe.warnings

    def test_grouped_identity_resolves_through_its_group(self):
        """A restriction naming a grouped tool resolves where a bare name cannot.

        Two tools share a bare name; the target needs flat names, so they are
        qualified apart.  Only the identity that states its group can still be
        resolved — the bare one is left as it was, with a warning.
        """
        request = {
            "model": "m",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "tools": [
                {
                    "type": "function",
                    "name": "search",
                    "description": "d",
                    "parameters": {},
                    "tool_group": "alpha",
                },
                {
                    "type": "function",
                    "name": "search",
                    "description": "d",
                    "parameters": {},
                    "tool_group": "beta",
                },
            ],
            "tool_choice": {
                "mode": "any",
                "allowed_tools": [
                    {"name": "search", "tool_group": "beta"},
                    {"name": "search"},
                ],
            },
        }
        request = qualify_tool_names(request, warnings=[])
        assert [t["name"] for t in request["tools"]] == ["alpha_search", "beta_search"]

        warnings: list[str] = []
        apply_upstream_tool_names(
            request, name_map=build_tool_name_map(request), warnings=warnings
        )
        assert request["tool_choice"]["allowed_tools"] == [
            {"name": "beta_search"},
            {"name": "search"},
        ]
        assert len(warnings) == 1 and "shared by tools in more than one" in warnings[0]


class TestReaderReportsWhatItCannotCarry:
    """The readers report a restriction that comes out wider than it went in."""

    def _responses_choice(self, tool_choice: dict) -> dict:
        return _responses_body([_responses_tool()], tool_choice=tool_choice)

    def test_entry_naming_no_tool_is_dropped_with_a_warning(self):
        """An entry the IR cannot hold — a server tool, say — is reported."""
        ctx = ConversionContext()
        ir = OpenAIResponsesConverter().request_from_provider(
            self._responses_choice(
                {
                    "type": "allowed_tools",
                    "mode": "required",
                    "tools": [
                        {"type": "mcp", "server_label": "deepwiki"},
                        {"type": "function", "name": "get_weather"},
                    ],
                }
            ),
            context=ctx,
        )
        assert ir["tool_choice"]["allowed_tools"] == [{"name": "get_weather"}]
        assert any("restriction is wider" in w for w in ctx.warnings), ctx.warnings

    def test_unrecognized_mode_is_reported(self):
        """An unknown mode reads as the widest one, and says so."""
        ctx = ConversionContext()
        ir = OpenAIResponsesConverter().request_from_provider(
            self._responses_choice(
                {
                    "type": "allowed_tools",
                    "mode": "whenever",
                    "tools": [{"type": "function", "name": "get_weather"}],
                }
            ),
            context=ctx,
        )
        assert ir["tool_choice"]["mode"] == "auto"
        assert any("unrecognized mode" in w for w in ctx.warnings), ctx.warnings

    def test_interactions_reports_a_dropped_entry(self):
        conv = GoogleInteractionsConverter()
        ctx = ConversionContext()
        ir = conv.request_from_provider(
            {
                "model": "gemini-3.7-flash",
                "input": "hi",
                "generation_config": {
                    "tool_choice": {
                        "allowed_tools": {
                            "mode": "any",
                            "tools": [{"server_label": "deepwiki"}, "get_weather"],
                        }
                    }
                },
            },
            context=ctx,
        )
        assert ir["tool_choice"]["allowed_tools"] == [{"name": "get_weather"}]
        assert any("restriction is wider" in w for w in ctx.warnings), ctx.warnings

    def test_a_restriction_naming_an_undeclared_tool_is_reported_by_the_converter(self):
        """The API answers 400 to this body, so a library caller is told."""
        conv = OpenAIResponsesConverter()
        ir = cast(
            IRRequest,
            {
                "model": "m",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "hi"}]}
                ],
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "d",
                        "parameters": _SCHEMA,
                    }
                ],
                "tool_choice": {
                    "mode": "any",
                    "allowed_tools": [{"name": "nope"}],
                },
            },
        )
        ctx = ConversionContext()
        out, _ = conv.request_to_provider(ir, context=ctx)
        assert any("nope" in w for w in ctx.warnings), ctx.warnings
        assert out["tool_choice"]["tools"] == [{"type": "function", "name": "nope"}]


class TestAllowedToolsIRValidation:
    """The new IR shape survives the runtime validator."""

    @pytest.mark.parametrize("strict", [True, False, None])
    def test_strict_is_a_boolean_field(self, strict):
        from llm_rosetta.types.ir.validation import validate_ir_request

        tool: dict = {
            "type": "function",
            "name": "f",
            "description": "d",
            "parameters": {},
        }
        if strict is not None:
            tool["strict"] = strict
        request = {
            "model": "m",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "tools": [tool],
            "tool_choice": {
                "mode": "any",
                "allowed_tools": [{"name": "f"}, {"name": "g", "tool_group": "ns"}],
            },
        }
        # No raise: the validator accepts both new fields.
        assert validate_ir_request(request)["tool_choice"]["allowed_tools"] == [
            {"name": "f"},
            {"name": "g", "tool_group": "ns"},
        ]
