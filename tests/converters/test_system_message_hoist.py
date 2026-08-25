"""Tests for late system message hoisting.

Verifies that system messages appearing mid-conversation are either moved
to system_instruction (leading) or rewritten as user messages with a
``<system>...</system>`` envelope (late), preserving prompt cache prefix
stability.

Attribution: envelope format adapted from codex-rosetta commits c749003b
and e7e7768e.
"""

from __future__ import annotations
from typing import Any, cast

from llm_rosetta.converters.base.helpers.system_message_hoist import (
    hoist_late_system_messages_ir,
)


def _sys(text: str, **kw) -> dict:
    return {"role": "system", "content": [{"type": "text", "text": text}], **kw}


def _user(text: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _asst(text: str) -> dict:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def _wrapped(text: str) -> str:
    return f"<system>\n{text}\n</system>"


class TestNoOp:
    def test_no_messages(self):
        req = {"model": "m", "messages": []}
        assert hoist_late_system_messages_ir(req) is req

    def test_no_system_messages(self):
        req = {"model": "m", "messages": [_user("hi"), _asst("hello")]}
        assert hoist_late_system_messages_ir(req) is req

    def test_empty_messages_key(self):
        req = {"model": "m"}
        assert hoist_late_system_messages_ir(req) is req


class TestLeadingSystem:
    def test_single_leading_moved_to_empty_si(self):
        req = {"model": "m", "messages": [_sys("Be helpful"), _user("hi")]}
        result = hoist_late_system_messages_ir(req)
        assert result["system_instruction"] == [{"type": "text", "text": "Be helpful"}]
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"

    def test_leading_appended_to_existing_si(self):
        req = {
            "model": "m",
            "system_instruction": [{"type": "text", "text": "Existing"}],
            "messages": [_sys("Extra"), _user("hi")],
        }
        result = hoist_late_system_messages_ir(req)
        assert len(result["system_instruction"]) == 2
        assert result["system_instruction"][0]["text"] == "Existing"
        assert result["system_instruction"][1]["text"] == "Extra"

    def test_multiple_leading(self):
        req = {"model": "m", "messages": [_sys("A"), _sys("B"), _user("hi")]}
        result = hoist_late_system_messages_ir(req)
        assert len(result["system_instruction"]) == 2
        assert result["system_instruction"][0]["text"] == "A"
        assert result["system_instruction"][1]["text"] == "B"
        assert len(result["messages"]) == 1

    def test_all_system(self):
        req = {"model": "m", "messages": [_sys("A"), _sys("B")]}
        result = hoist_late_system_messages_ir(req)
        assert len(result["system_instruction"]) == 2
        assert result["messages"] == []


class TestLateSystem:
    def test_single_late_rewritten_as_user(self):
        req = {
            "model": "m",
            "messages": [_user("hi"), _asst("hey"), _sys("Be formal"), _user("ok")],
        }
        result = hoist_late_system_messages_ir(req)
        # Late system rewritten as user; no merging with adjacent user
        assert len(result["messages"]) == 4
        rewritten = result["messages"][2]
        assert rewritten["role"] == "user"
        assert rewritten["content"][0]["text"] == _wrapped("Be formal")
        # Following user message unchanged
        assert result["messages"][3] == _user("ok")

    def test_multiple_late(self):
        req = {
            "model": "m",
            "messages": [_user("hi"), _sys("A"), _asst("ok"), _sys("B"), _user("bye")],
        }
        result = hoist_late_system_messages_ir(req)
        assert result["messages"][1]["content"][0]["text"] == _wrapped("A")
        assert result["messages"][3]["content"][0]["text"] == _wrapped("B")

    def test_empty_system_uses_envelope(self):
        req = {"model": "m", "messages": [_user("hi"), _sys("")]}
        result = hoist_late_system_messages_ir(req)
        assert result["messages"][1]["content"][0]["text"] == _wrapped("")

    def test_metadata_preserved(self):
        req = {
            "model": "m",
            "messages": [_user("hi"), _sys("formal", metadata={"source": "dev"})],
        }
        result = hoist_late_system_messages_ir(req)
        assert result["messages"][1]["metadata"] == {"source": "dev"}


class TestMixed:
    def test_leading_and_late(self):
        req = {
            "model": "m",
            "messages": [
                _sys("System prompt"),
                _user("hi"),
                _asst("hello"),
                _sys("Now be formal"),
                _user("ok"),
            ],
        }
        result = hoist_late_system_messages_ir(req)
        assert result["system_instruction"] == [
            {"type": "text", "text": "System prompt"}
        ]
        # Late system rewritten as user, following user unchanged
        assert len(result["messages"]) == 4
        assert result["messages"][0]["role"] == "user"  # "hi"
        assert result["messages"][2]["role"] == "user"  # rewritten
        assert result["messages"][2]["content"][0]["text"] == _wrapped("Now be formal")
        assert result["messages"][3] == _user("ok")

    def test_multi_part_system(self):
        msg = {
            "role": "system",
            "content": [
                {"type": "text", "text": "Part one"},
                {"type": "text", "text": "Part two"},
            ],
        }
        req = {"model": "m", "messages": [_user("hi"), msg]}
        result = hoist_late_system_messages_ir(req)
        # First and last text parts get envelope tags
        content = result["messages"][1]["content"]
        assert content[0]["text"] == "<system>\nPart one"
        assert content[1]["text"] == "Part two\n</system>"


class TestMultimodalPreservation:
    """Late system messages with non-text content preserve all parts."""

    def test_image_part_preserved(self):
        image = {"type": "image", "source": {"type": "base64", "data": "AA=="}}
        msg = {
            "role": "system",
            "content": [
                {"type": "text", "text": "Look at this"},
                image,
                {"type": "text", "text": "and respond"},
            ],
        }
        req = {"model": "m", "messages": [_user("hi"), msg]}
        result = hoist_late_system_messages_ir(req)
        rewritten = result["messages"][1]
        assert rewritten["role"] == "user"
        assert len(rewritten["content"]) == 3
        assert rewritten["content"][0]["text"] == "<system>\nLook at this"
        assert rewritten["content"][1] == image
        assert rewritten["content"][2]["text"] == "and respond\n</system>"

    def test_nontext_only_gets_boundary_sentinels(self):
        image = {"type": "image", "source": {"type": "base64", "data": "AA=="}}
        msg = {"role": "system", "content": [image]}
        req = {"model": "m", "messages": [_user("hi"), msg]}
        result = hoist_late_system_messages_ir(req)
        content = result["messages"][1]["content"]
        assert len(content) == 3
        assert content[0] == {"type": "text", "text": "<system>"}
        assert content[1] == image
        assert content[2] == {"type": "text", "text": "</system>"}

    def test_string_content_wrapped(self):
        msg = {"role": "system", "content": "plain text system"}
        req = {"model": "m", "messages": [_user("hi"), msg]}
        result = hoist_late_system_messages_ir(req)
        assert result["messages"][1]["content"][0]["text"] == _wrapped(
            "plain text system"
        )


class TestIdempotency:
    def test_double_apply(self):
        req = {"model": "m", "messages": [_user("hi"), _sys("Be formal"), _user("ok")]}
        r1 = hoist_late_system_messages_ir(req)
        r2 = hoist_late_system_messages_ir(r1)
        assert r1["messages"] == r2["messages"]
        assert "system_instruction" not in r1
        assert "system_instruction" not in r2

    def test_original_not_mutated(self):
        msgs = [_user("hi"), _sys("Be formal")]
        req = {"model": "m", "messages": msgs}
        result = hoist_late_system_messages_ir(req)
        assert req["messages"] is msgs
        assert len(msgs) == 2
        assert msgs[1]["role"] == "system"
        assert result is not req


class TestTransformIntegration:
    def test_factory_repr(self):
        from llm_rosetta.shims.transforms import hoist_late_system_messages

        t = hoist_late_system_messages()
        assert repr(t) == "hoist_late_system_messages()"

    def test_argo_anthropic_includes_hoist(self):
        from llm_rosetta.shims.provider_shim import get_shim

        shim = get_shim("argo--anthropic")
        assert shim is not None
        names = [repr(t) for t in shim.ir_transforms]
        assert "hoist_late_system_messages()" in names

    def test_openrouter_anthropic_includes_hoist(self):
        from llm_rosetta.shims.provider_shim import get_shim

        shim = get_shim("openrouter--anthropic")
        assert shim is not None
        names = [repr(t) for t in shim.ir_transforms]
        assert "hoist_late_system_messages()" in names

    def test_hoist_before_cache_breakpoints(self):
        from llm_rosetta.shims.provider_shim import get_shim

        shim = get_shim("argo--anthropic")
        assert shim is not None
        names = [repr(t) for t in shim.ir_transforms]
        hoist_idx = names.index("hoist_late_system_messages()")
        cache_idx = names.index("auto_cache_breakpoints()")
        assert hoist_idx < cache_idx

    def test_pipeline_with_ir_system_messages(self):
        """Verify hoist works in the pipeline when IR has mid-list system messages.

        Constructs IR directly (bypassing source converter) to simulate
        a source format that preserves system message positions.
        """
        from llm_rosetta.pipeline import apply_ir_transforms

        ir_request = {
            "model": "claude-haiku-4-5",
            "system_instruction": [{"type": "text", "text": "You are helpful."}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "Now be formal."}],
                },
                {"role": "user", "content": [{"type": "text", "text": "ok"}]},
            ],
        }
        result = apply_ir_transforms(ir_request, "argo--anthropic")
        # No system messages in the messages array
        for msg in result["messages"]:
            assert msg["role"] != "system", f"system message leaked: {msg}"
        # Late system rewritten as user with <system> envelope
        user_msgs = [m for m in result["messages"] if m["role"] == "user"]
        enveloped = [
            m
            for m in user_msgs
            if any(
                "<system>" in c.get("text", "")
                for c in m.get("content", [])
                if isinstance(c, dict)
            )
        ]
        assert len(enveloped) == 1
        # system_instruction preserved
        assert result["system_instruction"][0]["text"] == "You are helpful."

    def test_hoist_produces_correct_ir_for_anthropic(self):
        """Hoisted late system → user with envelope; no merging at IR level.

        The hoist deliberately does NOT merge consecutive user messages.
        That is the target converter's responsibility (e.g. Anthropic API
        requires alternating roles). This test verifies the IR output is
        correct and the envelope is present.
        """
        from llm_rosetta.pipeline import apply_ir_transforms

        ir_request = {
            "model": "claude-haiku-4-5",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
                {"role": "system", "content": [{"type": "text", "text": "Be formal"}]},
                {"role": "user", "content": [{"type": "text", "text": "ok"}]},
            ],
        }
        hoisted = apply_ir_transforms(ir_request, "argo--anthropic")

        # No system messages remain
        for msg in hoisted["messages"]:
            assert msg["role"] != "system", f"system message leaked: {msg}"

        # The rewritten message has the envelope
        rewritten = hoisted["messages"][2]
        assert rewritten["role"] == "user"
        assert "<system>" in rewritten["content"][0]["text"]
        assert "Be formal" in rewritten["content"][0]["text"]
        assert "</system>" in rewritten["content"][0]["text"]

        # Following user message is separate (not merged)
        assert hoisted["messages"][3]["role"] == "user"
        assert hoisted["messages"][3]["content"][0]["text"] == "ok"


class TestDeveloperRole:
    """Chat API developer role should be converted to IR system and hoisted."""

    def test_chat_developer_to_ir_system(self):
        """Leading Chat developer messages are extracted to system_instruction."""
        from llm_rosetta.converters.openai_chat import OpenAIChatConverter

        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "developer", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        }
        converter = OpenAIChatConverter()
        ir_request = cast(dict[str, Any], converter.request_from_provider(body))
        assert ir_request["system_instruction"][0]["text"] == "You are helpful."
        assert len(ir_request["messages"]) == 1
        assert ir_request["messages"][0]["role"] == "user"

    def test_late_chat_developer_hoisted(self):
        """Late Chat developer message is hoisted as user with envelope."""
        from llm_rosetta.pipeline import ConversionPipeline

        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "developer", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "developer", "content": "Be concise."},
                {"role": "user", "content": "What is 2+2?"},
            ],
        }
        pipeline = ConversionPipeline(
            "openai_chat", "openai_chat", target_shim="openai"
        )
        result = pipeline.convert_request(body)
        roles = [m["role"] for m in result["messages"]]
        # No system messages mid-conversation
        system_indices = [i for i, r in enumerate(roles) if r == "system"]
        assert all(i == 0 for i in system_indices), (
            f"system message at non-leading position: {system_indices}"
        )
        # Late developer appears as user with envelope
        user_msgs = [m for m in result["messages"] if m["role"] == "user"]
        enveloped = [
            m
            for m in user_msgs
            if isinstance(m.get("content"), str) and "<system>" in m["content"]
        ]
        assert len(enveloped) == 1
        assert "Be concise." in enveloped[0]["content"]

    def test_responses_late_developer_to_chat(self):
        """Responses developer messages hoisted when converting to Chat."""
        from llm_rosetta.pipeline import ConversionPipeline

        body = {
            "model": "gpt-4o",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [
                        {"type": "input_text", "text": "You are a helpful assistant."}
                    ],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hi there!"}],
                },
                {
                    "type": "message",
                    "role": "developer",
                    "content": [
                        {"type": "input_text", "text": "New instructions: be concise."}
                    ],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "What is 2+2?"}],
                },
            ],
            "stream": True,
        }
        pipeline = ConversionPipeline(
            "openai_responses", "openai_chat", target_shim="openai"
        )
        result = pipeline.convert_request(body)
        roles = [m["role"] for m in result["messages"]]
        # Late system hoisted to user
        non_leading_system = [i for i, r in enumerate(roles) if r == "system" and i > 0]
        assert not non_leading_system, f"late system messages: {non_leading_system}"
