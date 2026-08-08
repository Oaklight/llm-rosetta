"""Tests for late system message hoisting.

Verifies that system messages appearing mid-conversation are either moved
to system_instruction (leading) or rewritten as user messages with a
[System: ...] envelope (late), preserving prompt cache prefix stability.
"""

from __future__ import annotations

from llm_rosetta.converters.base.helpers.system_message_hoist import (
    hoist_late_system_messages_ir,
)


def _sys(text: str, **kw) -> dict:
    return {"role": "system", "content": [{"type": "text", "text": text}], **kw}


def _user(text: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _asst(text: str) -> dict:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


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
        assert len(result["messages"]) == 3
        merged = result["messages"][2]
        assert merged["role"] == "user"
        assert merged["content"][0]["text"] == "[System: Be formal]"
        assert merged["content"][1]["text"] == "ok"

    def test_multiple_late(self):
        req = {
            "model": "m",
            "messages": [_user("hi"), _sys("A"), _asst("ok"), _sys("B"), _user("bye")],
        }
        result = hoist_late_system_messages_ir(req)
        assert result["messages"][1]["content"][0]["text"] == "[System: A]"
        assert result["messages"][3]["content"][0]["text"] == "[System: B]"

    def test_empty_system_uses_placeholder(self):
        req = {"model": "m", "messages": [_user("hi"), _sys("")]}
        result = hoist_late_system_messages_ir(req)
        assert result["messages"][1]["content"][0]["text"] == "[System instruction]"

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
        assert len(result["messages"]) == 3
        assert result["messages"][0]["role"] == "user"  # "hi"
        assert result["messages"][2]["role"] == "user"  # merged
        assert result["messages"][2]["content"][0]["text"] == "[System: Now be formal]"
        assert result["messages"][2]["content"][1]["text"] == "ok"

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
        assert (
            result["messages"][1]["content"][0]["text"]
            == "[System: Part one\nPart two]"
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
        # Late system rewritten as user envelope
        user_msgs = [m for m in result["messages"] if m["role"] == "user"]
        enveloped = [
            m
            for m in user_msgs
            if any(
                "[System:" in c.get("text", "")
                for c in m.get("content", [])
                if isinstance(c, dict)
            )
        ]
        assert len(enveloped) == 1
        # system_instruction preserved
        assert result["system_instruction"][0]["text"] == "You are helpful."

    def test_no_consecutive_user_after_anthropic_conversion(self):
        """Hoisted late system → user must not create consecutive user roles.

        The Anthropic converter merges consecutive same-role messages, so
        the final output should always alternate roles.
        """
        from llm_rosetta.converters.anthropic import AnthropicConverter
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
        converter = AnthropicConverter()
        from typing import cast
        from llm_rosetta.types.ir.request import IRRequest

        result, _ = converter.request_to_provider(cast(IRRequest, hoisted))

        roles = [m["role"] for m in result["messages"]]
        for i in range(1, len(roles)):
            assert roles[i] != roles[i - 1], (
                f"consecutive {roles[i]} at index {i}: {roles}"
            )
