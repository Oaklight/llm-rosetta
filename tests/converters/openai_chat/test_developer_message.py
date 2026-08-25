"""Tests for developer message handling in OpenAI Chat converter.

Verifies that developer role messages are properly handled:
- Leading developer messages → system_instruction
- Late developer messages → preserved as IR system messages for hoist
- Full round-trip through hoist produces user-role <system> envelope
"""

from __future__ import annotations

from typing import Any

import pytest

from llm_rosetta.converters.openai_chat import OpenAIChatConverter
from llm_rosetta.converters.base.helpers.system_message_hoist import (
    hoist_late_system_messages_ir,
)


@pytest.fixture
def converter() -> OpenAIChatConverter:
    return OpenAIChatConverter()


def _chat_request(messages: list[dict[str, Any]]) -> dict[str, Any]:
    return {"model": "gpt-4", "messages": messages}


class TestLeadingDeveloperExtraction:
    """Leading developer messages should be extracted to system_instruction."""

    def test_leading_developer_to_system_instruction(self, converter: OpenAIChatConverter):
        req = _chat_request([
            {"role": "developer", "content": "You are helpful"},
            {"role": "user", "content": "hi"},
        ])
        ir = converter.request_from_provider(req)
        assert ir["system_instruction"] is not None
        texts = [p["text"] for p in ir["system_instruction"]]
        assert "You are helpful" in texts
        # developer should not appear in messages
        for msg in ir["messages"]:
            assert msg["role"] != "developer"

    def test_multiple_leading_developer_all_extracted(self, converter: OpenAIChatConverter):
        req = _chat_request([
            {"role": "developer", "content": "Instruction A"},
            {"role": "developer", "content": "Instruction B"},
            {"role": "user", "content": "hi"},
        ])
        ir = converter.request_from_provider(req)
        texts = [p["text"] for p in ir["system_instruction"]]
        assert "Instruction A" in texts
        assert "Instruction B" in texts


class TestLateDeveloperPreservation:
    """Late developer messages should stay in messages as system role for hoist."""

    def test_late_developer_stays_in_messages(self, converter: OpenAIChatConverter):
        req = _chat_request([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "developer", "content": "Now be formal"},
            {"role": "user", "content": "ok"},
        ])
        ir = converter.request_from_provider(req)
        # Late developer should be converted to system role in IR
        system_msgs = [m for m in ir["messages"] if m["role"] == "system"]
        assert len(system_msgs) == 1
        text = system_msgs[0]["content"][0]["text"]
        assert text == "Now be formal"

    def test_developer_content_preserved(self, converter: OpenAIChatConverter):
        """Developer message content (list format) is properly preserved."""
        req = _chat_request([
            {"role": "user", "content": "hi"},
            {"role": "developer", "content": [
                {"type": "text", "text": "Part one"},
                {"type": "text", "text": "Part two"},
            ]},
        ])
        ir = converter.request_from_provider(req)
        system_msgs = [m for m in ir["messages"] if m["role"] == "system"]
        assert len(system_msgs) == 1
        texts = [p["text"] for p in system_msgs[0]["content"]]
        assert "Part one" in texts
        assert "Part two" in texts


class TestFullRoundTrip:
    """End-to-end: Chat with late developer → IR → hoist → envelope."""

    def test_late_developer_gets_system_envelope(self, converter: OpenAIChatConverter):
        """Late developer message should end up as user-role <system> envelope after hoist."""
        req = _chat_request([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "developer", "content": "Now be formal"},
            {"role": "user", "content": "ok"},
        ])
        ir = converter.request_from_provider(req)

        # Apply hoist
        hoisted = hoist_late_system_messages_ir(ir)

        # system_instruction should have the leading system message
        assert hoisted["system_instruction"][0]["text"] == "You are helpful."

        # No system messages should remain in messages
        for msg in hoisted["messages"]:
            assert msg["role"] != "system", f"system message leaked: {msg}"

        # Late developer should be rewritten as user with <system> envelope
        user_msgs = [m for m in hoisted["messages"] if m["role"] == "user"]
        enveloped = [
            m for m in user_msgs
            if any(
                "<system>" in c.get("text", "")
                for c in m.get("content", [])
                if isinstance(c, dict)
            )
        ]
        assert len(enveloped) == 1
        assert "<system>\nNow be formal\n</system>" in enveloped[0]["content"][0]["text"]
