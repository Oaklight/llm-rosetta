"""Tests for the upstream-error tool-collision annotation."""

from __future__ import annotations

import json

from llm_rosetta.gateway.middleware.error_annotate import (
    annotate_tool_collisions,
    collision_hint,
)


class TestCollisionHint:
    def test_names_are_sorted(self):
        hint = collision_hint({"wait": 3, "exec": 2})
        assert hint.index("'exec'") < hint.index("'wait'")
        assert "2 distinct definitions" in hint
        assert "3 distinct definitions" in hint

    def test_has_marker(self):
        assert collision_hint({"a": 2}).startswith("[llm-rosetta]")


class TestAnnotateToolCollisions:
    def _body(self, message="Tool names must be unique."):
        return json.dumps(
            {"error": {"type": "invalid_request_error", "message": message}}
        )

    def test_none_collisions_is_identity(self):
        raw = self._body()
        assert annotate_tool_collisions(raw, None) is raw

    def test_empty_collisions_is_identity(self):
        raw = self._body()
        assert annotate_tool_collisions(raw, {}) is raw

    def test_str_body_stays_str_and_appends(self):
        out = annotate_tool_collisions(self._body(), {"exec": 2})
        assert isinstance(out, str)
        message = json.loads(out)["error"]["message"]
        assert message.startswith("Tool names must be unique.")
        assert "[llm-rosetta]" in message
        assert "'exec'" in message

    def test_bytes_body_stays_bytes(self):
        raw = self._body().encode()
        out = annotate_tool_collisions(raw, {"exec": 2})
        assert isinstance(out, bytes)
        assert b"[llm-rosetta]" in out

    def test_anthropic_envelope(self):
        raw = json.dumps(
            {
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "dup"},
            }
        )
        out = json.loads(annotate_tool_collisions(raw, {"wait": 2}))
        assert "[llm-rosetta]" in out["error"]["message"]

    def test_google_envelope(self):
        raw = json.dumps(
            {"error": {"code": 400, "message": "dup", "status": "INVALID_ARGUMENT"}}
        )
        out = json.loads(annotate_tool_collisions(raw, {"wait": 2}))
        assert "[llm-rosetta]" in out["error"]["message"]
        assert out["error"]["code"] == 400  # untouched

    def test_parse_failure_returns_input(self):
        raw = "not json"
        assert annotate_tool_collisions(raw, {"exec": 2}) == raw

    def test_non_object_json_returns_input(self):
        raw = "[1, 2, 3]"
        assert annotate_tool_collisions(raw, {"exec": 2}) == raw

    def test_missing_message_returns_input(self):
        raw = json.dumps({"error": {"type": "x"}})
        assert annotate_tool_collisions(raw, {"exec": 2}) == raw

    def test_non_str_message_returns_input(self):
        raw = json.dumps({"error": {"message": {"nested": True}}})
        assert annotate_tool_collisions(raw, {"exec": 2}) == raw

    def test_no_error_object_returns_input(self):
        raw = json.dumps({"status": "error"})
        assert annotate_tool_collisions(raw, {"exec": 2}) == raw

    def test_non_utf8_bytes_returns_input(self):
        raw = b"\xff\xfe not utf-8"
        assert annotate_tool_collisions(raw, {"exec": 2}) is raw

    def test_already_annotated_is_identity(self):
        raw = annotate_tool_collisions(self._body(), {"exec": 2})
        assert annotate_tool_collisions(raw, {"exec": 2}) == raw

    def test_empty_message_gets_hint_only(self):
        out = json.loads(annotate_tool_collisions(self._body(message=""), {"exec": 2}))
        message = out["error"]["message"]
        assert message.startswith("[llm-rosetta]")
        assert not message.startswith("\n")
