"""Tests for the upstream-error tool annotations."""

from __future__ import annotations

import json

from llm_rosetta.capabilities import ToolNameMap
from llm_rosetta_gateway.middleware.error_annotate import (
    ToolSchemaLocator,
    annotate_tool_collisions,
    annotate_tool_errors,
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

    def test_names_capped_with_remainder_summary(self):
        collisions = {f"tool{i:03d}": 2 for i in range(25)}
        hint = collision_hint(collisions)
        # Only the first 10 names are listed, in sorted order.
        assert hint.count("declared tool name") == 10
        assert "'tool000'" in hint
        assert "'tool009'" in hint
        assert "'tool010'" not in hint
        assert "and 15 more" in hint

    def test_no_remainder_summary_at_the_cap(self):
        collisions = {f"tool{i:03d}": 2 for i in range(10)}
        assert "more" not in collision_hint(collisions)

    def test_large_map_stays_small(self):
        # 200 names must not produce a multi-kilobyte message.
        collisions = {f"tool{i:03d}": 2 for i in range(200)}
        assert len(collision_hint(collisions)) < 1000


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

    def test_non_ascii_message_stays_readable(self):
        # ensure_ascii=False: non-ASCII text is not escaped on the way out.
        out = annotate_tool_collisions(self._body(message="重复的工具名"), {"exec": 2})
        assert isinstance(out, str)
        assert "重复的工具名" in out
        assert "\\u" not in out


# ---------------------------------------------------------------------------
# Tool-schema attribution (#1427)
# ---------------------------------------------------------------------------


def _chat_tools(names: list[str]) -> dict:
    """A Chat Completions body declaring *names* as plain functions."""
    return {
        "tools": [
            {"type": "function", "function": {"name": name, "parameters": {}}}
            for name in names
        ]
    }


def _genai_tools(names: list[str]) -> dict:
    """A Google body declaring *names* inside one functionDeclarations list."""
    return {"tools": [{"functionDeclarations": [{"name": name} for name in names]}]}


def _hint(locator: ToolSchemaLocator, message: str) -> str:
    """The locator's attribution for *message*, asserted present."""
    hint = locator.hint(message)
    assert hint is not None
    return hint


class TestToolSchemaLocator:
    """An upstream reference is turned into the declaration the client wrote."""

    def test_an_index_maps_to_the_clients_own_declaration(self):
        """``tools[16]`` on the wire is declaration #17 in the genai body."""
        locator = ToolSchemaLocator(
            upstream_format="chat_completions",
            client_format="google_generate",
            upstream_body=_chat_tools(
                [f"tool{i}" for i in range(16)] + ["send_message"]
            ),
            client_body=_genai_tools(
                [f"tool{i}" for i in range(16)] + ["send_message"]
            ),
        )
        hint = _hint(
            locator,
            "Invalid 'tools[16].function.name': string does not match pattern "
            "'^[a-zA-Z0-9_-]+$'",
        )
        assert "the client's tool 'send_message'" in hint
        assert "declaration #17" in hint
        assert "tools[0].functionDeclarations[16]" in hint
        # The within-tool path is translated out of Chat's ``function`` wrapper.
        assert "tools[0].functionDeclarations[16].name" in hint

    def test_a_sanitized_name_reports_the_clients_spelling(self):
        name_map = ToolNameMap()
        name_map.record_rename("mcp/srv read", None, "mcp_srv_read")
        locator = ToolSchemaLocator(
            upstream_format="chat_completions",
            client_format="chat_completions",
            upstream_body=_chat_tools(["mcp_srv_read"]),
            client_body=_chat_tools(["mcp/srv read"]),
            name_map=name_map,
        )
        hint = _hint(
            locator,
            "Invalid 'tools[0].function.name': string does not match pattern "
            "'^[a-zA-Z0-9_-]+$'",
        )
        assert "the client's tool 'mcp/srv read'" in hint

    def test_a_message_naming_only_the_function(self):
        locator = ToolSchemaLocator(
            upstream_format="chat_completions",
            client_format="google_generate",
            upstream_body=_chat_tools(["send_message"]),
            client_body=_genai_tools(["send_message"]),
        )
        hint = _hint(
            locator,
            'Invalid schema for function \'send_message\': "1" is not of type "integer"',
        )
        assert "declaration #1" in hint
        assert "its schema is at tools[0].functionDeclarations[0].parameters" in hint

    def test_a_context_tuple_becomes_a_schema_path(self):
        locator = ToolSchemaLocator(
            upstream_format="chat_completions",
            client_format="messages",
            upstream_body=_chat_tools(["send_message"]),
            client_body={"tools": [{"name": "send_message", "input_schema": {}}]},
        )
        hint = _hint(
            locator,
            "Invalid schema for function 'send_message': In context=('properties', "
            "'count'), 'type' is not valid",
        )
        # Messages spells the schema field ``input_schema``.
        assert "tools[0].input_schema.properties.count" in hint

    def test_anthropic_positional_spelling_resolves(self):
        locator = ToolSchemaLocator(
            upstream_format="messages",
            client_format="messages",
            upstream_body={"tools": [{"name": "one"}, {"name": "two"}]},
            client_body={"tools": [{"name": "one"}, {"name": "two"}]},
        )
        hint = _hint(
            locator, "tools.1.input_schema: Input should be a valid dictionary"
        )
        assert "the client's tool 'two'" in hint
        assert "declaration #2 (tools[1])" in hint

    def test_google_snake_case_positional_spelling_resolves(self):
        locator = ToolSchemaLocator(
            upstream_format="google_generate",
            client_format="google_generate",
            upstream_body=_genai_tools(["one", "two"]),
            client_body=_genai_tools(["one", "two"]),
        )
        hint = _hint(
            locator,
            "Invalid JSON payload received. Unknown name at "
            "'tools[0].function_declarations[1].parameters'",
        )
        assert "the client's tool 'two'" in hint
        assert "tools[0].functionDeclarations[1].parameters" in hint

    def test_a_nested_upstream_path_is_rendered_in_the_clients_dialect(self):
        """Google's nesting and Chat's ``function`` wrapper both disappear."""
        hint = _hint(
            ToolSchemaLocator(
                upstream_format="google_generate",
                client_format="chat_completions",
                upstream_body=_genai_tools(["one", "two"]),
                client_body=_chat_tools(["one", "two"]),
            ),
            "Invalid value at 'tools[0].function_declarations[1]"
            ".parameters.properties.count'",
        )
        assert "the client's tool 'two'" in hint
        assert "tools[1].function.parameters.properties.count" in hint

    def test_a_message_that_names_no_tool_gets_no_hint(self):
        locator = ToolSchemaLocator(
            upstream_format="chat_completions",
            client_format="chat_completions",
            upstream_body=_chat_tools(["exec"]),
            client_body=_chat_tools(["exec"]),
        )
        assert locator.hint("Context length exceeded") is None

    def test_a_tool_the_client_never_declared_gets_no_hint(self):
        locator = ToolSchemaLocator(
            upstream_format="chat_completions",
            client_format="chat_completions",
            upstream_body=_chat_tools(["ghost"]),
            client_body=_chat_tools(["exec"]),
        )
        assert locator.hint("Invalid schema for function 'ghost': bad") is None

    def test_an_index_past_the_end_gets_no_hint(self):
        locator = ToolSchemaLocator(
            upstream_format="chat_completions",
            client_format="chat_completions",
            upstream_body=_chat_tools(["exec"]),
            client_body=_chat_tools(["exec"]),
        )
        assert locator.hint("Invalid 'tools[9].function.name': bad") is None

    def test_the_client_body_is_read_lazily(self):
        """A message with no tool reference never touches either body."""
        locator = ToolSchemaLocator(
            upstream_format="chat_completions",
            client_format="chat_completions",
            upstream_body=None,
            client_body=None,
        )
        assert locator.hint("plain failure") is None


class TestAnnotateToolErrors:
    """Both diagnostics travel in one rewrite, behind one idempotency marker."""

    def _body(self, message: str = "Invalid schema for function 'exec': bad") -> str:
        return json.dumps(
            {"error": {"type": "invalid_request_error", "message": message}}
        )

    def _locator(self) -> ToolSchemaLocator:
        return ToolSchemaLocator(
            upstream_format="chat_completions",
            client_format="chat_completions",
            upstream_body=_chat_tools(["exec"]),
            client_body=_chat_tools(["exec"]),
        )

    def test_both_hints_are_appended_to_the_upstream_message(self):
        out = annotate_tool_errors(
            self._body(), {"exec": 2}, hint_for=self._locator().hint
        )
        message = json.loads(out)["error"]["message"]
        assert message.startswith("Invalid schema for function 'exec': bad")
        assert "declared tool name 'exec'" in message
        assert "the client's declaration #1" in message

    def test_the_message_is_kept_verbatim_above_the_hints(self):
        """The upstream's own words stay first and unchanged."""
        out = annotate_tool_errors(self._body(), hint_for=self._locator().hint)
        message = json.loads(out)["error"]["message"]
        assert message.split("\n\n")[0] == "Invalid schema for function 'exec': bad"

    def test_bytes_body_stays_bytes(self):
        out = annotate_tool_errors(self._body().encode(), hint_for=self._locator().hint)
        assert isinstance(out, bytes)
        assert b"the client's declaration #1" in out

    def test_nothing_to_add_is_identity(self):
        raw = self._body()
        assert annotate_tool_errors(raw, None, hint_for=None) is raw

    def test_a_hintless_message_is_identity(self):
        raw = self._body("model not found")
        assert annotate_tool_errors(raw, hint_for=self._locator().hint) is raw

    def test_an_annotated_message_is_not_annotated_again(self):
        raw = annotate_tool_errors(self._body(), hint_for=self._locator().hint)
        again = annotate_tool_errors(raw, {"exec": 2}, hint_for=self._locator().hint)
        assert again == raw
