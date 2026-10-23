"""A tool definition with no name never reaches the wire.

Every dialect identifies a tool by its name, so a nameless definition can only
be sent as an empty string, and both upstreams reject that with a 400 naming a
position in a body the client never wrote — measured live 2026-10-23:

    Chat Completions  400 Invalid 'tools[0].function.name': empty string
    Messages          400 tools.0.custom.name: String should have at least 1
                          character

A nameless definition is what an IR request carries when a client body holds a
tool type the source reader does not know and no name to fall back on, so the
request used to fail upstream for a reason the client could not see.
"""

from __future__ import annotations

from typing import Any

import pytest

from llm_rosetta.conversion_warnings import DROPPED, WarningSink, recording_to
from llm_rosetta.ir.passes import drop_nameless_tools
from llm_rosetta.pipeline import ConversionPipeline

NAMED = {
    "type": "function",
    "name": "lookup",
    "description": "look something up",
    "parameters": {"type": "object", "properties": {}},
}


def _request(*tools: Any) -> dict[str, Any]:
    return {"model": "m", "messages": [], "tools": list(tools)}


class TestThePass:
    def test_a_nameless_definition_is_dropped(self):
        result = drop_nameless_tools(_request(NAMED, {"description": "no name"}))

        assert result["tools"] == [NAMED]

    @pytest.mark.parametrize("name", ["", "   "])
    def test_an_empty_name_counts_as_nameless(self, name: str):
        result = drop_nameless_tools(_request({**NAMED, "name": name}))

        assert result["tools"] == []

    def test_a_named_definition_is_untouched(self):
        request = _request(NAMED)

        assert drop_nameless_tools(request) is request

    def test_a_request_without_tools_is_untouched(self):
        request: dict[str, Any] = {"model": "m", "messages": []}

        assert drop_nameless_tools(request) is request

    def test_the_drop_is_reported_once(self):
        sink = WarningSink()
        with recording_to(sink):
            drop_nameless_tools(_request({"description": "a"}, {"description": "b"}))

        assert len(sink) == 1
        assert sink[0].code == DROPPED
        assert sink[0].subject == "tools"
        assert "2 tool definition(s) with no name" in sink[0]
        # No name to identify them by, so the report names their positions.
        assert "tools[0]" in sink[0]
        assert "tools[1]" in sink[0]


class TestThroughThePipeline:
    """The case from the issue: a client body carrying a tool the reader cannot name."""

    BODY: dict[str, Any] = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "intrinsic", "intrinsic_kind": "web_search"}],
    }

    def test_the_empty_name_does_not_survive(self):
        pipe = ConversionPipeline(
            "messages", "messages", target_shim="minimax--messages"
        )

        body = pipe.convert_request(dict(self.BODY))

        assert not body.get("tools")
        assert any("with no name" in warning for warning in pipe.warnings)

    def test_a_cross_format_target_is_covered_too(self):
        pipe = ConversionPipeline("messages", "chat_completions")

        body = pipe.convert_request(dict(self.BODY))

        assert not body.get("tools")
        assert any("with no name" in warning for warning in pipe.warnings)

    def test_a_named_tool_beside_it_still_goes_through(self):
        request = dict(self.BODY)
        request["tools"] = [
            {"type": "intrinsic", "intrinsic_kind": "web_search"},
            {
                "name": "lookup",
                "description": "",
                "input_schema": {"type": "object"},
            },
        ]
        pipe = ConversionPipeline(
            "messages", "messages", target_shim="minimax--messages"
        )

        body = pipe.convert_request(request)

        assert [tool.get("name") for tool in body["tools"]] == ["lookup"]
        assert any("1 tool definition(s) with no name" in w for w in pipe.warnings)


class TestTheForcedChoice:
    """A dropped definition must not leave a ``tool_choice`` forcing it.

    The converter-side orphan fixer used to fire only when *no* tools were
    left; with one nameless definition beside a good one, a choice forcing the
    nameless tool survived into a body that still had tools, which no lane
    accepts.
    """

    BODY: dict[str, Any] = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {"type": "intrinsic", "intrinsic_kind": "web_search"},
            {"name": "lookup", "description": "", "input_schema": {"type": "object"}},
        ],
        "tool_choice": {"type": "tool", "name": ""},
    }

    def test_a_choice_forcing_the_nameless_tool_does_not_survive(self):
        pipe = ConversionPipeline(
            "messages", "messages", target_shim="minimax--messages"
        )

        body = pipe.convert_request(dict(self.BODY))

        assert [tool.get("name") for tool in body["tools"]] == ["lookup"]
        assert "tool_choice" not in body
        assert any("forcing it was removed" in warning for warning in pipe.warnings)

    def test_a_choice_forcing_a_surviving_tool_is_kept(self):
        request = dict(self.BODY)
        request["tools"] = [
            {"name": "lookup", "description": "", "input_schema": {"type": "object"}},
        ]
        request["tool_choice"] = {"type": "tool", "name": "lookup"}
        pipe = ConversionPipeline(
            "messages", "messages", target_shim="minimax--messages"
        )

        body = pipe.convert_request(request)

        assert body["tool_choice"] == {"type": "tool", "name": "lookup"}
        assert not any("forcing it was removed" in warning for warning in pipe.warnings)

    def test_the_lone_nameless_tool_takes_its_choice_with_it(self):
        """Pinned against the orphan stripper moving ahead of the tool passes."""
        request = dict(self.BODY)
        request["tools"] = [{"type": "intrinsic", "intrinsic_kind": "web_search"}]
        request["tool_choice"] = {"mode": "auto"}
        pipe = ConversionPipeline(
            "messages", "messages", target_shim="minimax--messages"
        )

        body = pipe.convert_request(request)

        assert "tools" not in body
        assert "tool_choice" not in body
