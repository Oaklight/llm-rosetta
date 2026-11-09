"""A reasoning part with no replayable id is reported, not silently dropped.

The Responses API replays a reasoning item only through its id
(``provider_metadata.responses_reasoning_id``).  A part converted from
another dialect carries none, so the writer cannot emit it — but the history
the client sends back is then not the history it recorded, and only a
conversion warning says so (issue #1491).
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from llm_rosetta.converters.llm.open_responses import OpenAIResponsesConverter
from llm_rosetta.types.ir import Message

_BARE = {"type": "reasoning", "reasoning": "thinking"}
_WITH_ID = {
    "type": "reasoning",
    "reasoning": "thinking",
    "provider_metadata": {"responses_reasoning_id": "rs_1"},
}


def _items(role: str, parts: list[dict[str, Any]]) -> tuple[list[dict], list[str]]:
    converter = OpenAIResponsesConverter()
    messages = cast(list[Message], [{"role": role, "content": parts}])
    return converter.message_ops.ir_messages_to_p(messages)


@pytest.mark.parametrize("role", ["assistant", "user"])
def test_a_reasoning_part_without_an_id_is_reported(role):
    items, warnings = _items(role, [_BARE, {"type": "text", "text": "answer"}])

    assert not [item for item in items if item["type"] == "reasoning"]
    assert any("no Responses reasoning id" in warning for warning in warnings), warnings


@pytest.mark.parametrize("role", ["assistant", "user"])
def test_a_reasoning_part_with_an_id_is_not_reported(role):
    """The positive case: a replayable item travels, so nothing is lost."""
    items, warnings = _items(role, [_WITH_ID, {"type": "text", "text": "answer"}])

    assert [item["type"] for item in items].count("reasoning") == 1
    assert not any("reasoning id" in warning for warning in warnings), warnings
