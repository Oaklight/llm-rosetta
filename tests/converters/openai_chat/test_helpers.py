"""Unit tests for module-level helpers extracted from the OpenAI Chat converter.

These target the small pure helpers introduced when reducing cognitive
complexity so they stay covered even if the parent functions change shape.
"""

from llm_rosetta.converters.openai_chat.message_ops import (
    OpenAIChatMessageOps,
    _handle_synthetic_tag,
    _parse_synthetic_tool_content_msg,
    _restore_reasoning_metadata,
)


def test_restore_reasoning_metadata_copies_details_and_encrypted() -> None:
    target: dict = {}
    parts = [
        {"type": "text", "text": "hi"},
        {
            "type": "reasoning",
            "reasoning": "think",
            "provider_metadata": {
                "openai_chat": {
                    "reasoning_details": [{"id": "r1"}],
                    "encrypted_content": "abc",
                }
            },
        },
    ]
    _restore_reasoning_metadata(target, parts)
    assert target == {
        "reasoning_details": [{"id": "r1"}],
        "encrypted_content": "abc",
    }


def test_restore_reasoning_metadata_skips_without_metadata() -> None:
    target: dict = {}
    parts = [{"type": "reasoning", "reasoning": "x"}]
    _restore_reasoning_metadata(target, parts)
    assert target == {}


def test_restore_reasoning_metadata_uses_first_reasoning_part() -> None:
    target: dict = {}
    parts = [
        {
            "type": "reasoning",
            "reasoning": "first",
            "provider_metadata": {"openai_chat": {"reasoning_details": ["A"]}},
        },
        {
            "type": "reasoning",
            "reasoning": "second",
            "provider_metadata": {"openai_chat": {"reasoning_details": ["B"]}},
        },
    ]
    _restore_reasoning_metadata(target, parts)
    assert target == {"reasoning_details": ["A"]}


def test_parse_synthetic_tool_content_msg_multi_sections() -> None:
    unpacked: dict = {}
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": '<tool-content call-id="c1">'},
            {"type": "image_url", "image_url": {"url": "u1"}},
            {"type": "text", "text": "</tool-content>"},
            {"type": "text", "text": '<tool-content call-id="c2">'},
            {"type": "image_url", "image_url": {"url": "u2"}},
            {"type": "text", "text": "</tool-content>"},
        ],
    }
    _parse_synthetic_tool_content_msg(msg, unpacked)
    assert list(unpacked.keys()) == ["c1", "c2"]
    assert unpacked["c1"][0]["image_url"]["url"] == "u1"
    assert unpacked["c2"][0]["image_url"]["url"] == "u2"


def test_parse_synthetic_tool_content_msg_unclosed_section_saved() -> None:
    unpacked: dict = {}
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": '<tool-content call-id="c1">'},
            {"type": "image_url", "image_url": {"url": "u1"}},
        ],
    }
    _parse_synthetic_tool_content_msg(msg, unpacked)
    assert "c1" in unpacked
    assert unpacked["c1"][0]["image_url"]["url"] == "u1"


def test_handle_synthetic_tag_ignores_non_text_part() -> None:
    assert (
        _handle_synthetic_tag(
            {"type": "image_url", "image_url": {"url": "x"}}, {}, None, []
        )
        is None
    )


def test_reorder_split_index_rebuild_helpers() -> None:
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "t1"}, {"id": "t2"}]},
        {"role": "tool", "tool_call_id": "t2", "content": "b"},
        {"role": "tool", "tool_call_id": "t1", "content": "a"},
    ]
    tool_msgs, non_tool = OpenAIChatMessageOps._split_tool_and_non_tool(messages)
    assert len(tool_msgs) == 2 and len(non_tool) == 1

    by_id = OpenAIChatMessageOps._index_tools_by_call_id(tool_msgs)
    assert set(by_id.keys()) == {"t1", "t2"}

    warnings: list[str] = []
    rebuilt = OpenAIChatMessageOps._rebuild_with_tool_pairing(
        non_tool, by_id, tool_msgs, warnings
    )
    # Tool messages emitted in tool_calls order, not original order.
    assert [m.get("tool_call_id") for m in rebuilt if m.get("role") == "tool"] == [
        "t1",
        "t2",
    ]
    assert warnings == []


def test_rebuild_with_tool_pairing_appends_unmatched() -> None:
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "t1"}]},
        {"role": "tool", "tool_call_id": "t1", "content": "a"},
        {"role": "tool", "tool_call_id": "orphan", "content": "b"},
    ]
    tool_msgs, non_tool = OpenAIChatMessageOps._split_tool_and_non_tool(messages)
    by_id = OpenAIChatMessageOps._index_tools_by_call_id(tool_msgs)
    warnings: list[str] = []
    rebuilt = OpenAIChatMessageOps._rebuild_with_tool_pairing(
        non_tool, by_id, tool_msgs, warnings
    )
    ids = [m.get("tool_call_id") for m in rebuilt if m.get("role") == "tool"]
    assert ids == ["t1", "orphan"]
    assert any("orphan" in w for w in warnings)
