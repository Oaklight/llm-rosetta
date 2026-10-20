"""Strict Chat upstreams require string tool content (#1121).

The Chat wire format allows list-valued ``role: "tool"`` content, but the
OpenAI-compatible gRPC lanes behind it reject anything but a string.  Since
``_SUPPORTS_MULTIMODAL_TOOL_RESULT`` is ``False`` for this converter, the writer
flattens text-only content and carries ``is_error`` as a marker line.
"""

from __future__ import annotations

from typing import Any

from llm_rosetta import ConversionPipeline
from llm_rosetta.converters.base.tools.error_marker import TOOL_ERROR_MARKER


def _convert(result_content: Any, *, is_error: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": "t1",
        "content": result_content,
    }
    if is_error:
        result["is_error"] = True
    return ConversionPipeline("messages", "chat_completions").convert_request(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "bash", "input": {}}
                    ],
                },
                {"role": "user", "content": [result]},
            ]
        }
    )


def _tool_message(body: dict[str, Any]) -> dict[str, Any]:
    tool_messages = [m for m in body["messages"] if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    return tool_messages[0]


def test_a_single_text_block_becomes_a_string() -> None:
    message = _tool_message(_convert([{"type": "text", "text": "ok"}]))

    assert message["content"] == "ok"


def test_multiple_text_blocks_join_with_newlines() -> None:
    message = _tool_message(
        _convert(
            [
                {"type": "text", "text": "line one"},
                {"type": "text", "text": "line two"},
            ]
        )
    )

    assert message["content"] == "line one\nline two"


def test_an_error_result_is_a_string_with_the_marker() -> None:
    message = _tool_message(_convert("Exit code 1", is_error=True))

    assert message["content"] == f"Exit code 1\n{TOOL_ERROR_MARKER}"


def test_an_error_after_a_text_list_is_a_string_with_the_marker() -> None:
    message = _tool_message(
        _convert([{"type": "text", "text": "Exit code 1"}], is_error=True)
    )

    assert message["content"] == f"Exit code 1\n{TOOL_ERROR_MARKER}"


def test_a_silent_error_is_the_marker_alone() -> None:
    message = _tool_message(_convert("", is_error=True))

    assert message["content"] == TOOL_ERROR_MARKER
