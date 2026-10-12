"""Tool-result error marking for formats with no native error field.

Anthropic and Google Interactions mark a failed tool result natively with
``is_error``; Google Generate encodes the bit in the payload structure
(``response.error`` instead of ``response.output``).  Chat Completions and
Responses have neither, so a ``true`` flag would otherwise vanish without
trace.

Both OpenAI formats can still carry the signal as a trailing text block.
The model reads a plain textual failure marker the same way it reads the
native flag, and the marker stays harmless when the tool output already
describes the failure.  It is deliberately *not* machine-parseable: there
is no envelope to unwrap on the way back, the text is just text.
"""

from __future__ import annotations

from typing import Any

#: Text of the trailing block appended to a failed tool result.
TOOL_ERROR_MARKER = "[Tool error]"


def mark_tool_error(
    content: str | list[Any],
    *,
    text_block_type: str,
) -> list[dict[str, Any]]:
    """Append the error marker to a converted tool result payload.

    A bare string is normalized into a single text block followed by the
    marker, so the caller emits one shape regardless of what the IR carried.

    Args:
        content: Converted payload — either a plain string or a list of
            provider content blocks.
        text_block_type: Provider block type for text (``"text"`` for Chat
            Completions, ``"input_text"`` for Responses).

    Returns:
        Provider content blocks, with the marker block last.
    """
    blocks: list[dict[str, Any]] = []
    if isinstance(content, str):
        if content:
            blocks.append({"type": text_block_type, "text": content})
    else:
        blocks.extend(content)
    blocks.append({"type": text_block_type, "text": TOOL_ERROR_MARKER})
    return blocks


def warn_tool_error(
    warnings: list[str] | None,
    *,
    format_name: str,
    tool_call_id: str,
) -> None:
    """Record that ``is_error`` was re-encoded instead of carried natively.

    Args:
        warnings: Warning sink from the conversion path.  No-op when absent,
            since some callers (the public ``tool_ops`` dispatchers) convert a
            single part without a conversion context.
        format_name: Target format name, used verbatim in the message.
        tool_call_id: ID of the tool result that carried the flag.
    """
    if warnings is None:
        return
    warnings.append(
        f"{format_name} has no standard tool error field; is_error=true on "
        f"tool_call_id {tool_call_id!r} was re-encoded as a trailing "
        f"{TOOL_ERROR_MARKER!r} text block instead of being dropped."
    )


def warn_tool_error_dropped(
    warnings: list[str] | None,
    *,
    format_name: str,
    tool_call_id: str,
) -> None:
    """Record that ``is_error`` had nowhere to go and was dropped.

    Used where the target emits no tool result item at all, so there is no
    payload to hang a marker on — an intrinsic result whose call becomes a
    native server item.  The flag is unrepresentable there; saying so is the
    only way to keep the conversion from being silently lossy.

    Args:
        warnings: Warning sink from the conversion path.  No-op when absent.
        format_name: Target format name, used verbatim in the message.
        tool_call_id: ID of the tool result that carried the flag.
    """
    if warnings is None:
        return
    warnings.append(
        f"{format_name} emits no output item for this tool result, so "
        f"is_error=true on tool_call_id {tool_call_id!r} was dropped."
    )
