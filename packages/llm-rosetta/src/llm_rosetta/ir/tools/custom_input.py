"""A custom tool call's input text and its IR ``tool_input``.

A custom tool takes plain text, not JSON.  The IR keeps the text as
``{"input": text}`` so ``tool_input`` is always a dict.  A text that is itself a
JSON object stays structured, so other converters can read its fields.  These
two functions are the one place that rule lives, for the body and the stream
legs of every dialect.

Known limits of the rule, both of which follow from keeping a JSON object
structured: the text of a JSON object is re-serialized on the way out
(``{"a":1}`` becomes ``{"a": 1}``), and the text ``{"input": "x"}`` reads as the
dict ``{"input": "x"}`` and is written as ``x``.  The Chat Completions reader
does not keep objects structured: it reads the same text as
``{"input": text}``.

A wrapper whose ``input`` is not a string (``{"input": true}``) is written as
the JSON of that value (``true``, ``null``, ``[1]``), not as a Python repr
(``True``, ``None``); before #1720 the body writer wrote the repr.  A value
JSON cannot encode (``bytes``, a ``set``) falls back to ``str()`` and does not
raise.
"""

from __future__ import annotations

import json
from typing import Any


def custom_input_text(tool_input: Any) -> str:
    """The plain-text ``input`` a custom tool call carries on the wire.

    Args:
        tool_input: The IR tool call's ``tool_input``.

    Returns:
        The text the wire carries.
    """
    if isinstance(tool_input, dict) and list(tool_input.keys()) == ["input"]:
        value = tool_input["input"]
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value)
        except TypeError:
            # A hand-built IR may hold a value JSON cannot encode; keep the
            # text ``str()`` gave before #1720 rather than raise.
            return str(value)
    return json.dumps(tool_input) if isinstance(tool_input, dict) else str(tool_input)


def custom_input_from_text(text: str) -> dict[str, Any]:
    """The IR ``tool_input`` for the text a custom tool call carried.

    Args:
        text: The wire text, possibly empty.

    Returns:
        The decoded dict when the text is a JSON object; otherwise
        ``{"input": text}`` (an empty text and a JSON scalar included).
    """
    if text:
        try:
            decoded = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            decoded = None
        if isinstance(decoded, dict):
            return decoded
    return {"input": text}


def custom_input_text_of_wrapper(wrapper: str) -> str:
    """The plain text carried by a delta that is a custom call's IR wrapper.

    The wrapper is the JSON of the call's ``tool_input``, delivered whole (see
    ``ToolCallDeltaEvent.input_wrapped``).  A text that does not parse is
    returned as it is.

    Args:
        wrapper: The ``arguments_delta`` of a flagged delta.

    Returns:
        The text the wire carries, by the same rule as :func:`custom_input_text`.
    """
    try:
        decoded = json.loads(wrapper)
    except (json.JSONDecodeError, TypeError):
        return wrapper
    return custom_input_text(decoded)
