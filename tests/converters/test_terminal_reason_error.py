"""A terminal turn the target dialect has no finish reason for.

Chat Completions cannot say "the generation failed" or "it was cancelled", so
the writer raises instead of telling the client the turn finished.  The
payload — the reason, the response id, the upstream metadata — rides on the
exception as attributes so a caller dumps it structurally.
"""

from __future__ import annotations

from typing import Any, cast

import pytest


def test_a_terminal_failure_is_a_typed_error():
    """The payload rides on the exception, not inside its message string."""
    from llm_rosetta.converters.base.errors import UnrepresentableFinishReason
    from llm_rosetta.converters.llm.chat_completions import ChatCompletionsConverter

    ir = {
        "id": "r1",
        "object": "response",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "partial"}],
                    "provider_metadata": {"upstream": "boom"},
                },
                "finish_reason": {"reason": "error"},
            }
        ],
    }

    with pytest.raises(UnrepresentableFinishReason) as raised:
        ChatCompletionsConverter().response_to_provider(cast(Any, ir))

    assert raised.value.reason == "error"
    assert raised.value.response_id == "r1"
    assert raised.value.provider_metadata == {"upstream": "boom"}
    assert "boom" not in str(raised.value)
