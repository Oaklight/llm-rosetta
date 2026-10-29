"""Anthropic converter constants — reason mappings and event types."""

# --- Reason mappings ---

ANTHROPIC_REASON_FROM_PROVIDER: dict[str, str] = {
    "end_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "stop_sequence": "stop",
    "refusal": "refusal",
}

ANTHROPIC_REASON_TO_PROVIDER: dict[str, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    # A filtered turn is the closest thing Anthropic has to a refusal, and it
    # has the word: `end_turn` would tell the client the model finished
    # talking (#1236).  Anthropic has no finer distinction here — the
    # filtered family (`SAFETY`, `RECITATION`, …) all arrive as this.
    "content_filter": "refusal",
    "refusal": "refusal",
}


# --- SSE event types ---


class AnthropicEventType:
    """Anthropic Messages API server-sent event type constants."""

    PING = "ping"
    MESSAGE_START = "message_start"
    CONTENT_BLOCK_START = "content_block_start"
    CONTENT_BLOCK_DELTA = "content_block_delta"
    CONTENT_BLOCK_STOP = "content_block_stop"
    MESSAGE_DELTA = "message_delta"
    MESSAGE_STOP = "message_stop"
