"""OpenAI Responses converter constants — event types, status mappings, and ID generation."""

# --- SSE event types ---


class ResponsesEventType:
    """OpenAI Responses API server-sent event type constants."""

    RESPONSE_CREATED = "response.created"
    RESPONSE_IN_PROGRESS = "response.in_progress"
    RESPONSE_COMPLETED = "response.completed"
    RESPONSE_FAILED = "response.failed"
    RESPONSE_DONE = "response.done"
    OUTPUT_ITEM_ADDED = "response.output_item.added"
    OUTPUT_ITEM_DONE = "response.output_item.done"
    CONTENT_PART_ADDED = "response.content_part.added"
    CONTENT_PART_DONE = "response.content_part.done"
    OUTPUT_TEXT_DELTA = "response.output_text.delta"
    OUTPUT_TEXT_DONE = "response.output_text.done"
    REASONING_SUMMARY_TEXT_DELTA = "response.reasoning_summary_text.delta"
    FUNCTION_CALL_ARGS_DELTA = "response.function_call_arguments.delta"
    FUNCTION_CALL_ARGS_DONE = "response.function_call_arguments.done"


# --- Status <-> Reason mappings ---

# from_provider: response status -> IR finish reason (simple cases)
RESPONSES_STATUS_TO_REASON: dict[str, str] = {
    "completed": "stop",
    "failed": "error",
    "cancelled": "cancelled",
}

# from_provider: incomplete_details.reason -> IR finish reason
RESPONSES_INCOMPLETE_REASON_TO_IR: dict[str, str] = {
    "max_output_tokens": "length",
    "content_filter": "content_filter",
}

# to_provider: IR finish reason -> response status
# NOTE: content_filter -> "completed" is a known gap, tracked in #90
RESPONSES_REASON_TO_STATUS: dict[str, str] = {
    "stop": "completed",
    "length": "incomplete",
    "error": "failed",
    "tool_calls": "completed",
    "content_filter": "completed",  # TODO: should be "incomplete", see #90
    "cancelled": "cancelled",
    "refusal": "completed",
}


# --- ID generation ---


def generate_message_id(response_id: str) -> str:
    """Generate a message item ID from the response ID."""
    return f"msg_{response_id or ''}"
