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
RESPONSES_REASON_TO_STATUS: dict[str, str] = {
    "stop": "completed",
    "length": "incomplete",
    "error": "failed",
    "tool_calls": "completed",
    "content_filter": "incomplete",
    "cancelled": "cancelled",
    "refusal": "completed",
}

# to_provider: IR finish reason -> incomplete_details.reason
RESPONSES_REASON_TO_INCOMPLETE_REASON: dict[str, str] = {
    "length": "max_output_tokens",
    "content_filter": "content_filter",
}


# --- ID generation ---


def generate_message_id(response_id: str) -> str:
    """Generate a message item ID from the response ID."""
    return f"msg_{response_id or ''}"


# --- Preserve-mode echo fields ---

# Fields from the OpenAI Responses API response that are not captured in IR.
# These are request echo-back parameters, lifecycle metadata, and config that
# the Responses API includes in every response resource.
RESPONSES_PRESERVE_FIELDS: set[str] = {
    # Request echo-back
    "background",
    "frequency_penalty",
    "instructions",
    "max_output_tokens",
    "max_tool_calls",
    "parallel_tool_calls",
    "presence_penalty",
    "previous_response_id",
    "prompt_cache_key",
    "prompt_cache_retention",
    "reasoning",
    "safety_identifier",
    "store",
    "temperature",
    "text",
    "tool_choice",
    "tools",
    "top_logprobs",
    "top_p",
    "truncation",
    "user",
    "metadata",
    # Lifecycle metadata
    "billing",
    "completed_at",
    "error",
    "incomplete_details",
}
