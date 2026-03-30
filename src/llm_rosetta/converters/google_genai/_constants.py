"""Google GenAI converter constants — reason mappings and ID generation."""

import uuid

# --- Reason mappings ---

GOOGLE_REASON_FROM_PROVIDER: dict[str, str] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "MALFORMED_FUNCTION_CALL": "error",
    "OTHER": "error",
}

GOOGLE_REASON_TO_PROVIDER: dict[str, str] = {
    "stop": "STOP",
    "length": "MAX_TOKENS",
    "content_filter": "SAFETY",
    "tool_calls": "STOP",
    "error": "OTHER",
}


# --- ID generation ---


def generate_tool_call_id() -> str:
    """Generate a unique tool call ID for Google function calls.

    Google's API does not provide tool call IDs, so we generate them
    using a ``call_`` prefix followed by 24 hex characters from a UUID.
    """
    return f"call_{uuid.uuid4().hex[:24]}"
