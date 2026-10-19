"""Google GenAI converter constants — reason mappings and ID generation."""

import uuid

# --- Reason mappings ---

GOOGLE_REASON_FROM_PROVIDER: dict[str, str] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
    "IMAGE_SAFETY": "content_filter",
    "LANGUAGE": "error",
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


# --- Output formats ---

#: Wire shapes the Google converter can emit. ``"rest"`` builds a body for the
#: Generative Language REST API (``generationConfig`` / top-level ``tools``);
#: ``"sdk"`` builds the ``google.genai`` client's constructor kwargs
#: (``config``), which is not a wire format — the REST endpoint rejects it with
#: HTTP 400 ``Unknown name "config"``.
GOOGLE_OUTPUT_FORMATS: tuple[str, ...] = ("rest", "sdk")


def validate_google_output_format(
    output_format: str, *, param: str = "output_format"
) -> str:
    """Validate a Google converter output-format selector.

    The converter selects the REST body by exact match and otherwise emits the
    SDK shape, so an unrecognised value would silently produce a body the REST
    endpoint rejects. Reject it here instead of failing on the upstream call.

    Args:
        output_format: The value to validate — ``"rest"`` or ``"sdk"``.
        param: Name of the parameter being validated, used in the error message
            so callers see their own spelling.

    Returns:
        ``output_format``, unchanged.

    Raises:
        ValueError: If ``output_format`` is not one of
            :data:`GOOGLE_OUTPUT_FORMATS`.
    """
    if output_format not in GOOGLE_OUTPUT_FORMATS:
        raise ValueError(
            f"{param} must be one of {GOOGLE_OUTPUT_FORMATS}, got {output_format!r}"
        )
    return output_format


# --- ID generation ---


def generate_tool_call_id() -> str:
    """Generate a unique tool call ID for Google function calls.

    Google's API does not provide tool call IDs, so we generate them
    using a ``call_`` prefix followed by 24 hex characters from a UUID.
    """
    return f"call_{uuid.uuid4().hex[:24]}"
