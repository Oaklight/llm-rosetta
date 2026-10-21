"""Preflight bodies and token counting."""

from __future__ import annotations

from typing import Any


from llm_rosetta.auto_detect import ProviderType


from ..logging import (
    get_logger,
)
from ..transport import (
    ProviderInfo,
    UpstreamTransport,
)

logger = get_logger()

_PREFLIGHT_TIMEOUT = 5.0  # seconds — bound worst-case TTFT impact


def _build_preflight_body(
    target_body: dict[str, Any], target_provider: ProviderType
) -> dict[str, Any]:
    """Build a minimal non-streaming request body for token counting.

    Uses a shallow copy of *target_body* (messages/tools are read-only)
    and only deep-copies nested dicts that are mutated, to avoid
    duplicating large conversation histories.
    """
    body = dict(target_body)

    # Disable streaming
    body.pop("stream", None)
    body.pop("stream_options", None)

    # Set minimal output
    if target_provider in ("google", "google_generate"):
        gc = dict(body.get("generationConfig", {}))
        gc["maxOutputTokens"] = 1
        gc.pop("responseModalities", None)
        body["generationConfig"] = gc
    elif target_provider == "google_interactions":
        gc = dict(body.get("generation_config", {}))
        gc["max_output_tokens"] = 1
        body["generation_config"] = gc
    elif target_provider in ("openai_responses", "open_responses"):
        body["max_output_tokens"] = 1
    else:
        body["max_tokens"] = 1

    # Strip thinking / reasoning to minimise cost
    body.pop("thinking", None)
    body.pop("reasoning", None)
    if isinstance(body.get("metadata"), dict):
        body["metadata"] = {
            k: v for k, v in body["metadata"].items() if k != "thinking"
        }

    return body


def _extract_preflight_input_tokens(
    response_body: dict[str, Any], target_provider: ProviderType
) -> int | None:
    """Extract input/prompt token count from a non-streaming response."""
    try:
        if target_provider in ("google", "google_generate"):
            return int(response_body["usageMetadata"]["promptTokenCount"])
        if target_provider == "google_interactions":
            return int(response_body["usage"]["total_input_tokens"])
        usage = response_body.get("usage", {})
        if target_provider in ("chat_completions",):
            return int(usage["prompt_tokens"])
        # Anthropic, OpenAI Responses, Open Responses
        return int(usage["input_tokens"])
    except (KeyError, TypeError, ValueError):
        return None


async def _run_preflight(
    transport: UpstreamTransport,
    provider_info: ProviderInfo,
    target_body: dict[str, Any],
    target_provider: ProviderType,
    model: str,
    *,
    extra_headers: dict[str, str] | None = None,
) -> int | None:
    """Send a preflight request and return input token count.

    Never raises — returns ``None`` on any failure so the real streaming
    request can proceed with ``input_tokens=0`` as before.
    """
    try:
        preflight_body = _build_preflight_body(target_body, target_provider)
        upstream_url = provider_info.upstream_url(model)
        pinfo = provider_info.with_timeout(_PREFLIGHT_TIMEOUT)
        resp = await transport.send(
            pinfo,
            upstream_url,
            preflight_body,
            extra_headers=extra_headers,
        )
        if resp.is_error:
            logger.debug("Preflight request failed with status %d", resp.status_code)
            return None
        if resp.body is None:
            return None
        count = _extract_preflight_input_tokens(resp.body, target_provider)
        logger.debug("Preflight input_tokens: %s", count)
        return count
    except Exception:
        logger.debug("Preflight request error", exc_info=True)
        return None
