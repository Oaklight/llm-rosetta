"""Error dump utilities: image offload, body hashing, compression.

When an upstream or conversion error occurs, the gateway can capture the
full request context for later replay or debugging.  This module
provides the low-level helpers that prepare and compress the dump
payload before handing it to :class:`PersistenceManager` for storage.

All public functions are fire-and-forget safe — they catch and log
exceptions internally so callers never need ``try/except`` wrappers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
import zlib
from copy import deepcopy
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .persistence import PersistenceManager

logger = logging.getLogger("llm-rosetta.observability")

# Matches inline base64 image data URIs in string values.
# Captures: full match for replacement, group(1) = media type, group(2) = base64 data.
_BASE64_IMAGE_RE = re.compile(
    r"data:(image/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=\s]{100,})"
)

# Bodies larger than this are skipped entirely (only metadata is stored).
MAX_BODY_BYTES = 10 * 1024 * 1024  # 10 MB


# ------------------------------------------------------------------
# Image offload
# ------------------------------------------------------------------


def offload_images(body: dict[str, Any]) -> dict[str, Any]:
    """Replace inline base64 image data with SHA256 digest references.

    Returns a deep copy of *body* with every ``data:image/…;base64,…``
    string replaced by a human-readable placeholder like
    ``[image sha256:abc123… 450KB]``.  The original body is not mutated.
    """
    return _walk_and_replace(deepcopy(body))


def _walk_and_replace(obj: Any) -> Any:
    """Recursively walk a JSON-like structure and replace base64 images."""
    if isinstance(obj, str):
        return _replace_images_in_string(obj)
    if isinstance(obj, dict):
        return {k: _walk_and_replace(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_replace(item) for item in obj]
    return obj


def _replace_images_in_string(s: str) -> str:
    """Replace all base64 image data URIs in a single string."""

    def _replacement(m: re.Match[str]) -> str:
        media_type = m.group(1)
        b64_data = m.group(2).replace("\n", "").replace("\r", "").replace(" ", "")
        raw_bytes = len(b64_data) * 3 // 4  # approximate decoded size
        digest = hashlib.sha256(b64_data.encode("ascii")).hexdigest()[:16]
        size_label = _human_size(raw_bytes)
        return f"[image {media_type} sha256:{digest} {size_label}]"

    return _BASE64_IMAGE_RE.sub(_replacement, s)


def _human_size(n: int) -> str:
    """Format byte count as a human-readable string."""
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.0f}KB"
    return f"{n / (1024 * 1024):.1f}MB"


# ------------------------------------------------------------------
# Hashing & compression
# ------------------------------------------------------------------


def compute_body_hash(body: dict[str, Any]) -> str:
    """SHA256 of canonicalized JSON (sorted keys, no whitespace)."""
    canonical = json.dumps(
        body, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compress_body(body: dict[str, Any]) -> tuple[bytes, int]:
    """JSON-serialize and zlib-compress a body dict.

    Returns:
        ``(compressed_bytes, original_size)`` tuple.
    """
    raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
    compressed = zlib.compress(raw, level=6)
    return compressed, len(raw)


def decompress_body(data: bytes) -> dict[str, Any]:
    """Decompress a zlib-compressed JSON body back to a dict."""
    raw = zlib.decompress(data)
    return json.loads(raw.decode("utf-8"))


# ------------------------------------------------------------------
# High-level dump function
# ------------------------------------------------------------------


def dump_error(
    persistence: PersistenceManager | None,
    *,
    request_body: dict[str, Any] | None,
    response_text: str | None = None,
    converted_body: dict[str, Any] | None = None,
    model: str | None = None,
    source_provider: str | None = None,
    target_provider: str | None = None,
    provider_name: str | None = None,
    status_code: int | None = None,
    error_phase: str | None = None,
    upstream_url: str | None = None,
    request_log_id: str | None = None,
) -> str | None:
    """Offload images, hash, compress, and store an error dump.

    This is the primary entry point for recording error context.  It is
    fire-and-forget safe: any exception during processing is caught and
    logged, and ``None`` is returned.

    Args:
        persistence: The persistence manager (may be ``None`` in tests or
            when persistence is disabled — in that case this is a no-op).
        request_body: The original request body dict.
        response_text: Upstream error response text (usually small).
        converted_body: The converted target-format body, if available.
        model: Model name from the request.
        source_provider: Source API format (e.g. ``"openai_chat"``).
        target_provider: Target API format (e.g. ``"anthropic"``).
        provider_name: Human-readable provider name.
        status_code: Upstream HTTP status code.
        error_phase: One of ``"upstream"``, ``"stream_header"``,
            ``"stream_chunk"``, ``"conversion"``.
        upstream_url: The upstream URL that was called.
        request_log_id: FK to the request log entry, if available.

    Returns:
        The dump ID on success, or ``None`` on failure / no-op.
    """
    if persistence is None:
        return None

    try:
        return _dump_error_impl(
            persistence,
            request_body=request_body,
            response_text=response_text,
            converted_body=converted_body,
            model=model,
            source_provider=source_provider,
            target_provider=target_provider,
            provider_name=provider_name,
            status_code=status_code,
            error_phase=error_phase,
            upstream_url=upstream_url,
            request_log_id=request_log_id,
        )
    except Exception:
        logger.debug("Failed to dump error", exc_info=True)
        return None


def _dump_error_impl(
    persistence: PersistenceManager,
    *,
    request_body: dict[str, Any] | None,
    response_text: str | None,
    converted_body: dict[str, Any] | None,
    model: str | None,
    source_provider: str | None,
    target_provider: str | None,
    provider_name: str | None,
    status_code: int | None,
    error_phase: str | None,
    upstream_url: str | None,
    request_log_id: str | None,
) -> str | None:
    """Inner implementation — may raise."""
    dump_id = uuid.uuid4().hex
    timestamp = datetime.now(timezone.utc).isoformat()

    # --- Process request body ---
    body_hash: str | None = None
    if request_body is not None:
        raw_json = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
        if len(raw_json) <= MAX_BODY_BYTES:
            offloaded = offload_images(request_body)
            body_hash = compute_body_hash(offloaded)
            compressed, orig_bytes = compress_body(offloaded)
            persistence.insert_dump_body(body_hash, compressed, orig_bytes)
        else:
            logger.debug(
                "Skipping body dump: %d bytes exceeds %d limit",
                len(raw_json),
                MAX_BODY_BYTES,
            )

    # --- Process converted body ---
    converted_body_hash: str | None = None
    if converted_body is not None:
        raw_json = json.dumps(converted_body, ensure_ascii=False).encode("utf-8")
        if len(raw_json) <= MAX_BODY_BYTES:
            offloaded_conv = offload_images(converted_body)
            converted_body_hash = compute_body_hash(offloaded_conv)
            compressed_conv, orig_conv = compress_body(offloaded_conv)
            persistence.insert_dump_body(
                converted_body_hash, compressed_conv, orig_conv
            )

    # --- Truncate response_text if excessively large ---
    if response_text and len(response_text) > 64 * 1024:
        response_text = response_text[: 64 * 1024] + "\n…[truncated]"

    # --- Insert the error dump record ---
    persistence.insert_error_dump(
        dump_id=dump_id,
        request_log_id=request_log_id,
        timestamp=timestamp,
        model=model,
        source_provider=source_provider,
        target_provider=target_provider,
        provider_name=provider_name,
        status_code=status_code,
        error_phase=error_phase,
        body_hash=body_hash,
        response_text=response_text,
        upstream_url=upstream_url,
        converted_body_hash=converted_body_hash,
    )

    logger.debug(
        "Dumped error %s (phase=%s, status=%s)", dump_id, error_phase, status_code
    )
    return dump_id
