"""Shared SSE transport for REST integration tests."""

from __future__ import annotations

import json
import time
from typing import Any, Iterator

import requests


def _iter_sse_json(response: requests.Response) -> Iterator[dict[str, Any]]:
    """Yield JSON payloads from ``data: `` lines until stream end or [DONE]."""
    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8")
        if not decoded.startswith("data: "):
            continue
        data_str = decoded[6:]
        if data_str.strip() == "[DONE]":
            return
        try:
            yield json.loads(data_str)
        except json.JSONDecodeError:
            continue


def stream_sse_events(
    url: str,
    headers: dict[str, str],
    provider_req: dict,
    *,
    max_retries: int = 3,
    timeout: int = 60,
) -> Iterator[dict[str, Any]]:
    """POST ``provider_req`` and yield parsed SSE JSON events.

    Retries on HTTP 429 with exponential backoff (``2 ** (attempt + 1)``
    seconds).  After ``max_retries`` retryable failures, a final attempt
    is made whose ``raise_for_status`` error is propagated to the caller.
    """
    for attempt in range(max_retries):
        response = requests.post(
            url, headers=headers, json=provider_req, timeout=timeout, stream=True
        )
        if response.status_code == 429:
            wait = 2 ** (attempt + 1)
            print(f"  [Rate limited, retrying in {wait}s...]")
            time.sleep(wait)
            continue
        response.raise_for_status()
        yield from _iter_sse_json(response)
        return

    # All retries exhausted on 429 — do one final unguarded attempt so
    # the caller sees the eventual raise_for_status error.
    response = requests.post(
        url, headers=headers, json=provider_req, timeout=timeout, stream=True
    )
    response.raise_for_status()
    yield from _iter_sse_json(response)
