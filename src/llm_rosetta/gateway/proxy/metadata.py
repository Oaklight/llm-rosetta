"""The provider metadata cache and internal-metadata stripping."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


from llm_rosetta.converters.base.helpers import strip_internal_metadata


from ..logging import (
    get_logger,
)

logger = get_logger()


@dataclass
class _CacheEntry:
    """A single cached provider_metadata entry with creation timestamp."""

    data: dict[str, Any]
    created: float = field(default_factory=time.monotonic)


class ProviderMetadataStore:
    """Stores provider_metadata across request boundaries with TTL and bounds.

    Args:
        ttl: Time-to-live in seconds for each entry.  Defaults to 30 minutes.
        max_size: Maximum number of entries.  Oldest is evicted on overflow.
    """

    def __init__(self, *, ttl: float = 1800.0, max_size: int = 10_000) -> None:
        self._store: dict[str, _CacheEntry] = {}
        self._ttl = ttl
        self._max_size = max_size

    def _evict_expired(self) -> None:
        now = time.monotonic()
        # Entries are inserted in monotonic time order and never refreshed,
        # so iteration order == creation order.  Stop at the first
        # non-expired entry instead of scanning the entire store.
        expired = []
        for k, e in self._store.items():
            if now - e.created > self._ttl:
                expired.append(k)
            else:
                break
        for k in expired:
            del self._store[k]

    def _evict_oldest(self) -> None:
        if len(self._store) >= self._max_size:
            # dict preserves insertion order (Python 3.7+) and entries are
            # never refreshed, so the first key is always the oldest.
            oldest_key = next(iter(self._store))
            del self._store[oldest_key]

    def cache_from_response(self, ir_response: dict[str, Any]) -> None:
        """Extract and cache provider_metadata from tool calls in an IR response."""
        self._evict_expired()
        for choice in ir_response.get("choices", []):
            msg = choice.get("message", {})
            for part in msg.get("content", []):
                if part.get("type") == "tool_call" and "provider_metadata" in part:
                    tool_call_id = part.get("tool_call_id")
                    if tool_call_id:
                        self._evict_oldest()
                        self._store[tool_call_id] = _CacheEntry(
                            data=part["provider_metadata"],
                        )
                        logger.debug(
                            "Cached provider_metadata for tool_call %s", tool_call_id
                        )

    def cache_from_stream_event(self, ir_event: dict[str, Any]) -> None:
        """Cache provider_metadata from a tool_call_start stream event."""
        if (
            ir_event.get("type") == "tool_call_start"
            and "provider_metadata" in ir_event
        ):
            self._evict_expired()
            self._evict_oldest()
            self._store[ir_event["tool_call_id"]] = _CacheEntry(
                data=ir_event["provider_metadata"],
            )

    def inject_into_request(self, ir_request: dict[str, Any]) -> None:
        """Inject cached provider_metadata into tool call parts in an IR request.

        Clients send the full conversation history on every request, so the
        same tool_call_id may appear in multiple requests.  Entries are kept
        alive (not popped) for subsequent turns.
        """
        self._evict_expired()
        logger.debug(
            "inject: store has %d entries: %s",
            len(self._store),
            list(self._store.keys()),
        )
        for msg in ir_request.get("messages", []):
            for part in msg.get("content", []):
                if part.get("type") == "tool_call":
                    tool_call_id = part.get("tool_call_id")
                    if tool_call_id and tool_call_id in self._store:
                        part["provider_metadata"] = self._store[tool_call_id].data

    def clear(self) -> None:
        """Remove all entries."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


_default_metadata_store = ProviderMetadataStore()


def _strip_internal_metadata(obj: dict | list) -> None:
    """Remove ``_llm_rosetta_provider_metadata`` from an outbound body, in place.

    Thin wrapper over
    :func:`llm_rosetta.converters.base.helpers.strip_internal_metadata`, kept
    as a defensive guard at the HTTP boundary: the converters already strip the
    carrier from their ``*_to_provider`` output (issue #941), so this is a
    belt-and-braces pass over bodies that may have been post-processed.
    """
    strip_internal_metadata(obj)
