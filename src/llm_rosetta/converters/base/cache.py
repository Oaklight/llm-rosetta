"""Process-level LRU caching for tool conversion and schema sanitization.

Eliminates repeated IR validation and schema sanitization for unchanged
tool definitions across conversation turns.  All caches are module-level
singletons (converters are recreated per request, so instance-level
caching would be useless).

Thread safety: not needed — the gateway runs a single-threaded async
event loop.

Mutation safety: cached values are returned **without deep copy**.
The conversion pipeline is read-only after each stage produces its
output.  If a future change introduces mutation of cached tool dicts,
tests will fail due to cross-test pollution (the ``clear_all_caches``
conftest fixture catches this).
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from typing import Any

_SENTINEL = object()
"""Cache miss sentinel — distinct from any valid cached value."""

# Default TTL: 30 minutes.  Long enough to cover most agent sessions
# without a miss; short enough that idle entries don't linger for days.
# The miss penalty is ~2ms, so even aggressive TTL is harmless.
DEFAULT_TTL: float = 1800.0


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------


def _canonical_json_bytes(obj: Any) -> bytes:
    """Serialize *obj* to deterministic JSON bytes.

    Uses ``sort_keys=True`` so dict key insertion order does not affect
    the output, and compact separators to minimise byte length.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def tools_cache_key(converter_tag: str, tools: list[Any]) -> int:
    """Compute a cache key for a tool definition list.

    The key incorporates the *converter_tag* (so different converters
    never collide) and the canonical JSON of each tool.  Uses Python's
    built-in ``hash()`` on bytes — 64-bit SipHash, collision probability
    ~10⁻¹⁵ at n=128 entries, more than sufficient for a bounded LRU.

    Args:
        converter_tag: Converter identifier (e.g. ``"anthropic"``).
        tools: List of provider or IR tool definition dicts.

    Returns:
        Integer hash suitable as an LRU cache key.
    """
    # Build a single bytes blob: tag + each tool's canonical JSON.
    parts: list[bytes] = [converter_tag.encode()]
    for t in tools:
        parts.append(_canonical_json_bytes(t))
    return hash(b"\x00".join(parts))


def schema_cache_key(
    schema: dict[str, Any],
    extra_strip_keys: frozenset[str] | None = None,
) -> int:
    """Compute a cache key for a single JSON Schema dict.

    Args:
        schema: The JSON Schema to hash.
        extra_strip_keys: Additional provider-specific keys to strip
            (e.g. Google's ``{"additionalProperties"}``).

    Returns:
        Integer hash suitable as an LRU cache key.
    """
    blob = _canonical_json_bytes(schema)
    if extra_strip_keys:
        blob += b"\x00" + ",".join(sorted(extra_strip_keys)).encode()
    return hash(blob)


# ---------------------------------------------------------------------------
# LRU cache with TTL
# ---------------------------------------------------------------------------


class LRUCache:
    """Bounded LRU cache with per-entry TTL.

    Each entry expires *ttl* seconds after it was last **written**
    (``put``).  Reads (``get``) do **not** extend the deadline — this
    keeps the semantics simple and predictable.  Expired entries are
    evicted lazily on ``get`` (treated as a miss).

    Not thread-safe (single-threaded async event loop assumed).

    Args:
        maxsize: Maximum number of entries before LRU eviction.
        ttl: Time-to-live in seconds for each entry.  ``None`` disables
            expiry (entries live until LRU-evicted or cleared).
    """

    __slots__ = ("_cache", "_maxsize", "_ttl", "_hits", "_misses", "_expirations")

    def __init__(
        self,
        maxsize: int = 16,
        ttl: float | None = DEFAULT_TTL,
    ) -> None:
        self._cache: OrderedDict[int, tuple[Any, float]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl
        self._hits = 0
        self._misses = 0
        self._expirations = 0

    def get(self, key: int) -> Any:
        """Return cached value, or :data:`_SENTINEL` on miss.

        Expired entries are evicted and counted as misses.
        On hit the entry is moved to the end (most-recently-used).
        """
        try:
            value, deadline = self._cache[key]
        except KeyError:
            self._misses += 1
            return _SENTINEL

        if self._ttl is not None and time.monotonic() >= deadline:
            # Entry has expired — evict it.
            del self._cache[key]
            self._expirations += 1
            self._misses += 1
            return _SENTINEL

        self._cache.move_to_end(key)
        self._hits += 1
        return value

    def put(self, key: int, value: Any) -> None:
        """Store *value* under *key*, evicting the LRU entry if full.

        The TTL deadline is set (or reset) on every ``put``.
        """
        deadline = (time.monotonic() + self._ttl) if self._ttl is not None else 0.0
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = (value, deadline)
            return
        if len(self._cache) >= self._maxsize:
            self._cache.popitem(last=False)  # evict oldest
        self._cache[key] = (value, deadline)

    def clear(self) -> None:
        """Remove all entries and reset counters."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        self._expirations = 0

    def info(self) -> dict[str, Any]:
        """Return cache statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "expirations": self._expirations,
            "currsize": len(self._cache),
            "maxsize": self._maxsize,
            "ttl": self._ttl,
        }


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

tools_from_p_cache = LRUCache(maxsize=16)
"""Provider→IR tool list cache.  Keyed by (converter_tag, tools_hash)."""

tools_to_p_cache = LRUCache(maxsize=16)
"""IR→Provider tool list cache.  Keyed by (converter_tag, ir_tools_hash)."""

sanitize_cache = LRUCache(maxsize=128)
"""Individual JSON Schema sanitization cache."""


def clear_all_caches() -> None:
    """Clear all tool conversion caches.  Used in test fixtures."""
    tools_from_p_cache.clear()
    tools_to_p_cache.clear()
    sanitize_cache.clear()


def cache_info() -> dict[str, dict[str, Any]]:
    """Return statistics for all caches (for diagnostics)."""
    return {
        "tools_from_p": tools_from_p_cache.info(),
        "tools_to_p": tools_to_p_cache.info(),
        "sanitize": sanitize_cache.info(),
    }
