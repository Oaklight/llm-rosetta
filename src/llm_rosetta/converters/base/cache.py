"""Process-level LRU caching for tool conversion, schema sanitization,
and message validation.

Uses **per-entry** caching: each tool definition, schema, and message is
cached individually by content hash.  This enables:

- **Partial hit**: 30/31 tools unchanged → 30 hits + 1 miss
- **Cross-agent sharing**: two agents sharing 25 tools → 25 shared entries
- **Sliding context window**: old messages TTL-expire, new ones added,
  overlapping ones hit cache

All caches are module-level singletons (converters are recreated per
request, so instance-level caching would be useless).

Thread safety: not needed — the gateway runs a single-threaded async
event loop.

Mutation safety: cached values are returned **without deep copy**.
The conversion pipeline is read-only after each stage produces its
output.  Use :func:`check_integrity` (called by the test conftest on
teardown) to catch code bugs that accidentally mutate cached objects.
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


def entry_cache_key(tag: str, entry: Any) -> int:
    """Compute a cache key for a single entry (tool, message, schema, etc.).

    The key incorporates *tag* (so different converters / directions
    never collide) and the canonical JSON of *entry*.  Uses Python's
    built-in ``hash()`` on bytes — 64-bit SipHash, collision probability
    ~10⁻¹⁵ at n=512 entries, more than sufficient for a bounded LRU.

    Args:
        tag: Namespace string (e.g. ``"anthropic:from_p"``).
        entry: The dict/object to hash.

    Returns:
        Integer hash suitable as an LRU cache key.
    """
    return hash(tag.encode() + b"\x00" + _canonical_json_bytes(entry))


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

    Each entry expires *ttl* seconds after it was last **accessed**
    (read or written).  Expired entries are evicted lazily on ``get``
    (treated as a miss).

    **Mutation contract**: cached values are returned by reference.
    Callers **must not** mutate the returned objects — doing so silently
    corrupts the cache for all subsequent requests.  Use
    :meth:`check_integrity` (called by the test conftest on teardown) to
    detect violations.

    Not thread-safe (single-threaded async event loop assumed).

    Args:
        maxsize: Maximum number of entries before LRU eviction.
        ttl: Time-to-live in seconds for each entry.  ``None`` disables
            expiry (entries live until LRU-evicted or cleared).
        verify: When ``True``, ``get()`` re-hashes the cached value on
            every hit and evicts it if mutated (self-healing but ~265µs
            overhead per hit).  Default ``False`` — use
            :meth:`check_integrity` in tests instead.
    """

    __slots__ = (
        "_cache",
        "_fingerprints",
        "_maxsize",
        "_ttl",
        "_verify",
        "_hits",
        "_misses",
        "_expirations",
        "_corruptions",
    )

    def __init__(
        self,
        maxsize: int = 16,
        ttl: float | None = DEFAULT_TTL,
        verify: bool = False,
    ) -> None:
        # value storage: key → (value, deadline)
        self._cache: OrderedDict[int, tuple[Any, float]] = OrderedDict()
        # mutation detection: key → content hash at put() time
        self._fingerprints: dict[int, int] = {}
        self._maxsize = maxsize
        self._ttl = ttl
        self._verify = verify
        self._hits = 0
        self._misses = 0
        self._expirations = 0
        self._corruptions = 0

    def get(self, key: int) -> Any:
        """Return cached value, or :data:`_SENTINEL` on miss.

        Checks key existence and TTL expiry.  On hit the entry is moved
        to the end (most-recently-used) and its TTL deadline is
        refreshed — so an actively-used entry never expires mid-session.

        When *verify* mode is enabled, also re-hashes the value to
        detect in-place mutation — corrupted entries are evicted and
        treated as misses.
        """
        try:
            value, deadline = self._cache[key]
        except KeyError:
            self._misses += 1
            return _SENTINEL

        if self._ttl is not None and time.monotonic() >= deadline:
            del self._cache[key]
            self._fingerprints.pop(key, None)
            self._expirations += 1
            self._misses += 1
            return _SENTINEL

        # Optional mutation guard (off by default, enable for debugging).
        if self._verify:
            current_fp = hash(_canonical_json_bytes(value))
            if current_fp != self._fingerprints.get(key):
                del self._cache[key]
                self._fingerprints.pop(key, None)
                self._corruptions += 1
                self._misses += 1
                return _SENTINEL

        # Refresh TTL on access so active sessions don't see spurious expiry.
        if self._ttl is not None:
            self._cache[key] = (value, time.monotonic() + self._ttl)
        self._cache.move_to_end(key)
        self._hits += 1
        return value

    def put(self, key: int, value: Any) -> None:
        """Store *value* under *key*, evicting the LRU entry if full.

        The TTL deadline is set (or reset) on every ``put``.
        A content fingerprint is recorded for :meth:`check_integrity`.
        """
        deadline = (time.monotonic() + self._ttl) if self._ttl is not None else 0.0
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = (value, deadline)
            self._fingerprints[key] = hash(_canonical_json_bytes(value))
            return
        if len(self._cache) >= self._maxsize:
            evicted_key, _ = self._cache.popitem(last=False)  # evict oldest
            self._fingerprints.pop(evicted_key, None)
        self._cache[key] = (value, deadline)
        self._fingerprints[key] = hash(_canonical_json_bytes(value))

    def clear(self) -> None:
        """Remove all entries and reset counters."""
        self._cache.clear()
        self._fingerprints.clear()
        self._hits = 0
        self._misses = 0
        self._expirations = 0
        self._corruptions = 0

    def check_integrity(self) -> list[int]:
        """Verify that no cached value has been mutated since ``put()``.

        Re-hashes every live entry and compares against the fingerprint
        recorded at insertion time.  Returns a list of keys whose values
        have changed — an empty list means the cache is clean.

        **Not called in the hot path.**  Designed for the test conftest
        teardown to catch code bugs that accidentally mutate cached
        objects.  Production ``get()`` does not check — mutations are a
        code bug (users never touch Python objects), not a runtime event.
        """
        corrupted: list[int] = []
        for key, (value, _deadline) in self._cache.items():
            current = hash(_canonical_json_bytes(value))
            if current != self._fingerprints.get(key):
                corrupted.append(key)
        return corrupted

    def info(self) -> dict[str, Any]:
        """Return cache statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "expirations": self._expirations,
            "corruptions": self._corruptions,
            "currsize": len(self._cache),
            "maxsize": self._maxsize,
            "ttl": self._ttl,
            "verify": self._verify,
        }


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

tool_entry_cache = LRUCache(maxsize=512)
"""Per-entry tool conversion cache.

Keyed by ``(converter_tag:direction, single_tool_json)``.
Stores individual tool conversion results (provider→IR or IR→provider).
At ~3KB per tool, 512 entries ≈ 1.5MB max.
"""

sanitize_cache = LRUCache(maxsize=512)
"""Per-schema sanitization cache.

Keyed by ``(schema_json, extra_strip_keys)``.
"""

validated_msg_cache = LRUCache(maxsize=4096)
"""Per-message validation status cache.

Stores ``True`` for messages that have passed IR validation.
At ~170 bytes per message hash, 4096 entries ≈ negligible memory
(only the key + True + deadline are stored, not the message content).
Covers ~18 concurrent 218-message conversations.
"""


def clear_all_caches() -> None:
    """Clear all conversion caches.  Used in test fixtures."""
    tool_entry_cache.clear()
    sanitize_cache.clear()
    validated_msg_cache.clear()


def cache_info() -> dict[str, dict[str, Any]]:
    """Return statistics for all caches (for diagnostics)."""
    return {
        "tool_entry": tool_entry_cache.info(),
        "sanitize": sanitize_cache.info(),
        "validated_msg": validated_msg_cache.info(),
    }


# ---------------------------------------------------------------------------
# Per-entry helper functions
# ---------------------------------------------------------------------------


def get_cached_tool(tag: str, tool: dict[str, Any]) -> Any:
    """Look up a single tool conversion result.

    Args:
        tag: Namespace (e.g. ``"anthropic:from_p"``).
        tool: Single tool definition dict.

    Returns:
        Cached conversion result, or :data:`_SENTINEL` on miss.
    """
    return tool_entry_cache.get(entry_cache_key(tag, tool))


def put_cached_tool(tag: str, tool: dict[str, Any], result: Any) -> None:
    """Cache a single tool conversion result.

    Args:
        tag: Namespace (e.g. ``"anthropic:from_p"``).
        tool: Single tool definition dict (used to compute key).
        result: The conversion result to cache.
    """
    tool_entry_cache.put(entry_cache_key(tag, tool), result)


def is_message_validated(msg: dict[str, Any]) -> bool:
    """Check if a message was previously validated.

    Args:
        msg: A single IR message dict.

    Returns:
        True if the message content hash is in the validation cache.
    """
    key = hash(_canonical_json_bytes(msg))
    return validated_msg_cache.get(key) is not _SENTINEL


def mark_message_validated(msg: dict[str, Any]) -> None:
    """Record a message as having passed IR validation.

    Args:
        msg: A single IR message dict.
    """
    key = hash(_canonical_json_bytes(msg))
    validated_msg_cache.put(key, True)
