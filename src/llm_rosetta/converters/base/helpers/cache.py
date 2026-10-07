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
import pickle
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


def _content_hash_bytes(obj: Any) -> bytes:
    """Serialize *obj* to bytes for cache key computation.

    Uses ``pickle.dumps(protocol=5)`` which is ~4-5x faster than
    ``json.dumps(sort_keys=True)`` for nested dicts.  Key-order
    dependent, but all IR converters produce identical key order
    (verified empirically).  Worst case on key-order divergence:
    cache miss → re-validate (safe, not a false positive).
    """
    return pickle.dumps(obj, protocol=5)


def _canonical_json_bytes(obj: Any) -> bytes:
    """Serialize *obj* to deterministic JSON bytes (slow path).

    Only used for mutation-detection fingerprints in :class:`LRUCache`
    where key-order independence across sessions matters.
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
# Generational set — lightweight eviction for boolean caches
# ---------------------------------------------------------------------------


class GenerationalSet:
    """Two-generation set with time-based rotation for boolean caches.

    Stores only keys (no values) — designed for caches that record
    ``True``/``False`` status (e.g. "has this IR entry been validated?").
    Lookup checks both current and previous generations (two O(1) set
    lookups).  Rotation drops the oldest generation without per-entry
    bookkeeping.

    Provides the same public interface as :class:`LRUCache` (``get``,
    ``put``, ``clear``, ``info``, ``check_integrity``) so it can be
    used as a drop-in replacement for boolean-only caches.

    Args:
        rotate_interval: Seconds between generation rotations.
            Entries from sessions ended more than ``rotate_interval``
            seconds ago naturally age out.
    """

    __slots__ = (
        "_current",
        "_previous",
        "_rotate_interval",
        "_last_rotate",
        "_hits",
        "_misses",
        "_rotations",
    )

    def __init__(self, rotate_interval: float = DEFAULT_TTL) -> None:
        self._current: set[int] = set()
        self._previous: set[int] = set()
        self._rotate_interval = rotate_interval
        self._last_rotate = time.monotonic()
        self._hits = 0
        self._misses = 0
        self._rotations = 0

    def _maybe_rotate(self) -> None:
        now = time.monotonic()
        if now - self._last_rotate >= self._rotate_interval:
            self._previous = self._current
            self._current = set()
            self._last_rotate = now
            self._rotations += 1

    def get(self, key: int) -> Any:
        """Return ``True`` on hit, :data:`_SENTINEL` on miss."""
        self._maybe_rotate()
        if key in self._current or key in self._previous:
            self._current.add(key)
            self._hits += 1
            return True
        self._misses += 1
        return _SENTINEL

    def put(self, key: int, value: Any) -> None:
        """Record *key* in the current generation (value is ignored)."""
        self._maybe_rotate()
        self._current.add(key)

    def clear(self) -> None:
        """Remove all entries and reset counters."""
        self._current.clear()
        self._previous.clear()
        self._last_rotate = time.monotonic()
        self._hits = 0
        self._misses = 0
        self._rotations = 0

    def check_integrity(self) -> list[int]:
        """Always returns empty — boolean values cannot be mutated."""
        return []

    def info(self) -> dict[str, Any]:
        """Return cache statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "expirations": 0,
            "corruptions": 0,
            "currsize": len(self._current) + len(self._previous),
            "maxsize": None,
            "ttl": self._rotate_interval,
            "verify": False,
            "rotations": self._rotations,
        }


# ---------------------------------------------------------------------------
# Epoch guard — list-level change detection for fast skip
# ---------------------------------------------------------------------------


def _sig_text(entry: dict[str, Any]) -> int:
    text = entry.get("text", "")
    return hash(("text", len(text), text[:16], text[-16:]))


def _sig_tool_call(entry: dict[str, Any]) -> int:
    inp = entry.get("input", {})
    first_key = next(iter(inp), "") if isinstance(inp, dict) else ""
    return hash(
        (
            "tool_call",
            entry.get("tool_call_id", ""),
            entry.get("tool_name", ""),
            len(inp) if isinstance(inp, (dict, str)) else 0,
            first_key,
        )
    )


def _sig_tool_result(entry: dict[str, Any]) -> int:
    result = entry.get("result")
    if isinstance(result, str):
        rsig = hash(("s", len(result), result[:16], result[-16:]))
    elif isinstance(result, list):
        rsig = hash(("l", len(result)))
    else:
        rsig = hash(type(result))
    return hash(("tool_result", entry.get("tool_call_id", ""), rsig))


def _sig_reasoning(entry: dict[str, Any]) -> int:
    r = entry.get("reasoning", "")
    s = entry.get("signature", "")
    return hash(
        (
            "reasoning",
            r[:16] if r else "",
            r[-16:] if r else "",
            s[:16] if s else "",
            s[-16:] if s else "",
        )
    )


def _sig_refusal(entry: dict[str, Any]) -> int:
    r = entry.get("refusal", "")
    return hash(("refusal", r[:16], r[-16:]))


def _sig_media(entry: dict[str, Any]) -> int:
    t = entry["type"]
    url = entry.get(f"{t}_url") or entry.get("url", "")
    if url:
        return hash((t, url[:16], url[-16:]))
    data = entry.get("data", "")
    return hash((t, len(data) if data else 0, entry.get("media_type", "")))


def _sig_citation(entry: dict[str, Any]) -> int:
    url = entry.get("url", "")
    if url:
        return hash(("citation", url[:64]))
    return hash(("citation", (entry.get("cited_text", "") or "")[:32]))


_SIGNAL_DISPATCH: dict[str, Any] = {
    "text": _sig_text,
    "tool_call": _sig_tool_call,
    "tool_result": _sig_tool_result,
    "reasoning": _sig_reasoning,
    "refusal": _sig_refusal,
    "image": _sig_media,
    "file": _sig_media,
    "audio": _sig_media,
    "citation": _sig_citation,
}


def _entry_signal(entry: Any) -> int:
    """Compute a lightweight per-entry signal for epoch detection.

    O(1) per entry — field access only, no serialization.  Covers all
    9 IR content part types plus ToolDefinition.
    """
    if not isinstance(entry, dict):
        return hash(type(entry))

    t = entry.get("type", "")
    handler = _SIGNAL_DISPATCH.get(t)
    if handler is not None:
        return handler(entry)

    # ToolDefinition (no "type" field — keyed by "name")
    name = entry.get("name")
    if name is not None:
        desc = entry.get("description", "")
        params = entry.get("parameters", {})
        n_props = len(params.get("properties", {})) if isinstance(params, dict) else 0
        n_req = len(params.get("required", [])) if isinstance(params, dict) else 0
        return hash(("tool_def", name, len(desc) if desc else 0, n_props, n_req))

    # Message envelope (role + content)
    role = entry.get("role")
    if role is not None:
        content = entry.get("content", [])
        if isinstance(content, list):
            parts_sig = 0
            for part in content:
                parts_sig ^= _entry_signal(part)
            return hash((role, len(content), parts_sig))
        return hash((role, hash(content)))

    return hash(pickle.dumps(entry, protocol=5))


def _list_epoch(entries: list[Any], tag: str) -> int:
    """Compute a list-level epoch signal.

    Combines list length, two pickle samples (first + last), and
    an XOR aggregate of per-entry lightweight signals.  Cost is
    O(n) but O(1) per entry (no serialization except two samples).
    """
    n = len(entries)
    if n == 0:
        return hash((tag, 0))

    agg = 0
    for entry in entries:
        agg ^= _entry_signal(entry)

    first_sample = pickle.dumps(entries[0], protocol=5) if n > 0 else b""
    last_sample = pickle.dumps(entries[-1], protocol=5) if n > 1 else first_sample

    return hash((tag, n, agg, first_sample, last_sample))


# Epoch cache: (field_name, tag) → last epoch value
_epoch_cache: dict[tuple[str, str], int] = {}

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

ir_validation_cache = GenerationalSet()
"""Unified IR validation status cache (the **hub**).

Stores ``True`` for any IR entry (tool, message, etc.) that has passed
TypedDict validation.  Keyed by ``(type_tag, canonical_json)`` — the
tag prevents cross-type hash collisions while allowing cross-converter
sharing (IR is converter-agnostic).

This is the **hub** in the hub-and-spoke architecture:
- Spokes: ``tool_entry_cache`` (converter-specific conversion results)
- Hub: ``ir_validation_cache`` (converter-agnostic validation status)

A spoke hit implies a hub hit (entry was validated on first conversion).
A hub hit does NOT imply a spoke hit (different converter may not have
converted this entry yet).
"""


def clear_all_caches() -> None:
    """Clear all conversion caches.  Used in test fixtures."""
    tool_entry_cache.clear()
    sanitize_cache.clear()
    ir_validation_cache.clear()
    _epoch_cache.clear()


def cache_info() -> dict[str, dict[str, Any]]:
    """Return statistics for all caches (for diagnostics)."""
    return {
        "tool_entry": tool_entry_cache.info(),
        "sanitize": sanitize_cache.info(),
        "ir_validation": ir_validation_cache.info(),
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


def _ir_validation_key(tag: str, entry: Any) -> int:
    """Compute a validation cache key for an IR entry.

    The *tag* namespaces by IR type (e.g. ``"ir.tool"``, ``"ir.message"``)
    so different IR types with the same content never collide.  No
    converter tag — IR validation is converter-agnostic.
    """
    return hash(tag.encode() + b"\x00" + _content_hash_bytes(entry))


def is_ir_validated(tag: str, entry: Any) -> bool:
    """Check if an IR entry was previously validated.

    Uses a content-hash lookup in the ``ir_validation_cache`` LRU.
    The tag prevents cross-type collisions (e.g. ``"ir.tool"`` vs
    ``"ir.message"``).

    Args:
        tag: IR type tag (e.g. ``"ir.tool"``, ``"ir.message"``).
        entry: A single IR entry dict.

    Returns:
        True if this entry has passed validation before.
    """
    return ir_validation_cache.get(_ir_validation_key(tag, entry)) is not _SENTINEL


def mark_ir_validated(tag: str, entry: Any) -> None:
    """Record an IR entry as having passed validation.

    Args:
        tag: IR type tag (e.g. ``"ir.tool"``, ``"ir.message"``).
        entry: A single IR entry dict.
    """
    ir_validation_cache.put(_ir_validation_key(tag, entry), True)
