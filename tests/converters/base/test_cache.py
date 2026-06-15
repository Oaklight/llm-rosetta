"""Unit tests for the process-level LRU cache infrastructure."""

from unittest.mock import patch

from llm_rosetta.converters.base.cache import (
    DEFAULT_TTL,
    LRUCache,
    _SENTINEL,
    _canonical_json_bytes,
    cache_info,
    clear_all_caches,
    schema_cache_key,
    tools_cache_key,
)


# ---------------------------------------------------------------------------
# _canonical_json_bytes
# ---------------------------------------------------------------------------


class TestCanonicalJsonBytes:
    def test_sort_keys(self):
        """Dict key order should not affect output."""
        a = _canonical_json_bytes({"b": 2, "a": 1})
        b = _canonical_json_bytes({"a": 1, "b": 2})
        assert a == b

    def test_compact_separators(self):
        result = _canonical_json_bytes({"key": "value"})
        assert b" " not in result  # no whitespace


# ---------------------------------------------------------------------------
# tools_cache_key
# ---------------------------------------------------------------------------


class TestToolsCacheKey:
    def test_deterministic(self):
        tools = [{"name": "foo", "type": "function"}]
        k1 = tools_cache_key("test", tools)
        k2 = tools_cache_key("test", tools)
        assert k1 == k2

    def test_varies_by_tag(self):
        tools = [{"name": "foo", "type": "function"}]
        k1 = tools_cache_key("anthropic", tools)
        k2 = tools_cache_key("openai_chat", tools)
        assert k1 != k2

    def test_varies_by_content(self):
        tools_a = [{"name": "foo", "type": "function"}]
        tools_b = [{"name": "bar", "type": "function"}]
        assert tools_cache_key("t", tools_a) != tools_cache_key("t", tools_b)

    def test_order_independent_within_dict(self):
        """Same dict content with different key order → same key."""
        tools_a = [{"type": "function", "name": "foo"}]
        tools_b = [{"name": "foo", "type": "function"}]
        assert tools_cache_key("t", tools_a) == tools_cache_key("t", tools_b)

    def test_list_order_matters(self):
        """Different tool ordering → different key (tools are ordered)."""
        a = [{"name": "a"}, {"name": "b"}]
        b = [{"name": "b"}, {"name": "a"}]
        assert tools_cache_key("t", a) != tools_cache_key("t", b)


# ---------------------------------------------------------------------------
# schema_cache_key
# ---------------------------------------------------------------------------


class TestSchemaCacheKey:
    def test_deterministic(self):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        assert schema_cache_key(schema) == schema_cache_key(schema)

    def test_extra_strip_keys_affects_key(self):
        schema = {"type": "object"}
        k1 = schema_cache_key(schema, None)
        k2 = schema_cache_key(schema, frozenset({"additionalProperties"}))
        assert k1 != k2


# ---------------------------------------------------------------------------
# LRUCache
# ---------------------------------------------------------------------------


class TestLRUCache:
    def test_basic_get_put(self):
        cache = LRUCache(maxsize=4)
        cache.put(1, "one")
        assert cache.get(1) == "one"

    def test_miss_returns_sentinel(self):
        cache = LRUCache(maxsize=4)
        assert cache.get(999) is _SENTINEL

    def test_eviction_at_maxsize(self):
        cache = LRUCache(maxsize=2)
        cache.put(1, "a")
        cache.put(2, "b")
        cache.put(3, "c")  # evicts key=1
        assert cache.get(1) is _SENTINEL
        assert cache.get(2) == "b"
        assert cache.get(3) == "c"

    def test_move_to_end_on_access(self):
        cache = LRUCache(maxsize=2)
        cache.put(1, "a")
        cache.put(2, "b")
        cache.get(1)  # access 1 → moves to end
        cache.put(3, "c")  # should evict 2 (now LRU), not 1
        assert cache.get(1) == "a"
        assert cache.get(2) is _SENTINEL
        assert cache.get(3) == "c"

    def test_update_existing_key(self):
        cache = LRUCache(maxsize=4)
        cache.put(1, "old")
        cache.put(1, "new")
        assert cache.get(1) == "new"
        assert cache.info()["currsize"] == 1

    def test_clear_resets_all(self):
        cache = LRUCache(maxsize=4)
        cache.put(1, "a")
        cache.get(1)  # 1 hit
        cache.get(2)  # 1 miss
        cache.clear()
        assert cache.get(1) is _SENTINEL
        info = cache.info()
        assert info["hits"] == 0
        assert info["misses"] == 1  # the miss from get(1) after clear

    def test_info_counters(self):
        cache = LRUCache(maxsize=4)
        cache.put(1, "a")
        cache.get(1)  # hit
        cache.get(1)  # hit
        cache.get(2)  # miss
        info = cache.info()
        assert info["hits"] == 2
        assert info["misses"] == 1
        assert info["currsize"] == 1
        assert info["maxsize"] == 4

    def test_no_ttl(self):
        """ttl=None disables expiry — entries live until LRU-evicted."""
        cache = LRUCache(maxsize=4, ttl=None)
        cache.put(1, "a")
        assert cache.get(1) == "a"
        assert cache.info()["ttl"] is None

    def test_ttl_expiry(self):
        """Entry should expire after TTL elapses."""
        cache = LRUCache(maxsize=4, ttl=10.0)
        base_time = 1000.0
        with patch(
            "llm_rosetta.converters.base.cache.time.monotonic", return_value=base_time
        ):
            cache.put(1, "a")

        # Before expiry
        with patch(
            "llm_rosetta.converters.base.cache.time.monotonic",
            return_value=base_time + 9.9,
        ):
            assert cache.get(1) == "a"

        # After expiry
        with patch(
            "llm_rosetta.converters.base.cache.time.monotonic",
            return_value=base_time + 10.0,
        ):
            assert cache.get(1) is _SENTINEL

        assert cache.info()["expirations"] == 1
        assert cache.info()["currsize"] == 0

    def test_put_resets_ttl(self):
        """Re-putting the same key should reset the TTL deadline."""
        cache = LRUCache(maxsize=4, ttl=10.0)
        base_time = 1000.0
        with patch(
            "llm_rosetta.converters.base.cache.time.monotonic", return_value=base_time
        ):
            cache.put(1, "a")

        # Re-put at t=8 → new deadline = t=18
        with patch(
            "llm_rosetta.converters.base.cache.time.monotonic",
            return_value=base_time + 8.0,
        ):
            cache.put(1, "b")

        # At t=15 the original deadline (t=10) would have expired,
        # but the re-put extended it to t=18
        with patch(
            "llm_rosetta.converters.base.cache.time.monotonic",
            return_value=base_time + 15.0,
        ):
            assert cache.get(1) == "b"

    def test_default_ttl(self):
        """Module-level singletons should use DEFAULT_TTL."""
        assert DEFAULT_TTL == 1800.0
        cache = LRUCache(maxsize=4)
        assert cache.info()["ttl"] == DEFAULT_TTL


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------


class TestModuleSingletons:
    def test_clear_all_caches(self):
        from llm_rosetta.converters.base.cache import (
            sanitize_cache,
            tools_from_p_cache,
            tools_to_p_cache,
        )

        tools_from_p_cache.put(1, "x")
        tools_to_p_cache.put(2, "y")
        sanitize_cache.put(3, "z")

        clear_all_caches()

        assert tools_from_p_cache.get(1) is _SENTINEL
        assert tools_to_p_cache.get(2) is _SENTINEL
        assert sanitize_cache.get(3) is _SENTINEL

    def test_cache_info_structure(self):
        info = cache_info()
        assert set(info.keys()) == {"tools_from_p", "tools_to_p", "sanitize"}
        for v in info.values():
            assert "hits" in v
            assert "misses" in v
            assert "expirations" in v
            assert "currsize" in v
            assert "maxsize" in v
            assert "ttl" in v
