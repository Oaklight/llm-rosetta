"""Unit tests for the per-entry LRU cache infrastructure."""

from unittest.mock import patch

from llm_rosetta.converters.base.helpers.cache import (
    DEFAULT_TTL,
    LRUCache,
    _SENTINEL,
    _canonical_json_bytes,
    cache_info,
    clear_all_caches,
    entry_cache_key,
    get_cached_tool,
    is_ir_validated,
    mark_ir_validated,
    put_cached_tool,
    schema_cache_key,
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
# entry_cache_key
# ---------------------------------------------------------------------------


class TestEntryCacheKey:
    def test_deterministic(self):
        tool = {"name": "foo", "type": "function"}
        k1 = entry_cache_key("test:from_p", tool)
        k2 = entry_cache_key("test:from_p", tool)
        assert k1 == k2

    def test_varies_by_tag(self):
        tool = {"name": "foo", "type": "function"}
        k1 = entry_cache_key("anthropic:from_p", tool)
        k2 = entry_cache_key("openai_chat:from_p", tool)
        assert k1 != k2

    def test_varies_by_direction(self):
        tool = {"name": "foo", "type": "function"}
        k1 = entry_cache_key("anthropic:from_p", tool)
        k2 = entry_cache_key("anthropic:to_p", tool)
        assert k1 != k2

    def test_varies_by_content(self):
        tool_a = {"name": "foo", "type": "function"}
        tool_b = {"name": "bar", "type": "function"}
        assert entry_cache_key("t", tool_a) != entry_cache_key("t", tool_b)

    def test_order_independent_within_dict(self):
        """Same dict content with different key order → same key."""
        tool_a = {"type": "function", "name": "foo"}
        tool_b = {"name": "foo", "type": "function"}
        assert entry_cache_key("t", tool_a) == entry_cache_key("t", tool_b)


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

    def test_check_integrity_clean(self):
        """check_integrity returns empty list when nothing is mutated."""
        cache = LRUCache(maxsize=4, ttl=None)
        cache.put(1, [{"name": "foo"}])
        cache.put(2, [{"name": "bar"}])
        assert cache.check_integrity() == []

    def test_check_integrity_detects_mutation(self):
        """check_integrity catches in-place mutation of cached values."""
        cache = LRUCache(maxsize=4, ttl=None)
        original = [{"name": "foo", "params": {"type": "object"}}]
        cache.put(1, original)
        original[0]["name"] = "MUTATED"
        assert cache.check_integrity() == [1]

    def test_check_integrity_detects_deep_mutation(self):
        """check_integrity catches nested dict mutation."""
        cache = LRUCache(maxsize=4, ttl=None)
        data = [{"name": "foo", "params": {"type": "object", "props": {}}}]
        cache.put(1, data)
        data[0]["params"]["props"]["new_key"] = "injected"
        assert cache.check_integrity() == [1]

    def test_verify_mode_evicts_mutated_on_get(self):
        """With verify=True, get() detects mutation and returns miss."""
        cache = LRUCache(maxsize=4, ttl=None, verify=True)
        data = [{"name": "foo"}]
        cache.put(1, data)
        assert cache.get(1) == data  # hit
        data[0]["name"] = "MUTATED"
        assert cache.get(1) is _SENTINEL  # self-healed miss
        assert cache.info()["corruptions"] == 1
        assert cache.info()["currsize"] == 0

    def test_verify_off_by_default(self):
        """With default verify=False, get() does not check fingerprint."""
        cache = LRUCache(maxsize=4, ttl=None)
        data = [{"name": "foo"}]
        cache.put(1, data)
        data[0]["name"] = "MUTATED"
        result = cache.get(1)
        assert result[0]["name"] == "MUTATED"
        assert cache.info()["corruptions"] == 0

    def test_no_ttl(self):
        """ttl=None disables expiry — entries live until LRU-evicted."""
        cache = LRUCache(maxsize=4, ttl=None)
        cache.put(1, "a")
        assert cache.get(1) == "a"
        assert cache.info()["ttl"] is None

    def test_ttl_expiry(self):
        """Entry should expire after TTL elapses with no intervening access."""
        cache = LRUCache(maxsize=4, ttl=10.0)
        base_time = 1000.0
        with patch(
            "llm_rosetta.converters.base.helpers.cache.time.monotonic",
            return_value=base_time,
        ):
            cache.put(1, "a")

        with patch(
            "llm_rosetta.converters.base.helpers.cache.time.monotonic",
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
            "llm_rosetta.converters.base.helpers.cache.time.monotonic",
            return_value=base_time,
        ):
            cache.put(1, "a")

        with patch(
            "llm_rosetta.converters.base.helpers.cache.time.monotonic",
            return_value=base_time + 8.0,
        ):
            cache.put(1, "b")

        with patch(
            "llm_rosetta.converters.base.helpers.cache.time.monotonic",
            return_value=base_time + 15.0,
        ):
            assert cache.get(1) == "b"

    def test_get_refreshes_ttl(self):
        """Reading an entry should refresh its TTL deadline."""
        cache = LRUCache(maxsize=4, ttl=10.0)
        base_time = 1000.0
        with patch(
            "llm_rosetta.converters.base.helpers.cache.time.monotonic",
            return_value=base_time,
        ):
            cache.put(1, "a")

        with patch(
            "llm_rosetta.converters.base.helpers.cache.time.monotonic",
            return_value=base_time + 8.0,
        ):
            assert cache.get(1) == "a"

        with patch(
            "llm_rosetta.converters.base.helpers.cache.time.monotonic",
            return_value=base_time + 15.0,
        ):
            assert cache.get(1) == "a"

        with patch(
            "llm_rosetta.converters.base.helpers.cache.time.monotonic",
            return_value=base_time + 26.0,
        ):
            assert cache.get(1) is _SENTINEL

    def test_default_ttl(self):
        """Module-level singletons should use DEFAULT_TTL."""
        assert DEFAULT_TTL == 1800.0
        cache = LRUCache(maxsize=4)
        assert cache.info()["ttl"] == DEFAULT_TTL


# ---------------------------------------------------------------------------
# Per-entry tool helpers
# ---------------------------------------------------------------------------


class TestToolEntryHelpers:
    def test_get_put_roundtrip(self):
        clear_all_caches()
        put_cached_tool(
            "test:from_p", {"name": "foo"}, {"type": "function", "name": "foo"}
        )
        result = get_cached_tool("test:from_p", {"name": "foo"})
        assert result == {"type": "function", "name": "foo"}

    def test_miss_returns_sentinel(self):
        clear_all_caches()
        assert get_cached_tool("test:from_p", {"name": "missing"}) is _SENTINEL

    def test_different_tags_dont_collide(self):
        clear_all_caches()
        tool = {"name": "foo"}
        put_cached_tool("anthropic:from_p", tool, "anthropic_result")
        put_cached_tool("openai_chat:from_p", tool, "openai_result")
        assert get_cached_tool("anthropic:from_p", tool) == "anthropic_result"
        assert get_cached_tool("openai_chat:from_p", tool) == "openai_result"


# ---------------------------------------------------------------------------
# Unified IR validation helpers
# ---------------------------------------------------------------------------


class TestIRValidationHelpers:
    def test_not_validated_initially(self):
        clear_all_caches()
        msg = {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        assert is_ir_validated("ir.message", msg) is False

    def test_mark_and_check(self):
        clear_all_caches()
        msg = {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        mark_ir_validated("ir.message", msg)
        assert is_ir_validated("ir.message", msg) is True

    def test_different_entries_independent(self):
        clear_all_caches()
        msg1 = {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        msg2 = {"role": "user", "content": [{"type": "text", "text": "world"}]}
        mark_ir_validated("ir.message", msg1)
        assert is_ir_validated("ir.message", msg1) is True
        assert is_ir_validated("ir.message", msg2) is False

    def test_dict_key_order_independent(self):
        clear_all_caches()
        msg1 = {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        msg2 = {"content": [{"type": "text", "text": "hi"}], "role": "user"}
        mark_ir_validated("ir.message", msg1)
        assert is_ir_validated("ir.message", msg2) is True

    def test_different_tags_independent(self):
        """Same content with different tags should not collide."""
        clear_all_caches()
        entry = {"name": "foo", "type": "function"}
        mark_ir_validated("ir.tool", entry)
        assert is_ir_validated("ir.tool", entry) is True
        assert is_ir_validated("ir.message", entry) is False

    def test_cross_converter_sharing(self):
        """IR validation is converter-agnostic — no converter tag."""
        clear_all_caches()
        tool = {"type": "function", "name": "foo", "description": "d", "parameters": {}}
        mark_ir_validated("ir.tool", tool)
        # Same IR tool, different "converter" — should still hit
        assert is_ir_validated("ir.tool", tool) is True


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------


class TestModuleSingletons:
    def test_clear_all_caches(self):
        from llm_rosetta.converters.base.helpers.cache import (
            ir_validation_cache,
            sanitize_cache,
            tool_entry_cache,
        )

        tool_entry_cache.put(1, "x")
        sanitize_cache.put(2, "y")
        ir_validation_cache.put(3, True)

        clear_all_caches()

        assert tool_entry_cache.get(1) is _SENTINEL
        assert sanitize_cache.get(2) is _SENTINEL
        assert ir_validation_cache.get(3) is _SENTINEL

    def test_cache_info_structure(self):
        info = cache_info()
        assert set(info.keys()) == {"tool_entry", "sanitize", "ir_validation"}
        for v in info.values():
            assert "hits" in v
            assert "misses" in v
            assert "expirations" in v
            assert "corruptions" in v
            assert "currsize" in v
            assert "maxsize" in v
            assert "ttl" in v
