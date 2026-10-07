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

    def test_dict_key_order_dependent(self):
        """pickle-based validation keys are key-order dependent.

        Different key order → cache miss → re-validate (safe).
        """
        clear_all_caches()
        msg1 = {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        msg2 = {"content": [{"type": "text", "text": "hi"}], "role": "user"}
        mark_ir_validated("ir.message", msg1)
        assert is_ir_validated("ir.message", msg2) is False

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


# ---------------------------------------------------------------------------
# _content_hash_bytes (pickle-based, Layer 1)
# ---------------------------------------------------------------------------


class TestContentHashBytes:
    def test_deterministic(self):
        from llm_rosetta.converters.base.helpers.cache import _content_hash_bytes

        obj = {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        assert _content_hash_bytes(obj) == _content_hash_bytes(obj)

    def test_different_content_different_bytes(self):
        from llm_rosetta.converters.base.helpers.cache import _content_hash_bytes

        a = _content_hash_bytes({"text": "hello"})
        b = _content_hash_bytes({"text": "world"})
        assert a != b

    def test_key_order_dependent(self):
        """pickle is key-order dependent — this is the trade-off for speed."""
        from llm_rosetta.converters.base.helpers.cache import _content_hash_bytes

        a = _content_hash_bytes({"a": 1, "b": 2})
        b = _content_hash_bytes({"b": 2, "a": 1})
        assert a != b

    def test_faster_than_json(self):
        """pickle should be meaningfully faster than json for nested dicts."""
        import json
        import time

        from llm_rosetta.converters.base.helpers.cache import _content_hash_bytes

        obj = {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello " * 50},
                {
                    "type": "tool_call",
                    "tool_call_id": "tc_1",
                    "tool_name": "fn",
                    "input": {"arg": "value " * 20},
                },
            ],
        }

        n = 1000
        t0 = time.perf_counter()
        for _ in range(n):
            _content_hash_bytes(obj)
        t_pickle = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(n):
            json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
        t_json = time.perf_counter() - t0

        speedup = t_json / t_pickle
        assert speedup > 2.0, f"Expected >2x speedup, got {speedup:.1f}x"


# ---------------------------------------------------------------------------
# GenerationalSet (Layer 2)
# ---------------------------------------------------------------------------


class TestGenerationalSet:
    def test_put_and_get(self):
        from llm_rosetta.converters.base.helpers.cache import GenerationalSet

        gs = GenerationalSet()
        gs.put(42, True)
        assert gs.get(42) is True

    def test_miss_returns_sentinel(self):
        from llm_rosetta.converters.base.helpers.cache import GenerationalSet

        gs = GenerationalSet()
        assert gs.get(999) is _SENTINEL

    def test_clear(self):
        from llm_rosetta.converters.base.helpers.cache import GenerationalSet

        gs = GenerationalSet()
        gs.put(1, True)
        gs.put(2, True)
        gs.clear()
        assert gs.get(1) is _SENTINEL
        assert gs.get(2) is _SENTINEL

    def test_rotation_evicts_old_generation(self):
        from llm_rosetta.converters.base.helpers.cache import GenerationalSet

        gs = GenerationalSet(rotate_interval=3600.0)
        gs.put(1, True)
        assert gs.get(1) is True

        # Force first rotation
        gs._last_rotate -= 7200.0
        gs._maybe_rotate()
        # Entry moved to previous generation — should still be findable
        assert gs.get(1) is True

        # Force second rotation — entry was promoted by get(),
        # so it survives this rotation too
        gs._last_rotate -= 7200.0
        gs._maybe_rotate()
        assert gs.get(1) is True

        # Now don't access it, force two rotations
        gs._last_rotate -= 7200.0
        gs._maybe_rotate()
        gs._last_rotate -= 7200.0
        gs._maybe_rotate()
        assert gs.get(1) is _SENTINEL

    def test_get_promotes_to_current(self):
        from llm_rosetta.converters.base.helpers.cache import GenerationalSet

        gs = GenerationalSet(rotate_interval=3600.0)
        gs.put(1, True)
        # Force rotation
        gs._last_rotate -= 7200.0
        gs._maybe_rotate()
        # Entry is in previous, get() should promote to current
        assert gs.get(1) is True
        assert 1 in gs._current

    def test_info_structure(self):
        from llm_rosetta.converters.base.helpers.cache import GenerationalSet

        gs = GenerationalSet()
        gs.put(1, True)
        gs.get(1)  # hit
        gs.get(2)  # miss
        info = gs.info()
        assert info["hits"] == 1
        assert info["misses"] == 1
        assert info["currsize"] == 1
        assert "ttl" in info

    def test_check_integrity_always_empty(self):
        from llm_rosetta.converters.base.helpers.cache import GenerationalSet

        gs = GenerationalSet()
        gs.put(1, True)
        assert gs.check_integrity() == []

    def test_hit_miss_counters(self):
        from llm_rosetta.converters.base.helpers.cache import GenerationalSet

        gs = GenerationalSet()
        gs.put(10, True)
        gs.get(10)
        gs.get(10)
        gs.get(99)  # miss
        info = gs.info()
        assert info["hits"] == 2
        assert info["misses"] == 1


# ---------------------------------------------------------------------------
# Epoch guard (_entry_signal + _list_epoch, Layer 3)
# ---------------------------------------------------------------------------


class TestEntrySignal:
    def test_text_part(self):
        from llm_rosetta.converters.base.helpers.cache import _entry_signal

        sig = _entry_signal({"type": "text", "text": "hello world"})
        assert isinstance(sig, int)
        # Same input → same signal
        assert sig == _entry_signal({"type": "text", "text": "hello world"})
        # Different text → different signal
        assert sig != _entry_signal({"type": "text", "text": "goodbye world"})

    def test_tool_call_part(self):
        from llm_rosetta.converters.base.helpers.cache import _entry_signal

        tc = {
            "type": "tool_call",
            "tool_call_id": "tc_1",
            "tool_name": "read_file",
            "input": {"path": "/tmp/x"},
        }
        sig = _entry_signal(tc)
        assert isinstance(sig, int)
        # Different tool → different signal
        tc2 = {**tc, "tool_name": "write_file"}
        assert sig != _entry_signal(tc2)

    def test_tool_result_part(self):
        from llm_rosetta.converters.base.helpers.cache import _entry_signal

        tr = {"type": "tool_result", "tool_call_id": "tc_1", "result": "output text"}
        sig = _entry_signal(tr)
        assert isinstance(sig, int)

    def test_reasoning_part(self):
        from llm_rosetta.converters.base.helpers.cache import _entry_signal

        r = {"type": "reasoning", "reasoning": "Let me think..."}
        sig = _entry_signal(r)
        assert isinstance(sig, int)

    def test_tool_definition(self):
        from llm_rosetta.converters.base.helpers.cache import _entry_signal

        td = {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }
        sig = _entry_signal(td)
        assert isinstance(sig, int)
        # Different tool → different signal
        td2 = {**td, "name": "write_file"}
        assert sig != _entry_signal(td2)

    def test_message_envelope(self):
        from llm_rosetta.converters.base.helpers.cache import _entry_signal

        msg = {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        sig = _entry_signal(msg)
        assert isinstance(sig, int)

    def test_image_part(self):
        from llm_rosetta.converters.base.helpers.cache import _entry_signal

        img = {"type": "image", "image_url": "https://example.com/img.png"}
        sig = _entry_signal(img)
        assert isinstance(sig, int)

    def test_citation_part(self):
        from llm_rosetta.converters.base.helpers.cache import _entry_signal

        cit = {"type": "citation", "url": "https://example.com/doc"}
        sig = _entry_signal(cit)
        assert isinstance(sig, int)


class TestListEpoch:
    def test_same_list_same_epoch(self):
        from llm_rosetta.converters.base.helpers.cache import _list_epoch

        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
        ]
        assert _list_epoch(msgs, "ir.message") == _list_epoch(msgs, "ir.message")

    def test_different_content_different_epoch(self):
        from llm_rosetta.converters.base.helpers.cache import _list_epoch

        msgs1 = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        msgs2 = [{"role": "user", "content": [{"type": "text", "text": "bye"}]}]
        assert _list_epoch(msgs1, "ir.message") != _list_epoch(msgs2, "ir.message")

    def test_different_length_different_epoch(self):
        from llm_rosetta.converters.base.helpers.cache import _list_epoch

        msgs1 = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        msgs2 = msgs1 + [
            {"role": "assistant", "content": [{"type": "text", "text": "yo"}]}
        ]
        assert _list_epoch(msgs1, "ir.message") != _list_epoch(msgs2, "ir.message")

    def test_empty_list(self):
        from llm_rosetta.converters.base.helpers.cache import _list_epoch

        assert isinstance(_list_epoch([], "ir.message"), int)

    def test_different_tag_different_epoch(self):
        from llm_rosetta.converters.base.helpers.cache import _list_epoch

        msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        assert _list_epoch(msgs, "ir.message") != _list_epoch(msgs, "ir.tool")

    def test_mutation_at_various_positions(self):
        """Mutations at any position should change the epoch."""
        from llm_rosetta.converters.base.helpers.cache import _list_epoch

        msgs = [
            {"role": "user", "content": [{"type": "text", "text": f"msg {i}"}]}
            for i in range(10)
        ]
        baseline = _list_epoch(msgs, "ir.message")

        for pos in [0, 3, 5, 9]:
            modified = [m.copy() for m in msgs]
            modified[pos] = {
                "role": "user",
                "content": [{"type": "text", "text": "CHANGED"}],
            }
            assert _list_epoch(modified, "ir.message") != baseline, (
                f"Mutation at position {pos} not detected"
            )

    def test_insertion_detected(self):
        from llm_rosetta.converters.base.helpers.cache import _list_epoch

        msgs = [
            {"role": "user", "content": [{"type": "text", "text": f"msg {i}"}]}
            for i in range(5)
        ]
        baseline = _list_epoch(msgs, "ir.message")
        extended = msgs + [
            {"role": "user", "content": [{"type": "text", "text": "new"}]}
        ]
        assert _list_epoch(extended, "ir.message") != baseline

    def test_removal_detected(self):
        from llm_rosetta.converters.base.helpers.cache import _list_epoch

        msgs = [
            {"role": "user", "content": [{"type": "text", "text": f"msg {i}"}]}
            for i in range(5)
        ]
        baseline = _list_epoch(msgs, "ir.message")
        shorter = msgs[:4]
        assert _list_epoch(shorter, "ir.message") != baseline


class TestEpochCacheIntegration:
    """Test that the epoch guard correctly skips per-entry validation."""

    def test_epoch_cache_cleared_on_clear_all(self):
        from llm_rosetta.converters.base.helpers.cache import _epoch_cache

        clear_all_caches()
        _epoch_cache[("test", "tag")] = 42
        clear_all_caches()
        assert len(_epoch_cache) == 0

    def test_second_call_skips_per_entry_via_epoch(self):
        """On second call with same data, epoch guard should short-circuit."""
        import copy

        from llm_rosetta.converters.anthropic import AnthropicConverter

        clear_all_caches()
        request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
                {"role": "user", "content": [{"type": "text", "text": "how are you?"}]},
            ],
        }

        conv = AnthropicConverter()
        ir1 = conv.request_from_provider(copy.deepcopy(request))
        info1 = cache_info()["ir_validation"]

        # Second call with identical input
        conv2 = AnthropicConverter()
        ir2 = conv2.request_from_provider(copy.deepcopy(request))
        info2 = cache_info()["ir_validation"]

        # Epoch guard should have skipped per-entry work on second call,
        # so no new misses should be recorded (epoch short-circuited)
        assert info2["misses"] == info1["misses"], (
            f"Expected no new misses (epoch guard should skip), "
            f"got {info2['misses']} vs {info1['misses']}"
        )
        # Both results should be identical
        assert ir1["messages"] == ir2["messages"]
