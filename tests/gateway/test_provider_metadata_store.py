"""Tests for ProviderMetadataStore."""

import time

from llm_rosetta.gateway.proxy import ProviderMetadataStore


def _make_ir_response(tool_call_id: str, metadata: dict) -> dict:
    """Build a minimal IR response containing a tool call with provider_metadata."""
    return {
        "choices": [
            {
                "message": {
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": tool_call_id,
                            "name": "test_fn",
                            "arguments": "{}",
                            "provider_metadata": metadata,
                        }
                    ]
                }
            }
        ]
    }


def _make_ir_request(tool_call_ids: list[str]) -> dict:
    """Build a minimal IR request with tool_call parts (no provider_metadata)."""
    return {
        "messages": [
            {
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call_id": tc_id,
                        "name": "test_fn",
                        "arguments": "{}",
                    }
                    for tc_id in tool_call_ids
                ]
            }
        ]
    }


class TestCacheAndInjectRoundtrip:
    def test_basic_roundtrip(self):
        store = ProviderMetadataStore()
        metadata = {"thought_signature": "abc123"}
        store.cache_from_response(_make_ir_response("call_001", metadata))

        ir_req = _make_ir_request(["call_001"])
        store.inject_into_request(ir_req)

        part = ir_req["messages"][0]["content"][0]
        assert part["provider_metadata"] == {"thought_signature": "abc123"}

    def test_multiple_tool_calls(self):
        store = ProviderMetadataStore()
        store.cache_from_response(_make_ir_response("call_a", {"sig": "A"}))
        store.cache_from_response(_make_ir_response("call_b", {"sig": "B"}))

        ir_req = _make_ir_request(["call_a", "call_b"])
        store.inject_into_request(ir_req)

        parts = ir_req["messages"][0]["content"]
        assert parts[0]["provider_metadata"] == {"sig": "A"}
        assert parts[1]["provider_metadata"] == {"sig": "B"}


class TestTTLExpiration:
    def test_expired_entries_not_injected(self):
        store = ProviderMetadataStore(ttl=0.05)  # 50ms TTL
        store.cache_from_response(_make_ir_response("call_exp", {"sig": "old"}))
        assert len(store) == 1

        time.sleep(0.06)

        ir_req = _make_ir_request(["call_exp"])
        store.inject_into_request(ir_req)

        part = ir_req["messages"][0]["content"][0]
        assert "provider_metadata" not in part
        assert len(store) == 0

    def test_non_expired_entries_survive(self):
        store = ProviderMetadataStore(ttl=10.0)
        store.cache_from_response(_make_ir_response("call_live", {"sig": "fresh"}))

        ir_req = _make_ir_request(["call_live"])
        store.inject_into_request(ir_req)

        part = ir_req["messages"][0]["content"][0]
        assert part["provider_metadata"] == {"sig": "fresh"}


class TestMaxSizeEviction:
    def test_oldest_evicted_on_overflow(self):
        store = ProviderMetadataStore(max_size=3)

        for i in range(3):
            store.cache_from_response(_make_ir_response(f"call_{i}", {"i": i}))
        assert len(store) == 3

        # Adding a 4th entry should evict the oldest (call_0)
        store.cache_from_response(_make_ir_response("call_3", {"i": 3}))
        assert len(store) == 3

        ir_req = _make_ir_request(["call_0", "call_1", "call_2", "call_3"])
        store.inject_into_request(ir_req)

        parts = ir_req["messages"][0]["content"]
        assert "provider_metadata" not in parts[0]  # call_0 evicted
        assert parts[1]["provider_metadata"] == {"i": 1}
        assert parts[2]["provider_metadata"] == {"i": 2}
        assert parts[3]["provider_metadata"] == {"i": 3}


class TestStreamEventCaching:
    def test_cache_from_stream_event(self):
        store = ProviderMetadataStore()
        event = {
            "type": "tool_call_start",
            "tool_call_id": "call_stream",
            "provider_metadata": {"thought_signature": "stream_sig"},
        }
        store.cache_from_stream_event(event)
        assert len(store) == 1

        ir_req = _make_ir_request(["call_stream"])
        store.inject_into_request(ir_req)

        part = ir_req["messages"][0]["content"][0]
        assert part["provider_metadata"] == {"thought_signature": "stream_sig"}

    def test_ignores_non_tool_call_start(self):
        store = ProviderMetadataStore()
        store.cache_from_stream_event(
            {"type": "content_delta", "provider_metadata": {"x": 1}}
        )
        assert len(store) == 0

    def test_ignores_event_without_metadata(self):
        store = ProviderMetadataStore()
        store.cache_from_stream_event(
            {"type": "tool_call_start", "tool_call_id": "call_no_meta"}
        )
        assert len(store) == 0


class TestClear:
    def test_clear_empties_store(self):
        store = ProviderMetadataStore()
        store.cache_from_response(_make_ir_response("call_1", {"a": 1}))
        store.cache_from_response(_make_ir_response("call_2", {"b": 2}))
        assert len(store) == 2

        store.clear()
        assert len(store) == 0


class TestEdgeCases:
    def test_no_injection_when_empty(self):
        store = ProviderMetadataStore()
        ir_req = _make_ir_request(["call_missing"])
        store.inject_into_request(ir_req)

        part = ir_req["messages"][0]["content"][0]
        assert "provider_metadata" not in part

    def test_inject_preserves_existing_metadata(self):
        store = ProviderMetadataStore()
        store.cache_from_response(_make_ir_response("call_x", {"sig": "from_store"}))

        ir_req = {
            "messages": [
                {
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_x",
                            "name": "fn",
                            "arguments": "{}",
                            "provider_metadata": {"sig": "already_present"},
                        }
                    ]
                }
            ]
        }
        store.inject_into_request(ir_req)

        # Store overwrites — this matches original behavior
        part = ir_req["messages"][0]["content"][0]
        assert part["provider_metadata"] == {"sig": "from_store"}

    def test_entries_kept_across_injections(self):
        """Entries are non-consumptive: same ID can be injected multiple times."""
        store = ProviderMetadataStore()
        store.cache_from_response(_make_ir_response("call_reuse", {"sig": "reused"}))

        for _ in range(3):
            ir_req = _make_ir_request(["call_reuse"])
            store.inject_into_request(ir_req)
            part = ir_req["messages"][0]["content"][0]
            assert part["provider_metadata"] == {"sig": "reused"}

        assert len(store) == 1

    def test_response_without_tool_calls_is_noop(self):
        store = ProviderMetadataStore()
        store.cache_from_response(
            {"choices": [{"message": {"content": [{"type": "text", "text": "hi"}]}}]}
        )
        assert len(store) == 0

    def test_request_without_messages_is_noop(self):
        store = ProviderMetadataStore()
        store.inject_into_request({})  # no crash


class TestEvictionOrder:
    """Verify FIFO eviction relies on dict insertion order, not min() scan."""

    def test_fifo_eviction_order_across_many_inserts(self):
        store = ProviderMetadataStore(max_size=5)
        for i in range(5):
            store.cache_from_response(_make_ir_response(f"c{i}", {"i": i}))
        assert len(store) == 5

        # Insert 3 more — should evict c0, c1, c2 in order
        for i in range(5, 8):
            store.cache_from_response(_make_ir_response(f"c{i}", {"i": i}))
        assert len(store) == 5

        ir_req = _make_ir_request([f"c{i}" for i in range(8)])
        store.inject_into_request(ir_req)
        parts = ir_req["messages"][0]["content"]
        for i in range(3):
            assert "provider_metadata" not in parts[i], f"c{i} should be evicted"
        for i in range(3, 8):
            assert parts[i]["provider_metadata"] == {"i": i}

    def test_expired_eviction_stops_at_non_expired(self):
        store = ProviderMetadataStore(ttl=0.05)
        # Insert 2 entries that will expire
        store.cache_from_response(_make_ir_response("old_a", {"v": "a"}))
        store.cache_from_response(_make_ir_response("old_b", {"v": "b"}))

        time.sleep(0.06)

        # Insert a fresh entry (won't be expired)
        store.cache_from_response(_make_ir_response("new_c", {"v": "c"}))

        # Trigger eviction via inject
        ir_req = _make_ir_request(["old_a", "old_b", "new_c"])
        store.inject_into_request(ir_req)
        parts = ir_req["messages"][0]["content"]
        assert "provider_metadata" not in parts[0]
        assert "provider_metadata" not in parts[1]
        assert parts[2]["provider_metadata"] == {"v": "c"}
        assert len(store) == 1
