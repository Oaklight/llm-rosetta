"""Tests for API key affinity — deterministic key selection for cache locality.

Covers:
- KeyRing.select() — positional lookup without advancing round-robin
- extract_prefix_from_ir() — stable fingerprint from IR request
- compute_affinity_index() — deterministic hash-based key index
- ProviderInfo.with_affinity() — shallow-copy with pinned key index
- KeyContext.key_hash — client identity field
"""

from __future__ import annotations

import pytest

from llm_rosetta.gateway.affinity import compute_affinity_index, extract_prefix_from_ir
from llm_rosetta.gateway.transport.provider_info import KeyRing, ProviderInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider_info(
    *,
    api_key: str = "test-key",
    base_url: str = "https://api.example.com",
    url_template: str = "{base_url}/v1/chat/completions",
) -> ProviderInfo:
    return ProviderInfo(
        "test",
        api_key=api_key,
        base_url=base_url,
        auth_header_fn=lambda key: {"Authorization": f"Bearer {key}"},
        url_template=url_template,
    )


# ---------------------------------------------------------------------------
# KeyRing.select()
# ---------------------------------------------------------------------------


class TestKeyRingSelect:
    def test_select_returns_correct_key(self):
        ring = KeyRing("key-a,key-b,key-c")
        assert ring.select(0) == "key-a"
        assert ring.select(1) == "key-b"
        assert ring.select(2) == "key-c"

    def test_select_wraps_around(self):
        ring = KeyRing("key-a,key-b,key-c")
        assert ring.select(3) == "key-a"
        assert ring.select(4) == "key-b"

    def test_select_does_not_affect_round_robin(self):
        ring = KeyRing("key-a,key-b")
        ring.select(0)  # should not change _idx
        ring.select(1)
        assert ring.next() == "key-a"  # still starts from 0
        assert ring.next() == "key-b"

    def test_select_single_key(self):
        ring = KeyRing("only-key")
        assert ring.select(0) == "only-key"
        assert ring.select(42) == "only-key"

    def test_select_empty_raises(self):
        ring = KeyRing("")
        with pytest.raises(ValueError):
            ring.select(0)


# ---------------------------------------------------------------------------
# extract_prefix_from_ir()
# ---------------------------------------------------------------------------


class TestExtractPrefixFromIR:
    def test_system_and_messages(self):
        ir = {
            "system_instruction": [{"type": "text", "text": "You are helpful."}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
                {"role": "user", "content": [{"type": "text", "text": "Bye"}]},
            ],
        }
        prefix = extract_prefix_from_ir(ir)
        assert "You are helpful." in prefix
        assert "Hello" in prefix
        assert "Hi" in prefix
        # third message should NOT be included (max_messages=2 default)
        assert "Bye" not in prefix

    def test_no_system(self):
        ir = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            ],
        }
        prefix = extract_prefix_from_ir(ir)
        assert "Hello" in prefix
        assert prefix  # not empty

    def test_empty_ir(self):
        assert extract_prefix_from_ir({}) == ""

    def test_none_like_ir(self):
        assert extract_prefix_from_ir({}) == ""

    def test_custom_max_messages(self):
        ir = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "msg1"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "msg2"}]},
                {"role": "user", "content": [{"type": "text", "text": "msg3"}]},
            ],
        }
        prefix = extract_prefix_from_ir(ir, max_messages=1)
        assert "msg1" in prefix
        assert "msg2" not in prefix

    def test_determinism(self):
        ir = {
            "system_instruction": [{"type": "text", "text": "system"}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            ],
        }
        assert extract_prefix_from_ir(ir) == extract_prefix_from_ir(ir)

    def test_system_only(self):
        ir = {"system_instruction": [{"type": "text", "text": "Be concise."}]}
        prefix = extract_prefix_from_ir(ir)
        assert "Be concise." in prefix


# ---------------------------------------------------------------------------
# compute_affinity_index()
# ---------------------------------------------------------------------------


class TestComputeAffinityIndex:
    def test_deterministic(self):
        idx1 = compute_affinity_index("hash123", "prefix", 5)
        idx2 = compute_affinity_index("hash123", "prefix", 5)
        assert idx1 == idx2

    def test_in_range(self):
        for n in [2, 3, 5, 10, 100]:
            idx = compute_affinity_index("test-hash", "test-prefix", n)
            assert idx is not None
            assert 0 <= idx < n

    def test_different_clients(self):
        idx_a = compute_affinity_index("client-a", "same-prefix", 100)
        idx_b = compute_affinity_index("client-b", "same-prefix", 100)
        # with 100 keys, different clients should (almost certainly) get
        # different indices — collision probability ~1%
        assert idx_a != idx_b

    def test_different_prefixes(self):
        idx_a = compute_affinity_index("same-client", "prefix-alpha", 100)
        idx_b = compute_affinity_index("same-client", "prefix-beta", 100)
        # with 100 keys, different prefixes should (almost certainly) get
        # different indices — collision probability ~1%
        assert idx_a != idx_b

    def test_returns_none_single_key(self):
        assert compute_affinity_index("hash", "prefix", 1) is None

    def test_returns_none_empty_hash(self):
        assert compute_affinity_index("", "prefix", 5) is None

    def test_returns_none_empty_prefix(self):
        assert compute_affinity_index("hash", "", 5) is None


# ---------------------------------------------------------------------------
# ProviderInfo.with_affinity()
# ---------------------------------------------------------------------------


class TestProviderInfoWithAffinity:
    def test_returns_clone(self):
        pi = _make_provider_info(api_key="key-a,key-b,key-c")
        clone = pi.with_affinity(1)
        assert clone is not pi

    def test_shares_key_ring(self):
        pi = _make_provider_info(api_key="key-a,key-b,key-c")
        clone = pi.with_affinity(1)
        assert clone.key_ring is pi.key_ring

    def test_uses_select(self):
        pi = _make_provider_info(api_key="key-a,key-b,key-c")
        clone = pi.with_affinity(1)
        headers = clone.auth_headers()
        assert headers == {"Authorization": "Bearer key-b"}

    def test_none_returns_self(self):
        pi = _make_provider_info(api_key="key-a,key-b")
        result = pi.with_affinity(None)
        assert result is pi

    def test_original_unaffected(self):
        pi = _make_provider_info(api_key="key-a,key-b")
        _clone = pi.with_affinity(1)
        # original should still use round-robin
        assert pi.auth_headers() == {"Authorization": "Bearer key-a"}
        assert pi.auth_headers() == {"Authorization": "Bearer key-b"}

    def test_affinity_is_idempotent(self):
        pi = _make_provider_info(api_key="key-a,key-b,key-c")
        clone = pi.with_affinity(2)
        # calling auth_headers multiple times should return same key
        h1 = clone.auth_headers()
        h2 = clone.auth_headers()
        assert h1 == h2 == {"Authorization": "Bearer key-c"}

    def test_chaining_with_timeout(self):
        pi = _make_provider_info(api_key="key-a,key-b")
        clone = pi.with_affinity(1).with_timeout(30.0)
        assert clone.timeout == 30.0
        assert clone.auth_headers() == {"Authorization": "Bearer key-b"}


# ---------------------------------------------------------------------------
# KeyContext.key_hash
# ---------------------------------------------------------------------------


class TestKeyContextKeyHash:
    def test_default_empty(self):
        from llm_rosetta.gateway.keystore import KeyContext

        ctx = KeyContext(label="test", allowed_shims=frozenset())
        assert ctx.key_hash == ""

    def test_explicit_hash(self):
        from llm_rosetta.gateway.keystore import KeyContext

        ctx = KeyContext(label="test", allowed_shims=frozenset(), key_hash="abc123")
        assert ctx.key_hash == "abc123"
