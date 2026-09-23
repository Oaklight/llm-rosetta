"""Tests for API key affinity — deterministic key selection for cache locality.

Covers:
- KeyRing.select() — rendezvous hashing without advancing round-robin
- KeyRing stability on refresh — minimal disruption when keys change
- extract_prefix_from_ir() — stable fingerprint from IR request
- compute_affinity_identity() — identity string for rendezvous hashing
- ProviderInfo.with_affinity() — shallow-copy with pinned identity
- KeyContext.key_hash — client identity field
"""

from __future__ import annotations

import pytest

from llm_rosetta.gateway.middleware.affinity import (
    compute_affinity_identity,
    extract_prefix_from_ir,
)
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
# KeyRing.select() — rendezvous hashing
# ---------------------------------------------------------------------------


class TestKeyRingSelect:
    def test_deterministic(self):
        ring = KeyRing("key-a,key-b,key-c")
        result1 = ring.select("identity-1")
        result2 = ring.select("identity-1")
        assert result1 == result2

    def test_returns_valid_key(self):
        keys = ["key-a", "key-b", "key-c"]
        ring = KeyRing(",".join(keys))
        for i in range(20):
            assert ring.select(f"identity-{i}") in keys

    def test_spreads_across_keys(self):
        ring = KeyRing("key-a,key-b,key-c")
        selected = {ring.select(f"identity-{i}") for i in range(100)}
        assert len(selected) == 3

    def test_does_not_affect_round_robin(self):
        ring = KeyRing("key-a,key-b")
        ring.select("some-identity")
        ring.select("other-identity")
        assert ring.next() == "key-a"
        assert ring.next() == "key-b"

    def test_single_key(self):
        ring = KeyRing("only-key")
        assert ring.select("identity-1") == "only-key"
        assert ring.select("identity-999") == "only-key"

    def test_empty_raises(self):
        ring = KeyRing("")
        with pytest.raises(ValueError):
            ring.select("identity")

    def test_order_independent(self):
        """Same keys in different order produce the same selection."""
        ring_abc = KeyRing("key-a,key-b,key-c")
        ring_cab = KeyRing("key-c,key-a,key-b")
        for i in range(20):
            identity = f"test-{i}"
            assert ring_abc.select(identity) == ring_cab.select(identity)


# ---------------------------------------------------------------------------
# KeyRing stability on refresh — core acceptance tests
# ---------------------------------------------------------------------------


class TestKeyRingStabilityOnRefresh:
    def test_same_keys_no_change(self):
        """Refreshing with the same keys changes zero mappings."""
        ring = KeyRing("key-a,key-b,key-c,key-d")
        identities = [f"id-{i}" for i in range(200)]

        before = [ring.select(ident) for ident in identities]
        ring.refresh("key-a,key-b,key-c,key-d")
        after = [ring.select(ident) for ident in identities]

        assert before == after

    def test_add_key_minimal_disruption(self):
        """Adding one key to 4 disrupts at most ~40% of mappings (expected ~20%)."""
        ring = KeyRing("key-a,key-b,key-c,key-d")
        identities = [f"id-{i}" for i in range(300)]

        before = [ring.select(ident) for ident in identities]
        ring.refresh("key-a,key-b,key-c,key-d,key-e")
        after = [ring.select(ident) for ident in identities]

        changed = sum(1 for b, a in zip(before, after) if b != a)
        # Expected ~1/5 = 60 out of 300.  Allow up to 40% (120) for statistical margin.
        assert changed <= 120, f"Too many mappings changed: {changed}/300"
        # Sanity: at least a few should change (the new key should win some)
        assert changed > 0, "No mappings changed — new key never wins"

    def test_remove_key_only_affects_removed(self):
        """Removing a key only redistributes that key's mappings."""
        ring = KeyRing("key-a,key-b,key-c,key-d")
        identities = [f"id-{i}" for i in range(300)]

        before = [ring.select(ident) for ident in identities]
        ring.refresh("key-a,key-b,key-d")  # remove key-c
        after = [ring.select(ident) for ident in identities]

        for b, a in zip(before, after):
            if b != "key-c":
                assert a == b, (
                    f"Mapping changed from {b} to {a} (only key-c removals should shift)"
                )

    def test_reorder_no_change(self):
        """Key order in the CSV doesn't affect selection."""
        ring = KeyRing("key-a,key-b,key-c")
        identities = [f"id-{i}" for i in range(200)]

        before = [ring.select(ident) for ident in identities]
        ring.refresh("key-c,key-a,key-b")
        after = [ring.select(ident) for ident in identities]

        assert before == after


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
# compute_affinity_identity()
# ---------------------------------------------------------------------------


class TestComputeAffinityIdentity:
    def test_returns_string(self):
        result = compute_affinity_identity("hash123", "prefix")
        assert isinstance(result, str)

    def test_deterministic(self):
        r1 = compute_affinity_identity("hash123", "prefix")
        r2 = compute_affinity_identity("hash123", "prefix")
        assert r1 == r2

    def test_different_clients_different_identities(self):
        r_a = compute_affinity_identity("client-a", "same-prefix")
        r_b = compute_affinity_identity("client-b", "same-prefix")
        assert r_a != r_b

    def test_different_prefixes_different_identities(self):
        r_a = compute_affinity_identity("same-client", "prefix-alpha")
        r_b = compute_affinity_identity("same-client", "prefix-beta")
        assert r_a != r_b

    def test_returns_none_empty_hash(self):
        assert compute_affinity_identity("", "prefix") is None

    def test_returns_none_empty_prefix(self):
        assert compute_affinity_identity("hash", "") is None


# ---------------------------------------------------------------------------
# ProviderInfo.with_affinity()
# ---------------------------------------------------------------------------


class TestProviderInfoWithAffinity:
    def test_returns_clone(self):
        pi = _make_provider_info(api_key="key-a,key-b,key-c")
        clone = pi.with_affinity("test-identity")
        assert clone is not pi

    def test_shares_key_ring(self):
        pi = _make_provider_info(api_key="key-a,key-b,key-c")
        clone = pi.with_affinity("test-identity")
        assert clone.key_ring is pi.key_ring

    def test_uses_rendezvous_select(self):
        pi = _make_provider_info(api_key="key-a,key-b,key-c")
        clone = pi.with_affinity("test-identity")
        headers = clone.auth_headers()
        selected_key = headers["Authorization"].removeprefix("Bearer ")
        assert selected_key in {"key-a", "key-b", "key-c"}

    def test_none_returns_self(self):
        pi = _make_provider_info(api_key="key-a,key-b")
        result = pi.with_affinity(None)
        assert result is pi

    def test_empty_returns_self(self):
        pi = _make_provider_info(api_key="key-a,key-b")
        result = pi.with_affinity("")
        assert result is pi

    def test_original_unaffected(self):
        pi = _make_provider_info(api_key="key-a,key-b")
        _clone = pi.with_affinity("test-identity")
        # original should still use round-robin
        assert pi.auth_headers() == {"Authorization": "Bearer key-a"}
        assert pi.auth_headers() == {"Authorization": "Bearer key-b"}

    def test_affinity_is_idempotent(self):
        pi = _make_provider_info(api_key="key-a,key-b,key-c")
        clone = pi.with_affinity("test-identity")
        h1 = clone.auth_headers()
        h2 = clone.auth_headers()
        assert h1 == h2

    def test_chaining_with_timeout(self):
        pi = _make_provider_info(api_key="key-a,key-b")
        clone = pi.with_affinity("test-identity").with_timeout(30.0)
        assert clone.timeout == 30.0
        selected_key = clone.auth_headers()["Authorization"].removeprefix("Bearer ")
        assert selected_key in {"key-a", "key-b"}


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
