"""Tests for per-key token usage attribution and aggregation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from llm_rosetta.gateway.admin.routes.keys import get_key_usage
from llm_rosetta.gateway.keystore import KeyContext, KeyStore, key_bucket
from llm_rosetta.gateway.middleware.auth import api_key_context_var
from llm_rosetta.gateway.ops.base import OpsContext
from llm_rosetta.gateway.ops.proxy import OpsProxyRequest
from llm_rosetta.observability import PersistenceManager, RequestLog


def bucket_of(raw_key: str) -> str:
    """The bucket a raw key must map to."""
    return key_bucket(hashlib.sha256(raw_key.encode()).hexdigest())


@pytest.fixture()
async def keystore(tmp_path):
    ks = await KeyStore.create(tmp_path / "keys.db")
    yield ks
    await ks.close()


@pytest.fixture()
async def pm(tmp_path):
    manager = await PersistenceManager.create(str(tmp_path / "gateway.db"))
    yield manager
    await manager.close()


def _log_row(bucket: str | None, ts_iso: str, total: int = 100, inp: int = 60):
    """A minimal request_log row dict with a fixed bucket and timestamp."""
    return {
        "id": f"id-{bucket}-{ts_iso}",
        "timestamp": ts_iso,
        "model": "m",
        "source_provider": "s",
        "target_provider": "t",
        "is_stream": False,
        "status_code": 200,
        "duration_ms": 1.0,
        "api_key_bucket": bucket,
        "input_tokens": inp,
        "output_tokens": total - inp,
        "total_tokens": total,
    }


def _ago(**kwargs) -> str:
    return (datetime.now(timezone.utc) - timedelta(**kwargs)).isoformat()


class TestKeyBucket:
    def test_deterministic(self):
        assert key_bucket("abc") == key_bucket("abc")

    def test_distinct_hashes_differ(self):
        assert key_bucket("abc") != key_bucket("xyz")

    def test_is_full_uuid_hex(self):
        """The full uuid5 hex, not a prefix — the bucket is a grouping key."""
        bucket = key_bucket("abc")
        assert len(bucket) == 32
        assert all(c in "0123456789abcdef" for c in bucket)

    async def test_create_stores_bucket_on_context(self, keystore):
        _, raw = await keystore.create_key(label="a")
        result = keystore.validate(raw)
        assert result is not None
        _, ctx = result
        assert ctx.usage_bucket == bucket_of(raw)

    async def test_rotate_derives_bucket_from_the_new_secret(self, keystore):
        key_id, raw = await keystore.create_key(label="a")
        before = keystore.validate(raw)[1].usage_bucket

        new_raw = await keystore.rotate(key_id)
        after = keystore.validate(new_raw)[1].usage_bucket

        # Pins the derivation, not merely "something different" — a fresh
        # uuid4 or a constant would satisfy `!=` alone.
        assert after != before
        assert after == bucket_of(new_raw)
        assert keystore.validate(raw) is None  # old secret is gone

    async def test_usage_keys_shape(self, keystore):
        key_id, raw = await keystore.create_key(label="lbl")
        assert await keystore.usage_keys() == [
            {"id": key_id, "label": "lbl", "usage_bucket": bucket_of(raw)}
        ]

    async def test_migration_backfills_legacy_rows(self, tmp_path):
        db_path = tmp_path / "keys.db"
        ks = await KeyStore.create(db_path)
        key_id, raw = await ks.create_key(label="legacy")

        # Simulate a pre-migration row: hash present, bucket missing.
        await ks._conn.execute(
            "UPDATE api_keys SET usage_bucket = NULL WHERE id = ?", (key_id,)
        )
        await ks._conn.commit()
        await ks.close()

        reopened = await KeyStore.create(db_path)
        assert await reopened.usage_keys() == [
            {"id": key_id, "label": "legacy", "usage_bucket": bucket_of(raw)}
        ]
        await reopened.close()


class TestQueryTokenUsageByKey:
    async def test_window_boundaries_and_sums(self, pm):
        # Ages straddle every cutoff: 4h/10h separate 5h, 20h/3d separate
        # 24h, 3d/20d separate 7d, 20d/40d separate 30d.
        await pm.insert_log_entries(
            [
                _log_row("b1", _ago(minutes=10)),  # 5h, 24h, 7d, 30d
                _log_row("b1", _ago(hours=4)),  # 5h, 24h, 7d, 30d
                _log_row("b1", _ago(hours=10)),  # 24h, 7d, 30d
                _log_row("b1", _ago(hours=20)),  # 24h, 7d, 30d
                _log_row("b1", _ago(days=3)),  # 7d, 30d
                _log_row("b1", _ago(days=20)),  # 30d only
                _log_row("b1", _ago(days=40)),  # outside every window
            ]
        )
        report = await pm.query_token_usage_by_key()

        assert report["windows"] == ["5h", "24h", "7d", "30d"]
        windows = report["keys"]["b1"]
        assert windows["5h"]["request_count"] == 2
        assert windows["24h"]["request_count"] == 4
        assert windows["7d"]["request_count"] == 5
        assert windows["30d"]["request_count"] == 6
        assert windows["5h"]["total_tokens"] == 200
        assert windows["30d"]["total_tokens"] == 600
        assert windows["5h"]["input_tokens"] == 120

    async def test_null_bucket_excluded(self, pm):
        await pm.insert_log_entries([_log_row(None, _ago(minutes=1))])
        report = await pm.query_token_usage_by_key()
        assert report["keys"] == {}

    async def test_buckets_are_separated(self, pm):
        await pm.insert_log_entries(
            [
                _log_row("b1", _ago(minutes=1), total=100),
                _log_row("b2", _ago(minutes=1), total=7),
            ]
        )
        report = await pm.query_token_usage_by_key()
        assert report["keys"]["b1"]["5h"]["total_tokens"] == 100
        assert report["keys"]["b2"]["5h"]["total_tokens"] == 7


class TestProxyWiring:
    """The one line that carries the credential bucket into the log row.

    Without it the column stays NULL, `query_token_usage_by_key` returns no
    buckets, and every panel figure silently reads zero.
    """

    @staticmethod
    def _op(log, is_stream):
        return OpsProxyRequest(
            OpsContext(request_log=log),
            model="m",
            source_provider="s",
            target_provider="t",
            provider_name="dummy",
            is_stream=is_stream,
            status_code=200,
            duration_ms=1.0,
            error_detail=None,
        )

    @pytest.mark.parametrize("is_stream", [False, True])
    async def test_records_the_bucket_on_the_log_row(self, keystore, pm, is_stream):
        _, raw = await keystore.create_key(label="wired")
        _, ctx = keystore.validate(raw)
        log = RequestLog(persistence=pm)

        token = api_key_context_var.set(ctx)
        try:
            await self._op(log, is_stream).execute()
        finally:
            api_key_context_var.reset(token)

        entries, _ = await log.get_entries(limit=10)
        assert entries[0]["api_key_bucket"] == ctx.usage_bucket
        assert entries[0]["api_key_label"] == "wired"
        assert entries[0]["is_stream"] is is_stream

        # And the report groups on it.
        report = await pm.query_token_usage_by_key()
        assert ctx.usage_bucket in report["keys"]

    async def test_anonymous_request_records_no_bucket(self, pm):
        log = RequestLog(persistence=pm)
        await self._op(log, False).execute()

        entries, _ = await log.get_entries(limit=10)
        assert "api_key_bucket" not in entries[0]
        assert (await pm.query_token_usage_by_key())["keys"] == {}

    async def test_internal_token_records_no_bucket(self, pm):
        """Unattributed is NULL, never an empty-string bucket."""
        ctx = KeyContext(
            label="internal", allowed_shims=frozenset({"*"}), key_hash="abc"
        )
        assert ctx.usage_bucket is None
        log = RequestLog(persistence=pm)

        token = api_key_context_var.set(ctx)
        try:
            await self._op(log, False).execute()
        finally:
            api_key_context_var.reset(token)

        entries, _ = await log.get_entries(limit=10)
        assert "api_key_bucket" not in entries[0]
        assert (await pm.query_token_usage_by_key())["keys"] == {}


class TestGetKeyUsageRoute:
    @staticmethod
    def _request(keystore, persistence):
        app = SimpleNamespace(keystore=keystore, persistence=persistence)
        return SimpleNamespace(app=app)

    async def test_zero_fills_keys_without_traffic(self, keystore, pm):
        key_id, raw = await keystore.create_key(label="k")
        await pm.insert_log_entries(
            [_log_row(bucket_of(raw), _ago(minutes=1), total=50)]
        )

        resp = await get_key_usage(self._request(keystore, pm))
        data = json.loads(resp.body)

        assert data["windows"] == ["5h", "24h", "7d", "30d"]
        entry = data["keys"][key_id]
        assert entry["label"] == "k"
        assert entry["windows"]["5h"]["total_tokens"] == 50
        assert entry["windows"]["24h"]["request_count"] == 1

        # A second key with no traffic is present and zero-filled.
        other_id, _ = await keystore.create_key(label="idle")
        resp = await get_key_usage(self._request(keystore, pm))
        data = json.loads(resp.body)
        assert data["keys"][other_id]["windows"]["5h"]["total_tokens"] == 0

    async def test_no_persistence_returns_zeroes(self, keystore):
        key_id, _ = await keystore.create_key(label="k")
        resp = await get_key_usage(self._request(keystore, None))
        data = json.loads(resp.body)

        assert data["windows"] == ["5h", "24h", "7d", "30d"]
        assert data["keys"][key_id]["windows"]["5h"]["total_tokens"] == 0

    async def test_orphan_bucket_is_dropped(self, keystore, pm):
        # Traffic recorded under a bucket that has no live key.
        await pm.insert_log_entries([_log_row("orphan", _ago(minutes=1))])
        resp = await get_key_usage(self._request(keystore, pm))
        data = json.loads(resp.body)
        assert data["keys"] == {}
