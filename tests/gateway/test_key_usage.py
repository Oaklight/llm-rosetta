"""Tests for per-key token usage attribution and aggregation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from llm_rosetta.gateway.admin.routes.keys import get_key_usage
from llm_rosetta.gateway.keystore import (
    INTERNAL_LABEL,
    INTERNAL_USAGE_BUCKET,
    KeyContext,
    KeyStore,
    key_bucket,
)
from llm_rosetta.gateway.middleware.auth import api_key_context_var
from llm_rosetta.gateway.ops.base import OpsContext
from llm_rosetta.gateway.ops.proxy import OpsProxyRequest
from llm_rosetta.observability import PersistenceManager, RequestLog
from llm_rosetta.observability.request_log import token_total


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


_ANY = object()


def _group(report, bucket, label=_ANY):
    """The windows of one bucket, optionally narrowed to one label.

    A bucket the caller renamed inside the window comes back as several
    groups, so the default merges them; pass ``label=None`` for the rows that
    carry no label at all.  Returns None when nothing matches.
    """
    merged: dict[str, dict[str, int]] = {}
    for group in report["groups"]:
        if group["bucket"] != bucket:
            continue
        if label is not _ANY and group["label"] != label:
            continue
        for name, counts in group["windows"].items():
            slot = merged.setdefault(name, {})
            for counter, value in counts.items():
                slot[counter] = slot.get(counter, 0) + value
    return merged or None


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
        windows = _group(report, "b1")
        assert windows["5h"]["request_count"] == 2
        assert windows["24h"]["request_count"] == 4
        assert windows["7d"]["request_count"] == 5
        assert windows["30d"]["request_count"] == 6
        assert windows["5h"]["total_tokens"] == 200
        assert windows["30d"]["total_tokens"] == 600
        assert windows["5h"]["input_tokens"] == 120

    async def test_null_bucket_is_reported_not_dropped(self, pm):
        """A row with no bucket is unattributed, not invisible.

        Dropping it here is what made the panel disagree with the dashboard:
        the dashboard sums every row, the panel only the attributed ones.
        """
        await pm.insert_log_entries([_log_row(None, _ago(minutes=1), total=42)])
        report = await pm.query_token_usage_by_key()
        assert _group(report, None)["5h"]["total_tokens"] == 42

    async def test_groups_split_by_label(self, pm):
        """One bucket renamed inside the window comes back as two groups.

        The caller folds them by bucket; the split is what lets it also
        account for rows that no live key owns.
        """
        await pm.insert_log_entries(
            [
                _log_row("b1", _ago(minutes=1), total=100),
                {**_log_row("b1", _ago(minutes=2), total=5), "api_key_label": "new"},
            ]
        )
        report = await pm.query_token_usage_by_key()
        assert _group(report, "b1", None)["5h"]["total_tokens"] == 100
        assert _group(report, "b1", "new")["5h"]["total_tokens"] == 5

    async def test_buckets_are_separated(self, pm):
        await pm.insert_log_entries(
            [
                _log_row("b1", _ago(minutes=1), total=100),
                _log_row("b2", _ago(minutes=1), total=7),
            ]
        )
        report = await pm.query_token_usage_by_key()
        assert _group(report, "b1")["5h"]["total_tokens"] == 100
        assert _group(report, "b2")["5h"]["total_tokens"] == 7


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
        assert _group(report, ctx.usage_bucket) is not None

    async def test_request_without_a_credential_records_no_bucket(self, pm):
        """Unattributed stays NULL, never an empty-string bucket."""
        log = RequestLog(persistence=pm)
        await self._op(log, False).execute()

        entries, _ = await log.get_entries(limit=10)
        assert "api_key_bucket" not in entries[0]
        # Reported under the None bucket rather than dropped, so the panel can
        # still account for it.
        assert _group(await pm.query_token_usage_by_key(), None) is not None

    async def test_internal_token_records_the_sentinel_bucket(self, pm):
        ctx = KeyContext(
            label=INTERNAL_LABEL,
            allowed_shims=frozenset({"*"}),
            key_hash="abc",
            usage_bucket=INTERNAL_USAGE_BUCKET,
        )
        log = RequestLog(persistence=pm)

        token = api_key_context_var.set(ctx)
        try:
            await self._op(log, False).execute()
        finally:
            api_key_context_var.reset(token)

        entries, _ = await log.get_entries(limit=10)
        assert entries[0]["api_key_bucket"] == INTERNAL_USAGE_BUCKET
        report = await pm.query_token_usage_by_key()
        assert _group(report, INTERNAL_USAGE_BUCKET, INTERNAL_LABEL) is not None


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
        assert "shared_label" not in entry

        # A second key with no traffic is present and zero-filled.
        other_id, _ = await keystore.create_key(label="idle")
        resp = await get_key_usage(self._request(keystore, pm))
        data = json.loads(resp.body)
        assert data["keys"][other_id]["windows"]["5h"]["total_tokens"] == 0

    async def test_a_label_two_keys_share_is_flagged(self, keystore, pm):
        """The row can only ever read zero — a log row's label cannot say
        which of the two keys spent the tokens — so the panel says why."""
        first_id, _ = await keystore.create_key(label="dup")
        second_id, _ = await keystore.create_key(label="dup")

        data = json.loads((await get_key_usage(self._request(keystore, pm))).body)

        assert data["keys"][first_id]["shared_label"] is True
        assert data["keys"][second_id]["shared_label"] is True

    async def test_no_persistence_returns_zeroes(self, keystore):
        key_id, _ = await keystore.create_key(label="k")
        resp = await get_key_usage(self._request(keystore, None))
        data = json.loads(resp.body)

        assert data["windows"] == ["5h", "24h", "7d", "30d"]
        assert data["keys"][key_id]["windows"]["5h"]["total_tokens"] == 0

    async def test_orphan_bucket_is_reported_as_unattributed(self, keystore, pm):
        # Traffic recorded under a bucket that has no live key — a deleted or
        # rotated-away secret. It must still be visible, or the panel's total
        # would disagree with the dashboard's.
        labelled = {
            **_log_row("orphan", _ago(minutes=1), total=70),
            "api_key_label": "gone",
        }
        await pm.insert_log_entries(
            [labelled, _log_row("orphan2", _ago(minutes=1), total=30)]
        )

        data = json.loads((await get_key_usage(self._request(keystore, pm))).body)

        assert data["keys"] == {}
        assert data["unattributed"]["windows"]["5h"]["total_tokens"] == 100
        # Split by label so the panel can say whose traffic it was.
        assert set(data["unattributed"]["by_label"]) == {"gone", "(no credential)"}
        assert data["unattributed"]["by_label"]["gone"]["5h"]["total_tokens"] == 70

    async def test_internal_traffic_gets_its_own_row(self, keystore, pm):
        await pm.insert_log_entries(
            [
                _log_row(INTERNAL_USAGE_BUCKET, _ago(minutes=1), total=25),
                _log_row(None, _ago(minutes=1), total=5),
            ]
        )
        # The rows carry the internal label in production; mirror that here.
        await pm._conn.execute("UPDATE request_log SET api_key_label = 'internal'")
        await pm._conn.commit()

        data = json.loads((await get_key_usage(self._request(keystore, pm))).body)

        # Both rows: the one already carrying the sentinel bucket, and the
        # bucketless one whose label says it is the panel's own traffic.
        assert data["internal"]["windows"]["5h"]["total_tokens"] == 30
        assert "unattributed" not in data

    async def test_a_live_key_named_internal_keeps_its_own_traffic(self, keystore, pm):
        """Historical rows labelled "internal" belong to that key, not the token."""
        key_id, raw = await keystore.create_key(label="internal")
        await pm.insert_log_entries(
            [
                _log_row(None, _ago(minutes=1), total=9),
                _log_row(bucket_of(raw), _ago(minutes=1), total=40),
            ]
        )

        data = json.loads((await get_key_usage(self._request(keystore, pm))).body)

        entry = data["keys"][key_id]
        assert entry["windows"]["5h"]["total_tokens"] == 40
        assert "internal" not in data
        assert data["unattributed"]["windows"]["5h"]["total_tokens"] == 9

    async def test_rows_add_up_to_the_gateway_total(self, keystore, pm):
        """Live keys + internal + unattributed == every logged row."""
        live_id, raw = await keystore.create_key(label="live")
        await pm.insert_log_entries(
            [
                _log_row(bucket_of(raw), _ago(minutes=1), total=11),
                _log_row(INTERNAL_USAGE_BUCKET, _ago(minutes=1), total=22),
                _log_row("orphan", _ago(minutes=1), total=33),
                _log_row(None, _ago(minutes=1), total=44),
            ]
        )

        data = json.loads((await get_key_usage(self._request(keystore, pm))).body)
        rows = [data["keys"][live_id], data["internal"], data["unattributed"]]
        assert sum(r["windows"]["5h"]["total_tokens"] for r in rows) == 110


class TestTokenTotal:
    """The one rule that fills `total_tokens`, wherever usage comes from."""

    def test_provider_total_wins(self):
        assert (
            token_total(
                {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 99}
            )
            == 99
        )

    def test_falls_back_to_the_additive_sum(self):
        """Anthropic sends no total, and its cache tokens are additive."""
        assert (
            token_total(
                {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "cache_read_tokens": 100,
                    "cache_creation_tokens": 7,
                }
            )
            == 122
        )

    def test_reasoning_is_never_added(self):
        """It rides inside the output tokens."""
        assert (
            token_total(
                {"prompt_tokens": 10, "completion_tokens": 50, "reasoning_tokens": 40}
            )
            == 60
        )

    def test_no_usage_at_all(self):
        assert token_total(None) is None
        assert token_total({}) is None
        assert token_total({"model": "gpt-4o"}) is None

    def test_partial_usage_still_totals(self):
        assert token_total({"completion_tokens": 5}) == 5


class TestBackfillUsageBuckets:
    async def test_attributes_rows_by_label(self, pm):
        await pm.insert_log_entries(
            [
                {**_log_row(None, _ago(minutes=1), total=10), "api_key_label": "prod"},
                {**_log_row(None, _ago(minutes=1), total=20), "api_key_label": "prod"},
                {**_log_row(None, _ago(minutes=1), total=30), "api_key_label": "other"},
            ]
        )

        updated = await pm.backfill_usage_buckets({"prod": "b-prod"})

        assert updated == 2
        report = await pm.query_token_usage_by_key()
        assert _group(report, "b-prod", "prod")["5h"]["total_tokens"] == 30
        # The label nobody claimed is left alone.
        assert _group(report, None, "other")["5h"]["total_tokens"] == 30

    async def test_never_touches_an_attributed_row(self, pm):
        await pm.insert_log_entries(
            [
                {
                    **_log_row("b-owned", _ago(minutes=1), total=10),
                    "api_key_label": "prod",
                },
                {**_log_row(None, _ago(minutes=1), total=20), "api_key_label": "prod"},
            ]
        )

        assert await pm.backfill_usage_buckets({"prod": "b-prod"}) == 1
        report = await pm.query_token_usage_by_key()
        assert _group(report, "b-owned", "prod")["5h"]["total_tokens"] == 10

    async def test_idempotent(self, pm):
        await pm.insert_log_entries(
            [{**_log_row(None, _ago(minutes=1), total=10), "api_key_label": "prod"}]
        )
        assert await pm.backfill_usage_buckets({"prod": "b-prod"}) == 1
        assert await pm.backfill_usage_buckets({"prod": "b-prod"}) == 0

    async def test_empty_map_is_a_noop(self, pm):
        await pm.insert_log_entries([_log_row(None, _ago(minutes=1))])
        assert await pm.backfill_usage_buckets({}) == 0


class TestLabelBucketMap:
    async def test_maps_each_label(self, keystore):
        _, raw = await keystore.create_key(label="prod")
        mapping = await keystore.label_bucket_map()
        assert mapping["prod"] == bucket_of(raw)

    async def test_drops_a_label_two_keys_share(self, keystore):
        """Two credentials behind one label cannot say which one spent."""
        await keystore.create_key(label="dup")
        await keystore.create_key(label="dup")
        assert "dup" not in await keystore.label_bucket_map()

    async def test_attributes_the_internal_label_to_the_panel_token(self, keystore):
        assert (await keystore.label_bucket_map())[
            INTERNAL_LABEL
        ] == INTERNAL_USAGE_BUCKET

    async def test_a_real_key_keeps_the_internal_label(self, keystore):
        """A key literally named "internal" owns its label, so rows labelled
        with it are not folded into the admin token's bucket."""
        _, raw = await keystore.create_key(label=INTERNAL_LABEL)
        assert (await keystore.label_bucket_map())[INTERNAL_LABEL] == bucket_of(raw)

    async def test_skips_keys_with_no_label(self, keystore):
        await keystore.create_key(label="")
        assert "" not in await keystore.label_bucket_map()


class TestRolling24hTotal:
    async def test_total_tokens_ride_along(self, pm):
        await pm.insert_log_entries(
            [
                {
                    **_log_row("b1", _ago(hours=1), total=100, inp=60),
                    "cache_read_tokens": 400,
                },
                _log_row("b1", _ago(hours=30), total=999),
            ]
        )

        snap = await pm.query_rolling_24h_tokens()

        # The canonical figure is the stored column, not a re-sum of the
        # subset columns — 400 cache tokens behind 100 total must not inflate
        # it past what the provider reported.
        assert snap["total_tokens_24h"] == 100
        assert snap["input_tokens_24h"] == 60
        assert snap["cache_read_tokens_24h"] == 400
