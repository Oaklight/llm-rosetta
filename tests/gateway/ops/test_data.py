"""Tests for data mutation Ops subclasses."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_rosetta.gateway.ops.base import OpsContext
from llm_rosetta.gateway.ops.data import (
    OpsClearData,
    OpsClearOpsLog,
    OpsCleanupData,
    OpsDeleteErrorDump,
    OpsRebuildMetrics,
    OpsTrimData,
    OpsVacuum,
)
from llm_rosetta.observability.ops_log import (
    EVENT_DATA_CLEARED,
    EVENT_DATA_CLEANUP,
    EVENT_DATA_REBUILT,
    EVENT_DATA_TRIMMED,
    EVENT_DATA_VACUUMED,
    EVENT_OPS_LOG_CLEARED,
    SEVERITY_WARNING,
)


def _make_ctx(
    *,
    persistence: Any = None,
    ops_log: Any = None,
    metrics: Any = None,
    request_log: Any = None,
) -> OpsContext:
    if ops_log is None:
        ops_log = MagicMock()
        ops_log.add = AsyncMock()
    return OpsContext(
        ops_log=ops_log,
        persistence=persistence,
        metrics=metrics,
        request_log=request_log,
    )


def _mock_persistence() -> MagicMock:
    p = MagicMock()
    p.clear_log = AsyncMock()
    p.clear_error_dumps = AsyncMock()
    p.delete_error_dump = AsyncMock(return_value=True)
    p.cleanup_by_age = AsyncMock(return_value={"deleted": 10})
    p.cleanup_logs_by_age = AsyncMock(return_value={"deleted": 5})
    p.cleanup_error_dumps_by_age = AsyncMock(return_value={"error_dumps_deleted": 3})
    p.cleanup_ops_log_by_age = AsyncMock(return_value={"deleted": 2})
    p.cleanup_before = AsyncMock(return_value={"deleted": 7})
    p.cleanup_range = AsyncMock(return_value={"deleted": 4})
    p.trim_to_cap = AsyncMock(return_value={"trimmed": 15})
    p.vacuum = AsyncMock(return_value={"freed_bytes": 1024})
    p.count_log_entries = AsyncMock(return_value=100)
    p.iter_log_rows_for_rebuild = _mock_iter_rows
    p.save_metrics = AsyncMock()
    return p


async def _mock_iter_rows():
    for row in [{"status_code": 200}, {"status_code": 500}]:
        yield row


class TestOpsClearData:
    @pytest.mark.asyncio
    async def test_clear_request_log(self):
        p = _mock_persistence()
        m = MagicMock()
        m.rebuild_counters = MagicMock(return_value=2)
        m.export_counters = MagicMock(return_value={"total": 2})
        rl = MagicMock()
        rl.clear = AsyncMock()
        ctx = _make_ctx(persistence=p, metrics=m, request_log=rl)

        result = await OpsClearData(ctx, table="request_log").execute()

        assert result["deleted"] == 100
        rl.clear.assert_awaited_once()
        # Counter rebuild triggered
        m.rebuild_counters.assert_called_once()
        p.save_metrics.assert_awaited_once()
        # ops_log entry written
        entry = ctx.ops_log.add.call_args[0][0]
        assert entry.event_type == EVENT_DATA_CLEARED
        assert entry.severity == SEVERITY_WARNING
        assert "request_log" in entry.message

    @pytest.mark.asyncio
    async def test_clear_error_dumps(self):
        p = _mock_persistence()
        ctx = _make_ctx(persistence=p)

        await OpsClearData(ctx, table="error_dumps").execute()

        p.clear_error_dumps.assert_awaited_once()
        entry = ctx.ops_log.add.call_args[0][0]
        assert entry.event_type == EVENT_DATA_CLEARED
        assert "error_dumps" in entry.details["table"]

    @pytest.mark.asyncio
    async def test_no_persistence(self):
        ctx = _make_ctx()
        result = await OpsClearData(ctx, table="request_log").execute()
        assert result == {}


class TestOpsDeleteErrorDump:
    @pytest.mark.asyncio
    async def test_delete_found(self):
        p = _mock_persistence()
        ctx = _make_ctx(persistence=p)

        found = await OpsDeleteErrorDump(ctx, dump_id="abc123").execute()

        assert found is True
        p.delete_error_dump.assert_awaited_once_with("abc123")
        entry = ctx.ops_log.add.call_args[0][0]
        assert entry.event_type == EVENT_DATA_CLEARED
        assert entry.details["dump_id"] == "abc123"
        assert entry.details["found"] is True

    @pytest.mark.asyncio
    async def test_delete_not_found(self):
        p = _mock_persistence()
        p.delete_error_dump = AsyncMock(return_value=False)
        ctx = _make_ctx(persistence=p)

        found = await OpsDeleteErrorDump(ctx, dump_id="missing").execute()

        assert found is False
        entry = ctx.ops_log.add.call_args[0][0]
        assert entry.details["found"] is False


class TestOpsCleanupData:
    @pytest.mark.asyncio
    async def test_cleanup_by_age_all(self):
        p = _mock_persistence()
        m = MagicMock()
        m.rebuild_counters = MagicMock(return_value=2)
        m.export_counters = MagicMock(return_value={})
        ctx = _make_ctx(persistence=p, metrics=m)

        result = await OpsCleanupData(ctx, mode="age", max_age_days=30).execute()

        assert result["deleted"] == 10
        p.cleanup_by_age.assert_awaited_once_with(30)
        entry = ctx.ops_log.add.call_args[0][0]
        assert entry.event_type == EVENT_DATA_CLEANUP
        assert entry.details["max_age_days"] == 30

    @pytest.mark.asyncio
    async def test_cleanup_request_log_only(self):
        p = _mock_persistence()
        m = MagicMock()
        m.rebuild_counters = MagicMock(return_value=1)
        m.export_counters = MagicMock(return_value={})
        ctx = _make_ctx(persistence=p, metrics=m)

        await OpsCleanupData(
            ctx, mode="age", tables=("request_log",), max_age_days=7
        ).execute()

        p.cleanup_logs_by_age.assert_awaited_once_with(7)
        m.rebuild_counters.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_error_dumps_only(self):
        p = _mock_persistence()
        ctx = _make_ctx(persistence=p)

        await OpsCleanupData(
            ctx, mode="age", tables=("error_dumps",), max_age_days=14
        ).execute()

        p.cleanup_error_dumps_by_age.assert_awaited_once_with(14)

    @pytest.mark.asyncio
    async def test_cleanup_before(self):
        p = _mock_persistence()
        ctx = _make_ctx(persistence=p)

        await OpsCleanupData(
            ctx,
            mode="before",
            tables=("error_dumps",),
            before="2026-01-01T00:00:00Z",
        ).execute()

        p.cleanup_before.assert_awaited_once_with(
            "2026-01-01T00:00:00Z", tables=("error_dumps",)
        )
        entry = ctx.ops_log.add.call_args[0][0]
        assert entry.details["before"] == "2026-01-01T00:00:00Z"

    @pytest.mark.asyncio
    async def test_cleanup_range(self):
        p = _mock_persistence()
        ctx = _make_ctx(persistence=p)

        await OpsCleanupData(
            ctx,
            mode="range",
            tables=("request_log", "error_dumps"),
            start="2026-01-01",
            end="2026-02-01",
        ).execute()

        p.cleanup_range.assert_awaited_once()
        entry = ctx.ops_log.add.call_args[0][0]
        assert entry.details["start"] == "2026-01-01"
        assert entry.details["end"] == "2026-02-01"


class TestOpsTrimData:
    @pytest.mark.asyncio
    async def test_trim(self):
        p = _mock_persistence()
        m = MagicMock()
        m.rebuild_counters = MagicMock(return_value=2)
        m.export_counters = MagicMock(return_value={})
        ctx = _make_ctx(persistence=p, metrics=m)

        result = await OpsTrimData(ctx).execute()

        assert result["trimmed"] == 15
        p.trim_to_cap.assert_awaited_once()
        entry = ctx.ops_log.add.call_args[0][0]
        assert entry.event_type == EVENT_DATA_TRIMMED


class TestOpsVacuum:
    @pytest.mark.asyncio
    async def test_vacuum(self):
        p = _mock_persistence()
        ctx = _make_ctx(persistence=p)

        result = await OpsVacuum(ctx).execute()

        assert result["freed_bytes"] == 1024
        p.vacuum.assert_awaited_once()
        entry = ctx.ops_log.add.call_args[0][0]
        assert entry.event_type == EVENT_DATA_VACUUMED


class TestOpsRebuildMetrics:
    @pytest.mark.asyncio
    async def test_rebuild(self):
        p = _mock_persistence()
        m = MagicMock()
        m.export_counters = MagicMock(side_effect=[{"before": True}, {"after": True}])
        m.rebuild_counters = MagicMock(return_value=2)
        ctx = _make_ctx(persistence=p, metrics=m)

        result = await OpsRebuildMetrics(ctx).execute()

        assert result["rebuilt_from"] == 2
        assert "before" in result
        assert "counters" in result
        entry = ctx.ops_log.add.call_args[0][0]
        assert entry.event_type == EVENT_DATA_REBUILT


class TestOpsClearOpsLog:
    @pytest.mark.asyncio
    async def test_clear(self):
        ops_log = MagicMock()
        ops_log.clear = AsyncMock(return_value=42)
        ops_log.add = AsyncMock()
        ctx = OpsContext(ops_log=ops_log)

        result = await OpsClearOpsLog(ctx).execute()

        assert result["cleared"] == 42
        ops_log.clear.assert_awaited_once()
        entry = ops_log.add.call_args[0][0]
        assert entry.event_type == EVENT_OPS_LOG_CLEARED
        assert entry.details["cleared_count"] == 42


class TestOpsPrune:
    @pytest.mark.asyncio
    async def test_prune_records_event(self):
        ops_log = MagicMock()
        ops_log.add = AsyncMock()
        ctx = OpsContext(ops_log=ops_log)

        from llm_rosetta.gateway.ops.data import OpsPrune

        await OpsPrune(ctx, table="request_log", count=50).execute()

        entry = ops_log.add.call_args[0][0]
        assert entry.event_type == EVENT_DATA_TRIMMED
        assert entry.source == "persistence"
        assert entry.details["table"] == "request_log"
        assert entry.details["pruned"] == 50
        assert "50" in entry.message

    @pytest.mark.asyncio
    async def test_prune_no_ops_log(self):
        from llm_rosetta.gateway.ops.data import OpsPrune

        ctx = OpsContext()
        await OpsPrune(ctx, table="error_dumps", count=10).execute()


class TestOpsPeriodicCleanup:
    @pytest.mark.asyncio
    async def test_periodic_cleanup_with_deletions(self):
        from llm_rosetta.gateway.ops.data import OpsPeriodicCleanup

        p = _mock_persistence()
        m = MagicMock()
        m.rebuild_counters = MagicMock(return_value=2)
        m.export_counters = MagicMock(return_value={})
        ops_log = MagicMock()
        ops_log.add = AsyncMock()
        ctx = OpsContext(ops_log=ops_log, persistence=p, metrics=m)

        result = await OpsPeriodicCleanup(ctx, rl_age=30, ol_age=60).execute()

        assert result["total"] > 0
        p.cleanup_logs_by_age.assert_awaited_once_with(30)
        p.cleanup_error_dumps_by_age.assert_awaited_once_with(30)
        p.cleanup_ops_log_by_age.assert_awaited_once_with(60)
        # ops_log entry written because total > 0
        ops_log.add.assert_called_once()
        entry = ops_log.add.call_args[0][0]
        assert entry.event_type == EVENT_DATA_CLEANUP
        assert entry.source == "persistence"

    @pytest.mark.asyncio
    async def test_periodic_cleanup_zero_deletions_no_record(self):
        from llm_rosetta.gateway.ops.data import OpsPeriodicCleanup

        p = MagicMock()
        p.cleanup_logs_by_age = AsyncMock(return_value={"deleted": 0})
        p.cleanup_error_dumps_by_age = AsyncMock(
            return_value={"error_dumps_deleted": 0}
        )
        p.cleanup_ops_log_by_age = AsyncMock(return_value={"deleted": 0})
        ops_log = MagicMock()
        ops_log.add = AsyncMock()
        ctx = OpsContext(ops_log=ops_log, persistence=p)

        result = await OpsPeriodicCleanup(ctx, rl_age=90, ol_age=90).execute()

        assert result["total"] == 0
        ops_log.add.assert_not_called()
