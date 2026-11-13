"""The data-mutation handlers: physical work for the destructive operations.

Each handler runs the matching :mod:`~llm_rosetta_gateway.ops.data` class with
its self as the job context, so the audit record, its event type and its
message stay exactly what the inline path wrote; only the scheduling changed.

The handlers are built from an ops context rather than importing one, because
the context is per-app state (stores, metrics, the ops log) and a module-level
singleton would tie two apps — or a test and the app — to one another.
"""

from __future__ import annotations

from typing import Any

from ..registry import Handler


def data_handlers(ops_ctx: Any) -> dict[str, Handler]:
    """Bind the data operations to *ops_ctx* and return kind -> handler."""
    from ...ops.data import (
        OpsCleanupData,
        OpsClearData,
        OpsClearOpsLog,
        OpsRebuildMetrics,
        OpsTrimData,
        OpsVacuum,
    )

    async def clear(ctx: Any) -> dict[str, Any]:
        table = ctx.job.payload.get("table", "request_log")
        if table == "ops_log":
            return await OpsClearOpsLog(ops_ctx, job=ctx).execute()
        return await OpsClearData(ops_ctx, table=table, job=ctx).execute()

    async def cleanup(ctx: Any) -> dict[str, Any]:
        payload = ctx.job.payload
        return await OpsCleanupData(
            ops_ctx,
            mode=payload.get("mode", "age"),
            tables=tuple(payload.get("tables") or ()),
            max_age_days=payload.get("max_age_days"),
            before=payload.get("before"),
            start=payload.get("start"),
            end=payload.get("end"),
            cutoff=payload.get("cutoff"),
            job=ctx,
        ).execute()

    async def trim(ctx: Any) -> dict[str, Any]:
        return await OpsTrimData(ops_ctx, job=ctx).execute()

    async def vacuum(ctx: Any) -> dict[str, Any]:
        return await OpsVacuum(ops_ctx, job=ctx).execute()

    async def rebuild(ctx: Any) -> dict[str, Any]:
        return await OpsRebuildMetrics(ops_ctx, job=ctx).execute()

    return {
        "data.clear": clear,
        "data.cleanup": cleanup,
        "data.trim": trim,
        "data.vacuum": vacuum,
        "data.rebuild": rebuild,
    }
