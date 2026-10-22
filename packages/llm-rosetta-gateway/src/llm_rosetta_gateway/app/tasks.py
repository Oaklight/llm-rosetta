"""Periodic background tasks: token rotation, cleanup and metric flush."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from llm_rosetta._compat import sync_compat as _sync_compat

if TYPE_CHECKING:
    pass

from llm_rosetta_gateway._vendor.httpserver import (
    App,
)

from ..logging import get_logger


logger = get_logger()

_DEFAULT_TOKEN_ROTATE_SECONDS = 15 * 60


async def _periodic_token_rotation(app: App) -> None:
    """Periodically rotate the internal token in the background."""
    config = getattr(app, "gateway_config", None)
    interval = _DEFAULT_TOKEN_ROTATE_SECONDS
    if config is not None:
        interval = getattr(config, "token_rotate_seconds", 0) or interval
    while True:
        await asyncio.sleep(interval)
        auth_state = getattr(app, "auth_state", None)
        if auth_state is None:
            continue
        auth_state.rotate_internal_token()
        app.internal_token = auth_state.internal_token  # type: ignore
        logger.debug("Background token rotation completed")


_FLUSH_METRICS_INTERVAL = 30  # seconds


_CLEANUP_INTERVAL = 7 * 24 * 3600  # weekly


_CLEANUP_INITIAL_DELAY = 120  # seconds


async def _periodic_cleanup(app: App) -> None:
    """Periodically delete records older than the configured max_age_days."""
    await asyncio.sleep(_CLEANUP_INITIAL_DELAY)
    while True:
        try:
            ops_ctx = getattr(app, "ops_ctx", None)
            config = getattr(app, "gateway_config", None)
            if (
                ops_ctx is not None
                and ops_ctx.persistence is not None
                and config is not None
            ):
                from ..ops.data import OpsPeriodicCleanup

                raw_cfg = getattr(config, "_raw", {})
                server = raw_cfg.get("server", {}) if isinstance(raw_cfg, dict) else {}
                rl_age = int((server.get("request_log") or {}).get("max_age_days", 90))
                ol_age = int((server.get("ops_log") or {}).get("max_age_days", 90))

                result = await OpsPeriodicCleanup(
                    ops_ctx, rl_age=rl_age, ol_age=ol_age
                ).execute()

                total = (result or {}).get("total", 0)
                if total > 0:
                    logger.info("Periodic cleanup: %d entries removed", total)
        except Exception as exc:
            logger.warning("Periodic cleanup failed: %s", exc)
        await asyncio.sleep(_CLEANUP_INTERVAL)


async def _periodic_flush(app: App) -> None:
    """Periodically flush metrics counters to disk."""
    while True:
        await asyncio.sleep(_FLUSH_METRICS_INTERVAL)
        persistence = getattr(app, "persistence", None)
        if persistence is None:
            continue
        metrics = getattr(app, "metrics", None)
        if metrics is not None:
            try:
                if await persistence.check_and_clear_rebuild_flag():
                    logger.info(
                        "Rebuild flag detected (external cleanup), rebuilding counters"
                    )
                    rows = [
                        row async for row in persistence.iter_log_rows_for_rebuild()
                    ]
                    metrics.rebuild_counters(iter(rows))
                await persistence.save_metrics(metrics.export_counters())
            except Exception as exc:
                logger.warning("Failed to flush metrics: %s", exc)


@_sync_compat
async def _flush_now(app: App) -> None:
    """Final async flush on shutdown."""
    persistence = getattr(app, "persistence", None)
    if persistence is None:
        return

    metrics = getattr(app, "metrics", None)
    if metrics is not None:
        try:
            await persistence.save_metrics(metrics.export_counters())
        except Exception as exc:
            logger.warning("Shutdown: failed to flush metrics: %s", exc)

    # Record shutdown event (skip pruning to avoid latency before close)
    ops_log = getattr(app, "ops_log", None)
    if ops_log is not None:
        from llm_rosetta_gateway.observability.ops_log import (
            EVENT_SHUTDOWN,
            OpsLogEntry,
            SEVERITY_INFO,
            SOURCE_GATEWAY,
        )

        await ops_log.add(
            OpsLogEntry.create(
                event_type=EVENT_SHUTDOWN,
                severity=SEVERITY_INFO,
                message="Gateway shutting down",
                source=SOURCE_GATEWAY,
            ),
            _skip_prune=True,
        )

    await persistence.close()

    keystore = getattr(app, "keystore", None)
    if keystore is not None:
        await keystore.close()

    logger.info("Persistence flushed and closed on shutdown")
