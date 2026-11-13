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
    """Periodically queue a cleanup of records older than max_age_days.

    The work itself is a job: it hides its ranges, deletes in batches and
    reports progress on the board, instead of running a multi-second delete
    inside this loop where nothing can see it.
    """
    await asyncio.sleep(_CLEANUP_INITIAL_DELAY)
    while True:
        try:
            jobs = getattr(app, "jobs", None)
            config = getattr(app, "gateway_config", None)
            if jobs is not None and config is not None:
                from ..jobs.enqueue import enqueue_cleanup

                raw_cfg = getattr(config, "_raw", {})
                server = raw_cfg.get("server", {}) if isinstance(raw_cfg, dict) else {}
                rl_age = int((server.get("request_log") or {}).get("max_age_days", 90))
                ol_age = int((server.get("ops_log") or {}).get("max_age_days", 90))

                queued = await enqueue_cleanup(
                    jobs,
                    mode="age",
                    tables=("request_log", "error_dumps"),
                    max_age_days=rl_age,
                    source="periodic",
                )
                if ol_age != rl_age:
                    await enqueue_cleanup(
                        jobs,
                        mode="age",
                        tables=("ops_log",),
                        max_age_days=ol_age,
                        source="periodic",
                    )
                logger.info("Periodic cleanup queued as job %s", queued.id)
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
                    jobs = getattr(app, "jobs", None)
                    if jobs is not None:
                        from ..jobs.enqueue import enqueue_rebuild

                        job = await enqueue_rebuild(jobs, source="persistence")
                        logger.info("Rebuild flag detected; queued as job %s", job.id)
                    else:
                        # No worker (in-memory mode): rebuild here, streaming so
                        # the table is not materialised.
                        logger.info(
                            "Rebuild flag detected (external cleanup), "
                            "rebuilding counters"
                        )
                        await metrics.rebuild_from_stream(
                            persistence.iter_log_rows_for_rebuild()
                        )
                jobs = getattr(app, "jobs", None)
                if jobs is not None:
                    await jobs.maybe_prune()
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
