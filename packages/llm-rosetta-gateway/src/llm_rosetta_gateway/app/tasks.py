"""Periodic background tasks: token rotation, cleanup and metric flush."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    pass

from llm_rosetta_gateway._vendor.httpserver import (
    App,
)

from ..observability.logging import get_logger


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


_SWEEP_INTERVAL = 60  # seconds


async def _periodic_sweep(app: App) -> None:
    """Periodically let each transport give back what nobody is using.

    A transport that holds upstream state — the ACP lane's agent sessions —
    expires it on its own schedule, but only when a request arrives.  An idle
    gateway would otherwise keep every such session alive until traffic came
    back, which is exactly when nobody is there to notice the cost.  A
    transport without a ``sweep`` has nothing to give back and is skipped.
    """
    while True:
        await asyncio.sleep(_SWEEP_INTERVAL)
        transports = dict(getattr(app, "transports", {}) or {})
        for transport in transports.values():
            sweep = getattr(transport, "sweep", None)
            if sweep is None:
                continue
            try:
                await sweep()
            except Exception as exc:  # noqa: BLE001 - a sweep must not die
                logger.warning("Periodic transport sweep failed: %s", exc)


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
                from ..jobs.enqueue import CLEANUP_KIND, enqueue_cleanup
                from ..jobs.model import JOB_SUCCEEDED

                raw_cfg = getattr(config, "_raw", {})
                server = raw_cfg.get("server", {}) if isinstance(raw_cfg, dict) else {}
                rl_age = int((server.get("request_log") or {}).get("max_age_days", 90))
                ol_age = int((server.get("ops_log") or {}).get("max_age_days", 90))

                # The two ages can differ, and then this task queues two
                # passes over different table sets.  Ask per set: one success
                # must not answer for the other, or a failed ops-log sweep is
                # deferred for a week by the request-log pass.
                passes: list[tuple[tuple[str, ...], int]] = [
                    (("request_log", "error_dumps"), rl_age)
                ]
                if ol_age != rl_age:
                    passes.append((("ops_log",), ol_age))

                # The jobs table outlives this task.  Without this check every
                # restart queues another cleanup two minutes later, however
                # recently the last one finished.  Only a *successful periodic*
                # pass counts: a failed or cancelled one did not do the work,
                # and a small manual range delete is not the weekly sweep.
                # Sleep the remainder of the interval rather than a whole one,
                # so a wake that lands milliseconds early is not doubled.
                # Wake at the earliest due pass and rerun both; rerunning the
                # other pass early is harmless and keeps the schedule simple.
                waits: list[float] = []
                for tables, _ in passes:
                    last = await jobs.store.last_job_at(
                        CLEANUP_KIND,
                        status=JOB_SUCCEEDED,
                        source="periodic",
                        tables=tables,
                    )
                    if last is None:
                        waits.append(0.0)
                        continue
                    try:
                        due = datetime.fromisoformat(last) + timedelta(
                            seconds=_CLEANUP_INTERVAL
                        )
                    except ValueError:
                        waits.append(0.0)
                        continue
                    waits.append((due - datetime.now(timezone.utc)).total_seconds())

                wait = min(waits)
                if wait > 0:
                    logger.debug("Periodic cleanup deferred for %.0fs", wait)
                    await asyncio.sleep(wait)
                    continue

                queued = None
                for tables, age in passes:
                    queued = await enqueue_cleanup(
                        jobs,
                        mode="age",
                        tables=tables,
                        max_age_days=age,
                        source="periodic",
                    )
                assert queued is not None  # passes is never empty
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
