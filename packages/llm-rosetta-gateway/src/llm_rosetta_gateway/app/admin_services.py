"""The services the admin panel is built on, started once per app.

``setup_admin`` used to construct the metrics collector, the persistence
manager, the request and ops logs, the job manager and the ops context itself.
Those are the gateway's own services, so the app layer builds them and the
admin package asks for them through ``AdminContext.start_services``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ..config import ConfigIO, GatewayConfig, JsoncConfigIO
from ..jobs import JobManager
from ..jobs.handlers.data import data_handlers
from ..observability import (
    DEFAULT_SUCCESS_MAX,
    CaptureState,
    MetricsCollector,
    OpsLog,
    PersistenceManager,
    ProfilerState,
    RequestLog,
)
from ..ops import OpsContext
from ..ops.data import OpsPrune

logger = logging.getLogger("llm-rosetta-gateway")


def _resolve_log_caps(config: GatewayConfig) -> tuple[int, int | None]:
    """Resolve (success_max, dump_max) from env vars and config.

    Precedence: env vars > config.request_log.{success_max,error_dump_max} >
    legacy config.request_log.max_entries > built-in defaults.
    """
    rl_cfg: dict[str, Any] = getattr(config, "request_log", {}) or {}

    def _parse_int_env(name: str) -> int | None:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            return None
        try:
            return int(raw)
        except ValueError:
            logger.warning("Ignoring non-integer %s=%r", name, raw)
            return None

    success_max = _parse_int_env("REQUEST_LOG_SUCCESS_MAX")
    if success_max is None:
        success_max = rl_cfg.get("success_max")
    legacy = rl_cfg.get("max_entries")
    if legacy is not None and success_max is None:
        logger.warning(
            "config: server.request_log.max_entries is deprecated; "
            "use success_max instead."
        )
        success_max = legacy

    # Warn if old error_max key is still in config
    if rl_cfg.get("error_max") is not None:
        logger.warning(
            "config: server.request_log.error_max is deprecated and ignored; "
            "use error_dump_max to control error dump retention instead."
        )

    dump_max = _parse_int_env("ERROR_DUMP_MAX")
    if dump_max is None:
        dump_max = rl_cfg.get("error_dump_max")
    if dump_max is not None:
        dump_max = int(dump_max)

    return (
        int(success_max) if success_max is not None else DEFAULT_SUCCESS_MAX,
        dump_max,
    )


async def _init_persistence(
    config: Any,
    config_path: str | None,
    data_dir: str | None,
    metrics: MetricsCollector,
) -> tuple[PersistenceManager, bool] | None:
    """Resolve data_dir, create PersistenceManager, and restore counters."""
    resolved_data_dir: str | None = data_dir
    if not resolved_data_dir:
        configured = getattr(config, "data_dir", None)
        if configured:
            if config_path and not os.path.isabs(configured):
                resolved_data_dir = os.path.join(
                    os.path.dirname(config_path), configured
                )
            else:
                resolved_data_dir = configured
        elif config_path:
            resolved_data_dir = os.path.join(os.path.dirname(config_path), "data")
    if not resolved_data_dir:
        return None

    success_max, dump_max = _resolve_log_caps(config)
    ol_cfg = getattr(config, "ops_log", {}) or {}
    ops_info_max = ol_cfg.get("info_max")
    ops_warn_max = ol_cfg.get("warn_max")
    if ops_info_max is not None:
        ops_info_max = int(ops_info_max)
    if ops_warn_max is not None:
        ops_warn_max = int(ops_warn_max)
    persistence = await PersistenceManager.create(
        resolved_data_dir,
        success_max=success_max,
        dump_max=dump_max,
        ops_info_max=ops_info_max,
        ops_warn_max=ops_warn_max,
    )

    saved_metrics = await persistence.load_metrics()
    if saved_metrics:
        metrics.load_counters(saved_metrics)
        logger.info(
            "Loaded metrics from disk (total_requests=%d)",
            metrics.total_requests,
        )

    # Detect counter drift (ungraceful shutdown) but defer the rebuild to
    # a background task so the server starts accepting connections immediately.
    # Compare lifetime counter (monotonic) against the physical log size.
    # lifetime < log means counters were never persisted for some requests.
    # The reverse (lifetime > log) is normal after pruning.  Hidden rows count
    # here: a cleared prefix is still on disk until its job runs, and leaving
    # it out would mask exactly the drift this check exists to catch.
    log_entries = await persistence.count_log_entries(include_hidden=True)
    _counter_rebuild_needed = False
    if await persistence.check_and_clear_rebuild_flag():
        logger.info(
            "Rebuild flag detected (external cleanup) — "
            "rebuild will run in the background after startup"
        )
        _counter_rebuild_needed = True
    elif metrics.lifetime_total_requests < log_entries:
        logger.warning(
            "Counter drift detected (lifetime=%d, log=%d) — "
            "rebuild will run in the background after startup",
            metrics.lifetime_total_requests,
            log_entries,
        )
        _counter_rebuild_needed = True

    return persistence, _counter_rebuild_needed


async def start_admin_services(
    app: Any,
    config: GatewayConfig,
    config_path: str | None,
    config_io: ConfigIO | None,
    data_dir: str | None,
) -> None:
    """Build the services the admin panel reads and store them on *app*.

    Sets ``metrics``, ``request_log``, ``ops_log``, ``persistence``, ``jobs``,
    ``ops_ctx``, ``config_path``, ``config_io``, ``profiler_state`` and
    ``capture_state``, and registers the job handlers that need the ops
    context.

    Args:
        app: The httpserver application.
        config: The parsed gateway configuration.
        config_path: The config file, or ``None`` (no persistence, no editing).
        config_io: The config reader and writer; :class:`JsoncConfigIO` when
            ``None``.
        data_dir: An explicit data directory, or ``None`` to resolve it from
            the config.
    """
    if config_io is None:
        config_io = JsoncConfigIO()
    metrics = MetricsCollector()

    result = await _init_persistence(config, config_path, data_dir, metrics)
    if result is not None:
        persistence, counter_rebuild_needed = result
    else:
        persistence, counter_rebuild_needed = None, False

    # Backfills are deferred to DeferredStartup (run in background after
    # the event loop starts).  Store the flag for the rebuild task.
    app._counter_rebuild_needed = counter_rebuild_needed  # type: ignore[attr-defined]

    # Request log delegates to persistence when available
    request_log = RequestLog(persistence=persistence)

    # Ops log: PersistenceManager creates the table, OpsLog wraps it
    ops_log = OpsLog(persistence=persistence)

    # On-demand deep profiling state
    profiler_state = ProfilerState()

    # In-memory content capture state
    capture_state = CaptureState()

    # Backfills (last_used, error_dump_log_ids) are deferred to
    # DeferredStartup — they run in a background thread after startup.

    app.metrics = metrics
    app.request_log = request_log
    app.ops_log = ops_log
    app.persistence = persistence
    # Background jobs: the envelope destructive operations use.  Constructed
    # here, beside the services a handler needs; `run_gateway` starts the
    # workers once the loop is serving, so a test that builds the app never
    # leaves a worker running.
    job_manager = JobManager(persistence) if persistence is not None else None
    app.jobs = job_manager
    # Unified ops context — used by gateway/ops/ operation layer
    _ops_ctx = OpsContext(
        ops_log=ops_log,
        request_log=request_log,
        metrics=metrics,
        persistence=persistence,
        config_io=config_io,
        config_path=config_path,
        jobs=job_manager,
    )
    app.ops_ctx = _ops_ctx

    # The data handlers need the ops context, which is built above; binding
    # them here keeps `jobs` free of an import back into `admin`.
    if job_manager is not None:
        for kind, handler in data_handlers(_ops_ctx).items():
            job_manager.registry.register(kind, handler)

    # Wire amortized prune audit callback
    if persistence is not None:

        async def _on_prune(table: str, count: int) -> None:
            await OpsPrune(_ops_ctx, table=table, count=count).execute()

        persistence.on_prune = _on_prune

    app.config_path = config_path
    app.config_io = config_io
    app.profiler_state = profiler_state
    app.capture_state = capture_state
