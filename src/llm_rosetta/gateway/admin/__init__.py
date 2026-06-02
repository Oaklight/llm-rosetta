"""Admin panel for the llm-rosetta gateway."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from .metrics import MetricsCollector
from .persistence import DEFAULT_ERROR_MAX, DEFAULT_SUCCESS_MAX, PersistenceManager
from .request_log import RequestLog

if TYPE_CHECKING:
    from ..config import GatewayConfig

__all__ = ["setup_admin", "MetricsCollector", "RequestLog", "PersistenceManager"]

logger = logging.getLogger("llm-rosetta-gateway")


def _resolve_log_caps(config: GatewayConfig) -> tuple[int, int]:
    """Resolve (success_max, error_max) from env vars and config.

    Precedence: env vars > config.request_log.{success,error}_max >
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
    error_max = _parse_int_env("REQUEST_LOG_ERROR_MAX")

    if success_max is None:
        success_max = rl_cfg.get("success_max")
    if error_max is None:
        error_max = rl_cfg.get("error_max")

    legacy = rl_cfg.get("max_entries")
    if legacy is not None and success_max is None:
        logger.warning(
            "config: server.request_log.max_entries is deprecated; "
            "use success_max (and optionally error_max) instead."
        )
        success_max = legacy

    return (
        int(success_max) if success_max is not None else DEFAULT_SUCCESS_MAX,
        int(error_max) if error_max is not None else DEFAULT_ERROR_MAX,
    )


def setup_admin(
    app: Any,
    config: GatewayConfig,
    config_path: str | None,
) -> None:
    """Initialize admin panel state on the app.

    Routes are registered separately via ``register_admin_routes`` before
    calling this function.
    """
    metrics = MetricsCollector()

    # Set up SQLite persistence alongside the config file
    persistence: PersistenceManager | None = None
    if config_path:
        data_dir = os.path.join(os.path.dirname(config_path), "data")
        success_max, error_max = _resolve_log_caps(config)
        persistence = PersistenceManager(
            data_dir, success_max=success_max, error_max=error_max
        )

        # Restore persisted metrics counters
        saved_metrics = persistence.load_metrics()
        if saved_metrics:
            metrics.load_counters(saved_metrics)
            logger.info(
                "Loaded metrics from disk (total_requests=%d)",
                metrics.total_requests,
            )

    # Backfill target_provider_name for legacy log entries
    if persistence is not None:
        model_to_provider = {model: config.models[model] for model in config.models}
        backfilled = persistence.backfill_provider_names(model_to_provider)
        if backfilled:
            logger.info(
                "Backfilled target_provider_name for %d log entries",
                backfilled,
            )

    # Request log delegates to persistence when available
    request_log = RequestLog(persistence=persistence)

    app.metrics = metrics
    app.request_log = request_log
    app.persistence = persistence
    app.gateway_config = config
    app.config_path = config_path
