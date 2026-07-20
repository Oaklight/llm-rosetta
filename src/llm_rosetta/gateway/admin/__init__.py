"""Admin panel for the llm-rosetta gateway."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from llm_rosetta.observability import (
    DEFAULT_ERROR_MAX,
    DEFAULT_SUCCESS_MAX,
    MetricsCollector,
    PersistenceManager,
    RequestLog,
)

if TYPE_CHECKING:
    from ..config import ConfigIO, GatewayConfig

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
    config_io: ConfigIO | None = None,
    custom_head: str | None = None,
    branding: dict[str, Any] | None = None,
) -> None:
    """Initialize admin panel state on the app.

    Args:
        app: The httpserver application.
        config: Parsed gateway configuration.
        config_path: Path to the config file on disk (``None`` disables
            persistence and config editing).
        config_io: Optional I/O adapter for reading/writing the config
            file.  Defaults to :class:`JsoncConfigIO` when ``None``.
            Supply a custom implementation for non-JSONC formats
            (e.g. YAML in argo-proxy).
        custom_head: Optional HTML fragment (``<style>``, ``<script>``,
            etc.) injected before ``</head>`` when serving the admin
            panel.  Allows downstream projects to override UI behavior
            without modifying the reference ``admin.html``.
        branding: Optional dict to customize the admin panel identity.
            Supported keys::

                title       — header title (default: "llm-rosetta")
                subtitle    — header subtitle (default: "gateway admin")
                version     — version shown in settings footer
                links       — list of {"label", "url", "icon"} for footer buttons
                              icon: "github" | "pypi" | "docker" | "docs" | None
                attribution — "Powered by ..." line in settings footer

    Routes are registered separately via ``register_admin_routes`` before
    calling this function.
    """
    if config_io is None:
        from ..config import JsoncConfigIO

        config_io = JsoncConfigIO()
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

    # On-demand deep profiling state
    from llm_rosetta.observability import ProfilerState

    profiler_state = ProfilerState()

    # In-memory content capture state
    from llm_rosetta.observability import CaptureState

    capture_state = CaptureState()

    app.metrics = metrics
    app.request_log = request_log
    app.persistence = persistence
    app.gateway_config = config
    app.config_path = config_path
    app.config_io = config_io
    app.profiler_state = profiler_state
    app.capture_state = capture_state
    # Build custom_head: user-supplied fragment + branding injection
    parts: list[str] = []
    if custom_head:
        parts.append(custom_head)
    if branding:
        import json as _json

        serialized = _json.dumps(branding).replace("</", r"<\/")
        parts.append(f"<script>window.__branding={serialized};</script>")
    app.admin_custom_head = "\n".join(parts)
