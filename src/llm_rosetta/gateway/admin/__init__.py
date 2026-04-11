"""Admin panel for the llm-rosetta gateway."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from .metrics import MetricsCollector
from .persistence import PersistenceManager
from .request_log import RequestLog

if TYPE_CHECKING:
    from starlette.applications import Starlette

    from ..config import GatewayConfig

__all__ = ["setup_admin", "MetricsCollector", "RequestLog", "PersistenceManager"]

logger = logging.getLogger("llm-rosetta-gateway")


def setup_admin(
    app: Starlette,
    config: GatewayConfig,
    config_path: str | None,
) -> None:
    """Initialize admin panel state on the app.

    Routes are added separately in ``create_app`` *before* the Starlette
    instance is constructed so that its Router compiles them properly.
    """
    metrics = MetricsCollector()
    request_log = RequestLog()

    # Set up file persistence alongside the config file
    persistence: PersistenceManager | None = None
    if config_path:
        data_dir = os.path.join(os.path.dirname(config_path), "data")
        persistence = PersistenceManager(data_dir)

        # Restore persisted request log
        saved_entries = persistence.load_log_entries(request_log._max_entries)
        if saved_entries:
            request_log.load_entries(saved_entries)
            logger.info("Loaded %d request log entries from disk", len(saved_entries))

        # Restore persisted metrics counters
        saved_metrics = persistence.load_metrics()
        if saved_metrics:
            metrics.load_counters(saved_metrics)
            logger.info(
                "Loaded metrics from disk (total_requests=%d)",
                metrics.total_requests,
            )

    app.state.metrics = metrics
    app.state.request_log = request_log
    app.state.persistence = persistence
    app.state.gateway_config = config
    app.state.config_path = config_path
