"""Admin panel for the llm-rosetta gateway."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from .metrics import MetricsCollector
from .persistence import PersistenceManager
from .request_log import RequestLog

if TYPE_CHECKING:
    from ..config import GatewayConfig

__all__ = ["setup_admin", "MetricsCollector", "RequestLog", "PersistenceManager"]

logger = logging.getLogger("llm-rosetta-gateway")


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
        persistence = PersistenceManager(data_dir)

        # Restore persisted metrics counters
        saved_metrics = persistence.load_metrics()
        if saved_metrics:
            metrics.load_counters(saved_metrics)
            logger.info(
                "Loaded metrics from disk (total_requests=%d)",
                metrics.total_requests,
            )

    # Request log delegates to persistence when available
    request_log = RequestLog(persistence=persistence)

    app.metrics = metrics
    app.request_log = request_log
    app.persistence = persistence
    app.gateway_config = config
    app.config_path = config_path
