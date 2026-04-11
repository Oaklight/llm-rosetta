"""Admin panel for the llm-rosetta gateway."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .metrics import MetricsCollector
from .request_log import RequestLog

if TYPE_CHECKING:
    from starlette.applications import Starlette

    from ..config import GatewayConfig

__all__ = ["setup_admin", "MetricsCollector", "RequestLog"]


def setup_admin(
    app: Starlette,
    config: GatewayConfig,
    config_path: str | None,
) -> None:
    """Initialize admin panel state on the app.

    Routes are added separately in ``create_app`` *before* the Starlette
    instance is constructed so that its Router compiles them properly.
    """
    app.state.metrics = MetricsCollector()
    app.state.request_log = RequestLog()
    app.state.gateway_config = config
    app.state.config_path = config_path
