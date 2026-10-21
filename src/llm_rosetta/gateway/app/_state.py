"""Shared app state: the config the running app was built with.

``create_app`` stores it once; the route handlers read it through the module
attribute so the assignment is visible to every importer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import GatewayConfig

#: The config the serving app was built from, set by ``create_app``.
_config: GatewayConfig | None = None


def set_config(config: GatewayConfig) -> None:
    """Record the config the app serves."""
    global _config
    _config = config
