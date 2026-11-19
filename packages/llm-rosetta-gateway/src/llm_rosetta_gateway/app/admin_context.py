"""The adapter that implements :class:`~llm_rosetta_gateway.admin_api.AdminContext`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import _state

if TYPE_CHECKING:
    from ..config import GatewayConfig


class GatewayAdminContext:
    """Serve the admin panel's needs from the gateway's own state."""

    def served_config(self) -> GatewayConfig | None:
        """Return the config the gateway serves, or ``None`` if none is set."""
        return _state.get_config()

    def publish_config(self, config: GatewayConfig) -> None:
        """Make *config* the one the route handlers read."""
        _state.set_config(config)
