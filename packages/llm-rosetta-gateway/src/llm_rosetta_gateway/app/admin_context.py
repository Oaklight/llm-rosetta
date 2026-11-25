"""The adapter that implements :class:`~llm_rosetta_gateway.admin_api.AdminContext`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

    def shim_source_catalogue(
        self, config: Any, report: Any, *, refresh: bool = False
    ) -> dict[str, Any]:
        """Every manifest source *config* reads, with every vendor it publishes."""
        from ..shims.source_view import catalogue_payload

        return catalogue_payload(config, report, refresh=refresh)

    def extension_listing(self) -> list[dict[str, Any]]:
        """Every extension found or installed, one JSON-safe record each."""
        from ..shims.extension_view import extension_listing

        return extension_listing()
