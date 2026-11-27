"""The adapter that implements :class:`~llm_rosetta_gateway.admin_api.AdminContext`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import _state

if TYPE_CHECKING:
    from ..config import GatewayConfig


_REFUSED = {
    "unknown_extension": 404,
    "unknown_program": 404,
    "program_required": 400,
    "no_install_spec": 409,
}


class GatewayAdminContext:
    """Serve the admin panel's needs from the gateway's own state.

    Args:
        app: The app this context belongs to; its ``ops_ctx`` (set once the
            admin panel is set up) is where installs are recorded.  ``None`` for
            the stock fallback of an app built by hand: nothing is recorded.
    """

    def __init__(self, app: Any = None) -> None:
        self._app = app
        self._installing: set[str] = set()

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

    async def install_extension_cli(
        self, ext_id: str, program: str | None
    ) -> dict[str, Any]:
        """Install a program an extension declares, at the version it pins."""
        from llm_rosetta.api import CliInstallError

        from ..admin_api import ExtensionInstallRefused
        from ..ops import OpsContext, OpsExtensionCliInstall
        from ..shims.extension_view import ProgramRequestError, declared_program

        try:
            declared = declared_program(ext_id, program)
        except ProgramRequestError as exc:
            raise ExtensionInstallRefused(
                _REFUSED[exc.reason], exc.reason, str(exc)
            ) from None
        if ext_id in self._installing:
            raise ExtensionInstallRefused(
                409, "busy", f"an install of {ext_id!r} is already running"
            )
        self._installing.add(ext_id)
        try:
            ops_ctx = getattr(self._app, "ops_ctx", None) or OpsContext()
            try:
                done = await OpsExtensionCliInstall(
                    ops_ctx, ext_id=ext_id, program=declared
                ).execute()
            except CliInstallError as exc:
                raise ExtensionInstallRefused(502, "failed", str(exc)) from None
        finally:
            self._installing.discard(ext_id)
        return {
            "extension": ext_id,
            "program": declared.name,
            "version": declared.install.version,
            **done,
        }
