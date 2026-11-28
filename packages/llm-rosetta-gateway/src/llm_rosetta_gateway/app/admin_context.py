"""The adapter that implements :class:`~llm_rosetta_gateway.admin_api.AdminContext`."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from ..shims.extension_view import pick_release, update_report
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

    @contextlib.contextmanager
    def _exclusive(self, ext_id: str) -> Iterator[None]:
        """Hold *ext_id* for one install, removal or update at a time."""
        from ..admin_api import ExtensionInstallRefused

        if ext_id in self._installing:
            raise ExtensionInstallRefused(
                409, "busy", f"an operation on {ext_id!r} is already running"
            )
        self._installing.add(ext_id)
        try:
            yield
        finally:
            self._installing.discard(ext_id)

    def _ops_ctx(self) -> Any:
        from ..ops import OpsContext

        return getattr(self._app, "ops_ctx", None) or OpsContext()

    def extension_removal_plan(self, ext_id: str) -> dict[str, Any] | None:
        """What removing the extension deletes and keeps, or ``None`` if unknown."""
        from ..shims.extension_view import removal_view

        return removal_view(ext_id)

    async def remove_extension(
        self, ext_id: str, *, replace: bool = False
    ) -> dict[str, Any]:
        """Remove an extension and the programs it installed, as the plan says."""
        from llm_rosetta.api import ExtensionInstallError, plan_removal

        from ..admin_api import ExtensionInstallRefused
        from ..ops import OpsExtensionRemove
        from ..shims.extension_view import locate_extension

        found = locate_extension(ext_id)
        if found is None:
            raise ExtensionInstallRefused(
                404, "unknown_extension", f"no extension {ext_id!r}"
            )
        plan = plan_removal(ext_id, path=found[0])
        if not plan.removable:
            raise ExtensionInstallRefused(
                409, "not_removable", plan.hint or f"{ext_id!r} cannot be removed here"
            )
        with self._exclusive(ext_id):
            try:
                done = await OpsExtensionRemove(
                    self._ops_ctx(), ext_id=ext_id, plan=plan, replace=replace
                ).execute()
            except ExtensionInstallError as exc:
                raise ExtensionInstallRefused(502, "failed", str(exc)) from None
        return {"extension": ext_id, **done}

    async def check_extension_updates(
        self, config: Any, report: Any, ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Compare installed extensions with the manifest sources, refetching them."""
        import asyncio

        loop = asyncio.get_running_loop()
        catalogue = await loop.run_in_executor(
            None, lambda: self.shim_source_catalogue(config, report, refresh=True)
        )
        from ..shims.extension_view import extension_versions

        return update_report(
            await loop.run_in_executor(None, extension_versions), catalogue, ids
        )

    async def update_extension(
        self, ext_id: str, config: Any, report: Any, version: str | None = None
    ) -> dict[str, Any]:
        """Install the newest release (or *version*) of an extension from its source."""
        import asyncio

        from llm_rosetta.api import (
            ExtensionInstallError,
            install_from_index,
            load_index,
            plan_removal,
        )

        from ..admin_api import ExtensionInstallRefused
        from ..ops import OpsExtensionUpdate
        from ..shims.extension_view import locate_extension

        found = locate_extension(ext_id)
        if found is None:
            raise ExtensionInstallRefused(
                404, "unknown_extension", f"no extension {ext_id!r}"
            )
        if not plan_removal(ext_id, path=found[0]).removable:
            raise ExtensionInstallRefused(
                409,
                "not_removable",
                "provided by the image: update it by rebuilding the image",
            )
        before = found[1]
        loop = asyncio.get_running_loop()
        catalogue = await loop.run_in_executor(
            None, lambda: self.shim_source_catalogue(config, report, refresh=False)
        )
        chosen = pick_release(catalogue, ext_id, version)
        if isinstance(chosen, str):
            raise ExtensionInstallRefused(
                404 if version is None else 400, "no_release", chosen
            )
        source_url, target = chosen

        def work() -> str | None:
            index = load_index(source_url)
            installed = install_from_index(ext_id, index, version=target)
            return next((i.version for i in installed if i.id == ext_id), target)

        with self._exclusive(ext_id):
            try:
                done = await OpsExtensionUpdate(
                    self._ops_ctx(), ext_id=ext_id, before=before, work=work
                ).execute()
            except ExtensionInstallError as exc:
                raise ExtensionInstallRefused(502, "failed", str(exc)) from None
        return {"extension": ext_id, "before": before, **done}

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
