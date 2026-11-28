"""The adapter that implements :class:`~llm_rosetta_gateway.admin_api.AdminContext`."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from ..shims.extension_view import pick_release, update_report

if TYPE_CHECKING:
    from ..config import GatewayConfig


_REFUSED = {
    "unknown_extension": 404,
    "unknown_program": 404,
    "program_required": 400,
    "no_install_spec": 409,
}

#: Refusal of an install while ``server.extension_install`` is false.
_INSTALL_OFF = (
    "installing extensions through the gateway is switched off: "
    "set server.extension_install to true to allow it"
)


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
        return getattr(self._app, "gateway_config", None)

    def publish_config(self, config: GatewayConfig) -> None:
        """Make *config* the one the route handlers read."""
        self._app.gateway_config = config

    async def reload_config(
        self, config_path: str, *, force_shims: bool = False
    ) -> GatewayConfig:
        """Re-read the config from disk, rebuild it and swap it in everywhere.

        Declarative shim sources are (re)applied before the config is rebuilt,
        but only when the ``shims`` section changed (or *force_shims* is set),
        so an unrelated edit never triggers a remote refetch.  Applying them
        drops shims whose source was removed.  Then the new config is served,
        the live settings that read it are updated (auth, log level, rate
        limits, trusted proxies, retention caps), the reload is recorded in the
        ops log, and credential-minting providers are seeded again.

        Args:
            config_path: The config file to read.
            force_shims: Reapply the shim sources even if their section is
                unchanged.

        Returns:
            The config now served.

        Raises:
            RuntimeError: The app has no config reader.
        """
        import asyncio

        from ..observability.logging import setup_logging
        from ..ops import OpsConfigReload, OpsContext, OpsProviderValidation
        from ..shims.shim_sources import apply_shim_sources, shims_signature
        from ..config import GatewayConfig

        app = self._app
        config_io = getattr(app, "config_io", None)
        if config_io is None:
            raise RuntimeError("No ConfigIO adapter configured on this application")
        raw = config_io.load(config_path)
        # Taken in the same turn as the read, before anything can await: the
        # reader records the registry for the document it just read, and this is
        # the config object built from that document, so it is the one that
        # reports it.  A reader that runs no load-time steps has no such
        # attribute, and reports nothing — the same answer as a config built
        # from a literal.
        maintenance = getattr(config_io, "last_load_registry", None)

        report = getattr(app, "shim_source_report", None)
        signature = shims_signature(raw)
        if force_shims or signature != getattr(app, "shim_source_signature", None):
            loop = asyncio.get_running_loop()
            report = await loop.run_in_executor(
                None, apply_shim_sources, raw, config_path
            )
            # The update verdicts cached on ``shim_update_check`` describe the
            # previous report, which pinned other commits, so they go with it.
            app.shim_source_report = report
            app.shim_source_signature = signature
            app.shim_update_check = None

        recovery = OpsProviderValidation(
            getattr(app, "ops_ctx", None) or OpsContext(),
            raw,
            report=report,
            persist=True,
        )
        recovery_result = await recovery.execute()
        app.provider_validation_warnings = recovery_result.get("disabled", [])
        new_config = GatewayConfig(
            raw, maintenance=maintenance, config_path=config_path
        )
        self.publish_config(new_config)

        self._sync_auth(new_config)

        setup_logging(
            verbose=new_config.verbose,
            log_bodies=new_config.log_bodies,
            log_format=new_config.log_format,
        )

        # Rate limiting is rebuilt only if its settings changed.
        rate_limit_state = getattr(app, "rate_limit_state", None)
        if rate_limit_state is not None:
            old_config = getattr(app, "_prev_gateway_config", None)
            changed = old_config is None or any(
                getattr(new_config, name, None) != getattr(old_config, name, None)
                for name in (
                    "rate_limit_enabled",
                    "rate_limit_algorithm",
                    "rate_limit_global",
                    "rate_limit_per_ip",
                    "rate_limit_per_key",
                    "rate_limit_per_model",
                    "rate_limit_exclude",
                )
            )
            if changed:
                rate_limit_state.rebuild(new_config)
        app._prev_gateway_config = new_config
        proxy_policy = getattr(app, "proxy_policy", None)
        if proxy_policy is not None:
            proxy_policy.trusted = new_config.trusted_proxies

        persistence = getattr(app, "persistence", None)
        if persistence is not None:
            request_log_cfg = new_config.request_log or {}
            if "success_max" in request_log_cfg:
                persistence.success_max = int(request_log_cfg["success_max"])
            if "error_dump_max" in request_log_cfg:
                persistence.dump_max = int(request_log_cfg["error_dump_max"])
            ops_log_cfg = new_config.ops_log or {}
            if "info_max" in ops_log_cfg:
                persistence.retention.ops_info_max = max(
                    100, int(ops_log_cfg["info_max"])
                )
            if "warn_max" in ops_log_cfg:
                persistence.retention.ops_warn_max = max(
                    100, int(ops_log_cfg["warn_max"])
                )

        await OpsConfigReload(
            OpsContext(ops_log=getattr(app, "ops_log", None)), new_config
        ).execute()

        # The rebuild above replaced every ProviderInfo, so each credential-
        # minting provider is back on the pending sentinel with no refresh loop.
        # Seeding otherwise runs only at process start, so without this the
        # provider never obtains a credential and every request to it is
        # rejected until the process restarts.
        deferred = getattr(app, "deferred_startup", None)
        if deferred is not None:
            await deferred.apply_config_reload(new_config)

        return new_config

    def _sync_auth(self, config: GatewayConfig) -> None:
        """Update the auth hook's state for a reloaded config."""
        auth_state = getattr(self._app, "auth_state", None)
        if auth_state is not None:
            # An admin password changed through the CLI or a config edit.
            if config.admin_password != auth_state.admin_password:
                auth_state.change_password(config.admin_password or "")
            auth_state.configure_sessions(
                idle_minutes=config.admin_session_idle_minutes,
                remember_days=config.admin_session_remember_days,
            )

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

    def _require_install_enabled(self) -> None:
        """Refuse an install or update when ``server.extension_install`` is off."""
        from ..admin_api import ExtensionInstallRefused

        config = self.served_config()
        if config is not None and not config.extension_install:
            raise ExtensionInstallRefused(403, "install_disabled", _INSTALL_OFF)

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

        self._require_install_enabled()
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

        self._require_install_enabled()
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
