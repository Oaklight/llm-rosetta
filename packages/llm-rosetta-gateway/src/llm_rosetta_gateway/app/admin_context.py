"""The adapter that implements :class:`~llm_rosetta_gateway.admin_api.AdminContext`."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from ..quota.lane_allowance import bind_lane_allowance
from ..shims.extension_view import pick_release, update_report

if TYPE_CHECKING:
    from ..config import ConfigIO, GatewayConfig


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


class _AppStateMembers:
    """The app attributes the admin routes read, behind their typed members.

    Implements :class:`~llm_rosetta_gateway.admin_api.AdminPanelState`,
    ``AdminAuthState``, ``AdminStores`` and ``AdminRuntime``.  Each member reads
    the attribute when it is asked, so a value set after the context was built
    is seen.
    """

    _app: Any

    # -- AdminPanelState ----------------------------------------------------

    def admin_custom_head(self) -> str:
        return getattr(self._app, "admin_custom_head", "")  # type: ignore[no-any-return]

    def set_admin_custom_head(self, head: str) -> None:
        self._app.admin_custom_head = head

    def admin_head_prefix(self) -> str:
        return getattr(self._app, "_admin_head_prefix", "")  # type: ignore[no-any-return]

    def set_admin_head_prefix(self, prefix: str) -> None:
        self._app._admin_head_prefix = prefix

    def disabled_tabs(self) -> Any:
        return getattr(self._app, "disabled_tabs", frozenset())

    def set_disabled_tabs(self, tabs: Any) -> None:
        self._app.disabled_tabs = tabs

    # -- AdminAuthState -----------------------------------------------------

    def auth_state(self) -> Any:
        return self._app.auth_state

    def internal_token(self) -> str:
        return getattr(self._app, "internal_token", None) or ""

    def set_internal_token(self, token: str) -> None:
        self._app.internal_token = token

    # -- AdminStores --------------------------------------------------------

    def keystore(self) -> Any | None:
        return getattr(self._app, "keystore", None)

    def usage_payload_cache(self) -> Any:
        return getattr(self._app, "_usage_payload", None)

    def set_usage_payload_cache(self, value: Any) -> None:
        self._app._usage_payload = value

    # -- AdminRuntime -------------------------------------------------------

    def bind_address(self) -> tuple[str, int]:
        return (
            getattr(self._app, "_bind_host", "127.0.0.1"),
            getattr(self._app, "_bind_port", 28765),
        )

    def transports(self) -> dict[str, Any]:
        return getattr(self._app, "transports", {}) or {}

    def deferred_startup(self) -> Any | None:
        return getattr(self._app, "deferred_startup", None)

    def config_path(self) -> str | None:
        return getattr(self._app, "config_path", None)

    def config_io(self) -> Any | None:
        return getattr(self._app, "config_io", None)


class GatewayAdminContext(_AppStateMembers):
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
        store = getattr(self._app, "lane_allowance", None)
        if store is not None:
            bind_lane_allowance(config, store)
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

    async def _audit(self, build: Any) -> None:
        """Run the op *build* makes from the ops context, if the app has one."""
        ctx = getattr(self._app, "ops_ctx", None)
        if ctx is not None:
            await build(ctx).execute()

    async def record_password_change(self) -> None:
        """The admin password was changed."""
        from ..ops.security import OpsPasswordChange

        await self._audit(OpsPasswordChange)

    async def record_token_rotation(self) -> None:
        """The internal proxy token was rotated."""
        from ..ops.security import OpsTokenRotate

        await self._audit(OpsTokenRotate)

    async def record_sessions_cleared(self, count: int) -> None:
        """Every admin session was invalidated; *count* were open."""
        from ..ops.security import OpsSessionLogoutAll

        await self._audit(lambda ctx: OpsSessionLogoutAll(ctx, count=count))

    async def record_key_created(self, key_id: str, label: str) -> None:
        """A gateway API key was created."""
        from ..ops.keys import OpsKeyCreate

        await self._audit(lambda ctx: OpsKeyCreate(ctx, key_id=key_id, label=label))

    async def record_key_updated(self, key_id: str, changed_fields: list[str]) -> None:
        """A gateway API key was edited."""
        from ..ops.keys import OpsKeyUpdate

        await self._audit(
            lambda ctx: OpsKeyUpdate(ctx, key_id=key_id, changed_fields=changed_fields)
        )

    async def record_key_deleted(self, key_id: str, label: str | None) -> None:
        """A gateway API key was deleted."""
        from ..ops.keys import OpsKeyDelete

        await self._audit(lambda ctx: OpsKeyDelete(ctx, key_id=key_id, label=label))

    async def record_key_rotated(self, key_id: str, label: str | None) -> None:
        """A gateway API key was rotated."""
        from ..ops.keys import OpsKeyRotate

        await self._audit(lambda ctx: OpsKeyRotate(ctx, key_id=key_id, label=label))

    async def record_self_update(self, version: str) -> None:
        """The gateway installed *version* and is restarting into it."""
        from ..ops.security import OpsSelfUpdate

        await self._audit(lambda ctx: OpsSelfUpdate(ctx, version=version))

    async def delete_error_dump(self, dump_id: str) -> bool | None:
        """Delete one error dump and audit it; ``None`` without persistence."""
        from ..ops.data import OpsDeleteErrorDump

        ctx = getattr(self._app, "ops_ctx", None)
        if ctx is None or ctx.persistence is None:
            return None
        return bool(await OpsDeleteErrorDump(ctx, dump_id=dump_id).execute())

    async def enqueue_rebuild(self, manager: Any) -> Any:
        """Queue the counter rebuild."""
        from ..jobs.enqueue import enqueue_rebuild

        return await enqueue_rebuild(manager)

    async def enqueue_clear(self, manager: Any, table: str) -> Any:
        """Hide every row of *table* and queue its deletion."""
        from ..jobs.enqueue import enqueue_clear

        return await enqueue_clear(manager, table)

    async def enqueue_cleanup(self, manager: Any, **options: Any) -> Any:
        """Hide the rows a cleanup selects and queue their deletion."""
        from ..jobs.enqueue import enqueue_cleanup

        return await enqueue_cleanup(manager, **options)

    async def enqueue_vacuum(self, manager: Any) -> Any:
        """Queue a database vacuum."""
        from ..jobs.enqueue import enqueue_vacuum

        return await enqueue_vacuum(manager)

    async def enqueue_trim(self, manager: Any) -> Any:
        """Queue a trim of the logs to their retention caps."""
        from ..jobs.enqueue import enqueue_trim

        return await enqueue_trim(manager)

    def maintenance_registries(self) -> list[Any]:
        """Every registry this app reports on, in the order an operator reads it."""
        from .admin_maintenance import registries_for

        return registries_for(self._app)

    async def run_maintenance_step(self, name: str) -> Any | None:
        """Run one step by name and return its result, or ``None`` if unknown."""
        from .admin_maintenance import run_step

        return await run_step(self._app, name)

    async def start_services(
        self,
        config: GatewayConfig,
        config_path: str | None,
        config_io: ConfigIO | None,
        data_dir: str | None,
    ) -> None:
        """Build the services the admin panel runs on and store them on the app."""
        from .admin_services import start_admin_services

        await start_admin_services(self._app, config, config_path, config_io, data_dir)

    async def record_admin_setup(self) -> None:
        """Write the audit row that says the admin panel is wired."""
        from ..ops import OpsAdminSetup

        await OpsAdminSetup(self._app.ops_ctx).execute()

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

    # -- the running services (the observability routes) ---------------------

    def persistence(self) -> Any | None:
        """The persistence manager, or ``None`` in memory mode."""
        return getattr(self._app, "persistence", None)

    def metrics(self) -> Any:
        """The metrics collector."""
        return getattr(self._app, "metrics", None)

    def request_log(self) -> Any:
        """The request log."""
        return getattr(self._app, "request_log", None)

    def ops_log(self) -> Any | None:
        """The ops log."""
        return getattr(self._app, "ops_log", None)

    def job_manager(self) -> Any | None:
        """The background job manager, or ``None`` without persistence."""
        return getattr(self._app, "jobs", None)

    def rate_limit_state(self) -> Any | None:
        """The rate limiter state, or ``None`` when rate limiting is off."""
        return getattr(self._app, "rate_limit_state", None)

    def profiler_state(self) -> Any:
        """The request profiler's state."""
        return getattr(self._app, "profiler_state", None)

    def capture_state(self) -> Any:
        """The content capture state."""
        return getattr(self._app, "capture_state", None)

    def admin_password_is_set(self) -> bool:
        """Whether an admin password protects the panel."""
        auth_state = getattr(self._app, "auth_state", None)
        return bool(getattr(auth_state, "admin_password", None))

    # -- AdminShimSourceState -------------------------------------------------

    def shim_source_report(self) -> Any | None:
        """The report of the last shim-source apply."""
        return getattr(self._app, "shim_source_report", None)

    def shim_source_signature(self) -> str | None:
        """The signature of the ``shims`` section that report came from."""
        return getattr(self._app, "shim_source_signature", None)

    def shim_update_check(self) -> Any | None:
        """The cached update verdicts for the current report."""
        return getattr(self._app, "shim_update_check", None)

    def set_shim_update_check(self, payload: Any | None) -> None:
        """Cache the update verdicts for the current report."""
        self._app.shim_update_check = payload

    def install_shim_report(self, report: Any, signature: str) -> None:
        """Record a freshly applied report and drop the verdicts cached for the old one."""
        self._app.shim_source_report = report
        self._app.shim_source_signature = signature
        self._app.shim_update_check = None

    def provider_validation_warnings(self) -> list[Any]:
        """The providers the last reload disabled."""
        return list(getattr(self._app, "provider_validation_warnings", None) or [])
