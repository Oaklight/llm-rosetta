"""The surface the admin panel depends on.

``admin/`` is served by the gateway but must not reach into its internals.
Everything it needs from the running gateway is declared here as a
:class:`~typing.Protocol`, grouped by capability, and implemented by one
adapter (``app/admin_context.py``) that ``create_app`` exposes as
``app.admin``.

Only members the admin code uses today belong here.  A new need adds a member
to the matching capability (or a new capability) and its adapter
implementation; it does not add a gateway import to ``admin/``.
``tests/gateway/test_admin_import_boundary.py`` enforces the boundary and
keeps the list of not-yet-migrated imports shrinking.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


if TYPE_CHECKING:
    import re

    from .config import ConfigIO, GatewayConfig

__all__ = [
    "AdminCatalogue",
    "AdminAudit",
    "AdminAuthState",
    "AdminConfigAccess",
    "AdminContext",
    "AdminJobs",
    "AdminMaintenance",
    "AdminPanelState",
    "AdminRuntime",
    "AdminStores",
    "AdminExtensions",
    "AdminServices",
    "AdminShimSources",
    "ConfigIO",
    "ExtensionInstallRefused",
    "GatewayConfig",
    "MaintenanceLoadTimeStep",
    "RetiredNameError",
    "admin_context",
    "app_admin_context",
    "bind_catalogue",
    "catalogue",
]


@runtime_checkable
class AdminConfigAccess(Protocol):
    """Read and replace the config the gateway is serving."""

    def served_config(self) -> GatewayConfig | None:
        """Return the config the gateway serves, or ``None`` if none is set."""
        ...

    def publish_config(self, config: GatewayConfig) -> None:
        """Make *config* the one the gateway serves from now on."""
        ...

    async def reload_config(
        self, config_path: str, *, force_shims: bool = False
    ) -> GatewayConfig:
        """Re-read *config_path*, rebuild the config and swap it in everywhere.

        Reapplies the declarative shim sources when their section changed (or
        *force_shims* is set), serves the new config, updates the live settings
        that read it, records the reload and seeds credential-minting providers
        again.  Returns the config now served.
        """
        ...


class RetiredNameError(ValueError):
    """A config entry spells a name that was retired.

    Attributes:
        detail: The retired-names message alone (it ends with the
            ``llm-rosetta gateway migrate`` command).
        where: The JSON path of the offending value, or ``None``.
        path: The config file, or ``None`` until the caller adds it.
    """

    def __init__(
        self, detail: str, *, where: str | None = None, path: str | None = None
    ) -> None:
        self.detail = detail
        self.where = where
        self.path = path
        super().__init__(": ".join(part for part in (path, where, detail) if part))

    def located(
        self, *, where: str | None = None, path: str | None = None
    ) -> RetiredNameError:
        """A copy with *where* and *path* filled in where they were not known."""
        return RetiredNameError(
            self.detail, where=self.where or where, path=self.path or path
        )


@runtime_checkable
class AdminRegistries(Protocol):
    """What the gateway knows about model types, provider types and limits.

    Stateless facts: the registries and constants the config editor validates
    against and the panel renders from.
    """

    def model_types(self) -> list[Any]:
        """Every registered model type descriptor, in registration order."""
        ...

    def registered_type_names(self) -> list[str]:
        """The names of the registered model types."""
        ...

    def retired_type_message(self, name: object) -> str | None:
        """The retired-names message when *name* is a retired model type."""
        ...

    def entry_capabilities(self) -> frozenset[str]:
        """The capability names a ``models`` entry may declare."""
        ...

    def declared_model_type(self, entry: Any, where: str | None = None) -> str:
        """The model type a ``models`` entry declares.

        Raises:
            RetiredNameError: The entry spells its type with a retired name.
        """
        ...

    def face_value(self, entry: dict[str, Any], key: str, type_name: str) -> Any:
        """One endpoint setting: the face block first, then the flat key."""
        ...

    def base_formats(self) -> tuple[str, ...]:
        """The base formats a provider may use, in picker order."""
        ...

    def base_type_hints(self, provider_type: str) -> dict[str, str]:
        """The placeholder ``base_url`` / ``api_key_env`` hints of a base format."""
        ...

    def default_supports_custom_tools(self, provider_type: str) -> bool:
        """Whether a base format supports custom tools by default."""
        ...

    def known_provider_types(self) -> list[str]:
        """The built-in provider type names."""
        ...

    def default_models_path(self, provider_type: str) -> str:
        """The model-listing path appended to a provider's ``base_url``."""
        ...

    def version_suffix_search(self, base_url: str) -> re.Match[str] | None:
        """Match a trailing ``/v1`` style version segment of *base_url*."""
        ...

    def max_custom_css(self) -> int:
        """The most characters ``server.admin_custom_css`` may hold."""
        ...

    def gateway_version(self) -> str:
        """The gateway distribution's version."""
        ...

    def parse_quota(self, quota: str) -> dict[str, Any]:
        """Parse a rate quota such as ``"100/s"``; raises ``ValueError`` if bad."""
        ...

    def detect_routing_loops(
        self, models: Mapping[str, Any], upstream_names: Mapping[str, str]
    ) -> list[str]:
        """Warnings for models whose upstream names route back into the gateway."""
        ...

    def embeddings_formats(self) -> list[str]:
        """The wire formats an ``embeddings`` provider may use."""
        ...

    def rerank_formats(self) -> list[str]:
        """The wire formats a ``rerank`` provider may use."""
        ...

    def config_lock(self, path: str) -> AbstractAsyncContextManager[None]:
        """Hold the in-process and cross-process lock on the config file *path*."""
        ...

    def provider_preflight_error(
        self, providers: Mapping[str, Any], data_dir: str | None = None
    ) -> str | None:
        """Build each provider in *providers* and return the first refusal.

        *providers* is the raw ``providers`` section with environment
        placeholders already expanded and disabled entries removed.  Nothing is
        kept: the point is to refuse a document the loader would reject before
        it is written.  *data_dir* is the resolved data directory.
        """
        ...

    def admin_cookie_name(self) -> str:
        """The name of the admin session cookie."""
        ...

    def request_is_https(self, request: Any) -> bool:
        """Whether *request* arrived over HTTPS, as the trusted-proxy policy sees it."""
        ...

    def public_token_status(self, status: Any) -> dict[str, Any]:
        """A credential-minting provider's token status, without any secret."""
        ...

    async def refresh_credential(self, pinfo: Any, *, debounce: bool = True) -> bool:
        """Re-mint the credential of *pinfo*; ``False`` when it cannot be."""
        ...

    def token_pending_sentinel(self) -> str:
        """The key a credential-minting provider holds until its first mint."""
        ...

    def account_token_for(self, account_id: str) -> str | None:
        """The opaque token that names an account in the panel's URLs."""
        ...


@runtime_checkable
class AdminQuotaRouting(Protocol):
    """What routing and quota say about a provider, read from process-wide state."""

    async def read_provider_quota(
        self, config: Any, name: str, *, force: bool = False
    ) -> dict[str, Any]:
        """Probe (or read the cached probe of) one provider's usage allowance."""
        ...

    def provider_allowance(self, pinfo: Any) -> Any:
        """One provider's allowance as a yes/no and a comparable number."""
        ...

    def provider_cooldown(self, name: str, pinfo: Any) -> dict[str, Any] | None:
        """A provider-wide block, or ``None``."""
        ...

    def lane_eligibility(
        self, pinfo: Any, name: str | None = None, *, lane: str
    ) -> Any:
        """Whether one lane of *pinfo* may serve, and why not."""
        ...

    def provider_eligibility(self, pinfo: Any, name: str | None = None) -> Any:
        """Whether *pinfo* may serve as a provider, and why not."""
        ...

    def rate_limited_reason(self) -> str:
        """The eligibility reason code for a rate-limited lane."""
        ...

    def parse_model_routing(self, raw: Any, *, label: str) -> Any:
        """Parse one model's ``routing`` block; raises ``ValueError`` if invalid."""
        ...


@runtime_checkable
class AdminShimSourceFacts(Protocol):
    """The declarative shim sources, read and applied without an app."""

    def shims_signature(self, raw: Mapping[str, Any] | None) -> str:
        """A stable hash of the ``shims`` section of *raw*."""
        ...

    def resolve_source_config(
        self, section: Mapping[str, Any] | None, config_path: str | None
    ) -> str:
        """Resolve a ``shims`` section into its cache root; raises if invalid."""
        ...

    def apply_shim_sources(
        self,
        raw: Mapping[str, Any],
        config_path: str | None = None,
        *,
        force_refresh: bool = False,
    ) -> Any:
        """Apply the declared sources and return the load report.  Blocking."""
        ...

    def check_source_updates(self, report: Any, *, proxy: str | None = None) -> Any:
        """Compare each installed source with what it publishes.  Blocking."""
        ...


@runtime_checkable
class AdminSupport(Protocol):
    """Vendored helpers, the update check and log vocabularies the panel uses."""

    def http_client(self, **options: Any) -> Any:
        """An async HTTP client; use it as ``async with``."""
        ...

    def is_http_response(self, value: Any) -> bool:
        """Whether *value* is a buffered response of :meth:`http_client`."""
        ...

    def yaml_load(self, text: str) -> Any:
        """Parse a YAML document."""
        ...

    def scrub_credential_patterns(self, text: str) -> str:
        """*text* with anything shaped like a credential removed."""
        ...

    def decompress_error_dump_body(self, data: bytes) -> dict[str, Any]:
        """Decode one stored error-dump body."""
        ...

    def ops_event_types(self) -> list[str]:
        """Every ops-log event type."""
        ...

    def ops_sources(self) -> list[str]:
        """Every ops-log source."""
        ...

    def internal_key_label(self) -> str:
        """The label under which the gateway's own requests are counted."""
        ...

    def internal_usage_bucket(self) -> str:
        """The usage subject of the gateway's own requests."""
        ...

    async def check_for_update(
        self, proxy: str | None = None, *, force: bool = False
    ) -> dict[str, Any]:
        """What PyPI offers and whether this process may update itself."""
        ...

    def read_changelog(self, language: str) -> str:
        """The changelog bundled with this build, as markdown."""
        ...

    async def install_update(self, version: str, proxy: str | None) -> dict[str, Any]:
        """Install *version* with pip; ``{"ok", "output", "dists"}``."""
        ...

    def restart_process(self) -> None:
        """Replace this process image with a fresh interpreter."""
        ...


@runtime_checkable
class AdminCatalogue(
    AdminRegistries,
    AdminQuotaRouting,
    AdminShimSourceFacts,
    AdminSupport,
    Protocol,
):
    """Everything the admin panel reads that needs no app.

    Process-wide facts and helpers: the registries and constants, the quota and
    eligibility state, the shim-source operations, and the vendored helpers.
    Bound once by ``create_app`` and reached with :func:`catalogue`.
    """


class MaintenanceLoadTimeStep(Exception):
    """A step that runs when a document is loaded, not on demand.

    Its context is the document a *previous* load read, so re-running it from a
    request would correct a stale copy and, for a caller that writes the result
    back, rewrite the config from it.  The surface refuses instead.
    """


@runtime_checkable
class AdminMaintenance(Protocol):
    """The maintenance steps of every store, for the maintenance page."""

    def maintenance_registries(self) -> list[Any]:
        """Every registry this app reports on, in the order an operator reads it.

        Each has ``name``, ``policy``, ``steps`` and ``results``.
        """
        ...

    async def run_maintenance_step(self, name: str) -> Any | None:
        """Run one step by name and return its result, or ``None`` if unknown.

        A step that fails still answers: the recorded result says why.

        Raises:
            MaintenanceLoadTimeStep: The step belongs to the load-time registry.
        """
        ...


@runtime_checkable
class AdminJobs(Protocol):
    """Queue the destructive data operations as background jobs."""

    async def enqueue_rebuild(self, manager: Any) -> Any:
        """Queue the counter rebuild."""
        ...

    async def enqueue_clear(self, manager: Any, table: str) -> Any:
        """Hide every row of *table* and queue its deletion."""
        ...

    async def enqueue_cleanup(self, manager: Any, **options: Any) -> Any:
        """Hide the rows a cleanup selects and queue their deletion."""
        ...

    async def enqueue_vacuum(self, manager: Any) -> Any:
        """Queue a database vacuum."""
        ...

    async def enqueue_trim(self, manager: Any) -> Any:
        """Queue a trim of the logs to their retention caps."""
        ...


@runtime_checkable
class AdminAudit(Protocol):
    """Record what the operator did in the ops log.

    Each does nothing when the app has no ops context.
    """

    async def record_password_change(self) -> None:
        """The admin password was changed."""
        ...

    async def record_token_rotation(self) -> None:
        """The internal proxy token was rotated."""
        ...

    async def record_sessions_cleared(self, count: int) -> None:
        """Every admin session was invalidated; *count* were open."""
        ...

    async def record_key_created(self, key_id: str, label: str) -> None:
        """A gateway API key was created."""
        ...

    async def record_key_updated(self, key_id: str, changed_fields: list[str]) -> None:
        """A gateway API key was edited."""
        ...

    async def record_key_deleted(self, key_id: str, label: str | None) -> None:
        """A gateway API key was deleted."""
        ...

    async def record_key_rotated(self, key_id: str, label: str | None) -> None:
        """A gateway API key was rotated."""
        ...

    async def record_self_update(self, version: str) -> None:
        """The gateway installed *version* and is restarting into it."""
        ...

    async def delete_error_dump(self, dump_id: str) -> bool | None:
        """Delete one error dump and audit it.

        Returns:
            ``True`` if it existed, ``False`` if not, ``None`` when the app has
            no persistence.
        """
        ...


@runtime_checkable
class AdminServices(Protocol):
    """The gateway services the admin panel runs on, started once per app."""

    async def start_services(
        self,
        config: GatewayConfig,
        config_path: str | None,
        config_io: ConfigIO | None,
        data_dir: str | None,
    ) -> None:
        """Build the metrics, logs, persistence, jobs and ops context.

        Stores them on the app (``metrics``, ``request_log``, ``ops_log``,
        ``persistence``, ``jobs``, ``ops_ctx``, ``config_path``, ``config_io``,
        ``profiler_state``, ``capture_state``).  *config_io* defaults to the
        JSONC reader; *data_dir* overrides the config's data directory.
        """
        ...

    async def record_admin_setup(self) -> None:
        """Write the audit row that says the admin panel is wired."""
        ...


@runtime_checkable
class AdminShimSources(Protocol):
    """What the manifest shim sources offer (#1660 S9)."""

    def shim_source_catalogue(
        self, config: Any, report: Any, *, refresh: bool = False
    ) -> dict[str, Any]:
        """Every manifest source *config* reads, with every vendor it publishes.

        Returns ``{"core": {surface, ir, version}, "sources": [...],
        "install": [...]}``.  Reads the manifests through the installer's
        cache; *refresh* refetches them.  Blocking: call it off the loop.
        """
        ...


@runtime_checkable
class AdminServiceState(Protocol):
    """The gateway's running services, which the observability routes read.

    Each returns the service the app holds, or ``None`` when the app was built
    without it (an in-memory gateway has no persistence and no job manager).
    """

    def persistence(self) -> Any | None:
        """The persistence manager, or ``None`` in memory mode."""
        ...

    def metrics(self) -> Any:
        """The metrics collector; ``None`` only for an app built by hand without it."""
        ...

    def request_log(self) -> Any:
        """The request log; ``None`` only for an app built by hand without it."""
        ...

    def ops_log(self) -> Any | None:
        """The ops log."""
        ...

    def job_manager(self) -> Any | None:
        """The background job manager, or ``None`` without persistence."""
        ...

    def rate_limit_state(self) -> Any | None:
        """The rate limiter state, or ``None`` when rate limiting is off."""
        ...

    def profiler_state(self) -> Any:
        """The request profiler's state; ``None`` only for an app built by hand without it."""
        ...

    def capture_state(self) -> Any:
        """The content capture state; ``None`` only for an app built by hand without it."""
        ...

    def admin_password_is_set(self) -> bool:
        """Whether an admin password protects the panel."""
        ...


class ExtensionInstallRefused(Exception):
    """An install the panel asked for did not happen.

    Attributes:
        status: The HTTP status that fits: 403 (``server.extension_install``
            is false), 404 (no such extension or program),
            400 (the program must be named), 409 (nothing to install, or one
            install of that extension is already running) or 502 (the install
            ran and failed).
        reason: A short code the panel can switch on.
        message: One sentence for the operator.
    """

    def __init__(self, status: int, reason: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.reason = reason
        self.message = message


@runtime_checkable
class AdminExtensions(Protocol):
    """What the installed extensions look like (epic #1703)."""

    def extension_listing(self) -> list[dict[str, Any]]:
        """Every extension found or installed, one JSON-safe record each.

        A record holds ``id``, ``version``, ``source`` (``entry point``,
        ``namespace`` or ``extensions directory``), ``state`` (``loaded``,
        ``skipped``, ``failed`` or ``installed`` when it is on disk and not yet
        loaded), ``reason``, ``path``, ``maturity``, ``shims`` (the names it
        registered), ``transports``, ``external`` (one status per program it
        drives) and ``missing_dependencies``.  A program's status holds
        ``state`` (the check's: ``ok``, ``missing``, ``broken``, ``too_old``,
        ``unknown``), ``status`` (the card's: ``ready``, ``missing``, ``broken``,
        ``too_old``, ``unknown``), ``found``, ``minimum``, ``install`` (the pinned
        ``{version, source}`` or ``None``) and ``install_command`` (the
        ``llm-rosetta ext install-cli`` line that installs it, or ``None`` when
        it is ready or declares no install).  The check also looks in the
        directory ``ext install-cli`` fills.  Blocking: it runs the external
        programs' version checks, so call it off the loop.
        """
        ...

    async def install_extension_cli(
        self, ext_id: str, program: str | None
    ) -> dict[str, Any]:
        """Install a program an extension declares, at the version it pins.

        Only the extension id and the program's name come from the caller; the
        version, the source and the checksum are the manifest's.  One install
        per extension runs at a time.  The attempt is recorded in the ops log
        (extension, program, version and outcome; never the environment or the
        install's output).

        Returns:
            ``{"extension", "program", "state", "version", "found"}``; *state*
            is ``installed`` or ``unchanged``.

        Raises:
            ExtensionInstallRefused: The request names nothing installable, one
                install of that extension is running, or the install failed.
        """
        ...

    def extension_removal_plan(self, ext_id: str) -> dict[str, Any] | None:
        """What removing the extension deletes and keeps, or ``None`` if unknown.

        ``{"id", "version", "origin", "removable", "delete", "keep", "hint"}``;
        *delete* and *keep* are lists of ``{path, size, note}``.  *origin* is
        ``directory`` (removable) or ``image`` (provided by the image, e.g. a pip
        package; *hint* says how to remove it).  Blocking: it measures sizes.
        """
        ...

    async def remove_extension(
        self, ext_id: str, *, replace: bool = False
    ) -> dict[str, Any]:
        """Remove an extension and the programs it installed, as the plan says.

        One operation per extension runs at a time.  The attempt is recorded in
        the ops log (extension, version, paths deleted or failed, paths kept).

        Returns:
            ``{"extension", "deleted", "failed"}``.

        Raises:
            ExtensionInstallRefused: The extension is unknown (404), provided by
                the image (409 ``not_removable``), busy (409) or the removal
                failed (502).
        """
        ...

    async def check_extension_updates(
        self, config: Any, report: Any, ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Compare installed extensions with the manifest sources, refetching them.

        One record per extension: ``id``, ``current``, ``available`` (the newest
        release this core runs), ``newest``, ``source``, ``versions`` and
        ``status`` (``update``, ``current``, ``not_published``,
        ``no_compatible`` or ``failed``) with ``error`` for a failure.  A source
        that cannot be read is named in the error, never skipped silently.
        """
        ...

    async def update_extension(
        self, ext_id: str, config: Any, report: Any, version: str | None = None
    ) -> dict[str, Any]:
        """Install the newest release (or *version*) of an extension from its source.

        Recorded in the ops log (extension, version before and after).

        Returns:
            ``{"extension", "before", "after"}``.

        Raises:
            ExtensionInstallRefused: Unknown extension or version (404/400),
                provided by the image (409), busy (409) or the install failed
                (502).
        """
        ...


@runtime_checkable
class AdminPanelState(Protocol):
    """The panel's own page state: its head fragment and the hidden tabs."""

    def admin_custom_head(self) -> str:
        """The HTML injected before ``</head>`` of the admin page (``""`` if none)."""
        ...

    def set_admin_custom_head(self, head: str) -> None:
        """Replace the head the admin page is served with."""
        ...

    def admin_head_prefix(self) -> str:
        """The head fragments built at setup, before the operator's CSS override."""
        ...

    def set_admin_head_prefix(self, prefix: str) -> None:
        """Store the head fragments a theme rebuild starts from."""
        ...

    def disabled_tabs(self) -> Any:
        """The tab identifiers hidden from the panel (a set; empty if none)."""
        ...

    def set_disabled_tabs(self, tabs: Any) -> None:
        """Set the tab identifiers hidden from the panel."""
        ...


@runtime_checkable
class AdminAuthState(Protocol):
    """The admin sign-in state and the token the panel calls the gateway with."""

    def auth_state(self) -> Any:
        """The mutable auth state: sessions, the password, the login limiter."""
        ...

    def internal_token(self) -> str:
        """The ephemeral token the panel's own test requests carry (``""`` if none)."""
        ...

    def set_internal_token(self, token: str) -> None:
        """Make *token* the internal token (after a rotation)."""
        ...


@runtime_checkable
class AdminStores(Protocol):
    """The key store the admin routes read, and the usage report's short cache."""

    def keystore(self) -> Any | None:
        """The API key store, or ``None`` when none is configured."""
        ...

    def usage_payload_cache(self) -> Any:
        """The cached per-key usage report as ``(monotonic time, payload)``, or ``None``."""
        ...

    def set_usage_payload_cache(self, value: Any) -> None:
        """Store the per-key usage report with the time it was built."""
        ...


@runtime_checkable
class AdminRuntime(Protocol):
    """What the running process offers the panel: where it listens and its lanes."""

    def bind_address(self) -> tuple[str, int]:
        """The host and port the gateway listens on (``127.0.0.1``, ``28765`` by default)."""
        ...

    def transports(self) -> dict[str, Any]:
        """The non-HTTP transports by the name a provider declares."""
        ...

    def deferred_startup(self) -> Any | None:
        """The deferred-startup controller, or ``None`` when startup was not deferred."""
        ...

    def config_path(self) -> str | None:
        """The config file this gateway was started from, or ``None`` when it has none."""
        ...

    def config_io(self) -> Any | None:
        """The config reader and writer (``ConfigIO``), or ``None`` when none is set."""
        ...


@runtime_checkable
class AdminShimSourceState(Protocol):
    """The shim-source report the gateway holds, and what the last reload found wrong."""

    def shim_source_report(self) -> Any | None:
        """The report of the last shim-source apply, or ``None`` before one ran."""
        ...

    def shim_source_signature(self) -> str | None:
        """The signature of the ``shims`` section that report was built from."""
        ...

    def shim_update_check(self) -> Any | None:
        """The cached update verdicts for that report, or ``None`` when none are held."""
        ...

    def set_shim_update_check(self, payload: Any | None) -> None:
        """Cache the update verdicts for the current report, or drop them with ``None``."""
        ...

    def install_shim_report(self, report: Any, signature: str) -> None:
        """Record a freshly applied report; the verdicts cached for the old one are dropped."""
        ...

    def provider_validation_warnings(self) -> list[Any]:
        """The providers the last reload disabled, with the reason for each."""
        ...


@runtime_checkable
class AdminContext(
    AdminAudit,
    AdminAuthState,
    AdminConfigAccess,
    AdminExtensions,
    AdminJobs,
    AdminMaintenance,
    AdminPanelState,
    AdminRuntime,
    AdminServices,
    AdminServiceState,
    AdminShimSources,
    AdminShimSourceState,
    AdminStores,
    Protocol,
):
    """Everything the admin panel may ask of the running gateway.

    Capabilities are base classes, one per concern: the audit trail, config,
    extensions, jobs, maintenance, the services admin runs on, and shim sources.
    What needs no app (registries, constants, the quota and eligibility state,
    vendored helpers) is on :class:`AdminCatalogue` instead.
    """


_catalogue: AdminCatalogue | None = None


def bind_catalogue(impl: AdminCatalogue | None) -> None:
    """Set the :class:`AdminCatalogue` that :func:`catalogue` returns.

    ``create_app`` binds the gateway's own; a test binds a stand-in, or
    ``None`` to unbind.
    """
    global _catalogue
    _catalogue = impl


def catalogue() -> AdminCatalogue:
    """Return the bound :class:`AdminCatalogue`.

    The registries it reads are process-wide, so no app is needed.

    Raises:
        RuntimeError: Nothing is bound: ``create_app`` binds it.
    """
    if _catalogue is None:
        raise RuntimeError(
            "no admin catalogue is bound: create_app binds it, and admin needs it"
        )
    return _catalogue


def admin_context(request: Any) -> AdminContext:
    """Return the :class:`AdminContext` the app of *request* exposes.

    Raises:
        RuntimeError: The app has no ``admin`` context.
    """
    return app_admin_context(request.app)


def app_admin_context(app: Any) -> AdminContext:
    """Return the :class:`AdminContext` that *app* exposes.

    ``create_app`` sets ``app.admin``; an app that lacks it was not built by
    ``create_app``, and admin has nothing to fall back on.

    Raises:
        RuntimeError: The app has no ``admin`` context.
    """
    context = getattr(app, "admin", None)
    if context is None:
        raise RuntimeError(
            "app.admin is not set: create_app sets it, and admin needs it"
        )
    return context  # type: ignore[no-any-return]
