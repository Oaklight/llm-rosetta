"""App factory, extension points, auth/CORS/lifecycle installation, and the runner."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from llm_rosetta._compat import sync_compat as _sync_compat

from ._state import set_config

if TYPE_CHECKING:
    from ..shims import ShimLoadReport

from llm_rosetta_gateway._vendor.httpserver import (
    App,
    JSONResponse,
    Response,
)

from ..middleware.auth import (
    AuthState,
    create_auth_hook,
)
from ..config import GatewayConfig
from ..middleware.error_format import (
    apply_cors_headers,
    is_admin_path as _is_admin_path,
)
from ..keystore import KeyStore
from ..middleware.request_context import request_context_var, setup_request_context
from ..ops.rejected import record_read_rejection, record_unrecorded_rejection
from ..logging import get_logger

from ..proxy import (
    ProviderMetadataStore,
    close_resources,
)
from .routes_compact import (
    route_responses_compact,
)
from .routes_llm import (
    _register_llm_routes,
)
from .routes_typed import (
    _register_non_llm_routes,
)
from .routes_models import (
    handle_list_models,
    handle_list_models_google,
)
from .routes_health import (
    handle_health,
    handle_health_live,
    handle_health_ready,
)
from .tasks import (
    _CLEANUP_INITIAL_DELAY,
    _CLEANUP_INTERVAL,
    _flush_now,
    _periodic_cleanup,
    _periodic_flush,
    _periodic_token_rotation,
)

logger = get_logger()


def _resolve_data_dir_for_app(
    config: GatewayConfig, config_path: str | None
) -> str | None:
    """Resolve data directory from config, anchoring relative paths to config dir.

    The rule lives in :func:`llm_rosetta_gateway.config.settings.resolve_data_dir`
    — the config build needs the same answer for its providers' credential
    records — and is idempotent on an already-resolved value.
    """
    from ..config.settings import resolve_data_dir

    return resolve_data_dir(config.data_dir, config_path or config.config_path)


async def _setup_auth(
    config: GatewayConfig,
    config_path: str | None,
    data_dir: str | None = None,
) -> tuple[str, KeyStore, AuthState]:
    """Create KeyStore and build AuthState."""
    import os
    import secrets

    internal_token = f"rsk-internal-{secrets.token_hex(16)}"

    if config.api_keys_db:
        keys_db_path = config.api_keys_db
    elif data_dir:
        new_path = os.path.join(data_dir, "keys.db")
        old_path = (
            os.path.join(os.path.dirname(os.path.abspath(config_path)), "keys.db")
            if config_path
            else None
        )
        if os.path.exists(new_path):
            keys_db_path = new_path
        elif old_path and os.path.exists(old_path):
            keys_db_path = old_path
            logger.warning(
                "Using keys.db from legacy location %s — "
                "move it to %s to silence this warning.",
                old_path,
                new_path,
            )
        else:
            keys_db_path = new_path
    elif config_path:
        keys_db_path = os.path.join(
            os.path.dirname(os.path.abspath(config_path)), "keys.db"
        )
    else:
        keys_db_path = "keys.db"

    keystore = await KeyStore.create(keys_db_path)

    auth_state = AuthState(
        keystore=keystore,
        internal_token=internal_token,
        admin_password=config.admin_password,
        open_on_no_keys=config.open_on_no_keys,
        session_idle_minutes=config.admin_session_idle_minutes,
        session_remember_days=config.admin_session_remember_days,
    )
    if not auth_state._has_keys():
        _no_key_msg = (
            "No API keys configured — all /v1/* endpoints are OPEN "
            "(server.open_on_no_keys=true). Generate keys in the admin "
            "panel to restrict access."
            if config.open_on_no_keys
            else "No API keys configured — all /v1/* endpoints are BLOCKED. "
            "Generate keys in the admin panel, or set "
            "server.open_on_no_keys=true to allow anonymous access."
        )
        logger.warning(_no_key_msg)

    return internal_token, keystore, auth_state


def _install_root_redirect(app: App, target: str | None) -> None:
    """Register a GET / → *target* redirect when *target* is set."""
    if not target:
        return

    @app.route("/", methods=["GET"])
    async def root_redirect(request: Any) -> Response:
        return Response(body=b"", status_code=307, headers={"Location": target})


@dataclass
class GatewayExtensions:
    """Extension points for downstream projects that wrap the gateway.

    Pass an instance to :func:`create_app` to customise transport, auth,
    middleware, routes, and admin panel without duplicating the factory.
    All fields are optional — defaults preserve the standard gateway behaviour.
    """

    transport: Any | None = None
    """Pre-built transport instance.  When *None*, an :class:`HttpTransport`
    is created from *config.upstream_timeout*."""

    before_hooks: list[Callable] = field(default_factory=list)
    """Extra ``before_request`` hooks registered **after** built-in auth."""

    after_hooks: list[Callable] = field(default_factory=list)
    """Extra ``after_request`` hooks registered **after** built-in CORS."""

    extra_routes: list[tuple[str, list[str], Callable]] = field(default_factory=list)
    """Additional routes as ``(path, methods, handler)`` tuples."""

    branding: dict | None = None
    """Admin panel branding dict forwarded to :func:`setup_admin`."""

    custom_head: str | None = None
    """Extra ``<head>`` HTML injected into the admin page."""

    disabled_tabs: list[str] | None = None
    """Admin tabs to hide (e.g. ``["keys"]``)."""

    config_io: Any | None = None
    """Custom :class:`ConfigIO` for admin config persistence."""

    max_body_size: int | None = None
    """Override the default 50 MB max request body size."""

    enable_rate_limiting: bool = True
    """Set to *False* to skip built-in rate-limiting middleware."""

    skip_default_routes: bool = False
    """When *True*, the standard proxy routes are **not** registered.
    Use this when the downstream project provides its own handlers."""

    skip_builtin_auth: bool = False
    """When *True*, the API-key auth hook is **not** registered.
    ``_setup_auth`` still runs (admin needs *internal_token*)."""

    skip_admin_setup: bool = False
    """When *True*, :func:`setup_admin` is **not** called.
    The downstream project is responsible for calling it later
    (e.g. after async initialisation)."""


def _install_cors(app: App, admin_cors_origins: list[str]) -> None:
    """Register CORS after-request hook and OPTIONS preflight handler."""

    def _apply_cors(response: Any, origin: str | None) -> None:
        if admin_cors_origins and origin and origin in admin_cors_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
            response.headers["Access-Control-Allow-Methods"] = "*"
            response.headers["Access-Control-Allow-Headers"] = "*"

    @app.after_request
    async def add_cors_headers(request: Any, response: Any) -> Any:
        if _is_admin_path(request.path):
            _apply_cors(response, request.headers.get("origin"))
            if request.path.startswith("/admin/api/"):
                response.headers.setdefault(
                    "Cache-Control", "no-cache, no-store, must-revalidate"
                )
        else:
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "*"
            response.headers["Access-Control-Allow-Headers"] = "*"
        return response

    @app.route("/<path:_path>", methods=["OPTIONS"])
    async def cors_preflight(request: Any, _path: str = "") -> Response:
        resp = Response(body=b"", status_code=204)
        if not _is_admin_path(request.path):
            resp.headers["Access-Control-Allow-Origin"] = "*"
            resp.headers["Access-Control-Allow-Methods"] = "*"
            resp.headers["Access-Control-Allow-Headers"] = "*"
        else:
            _apply_cors(resp, request.headers.get("origin"))
        return resp


def _install_rate_limiting(app: App, config: GatewayConfig) -> None:
    """Set up rate-limiting before/after hooks and attach state to *app*."""
    from ..middleware.ratelimit import (
        RateLimitState,
        create_rate_limit_after_hook,
        create_rate_limit_hook,
    )

    rate_limit_state = RateLimitState()
    rate_limit_state.rebuild(config)
    app.before_request(create_rate_limit_hook(rate_limit_state))
    app.rate_limit_state = rate_limit_state  # type: ignore
    app.after_request(create_rate_limit_after_hook())


def _install_rejection_recording(app: App, config: GatewayConfig) -> None:
    """Log the requests refused before any route handler saw them (#1217).

    Two signals cover the two places that happens: the response signal sees
    every error answered after dispatch began (auth, rate limit, no route), and
    the rejection signal sees the reads the server refused before a request
    existed.
    """
    trust_proxy = config.rate_limit_trust_proxy

    @app.on_response_started
    async def _record_rejection(request: Any, response: Any) -> None:
        await record_unrecorded_rejection(request, response)

    @app.on_request_rejected
    async def _record_read_rejection(rejection: Any) -> None:
        await record_read_rejection(app, rejection, trust_proxy=trust_proxy)


def _install_lifecycle_hooks(app: App) -> None:
    """Register httpserver response lifecycle hooks for TTFB and disconnect tracking."""

    @app.on_response_started
    async def _on_response_started(request: Any, response: Any) -> None:
        ctx = request_context_var.get()
        if ctx is None or ctx.request_start is None:
            return
        ttfb_ms = round((time.monotonic() - ctx.request_start) * 1000, 2)
        request.state.ttfb_ms = ttfb_ms

    @app.on_response_completed
    async def _on_response_completed(request: Any, response: Any) -> None:
        ttfb_ms = getattr(request.state, "ttfb_ms", None)
        entry_id = getattr(request.state, "log_entry_id", None)
        if ttfb_ms is not None and entry_id is not None:
            request_log = getattr(request.app, "request_log", None)
            if request_log is not None:
                await request_log.update_profile(entry_id, {"ttfb_ms": ttfb_ms})

    @app.on_client_disconnect
    async def _on_client_disconnect(request: Any) -> None:
        metrics = getattr(request.app, "metrics", None)
        if metrics is not None:
            metrics.record_disconnect()

        entry_id = getattr(request.state, "log_entry_id", None)
        if entry_id is not None:
            request_log = getattr(request.app, "request_log", None)
            if request_log is not None:
                await request_log.update_profile(
                    entry_id, {"client_disconnected": True}
                )

        ctx = request_context_var.get()
        logger.debug(
            "Client disconnected mid-response: client=%s path=%s request_id=%s",
            ctx.client_ip if ctx else "unknown",
            request.path,
            ctx.request_id if ctx else "unknown",
        )


def _report_shim_removals(config: GatewayConfig, report: ShimLoadReport | None) -> None:
    """Warn about providers still on a built-in shim that is going away.

    Logged here, once, because both the CLI and a programmatic caller pass
    through ``create_app``.  The report is only an input: a shim a declared
    source supplies is not going anywhere, so it is not reported.
    """
    from ..shim_sources import report_shim_removals

    report_shim_removals(config.raw_providers, report)


@_sync_compat
async def create_app(
    config: GatewayConfig,
    config_path: str | None = None,
    extensions: GatewayExtensions | None = None,
    *,
    shim_source_report: ShimLoadReport | None = None,
    load_shim_sources: bool = True,
) -> App:
    """Create the httpserver application.

    Args:
        config: The parsed gateway config.
        config_path: Path to the config file being served.
        extensions: Optional host hooks.
        shim_source_report: Result of an earlier shim-source application.  The
            CLI passes its report so the sources are not fetched twice; when
            ``None`` and ``config.shims`` is set, the section is applied here so
            programmatic callers that built ``GatewayConfig`` directly still get
            their declared shims.
        load_shim_sources: Set ``False`` to skip the automatic application
            (tests, or callers that already applied the sources themselves).
    """
    set_config(config)
    ext = extensions or GatewayExtensions()

    if shim_source_report is None and load_shim_sources and config.shims:
        from ..shim_sources import apply_shim_section

        shim_source_report = apply_shim_section(
            config.shims, config_path, proxy=config.proxy
        )

    _report_shim_removals(config, shim_source_report)

    # Expose global proxy as env vars so downstream code (e.g. image
    # downloads in converters) can use it without threading config through.
    import os

    if config.proxy:
        os.environ.setdefault("HTTP_PROXY", config.proxy)
        os.environ.setdefault("HTTPS_PROXY", config.proxy)

    from ..transport import HttpTransport

    metadata_store = ProviderMetadataStore()
    transport = (
        ext.transport
        if ext.transport is not None
        else HttpTransport(timeout=config.upstream_timeout)
    )

    app = App(
        max_body_size=ext.max_body_size
        if ext.max_body_size is not None
        else 50_000_000,
        read_timeout=config.read_timeout,
    )

    # Expose the shim-source outcome to the admin API, and record the applied
    # signature so the reload path can skip a rebuild when the section is
    # unchanged.  ``App`` does not declare these, so set them dynamically.
    from ..shim_sources import shims_signature as _shims_signature

    setattr(app, "shim_source_report", shim_source_report)
    setattr(app, "shim_source_signature", _shims_signature({"shims": config.shims}))

    # --- Routes ---
    if not ext.skip_default_routes:
        # LLM routes: registered from the model type registry
        _register_llm_routes(app)

        # Non-LLM routes: registered from the model type registry
        _register_non_llm_routes(app, config)

        # Context compaction: an open_responses-family operation with no
        # neutral form, so it is relayed rather than converted (see the module).
        app.route("/v1/responses/compact", methods=["POST"])(route_responses_compact)

        # Utility / listing / health routes
        app.route("/v1/models", methods=["GET"])(handle_list_models)
        app.route("/v1beta/models", methods=["GET"])(handle_list_models_google)
        app.route("/health", methods=["GET"])(handle_health)
        app.route("/health/live", methods=["GET"])(handle_health_live)
        app.route("/health/ready", methods=["GET"])(handle_health_ready)

    _install_root_redirect(app, config.root_redirect)

    # --- Resolve data directory (shared by keystore + persistence) ---
    resolved_data_dir = _resolve_data_dir_for_app(config, config_path)

    # --- Request context (earliest hook — before auth and rate limiting) ---
    app.before_request(setup_request_context(trust_proxy=config.rate_limit_trust_proxy))

    # --- Hop-count loop detection (before auth — no point authenticating a loop) ---
    from ..middleware.hop_limit import create_hop_limit_hook

    app.before_request(create_hop_limit_hook())

    # --- Auth (SQLite keystore + config fallback) ---
    internal_token, keystore, auth_state = await _setup_auth(
        config, config_path, data_dir=resolved_data_dir
    )
    if not ext.skip_builtin_auth:
        app.before_request(create_auth_hook(auth_state))

    # --- Rate Limiting (after auth so api_key_context_var is set) ---
    if ext.enable_rate_limiting:
        _install_rate_limiting(app, config)

    # --- Extension hooks (after built-in auth/rate-limit) ---
    for hook in ext.before_hooks:
        app.before_request(hook)

    _install_cors(app, config.admin_cors_origins)

    for hook in ext.after_hooks:
        app.after_request(hook)

    # --- Extension routes ---
    for path, methods, handler in ext.extra_routes:
        app.route(path, methods=methods)(handler)

    # --- .well-known (always registered, not a proxy route) ---
    @app.route("/.well-known/change-password", methods=["GET"])
    async def well_known_change_password(request: Any) -> Response:
        return Response(
            body=b"", status_code=302, headers={"Location": "/admin#change-password"}
        )

    @app.errorhandler(404)
    async def handle_404(request: Any, exc: Any) -> Response:
        resp = JSONResponse({"error": "Not Found"}, status_code=404)
        if not _is_admin_path(request.path):
            apply_cors_headers(resp)
        return resp

    @app.errorhandler(405)
    async def handle_405(request: Any, exc: Any) -> Response:
        resp = JSONResponse({"error": "Method Not Allowed"}, status_code=405)
        if not _is_admin_path(request.path):
            apply_cors_headers(resp)
        return resp

    # --- Admin routes ---
    from ..admin import setup_admin
    from ..admin.routes import register_admin_routes

    register_admin_routes(app)

    # --- App-level state ---
    app.transport = transport  # type: ignore
    app.metadata_store = metadata_store  # type: ignore
    app.internal_token = internal_token  # type: ignore
    app.auth_state = auth_state  # type: ignore
    app.keystore = keystore  # type: ignore

    if not ext.skip_admin_setup:
        await setup_admin(
            app,
            config,
            config_path,
            config_io=ext.config_io,
            disabled_tabs=ext.disabled_tabs,
            custom_head=ext.custom_head,
            branding=ext.branding,
            data_dir=resolved_data_dir,
        )

    # --- Response lifecycle hooks (httpserver 0.5.0+) ---
    _install_lifecycle_hooks(app)
    _install_rejection_recording(app, config)

    return app


async def run_gateway(
    app: App,
    host: str,
    port: int,
    *,
    socket: str | None = None,
    ssl_context: Any | None = None,
) -> None:
    """Start the gateway with lifecycle management."""
    # Expose bind address so admin test tasks can self-call.
    setattr(app, "_bind_host", host)
    setattr(app, "_bind_port", port)
    # Record startup event
    ops_log = getattr(app, "ops_log", None)
    if ops_log is not None:
        from llm_rosetta_gateway.observability.ops_log import (
            EVENT_STARTUP,
            OpsLogEntry,
            SEVERITY_INFO,
            SOURCE_GATEWAY,
        )

        config = getattr(app, "gateway_config", None)
        await ops_log.add(
            OpsLogEntry.create(
                event_type=EVENT_STARTUP,
                severity=SEVERITY_INFO,
                message=f"Gateway starting on {host}:{port}",
                details={
                    "host": host,
                    "port": port,
                    "provider_count": len(config.providers) if config else 0,
                    "model_count": len(config.models) if config else 0,
                },
                source=SOURCE_GATEWAY,
            )
        )

    flush_task = asyncio.create_task(_periodic_flush(app))
    cleanup_task = asyncio.create_task(_periodic_cleanup(app))
    rotation_task = asyncio.create_task(_periodic_token_rotation(app))

    persistence = getattr(app, "persistence", None)
    if persistence is not None:
        persistence.start_wal_task()

    logger.info(
        "Periodic cleanup armed (interval=%ds, initial_delay=%ds)",
        _CLEANUP_INTERVAL,
        _CLEANUP_INITIAL_DELAY,
    )

    # Deferred startup: token seeding, counter rebuild, backfills
    from ..maintenance.startup import DeferredStartup

    config = getattr(app, "gateway_config", None)
    deferred = DeferredStartup(config, app) if config else None
    if deferred is not None:
        setattr(app, "deferred_startup", deferred)
        # Expose on config so resolve() can check provider states
        setattr(config, "_deferred_startup", deferred)
        await deferred.start()

    # Start background refresh loops for providers that do NOT mint their own
    # credential (those are handled by DeferredStartup.seed_tokens).
    from ..transport.token_refresh import start_token_refreshers

    static_providers = {
        name: pinfo
        for name, pinfo in (config.providers.items() if config else {})
        if not pinfo.mints_credential
    }
    refresh_tasks = await start_token_refreshers(static_providers)

    try:
        await app._serve(host, port, socket=socket, ssl_context=ssl_context)
    finally:
        if deferred is not None:
            await deferred.shutdown()
        # Cancel token refresh tasks (both static and deferred-startup ones)
        all_refresh = refresh_tasks + getattr(app, "_token_refresh_tasks", [])
        for task in all_refresh:
            task.cancel()
        flush_task.cancel()
        cleanup_task.cancel()
        rotation_task.cancel()
        for task in all_refresh:
            try:
                await task
            except asyncio.CancelledError:
                pass
        for bg in (flush_task, cleanup_task, rotation_task):
            try:
                await bg
            except asyncio.CancelledError:
                pass
        await _flush_now(app)
        await close_resources(
            transport=app.transport,  # type: ignore
            metadata_store=app.metadata_store,  # type: ignore
        )
