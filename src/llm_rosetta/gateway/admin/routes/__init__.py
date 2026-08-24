"""Admin panel route registration.

This package splits the admin route handlers into focused modules:

- auth        — HTML serving, admin login, rate limiting
- config      — Config CRUD, upstream model fetch, bulk-add
- keys        — Gateway API key management
- observability — Metrics, request log, network diagnostics
- testing     — Async model test tasks
- profiling   — On-demand pyinstrument profiling
"""

from __future__ import annotations

import functools
from typing import Any

from llm_rosetta._vendor.httpserver import JSONResponse

from ._shared import (  # noqa: F401  (re-exported for backward compat)
    _ENV_VAR_RE,
    _build_provider_entry,
    _get_config_path,
    _handle_provider_rename,
    _mask_api_key,
    _qp,
    _reload_gateway_config,
    _sync_auth_middleware,
)
from .auth import (
    admin_check,
    admin_login,
    change_password,
    rotate_token,
    serve_admin_html,
    serve_admin_static,
)
from .config import (
    bulk_add_models,
    delete_model,
    delete_provider,
    fetch_upstream_models,
    get_config,
    put_model,
    put_provider,
    put_server_settings,
    reload_config,
    bulk_update_models,
    toggle_model,
    toggle_provider,
)
from .keys import (
    create_api_key,
    delete_api_key,
    get_api_keys,
    get_internal_token,
    rotate_api_key,
    update_api_key,
)
from .observability import (
    clear_error_dumps,
    clear_requests,
    get_error_dump_body,
    get_error_dump_detail,
    get_error_dumps,
    get_host_ip,
    get_metrics,
    get_provider_key,
    get_request_key_labels,
    get_requests,
    network_diagnostics,
    rebuild_metrics,
)
from .capture import (
    clear_capture_results,
    disable_capture,
    enable_capture,
    get_capture_result,
    get_capture_results,
    get_capture_status,
)
from .profiling import (
    clear_profiling_results,
    disable_profiling,
    download_profiling_results,
    enable_profiling,
    get_profiling_result,
    get_profiling_results,
    get_profiling_status,
)
from .testing import (
    cancel_test,
    get_test_result,
    start_test,
)


def _guard(tab_id: str, handler: Any) -> Any:
    """Wrap a route handler to return 404 when its tab is disabled.

    Read-only endpoints (``get_config``, ``get_metrics``,
    ``get_request_key_labels``) are intentionally left unguarded —
    disabled tabs are a UI convenience, not security isolation, and
    other components (e.g. argo-proxy's ``ArgoConfigIO``) may depend
    on them regardless of tab visibility.
    """

    @functools.wraps(handler)
    async def _guarded(request: Any, **kw: Any) -> Any:
        if tab_id in getattr(request.app, "disabled_tabs", ()):
            return JSONResponse(
                {"error": "Tab disabled", "disabled": True}, status_code=404
            )
        return await handler(request, **kw)

    return _guarded


def register_admin_routes(app: Any) -> None:
    """Register all admin panel routes on the httpserver App."""
    # HTML + static assets
    app.route("/admin", methods=["GET"])(serve_admin_html)
    app.route("/admin/", methods=["GET"])(serve_admin_html)
    app.route("/admin/static/<path:path>", methods=["GET"])(serve_admin_static)
    # Admin auth
    app.route("/admin/api/login", methods=["POST"])(admin_login)
    app.route("/admin/api/auth-check", methods=["GET"])(admin_check)
    app.route("/admin/api/config/password", methods=["PUT"])(change_password)
    app.route("/admin/api/token/rotate", methods=["POST"])(rotate_token)
    # Config CRUD (providers + models tabs)
    app.route("/admin/api/config", methods=["GET"])(get_config)
    app.route("/admin/api/config/providers/<name>", methods=["PUT"])(
        _guard("providers", put_provider)
    )
    app.route("/admin/api/config/providers/<name>", methods=["DELETE"])(
        _guard("providers", delete_provider)
    )
    app.route("/admin/api/config/providers/<name>/toggle", methods=["POST"])(
        _guard("providers", toggle_provider)
    )
    app.route("/admin/api/config/providers/<name>/key", methods=["GET"])(
        get_provider_key
    )
    app.route("/admin/api/config/models/<path:name>", methods=["PUT"])(
        _guard("models", put_model)
    )
    app.route("/admin/api/config/models/<path:name>", methods=["DELETE"])(
        _guard("models", delete_model)
    )
    app.route("/admin/api/config/models/<path:name>/toggle", methods=["POST"])(
        _guard("models", toggle_model)
    )
    app.route("/admin/api/config/models/bulk", methods=["POST"])(
        _guard("models", bulk_update_models)
    )
    app.route("/admin/api/config/providers/<name>/models", methods=["GET"])(
        fetch_upstream_models
    )
    app.route("/admin/api/config/models", methods=["POST"])(
        _guard("models", bulk_add_models)
    )
    app.route("/admin/api/config/server", methods=["PUT"])(put_server_settings)
    app.route("/admin/api/config/reload", methods=["POST"])(reload_config)
    # Metrics (dashboard tab)
    app.route("/admin/api/metrics", methods=["GET"])(get_metrics)
    app.route("/admin/api/metrics/rebuild", methods=["POST"])(
        _guard("dashboard", rebuild_metrics)
    )
    # Request log (logs tab)
    app.route("/admin/api/requests", methods=["GET"])(_guard("logs", get_requests))
    app.route("/admin/api/requests/key-labels", methods=["GET"])(get_request_key_labels)
    app.route("/admin/api/requests", methods=["DELETE"])(_guard("logs", clear_requests))
    # Network diagnostics
    app.route("/admin/api/diagnostics/network", methods=["GET"])(network_diagnostics)
    app.route("/admin/api/diagnostics/host-ip", methods=["GET"])(get_host_ip)
    # Error dumps
    app.route("/admin/api/error-dumps", methods=["GET"])(get_error_dumps)
    app.route("/admin/api/error-dumps/<dump_id>", methods=["GET"])(
        get_error_dump_detail
    )
    app.route("/admin/api/error-dumps/<dump_id>/body", methods=["GET"])(
        get_error_dump_body
    )
    app.route("/admin/api/error-dumps", methods=["DELETE"])(clear_error_dumps)
    # API key management (keys tab)
    app.route("/admin/api/keys", methods=["GET"])(_guard("keys", get_api_keys))
    app.route("/admin/api/keys", methods=["POST"])(_guard("keys", create_api_key))
    app.route("/admin/api/keys/<key_id>", methods=["PUT"])(
        _guard("keys", update_api_key)
    )
    app.route("/admin/api/keys/<key_id>", methods=["DELETE"])(
        _guard("keys", delete_api_key)
    )
    app.route("/admin/api/keys/<key_id>/rotate", methods=["POST"])(
        _guard("keys", rotate_api_key)
    )
    app.route("/admin/api/internal-token", methods=["GET"])(get_internal_token)
    # Async model test
    app.route("/admin/api/test", methods=["POST"])(start_test)
    app.route("/admin/api/test/<task_id>", methods=["GET"])(get_test_result)
    app.route("/admin/api/test/<task_id>/poll", methods=["POST"])(get_test_result)
    app.route("/admin/api/test/<task_id>", methods=["DELETE"])(cancel_test)
    # Content capture (dashboard tab)
    app.route("/admin/api/capture/status", methods=["GET"])(
        _guard("dashboard", get_capture_status)
    )
    app.route("/admin/api/capture/enable", methods=["POST"])(
        _guard("dashboard", enable_capture)
    )
    app.route("/admin/api/capture/disable", methods=["POST"])(
        _guard("dashboard", disable_capture)
    )
    app.route("/admin/api/capture/results", methods=["GET"])(
        _guard("dashboard", get_capture_results)
    )
    app.route("/admin/api/capture/results/<index>", methods=["GET"])(
        _guard("dashboard", get_capture_result)
    )
    app.route("/admin/api/capture/results", methods=["DELETE"])(
        _guard("dashboard", clear_capture_results)
    )
    # Profiling (dashboard tab)
    app.route("/admin/api/profiling/status", methods=["GET"])(
        _guard("dashboard", get_profiling_status)
    )
    app.route("/admin/api/profiling/enable", methods=["POST"])(
        _guard("dashboard", enable_profiling)
    )
    app.route("/admin/api/profiling/disable", methods=["POST"])(
        _guard("dashboard", disable_profiling)
    )
    app.route("/admin/api/profiling/results", methods=["GET"])(
        _guard("dashboard", get_profiling_results)
    )
    app.route("/admin/api/profiling/results/<index>", methods=["GET"])(
        _guard("dashboard", get_profiling_result)
    )
    app.route("/admin/api/profiling/results/download", methods=["GET"])(
        _guard("dashboard", download_profiling_results)
    )
    app.route("/admin/api/profiling/results", methods=["DELETE"])(
        _guard("dashboard", clear_profiling_results)
    )
