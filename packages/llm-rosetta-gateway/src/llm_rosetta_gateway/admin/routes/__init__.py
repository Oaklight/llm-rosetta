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

from typing import Any


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
    admin_logout,
    change_password,
    logout_all_sessions,
    rotate_token,
    serve_admin_html,
    serve_admin_static,
    serve_favicon,
)
from ._shared import _guard
from .config import register_config_routes
from .keys import (
    create_api_key,
    delete_api_key,
    get_api_keys,
    get_internal_token,
    get_key_usage,
    rotate_api_key,
    update_api_key,
)
from .maintenance import (
    backfill_dump_log_ids,
    backfill_keys_last_used,
    get_maintenance,
    post_maintenance_run,
)
from .observability import (
    cleanup_error_dumps_by_age,
    cleanup_requests_by_age,
    clear_error_dumps,
    cleanup_ops_log_by_age,
    clear_ops_log,
    clear_requests,
    delete_error_dump,
    db_cleanup,
    db_cleanup_before,
    db_cleanup_range,
    db_trim,
    db_vacuum,
    export_error_dumps,
    get_circuit_breaker_states,
    get_error_dump_body,
    get_error_dump_detail,
    get_error_dumps,
    get_host_ip,
    get_metrics,
    get_ops_log,
    get_ops_log_event_types,
    get_ops_log_sources,
    get_rate_limit_status,
    get_request_key_labels,
    get_token_usage,
    get_request_by_id,
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
from .connectivity import test_provider_connectivity
from .credential import (
    delete_provider_credential,
    get_provider_credential,
    put_provider_credential,
)
from .quota import get_provider_quota
from .testing import (
    cancel_test,
    get_test_result,
    start_test,
)


def register_admin_routes(app: Any) -> None:
    """Register all admin panel routes on the httpserver App."""
    # HTML + favicon
    app.route("/admin", methods=["GET"])(serve_admin_html)
    app.route("/admin/", methods=["GET"])(serve_admin_html)
    app.route("/admin/static/<path:path>", methods=["GET"])(serve_admin_static)
    app.route("/favicon.ico", methods=["GET"])(serve_favicon)
    # Admin auth
    app.route("/admin/api/login", methods=["POST"])(admin_login)
    app.route("/admin/api/auth-check", methods=["GET"])(admin_check)
    app.route("/admin/api/logout", methods=["POST"])(admin_logout)
    app.route("/admin/api/config/password", methods=["PUT"])(change_password)
    app.route("/admin/api/token/rotate", methods=["POST"])(rotate_token)
    app.route("/admin/api/sessions/logout-all", methods=["POST"])(logout_all_sessions)
    # Config CRUD (providers + models tabs, settings, shims)
    register_config_routes(app)
    # Metrics (dashboard tab)
    app.route("/admin/api/metrics", methods=["GET"])(get_metrics)
    app.route("/admin/api/metrics/rebuild", methods=["POST"])(
        _guard("dashboard", rebuild_metrics)
    )
    app.route("/admin/api/token-usage", methods=["GET"])(
        _guard("dashboard", get_token_usage)
    )
    # Circuit breaker states (dashboard tab)
    app.route("/admin/api/circuit-breakers", methods=["GET"])(
        _guard("dashboard", get_circuit_breaker_states)
    )
    # Rate limit introspection (dashboard tab)
    app.route("/admin/api/rate-limits", methods=["GET"])(
        _guard("dashboard", get_rate_limit_status)
    )
    # Request log (logs tab)
    app.route("/admin/api/requests", methods=["GET"])(_guard("logs", get_requests))
    app.route("/admin/api/error-dumps/backfill-log-ids", methods=["POST"])(
        _guard("logs", backfill_dump_log_ids)
    )
    # Maintenance (dashboard tab): what the deferred steps did, and re-running one
    app.route("/admin/api/maintenance", methods=["GET"])(
        _guard("dashboard", get_maintenance)
    )
    app.route("/admin/api/maintenance/<name>", methods=["POST"])(
        _guard("dashboard", post_maintenance_run)
    )
    app.route("/admin/api/requests/key-labels", methods=["GET"])(get_request_key_labels)
    app.route("/admin/api/requests/<entry_id>", methods=["GET"])(
        _guard("logs", get_request_by_id)
    )
    app.route("/admin/api/requests", methods=["DELETE"])(_guard("logs", clear_requests))
    # Ops log (logs tab)
    app.route("/admin/api/ops-log", methods=["GET"])(_guard("logs", get_ops_log))
    app.route("/admin/api/ops-log", methods=["DELETE"])(_guard("logs", clear_ops_log))
    app.route("/admin/api/ops-log/event-types", methods=["GET"])(
        _guard("logs", get_ops_log_event_types)
    )
    app.route("/admin/api/ops-log/sources", methods=["GET"])(
        _guard("logs", get_ops_log_sources)
    )
    app.route("/admin/api/ops-log/cleanup", methods=["POST"])(
        _guard("logs", cleanup_ops_log_by_age)
    )
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
    app.route("/admin/api/error-dumps/<dump_id>", methods=["DELETE"])(delete_error_dump)
    app.route("/admin/api/db/cleanup", methods=["POST"])(db_cleanup)
    app.route("/admin/api/db/cleanup-before", methods=["POST"])(db_cleanup_before)
    app.route("/admin/api/db/cleanup-range", methods=["POST"])(db_cleanup_range)
    app.route("/admin/api/db/trim", methods=["POST"])(db_trim)
    app.route("/admin/api/db/vacuum", methods=["POST"])(db_vacuum)
    app.route("/admin/api/requests/cleanup", methods=["POST"])(cleanup_requests_by_age)
    app.route("/admin/api/error-dumps/cleanup", methods=["POST"])(
        cleanup_error_dumps_by_age
    )
    app.route("/admin/api/error-dumps/export", methods=["GET"])(export_error_dumps)
    # API key management (keys tab)
    app.route("/admin/api/keys", methods=["GET"])(_guard("keys", get_api_keys))
    app.route("/admin/api/keys", methods=["POST"])(_guard("keys", create_api_key))
    app.route("/admin/api/keys/usage", methods=["GET"])(_guard("keys", get_key_usage))
    app.route("/admin/api/keys/backfill-last-used", methods=["POST"])(
        _guard("keys", backfill_keys_last_used)
    )
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
    # Provider connectivity test
    app.route("/admin/api/config/providers/<name>/test-connectivity", methods=["POST"])(
        test_provider_connectivity
    )
    # Provider quota reading (one entry per credential)
    app.route("/admin/api/config/providers/<name>/quota", methods=["GET"])(
        get_provider_quota
    )
    # Shim-declared credentials: read, install a record, sign out
    app.route("/admin/api/config/providers/<name>/credential", methods=["GET"])(
        get_provider_credential
    )
    app.route("/admin/api/config/providers/<name>/credential", methods=["PUT"])(
        put_provider_credential
    )
    app.route("/admin/api/config/providers/<name>/credential", methods=["DELETE"])(
        delete_provider_credential
    )
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
