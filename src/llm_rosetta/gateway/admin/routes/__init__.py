"""Admin panel route registration.

This package splits the admin route handlers into focused modules:

- auth        — HTML serving, admin login, rate limiting
- config      — Config CRUD, upstream model fetch, bulk-add
- keys        — Gateway API key management
- observability — Metrics, request log, network diagnostics
- testing     — Async model test tasks
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
    serve_admin_html,
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
    toggle_provider,
)
from .keys import (
    create_api_key,
    delete_api_key,
    get_api_keys,
    get_internal_token,
    reveal_api_key,
    update_api_key,
)
from .observability import (
    clear_requests,
    get_metrics,
    get_provider_key,
    get_requests,
    network_diagnostics,
)
from .testing import (
    cancel_test,
    get_test_result,
    start_test,
)


def register_admin_routes(app: Any) -> None:
    """Register all admin panel routes on the httpserver App."""
    # HTML
    app.route("/admin", methods=["GET"])(serve_admin_html)
    app.route("/admin/", methods=["GET"])(serve_admin_html)
    # Admin auth
    app.route("/admin/api/login", methods=["POST"])(admin_login)
    app.route("/admin/api/auth-check", methods=["GET"])(admin_check)
    # Config CRUD
    app.route("/admin/api/config", methods=["GET"])(get_config)
    app.route("/admin/api/config/providers/<name>", methods=["PUT"])(put_provider)
    app.route("/admin/api/config/providers/<name>", methods=["DELETE"])(delete_provider)
    app.route("/admin/api/config/providers/<name>/toggle", methods=["POST"])(
        toggle_provider
    )
    app.route("/admin/api/config/providers/<name>/key", methods=["GET"])(
        get_provider_key
    )
    app.route("/admin/api/config/models/<path:name>", methods=["PUT"])(put_model)
    app.route("/admin/api/config/models/<path:name>", methods=["DELETE"])(delete_model)
    app.route("/admin/api/config/providers/<name>/models", methods=["GET"])(
        fetch_upstream_models
    )
    app.route("/admin/api/config/models", methods=["POST"])(bulk_add_models)
    app.route("/admin/api/config/server", methods=["PUT"])(put_server_settings)
    app.route("/admin/api/config/reload", methods=["POST"])(reload_config)
    # Metrics
    app.route("/admin/api/metrics", methods=["GET"])(get_metrics)
    # Request log
    app.route("/admin/api/requests", methods=["GET"])(get_requests)
    app.route("/admin/api/requests", methods=["DELETE"])(clear_requests)
    # Network diagnostics
    app.route("/admin/api/diagnostics/network", methods=["GET"])(network_diagnostics)
    # API key management
    app.route("/admin/api/keys", methods=["GET"])(get_api_keys)
    app.route("/admin/api/keys", methods=["POST"])(create_api_key)
    app.route("/admin/api/keys/<key_id>", methods=["PUT"])(update_api_key)
    app.route("/admin/api/keys/<key_id>", methods=["DELETE"])(delete_api_key)
    app.route("/admin/api/keys/<key_id>/reveal", methods=["GET"])(reveal_api_key)
    app.route("/admin/api/internal-token", methods=["GET"])(get_internal_token)
    # Async model test
    app.route("/admin/api/test", methods=["POST"])(start_test)
    app.route("/admin/api/test/<task_id>", methods=["GET"])(get_test_result)
    app.route("/admin/api/test/<task_id>/poll", methods=["POST"])(get_test_result)
    app.route("/admin/api/test/<task_id>", methods=["DELETE"])(cancel_test)
