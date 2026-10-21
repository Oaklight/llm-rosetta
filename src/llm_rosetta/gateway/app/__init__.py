"""app package."""

from __future__ import annotations

from .routes_llm import (  # noqa: F401
    _resolve_or_error,
    _proxy_handler,
    _handle_llm,
    handle_chat_completion,
    handle_messages,
    handle_open_responses,
    handle_google_generate,
    handle_google_interactions,
    _LLM_ROUTE_HANDLERS,
    _register_llm_routes,
    _llm_route_handler,
)

from .routes_models import (  # noqa: F401
    handle_list_models,
    handle_list_models_google,
)

from .routes_health import (  # noqa: F401
    handle_health,
    handle_health_live,
    handle_health_ready,
)

from .routes_typed import (  # noqa: F401
    _register_non_llm_routes,
    _make_typed_handler,
)

from .tasks import (  # noqa: F401
    _DEFAULT_TOKEN_ROTATE_SECONDS,
    _periodic_token_rotation,
    _FLUSH_METRICS_INTERVAL,
    _CLEANUP_INTERVAL,
    _CLEANUP_INITIAL_DELAY,
    _periodic_cleanup,
    _periodic_flush,
    _flush_now,
)

from .factory import (  # noqa: F401
    _resolve_data_dir_for_app,
    _setup_auth,
    _install_root_redirect,
    GatewayExtensions,
    _install_cors,
    _install_rate_limiting,
    _install_lifecycle_hooks,
    _report_shim_removals,
    create_app,
    run_gateway,
)


__all__ = [
    "GatewayExtensions",
    "create_app",
    "run_gateway",
    "_resolve_or_error",
    "_proxy_handler",
    "_handle_llm",
    "handle_chat_completion",
    "handle_messages",
    "handle_open_responses",
    "handle_google_generate",
    "handle_google_interactions",
    "_LLM_ROUTE_HANDLERS",
    "_register_llm_routes",
    "_llm_route_handler",
    "handle_list_models",
    "handle_list_models_google",
    "handle_health",
    "handle_health_live",
    "handle_health_ready",
    "_register_non_llm_routes",
    "_make_typed_handler",
    "_DEFAULT_TOKEN_ROTATE_SECONDS",
    "_periodic_token_rotation",
    "_FLUSH_METRICS_INTERVAL",
    "_CLEANUP_INTERVAL",
    "_CLEANUP_INITIAL_DELAY",
    "_periodic_cleanup",
    "_periodic_flush",
    "_flush_now",
    "_config",
    "_resolve_data_dir_for_app",
    "_setup_auth",
    "_install_root_redirect",
    "GatewayExtensions",
    "_install_cors",
    "_install_rate_limiting",
    "_install_lifecycle_hooks",
    "_report_shim_removals",
    "create_app",
    "run_gateway",
]
