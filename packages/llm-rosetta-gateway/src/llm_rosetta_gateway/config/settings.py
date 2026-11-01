"""Config sections: server, auth, limits, debug, and validation.

Each function reads one section of the document into the config object, or
rejects it — they are grouped here because they all follow that shape.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from .io import _ENV_VAR_RE
from .resolve import detect_routing_loops

if TYPE_CHECKING:
    from .core import GatewayConfig


logger = logging.getLogger("llm-rosetta-gateway")

#: Rate-limit algorithms the config accepts.
_VALID_RL_ALGORITHMS = frozenset({"token_bucket", "sliding_window", "fixed_window"})

#: Cap for ``server.admin_custom_css``.  The text is served inside every admin
#: page, so an unbounded value would bloat each response.
MAX_ADMIN_CUSTOM_CSS = 64 * 1024


def _parse_admin_custom_css(value: Any) -> str:
    """Validate ``server.admin_custom_css``.

    Args:
        value: The configured value, or ``None`` when the key is absent.

    Returns:
        The CSS text with the surrounding whitespace removed; an empty string
        when unset.

    Raises:
        ValueError: If the value is not a string, or exceeds
            :data:`MAX_ADMIN_CUSTOM_CSS`.
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("config: 'server.admin_custom_css' must be a string")
    if len(value) > MAX_ADMIN_CUSTOM_CSS:
        raise ValueError(
            "config: 'server.admin_custom_css' is limited to "
            f"{MAX_ADMIN_CUSTOM_CSS} characters"
        )
    return value.strip()


def resolve_data_dir(configured: str | None, config_path: str | None) -> str | None:
    """Resolve the gateway's data directory, anchoring relative paths.

    ``server.data_dir`` is written by hand, so it may be relative — and a
    relative path means "beside the config file", not "beside the working
    directory the gateway happened to start in".  With nothing configured the
    default is ``<config dir>/data``, which is what the persistence layer has
    always used.

    Args:
        configured: The value in ``server.data_dir``, or ``None``.
        config_path: Path of the config document, when the caller has one.
            Without it a relative value is returned as written — the caller
            has no anchor to resolve against.

    Returns:
        An absolute path, or ``None`` when there is neither a configured value
        nor a config path.
    """
    if configured:
        if config_path and not os.path.isabs(configured):
            return os.path.join(os.path.dirname(config_path), configured)
        return configured
    if config_path:
        return os.path.join(os.path.dirname(config_path), "data")
    return None


def _non_negative_int(value: Any, default: int, label: str) -> int:
    """Parse a session lifetime, rejecting anything below zero."""
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"config: {label} must be an integer, got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"config: {label} must be 0 or greater, got {parsed}")
    return parsed


def apply_server_settings(config: GatewayConfig, _server: dict[str, Any]) -> None:
    """Parse server section: host, port, proxy, timeouts, CORS, etc."""
    config.host = _server.get("host", "0.0.0.0")
    config.port = _server.get("port", 8765)
    config.proxy = _server.get("proxy")
    config.socket = _server.get("socket")
    config.credential_visible = _server.get("credential_visible", False)
    config.admin_custom_css = _parse_admin_custom_css(_server.get("admin_custom_css"))
    # Force off when admin panel has no password protection
    if not _server.get("admin_password"):
        config.credential_visible = False

    # When no API keys are configured, ``open_on_no_keys`` decides whether
    # the standalone gateway serves /v1/* anonymously (True) or rejects
    # every request with 403 (False).  Defaults to False (secure by
    # default).  Set to True for trusted, localhost-only deployments.
    config.open_on_no_keys = bool(_server.get("open_on_no_keys", False))

    # When a model has providers in multiple API formats, prefer the
    # one matching the client's format to avoid unnecessary cross-format
    # conversion.  The preference is tiered and relative to an anchor
    # provider, and lives in the selection chain as a stage.  Off by
    # default: enabling changes which upstream a multi-format model
    # reaches.
    config.prefer_same_format = bool(_server.get("prefer_same_format", False))

    config.admin_password = _server.get("admin_password")
    if config.admin_password and _ENV_VAR_RE.search(config.admin_password):
        raise ValueError(
            "config: admin_password contains an unresolved ${...} placeholder. "
            "Set the environment variable or use a literal password."
        )

    # Admin session lifetimes.  ``idle_minutes: 0`` never signs a session out
    # for inactivity; ``remember_days: 0`` removes "stay signed in" from the
    # login page and makes every session a browser-session cookie.
    from ..middleware.auth import (
        DEFAULT_SESSION_IDLE_MINUTES,
        DEFAULT_SESSION_REMEMBER_DAYS,
    )

    session = _server.get("admin_session") or {}
    config.admin_session_idle_minutes = _non_negative_int(
        session.get("idle_minutes"),
        DEFAULT_SESSION_IDLE_MINUTES,
        "session idle_minutes",
    )
    config.admin_session_remember_days = _non_negative_int(
        session.get("remember_days"),
        DEFAULT_SESSION_REMEMBER_DAYS,
        "session remember_days",
    )

    # CORS allow-list for /admin/* endpoints.
    # Default [] means same-origin only (no Access-Control-Allow-Origin).
    config.admin_cors_origins = _server.get("admin_cors_origins", []) or []

    # Optional root redirect: GET / returns 307 to the given path.
    config.root_redirect = _server.get("root_redirect")
    # The console's update probe asks PyPI for the newest release so the
    # panel can offer it.  Off means no outbound request and no bubble.
    # Only a literal ``true`` enables either switch: a quoted string in a
    # hand-written file must not turn an outbound check -- or a pip run that
    # replaces the running code -- on by accident.
    config.update_check = _server.get("update_check", True) is True
    config.self_update = _server.get("self_update", True) is True

    # upstream_timeout: entire upstream HTTP lifecycle (connect + headers
    # + per-chunk streaming reads).  read_timeout: how long the inbound
    # httpserver waits for the client to send a complete request.
    config.upstream_timeout = max(1.0, float(_server.get("upstream_timeout", 300.0)))
    config.read_timeout = max(1.0, float(_server.get("read_timeout", 300.0)))

    # Request-log retention knobs (consumed by setup_admin).
    config.request_log = _server.get("request_log", {}) or {}
    config.ops_log = _server.get("ops_log", {}) or {}
    # Resolved here rather than at the point of use: a provider that mints its
    # own credential needs its record directory while the config is being
    # built, and the persistence layer needs the same answer later.
    config.data_dir = resolve_data_dir(_server.get("data_dir"), config.config_path)

    # TLS termination: paths to PEM cert/key files.
    config.tls_cert = _server.get("tls_cert")
    config.tls_key = _server.get("tls_key")
    if bool(config.tls_cert) != bool(config.tls_key):
        raise ValueError(
            "config: tls_cert and tls_key must both be set or both be omitted"
        )


def apply_auth_settings(config: GatewayConfig, _server: dict[str, Any]) -> None:
    """Parse API key storage settings from the server section."""
    # Custom SQLite DB path for API key storage (default: alongside config)
    config.api_keys_db = _server.get("api_keys_db")


def apply_circuit_breaker_settings(
    config: GatewayConfig, _server: dict[str, Any]
) -> None:
    """Parse circuit breaker settings from the server section.

    Supports global defaults under ``server.circuit_breaker`` and
    per-provider overrides under each provider's ``circuit_breaker``
    sub-object.  The circuit breaker is disabled by default.
    """
    from ..middleware.circuit_breaker import (
        CircuitBreakerConfig,
        CircuitBreakerRegistry,
    )

    cb = _server.get("circuit_breaker", {}) or {}
    default_cfg = CircuitBreakerConfig(
        enabled=bool(cb.get("enabled", False)),
        error_threshold=int(cb.get("error_threshold", 5)),
        cooldown_seconds=float(cb.get("cooldown_seconds", 30.0)),
        half_open_max_probes=int(cb.get("half_open_max_probes", 1)),
    )
    config.circuit_breaker_registry = CircuitBreakerRegistry(default_cfg)

    # Per-provider overrides
    for pname, pcfg in config._raw_providers.items():
        if not isinstance(pcfg, dict):
            continue
        pcb = pcfg.get("circuit_breaker")
        if not isinstance(pcb, dict):
            continue
        override = CircuitBreakerConfig(
            enabled=bool(pcb.get("enabled", default_cfg.enabled)),
            error_threshold=int(
                pcb.get("error_threshold", default_cfg.error_threshold)
            ),
            cooldown_seconds=float(
                pcb.get("cooldown_seconds", default_cfg.cooldown_seconds)
            ),
            half_open_max_probes=int(
                pcb.get("half_open_max_probes", default_cfg.half_open_max_probes)
            ),
        )
        config.circuit_breaker_registry.set_override(pname, override)


def apply_rate_limit_settings(config: GatewayConfig, _server: dict[str, Any]) -> None:
    """Parse rate limiting settings from the server section."""
    rl = _server.get("rate_limit", {}) or {}
    config.rate_limit_enabled = bool(rl.get("enabled", False))
    algorithm = rl.get("algorithm", "sliding_window")
    if algorithm not in config._VALID_RL_ALGORITHMS:
        logger.warning(
            "config: invalid rate_limit algorithm %r, falling back to 'sliding_window'",
            algorithm,
        )
        algorithm = "sliding_window"
    config.rate_limit_algorithm = algorithm
    config.rate_limit_global = rl.get("global")
    config.rate_limit_per_ip = rl.get("per_ip")
    config.rate_limit_per_key = rl.get("per_key")
    config.rate_limit_per_model = rl.get("per_model")
    config.rate_limit_exclude = rl.get("exclude_paths", ["/health", "/admin"])
    # When True, trust X-Forwarded-For / X-Real-IP for per-IP limiting.
    # Default False — use direct peer address (safe when exposed directly).
    config.rate_limit_trust_proxy = bool(rl.get("trust_proxy", False))


def apply_debug_settings(config: GatewayConfig, _debug: dict[str, Any]) -> None:
    """Parse debug/logging settings with env-var overrides.

    Env vars ``LLM_ROSETTA_VERBOSE``, ``LLM_ROSETTA_LOG_BODIES``,
    and ``LLM_ROSETTA_LOG_FORMAT`` override their config counterparts.
    """
    config.verbose = _debug.get("verbose", False) or os.environ.get(
        "LLM_ROSETTA_VERBOSE", ""
    ).lower() in ("1", "true", "yes")
    config.log_bodies = _debug.get("log_bodies", False) or os.environ.get(
        "LLM_ROSETTA_LOG_BODIES", ""
    ).lower() in ("1", "true", "yes")
    config.error_dumps_enabled = _debug.get("error_dumps", True)

    # Log format: "text" (colourised, default for TTY), "json" (structured
    # one-line JSON, default for non-TTY), or "auto" (detect from stderr
    # TTY status).  Env var LLM_ROSETTA_LOG_FORMAT overrides config.
    _env_log_format = os.environ.get("LLM_ROSETTA_LOG_FORMAT", "").strip().lower()
    _cfg_log_format = _debug.get("log_format", "auto")
    raw_log_format = _env_log_format or _cfg_log_format
    if raw_log_format not in ("json", "text", "auto"):
        logger.warning(
            "config: invalid log_format %r, falling back to 'auto'",
            raw_log_format,
        )
        raw_log_format = "auto"
    config.log_format = raw_log_format  # resolved later by setup_logging


def validate(config: GatewayConfig, raw_models: dict[str, Any]) -> None:
    if not config._raw_providers:
        logger.warning("config: no enabled providers — all providers may be disabled")
        return
    if not config.models:
        logger.warning(
            "config: no routable models — models may reference disabled providers"
        )
        return
    for model, route in config.models.items():
        entry = raw_models.get(model)
        # Typed models are moved out of the LLM pool later; an embedding
        # model legitimately points at an embedding provider.
        declared_type = entry.get("type", "llm") if isinstance(entry, dict) else "llm"
        for pname in route.provider_names:
            if pname not in config._raw_providers:
                raise ValueError(
                    f"config: model '{model}' references unknown provider '{pname}'"
                )
            if declared_type != "llm":
                continue
            identity = config.provider_identities.get(pname)
            if identity is not None and identity.model_type != "llm":
                raise ValueError(
                    f"config: model '{model}' is an LLM model but provider "
                    f"'{pname}' serves {identity.model_type} "
                    f"({identity.format})"
                )

    # Warn about potential routing loops.
    for warning in detect_routing_loops(config.models, config.model_upstream_names):
        logger.warning("config: %s", warning)


def parse_custom_tools_overrides(
    raw_providers: dict[str, dict[str, str]],
) -> dict[str, bool]:
    """Extract per-provider supports_custom_tools overrides."""
    result: dict[str, bool] = {}
    for pname, pcfg in raw_providers.items():
        if isinstance(pcfg, dict) and "supports_custom_tools" in pcfg:
            result[pname] = bool(pcfg["supports_custom_tools"])
    return result


def parse_hoist_system_overrides(
    raw_providers: dict[str, dict[str, str]],
) -> dict[str, bool]:
    """Extract per-provider hoist_system_messages overrides."""
    result: dict[str, bool] = {}
    for pname, pcfg in raw_providers.items():
        if isinstance(pcfg, dict) and "hoist_system_messages" in pcfg:
            result[pname] = bool(pcfg["hoist_system_messages"])
    return result


def parse_preflight_token_count_overrides(
    raw_providers: dict[str, dict[str, str]],
) -> dict[str, bool]:
    """Extract per-provider preflight_token_count overrides."""
    result: dict[str, bool] = {}
    for pname, pcfg in raw_providers.items():
        if isinstance(pcfg, dict) and "preflight_token_count" in pcfg:
            result[pname] = bool(pcfg["preflight_token_count"])
    return result


def parse_key_affinity_overrides(
    raw_providers: dict[str, dict[str, str]],
) -> dict[str, bool]:
    """Extract per-provider key_affinity overrides."""
    result: dict[str, bool] = {}
    for pname, pcfg in raw_providers.items():
        if isinstance(pcfg, dict) and "key_affinity" in pcfg:
            result[pname] = bool(pcfg["key_affinity"])
    return result


def parse_max_tool_description_length_overrides(
    raw_providers: dict[str, dict[str, str]],
) -> dict[str, int]:
    """Extract per-provider max_tool_description_length overrides."""
    result: dict[str, int] = {}
    for pname, pcfg in raw_providers.items():
        if isinstance(pcfg, dict) and "max_tool_description_length" in pcfg:
            val = pcfg["max_tool_description_length"]
            if val is not None:
                result[pname] = int(val)
    return result
