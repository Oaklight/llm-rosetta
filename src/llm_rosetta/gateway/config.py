"""Gateway configuration: JSONC loading, env-var substitution, validation."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections.abc import Generator
from contextlib import contextmanager, suppress
from typing import Any, Protocol, runtime_checkable

from llm_rosetta.auto_detect import ProviderType
from llm_rosetta.routing import ResolvedRoute

from .providers import build_provider_info
from .transport import ProviderInfo

logger = logging.getLogger("llm-rosetta-gateway")

# ---------------------------------------------------------------------------
# Config file search paths (checked in order)
# ---------------------------------------------------------------------------

PATHS_TO_TRY = [
    "./config.jsonc",
    os.path.expanduser("~/.config/llm-rosetta-gateway/config.jsonc"),
    os.path.expanduser("~/.llm-rosetta-gateway/config.jsonc"),
]

# ---------------------------------------------------------------------------
# JSONC loader
# ---------------------------------------------------------------------------

_JSONC_COMMENT_RE = re.compile(
    r'("(?:[^"\\]|\\.)*")|//[^\n]*|/\*[\s\S]*?\*/', re.MULTILINE
)
_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")


def _strip_jsonc_comments(text: str) -> str:
    """Remove // and /* */ comments from JSONC, preserving strings."""

    def _replace(m: re.Match) -> str:
        if m.group(1) is not None:
            return m.group(1)  # quoted string — keep it
        return ""

    return _JSONC_COMMENT_RE.sub(_replace, text)


def _substitute_env_vars(text: str) -> str:
    """Replace ${ENV_VAR} placeholders with environment variable values."""

    def _replace(m: re.Match) -> str:
        var_name = m.group(1)
        value = os.environ.get(var_name)
        if value is None:
            logger.warning("Environment variable %s is not set", var_name)
            return m.group(0)  # leave placeholder intact
        return value

    return _ENV_VAR_RE.sub(_replace, text)


def load_config(path: str) -> dict[str, Any]:
    """Load and parse a JSONC config file with env-var substitution."""
    with open(path) as f:
        raw = f.read()
    stripped = _strip_jsonc_comments(raw)
    substituted = _substitute_env_vars(stripped)
    return json.loads(substituted)


def write_config(path: str, data: dict[str, Any]) -> None:
    """Write a config dict as formatted JSON to *path* atomically.

    Creates parent directories if needed.  Writes to a temporary file in
    the same directory, flushes to disk, then atomically replaces the
    target via ``os.replace``.  Readers never see a partially-written file.

    Comments in the original JSONC file (if any) are **not** preserved.
    """
    import tempfile

    dir_ = os.path.dirname(path) or "."
    os.makedirs(dir_, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp)
        raise


# ---------------------------------------------------------------------------
# Config-file read-modify-write serialization
# ---------------------------------------------------------------------------


def _lock_exclusive(f: Any) -> None:
    """Acquire an exclusive lock on an open file (cross-platform)."""
    if sys.platform == "win32":
        import msvcrt  # noqa: F811  # Windows-only

        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined]
    else:
        import fcntl

        fcntl.flock(f, fcntl.LOCK_EX)


@contextmanager
def config_lock(path: str) -> Generator[None, None, None]:
    """Serialize read-modify-write cycles on the same config file.

    Admin routes must wrap their ``load_raw → modify → save`` sequences
    with this lock.  Without it, concurrent requests that each load the
    same version can silently overwrite each other's changes.

    Uses a ``.lock`` sidecar file with ``fcntl.flock`` (Unix) or
    ``msvcrt.locking`` (Windows) for cross-process mutual exclusion.
    This protects against multiple gateway instances sharing the same
    config file (e.g. dev + prod on the same host).
    """
    lock_path = os.path.realpath(path) + ".lock"
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    with open(lock_path, "a+") as lf:
        _lock_exclusive(lf)
        yield


def load_config_raw(path: str) -> dict[str, Any]:
    """Load and parse a JSONC config file *without* env-var substitution.

    Useful for reading config that will be written back (e.g. ``add`` CLI).
    """
    with open(path) as f:
        raw = f.read()
    stripped = _strip_jsonc_comments(raw)
    return json.loads(stripped)


@runtime_checkable
class ConfigIO(Protocol):
    """Protocol for reading/writing gateway config files.

    The default :class:`JsoncConfigIO` handles JSONC files.  Downstream
    projects (e.g. argo-proxy) can supply an alternative implementation
    for other formats (YAML, TOML, etc.) via ``setup_admin(..., config_io=...)``.
    """

    def load(self, path: str) -> dict[str, Any]:
        """Read config with env-var substitution (for runtime use)."""
        ...

    def load_raw(self, path: str) -> dict[str, Any]:
        """Read config without env-var substitution (for edit round-trips)."""
        ...

    def save(self, path: str, data: dict[str, Any]) -> None:
        """Write config back to disk."""
        ...


class JsoncConfigIO:
    """Default :class:`ConfigIO` — reads/writes JSONC with env-var substitution."""

    def load(self, path: str) -> dict[str, Any]:
        return load_config(path)

    def load_raw(self, path: str) -> dict[str, Any]:
        return load_config_raw(path)

    def save(self, path: str, data: dict[str, Any]) -> None:
        write_config(path, data)


def discover_config(explicit_path: str | None = None) -> str | None:
    """Find the first existing config file.

    If *explicit_path* is given, return it unconditionally (caller is
    responsible for handling missing files).  Otherwise search
    ``PATHS_TO_TRY`` in order and return the first hit, or ``None``.
    """
    if explicit_path is not None:
        return explicit_path
    for path in PATHS_TO_TRY:
        if os.path.isfile(path):
            return path
    return None


# ---------------------------------------------------------------------------
# Config class
# ---------------------------------------------------------------------------


class GatewayConfig:
    """Parsed and validated gateway configuration."""

    # Default capabilities when not specified in config.
    DEFAULT_CAPABILITIES: list[str] = ["text"]

    def __init__(self, raw: dict[str, Any]) -> None:
        all_providers: dict[str, dict[str, str]] = raw.get("providers", {})

        # Filter out disabled providers (enabled defaults to True)
        self._raw_providers: dict[str, dict[str, str]] = {
            name: cfg
            for name, cfg in all_providers.items()
            if cfg.get("enabled", True) is not False
        }

        self.provider_types, self.provider_shim_names = self._resolve_provider_types(
            self._raw_providers
        )

        self.provider_supports_custom_tools = self._parse_custom_tools_overrides(
            self._raw_providers
        )

        self.provider_hoist_system_messages = self._parse_hoist_system_overrides(
            self._raw_providers
        )

        self.models, self.model_capabilities, self.model_upstream_names = (
            self._parse_models(raw.get("models", {}), self._raw_providers)
        )

        raw_models = raw.get("models", {})
        (
            self.model_url_templates,
            self.model_stream_url_templates,
            self.model_reasoning_overrides,
            self.model_flatten_system,
        ) = self._parse_model_overrides(raw_models)

        _server = raw.get("server", {})
        self._apply_server_settings(_server)
        self._apply_auth_settings(_server)

        self._apply_debug_settings(raw.get("debug", {}))

        self._validate()

        # Build ProviderInfo objects (with key rotation support)
        self.providers: dict[str, ProviderInfo] = {
            name: build_provider_info(
                self.provider_types[name], cfg, global_proxy=self.proxy
            )
            for name, cfg in self._raw_providers.items()
        }

    @staticmethod
    def _parse_custom_tools_overrides(
        raw_providers: dict[str, dict[str, str]],
    ) -> dict[str, bool]:
        """Extract per-provider supports_custom_tools overrides."""
        result: dict[str, bool] = {}
        for pname, pcfg in raw_providers.items():
            if isinstance(pcfg, dict) and "supports_custom_tools" in pcfg:
                result[pname] = bool(pcfg["supports_custom_tools"])
        return result

    @staticmethod
    def _parse_hoist_system_overrides(
        raw_providers: dict[str, dict[str, str]],
    ) -> dict[str, bool]:
        """Extract per-provider hoist_system_messages overrides."""
        result: dict[str, bool] = {}
        for pname, pcfg in raw_providers.items():
            if isinstance(pcfg, dict) and "hoist_system_messages" in pcfg:
                result[pname] = bool(pcfg["hoist_system_messages"])
        return result

    @staticmethod
    def _parse_model_overrides(
        raw_models: dict[str, Any],
    ) -> tuple[
        dict[str, str], dict[str, str], dict[str, dict[str, Any]], dict[str, bool]
    ]:
        """Extract per-model URL templates, reasoning overrides, and flatten_system flags."""
        url_templates: dict[str, str] = {}
        stream_url_templates: dict[str, str] = {}
        reasoning_overrides: dict[str, dict[str, Any]] = {}
        flatten_system: dict[str, bool] = {}
        for model_name, value in raw_models.items():
            if isinstance(value, dict):
                if "url_template" in value:
                    url_templates[model_name] = value["url_template"]
                if "stream_url_template" in value:
                    stream_url_templates[model_name] = value["stream_url_template"]
                if value.get("reasoning_override"):
                    reasoning_overrides[model_name] = value["reasoning_override"]
                if "flatten_system" in value:
                    flatten_system[model_name] = bool(value["flatten_system"])
                    continue
            if re.search(r"gemini", model_name, re.IGNORECASE):
                flatten_system[model_name] = True
        return url_templates, stream_url_templates, reasoning_overrides, flatten_system

    def _apply_server_settings(self, _server: dict[str, Any]) -> None:
        """Parse server section: host, port, proxy, timeouts, CORS, etc."""
        self.host: str = _server.get("host", "0.0.0.0")
        self.port: int = _server.get("port", 8765)
        self.proxy: str | None = _server.get("proxy")
        self.socket: str | None = _server.get("socket")
        self.credential_visible: bool = _server.get("credential_visible", True)

        # When no API keys are configured, ``open_on_no_keys`` decides whether
        # the standalone gateway serves /v1/* anonymously (True) or rejects
        # every request with 403 (False).  Defaults to False (secure by
        # default).  Set to True for trusted, localhost-only deployments.
        self.open_on_no_keys: bool = bool(_server.get("open_on_no_keys", False))

        self.admin_password: str | None = _server.get("admin_password")
        if self.admin_password and _ENV_VAR_RE.search(self.admin_password):
            raise ValueError(
                "config: admin_password contains an unresolved ${...} placeholder. "
                "Set the environment variable or use a literal password."
            )

        # CORS allow-list for /admin/* endpoints.
        # Default [] means same-origin only (no Access-Control-Allow-Origin).
        self.admin_cors_origins: list[str] = _server.get("admin_cors_origins", []) or []

        # Optional root redirect: GET / returns 307 to the given path.
        self.root_redirect: str | None = _server.get("root_redirect")

        # upstream_timeout: entire upstream HTTP lifecycle (connect + headers
        # + per-chunk streaming reads).  read_timeout: how long the inbound
        # httpserver waits for the client to send a complete request.
        self.upstream_timeout: float = max(
            1.0, float(_server.get("upstream_timeout", 300.0))
        )
        self.read_timeout: float = max(1.0, float(_server.get("read_timeout", 300.0)))

        # Request-log retention knobs (consumed by setup_admin).
        self.request_log: dict[str, Any] = _server.get("request_log", {}) or {}

    def _apply_auth_settings(self, _server: dict[str, Any]) -> None:
        """Parse API key auth settings from the server section.

        ``server.api_keys`` (list) takes precedence over the legacy
        ``server.api_key`` (single string).  A lone ``api_key`` is promoted
        to a synthetic ``api_keys`` entry for backward compatibility.
        """
        self.api_keys: list[dict[str, str]] = _server.get("api_keys", [])
        if not self.api_keys and _server.get("api_key"):
            self.api_keys = [
                {
                    "id": "default",
                    "key": _server["api_key"],
                    "label": "default",
                    "created": "",
                }
            ]

        # Custom SQLite DB path for API key storage (default: alongside config)
        self.api_keys_db: str | None = _server.get("api_keys_db")

    def _apply_debug_settings(self, _debug: dict[str, Any]) -> None:
        """Parse debug/logging settings with env-var overrides.

        Env vars ``LLM_ROSETTA_VERBOSE``, ``LLM_ROSETTA_LOG_BODIES``,
        and ``LLM_ROSETTA_LOG_FORMAT`` override their config counterparts.
        """
        self.verbose: bool = _debug.get("verbose", False) or os.environ.get(
            "LLM_ROSETTA_VERBOSE", ""
        ).lower() in ("1", "true", "yes")
        self.log_bodies: bool = _debug.get("log_bodies", False) or os.environ.get(
            "LLM_ROSETTA_LOG_BODIES", ""
        ).lower() in ("1", "true", "yes")
        self.error_dumps_enabled: bool = _debug.get("error_dumps", True)

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
        self.log_format: str = raw_log_format  # resolved later by setup_logging

    def _validate(self) -> None:
        if not self._raw_providers:
            logger.warning(
                "config: no enabled providers — all providers may be disabled"
            )
            return
        if not self.models:
            logger.warning(
                "config: no routable models — models may reference disabled providers"
            )
            return
        for model, provider in self.models.items():
            if provider not in self._raw_providers:
                raise ValueError(
                    f"config: model '{model}' references unknown provider '{provider}'"
                )

    @staticmethod
    def _resolve_provider_types(
        raw_providers: dict[str, dict[str, str]],
    ) -> tuple[dict[str, str], dict[str, str | None]]:
        """Resolve each provider's API standard type via shim registry.

        Resolution order per provider:
          1. ``shim`` field → resolve via shim registry
          2. ``type`` field → resolve via shim registry
          3. provider name itself (backward-compatible fallback)

        Returns:
            Tuple of (provider_types, provider_shim_names).
        """
        from llm_rosetta.shims import resolve_base

        provider_types: dict[str, str] = {}
        provider_shim_names: dict[str, str | None] = {}
        for name, cfg in raw_providers.items():
            if "shim" in cfg:
                provider_types[name] = resolve_base(cfg["shim"])
                provider_shim_names[name] = cfg["shim"]
            elif "type" in cfg:
                provider_types[name] = resolve_base(cfg["type"])
                provider_shim_names[name] = cfg["type"]
            else:
                provider_types[name] = name
                provider_shim_names[name] = name
        return provider_types, provider_shim_names

    @classmethod
    def _parse_models(
        cls,
        raw_models: dict[str, Any],
        raw_providers: dict[str, dict[str, str]],
    ) -> tuple[dict[str, ProviderType], dict[str, list[str]], dict[str, str]]:
        """Parse model routing entries from config.

        Supports both string and dict formats:
          - ``"model": "provider"`` (legacy)
          - ``"model": {"provider": "p", "capabilities": [...]}``
          - ``"model": {"provider": "p", "upstream_model": "actual_name"}``

        Models referencing disabled/missing providers are silently skipped.

        Returns:
            Tuple of (models, model_capabilities, model_upstream_names).
        """
        models: dict[str, ProviderType] = {}
        model_capabilities: dict[str, list[str]] = {}
        model_upstream_names: dict[str, str] = {}
        for name, value in raw_models.items():
            if isinstance(value, str):
                provider_name = value
            elif isinstance(value, dict):
                provider_name = value["provider"]
            else:
                raise ValueError(f"config: invalid model entry for '{name}'")

            # Skip disabled models
            if isinstance(value, dict) and value.get("enabled") is False:
                continue

            if provider_name not in raw_providers:
                continue

            models[name] = provider_name
            if isinstance(value, str):
                model_capabilities[name] = list(cls.DEFAULT_CAPABILITIES)
            else:
                model_capabilities[name] = value.get(
                    "capabilities", list(cls.DEFAULT_CAPABILITIES)
                )
                upstream = value.get("upstream_model")
                if upstream:
                    model_upstream_names[name] = upstream
        return models, model_capabilities, model_upstream_names

    @property
    def api_key(self) -> str | None:
        """First configured key (for backward-compat middleware init)."""
        return self.api_keys[0]["key"] if self.api_keys else None

    def resolve(
        self,
        source_provider: ProviderType,
        model: str,
    ) -> tuple[ResolvedRoute, ProviderInfo]:
        """Resolve *model* to a :class:`ResolvedRoute` and :class:`ProviderInfo`.

        Consolidates model lookup, provider type resolution, shim binding,
        capability detection, and reasoning overrides into a single typed
        result.

        Args:
            source_provider: API standard of the incoming request.
            model: Model name as specified by the client.

        Returns:
            ``(route, provider_info)`` — the route contains all
            pipeline-relevant fields; ``provider_info`` is the
            transport-level connection config.

        Raises:
            KeyError: If the model is not in the routing table.
        """
        from typing import cast

        provider_name = self.models[model]
        provider_type = self.provider_types[provider_name]
        shim_name = self.provider_shim_names.get(provider_name)
        upstream_model = self.model_upstream_names.get(model)
        caps = self.model_capabilities.get(model, list(self.DEFAULT_CAPABILITIES))
        reasoning = self.model_reasoning_overrides.get(model)
        flatten_system = self.model_flatten_system.get(model, False)
        from llm_rosetta.shims import resolve_shim

        _shim = resolve_shim(shim_name) if shim_name else None
        custom_tools = self.provider_supports_custom_tools.get(
            provider_name, _shim.supports_custom_tools if _shim else False
        )
        hoist_system = self.provider_hoist_system_messages.get(
            provider_name, _shim.hoist_system_messages if _shim else True
        )

        route = ResolvedRoute(
            source_provider=source_provider,
            target_provider=cast(ProviderType, provider_type),
            provider_name=provider_name,
            shim_name=shim_name,
            upstream_model=upstream_model,
            model_capabilities=caps,
            reasoning_override=reasoning,
            flatten_system=flatten_system,
            supports_custom_tools=custom_tools,
            hoist_system_messages=hoist_system,
        )

        pinfo = self.providers[provider_name]

        # Per-model URL template override: create a shallow copy of the
        # provider info with modified template(s) so the transport layer
        # hits the right endpoint.
        model_url_tpl = self.model_url_templates.get(model)
        model_stream_tpl = self.model_stream_url_templates.get(model)
        if model_url_tpl or model_stream_tpl:
            pinfo = pinfo.with_url_templates(model_url_tpl, model_stream_tpl)

        return route, pinfo
