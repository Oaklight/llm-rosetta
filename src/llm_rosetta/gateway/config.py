"""Gateway configuration: JSONC loading, env-var substitution, validation."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from llm_rosetta.auto_detect import ProviderType

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
    """Write a config dict as formatted JSON to *path*.

    Creates parent directories if needed.  Comments in the original
    JSONC file (if any) are **not** preserved.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_config_raw(path: str) -> dict[str, Any]:
    """Load and parse a JSONC config file *without* env-var substitution.

    Useful for reading config that will be written back (e.g. ``add`` CLI).
    """
    with open(path) as f:
        raw = f.read()
    stripped = _strip_jsonc_comments(raw)
    return json.loads(stripped)


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

        self.models, self.model_capabilities, self.model_upstream_names = (
            self._parse_models(raw.get("models", {}), self._raw_providers)
        )

        # Per-model reasoning overrides from config.jsonc (admin UI edits).
        # Keyed by gateway model name (same as self.models keys).
        self.model_reasoning_overrides: dict[str, dict[str, Any]] = {}
        for model_name, value in raw.get("models", {}).items():
            if isinstance(value, dict) and value.get("reasoning_override"):
                self.model_reasoning_overrides[model_name] = value["reasoning_override"]

        _server = raw.get("server", {})
        self.host: str = _server.get("host", "0.0.0.0")
        self.port: int = _server.get("port", 8765)
        self.proxy: str | None = _server.get("proxy")
        self.socket: str | None = _server.get("socket")
        self.credential_visible: bool = _server.get("credential_visible", True)
        self.admin_password: str | None = _server.get("admin_password")
        if self.admin_password and _ENV_VAR_RE.search(self.admin_password):
            raise ValueError(
                "config: admin_password contains an unresolved ${...} placeholder. "
                "Set the environment variable or use a literal password."
            )

        # CORS allow-list for /admin/* endpoints.
        # Default [] means same-origin only (no Access-Control-Allow-Origin header).
        # To permit a specific trusted origin set e.g. in your config (JSONC):
        #   "server": {
        #     "admin_cors_origins": ["https://my-admin.example.com"]
        #   }
        self.admin_cors_origins: list[str] = _server.get("admin_cors_origins", []) or []

        # Request-log retention knobs (consumed by setup_admin).  Kept as
        # a raw dict here so admin layer owns the resolution policy.
        self.request_log: dict[str, Any] = _server.get("request_log", {}) or {}

        # Multi-key auth: server.api_keys takes precedence over server.api_key
        self.api_keys: list[dict[str, str]] = _server.get("api_keys", [])
        if not self.api_keys and _server.get("api_key"):
            # Backward compat: single api_key → synthetic entry
            self.api_keys = [
                {
                    "id": "default",
                    "key": _server["api_key"],
                    "label": "default",
                    "created": "",
                }
            ]
        # O(1) lookup set and key→label map for auth middleware
        self.api_key_set: frozenset[str] = frozenset(
            entry["key"] for entry in self.api_keys
        )
        self.api_key_labels: dict[str, str] = {
            entry["key"]: entry.get("label", "") for entry in self.api_keys
        }

        # Debug / logging options (config + env-var overrides)
        _debug = raw.get("debug", {})
        self.verbose: bool = _debug.get("verbose", False) or os.environ.get(
            "LLM_ROSETTA_VERBOSE", ""
        ).lower() in ("1", "true", "yes")
        self.log_bodies: bool = _debug.get("log_bodies", False) or os.environ.get(
            "LLM_ROSETTA_LOG_BODIES", ""
        ).lower() in ("1", "true", "yes")

        self._validate()

        # Build ProviderInfo objects (with key rotation support)
        self.providers: dict[str, ProviderInfo] = {
            name: build_provider_info(
                self.provider_types[name], cfg, global_proxy=self.proxy
            )
            for name, cfg in self._raw_providers.items()
        }

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

    def resolve_model(
        self, model: str
    ) -> tuple[str, ProviderInfo, str | None, str | None, str]:
        """Return (provider_type, provider_info, shim_name, upstream_model, provider_name).

        ``provider_type`` is the API standard (e.g. ``"openai_chat"``),
        resolved from the provider's ``type`` field or its name as fallback.

        ``shim_name`` is the original shim/type identifier before base
        resolution (e.g. ``"volcengine"``), used for transform lookup.

        ``upstream_model`` is the actual model identifier to send to the
        upstream provider, or ``None`` when the gateway model name is used
        as-is.  This enables model aliasing — e.g. gateway name
        ``"argo:claude-opus-4.5"`` mapping to upstream ``"claudeopus45"``.

        ``provider_name`` is the user-configured provider name (e.g.
        ``"deepseek"``, ``"my-openai"``), used for admin UI filtering
        and display.

        Raises KeyError if the model is not in the routing table.
        """
        provider_name = self.models[model]
        provider_type = self.provider_types[provider_name]
        shim_name = self.provider_shim_names.get(provider_name)
        upstream_model = self.model_upstream_names.get(model)
        return (
            provider_type,
            self.providers[provider_name],
            shim_name,
            upstream_model,
            provider_name,
        )
