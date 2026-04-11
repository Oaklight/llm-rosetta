"""Gateway configuration: JSONC loading, env-var substitution, validation."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from llm_rosetta.auto_detect import ProviderType

from .providers import ProviderInfo, build_provider_info

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

    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw_providers: dict[str, dict[str, str]] = raw.get("providers", {})
        self.models: dict[str, ProviderType] = raw.get("models", {})
        self.host: str = raw.get("server", {}).get("host", "0.0.0.0")
        self.port: int = raw.get("server", {}).get("port", 8765)
        self.proxy: str | None = raw.get("server", {}).get("proxy")

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
            name: build_provider_info(name, cfg, global_proxy=self.proxy)
            for name, cfg in self._raw_providers.items()
        }

    def _validate(self) -> None:
        if not self._raw_providers:
            raise ValueError("config: 'providers' section is empty")
        if not self.models:
            raise ValueError("config: 'models' section is empty")
        for model, provider in self.models.items():
            if provider not in self._raw_providers:
                raise ValueError(
                    f"config: model '{model}' references unknown provider '{provider}'"
                )

    def resolve_model(self, model: str) -> tuple[ProviderType, ProviderInfo]:
        """Return (provider_type, provider_info) for a model name.

        Raises KeyError if the model is not in the routing table.
        """
        provider_type = self.models[model]
        return provider_type, self.providers[provider_type]
