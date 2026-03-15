"""Gateway configuration: JSONC loading, env-var substitution, validation."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from llm_rosetta.auto_detect import ProviderType

logger = logging.getLogger("llm-rosetta-gateway")

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


# ---------------------------------------------------------------------------
# Config class
# ---------------------------------------------------------------------------


class GatewayConfig:
    """Parsed and validated gateway configuration."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.providers: dict[str, dict[str, str]] = raw.get("providers", {})
        self.models: dict[str, ProviderType] = raw.get("models", {})
        self.host: str = raw.get("server", {}).get("host", "0.0.0.0")
        self.port: int = raw.get("server", {}).get("port", 8765)
        self._validate()

    def _validate(self) -> None:
        if not self.providers:
            raise ValueError("config: 'providers' section is empty")
        if not self.models:
            raise ValueError("config: 'models' section is empty")
        for model, provider in self.models.items():
            if provider not in self.providers:
                raise ValueError(
                    f"config: model '{model}' references unknown provider '{provider}'"
                )

    def resolve_model(self, model: str) -> tuple[ProviderType, dict[str, str]]:
        """Return (provider_type, provider_config) for a model name.

        Raises KeyError if the model is not in the routing table.
        """
        provider_type = self.models[model]
        return provider_type, self.providers[provider_type]
