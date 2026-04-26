"""Built-in provider shim definitions.

Importing this module registers the official and well-known third-party
provider shims into the global registry.
"""

from __future__ import annotations

from .provider_shim import ModelShim, ProviderShim, register_shim
from .transforms import strip_fields

# ---------------------------------------------------------------------------
# Reusable model shim tuples
# ---------------------------------------------------------------------------

_OPENAI_MODELS: tuple[ModelShim, ...] = (
    ModelShim("o1-*", frozenset({"reasoning", "tools", "vision"})),
    ModelShim("o3-*", frozenset({"reasoning", "tools", "vision"})),
    ModelShim("o4-*", frozenset({"reasoning", "tools", "vision"})),
    ModelShim("gpt-*", frozenset({"tools", "vision"})),
)

# ---------------------------------------------------------------------------
# Official provider shims
# ---------------------------------------------------------------------------

OPENAI = ProviderShim(
    name="openai",
    base="openai_chat",
    default_base_url="https://api.openai.com/v1",
    default_api_key_env="OPENAI_API_KEY",
    models=_OPENAI_MODELS,
)

OPENAI_RESPONSES = ProviderShim(
    name="openai_responses",
    base="openai_responses",
    default_base_url="https://api.openai.com/v1",
    default_api_key_env="OPENAI_API_KEY",
    models=_OPENAI_MODELS,
)

ANTHROPIC = ProviderShim(
    name="anthropic",
    base="anthropic",
    default_base_url="https://api.anthropic.com",
    default_api_key_env="ANTHROPIC_API_KEY",
    models=(ModelShim("claude-*", frozenset({"reasoning", "tools", "vision"})),),
)

GOOGLE = ProviderShim(
    name="google",
    base="google",
    default_base_url="https://generativelanguage.googleapis.com",
    default_api_key_env="GOOGLE_API_KEY",
    models=(
        ModelShim("gemini-2.5-*", frozenset({"reasoning", "tools", "vision"})),
        ModelShim("gemini-*", frozenset({"tools", "vision"})),
    ),
)

# ---------------------------------------------------------------------------
# Third-party provider shims
# ---------------------------------------------------------------------------

DEEPSEEK = ProviderShim(
    name="deepseek",
    base="openai_chat",
    default_base_url="https://api.deepseek.com",
    default_api_key_env="DEEPSEEK_API_KEY",
    models=(ModelShim("deepseek-*", frozenset({"reasoning", "tools"})),),
)

VOLCENGINE = ProviderShim(
    name="volcengine",
    base="openai_chat",
    default_base_url=None,
    default_api_key_env="VOLCENGINE_API_KEY",
    to_transforms=(strip_fields("logprobs", "top_logprobs"),),
)

# ---------------------------------------------------------------------------
# Auto-register all built-in shims
# ---------------------------------------------------------------------------

_BUILTIN_SHIMS: tuple[ProviderShim, ...] = (
    OPENAI,
    OPENAI_RESPONSES,
    ANTHROPIC,
    GOOGLE,
    DEEPSEEK,
    VOLCENGINE,
)


def _register_builtins() -> None:
    """Register all built-in shims into the global registry."""
    for shim in _BUILTIN_SHIMS:
        register_shim(shim)


_register_builtins()
