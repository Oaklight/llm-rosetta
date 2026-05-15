"""Provider shim layer — identity cards for LLM providers.

Importing this package automatically registers the built-in shims
(OpenAI, Anthropic, Google, DeepSeek, Volcengine, etc.).
"""

from .provider_shim import (
    ProviderShim,
    get_shim,
    list_shims,
    register_shim,
    resolve_base,
    unregister_shim,
)
from .transforms import (
    Transform,
    apply_transforms,
    rename_field,
    set_defaults,
    strip_fields,
)

# Scan provider directories and register shims from YAML + transforms.py.
from .providers import load_providers as _load_providers

_load_providers()

__all__ = [
    "ProviderShim",
    "register_shim",
    "unregister_shim",
    "get_shim",
    "list_shims",
    "resolve_base",
    # Transforms
    "Transform",
    "apply_transforms",
    "strip_fields",
    "rename_field",
    "set_defaults",
]
