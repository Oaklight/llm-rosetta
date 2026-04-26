"""Provider shim layer — identity cards for LLM providers and models.

Importing this package automatically registers the built-in shims
(OpenAI, Anthropic, Google, DeepSeek, Volcengine, etc.).
"""

from .provider_shim import (
    ModelShim,
    ProviderShim,
    get_shim,
    list_shims,
    register_shim,
    resolve_base,
    unregister_shim,
)
from .transforms import (
    Transform,
    Transformable,
    apply_transforms,
    rename_field,
    resolve_transforms,
    set_defaults,
    strip_fields,
)

# Importing builtins triggers auto-registration of built-in shims.
from . import builtins as _builtins  # noqa: F401

__all__ = [
    "ModelShim",
    "ProviderShim",
    "Transform",
    "Transformable",
    "apply_transforms",
    "get_shim",
    "list_shims",
    "register_shim",
    "rename_field",
    "resolve_base",
    "resolve_transforms",
    "set_defaults",
    "strip_fields",
    "unregister_shim",
]
