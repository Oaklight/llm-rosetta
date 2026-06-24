"""Provider shim layer — identity cards for LLM providers.

Importing this package automatically registers the built-in shims
(OpenAI, Anthropic, Google, DeepSeek, Volcengine, etc.) and discovers
any plugin shims declared via ``llm_rosetta.shim_providers`` entry points.

Public API:

- **Registration** (startup): ``register_shim``, ``load_providers_from_dir``
- **Query** (per request): ``get_shim``, ``list_shims``, ``resolve_base``
- **Transforms**: ``apply_transforms``, ``strip_fields``, ``rename_field``, ``set_defaults``
"""

from .provider_shim import (
    ProviderShim,
    ReasoningCapability,
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
from .providers import load_providers_from_dir

_load_providers()

__all__ = [
    "ProviderShim",
    "ReasoningCapability",
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
    "load_providers_from_dir",
]
