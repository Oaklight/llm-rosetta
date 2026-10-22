"""Provider shim layer — identity cards for LLM providers.

Importing this package automatically registers the built-in shims
(OpenAI, Anthropic, Google, DeepSeek, Volcengine, etc.) and discovers
any plugin shims declared via ``llm_rosetta.shim_providers`` entry points.

Public API:

- **Registration** (startup): ``register_shim``, ``load_providers_from_dir``
- **Query** (per request): ``get_shim``, ``list_shims``, ``resolve_format``,
  ``resolve_model_type`` (``resolve_base`` is the deprecated alias)
- **Transforms**: ``apply_transforms``, ``strip_fields``, ``rename_field``, ``set_defaults``

External shim sources — directory and archive loaders, the standalone
package catalog and scheduled removals — live in
:mod:`llm_rosetta_gateway.shims`; they need the gateway's HTTP client.
"""

from .provider_shim import (
    ConnectionConfig,
    ProviderShim,
    ReasoningCapability,
    ToolsConfig,
    get_shim,
    list_shims,
    register_shim,
    resolve_base,
    resolve_format,
    resolve_model_type,
    resolve_shim,
    unregister_shim,
)
from llm_rosetta.transforms import (
    IRTransform,
    Transform,
    TransformContext,
    apply_ir_transforms,
    apply_transforms,
    default_message_field,
    default_tool_description,
    force_fields,
    rewrite_harmony_tool_calls,
    rename_field,
    replace_message_field,
    set_defaults,
    strip_fields,
    strip_fields_for_model,
)

# Scan provider directories and register shims from YAML + transforms.py.
from .providers import load_providers as _load_providers
from .providers import load_providers_from_dir

_load_providers()

__all__ = [
    "ConnectionConfig",
    "ToolsConfig",
    "ProviderShim",
    "ReasoningCapability",
    "register_shim",
    "unregister_shim",
    "get_shim",
    "list_shims",
    "resolve_base",
    "resolve_format",
    "resolve_model_type",
    "resolve_shim",
    # Transforms
    "Transform",
    "apply_transforms",
    "force_fields",
    "strip_fields",
    "rename_field",
    "set_defaults",
    "replace_message_field",
    "default_message_field",
    "default_tool_description",
    "rewrite_harmony_tool_calls",
    "strip_fields_for_model",
    # IR transforms
    "IRTransform",
    "TransformContext",
    "apply_ir_transforms",
    "load_providers_from_dir",
]
