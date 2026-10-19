"""Provider shim layer — identity cards for LLM providers.

Importing this package automatically registers the built-in shims
(OpenAI, Anthropic, Google, DeepSeek, Volcengine, etc.) and discovers
any plugin shims declared via ``llm_rosetta.shim_providers`` entry points.

Public API:

- **Registration** (startup): ``register_shim``, ``load_providers_from_dir``
- **Query** (per request): ``get_shim``, ``list_shims``, ``resolve_format``,
  ``resolve_model_type`` (``resolve_base`` is the deprecated alias)
- **Transforms**: ``apply_transforms``, ``strip_fields``, ``rename_field``, ``set_defaults``
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

# Declarative external sources (directories + remote archives).  Imported after
# the built-ins so a rebuild can rely on them already being registered.
from .sources import (  # noqa: E402
    ShimArchiveMismatch,
    ShimLoadReport,
    ShimSourceError,
    ShimSourceStatus,
    rebuild_shims,
    resolve_remote,
)

# Known standalone shim packages, then the shims with a removal scheduled.
from .catalog import (  # noqa: E402
    SHIM_SOURCE_PACKAGES,
    ShimSourcePackage,
    find_source_package,
    get_source_package,
)
from .removals import (  # noqa: E402
    SHIM_REMOVALS,
    RemovalReference,
    ShimRemoval,
    ShimRemovalAdvisory,
    detect_shim_removals,
    parse_version,
    removals_due,
    scan_removal_references,
)

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
    # External shim sources
    "rebuild_shims",
    "resolve_remote",
    "ShimLoadReport",
    "ShimSourceStatus",
    "ShimSourceError",
    "ShimArchiveMismatch",
    # Scheduled removals
    "SHIM_REMOVALS",
    "ShimRemoval",
    "ShimRemovalAdvisory",
    "RemovalReference",
    "detect_shim_removals",
    "parse_version",
    "removals_due",
    "scan_removal_references",
    # Standalone shim packages (Settings presets)
    "SHIM_SOURCE_PACKAGES",
    "ShimSourcePackage",
    "find_source_package",
    "get_source_package",
    # Transforms
    "Transform",
    "apply_transforms",
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
