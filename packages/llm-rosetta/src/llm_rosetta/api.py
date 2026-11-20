"""The small public surface that hosts import from ``llm_rosetta`` (#1630).

The gateway and external shim authors build on core through this module and
four contract modules.  Everything else under ``llm_rosetta`` is internal and
may move in any release.

The five allowed import paths are:

- ``llm_rosetta.api`` (this module)
- ``llm_rosetta.routing``
- ``llm_rosetta.observability.usage`` (the package, never its submodules)
- ``llm_rosetta.shims.auth_exchange``
- ``llm_rosetta.shims.quota_probe``

The four contract modules stay where they are and are not re-exported here, so
each name has one home.

Stability rule: within one minor release this module only gains names.  A name
that is renamed or removed keeps a deprecated alias for one minor release.  A
breaking change needs an entry in ``scripts/dev/public_api_allowlist.json``,
and ``scripts/dev/public_api_diff.py`` flags it.

This module holds re-exports only.  Core code must never import it: the
dependency runs from the host to core, and an import from inside core would
create a cycle (``tests/test_core_import_boundary.py`` guards this).  Each name
is imported from its most stable home, never from a private ``_`` module.

``__all__`` is a literal list.  Names are grouped by capability, and each group
is sorted.
"""

from __future__ import annotations

from llm_rosetta import ProviderType, __version__, get_converter_for_provider
from llm_rosetta.capabilities import ToolNameMap
from llm_rosetta.converters import get_converter_class
from llm_rosetta.converters.base.errors import UnrepresentableFinishReason
from llm_rosetta.converters.base.helpers import strip_internal_metadata
from llm_rosetta.fidelity import FidelityChecker
from llm_rosetta.formats import (
    FORMAT_SPECS,
    MODEL_TYPES,
    FormatSpec,
    canonical_model_type,
    legacy_model_type,
)
from llm_rosetta.formats.catalog import default_path
from llm_rosetta.formats.tool_locations import ToolDeclaration, read_tool_declarations
from llm_rosetta.model_vocabulary import FORMATS_BY_TYPE
from llm_rosetta.pipeline import (
    ConversionError,
    ConversionPipeline,
    DecisionConversionPipeline,
    EmbeddingConversionPipeline,
    RerankConversionPipeline,
    get_decision_converter,
    get_embedding_converter,
    get_rerank_converter,
)
from llm_rosetta.profiling import DeepProfiler
from llm_rosetta.provider_names import (
    canonical_provider_name,
    canonical_shim_name,
    warn_legacy_provider_names,
    warn_legacy_shim_names,
)
from llm_rosetta.shims import (
    ConnectionConfig,
    ProviderShim,
    ReasoningCapability,
    get_shim,
    list_shims,
    load_providers_from_dir,
    register_shim,
    resolve_format,
    resolve_model_type,
    resolve_shim,
    restore_registry,
    snapshot_registry,
    unregister_shim,
)
from llm_rosetta.shims.provider_shim import (
    SoftErrorPattern,
    VendorManifest,
    match_model_reasoning,
)
from llm_rosetta.shims.providers import get_model_list_transform, load_providers
from llm_rosetta.transforms import flatten_system_content
from llm_rosetta.types.ir import is_image_part, is_tool_result_part

__all__ = [
    # Chat conversion
    "ConversionError",
    "ConversionPipeline",
    "UnrepresentableFinishReason",
    "flatten_system_content",
    "get_converter_class",
    "get_converter_for_provider",
    # Typed pipelines (embeddings, rerank, decisions)
    "DecisionConversionPipeline",
    "EmbeddingConversionPipeline",
    "RerankConversionPipeline",
    "get_decision_converter",
    "get_embedding_converter",
    "get_rerank_converter",
    # Format catalog and model-type vocabulary
    "FORMATS_BY_TYPE",
    "FORMAT_SPECS",
    "FormatSpec",
    "MODEL_TYPES",
    "canonical_model_type",
    "default_path",
    "legacy_model_type",
    # Provider and shim names
    "ProviderType",
    "canonical_provider_name",
    "canonical_shim_name",
    "warn_legacy_provider_names",
    "warn_legacy_shim_names",
    # Shim registry, read side
    "ConnectionConfig",
    "ProviderShim",
    "ReasoningCapability",
    "SoftErrorPattern",
    "VendorManifest",
    "get_model_list_transform",
    "get_shim",
    "list_shims",
    "match_model_reasoning",
    "resolve_format",
    "resolve_model_type",
    "resolve_shim",
    # Shim registry, write side and loaders
    "load_providers",
    "load_providers_from_dir",
    "register_shim",
    "restore_registry",
    "snapshot_registry",
    "unregister_shim",
    # Fidelity and profiling
    "DeepProfiler",
    "FidelityChecker",
    # IR the host inspects
    "ToolDeclaration",
    "ToolNameMap",
    "is_image_part",
    "is_tool_result_part",
    "read_tool_declarations",
    "strip_internal_metadata",
    # Version
    "__version__",
]
