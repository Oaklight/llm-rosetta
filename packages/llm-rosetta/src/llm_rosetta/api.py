"""The small public surface that hosts import from ``llm_rosetta`` (#1630).

A host (the gateway) builds on core through this module and
:mod:`llm_rosetta.contracts`; a shim builds on the contracts alone.
Everything else under ``llm_rosetta`` is internal and may move in any release
(``llm_rosetta._tiers`` states each module's tier).

The two allowed import prefixes are:

- ``llm_rosetta.api`` (this module)
- ``llm_rosetta.contracts`` and its submodules (``contracts.usage`` as a
  package, never its submodules)

Every name has one home.  ``llm_rosetta.contracts`` holds what outside parties
implement or receive (the shim data classes, the command and sign-in contracts,
``SURFACE_VERSION``); this module holds what a host calls (the registry, the
loaders, the installer, the converters, ``run_cli``).  A contract name is not
repeated here, and a name that neither needs is not exported at all.  The test
helpers that snapshot and restore the registry are in
:mod:`llm_rosetta.testing`.

Stability rule: within one minor release this module only gains names, and
``SURFACE_VERSION`` (defined in :mod:`llm_rosetta.contracts`) stays put; it is
bumped on a removal or a changed meaning, never on an addition.  A name
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

from llm_rosetta import (
    ProviderType,
    __version__,
    get_converter_for_provider,
    retired_names,
)
from llm_rosetta.ir.passes.tool_names import ToolNameMap
from llm_rosetta.cli import main as run_cli
from llm_rosetta.contracts.command import (
    CORE_COMMAND_NAMES,
    CommandSpec,
    register_command,
    resolve_command,
)
from llm_rosetta.converters import get_converter_class
from llm_rosetta.extensions import (
    ENV_VAR as EXTENSIONS_DIR_ENV,
)
from llm_rosetta.extensions import (
    ExtensionInstallError,
    ExtensionStatus,
    ExternalStatus,
    InstalledExtension,
    RetiredEntryPoint,
    activate_extensions_dir,
    check_external,
    configure_extensions_dir,
    extensions_dir,
    import_extension_module,
    install_with_pip,
    install_wheel,
    install_wheel_url,
    installed_extensions,
    list_extensions,
    missing_dependencies,
    pip_available,
    remove_extension,
    retired_entry_points,
)
from llm_rosetta.converters.base.errors import UnrepresentableFinishReason
from llm_rosetta.converters.base.helpers import strip_internal_metadata
from llm_rosetta.fidelity import FidelityChecker
from llm_rosetta.formats import (
    FORMAT_SPECS,
    MODEL_TYPES,
    FormatSpec,
    spec_for,
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
    typed_response_usage,
)
from llm_rosetta.ir.passes.enforcement import reasoning_capability
from llm_rosetta.profiling import DeepProfiler
from llm_rosetta.shims import (
    get_shim,
    list_shims,
    load_providers_from_dir,
    register_shim,
    resolve_format,
    resolve_model_type,
    resolve_shim,
    unregister_shim,
)
from llm_rosetta.contracts.shim import SoftErrorPattern
from llm_rosetta.shims.provider_shim import match_model_reasoning, staging_registry
from llm_rosetta.shims._loader import (
    builtin_providers_dir,
    get_model_list_transform,
    leaf_connection,
    load_providers,
    manifest_vocabulary,
    merge_connection,
    plugin_source,
    validate_shim_declaration,
)

from llm_rosetta.contracts.transform import flatten_system_content
from llm_rosetta.ir.model import is_image_part, is_tool_result_part


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
    "typed_response_usage",
    # Format catalog and model-type vocabulary
    "FORMATS_BY_TYPE",
    "FORMAT_SPECS",
    "FormatSpec",
    "MODEL_TYPES",
    "default_path",
    "spec_for",
    # Provider and shim names
    "ProviderType",
    "retired_names",
    # Shim registry, read side
    "SoftErrorPattern",
    "get_model_list_transform",
    "get_shim",
    "list_shims",
    "match_model_reasoning",
    "reasoning_capability",
    "resolve_format",
    "resolve_model_type",
    "resolve_shim",
    # Shim registry, write side and loaders
    "builtin_providers_dir",
    "leaf_connection",
    "load_providers",
    "load_providers_from_dir",
    "manifest_vocabulary",
    "merge_connection",
    "plugin_source",
    "register_shim",
    "staging_registry",
    "unregister_shim",
    "validate_shim_declaration",
    # Subcommands a distribution adds to the ``llm-rosetta`` dispatcher
    "CORE_COMMAND_NAMES",
    "CommandSpec",
    "register_command",
    "resolve_command",
    "run_cli",
    # Installed extensions and what happened to each at load time
    "ExtensionStatus",
    "list_extensions",
    # Installing extensions (a standard-library installer and directory manager)
    "EXTENSIONS_DIR_ENV",
    "ExtensionInstallError",
    "ExternalStatus",
    "InstalledExtension",
    "RetiredEntryPoint",
    "activate_extensions_dir",
    "check_external",
    "configure_extensions_dir",
    "extensions_dir",
    "import_extension_module",
    "install_wheel",
    "install_wheel_url",
    "install_with_pip",
    "installed_extensions",
    "missing_dependencies",
    "pip_available",
    "remove_extension",
    "retired_entry_points",
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
