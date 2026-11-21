"""Tests to enforce the curated public API surface.

Ensures that:
- Each __all__ contains only the expected stable exports
- Every item listed in __all__ is actually importable
- Export counts stay within bounds to prevent accidental surface growth
"""

from __future__ import annotations

import importlib
from typing import TypedDict

import pytest


class _ModuleSpec(TypedDict):
    items: list[str] | None
    max_count: int


# Expected stable exports per module
EXPECTED_EXPORTS: dict[str, _ModuleSpec] = {
    "llm_rosetta": {
        "items": [
            # Converters
            "BaseConverter",
            "BaseSimpleConverter",
            "ChatCompletionsConverter",
            "MessagesConverter",
            "GoogleGenerateConverter",
            "GoogleInteractionsConverter",
            "OpenAIResponsesConverter",
            "OpenResponsesConverter",
            "AcpConverter",
            # Embedding converters
            "OpenAIEmbeddingConverter",
            "JinaEmbeddingConverter",
            "VoyageEmbeddingConverter",
            "CohereEmbeddingConverter",
            "GoogleEmbeddingConverter",
            "MiniMaxEmbeddingConverter",
            # Rerank converters
            "JinaRerankConverter",
            "CohereRerankConverter",
            "VoyageRerankConverter",
            # Conversion context
            "ConversionContext",
            "StreamContext",
            # Tool definition convenience API
            "tool_ops",
            # Auto-detection and conversion
            "detect_provider",
            "get_converter_for_provider",
            "convert",
            "convert_response",
            "ProviderType",
            # Conversion pipeline
            "ConversionPipeline",
            "ConversionError",
            "DecisionConversionPipeline",
            "EmbeddingConversionPipeline",
            "RerankConversionPipeline",
            # Provider shim layer
            "ConnectionConfig",
            "ToolsConfig",
            "ProviderShim",
            "register_shim",
            "unregister_shim",
            "get_shim",
            "list_shims",
            "resolve_format",
            "resolve_model_type",
            # Transforms
            "Transform",
            "apply_transforms",
            "force_fields",
            "strip_fields",
            "rename_field",
            "set_defaults",
        ],
        "max_count": 60,
    },
    # The host-facing surface (#1630).  Re-exports only; a name enters it with a
    # gateway call site or an external-shim use.  The ceiling leaves room for
    # the six functions added in PR 6 (``spec_for``, ``legacy_names`` (removed again in 0.14.0, never released),
    # ``typed_response_usage``, ``staging_registry``, ``reasoning_capability``,
    # ``describe_vendor_manifest``) and ``SURFACE_VERSION``.  The manifest-text
    # trio (``leaf_connection``, ``manifest_vocabulary``, ``merge_connection``)
    # replaces the gateway's last private reaches (#1630); ``plugin_source``,
    # ``builtin_providers_dir`` and the four scaffold entry points replace the
    # gateway's last non-surface imports.  ``ManifestVocabulary`` lets a host
    # annotate what ``manifest_vocabulary()`` returns (#1630).
    "llm_rosetta.api": {
        "items": [
            # Chat conversion
            "ConversionError",
            "ConversionPipeline",
            "UnrepresentableFinishReason",
            "flatten_system_content",
            "get_converter_class",
            "get_converter_for_provider",
            # Typed pipelines
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
            "MODEL_TYPES",
            "FormatSpec",
            "default_path",
            "spec_for",
            # Provider and shim names
            "ProviderType",
            "retired_names",
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
            "reasoning_capability",
            "resolve_format",
            "resolve_model_type",
            "resolve_shim",
            # Shim registry, write side and loaders
            "ManifestVocabulary",
            "builtin_providers_dir",
            "describe_vendor_manifest",
            "leaf_connection",
            "load_providers",
            "load_providers_from_dir",
            "manifest_vocabulary",
            "merge_connection",
            "plugin_source",
            "register_shim",
            "restore_registry",
            "snapshot_registry",
            "staging_registry",
            "unregister_shim",
            "validate_shim_declaration",
            # Shim scaffolding
            "add_scaffold_arguments",
            "run_scaffold",
            "run_scaffold_check",
            "run_scaffold_migrate",
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
            "SURFACE_VERSION",
            "__version__",
        ],
        "max_count": 63,
    },
    "llm_rosetta.types.ir": {
        "items": None,  # too many to enumerate; just check count
        # 99 with the batch job envelope (epic #1509): five record types, the
        # state and mode literals, and the counts mapping.
        "max_count": 100,
    },
    "llm_rosetta.types": {
        "items": None,
        "max_count": 95,
    },
    "llm_rosetta.converters.base": {
        "items": [
            "BaseConverter",
            "BaseSimpleConverter",
            "BaseRerankConverter",
            "BaseEmbeddingConverter",
            "BaseDecisionConverter",
            "ConversionContext",
            "ConversionWarning",
            "MetadataMode",
            "StreamContext",
            "WarningSink",
            "current_sink",
            "report_drop",
        ],
        "max_count": 13,
    },
    "llm_rosetta.contracts.usage": {
        "items": [
            "ADDITIVE_CACHE_STANDARDS",
            "DEFAULT_MODEL_TYPE",
            "GOOGLE_METADATA",
            "INTERNAL_LABEL",
            "INTERNAL_SUBJECT",
            "IR_FIELDS",
            "LEDGER_COLUMNS",
            "MESSAGES_FIELDS",
            "NO_SUBJECT_LABEL",
            "OPENAI_DETAILS",
            "STANDARD_SHAPES",
            "USAGE_SHAPES",
            "USAGE_WINDOW_LABELS",
            "USAGE_WINDOWS",
            "SubjectUsage",
            "UsageLedger",
            "UsageRecord",
            "UsageReport",
            "UsageShape",
            "add_windows",
            "cache_is_additive",
            "shape_for",
            "token_total",
            "usage_columns",
            "zero_windows",
        ],
        "max_count": 25,
    },
    # The extension contracts (#1657): literal lists, so a name enters or
    # leaves a contract only with a change here.
    "llm_rosetta.contracts": {
        "items": [
            "AUTH_EXCHANGE",
            "AUTH_MODULE",
            "IR_TRANSFORMS",
            "IR_VERSION",
            "MODEL_LIST_TRANSFORM",
            "POST_IR_TRANSFORMS",
            "PRE_IR_TRANSFORMS",
            "QUOTA_MODULE",
            "QUOTA_PROBES",
            "RESPONSE_BODY_TRANSFORMS",
            "SURFACE_VERSION",
            "TRANSFORMS_MODULE",
        ],
        "max_count": 12,
    },
    "llm_rosetta.contracts.shim": {
        "items": [
            "TYPE_PARAM_KEYS",
            "BatchDeclaration",
            "ConnectionConfig",
            "ProviderShim",
            "QuotaDeclaration",
            "ReasoningCapability",
            "ReplayMissing",
            "SoftErrorPattern",
            "ToolSearchMode",
            "ToolsConfig",
            "UnsignedBlocks",
            "VendorManifest",
            "load_providers_from_dir",
        ],
        "max_count": 13,
    },
    "llm_rosetta.contracts.transform": {
        "items": [
            "IRTransform",
            "NamedIRTransform",
            "NamedTransform",
            "Transform",
            "TransformContext",
            "apply_ir_transforms",
            "apply_transforms",
            "backfill_reasoning_content",
            "default_message_field",
            "default_tool_description",
            "flatten_system_content",
            "force_fields",
            "rename_field",
            "replace_message_field",
            "rewrite_harmony_tool_calls",
            "set_defaults",
            "strip_fields",
            "strip_fields_for_model",
            "auto_cache_breakpoints",
            "hoist_late_system_messages",
            "truncate_images",
            "unwind_parallel_tool_calls",
        ],
        "max_count": 22,
    },
    "llm_rosetta.contracts.auth": {
        "items": [
            "DEFAULT_CREDENTIAL_ROLE",
            "FIELD_KEY_PATTERN",
            "FIELD_KEY_RE",
            "FIELD_LIMIT",
            "FIELD_TEXT_LIMIT",
            "AuthContext",
            "AuthExchange",
            "AuthExchangeError",
            "BaseAuthExchange",
            "InteractiveSetupRequired",
            "credential_for",
        ],
        "max_count": 11,
    },
    # ``api`` re-exports the module object, so its ``__all__`` is the surface a
    # host reads through ``api.retired_names`` (#919 decision 4, #1630).  Every
    # attribute the gateway and core use is listed.
    "llm_rosetta.retired_names": {
        "items": [
            "KINDS",
            "MIGRATE_HINT",
            "MIGRATION_GUIDE_URL",
            "MODEL_TYPE",
            "PROVIDER_TYPE",
            "RETIRED_MODULES",
            "RETIRED_NAMES",
            "SHIM",
            "RetiredModule",
            "RetiredName",
            "check_not_retired",
            "lookup",
            "replacement_for",
            "retired_key_message",
            "retired_module",
            "retired_module_message",
            "retired_name_message",
        ],
        "max_count": 17,
    },
}


@pytest.mark.parametrize("module_path", list(EXPECTED_EXPORTS.keys()))
class TestPublicAPI:
    def test_all_items_importable(self, module_path: str) -> None:
        """Every item in __all__ must be importable from the module."""
        mod = importlib.import_module(module_path)
        all_exports: list[str] = getattr(mod, "__all__", [])
        for name in all_exports:
            assert hasattr(mod, name), (
                f"{module_path}.__all__ lists '{name}' but it is not "
                f"importable from {module_path}"
            )

    def test_export_count_within_bounds(self, module_path: str) -> None:
        """Export count must not exceed the configured maximum."""
        mod = importlib.import_module(module_path)
        all_exports: list[str] = getattr(mod, "__all__", [])
        max_count = EXPECTED_EXPORTS[module_path]["max_count"]
        assert len(all_exports) <= max_count, (
            f"{module_path}.__all__ has {len(all_exports)} items, "
            f"exceeding the maximum of {max_count}. "
            f"If this is intentional, update EXPECTED_EXPORTS in test_public_api.py."
        )

    def test_expected_items_present(self, module_path: str) -> None:
        """If specific items are expected, they must all be present."""
        expected_items = EXPECTED_EXPORTS[module_path]["items"]
        if expected_items is None:
            pytest.skip("No specific items to check for this module")
        mod = importlib.import_module(module_path)
        all_exports = set(getattr(mod, "__all__", []))
        for name in expected_items:
            assert name in all_exports, (
                f"Expected '{name}' in {module_path}.__all__ but it is missing"
            )

    def test_no_unexpected_items(self, module_path: str) -> None:
        """If specific items are expected, no extra items should be present."""
        expected_items = EXPECTED_EXPORTS[module_path]["items"]
        if expected_items is None:
            pytest.skip("No specific items to check for this module")
        mod = importlib.import_module(module_path)
        all_exports = set(getattr(mod, "__all__", []))
        expected_set = set(expected_items)
        unexpected = all_exports - expected_set
        assert not unexpected, (
            f"{module_path}.__all__ contains unexpected items: {unexpected}. "
            f"If these are intentional stable exports, add them to "
            f"EXPECTED_EXPORTS in test_public_api.py."
        )
