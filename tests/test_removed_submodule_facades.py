"""The silent submodule facades are gone; the package-level names stay (#919)."""

from __future__ import annotations

import importlib

import pytest

REMOVED = [
    "llm_rosetta.converters.base.helpers.tool_group",
    "llm_rosetta.converters.base.helpers.truncate",
    "llm_rosetta.converters.base.tools.allowed_tools",
    "llm_rosetta.converters.base.tools.batch",
    "llm_rosetta.converters.base.tools.call_id",
    "llm_rosetta.converters.base.tools.call_unwind",
    "llm_rosetta.converters.base.tools.error_marker",
    "llm_rosetta.converters.base.tools.intrinsic",
    "llm_rosetta.converters.base.tools.orphan_fix",
    "llm_rosetta.converters.base.tools.strict_mode",
    "llm_rosetta_gateway.middleware.sanitize",
]


@pytest.mark.parametrize("path", REMOVED)
def test_removed_facade_does_not_import(path: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(path)


def test_package_level_names_still_resolve_to_the_canonical_objects() -> None:
    from llm_rosetta.converters.base.helpers import truncate_with_digest
    from llm_rosetta.converters.base.tools import (
        BaseToolOps,
        intrinsic_kind_of,
        sanitize_tool_call_id,
    )
    from llm_rosetta.ir.primitives.truncate import truncate_with_digest as canon_t
    from llm_rosetta.ir.tools.call_id import sanitize_tool_call_id as canon_id
    from llm_rosetta.ir.tools.intrinsic import intrinsic_kind_of as canon_k

    assert BaseToolOps is not None
    assert truncate_with_digest is canon_t
    assert sanitize_tool_call_id is canon_id
    assert intrinsic_kind_of is canon_k


# The package-level API is documented stable and now the only path to these
# names, so its ``__all__`` is pinned literally.  Adding a name is a deliberate
# edit here; removing one is a breaking change that needs an allowlist entry.
TOOLS_ALL = [
    "BaseToolOps",
    "IntrinsicLeg",
    "TOOL_CONTENT_CLOSE_TAG",
    "TOOL_CONTENT_OPEN_TAG_RE",
    "TOOL_ERROR_MARKER",
    "UnrepairableToolHistory",
    "allowed_tool_names",
    "allowed_tool_refs",
    "assign_tool_batch_ids",
    "convert_content_blocks_to_ir",
    "convert_ir_content_blocks_to_p",
    "convert_nullable_to_type_array",
    "extract_part_ids",
    "get_definition_kind",
    "get_intrinsic_kind",
    "get_native_definition",
    "has_multimodal_content",
    "inject_packed_tool_content",
    "intrinsic_kind_of",
    "ir_strict",
    "is_intrinsic",
    "is_intrinsic_part",
    "is_synthetic_tool_content_msg",
    "log_orphan_warnings",
    "make_intrinsic_tool_call",
    "make_intrinsic_tool_definition",
    "make_intrinsic_tool_result",
    "mark_tool_error",
    "mark_tool_error_text",
    "merge_tool_messages",
    "normalize_numeric_constraints",
    "normalize_tool_pairing",
    "pack_multimodal_tool_result",
    "prefixed_tool_call_id",
    "provider_allowed_tools_to_ir",
    "sanitize_schema",
    "sanitize_tool_call_id",
    "sanitize_tool_parameters",
    "sink_leg",
    "strip_orphaned_tool_config",
    "unpack_tool_content",
    "unwind_parallel_tool_calls_ir",
    "warn_tool_error",
    "warn_tool_error_dropped",
    "write_ir_strict",
]
HELPERS_ALL = [
    "CARRIER_KEY",
    "DEFAULT_REASONING_CAPS",
    "DUMMY_THOUGHT_SIGNATURE",
    "OPAQUE_SAME_ORIGIN",
    "apply_reasoning_config",
    "carrier_metadata",
    "get_opaque_origin",
    "google_thought_signature",
    "hoist_late_system_messages_ir",
    "inject_google_function_call_signature",
    "inject_interactions_function_call_signature",
    "is_own_origin",
    "set_google_thought_signature",
    "set_opaque_origin",
    "strip_internal_metadata",
    "truncate_with_digest",
]


@pytest.mark.parametrize(
    ("module", "expected"),
    [
        ("llm_rosetta.converters.base.tools", TOOLS_ALL),
        ("llm_rosetta.converters.base.helpers", HELPERS_ALL),
    ],
)
def test_the_package_all_is_pinned_literally(module: str, expected: list[str]) -> None:
    package = importlib.import_module(module)
    assert sorted(package.__all__) == expected
    assert len(package.__all__) == len(set(package.__all__))
    for name in expected:
        assert hasattr(package, name), name
