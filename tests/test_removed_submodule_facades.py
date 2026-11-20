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
