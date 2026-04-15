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
            "OpenAIChatConverter",
            "AnthropicConverter",
            "GoogleGenAIConverter",
            "GoogleConverter",
            "OpenAIResponsesConverter",
            # Conversion context
            "ConversionContext",
            "StreamContext",
            # Tool definition convenience API
            "tool_ops",
            # Auto-detection and conversion
            "detect_provider",
            "get_converter_for_provider",
            "convert",
            "ProviderType",
        ],
        "max_count": 16,
    },
    "llm_rosetta.types.ir": {
        "items": None,  # too many to enumerate; just check count
        "max_count": 60,
    },
    "llm_rosetta.types": {
        "items": None,
        "max_count": 60,
    },
    "llm_rosetta.converters.base": {
        "items": [
            "BaseConverter",
            "ConversionContext",
            "MetadataMode",
            "StreamContext",
        ],
        "max_count": 5,
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
