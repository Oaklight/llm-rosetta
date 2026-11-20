"""The v0.13.0 helper paths left in 0.14.0 (#919).

``converters/base/helpers/{image_limit,cache_breakpoints,system_message_hoist}``
were importable in 0.13.0.  They moved to ``llm_rosetta.ir.primitives``.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "name", ["image_limit", "cache_breakpoints", "system_message_hoist"]
)
def test_0130_helper_path_is_gone(name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"llm_rosetta.converters.base.helpers.{name}")

    importlib.import_module(f"llm_rosetta.ir.primitives.{name}")
