"""The v0.13.0 helper paths warn until 0.16.0 (#919).

``converters/base/helpers/{image_limit,cache_breakpoints,system_message_hoist}``
were importable in 0.13.0, so their shims announce the move rather than
disappearing silently.  The other moved modules never shipped under their old
names and re-export without a warning.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "name", ["image_limit", "cache_breakpoints", "system_message_hoist"]
)
def test_0130_helper_path_warns(name: str) -> None:
    module = importlib.import_module(f"llm_rosetta.converters.base.helpers.{name}")

    with pytest.warns(DeprecationWarning, match="ir.primitives"):
        importlib.reload(module)


def test_helper_shim_reexports_the_same_objects() -> None:
    from llm_rosetta.converters.base.helpers import image_limit as legacy
    from llm_rosetta.ir.primitives import image_limit as canonical

    assert legacy.truncate_images is canonical.truncate_images
    assert legacy.strip_images_for_non_vision is canonical.strip_images_for_non_vision
