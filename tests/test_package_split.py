"""The gateway is its own distribution; the old import root is a shim (#1157).

`llm-rosetta` no longer ships the gateway.  ``llm_rosetta.gateway`` stays
importable until 0.16.0 (#919) but resolves to ``llm_rosetta_gateway``: the
same module objects, so state is never duplicated, and a warning so the
migration is visible.
"""

from __future__ import annotations

import importlib
import sys
import warnings

from pathlib import Path


def test_importing_the_old_root_warns() -> None:
    for name in [n for n in sys.modules if n == "llm_rosetta.gateway"]:
        sys.modules.pop(name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("llm_rosetta.gateway")

    messages = [
        str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert any("llm_rosetta_gateway" in message for message in messages)


def test_old_submodules_are_the_same_objects() -> None:
    old = importlib.import_module("llm_rosetta.gateway.config")
    new = importlib.import_module("llm_rosetta_gateway.config")

    assert old is new


def test_the_core_package_ships_only_a_shim_for_the_gateway() -> None:
    import llm_rosetta

    core = Path(llm_rosetta.__file__).parent
    gateway_dir = core / "gateway"

    assert sorted(path.name for path in gateway_dir.glob("*.py")) == ["__init__.py"]
    assert not (gateway_dir / "app").exists()
