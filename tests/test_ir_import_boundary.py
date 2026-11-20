"""The IR layer must not import ``converters``, at any depth.

The direction is ``types.ir <- ir.primitives/ir.tools <- ir.passes <-
converters``.  The whole of ``ir`` (passes included) imports nothing from
``converters``: the context contract a pass needs lives in ``ir.context``, and
``converters.base.context`` re-exports it for old import paths.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from _import_scan import import_hits

_IR = Path(__file__).parents[1] / "packages/llm-rosetta/src/llm_rosetta/ir"
_LAYERS = (_IR / "primitives", _IR / "tools")
_CONVERTERS = "llm_rosetta.converters"


def _is_converters(name: str) -> bool:
    return name == _CONVERTERS or name.startswith(_CONVERTERS + ".")


def _converter_imports(path: Path, source: str) -> list[str]:
    """Return every import in *source* that reaches ``llm_rosetta.converters``."""
    package = ("llm_rosetta", "ir", *path.relative_to(_IR).parent.parts)
    return import_hits(path, source, package, _is_converters)


def test_primitives_and_tools_never_import_converters():
    offenders: list[str] = []
    for layer in _LAYERS:
        for path in layer.rglob("*.py"):
            offenders.extend(_converter_imports(path, path.read_text()))
    assert offenders == []


def test_ir_as_a_whole_never_imports_converters():
    """No allowed exceptions: ``ir.passes`` uses ``ir.context`` instead."""
    offenders: list[str] = []
    for path in _IR.rglob("*.py"):
        offenders.extend(_converter_imports(path, path.read_text()))
    assert offenders == []


def test_context_old_path_re_exports_the_same_objects():
    from llm_rosetta.converters.base import context as legacy
    import llm_rosetta.ir.context as canonical

    surface = (
        "ConversionContext",
        "StreamContext",
        "ImageFetcher",
        "MetadataMode",
        "_REQUEST_TOOL_VOCABULARY_KEY",
        "carry_registered_call_identity",
    )
    # The shim's whole surface is pinned: dropping any name, or adding one
    # nobody checks, turns this red.
    assert set(legacy.__all__) == set(surface)
    for name in surface:
        assert getattr(legacy, name) is getattr(canonical, name)


def test_guard_catches_absolute_relative_and_dynamic_imports():
    top = _IR / "primitives" / "ir.py"
    nested = _IR / "primitives" / "nested" / "helper.py"

    assert _converter_imports(top, "from llm_rosetta.converters import x")
    assert _converter_imports(top, "from ...converters.base import context")
    assert _converter_imports(nested, "from ....converters.base import context")
    assert _converter_imports(
        top, 'import importlib\nimportlib.import_module("llm_rosetta.converters.x")'
    )
    assert _converter_imports(top, '__import__("llm_rosetta.converters.x")')

    clean = "from llm_rosetta.types.ir import Message\nfrom .image_limit import x\n"
    assert _converter_imports(top, clean) == []


def test_shims_import_works_in_a_fresh_interpreter():
    """The cycle is gone: the shim package imports with no deferred breaker.

    The boundary scan above is what proves there is no function-level escape
    hatch left; this is the end-to-end check that the graph resolves.
    """
    subprocess.check_call([sys.executable, "-c", "import llm_rosetta.shims"])


def test_ir_init_imports_nothing():
    """``ir.passes`` imports the shim contract, so ``ir`` must stay lazy."""
    tree = ast.parse((_IR / "__init__.py").read_text())
    imports = [
        (node.level, node.module)
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    ]
    assert imports in ([], [(0, "__future__")])


def test_old_paths_are_identity_reattachments():
    """The facades re-export the same objects, not copies."""
    from llm_rosetta.capabilities import enforce_vision as legacy_pass
    from llm_rosetta.converters.base.tools import sanitize_tool_call_id as legacy_id
    from llm_rosetta.converters.base.tools.intrinsic import (
        intrinsic_kind_of as legacy_intrinsic,
    )
    from llm_rosetta.ir.passes import enforce_vision as canonical_pass
    from llm_rosetta.ir.tools import sanitize_tool_call_id as canonical_id
    from llm_rosetta.ir.tools.intrinsic import intrinsic_kind_of as canonical_intrinsic
    from llm_rosetta.transforms import strip_fields as legacy_transform
    from llm_rosetta.ir.primitives.body import strip_fields as canonical_transform

    assert legacy_pass is canonical_pass
    assert legacy_id is canonical_id
    assert legacy_intrinsic is canonical_intrinsic
    assert legacy_transform is canonical_transform
