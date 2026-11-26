"""The IR layer must not import ``converters``, at any depth.

The direction is ``ir.model <- ir.primitives/ir.tools <- ir.passes <-
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
_MODEL = _IR / "model"
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


def _hits(path: Path, predicate) -> list[str]:
    package = ("llm_rosetta", "ir", *path.relative_to(_IR).parent.parts)
    return import_hits(path, path.read_text(), package, predicate)


def _outside_the_model(name: str) -> bool:
    """A ``llm_rosetta`` import that is neither the model itself nor ``_vendor``."""
    if name != "llm_rosetta" and not name.startswith("llm_rosetta."):
        return False
    allowed = ("llm_rosetta.ir.model", "llm_rosetta._vendor")
    return not any(name == a or name.startswith(a + ".") for a in allowed)


def test_the_model_imports_only_the_standard_library_and_the_vendor_dir():
    """``ir.model`` is the bottom of the IR: data classes, guards, helpers."""
    offenders: list[str] = []
    for path in _MODEL.rglob("*.py"):
        offenders.extend(_hits(path, _outside_the_model))
    assert offenders == []


def test_primitives_and_tools_never_import_the_passes():
    """``ir.passes`` sits above ``ir.primitives`` and ``ir.tools``."""
    offenders: list[str] = []
    for layer in _LAYERS:
        for path in layer.rglob("*.py"):
            offenders.extend(
                _hits(
                    path,
                    lambda n: (
                        n == "llm_rosetta.ir.passes"
                        or n.startswith("llm_rosetta.ir.passes.")
                    ),
                )
            )
    assert offenders == []


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

    clean = "from llm_rosetta.ir.model import Message\nfrom .image_limit import x\n"
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
    """The re-exports hand out the same objects, not copies."""
    from llm_rosetta.contracts.transform import strip_fields as contract_transform
    from llm_rosetta.converters.base.tools import sanitize_tool_call_id as legacy_id
    from llm_rosetta.ir.primitives.body import strip_fields as canonical_transform
    from llm_rosetta.ir.tools import sanitize_tool_call_id as canonical_id

    assert legacy_id is canonical_id
    assert contract_transform is canonical_transform


def test_the_removed_facade_stays_removed():
    """The ``llm_rosetta.transforms`` facade was removed before 0.14.0 (#1657,
    D3): shims import ``llm_rosetta.contracts.transform``, core imports
    ``llm_rosetta.ir``.  ``llm_rosetta.capabilities`` is the 0.13.0 flat
    module again, a deprecated forwarder (``tests/test_compat_forwarders.py``)."""
    import importlib.util

    assert importlib.util.find_spec("llm_rosetta.transforms") is None


def test_core_does_not_import_the_compat_forwarders():
    """The 0.13.0 forwarders are for outside users; core reads the new homes."""
    hits = []
    forwarders = (_IR.parent / "routing.py", _IR.parent / "capabilities.py")
    for path in sorted(_IR.parent.rglob("*.py")):
        if "_vendor" in path.parts or path in forwarders:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in (
                "llm_rosetta.routing",
                "llm_rosetta.capabilities",
            ):
                hits.append(f"{path.relative_to(_IR.parent)}:{node.lineno}")
    assert hits == []
