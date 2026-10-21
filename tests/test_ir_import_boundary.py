"""The IR primitive/tool layer must not import ``converters``, at any depth.

The direction is ``types.ir <- ir.primitives/ir.tools <- ir.passes <-
converters``.  ``ir.passes`` is deliberately exempt: a pass applies a target's
declared capabilities, so it may use the converter *base* contract
(``ConversionContext``) and the shim contract — but not a converter family.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

_IR = Path(__file__).parents[1] / "packages/llm-rosetta/src/llm_rosetta/ir"
_LAYERS = (_IR / "primitives", _IR / "tools")
_CONVERTERS = "llm_rosetta.converters"


def _absolute_module(path: Path, node: ast.ImportFrom) -> str:
    """Resolve an ImportFrom to its absolute module name."""
    if not node.level:
        return node.module or ""
    package = ["llm_rosetta", "ir", *path.relative_to(_IR).parent.parts]
    base = package[: len(package) - (node.level - 1)]
    if node.module:
        base += node.module.split(".")
    return ".".join(base)


def _is_converters(name: str) -> bool:
    return name == _CONVERTERS or name.startswith(_CONVERTERS + ".")


def _from_hit(path: Path, node: ast.ImportFrom) -> str | None:
    module = _absolute_module(path, node)
    if _is_converters(module):
        return f"{path}:{node.lineno} {module}"
    for alias in node.names:
        if alias.name == "*":
            continue
        qualified = f"{module}.{alias.name}" if module else alias.name
        if _is_converters(qualified):
            return f"{path}:{node.lineno} {qualified}"
    return None


def _import_hit(path: Path, node: ast.Import) -> str | None:
    for alias in node.names:
        if _is_converters(alias.name):
            return f"{path}:{node.lineno} {alias.name}"
    return None


def _dynamic_target(node: ast.Call) -> str | None:
    is_import = (isinstance(node.func, ast.Name) and node.func.id == "__import__") or (
        isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"
    )
    if not is_import or not node.args:
        return None
    arg = node.args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    return None


def _node_hit(path: Path, node: ast.AST) -> str | None:
    if isinstance(node, ast.ImportFrom):
        return _from_hit(path, node)
    if isinstance(node, ast.Import):
        return _import_hit(path, node)
    if isinstance(node, ast.Call):
        target = _dynamic_target(node)
        if target is not None and _is_converters(target):
            return f"{path}:{node.lineno} <dynamic> {target}"
    return None


def _converter_imports(path: Path, source: str) -> list[str]:
    """Return every import in *source* that reaches ``llm_rosetta.converters``."""
    hits = (_node_hit(path, node) for node in ast.walk(ast.parse(source)))
    return [hit for hit in hits if hit is not None]


def test_primitives_and_tools_never_import_converters():
    offenders: list[str] = []
    for layer in _LAYERS:
        for path in layer.rglob("*.py"):
            offenders.extend(_converter_imports(path, path.read_text()))
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
