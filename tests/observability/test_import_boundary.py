"""The observability layer must not import the gateway, at any depth."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

_OBSERVABILITY = (
    Path(__file__).parents[2] / "packages/llm-rosetta/src/llm_rosetta/observability"
)
_GATEWAY = "llm_rosetta_gateway"
#: The deprecation shim in the core tree resolves to the same package.
_GATEWAY_ALIAS = "llm_rosetta.gateway"


def _absolute_module(path: Path, node: ast.ImportFrom) -> str:
    """Resolve an ImportFrom to its absolute module name.

    Relative imports are the natural way to reach the gateway from inside the
    package (``from ..gateway...``), so ``level`` has to be applied, not
    ignored.
    """
    if not node.level:
        return node.module or ""
    package = [
        "llm_rosetta",
        "observability",
        *path.relative_to(_OBSERVABILITY).parent.parts,
    ]
    base = package[: len(package) - (node.level - 1)]
    if node.module:
        base += node.module.split(".")
    return ".".join(base)


def _is_gateway(name: str) -> bool:
    return name in (_GATEWAY, _GATEWAY_ALIAS) or name.startswith(
        (_GATEWAY + ".", _GATEWAY_ALIAS + ".")
    )


def _from_hit(path: Path, node: ast.ImportFrom) -> str | None:
    module = _absolute_module(path, node)
    if _is_gateway(module):
        return f"{path}:{node.lineno} {module}"
    for alias in node.names:
        if alias.name == "*":
            continue
        qualified = f"{module}.{alias.name}" if module else alias.name
        if _is_gateway(qualified):
            return f"{path}:{node.lineno} {qualified}"
    return None


def _import_hit(path: Path, node: ast.Import) -> str | None:
    for alias in node.names:
        if _is_gateway(alias.name):
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


def _dynamic_hit(path: Path, node: ast.Call) -> str | None:
    target = _dynamic_target(node)
    if target is not None and _is_gateway(target):
        return f"{path}:{node.lineno} <dynamic> {target}"
    return None


def _node_hit(path: Path, node: ast.AST) -> str | None:
    if isinstance(node, ast.ImportFrom):
        return _from_hit(path, node)
    if isinstance(node, ast.Import):
        return _import_hit(path, node)
    if isinstance(node, ast.Call):
        return _dynamic_hit(path, node)
    return None


def _gateway_imports(path: Path, source: str) -> list[str]:
    """Return every import in *source* that reaches ``llm_rosetta_gateway``."""
    hits = (_node_hit(path, node) for node in ast.walk(ast.parse(source)))
    return [hit for hit in hits if hit is not None]


def test_observability_sources_never_import_the_gateway():
    """Catch a function-local regression, which a plain import would not.

    The removed violation lived inside ``dump_error``; importing the module
    never executed it.  Walking the AST finds imports at any depth, so a
    reintroduced gateway import — module scope or inside a function — fails.
    """
    offenders: list[str] = []
    for path in _OBSERVABILITY.rglob("*.py"):
        offenders.extend(_gateway_imports(path, path.read_text()))
    assert offenders == []


def test_guard_catches_absolute_relative_and_from_import():
    """The guard has teeth for the shapes a reviewer would actually write."""
    top = _OBSERVABILITY / "error_dump.py"
    nested = _OBSERVABILITY / "usage" / "cli.py"

    assert _gateway_imports(top, "from llm_rosetta_gateway import x")
    assert _gateway_imports(top, "from ..gateway.middleware import sanitize")
    assert _gateway_imports(top, "from .. import gateway")
    assert _gateway_imports(top, "from llm_rosetta import gateway as _gw")
    assert _gateway_imports(
        top, 'import importlib\nimportlib.import_module("llm_rosetta_gateway.x")'
    )
    assert _gateway_imports(top, '__import__("llm_rosetta_gateway.x")')
    # From observability.usage one more dot is needed to leave observability.
    assert not _gateway_imports(nested, "from ..gateway import x")
    assert _gateway_imports(nested, "from ...gateway import x")

    clean = "from .sanitize import scrub_credential_patterns\nimport re\n"
    assert _gateway_imports(top, clean) == []


def test_error_dump_import_does_not_load_gateway():
    """Importing an observability module must not pull the gateway in."""
    code = (
        "import sys; import llm_rosetta.observability.error_dump; "
        "print(int(any(m.startswith('llm_rosetta_gateway') for m in sys.modules)))"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert out == "0"


def test_scrub_is_identical_through_both_paths():
    from llm_rosetta_gateway.middleware.sanitize import (
        scrub_credential_patterns as legacy,
    )
    from llm_rosetta.observability.sanitize import (
        scrub_credential_patterns as canonical,
    )

    assert legacy is canonical
    assert canonical("Bearer abcdef") == "[REDACTED]"
