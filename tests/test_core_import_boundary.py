"""The translation core must not reach into the gateway package.

Two guards, because they catch different regressions:

* the AST scan finds an import at any depth — module scope, inside a
  function, absolute, relative or dynamic — in every non-gateway module;
* the subprocess check proves that importing the public core entry points
  does not initialize the gateway or the gateway-side telemetry stack,
  which an import statement alone can cause through a package's
  ``__init__``.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).parents[1] / "packages/llm-rosetta/src/llm_rosetta"
_GATEWAY = "llm_rosetta_gateway"
#: The deprecation shim in the core tree resolves to the same package.
_GATEWAY_ALIAS = "llm_rosetta.gateway"

# Modules the core may never initialize by accident.  ``usage`` is
# deliberately absent: it is the core-side metering subsystem and
# ``pipeline.chat`` reads its semantics.  The HTTP client is gateway-side
# too — the external shim-source loader lives in ``gateway.shims``.
_GATEWAY_SIDE = {
    "llm_rosetta.observability.capture",
    "llm_rosetta.observability.error_dump",
    "llm_rosetta.observability.maintenance",
    "llm_rosetta.observability.metrics",
    "llm_rosetta.observability.ops_log",
    "llm_rosetta.observability.persistence",
    "llm_rosetta.observability.profiling",
    "llm_rosetta.observability.request_log",
    "llm_rosetta.observability.retention",
    "llm_rosetta._vendor.httpclient",
    # The loader modules themselves: eager re-imports from ``shims`` would
    # bring them back, and the HTTP client is only reached from inside one.
    "llm_rosetta.shims.sources",
    "llm_rosetta.shims.removals",
    "llm_rosetta.shims.catalog",
}


def _absolute_module(path: Path, node: ast.ImportFrom) -> str:
    """Resolve an ImportFrom to its absolute module name."""
    if not node.level:
        return node.module or ""
    package = ["llm_rosetta", *path.relative_to(_PKG_ROOT).parent.parts]
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


def _gateway_imports(path: Path, source: str) -> list[str]:
    hits: list[str] = []
    for node in ast.walk(ast.parse(source)):
        hit: str | None = None
        if isinstance(node, ast.ImportFrom):
            hit = _from_hit(path, node)
        elif isinstance(node, ast.Import):
            hit = _import_hit(path, node)
        elif isinstance(node, ast.Call):
            target = _dynamic_target(node)
            if target is not None and _is_gateway(target):
                hit = f"{path}:{node.lineno} <dynamic> {target}"
        if hit is not None:
            hits.append(hit)
    return hits


def test_core_sources_never_import_the_gateway():
    offenders: list[str] = []
    for path in _PKG_ROOT.rglob("*.py"):
        rel = path.relative_to(_PKG_ROOT)
        if rel.parts[0] in {"gateway", "_vendor"}:
            continue
        offenders.extend(_gateway_imports(path, path.read_text()))
    assert offenders == []


def test_guard_catches_absolute_relative_and_dynamic_imports():
    probe = _PKG_ROOT / "pipeline" / "chat.py"

    assert _gateway_imports(probe, "from llm_rosetta_gateway import x")
    assert _gateway_imports(probe, "from ..gateway.proxy import x")
    assert _gateway_imports(probe, "from .. import gateway as _gw")
    assert _gateway_imports(
        probe, 'import importlib\nimportlib.import_module("llm_rosetta_gateway.x")'
    )
    assert _gateway_imports(probe, '__import__("llm_rosetta_gateway.x")')
    assert _gateway_imports(probe, "from llm_rosetta import gateway as _gw")
    assert _gateway_imports(probe, "from .usage.semantics import shape_for") == []


def test_core_entry_points_load_no_gateway_module():
    code = f"""
import sys
import llm_rosetta.converters
import llm_rosetta.ir
import llm_rosetta.pipeline
import llm_rosetta.shims

bad = sorted(
    m for m in sys.modules
    if m == "llm_rosetta_gateway" or m.startswith("llm_rosetta_gateway.")
)
bad += sorted(sys.modules.keys() & {_GATEWAY_SIDE!r})
print("\\n".join(bad))
"""
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert out == "", out


def test_observability_facade_still_exposes_every_public_name():
    import llm_rosetta.observability as observability

    missing = [
        name for name in observability.__all__ if not hasattr(observability, name)
    ]
    assert missing == []

    try:
        observability.__getattr__("does_not_exist")
    except AttributeError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("unknown names must raise AttributeError")
