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

from _import_scan import absolute_module, dynamic_target, import_hits, package_parts

_PKG_ROOT = Path(__file__).parents[1] / "packages/llm-rosetta/src/llm_rosetta"
_GATEWAY = "llm_rosetta_gateway"
#: The deprecation shim in the core tree resolves to the same package.
_GATEWAY_ALIAS = "llm_rosetta.gateway"

# Modules the core may never initialize by accident.  ``usage`` is
# deliberately absent: it is the core-side metering subsystem and
# ``pipeline.chat`` reads its semantics.  The HTTP client is gateway-side
# too — the external shim-source loader lives in ``gateway.shims``.
_GATEWAY_SIDE = {
    "llm_rosetta_gateway.observability.capture",
    "llm_rosetta_gateway.observability.error_dump",
    "llm_rosetta_gateway.observability.maintenance",
    "llm_rosetta_gateway.observability.metrics",
    "llm_rosetta_gateway.observability.ops_log",
    "llm_rosetta_gateway.observability.persistence",
    "llm_rosetta_gateway.observability.profiling",
    "llm_rosetta_gateway.observability.request_log",
    "llm_rosetta_gateway.observability.retention",
    "llm_rosetta_gateway._vendor.httpclient",
    # The loader modules themselves: eager re-imports from ``shims`` would
    # bring them back, and the HTTP client is only reached from inside one.
    "llm_rosetta.shims.sources",
    "llm_rosetta.shims.removals",
    "llm_rosetta.shims.catalog",
}

#: The telemetry modules' old paths.  They still import through a deprecation
#: alias that resolves to the gateway package (#1161), so a core module
#: reaching for one is the same regression as importing the gateway directly.
_GATEWAY_TELEMETRY_ALIASES = {
    "llm_rosetta.observability.capture",
    "llm_rosetta.observability.error_dump",
    "llm_rosetta.observability.maintenance",
    "llm_rosetta.observability.metrics",
    "llm_rosetta.observability.ops_log",
    "llm_rosetta.observability.persistence",
    "llm_rosetta.observability.profiling",
    "llm_rosetta.observability.request_log",
    "llm_rosetta.observability.retention",
    "llm_rosetta.observability.sanitize",
}


def _is_gateway(name: str) -> bool:
    if name in (_GATEWAY, _GATEWAY_ALIAS) or name.startswith(
        (_GATEWAY + ".", _GATEWAY_ALIAS + ".")
    ):
        return True
    return any(
        name == moved or name.startswith(moved + ".")
        for moved in _GATEWAY_TELEMETRY_ALIASES
    )


def _gateway_imports(path: Path, source: str) -> list[str]:
    package = package_parts(path, _PKG_ROOT, "llm_rosetta")
    return import_hits(path, source, package, _is_gateway)


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

    # The literal ``__all__`` and the two resolution maps must agree; a name
    # added to one and not the other is invisible until someone imports it.
    declared = set(observability._LAZY) | set(observability._MOVED)
    assert set(observability.__all__) == declared

    try:
        observability.__getattr__("does_not_exist")
    except AttributeError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("unknown names must raise AttributeError")


#: The host-facing surface (#1630).  Core code must not import it: the
#: dependency runs from the host to core, and an import from inside core
#: would be a cycle.  A module-scope import fails earlier than this test: it
#: dies at import time with ``ImportError: cannot import name '__version__'
#: from partially initialized module 'llm_rosetta'``, because ``api`` imports
#: back into the package.  That is a circular import, not this test's message,
#: so a red suite caused by one points here: look for an ``llm_rosetta.api``
#: import in a core module.
_API = "llm_rosetta.api"


def _is_api(name: str) -> bool:
    return name == _API or name.startswith(_API + ".")


def _api_imports(path: Path, source: str) -> list[str]:
    hits: list[str] = []
    for node in ast.walk(ast.parse(source)):
        found: str | None
        if isinstance(node, ast.ImportFrom):
            module = absolute_module(
                package_parts(path, _PKG_ROOT, "llm_rosetta"), node
            )
            names = [module] + [
                f"{module}.{alias.name}" for alias in node.names if alias.name != "*"
            ]
            found = next((n for n in names if _is_api(n)), None)
        elif isinstance(node, ast.Import):
            found = next((a.name for a in node.names if _is_api(a.name)), None)
        elif isinstance(node, ast.Call):
            target = dynamic_target(node)
            found = target if target is not None and _is_api(target) else None
        else:
            continue
        if found is not None:
            hits.append(f"{path}:{node.lineno} {found}")
    return hits


def test_core_sources_never_import_the_api_surface():
    offenders: list[str] = []
    for path in _PKG_ROOT.rglob("*.py"):
        rel = path.relative_to(_PKG_ROOT)
        if rel.parts[0] == "_vendor" or rel == Path("api.py"):
            continue
        offenders.extend(_api_imports(path, path.read_text()))
    assert offenders == []


def test_api_guard_catches_absolute_relative_and_dynamic_imports():
    probe = _PKG_ROOT / "pipeline" / "chat.py"

    assert _api_imports(probe, "from llm_rosetta.api import x")
    assert _api_imports(probe, "from llm_rosetta import api")
    assert _api_imports(probe, "from .. import api as _api")
    assert _api_imports(probe, "from ..api import x")
    assert _api_imports(probe, "import llm_rosetta.api")
    assert _api_imports(
        probe, 'import importlib\nimportlib.import_module("llm_rosetta.api")'
    )
    assert _api_imports(probe, "from llm_rosetta import apis") == []
    assert _api_imports(probe, "from llm_rosetta.pipeline import x") == []
