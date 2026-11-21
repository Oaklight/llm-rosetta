"""The gateway's packages form layers, and imports only go down (#1672 G5).

The layers are read off the import graph as it stands, bottom to top:

* ``foundation`` — telemetry and logging (``observability``), the upstream
  transport, the shim sources, the model-type registry, and the leaf modules
  the root keeps.  None of them imports another gateway layer at module level.
* ``state`` — what the gateway remembers and decides from: quota, routing,
  the maintenance steps, the API key store and the job queue.
* ``policy`` — config resolution (with the provider registry) and the request
  middleware.
* ``operations`` — the audited ops layer.
* ``engine`` — the proxy (with the request lifecycle) and the typed pipelines.
* ``serving`` — the HTTP app, the admin panel and the admin's boundary
  (``admin_api``).
* ``entry`` — the CLI and the package's entry points.

Guards:

* every package and root module of the gateway sits in exactly one layer, so
  a new one has to be placed before it lands;
* the package root holds only the modules on ``ROOT_MODULES``, each with its
  reason: everything else lives in a package;
* a module-level import never reaches a higher layer, and the module-level
  package graph has no cycle;
* a function-local import reaches a higher layer only where
  ``LAZY_UPWARD`` names it.  The entries are the ones the tree had when the
  layers were drawn; the list may only shrink, so a new one fails, and so does
  an entry that no longer occurs.

What admin may import is guarded separately, file by file, by
``tests/gateway/test_admin_import_boundary.py``; this test only places admin
in the serving layer.

Imports under ``if TYPE_CHECKING:`` are not counted: they never run.  The
package root itself may be imported for ``__version__``; its other names are
entry points and count as an import of the entry layer.  ``_vendor`` is not a
layer: any module may use it.

Print the current graph's function-local upward imports with
``PYTHONPATH=tests python tests/gateway/test_layers.py``.
"""

from __future__ import annotations

import functools
from pathlib import Path

from _import_scan import DYNAMIC_NONLITERAL, FROM, ImportSite, import_sites

_GATEWAY = "llm_rosetta_gateway"
_ROOT = Path(__file__).parents[2] / "packages/llm-rosetta-gateway/src" / _GATEWAY

#: Bottom to top.  A package or root module belongs to exactly one layer.
LAYERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "foundation",
        (
            "_compat",
            "model_types",
            "observability",
            "shim_sources",
            "shims",
            "surface",
            "transport",
            "updates",
        ),
    ),
    ("state", ("jobs", "keystore", "maintenance", "quota", "routing")),
    ("policy", ("config", "middleware")),
    ("operations", ("ops",)),
    ("engine", ("pipelines", "proxy")),
    ("serving", ("admin", "admin_api", "app")),
    ("entry", ("__init__", "__main__", "cli")),
)

#: The modules the package root keeps, and why each is not in a package.
ROOT_MODULES: dict[str, str] = {
    "__init__": "the package: version and lazy public names",
    "__main__": "python -m llm_rosetta_gateway",
    "_compat": "sync/async compatibility helpers, scheduled for deletion (#1672 G7)",
    "admin_api": "the surface admin depends on; its own boundary",
    "keystore": "imported from outside the gateway (argo-proxy: KeyContext), so "
    "moving it needs a deprecated forwarder",
    "shim_sources": "moves into shims/ with the shim subsystem (#1672 G1)",
    "surface": "the core surface level this gateway needs; its boundary with core",
    "updates": "reads the changelog the release workflow writes beside the "
    "package root, so a move changes that lookup",
}

#: Function-local imports that reach a higher layer, as ``(file, layer unit)``
#: with *file* relative to the gateway package.  Shrink only.
LAZY_UPWARD: frozenset[tuple[str, str]] = frozenset(
    {
        ("jobs/handlers/data.py", "ops"),
        ("maintenance/migrate_0140.py", "config"),
        ("maintenance/migrations.py", "config"),
        ("maintenance/names.py", "config"),
        ("model_types/__init__.py", "pipelines"),
        ("transport/provider_info.py", "routing"),
    }
)

_LAYER_OF: dict[str, int] = {
    unit: rank for rank, (_name, units) in enumerate(LAYERS) for unit in units
}


def _files() -> list[Path]:
    return sorted(p for p in _ROOT.rglob("*.py") if "_vendor" not in p.parts)


def _rel(path: Path) -> str:
    return path.relative_to(_ROOT).as_posix()


def _unit_of_file(path: Path) -> str:
    parts = path.relative_to(_ROOT).parts
    return parts[0] if len(parts) > 1 else Path(parts[0]).stem


def _package_parts(path: Path) -> tuple[str, ...]:
    return (_GATEWAY, *path.relative_to(_ROOT).parent.parts)


@functools.cache
def _units() -> frozenset[str]:
    return frozenset(_unit_of_file(p) for p in _files())


def _unit_of_module(module: str) -> str | None:
    """The layer unit a gateway module belongs to; None outside the layering.

    A root name that is not a submodule (``create_app``, ``main``) is one of
    the package's lazy entry points, so it is the root ``__init__``.
    """
    parts = module.split(".")
    if parts[0] != _GATEWAY:
        return None
    if len(parts) == 1:
        return "__init__"
    if parts[1] in ("_vendor", "__version__"):
        return None
    return parts[1] if parts[1] in _units() else "__init__"


def _site_modules(site: ImportSite) -> list[str]:
    """The modules one import names.

    A ``from pkg.sub import name`` depends on ``pkg.sub``'s unit whatever
    *name* is; ``from llm_rosetta_gateway import name`` names the root's
    *name*, which is the unit when it is a submodule.
    """
    if site.kind == FROM and site.module == _GATEWAY:
        return [f"{_GATEWAY}.{n}" for n in site.names if n != "*"]
    if site.kind == FROM:
        return [site.module]
    return list(site.targets())


def _edges(source: str, path: Path) -> list[tuple[str, bool, int]]:
    """``(target unit, function-level, line)`` for every counted import."""
    edges = []
    for site in import_sites(source, _package_parts(path)):
        if site.type_checking or site.kind == DYNAMIC_NONLITERAL:
            continue
        for module in _site_modules(site):
            unit = _unit_of_module(module)
            if unit is not None:
                edges.append((unit, site.function_level, site.lineno))
    return edges


def _upward(path: Path, source: str) -> list[tuple[str, str, bool, int]]:
    """Imports in *path* that reach a higher layer than its own."""
    own = _unit_of_file(path)
    rank = _LAYER_OF.get(own)
    if rank is None:
        return []
    found = []
    for unit, lazy, line in _edges(source, path):
        if unit != own and _LAYER_OF.get(unit, -1) > rank:
            found.append((_rel(path), unit, lazy, line))
    return found


def _lazy_upward_pairs() -> set[tuple[str, str]]:
    return {
        (rel, unit)
        for path in _files()
        for rel, unit, lazy, _line in _upward(path, path.read_text())
        if lazy
    }


def test_every_package_and_root_module_has_one_layer():
    units = {_unit_of_file(p) for p in _files()}
    declared = [u for _name, us in LAYERS for u in us]
    assert len(declared) == len(set(declared)), "a unit sits in two layers"
    assert sorted(units - set(declared)) == [], "place these in a layer"
    assert sorted(set(declared) - units) == [], "remove these from LAYERS"


def test_the_root_holds_only_the_listed_modules():
    root = {p.stem for p in _ROOT.glob("*.py")}
    assert sorted(root - set(ROOT_MODULES)) == [], (
        "a new root module: put it in a package, or give it a reason in ROOT_MODULES"
    )
    assert sorted(set(ROOT_MODULES) - root) == [], "remove from ROOT_MODULES"


def test_module_level_imports_never_reach_a_higher_layer():
    offenders = [
        f"{rel}:{line} -> {unit}"
        for path in _files()
        for rel, unit, lazy, line in _upward(path, path.read_text())
        if not lazy
    ]
    assert offenders == []


def test_the_module_level_package_graph_has_no_cycle():
    graph: dict[str, set[str]] = {}
    for path in _files():
        own = _unit_of_file(path)
        for unit, lazy, _line in _edges(path.read_text(), path):
            if not lazy and unit != own:
                graph.setdefault(own, set()).add(unit)

    state: dict[str, int] = {}  # 1 = on the stack, 2 = done
    cycle: list[str] = []

    def visit(node: str, trail: list[str]) -> bool:
        state[node] = 1
        for nxt in sorted(graph.get(node, ())):
            if state.get(nxt) == 1:
                cycle.extend(trail[trail.index(nxt) :] + [node, nxt])
                return True
            if nxt not in state and visit(nxt, [*trail, nxt]):
                return True
        state[node] = 2
        return False

    for start in sorted(graph):
        if start not in state and visit(start, [start]):
            break
    assert cycle == []


def test_function_local_upward_imports_only_shrink():
    current = _lazy_upward_pairs()
    assert sorted(current - LAZY_UPWARD) == [], (
        "a new function-local import into a higher layer; move the code down "
        "or the dependency up instead of reaching across"
    )
    assert sorted(LAZY_UPWARD - current) == [], "remove these from LAZY_UPWARD"


def test_the_guard_sees_eager_lazy_relative_and_dynamic_imports():
    probe = _ROOT / "observability" / "sanitize.py"

    def up(source: str) -> list[tuple[str, bool]]:
        return [(unit, lazy) for _r, unit, lazy, _l in _upward(probe, source)]

    assert up("from ..app import create_app") == [("app", False)]
    assert up("from llm_rosetta_gateway.proxy import engine") == [("proxy", False)]
    assert up("from .. import config") == [("config", False)]
    assert up("import llm_rosetta_gateway.cli") == [("cli", False)]
    assert up('import importlib\nimportlib.import_module("llm_rosetta_gateway.ops")')
    assert up("def f():\n    from ..routing import x\n") == [("routing", True)]
    # Not counted: same unit, lower layer, _vendor, __version__, TYPE_CHECKING.
    assert up("from . import metrics") == []
    assert up("from .._compat import sync_compat") == []
    assert up("from .._vendor import yaml") == []
    assert up("from .. import __version__") == []
    assert (
        up(
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from ..app import x"
        )
        == []
    )
    # The package root's entry-point names are the entry layer.
    assert up("from llm_rosetta_gateway import create_app") == [("__init__", False)]
    assert up("from .. import main") == [("__init__", False)]


if __name__ == "__main__":
    print("LAZY_UPWARD: frozenset[tuple[str, str]] = frozenset(\n    {")
    for _file, _unit in sorted(_lazy_upward_pairs()):
        print(f'        ("{_file}", "{_unit}"),')
    print("    }\n)")
