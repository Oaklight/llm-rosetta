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
  ``LAZY_UPWARD`` names it, by file, target unit and the enclosing function's
  qualified name, **with a count**.  The entries are the ones the tree had
  when the layers were drawn; the set may only shrink, so a new key fails, and
  so does a key that no longer occurs or one that now occurs more often than
  the freeze says.  A second such import in a function already named is a
  count change, and a same-named method in another class is a different key,
  so neither passes.

* every unit-to-unit import, downward and same-layer ones included, is a row
  of ``GATEWAY_EDGES``: a new one fails with its file and line, and a row whose
  import has gone fails too, so the table is the whole package graph;

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
from collections import Counter
from pathlib import Path

from _import_scan import DYNAMIC_NONLITERAL, FROM, ImportSite, import_sites

_GATEWAY = "llm_rosetta_gateway"
_ROOT = Path(__file__).parents[2] / "packages/llm-rosetta-gateway/src" / _GATEWAY

#: Bottom to top.  A package or root module belongs to exactly one layer.
LAYERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "foundation",
        (
            "model_types",
            "observability",
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
    ("entry", ("__init__", "cli")),
)

#: The modules the package root keeps, and why each is not in a package.
ROOT_MODULES: dict[str, str] = {
    "__init__": "the package: version and lazy public names",
    "admin_api": "the surface admin depends on; its own boundary",
    "surface": "the core surface level this gateway needs; its boundary with core",
    "updates": "reads the changelog the release workflow writes beside the "
    "package root, so a move changes that lookup",
}

#: Function-local imports that reach a higher layer, as ``(file, layer unit,
#: qualified function)`` with *file* relative to the gateway package, mapped to
#: how many such imports the function holds.  The function is the one the
#: import sits in (``ClassName.method`` for a method), so a second one in
#: another function, or a same-named method in another class, is a different
#: entry, and a second import in the same function changes its count.  Shrink
#: only.
LAZY_UPWARD: dict[tuple[str, str, str], int] = {
    ("jobs/handlers/data.py", "ops", "data_handlers"): 1,
}


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


def _edges(source: str, path: Path) -> list[tuple[str, str, int]]:
    """``(target unit, enclosing function, line)`` for every counted import.

    The function is empty for a module-level import.
    """
    edges = []
    for site in import_sites(source, _package_parts(path)):
        if site.type_checking or site.kind == DYNAMIC_NONLITERAL:
            continue
        for module in _site_modules(site):
            unit = _unit_of_module(module)
            if unit is not None:
                edges.append((unit, site.function, site.lineno))
    return edges


def _upward(path: Path, source: str) -> list[tuple[str, str, str, int]]:
    """Imports in *path* that reach a higher layer than its own.

    Each entry is ``(file, target unit, enclosing function, line)``; the
    function is empty for a module-level import.
    """
    own = _unit_of_file(path)
    rank = _LAYER_OF.get(own)
    if rank is None:
        return []
    found = []
    for unit, function, line in _edges(source, path):
        if unit != own and _LAYER_OF.get(unit, -1) > rank:
            found.append((_rel(path), unit, function, line))
    return found


def _lazy_upward_sites() -> Counter[tuple[str, str, str]]:
    """Count each ``(file, target unit, qualified function)`` function-local import."""
    return Counter(
        (rel, unit, function)
        for path in _files()
        for rel, unit, function, _line in _upward(path, path.read_text())
        if function
    )


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
        for rel, unit, function, line in _upward(path, path.read_text())
        if not function
    ]
    assert offenders == []


def test_the_module_level_package_graph_has_no_cycle():
    graph: dict[str, set[str]] = {}
    for path in _files():
        own = _unit_of_file(path)
        for unit, function, _line in _edges(path.read_text(), path):
            if not function and unit != own:
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
    current = _lazy_upward_sites()
    added = sorted(set(current) - set(LAZY_UPWARD))
    assert added == [], (
        "a new function-local import into a higher layer; move the code down "
        "or the dependency up instead of reaching across"
    )
    grown = sorted(key for key in current if current[key] > LAZY_UPWARD.get(key, 0))
    assert grown == [], (
        "another function-local import into a higher layer in a function "
        "LAZY_UPWARD already names; raise its count only if the move is agreed"
    )
    gone = sorted(set(LAZY_UPWARD) - set(current))
    assert gone == [], "remove these from LAZY_UPWARD"
    shrunk = sorted(key for key in current if current[key] < LAZY_UPWARD[key])
    assert shrunk == [], "lower these counts in LAZY_UPWARD"


#: Every import from one gateway unit into another, module level and
#: function-local alike (``TYPE_CHECKING`` and dynamic non-literal imports are
#: not counted, as above).  ``LAYERS`` forbids the upward ones; this table is
#: the rest: a *downward* or same-layer import is allowed, but a new one is a
#: new dependency a reviewer should see, so it fails until it is listed here, and
#: an entry whose import has gone fails until it is deleted.
GATEWAY_EDGES: dict[str, tuple[str, ...]] = {
    "admin": (
        "admin_api",
        "config",
        "jobs",
        "keystore",
        "maintenance",
        "middleware",
        "observability",
        "ops",
        "quota",
        "routing",
        "shims",
        "transport",
        "updates",
    ),
    "admin_api": (),
    "app": (
        "admin",
        "admin_api",
        "config",
        "jobs",
        "keystore",
        "maintenance",
        "middleware",
        "model_types",
        "observability",
        "ops",
        "pipelines",
        "proxy",
        "quota",
        "routing",
        "shims",
        "surface",
        "transport",
    ),
    "cli": (
        "app",
        "config",
        "observability",
        "ops",
        "shims",
    ),
    "config": (
        "middleware",
        "model_types",
        "observability",
        "quota",
        "routing",
        "shims",
        "transport",
    ),
    "jobs": (
        "observability",
        "ops",
    ),
    "keystore": ("observability",),
    "maintenance": (
        "keystore",
        "observability",
        "quota",
        "transport",
    ),
    "middleware": (
        "keystore",
        "observability",
    ),
    "model_types": (),
    "ops": (
        "config",
        "middleware",
        "model_types",
        "observability",
        "routing",
        "shims",
    ),
    "pipelines": (
        "config",
        "middleware",
        "model_types",
        "observability",
        "proxy",
        "quota",
        "transport",
    ),
    "proxy": (
        "maintenance",
        "middleware",
        "observability",
        "ops",
        "quota",
        "routing",
        "transport",
    ),
    "quota": (
        "observability",
        "transport",
    ),
    "routing": (
        "quota",
        "transport",
    ),
    "transport": (),
}


def _gateway_edges() -> dict[tuple[str, str], str]:
    """Every unit-to-unit import, with one ``file:line`` that shows it."""
    found: dict[tuple[str, str], str] = {}
    for path in _files():
        own = _unit_of_file(path)
        for unit, _function, line in _edges(path.read_text(), path):
            if unit != own:
                found.setdefault((own, unit), f"{_rel(path)}:{line}")
    return found


def test_no_gateway_package_gains_an_import_the_table_does_not_list():
    allowed = {(src, dst) for src, dsts in GATEWAY_EDGES.items() for dst in dsts}
    new = [
        f"{src} -> {dst} ({where})"
        for (src, dst), where in sorted(_gateway_edges().items())
        if (src, dst) not in allowed
    ]
    assert new == [], (
        "a new package-level import in the gateway; if it is meant, add it to "
        "GATEWAY_EDGES in tests/gateway/test_layers.py so a reviewer sees it"
    )


def test_the_gateway_edge_table_lists_no_import_that_has_gone():
    actual = set(_gateway_edges())
    gone = [
        f"{src} -> {dst}"
        for src, dsts in GATEWAY_EDGES.items()
        for dst in dsts
        if (src, dst) not in actual
    ]
    assert gone == [], "delete these rows from GATEWAY_EDGES: the import is gone"


def test_the_guard_sees_eager_lazy_relative_and_dynamic_imports():
    probe = _ROOT / "observability" / "sanitize.py"

    def up(source: str) -> list[tuple[str, str]]:
        return [(unit, function) for _r, unit, function, _l in _upward(probe, source)]

    assert up("from ..app import create_app") == [("app", "")]
    assert up("from llm_rosetta_gateway.proxy import engine") == [("proxy", "")]
    assert up("from .. import config") == [("config", "")]
    assert up("import llm_rosetta_gateway.cli") == [("cli", "")]
    assert up('import importlib\nimportlib.import_module("llm_rosetta_gateway.ops")')
    assert up("def f():\n    from ..routing import x\n") == [("routing", "f")]
    # The function is part of the site: two functions are two entries.
    assert up(
        "def f():\n    from ..routing import x\ndef g():\n    from ..ops import y\n"
    ) == [
        ("routing", "f"),
        ("ops", "g"),
    ]
    assert up("def f():\n    def g():\n        from ..ops import y\n") == [("ops", "g")]
    # Not counted: same unit, lower layer, _vendor, __version__, TYPE_CHECKING.
    assert up("from . import metrics") == []
    assert up("from ..observability import sanitize") == []
    assert up("from .._vendor import yaml") == []
    assert up("from .. import __version__") == []
    assert (
        up(
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from ..app import x"
        )
        == []
    )
    # The package root's entry-point names are the entry layer.
    assert up("from llm_rosetta_gateway import create_app") == [("__init__", "")]
    assert up("from .. import main") == [("__init__", "")]


if __name__ == "__main__":
    print("LAZY_UPWARD: dict[tuple[str, str, str], int] = {")
    for (_file, _unit, _function), _count in sorted(_lazy_upward_sites().items()):
        print(f'        ("{_file}", "{_unit}", "{_function}"): {_count},')
    print("    }\n)")
