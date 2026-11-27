"""The gateway reaches ``llm_rosetta`` only through a named surface (#1630).

The gateway used to import core through whatever module path it found, so a
core refactor could break it and only re-exports kept it safe.  The surface is
two prefixes (#1657).  Everything under ``packages/llm-rosetta-gateway/src``
(``admin/`` included, ``_vendor/`` excluded) may import ``llm_rosetta`` only
through them:

* ``llm_rosetta.api`` (the flat host surface);
* ``llm_rosetta.contracts`` and its submodules (the extension contracts), with
  one exception: ``llm_rosetta.contracts.usage`` is a package whose
  submodules are internal, so it is reached as a package only.

Where an allowed module defines ``__all__``, an imported name must be in it.

The rule is strict: any import of core from the gateway outside the two
prefixes fails, and so does any private name (leading underscore, or a private
module), wherever it is imported from.  There is no allowlist for the gateway
source; a missing capability becomes a name on ``llm_rosetta.api`` (additive,
with a test), never an exception here.  The gateway asserts the core's
``api.SURFACE_VERSION`` at startup (``llm_rosetta_gateway.surface``).

``tests/gateway/**`` follows D5 of #1630: a non-private internal module is
tolerated, a private name is not, except the entries on
``_TESTS_PRIVATE_ALLOWLIST``, which can only shrink.

Print the counts with ``python tests/gateway/test_core_surface_boundary.py``;
``--report-tests`` does the same for ``tests/gateway/**``.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from _import_scan import (  # noqa: E402
    CONTRACT_PACKAGE_ONLY,
    DYNAMIC,
    DYNAMIC_NONLITERAL,
    FROM,
    ImportSite,
    import_sites,
    package_parts,
)

_ROOT = Path(__file__).parents[2]
_CORE = _ROOT / "packages/llm-rosetta/src/llm_rosetta"
_GATEWAY = _ROOT / "packages/llm-rosetta-gateway/src/llm_rosetta_gateway"
_TESTS = _ROOT / "tests/gateway"

_CORE_ROOT = "llm_rosetta"
_NON_LITERAL = "<non-literal dynamic import>"

#: The two surface prefixes.  A module equal to one of them, or under one of
#: them, is allowed, except a submodule of a package-only path.
_SURFACE = (
    "llm_rosetta.api",
    "llm_rosetta.contracts",
)
#: Packages inside the surface whose submodules are not part of it (shared
#: with the shim-sidecar boundary test).
_PACKAGE_ONLY = CONTRACT_PACKAGE_ONLY


@dataclass(frozen=True)
class Edge:
    """One imported (module, name) of one import site."""

    file: str
    lineno: int
    module: str
    name: str
    function_level: bool
    type_checking: bool
    kind: str

    @property
    def private(self) -> bool:
        """A private name, or a name inside a private module."""
        if self.name.startswith("_") and not self.name.startswith("__"):
            return True
        return any(
            part.startswith("_") and part != "_vendor"
            for part in self.module.split(".")[1:]
        )


def _is_core(module: str) -> bool:
    return module == _CORE_ROOT or module.startswith(_CORE_ROOT + ".")


def _is_core_submodule(module: str, name: str) -> bool:
    base = _CORE.joinpath(*module.split(".")[1:], name)
    return base.is_dir() or base.with_suffix(".py").is_file()


def _is_allowed(module: str) -> bool:
    if any(module.startswith(path + ".") for path in _PACKAGE_ONLY):
        return False
    return any(module == path or module.startswith(path + ".") for path in _SURFACE)


def _edges_of(file: str, site: ImportSite) -> Iterable[Edge]:
    def edge(module: str, name: str) -> Edge:
        return Edge(
            file,
            site.lineno,
            module,
            name,
            site.function_level,
            site.type_checking,
            site.kind,
        )

    if site.kind == DYNAMIC_NONLITERAL:
        yield edge(_NON_LITERAL, "")
    elif site.kind == FROM:
        if not _is_core(site.module):
            return
        for name in site.names:
            if name != "*" and _is_core_submodule(site.module, name):
                yield edge(f"{site.module}.{name}", "")
            else:
                yield edge(site.module, name)
    else:
        for module in site.targets():
            if _is_core(module):
                yield edge(module, "")


def core_edges(file: str, source: str, package: tuple[str, ...]) -> list[Edge]:
    """Return every core import of *source* as (module, name) edges.

    Args:
        file: Path relative to the scanned tree (the allowlist key).
        source: Python source text.
        package: The file's package as dotted parts.
    """
    return [
        found
        for site in import_sites(source, package)
        for found in _edges_of(file, site)
    ]


def _scan_tree(root: Path, root_package: str) -> list[Edge]:
    edges: list[Edge] = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if "_vendor" in rel.parts:
            continue
        edges += core_edges(
            rel.as_posix(),
            path.read_text(),
            package_parts(path, root, root_package),
        )
    return edges


def gateway_edges() -> list[Edge]:
    """Scan the gateway package (``_vendor/`` excluded)."""
    return _scan_tree(_GATEWAY, "llm_rosetta_gateway")


def declared_all(module: str) -> frozenset[str] | None:
    """Return a module's ``__all__``, or None when it has none or does not exist."""
    if importlib.util.find_spec(module) is None:
        return None
    declared = getattr(importlib.import_module(module), "__all__", None)
    return None if declared is None else frozenset(declared)


def outside_surface(edges: Iterable[Edge]) -> set[tuple[str, str]]:
    """Return the ``(file, module)`` pairs outside the surface."""
    return {
        (e.file, e.module) for e in edges if not e.private and not _is_allowed(e.module)
    }


def private_offenders(edges: Iterable[Edge]) -> set[tuple[str, str, str]]:
    """Return the ``(file, module, name)`` private reaches."""
    return {(e.file, e.module, e.name) for e in edges if e.private}


def undeclared_names(edges: Iterable[Edge]) -> list[str]:
    """Return imports of a surface name that the module's ``__all__`` omits."""
    found: list[str] = []
    for e in edges:
        if e.private or not e.name or e.name == "*" or not _is_allowed(e.module):
            continue
        declared = declared_all(e.module)
        if declared is not None and e.name not in declared:
            found.append(f"{e.file}:{e.lineno} {e.module}.{e.name}")
    return found


def _lines(edges: list[Edge], keep) -> list[str]:
    """``file:line module.name`` of every edge *keep* accepts, sorted."""
    return sorted(
        f"{e.file}:{e.lineno} {e.module}{'.' + e.name if e.name else ''}"
        for e in edges
        if keep(e)
    )


def test_a_core_import_is_on_the_surface():
    edges = list(gateway_edges())
    outside = outside_surface(edges)
    assert _lines(edges, lambda e: (e.file, e.module) in outside) == [], (
        "the gateway imports core outside the named surface "
        f"({', '.join(_SURFACE)}); add an additive name to llm_rosetta.api "
        "instead"
    )


def test_the_gateway_reaches_no_private_core_name():
    edges = list(gateway_edges())
    assert _lines(edges, lambda e: e.private) == []


def test_surface_imports_use_declared_names():
    assert undeclared_names(gateway_edges()) == []


def test_the_scan_sees_every_import_form():
    gw = ("llm_rosetta_gateway", "app")

    def modules(source: str) -> set[tuple[str, str]]:
        return {(e.module, e.name) for e in core_edges("app/x.py", source, gw)}

    assert modules("from llm_rosetta.formats import FORMAT_SPECS") == {
        ("llm_rosetta.formats", "FORMAT_SPECS")
    }
    # A submodule imported by name is the submodule, not an attribute.
    assert modules("from llm_rosetta.authoring import scaffold") == {
        ("llm_rosetta.authoring.scaffold", "")
    }
    assert modules("import llm_rosetta.authoring.scaffold as s") == {
        ("llm_rosetta.authoring.scaffold", "")
    }
    assert modules("from llm_rosetta import __version__") == {
        ("llm_rosetta", "__version__")
    }
    assert modules(
        "def f():\n    from llm_rosetta.auto_detect import detect_provider"
    ) == {("llm_rosetta.auto_detect", "detect_provider")}
    assert modules(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n    from llm_rosetta.contracts.routing import ResolvedRoute"
    ) == {("llm_rosetta.contracts.routing", "ResolvedRoute")}
    assert modules('import importlib\nimportlib.import_module("llm_rosetta.x")') == {
        ("llm_rosetta.x", "")
    }
    assert modules('__import__("llm_rosetta.x")') == {("llm_rosetta.x", "")}
    # The gateway's own relative imports are not core imports.
    assert modules("from ..proxy import engine\nfrom . import x") == set()
    assert modules("import json\nfrom llm_rosetta_gateway import x") == set()


def test_a_non_literal_dynamic_import_is_an_offender():
    edges = core_edges(
        "app/x.py", "import importlib\nimportlib.import_module(name)", ()
    )
    assert outside_surface(edges) == {("app/x.py", _NON_LITERAL)}


def test_the_rule_is_strict_for_the_gateway_source():
    def edges(source: str) -> list[Edge]:
        return core_edges("x.py", source, ("llm_rosetta_gateway",))

    # A public internal module is outside the surface, with no list to hide on.
    assert outside_surface(edges("from llm_rosetta.shims import get_shim")) == {
        ("x.py", "llm_rosetta.shims")
    }
    assert outside_surface(edges("from llm_rosetta.api import get_shim")) == set()
    # A private name is a violation even on a surface module.
    assert private_offenders(edges("from llm_rosetta.api import _x")) == {
        ("x.py", "llm_rosetta.api", "_x")
    }


def test_the_surface_rule():
    for module in _SURFACE:
        assert _is_allowed(module)
    for module in ("auth", "quota", "routing", "shim", "transform", "usage"):
        assert _is_allowed(f"llm_rosetta.contracts.{module}")
    assert not _is_allowed("llm_rosetta.contracts.usage.report")
    assert not _is_allowed("llm_rosetta.contractsx")
    # The 0.13.0 forwarders are compat for outside users, not for the gateway.
    assert not _is_allowed("llm_rosetta.routing")
    assert not _is_allowed("llm_rosetta.capabilities")
    assert not _is_allowed("llm_rosetta.observability")
    assert not _is_allowed("llm_rosetta.shims")
    assert not _is_allowed("llm_rosetta")
    assert not _is_allowed("llm_rosetta.apis")


def test_private_names_and_modules_are_private():
    def edges(source: str) -> list[Edge]:
        return core_edges("x.py", source, ("llm_rosetta_gateway",))

    assert edges("from llm_rosetta._compat import sync_compat")[0].private
    assert edges("from llm_rosetta.shims._loader import _leaf_connection")[0].private
    assert not edges("from llm_rosetta import __version__")[0].private
    assert not edges("from llm_rosetta._vendor import yaml")[0].private


def test_an_undeclared_name_on_the_surface_is_reported():
    edges = core_edges(
        "x.py",
        "from llm_rosetta.contracts.routing import ResolvedRoute, NoSuchName",
        ("llm_rosetta_gateway",),
    )
    assert [line.split(" ", 1)[1] for line in undeclared_names(edges)] == [
        "llm_rosetta.contracts.routing.NoSuchName"
    ]
    # The flat surface is checked against its own ``__all__`` too.
    api = core_edges("x.py", "from llm_rosetta.api import NoSuchName", ("p",))
    assert [line.split(" ", 1)[1] for line in undeclared_names(api)] == [
        "llm_rosetta.api.NoSuchName"
    ]


# --- Tests allowlist: (file relative to tests/gateway, module, name). ---
# D5 of #1630: a gateway test may import non-private core modules, but not a
# private name or a private module.  Shrink only; never add an entry.  Lower
# _TESTS_PRIVATE_CEILING with every deletion.
_TESTS_PRIVATE_CEILING = 1
_TESTS_PRIVATE_ALLOWLIST: frozenset[tuple[str, str, str]] = frozenset(
    {
        # Tests a core internal: the fold's echo list must equal the catalog's.
        ("test_stream_bridge.py", "llm_rosetta.ir.primitives.fold", "_ECHO_FIELDS"),
    }
)


def _tests_edges() -> list[Edge]:
    """Scan ``tests/gateway`` (this file's own fixtures included)."""
    return _scan_tree(_TESTS, "tests")


def test_gateway_tests_reach_no_private_core_name_off_the_list():
    reaches = private_offenders(_tests_edges())
    assert sorted(reaches - _TESTS_PRIVATE_ALLOWLIST) == [], (
        "a gateway test imports a private core name; use llm_rosetta.api "
        "(snapshot_registry / restore_registry, load_providers, ...)"
    )
    assert sorted(_TESTS_PRIVATE_ALLOWLIST - reaches) == [], "stale: delete the entry"
    assert len(_TESTS_PRIVATE_ALLOWLIST) <= _TESTS_PRIVATE_CEILING


def _report(edges: list[Edge], label: str) -> str:
    files = {e.file for e in edges}
    pairs = {(e.file, e.module) for e in edges}
    symbols = {(e.module, e.name) for e in edges}
    modules = {e.module for e in edges}
    off = [e for e in edges if not _is_allowed(e.module) or e.private]
    rows = {
        "files that import core": len(files),
        "import sites": len(edges),
        "distinct (file, module) pairs": len(pairs),
        "distinct (module, symbol)": len(symbols),
        "distinct core modules": len(modules),
        "function-level sites": sum(e.function_level for e in edges),
        "TYPE_CHECKING sites": sum(e.type_checking for e in edges),
        "private sites": sum(e.private for e in edges),
        "private symbols": len({(e.module, e.name) for e in edges if e.private}),
        "dynamic sites (literal)": sum(e.kind == DYNAMIC for e in edges),
        "dynamic sites (non-literal)": sum(e.module == _NON_LITERAL for e in edges),
        "sites outside the surface": len(off),
        "(file, module) pairs outside the surface": len(
            {(e.file, e.module) for e in off}
        ),
    }
    out = [f"# {label}"] + [f"# {value:>5}  {key}" for key, value in rows.items()]
    by_module = Counter(e.module for e in off)
    out.append("# outside the surface, by module (sites):")
    out += [f"#   {n:>3} {m}" for m, n in by_module.most_common()]
    return "\n".join(out)


def main(argv: list[str]) -> None:
    """Print the gateway counts, or with ``--report-tests`` the test-tree counts."""
    if "--report-tests" in argv:
        print(_report(_scan_tree(_TESTS, "tests"), "tests/gateway/** (report only)"))
        return
    edges = gateway_edges()
    print(_report(edges, "gateway src (_vendor excluded)"))
    bad = undeclared_names(edges)
    if bad:
        print("\n# names missing from the surface module's __all__:")
        print("\n".join(f"#   {line}" for line in bad))


if __name__ == "__main__":
    main(sys.argv[1:])
