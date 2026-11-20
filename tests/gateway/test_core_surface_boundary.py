"""The gateway reaches ``llm_rosetta`` only through a named surface (#1630).

The gateway used to import core through whatever module path it found, so a
core refactor could break it and only re-exports kept it safe.  The surface is
five module paths.  Everything under ``packages/llm-rosetta-gateway/src``
(``admin/`` included, ``_vendor/`` excluded) may import ``llm_rosetta`` only
through them:

* ``llm_rosetta.api`` (the flat host surface; an imported name must be in its
  ``__all__``);
* ``llm_rosetta.shims.auth_exchange`` and ``llm_rosetta.shims.quota_probe``
  (the contracts external shims already import);
* ``llm_rosetta.routing``;
* ``llm_rosetta.observability.usage`` as a package, never one of its submodules.

Where an allowed module defines ``__all__``, an imported name must be in it.

Every import outside the surface is a *legacy offender*, held on
``_LEGACY_ALLOWLIST`` keyed on ``(file, module)`` (no line numbers, so a rebase
does not invalidate it).  The list can only shrink, and three guards enforce
that: a new offender fails, a stale entry fails, and the size may not exceed a
ceiling nor leave the frozen copy ``_LEGACY_FROZEN`` (so an offender cannot
enter by replacing another).  When you delete entries, lower the ceiling and
delete them from the frozen copy too.

A private name (leading underscore, or a private module) has its own small
allowlist, ``_PRIVATE_ALLOWLIST``.  PR 9 of the #1630 plan replaces those
reaches and deletes the list; the four left are the manifest helpers of
``admin/routes/config/shim_manifests.py`` (see the PR for what blocks them).

Regenerate both lists with ``python tests/gateway/test_core_surface_boundary.py``.
``--report-tests`` prints the same counts for ``tests/gateway/**``, which this
test does not scan yet.
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

#: The five surface paths.  A module equal to one of them, or under one of
#: them, is allowed, except under a package-only path.
_SURFACE = (
    "llm_rosetta.api",
    "llm_rosetta.shims.auth_exchange",
    "llm_rosetta.shims.quota_probe",
    "llm_rosetta.routing",
    "llm_rosetta.observability.usage",
)
#: Surface paths whose submodules are not part of the surface.
_PACKAGE_ONLY = frozenset({"llm_rosetta.observability.usage"})

# --- Legacy allowlist: (file relative to the gateway package, core module). ---
# Shrink only.  Lower _LEGACY_CEILING and edit _LEGACY_FROZEN with it.
# The shims split (#1633) added one pair, `shims/_scan.py` ->
# `llm_rosetta.shims.providers`, on a dependency `shims/sources.py` already
# had; #1639 removed `shims/catalog.py` -> `llm_rosetta.provider_names`; #1637
# removed the three `llm_rosetta._vendor.yaml` imports (the gateway vendors its
# own yaml).
_LEGACY_CEILING = 64
_LEGACY_ALLOWLIST: frozenset[tuple[str, str]] = frozenset(
    {
        ("admin/routes/_shared.py", "llm_rosetta.shims"),
        ("admin/routes/config/bulk.py", "llm_rosetta.formats"),
        ("admin/routes/config/models.py", "llm_rosetta.formats"),
        ("admin/routes/config/payload.py", "llm_rosetta"),
        ("admin/routes/config/payload.py", "llm_rosetta.formats"),
        ("admin/routes/config/payload.py", "llm_rosetta.shims"),
        ("admin/routes/config/payload.py", "llm_rosetta.shims.provider_shim"),
        ("admin/routes/config/providers.py", "llm_rosetta.shims"),
        ("admin/routes/config/shim_files.py", "llm_rosetta.shims"),
        ("admin/routes/config/shim_files.py", "llm_rosetta.shims.providers"),
        ("admin/routes/config/shim_manifests.py", "llm_rosetta.shims.provider_shim"),
        ("admin/routes/config/shim_manifests.py", "llm_rosetta.shims.providers"),
        ("admin/routes/config/upstream.py", "llm_rosetta.shims"),
        ("admin/routes/config/upstream.py", "llm_rosetta.shims.providers"),
        ("admin/routes/examples.py", "llm_rosetta"),
        ("admin/routes/examples.py", "llm_rosetta.converters.decisions"),
        ("admin/routes/examples.py", "llm_rosetta.converters.embeddings"),
        ("admin/routes/examples.py", "llm_rosetta.converters.rerank"),
        ("admin/routes/examples.py", "llm_rosetta.formats.catalog"),
        ("admin/routes/examples.py", "llm_rosetta.formats.spec"),
        ("admin/routes/observability.py", "llm_rosetta.provider_names"),
        ("admin/routes/observability.py", "llm_rosetta.shims"),
        ("app/routes_health.py", "llm_rosetta"),
        ("app/routes_llm.py", "llm_rosetta.auto_detect"),
        ("app/routes_llm.py", "llm_rosetta.converters.registry"),
        ("app/routes_llm.py", "llm_rosetta.provider_names"),
        ("app/routes_models.py", "llm_rosetta.formats"),
        ("banner.py", "llm_rosetta"),
        ("cli.py", "llm_rosetta"),
        ("cli.py", "llm_rosetta.shims"),
        ("cli.py", "llm_rosetta.shims.scaffold"),
        ("config/__init__.py", "llm_rosetta.auto_detect"),
        ("config/core.py", "llm_rosetta.formats"),
        ("config/parse.py", "llm_rosetta.formats"),
        ("config/parse.py", "llm_rosetta.shims"),
        ("config/resolve.py", "llm_rosetta.auto_detect"),
        ("config/resolve.py", "llm_rosetta.model_vocabulary"),
        ("config/resolve.py", "llm_rosetta.provider_names"),
        ("config/resolve.py", "llm_rosetta.shims"),
        ("config/typed.py", "llm_rosetta.formats"),
        ("config/typed.py", "llm_rosetta.provider_names"),
        ("config/typed.py", "llm_rosetta.shims"),
        ("fidelity_check.py", "llm_rosetta.fidelity"),
        ("lifecycle.py", "llm_rosetta.auto_detect"),
        ("maintenance/migrations.py", "llm_rosetta.formats"),
        ("maintenance/names.py", "llm_rosetta.provider_names"),
        ("observability/persistence/__init__.py", "llm_rosetta.formats"),
        ("observability/persistence/request_log.py", "llm_rosetta.formats"),
        ("observability/persistence/usage.py", "llm_rosetta.formats"),
        ("observability/profiling.py", "llm_rosetta.profiling"),
        ("observability/usage/cli.py", "llm_rosetta"),
        ("ops/provider_validation.py", "llm_rosetta.shims"),
        ("providers.py", "llm_rosetta.formats"),
        ("providers.py", "llm_rosetta.provider_names"),
        ("providers.py", "llm_rosetta.shims"),
        ("quota/service.py", "llm_rosetta.shims"),
        ("shims/_scan.py", "llm_rosetta.shims.providers"),
        ("shims/manifest.py", "llm_rosetta"),
        ("shims/removals.py", "llm_rosetta.provider_names"),
        ("shims/sources.py", "llm_rosetta.shims.provider_shim"),
        ("shims/sources.py", "llm_rosetta.shims.providers"),
        ("source_formats.py", "llm_rosetta.provider_names"),
        ("transport/sse_format.py", "llm_rosetta.formats"),
        ("updates.py", "llm_rosetta"),
    }
)
_LEGACY_FROZEN: frozenset[tuple[str, str]] = frozenset(
    {
        ("admin/routes/_shared.py", "llm_rosetta.shims"),
        ("admin/routes/config/bulk.py", "llm_rosetta.formats"),
        ("admin/routes/config/models.py", "llm_rosetta.formats"),
        ("admin/routes/config/payload.py", "llm_rosetta"),
        ("admin/routes/config/payload.py", "llm_rosetta.formats"),
        ("admin/routes/config/payload.py", "llm_rosetta.shims"),
        ("admin/routes/config/payload.py", "llm_rosetta.shims.provider_shim"),
        ("admin/routes/config/providers.py", "llm_rosetta.shims"),
        ("admin/routes/config/shim_files.py", "llm_rosetta.shims"),
        ("admin/routes/config/shim_files.py", "llm_rosetta.shims.providers"),
        ("admin/routes/config/shim_manifests.py", "llm_rosetta.shims.provider_shim"),
        ("admin/routes/config/shim_manifests.py", "llm_rosetta.shims.providers"),
        ("admin/routes/config/upstream.py", "llm_rosetta.shims"),
        ("admin/routes/config/upstream.py", "llm_rosetta.shims.providers"),
        ("admin/routes/examples.py", "llm_rosetta"),
        ("admin/routes/examples.py", "llm_rosetta.converters.decisions"),
        ("admin/routes/examples.py", "llm_rosetta.converters.embeddings"),
        ("admin/routes/examples.py", "llm_rosetta.converters.rerank"),
        ("admin/routes/examples.py", "llm_rosetta.formats.catalog"),
        ("admin/routes/examples.py", "llm_rosetta.formats.spec"),
        ("admin/routes/observability.py", "llm_rosetta.provider_names"),
        ("admin/routes/observability.py", "llm_rosetta.shims"),
        ("app/routes_health.py", "llm_rosetta"),
        ("app/routes_llm.py", "llm_rosetta.auto_detect"),
        ("app/routes_llm.py", "llm_rosetta.converters.registry"),
        ("app/routes_llm.py", "llm_rosetta.provider_names"),
        ("app/routes_models.py", "llm_rosetta.formats"),
        ("banner.py", "llm_rosetta"),
        ("cli.py", "llm_rosetta"),
        ("cli.py", "llm_rosetta.shims"),
        ("cli.py", "llm_rosetta.shims.scaffold"),
        ("config/__init__.py", "llm_rosetta.auto_detect"),
        ("config/core.py", "llm_rosetta.formats"),
        ("config/parse.py", "llm_rosetta.formats"),
        ("config/parse.py", "llm_rosetta.shims"),
        ("config/resolve.py", "llm_rosetta.auto_detect"),
        ("config/resolve.py", "llm_rosetta.model_vocabulary"),
        ("config/resolve.py", "llm_rosetta.provider_names"),
        ("config/resolve.py", "llm_rosetta.shims"),
        ("config/typed.py", "llm_rosetta.formats"),
        ("config/typed.py", "llm_rosetta.provider_names"),
        ("config/typed.py", "llm_rosetta.shims"),
        ("fidelity_check.py", "llm_rosetta.fidelity"),
        ("lifecycle.py", "llm_rosetta.auto_detect"),
        ("maintenance/migrations.py", "llm_rosetta.formats"),
        ("maintenance/names.py", "llm_rosetta.provider_names"),
        ("observability/persistence/__init__.py", "llm_rosetta.formats"),
        ("observability/persistence/request_log.py", "llm_rosetta.formats"),
        ("observability/persistence/usage.py", "llm_rosetta.formats"),
        ("observability/profiling.py", "llm_rosetta.profiling"),
        ("observability/usage/cli.py", "llm_rosetta"),
        ("ops/provider_validation.py", "llm_rosetta.shims"),
        ("providers.py", "llm_rosetta.formats"),
        ("providers.py", "llm_rosetta.provider_names"),
        ("providers.py", "llm_rosetta.shims"),
        ("quota/service.py", "llm_rosetta.shims"),
        ("shims/_scan.py", "llm_rosetta.shims.providers"),
        ("shims/manifest.py", "llm_rosetta"),
        ("shims/removals.py", "llm_rosetta.provider_names"),
        ("shims/sources.py", "llm_rosetta.shims.provider_shim"),
        ("shims/sources.py", "llm_rosetta.shims.providers"),
        ("source_formats.py", "llm_rosetta.provider_names"),
        ("transport/sse_format.py", "llm_rosetta.formats"),
        ("updates.py", "llm_rosetta"),
    }
)

# --- Private allowlist: (file, module, name).  PR 9 of #1630 deletes it. ---
_PRIVATE_CEILING = 4
_PRIVATE_ALLOWLIST: frozenset[tuple[str, str, str]] = frozenset(
    {
        (
            "admin/routes/config/shim_manifests.py",
            "llm_rosetta.shims.providers",
            "_CONNECTION_KEYS",
        ),
        (
            "admin/routes/config/shim_manifests.py",
            "llm_rosetta.shims.providers",
            "_IDENTITY_KEYS",
        ),
        (
            "admin/routes/config/shim_manifests.py",
            "llm_rosetta.shims.providers",
            "_leaf_connection",
        ),
        (
            "admin/routes/config/shim_manifests.py",
            "llm_rosetta.shims.providers",
            "_merged_connection",
        ),
    }
)


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
    for path in _SURFACE:
        if module == path:
            return True
        if module.startswith(path + ".") and path not in _PACKAGE_ONLY:
            return True
    return False


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


def legacy_offenders(edges: Iterable[Edge]) -> set[tuple[str, str]]:
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


def test_a_core_import_is_on_the_surface_or_on_the_legacy_list():
    offenders = legacy_offenders(gateway_edges())
    new = sorted(offenders - _LEGACY_ALLOWLIST)
    assert new == [], (
        "the gateway imports core outside the named surface "
        f"({', '.join(_SURFACE)}); use the surface, do not add to the list"
    )


def test_every_legacy_entry_is_still_an_offender():
    stale = sorted(_LEGACY_ALLOWLIST - legacy_offenders(gateway_edges()))
    assert stale == [], "delete these entries from _LEGACY_ALLOWLIST (and _FROZEN)"


def test_the_legacy_allowlist_cannot_grow_or_swap():
    assert len(_LEGACY_ALLOWLIST) <= _LEGACY_CEILING
    assert _LEGACY_ALLOWLIST <= _LEGACY_FROZEN, "an entry was added or replaced"


def test_private_core_names_are_only_the_known_reaches():
    reaches = private_offenders(gateway_edges())
    assert sorted(reaches - _PRIVATE_ALLOWLIST) == []
    assert sorted(_PRIVATE_ALLOWLIST - reaches) == [], "stale: delete the entry"
    assert len(_PRIVATE_ALLOWLIST) <= _PRIVATE_CEILING


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
    assert modules("from llm_rosetta.shims import quota_probe") == {
        ("llm_rosetta.shims.quota_probe", "")
    }
    assert modules("import llm_rosetta.shims.scaffold as s") == {
        ("llm_rosetta.shims.scaffold", "")
    }
    assert modules("from llm_rosetta import __version__") == {
        ("llm_rosetta", "__version__")
    }
    assert modules(
        "def f():\n    from llm_rosetta.auto_detect import detect_provider"
    ) == {("llm_rosetta.auto_detect", "detect_provider")}
    assert modules(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n    from llm_rosetta.routing import ResolvedRoute"
    ) == {("llm_rosetta.routing", "ResolvedRoute")}
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
    assert legacy_offenders(edges) == {("app/x.py", _NON_LITERAL)}


def test_the_surface_rule():
    for module in _SURFACE:
        assert _is_allowed(module)
    assert _is_allowed("llm_rosetta.routing.strategies")
    assert not _is_allowed("llm_rosetta.observability.usage.report")
    assert not _is_allowed("llm_rosetta.observability")
    assert not _is_allowed("llm_rosetta.shims")
    assert not _is_allowed("llm_rosetta")
    assert not _is_allowed("llm_rosetta.apis")


def test_private_names_and_modules_are_private():
    def edges(source: str) -> list[Edge]:
        return core_edges("x.py", source, ("llm_rosetta_gateway",))

    assert edges("from llm_rosetta._compat import sync_compat")[0].private
    assert edges("from llm_rosetta.shims.providers import _leaf_connection")[0].private
    assert not edges("from llm_rosetta import __version__")[0].private
    assert not edges("from llm_rosetta._vendor import yaml")[0].private


def test_an_undeclared_name_on_the_surface_is_reported():
    edges = core_edges(
        "x.py",
        "from llm_rosetta.routing import ResolvedRoute, NoSuchName",
        ("llm_rosetta_gateway",),
    )
    assert [line.split(" ", 1)[1] for line in undeclared_names(edges)] == [
        "llm_rosetta.routing.NoSuchName"
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
_TESTS_PRIVATE_CEILING = 7
_TESTS_PRIVATE_ALLOWLIST: frozenset[tuple[str, str, str]] = frozenset(
    {
        (
            "test_admin_credential_payload.py",
            "llm_rosetta.shims.providers",
            "_load_single_provider",
        ),
        (
            "test_admin_shim_manifests.py",
            "llm_rosetta.shims.providers",
            "_CONNECTION_KEYS",
        ),
        (
            "test_proxy_transforms.py",
            "llm_rosetta.shims.provider_shim",
            "_reset_registry",
        ),
        ("test_quota.py", "llm_rosetta.shims", "_load_providers"),
        (
            "test_shim_selection.py",
            "llm_rosetta.shims.provider_shim",
            "_reset_registry",
        ),
        (
            "test_shim_sources.py",
            "llm_rosetta.shims.provider_shim",
            "_reset_registry",
        ),
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


def _render(name: str, annotation: str, items: Iterable[tuple[str, ...]]) -> str:
    lines = [f"{name}: {annotation} = frozenset(", "    {"]
    lines += [f"        {item!r}," for item in sorted(items)]
    lines += ["    }", ")"]
    return "\n".join(lines)


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
        "  of which legacy-list entries": len(legacy_offenders(edges)),
    }
    out = [f"# {label}"] + [f"# {value:>5}  {key}" for key, value in rows.items()]
    by_module = Counter(e.module for e in off)
    out.append("# outside the surface, by module (sites):")
    out += [f"#   {n:>3} {m}" for m, n in by_module.most_common()]
    return "\n".join(out)


def main(argv: list[str]) -> None:
    """Print the allowlists, or with ``--report-tests`` the test-tree counts."""
    if "--report-tests" in argv:
        print(_report(_scan_tree(_TESTS, "tests"), "tests/gateway/** (report only)"))
        return
    edges = gateway_edges()
    print(_report(edges, "gateway src (_vendor excluded)"))
    legacy = legacy_offenders(edges)
    private = private_offenders(edges)
    print(f"\n_LEGACY_CEILING = {len(legacy)}")
    print(_render("_LEGACY_ALLOWLIST", "frozenset[tuple[str, str]]", legacy))
    print("\n# The frozen copy is the same literal; edit both when deleting entries.")
    print(_render("_LEGACY_FROZEN", "frozenset[tuple[str, str]]", legacy))
    print(f"\n_PRIVATE_CEILING = {len(private)}")
    print(_render("_PRIVATE_ALLOWLIST", "frozenset[tuple[str, str, str]]", private))
    bad = undeclared_names(edges)
    if bad:
        print("\n# names missing from the surface module's __all__:")
        print("\n".join(f"#   {line}" for line in bad))


if __name__ == "__main__":
    main(sys.argv[1:])
