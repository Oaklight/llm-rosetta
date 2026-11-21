"""Unit tests for the public-API diff used by the API compatibility CI job."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "dev" / "public_api_diff.py"
ALLOWLIST = REPO_ROOT / "scripts" / "dev" / "public_api_allowlist.json"
SRC = REPO_ROOT / "packages" / "llm-rosetta" / "src" / "llm_rosetta"

#: Listed modules with no ``__all__`` of their own.  Their names are
#: re-exported by a package that *is* compared.
NO_ALL_REEXPORTED = {
    "llm_rosetta.auto_detect": "re-exported from llm_rosetta",
    "llm_rosetta_gateway.observability.metrics": "re-exported from llm_rosetta_gateway.observability",
    "llm_rosetta_gateway.observability.persistence": "re-exported from llm_rosetta_gateway.observability",
    "llm_rosetta_gateway.observability.request_log": "re-exported from llm_rosetta_gateway.observability",
    "llm_rosetta_gateway.observability.ops_log": "re-exported from llm_rosetta_gateway.observability",
}

#: Top-level modules that declare ``__all__`` but are implementation detail.
INTERNAL_TOP_LEVEL = {
    "llm_rosetta.formats",
    "llm_rosetta.model_vocabulary",
    "llm_rosetta.profiling",
}


def _module_name(path: Path) -> str:
    parts = path.relative_to(SRC.parent).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _declares_all(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in targets
        ):
            return True
        if isinstance(node, ast.ImportFrom) and any(
            imported.name == "__all__" for imported in node.names
        ):
            return True
    return False


def _declared_all_modules() -> set[str]:
    """Modules that declare ``__all__``, found statically.

    The walk is static on purpose: importing every module would execute the
    gateway CLI's argv parsing.
    """
    return {
        _module_name(path)
        for path in SRC.rglob("*.py")
        if "_vendor" not in path.parts and _declares_all(path)
    }


def _load_script():
    spec = importlib.util.spec_from_file_location("public_api_diff", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


api = _load_script()


def _param(name, kind="POSITIONAL_OR_KEYWORD", required=True):
    return {"name": name, "kind": kind, "required": required}


def _export(kind="function", *params):
    return {"kind": kind, "params": list(params)}


def _surfaces(base_exports, head_exports, module="llm_rosetta"):
    base = {name: {"status": "absent"} for name in api.PUBLIC_MODULES}
    head = {name: {"status": "absent"} for name in api.PUBLIC_MODULES}
    base[module] = {"status": "ok", "exports": base_exports}
    head[module] = {"status": "ok", "exports": head_exports}
    return base, head


def _findings(base_exports, head_exports, module="llm_rosetta"):
    base, head = _surfaces(base_exports, head_exports, module)
    return api.compare(base, head)


def test_removed_export_is_breaking():
    findings = _findings({"OldName": _export()}, {})
    assert [(finding["severity"], finding["key"]) for finding in findings] == [
        ("breaking", "llm_rosetta::OldName::removed")
    ]


def test_added_export_is_informational():
    findings = _findings({}, {"NewName": _export()})
    assert [(finding["severity"], finding["key"]) for finding in findings] == [
        ("info", "llm_rosetta::NewName::added")
    ]


def test_removed_parameter_is_breaking():
    findings = _findings(
        {"convert": _export("function", _param("old"))},
        {"convert": _export("function")},
    )
    assert [(finding["severity"], finding["param"]) for finding in findings] == [
        ("breaking", "old")
    ]


def test_added_required_parameter_is_breaking():
    findings = _findings(
        {"convert": _export("function")},
        {"convert": _export("function", _param("new_required"))},
    )
    assert [(finding["severity"], finding["param"]) for finding in findings] == [
        ("breaking", "new_required")
    ]


def test_added_optional_parameter_is_informational():
    findings = _findings(
        {"convert": _export("function")},
        {"convert": _export("function", _param("new_optional", required=False))},
    )
    assert [(finding["severity"], finding["param"]) for finding in findings] == [
        ("info", "new_optional")
    ]


def test_parameter_becoming_required_is_breaking():
    findings = _findings(
        {"convert": _export("function", _param("arg", required=False))},
        {"convert": _export("function", _param("arg", required=True))},
    )
    assert [(finding["severity"], finding["change"]) for finding in findings] == [
        ("breaking", "parameter-became-required")
    ]


def test_parameter_kind_change_is_breaking():
    findings = _findings(
        {"convert": _export("function", _param("arg"))},
        {"convert": _export("function", _param("arg", kind="KEYWORD_ONLY"))},
    )
    assert [(finding["severity"], finding["change"]) for finding in findings] == [
        ("breaking", "parameter-kind-changed")
    ]


def test_private_parameters_are_ignored():
    findings = _findings(
        {"StreamContext": _export("class", _param("_internal_state"))},
        {"StreamContext": _export("class")},
    )
    assert findings == []


def test_export_kind_change_is_breaking():
    findings = _findings({"Thing": _export("class")}, {"Thing": _export("function")})
    assert [(finding["severity"], finding["change"]) for finding in findings] == [
        ("breaking", "kind-changed")
    ]


def test_removed_module_reports_each_export():
    base, head = _surfaces({"Keep": _export()}, {"Keep": _export()})
    head["llm_rosetta"] = {"status": "absent"}
    findings = api.compare(base, head)
    assert [finding["key"] for finding in findings] == [
        "llm_rosetta::Keep::module-removed"
    ]
    assert findings[0]["severity"] == "breaking"


def test_report_passes_when_break_is_acknowledged():
    findings = _findings({"OldName": _export()}, {})
    text, status = api.report(
        findings, {"llm_rosetta::OldName::removed": "planned"}, "v1", "HEAD"
    )
    assert status == 0
    assert "ACKNOWLEDGED BREAKS" in text


def test_report_fails_when_break_is_unacknowledged():
    findings = _findings({"OldName": _export()}, {})
    text, status = api.report(findings, {}, "v1", "HEAD")
    assert status == 1
    assert "BREAKING" in text


def test_report_fails_when_acknowledged_break_has_no_reason():
    findings = _findings({"OldName": _export()}, {})
    text, status = api.report(
        findings, {"llm_rosetta::OldName::removed": "   "}, "v1", "HEAD"
    )
    assert status == 1
    assert "acknowledged without reason" in text


def test_report_fails_on_stale_allowlist_entries():
    findings = _findings({}, {})
    text, status = api.report(
        findings, {"llm_rosetta::Gone::removed": "stale"}, "v1", "HEAD"
    )
    assert status == 1
    assert "STALE ALLOWLIST ENTRIES" in text


def test_inferred_surface_is_reported():
    base: dict[str, dict] = {name: {"status": "absent"} for name in api.PUBLIC_MODULES}
    head: dict[str, dict] = {name: {"status": "absent"} for name in api.PUBLIC_MODULES}
    base["llm_rosetta"] = {
        "status": "ok",
        "source": "declared",
        "exports": {"f": _export()},
    }
    head["llm_rosetta"] = {
        "status": "ok",
        "source": "inferred",
        "exports": {"f": _export()},
    }
    findings = api.compare(base, head)
    assert [finding["key"] for finding in findings] == [
        "llm_rosetta::<module>::surface-inferred"
    ]


def test_inferred_surface_sees_constants_and_intra_package_reexports(tmp_path):
    """No ``__all__`` means reachability, not just definitions (#1123).

    A constant has no ``__module__`` and a re-exported alias reports the
    alphabet it came from (``typing`` for a ``Literal``), so the old
    ``__module__ == module_name`` fallback saw neither.  The source does.
    """
    package = tmp_path / "inferred_surface_fixture"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "defs.py").write_text(
        "\n\n".join(
            [
                "MAX_ENTRIES = 3",
                "logger = object()",
                "",
                "def helper():",
                "    return 1",
                "",
                "class Widget:",
                "    pass",
            ]
        ),
        encoding="utf-8",
    )
    (package / "surface.py").write_text(
        "\n".join(
            [
                "import os",
                "from typing import Any",
                "from .defs import MAX_ENTRIES, Widget, helper",
                "OWN_LIMIT = 5",
            ]
        ),
        encoding="utf-8",
    )
    sys.path.insert(0, str(tmp_path))
    try:
        module = importlib.import_module("inferred_surface_fixture.surface")
        names = api._inferred_names(module, "inferred_surface_fixture.surface")
    finally:
        sys.path.remove(str(tmp_path))
        for name in ("inferred_surface_fixture.surface", "inferred_surface_fixture"):
            sys.modules.pop(name, None)

    assert names == ["MAX_ENTRIES", "OWN_LIMIT", "Widget", "helper"]


def test_inferred_surface_keeps_data_constants_of_a_real_module():
    """Deleting ``ALL_EVENT_TYPES`` must be visible to the checker."""
    module = importlib.import_module("llm_rosetta_gateway.observability.ops_log")
    assert getattr(module, "__all__", None) is None
    names = api._inferred_names(module, "llm_rosetta_gateway.observability.ops_log")
    assert "ALL_EVENT_TYPES" in names
    assert "ALL_SEVERITIES" in names


def test_listed_modules_have_a_real_surface():
    # Runtime, not static: a listed module can take its ``__all__`` from a
    # ``sys.modules`` alias, with no file that declares it.
    missing = [
        module_name
        for module_name in api.PUBLIC_MODULES
        if getattr(importlib.import_module(module_name), "__all__", None) is None
    ]
    missing = [name for name in missing if name not in NO_ALL_REEXPORTED]
    assert not missing, (
        "These PUBLIC_MODULES entries have no __all__; add one or record where "
        f"their names are re-exported: {missing}"
    )


def test_top_level_modules_with_all_are_listed():
    top_level = {
        module
        for module in _declared_all_modules()
        if module.startswith("llm_rosetta.") and module.count(".") == 1
    }
    missing = top_level - set(api.PUBLIC_MODULES) - INTERNAL_TOP_LEVEL
    assert not missing, (
        f"Top-level modules with __all__ are not covered: {sorted(missing)}"
    )


@pytest.mark.parametrize(
    "module_name", ["llm_rosetta.tool_ops", "llm_rosetta.pipeline"]
)
def test_all_covers_every_public_definition(module_name):
    """A listed module must not silently drop a public name from its surface."""
    module = importlib.import_module(module_name)
    defined = {
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and getattr(value, "__module__", None) == module_name
    }
    assert defined <= set(module.__all__), sorted(defined - set(module.__all__))


def test_checked_in_allowlist_entries_are_complete():
    entries = json.loads(ALLOWLIST.read_text(encoding="utf-8"))["acknowledged"]
    assert entries
    for entry in entries:
        assert entry["key"].count("::") >= 2
        assert entry["reason"].strip()


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        # The list is version-sorted, so dev tags appear above stable ones.
        (["v0.14.0.dev4", "v0.14.0.dev3", "v0.13.0", "v0.12.0"], "v0.13.0"),
        (["v0.14.0rc1", "v0.13.0"], "v0.13.0"),
        (["v1.0.0-beta", "v0.13.0"], "v0.13.0"),
        (["v0.14.0.dev4", "v0.14.0rc1"], None),
        (["v0.13.0", "v0.12.0"], "v0.13.0"),
        ([" v0.13.0\n"], "v0.13.0"),
        ([], None),
    ],
)
def test_select_stable_tag_ignores_prereleases(tags, expected):
    """A dev or rc tag must never be used as the stable baseline."""
    assert api.select_stable_tag(tags) == expected


@pytest.mark.parametrize("module_name", api.PUBLIC_MODULES)
def test_public_module_is_importable(module_name):
    """A typo in PUBLIC_MODULES would otherwise be skipped silently."""
    importlib.import_module(module_name)


def _make_packages_tree(root: Path) -> None:
    (root / "packages/llm-rosetta/src/llm_rosetta").mkdir(parents=True)
    (root / "packages/llm-rosetta-gateway/src/llm_rosetta_gateway").mkdir(parents=True)


def test_src_roots_include_the_gateway_root_of_a_packages_tree(tmp_path):
    _make_packages_tree(tmp_path)
    roots = api._src_roots(tmp_path)
    assert roots == [
        (tmp_path / "packages/llm-rosetta/src").resolve(),
        (tmp_path / "packages/llm-rosetta-gateway/src").resolve(),
    ]


def test_src_roots_of_the_real_checkout_include_the_gateway():
    roots = api._src_roots(REPO_ROOT)
    assert any((root / "llm_rosetta_gateway").is_dir() for root in roots)
    assert any((root / "llm_rosetta").is_dir() for root in roots)


def test_src_roots_of_an_old_layout_tree_has_only_the_core_root(tmp_path):
    (tmp_path / "src/llm_rosetta").mkdir(parents=True)
    assert api._src_roots(tmp_path) == [(tmp_path / "src").resolve()]


def test_src_roots_rejects_a_tree_without_sources(tmp_path):
    with pytest.raises(SystemExit):
        api._src_roots(tmp_path)


def test_core_root_is_named_by_what_it_holds(tmp_path):
    _make_packages_tree(tmp_path)
    assert api._core_root(tmp_path) == (tmp_path / "packages/llm-rosetta/src").resolve()


def test_core_root_rejects_a_gateway_only_tree(tmp_path):
    (tmp_path / "packages/llm-rosetta-gateway/src/llm_rosetta_gateway").mkdir(
        parents=True
    )
    with pytest.raises(SystemExit, match="no llm_rosetta source directory"):
        api._core_root(tmp_path)


def _copy_clean_tree(destination: Path) -> Path:
    """Copy the two source roots into a tree no editable install points at."""
    ignore = shutil.ignore_patterns("__pycache__", "*.egg-info", "*.pyc")
    for package in ("llm-rosetta", "llm-rosetta-gateway"):
        shutil.copytree(
            REPO_ROOT / "packages" / package / "src",
            destination / "packages" / package / "src",
            ignore=ignore,
        )
    return destination


def test_the_script_runs_against_a_clean_tree(tmp_path):
    """``--base <tree>`` on a bare copy runs end to end (#1644 review, #1630).

    CI reaches the base through ``--base-ref`` and a git worktree; a plain
    ``--base`` directory is the path that once died with ``ImportError``
    while the installed copy masked it.  The copy has no editable install,
    so every module must come from the copy itself (the dump's origin checks
    fail the run otherwise), and a tree compared with itself has no finding.
    """
    tree = _copy_clean_tree(tmp_path / "tree")
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text('{"acknowledged": []}', encoding="utf-8")

    surface = api.dump_in_subprocess(tree)
    assert {name: entry["status"] for name, entry in surface.items()} == {
        name: "ok" for name in api.COMPARED_MODULES
    }
    status = api.main(
        ["--base", str(tree), "--head", str(tree), "--allowlist", str(allowlist)]
    )
    assert status == 0


# ---------------------------------------------------------------------------
# Each tree is measured with its own tier table (#1657, #1659 review)
# ---------------------------------------------------------------------------


def _write_tiers(root: Path, **tiers: tuple[str, ...]) -> None:
    core = root / "packages/llm-rosetta/src/llm_rosetta"
    core.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{tier} = {tuple(tiers.get(tier, ()))!r}"
        for tier in ("USER", "EXTENSION", "COMPAT", "REMOVED")
    ]
    (core / "_tiers.py").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_tree_modules_reads_the_trees_own_tier_table(tmp_path):
    _write_tiers(
        tmp_path,
        USER=("llm_rosetta", "llm_rosetta.dropped_later"),
        EXTENSION=("llm_rosetta.api",),
    )
    assert api.tree_modules(tmp_path) == (
        "llm_rosetta",
        "llm_rosetta.dropped_later",
        "llm_rosetta.api",
        *api.GATEWAY_PUBLIC_MODULES,
    )


def test_tree_modules_falls_back_to_the_script_list_without_a_table(tmp_path):
    (tmp_path / "src/llm_rosetta").mkdir(parents=True)
    assert api.tree_modules(tmp_path) == api.COMPARED_MODULES


def test_tree_modules_of_this_checkout_is_the_script_list():
    assert api.tree_modules(REPO_ROOT) == api.COMPARED_MODULES


def test_tree_modules_rejects_a_module_listed_twice(tmp_path):
    _write_tiers(tmp_path, USER=("llm_rosetta",), COMPAT=("llm_rosetta",))
    with pytest.raises(SystemExit, match="twice"):
        api.tree_modules(tmp_path)


def test_a_module_dropped_from_the_head_tiers_is_reported():
    """``compare()`` reports a module only the base dump has.

    This drives ``compare()`` with two hand-built dumps, as a base that ships
    its own ``_tiers.py`` (0.14.0 on) produces them: the head dump has no
    entry for the module at all (not even ``absent``).  Every name it
    exported is a breaking finding.  It does not exercise ``tree_modules()``:
    against v0.13.0, which has no tier table, the base dump is built from the
    head's list, so a module dropped from the head's tiers is not in either
    dump; ``test_v0_13_0_public_modules_stay_in_the_head_tiers`` covers that
    base.
    """
    base = {
        "llm_rosetta": {"status": "ok", "exports": {"keep": _export()}},
        "llm_rosetta.dropped": {
            "status": "ok",
            "exports": {"one": _export(), "two": _export()},
        },
    }
    head = {"llm_rosetta": {"status": "ok", "exports": {"keep": _export()}}}
    findings = api.compare(base, head)
    assert [(finding["severity"], finding["key"]) for finding in findings] == [
        ("breaking", "llm_rosetta.dropped::one::module-removed"),
        ("breaking", "llm_rosetta.dropped::two::module-removed"),
    ]
    assert "not in the head tiers" in findings[0]["detail"]
    _text, status = api.report(findings, {}, "v1", "HEAD")
    assert status == 1


#: The modules v0.13.0's public surface had: derived once from the v0.13.0 tag
#: (the modules of the script's list that dump ``ok`` from that tree) and
#: frozen here.  v0.13.0 ships no ``_tiers.py``, so its dump is built from the
#: head's list; a module the head drops from its tiers would vanish from both
#: dumps and its removal would pass silently.  This list closes that gap for
#: the 0.13.0 base.
V0_13_0_PUBLIC_MODULES = (
    "llm_rosetta",
    "llm_rosetta.auto_detect",
    "llm_rosetta.capabilities",
    "llm_rosetta.converters",
    "llm_rosetta.converters.base",
    "llm_rosetta.converters.rerank",
    "llm_rosetta.fidelity",
    "llm_rosetta.gateway",
    "llm_rosetta.observability",
    "llm_rosetta.pipeline",
    "llm_rosetta.routing",
    "llm_rosetta.shims",
    "llm_rosetta.shims.provider_shim",
    "llm_rosetta.tool_ops",
    "llm_rosetta.types",
    "llm_rosetta.types.ir",
    "llm_rosetta.types.ir.extensions",
)


def test_v0_13_0_public_modules_stay_in_the_head_tiers():
    """Every v0.13.0 public module is still compared against v0.13.0.

    A module may leave the user, extension and compat tiers only by moving to
    ``REMOVED``, which keeps it in the comparison, so its removal shows as
    acknowledged allowlist entries rather than vanishing.
    """
    from llm_rosetta import _tiers  # noqa: PLC0415

    compared = set(_tiers.USER + _tiers.EXTENSION + _tiers.COMPAT + _tiers.REMOVED)
    assert sorted(set(V0_13_0_PUBLIC_MODULES) - compared) == []


def test_a_module_only_the_head_lists_is_an_addition():
    base = {"llm_rosetta": {"status": "ok", "exports": {}}}
    head = {
        "llm_rosetta": {"status": "ok", "exports": {}},
        "llm_rosetta.new": {"status": "ok", "exports": {"f": _export()}},
    }
    assert [finding["key"] for finding in api.compare(base, head)] == [
        "llm_rosetta.new::<module>::module-added"
    ]


def test_check_dump_complete_accepts_one_entry_per_listed_module():
    api.check_dump_complete({"a": {}, "b": {}}, ("a", "b"), "tree")


@pytest.mark.parametrize(
    "surface",
    [{"a": {}}, {"a": {}, "b": {}, "c": {}}, {"a": {}, "c": {}}],
    ids=["missing", "extra", "different"],
)
def test_check_dump_complete_rejects_a_dump_that_disagrees(surface):
    with pytest.raises(SystemExit, match="tier"):
        api.check_dump_complete(surface, ("a", "b"), "tree")


def test_the_dump_of_this_checkout_covers_its_tier_table():
    from llm_rosetta import _tiers  # noqa: PLC0415

    surface = api.dump_in_subprocess(REPO_ROOT)
    listed = (
        len(_tiers.USER)
        + len(_tiers.EXTENSION)
        + len(_tiers.COMPAT)
        + len(_tiers.REMOVED)
        + len(api.GATEWAY_PUBLIC_MODULES)
    )
    assert len(surface) == listed
