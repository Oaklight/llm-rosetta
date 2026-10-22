"""Unit tests for the public-API diff used by the API compatibility CI job."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
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
    "llm_rosetta.capabilities",
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
    # Runtime, not static: ``converters.google_generate`` is a ``sys.modules``
    # alias on some branches, so it has ``__all__`` without a file that declares it.
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
