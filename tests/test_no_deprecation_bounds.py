"""The completeness gate for the 0.14.0 removals.

A removal is only done when nothing is left to say it was pending.  This test
walks every distribution's source (``_vendor`` excluded) and fails on:

* a ``warnings.warn(..., DeprecationWarning)`` call outside the short list of
  deprecations that are kept on purpose;
* a source line that names a removal release (``0.16.0``) or the removal issue
  (``#919``): a deadline written into code goes stale the day it is missed;
* a module that is nothing but re-exports and a warning, which is what a
  forwarding shim looks like;
* a removed path or symbol that imports again.

The list of removed public names is read from ``public_api_allowlist.json``, so
a removal that is acknowledged there is pinned here without a second list.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [
    ROOT / "packages/llm-rosetta/src",
    ROOT / "packages/llm-rosetta-gateway/src",
]

#: The deprecations that stay, and why.  Shrink only: a file that no longer
#: warns must leave this table, and a new warning needs a reason here.
KEPT_DEPRECATIONS = dict(
    [
        (
            "llm_rosetta/gateway/__init__.py",
            "the llm_rosetta.gateway import root forwards to llm_rosetta_gateway (C01)",
        ),
        (
            "llm_rosetta/observability/__init__.py",
            "the llm_rosetta.observability finder forwards to the gateway (C02)",
        ),
        (
            "llm_rosetta/routing.py",
            "0.13.0 flat module, forwards to llm_rosetta.contracts.routing for one minor",
        ),
        (
            "llm_rosetta/capabilities.py",
            "0.13.0 flat module, forwards to the passes' current homes for one minor",
        ),
        (
            "llm_rosetta/pipeline/chat/ir_transforms.py",
            "apply_ir_transforms keeps its older signature for external callers",
        ),
        (
            "llm_rosetta/pipeline/chat/pipeline.py",
            "ConversionPipeline(shim=...) is the older spelling of target_shim",
        ),
        (
            "llm_rosetta/types/ir/extensions.py",
            "an IR extension field that readers still produce",
        ),
        (
            "llm_rosetta_gateway/observability/persistence/manager.py",
            "PersistenceManager(error_max=...) is the older spelling of dump_max",
        ),
    ]
)

#: Forwarding shells that stay.  ``types/ir/extensions.py`` shipped in 0.13.0 as
#: a deprecated forwarder to ``extensions_experimental`` and is not in the
#: removal inventory; whether it goes is a maintainer decision.  Shrink only.
KEPT_SHELLS = dict(
    [
        (
            "llm_rosetta/types/ir/extensions.py",
            "deprecated forwarder to extensions_experimental, outside the inventory",
        ),
        (
            "llm_rosetta/routing.py",
            "0.13.0 flat module kept for one minor (#1657 compat decision)",
        ),
        (
            "llm_rosetta/capabilities.py",
            "0.13.0 flat module kept for one minor (#1657 compat decision)",
        ),
    ]
)

#: A deadline written into source goes stale.  The shim-removal tests carry the
#: release numbers they need as data, and live under ``tests/``, not here.
DEADLINES = ("0.16.0", "#919")


def _sources() -> list[Path]:
    files: list[Path] = []
    for root in SOURCES:
        for path in sorted(root.rglob("*.py")):
            if "_vendor" in path.parts:
                continue
            files.append(path)
    return files


def _rel(path: Path) -> str:
    for root in SOURCES:
        if root in path.parents:
            return path.relative_to(root).as_posix()
    return path.as_posix()


def _is_warn_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    named = (isinstance(func, ast.Attribute) and func.attr == "warn") or (
        isinstance(func, ast.Name) and func.id == "warn"
    )
    if not named:
        return False
    names = [arg.id for arg in node.args if isinstance(arg, ast.Name)]
    names += [kw.value.id for kw in node.keywords if isinstance(kw.value, ast.Name)]
    return "DeprecationWarning" in names


def _deprecation_files() -> set[str]:
    found: set[str] = set()
    for path in _sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if any(_is_warn_call(node) for node in ast.walk(tree)):
            found.add(_rel(path))
    return found


def test_only_the_kept_deprecations_warn() -> None:
    warned = _deprecation_files()
    unexpected = sorted(warned - set(KEPT_DEPRECATIONS))
    assert unexpected == [], (
        "a DeprecationWarning left in source; remove the deprecated path, or "
        f"give it a reason in KEPT_DEPRECATIONS: {unexpected}"
    )


def test_every_kept_deprecation_still_warns() -> None:
    stale = sorted(set(KEPT_DEPRECATIONS) - _deprecation_files())
    assert stale == [], f"delete these from KEPT_DEPRECATIONS: {stale}"


def test_no_deadline_is_written_into_source() -> None:
    hits = []
    for path in _sources():
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for deadline in DEADLINES:
                if deadline in line:
                    hits.append(f"{_rel(path)}:{number}: {line.strip()}")
    assert hits == [], "\n".join(hits)


def _only_reexports_and_a_warning(tree: ast.Module) -> bool:
    """True for a module whose body is imports, ``__all__`` and a warn call."""
    warns = False
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # the docstring
        if isinstance(node, ast.Expr) and _is_warn_call(node.value):
            warns = True
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if all(isinstance(t, ast.Name) and t.id.startswith("__") for t in targets):
                continue
        return False
    return warns


def test_no_module_is_only_a_reexport_and_a_warning() -> None:
    shells = set()
    for path in _sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if _only_reexports_and_a_warning(tree):
            shells.add(_rel(path))
    assert sorted(shells - set(KEPT_SHELLS)) == [], "forwarding shells left"
    assert sorted(set(KEPT_SHELLS) - shells) == [], "delete these from KEPT_SHELLS"


def test_the_shell_detector_sees_a_shell() -> None:
    shell = ast.parse(
        '"""old"""\n'
        "import warnings\n"
        "from new import thing\n"
        "__all__ = ['thing']\n"
        "warnings.warn('moved', DeprecationWarning)\n"
    )
    real = ast.parse("import warnings\n\ndef f():\n    pass\n")
    assert _only_reexports_and_a_warning(shell)
    assert not _only_reexports_and_a_warning(real)


#: Paths and symbols removed in 0.14.0 that must not import again.  The rest of
#: the removed public names come from ``public_api_allowlist.json`` below.
REMOVED_MODULES = [
    "llm_rosetta.provider_names",
    "llm_rosetta.shims.transforms",
    "llm_rosetta.converters.openai_chat",
    "llm_rosetta.converters.anthropic",
    "llm_rosetta.converters.openai_responses",
    "llm_rosetta.converters.google_genai",
    "llm_rosetta.converters.google_generate",
    "llm_rosetta.converters.google_interactions",
    "llm_rosetta.converters.embedding",
    "llm_rosetta.converters.base.helpers.cache_breakpoints",
    "llm_rosetta.converters.base.helpers.image_limit",
    "llm_rosetta.converters.base.helpers.system_message_hoist",
    "llm_rosetta.converters.base.helpers.tool_group",
    "llm_rosetta.converters.base.helpers.truncate",
    "llm_rosetta.converters.base.tools.orphan_fix",
    "llm_rosetta.converters.base.tools.call_id",
    "llm_rosetta_gateway.auth",
    "llm_rosetta_gateway.headers",
    "llm_rosetta_gateway.embeddings",
    "llm_rosetta_gateway.migrations",
    "llm_rosetta_gateway.middleware.sanitize",
]

REMOVED_SYMBOLS = [
    ("llm_rosetta", "normalize_provider_name"),
    ("llm_rosetta.api", "canonical_provider_name"),
    ("llm_rosetta.api", "canonical_shim_name"),
    ("llm_rosetta.api", "legacy_names"),
    ("llm_rosetta.api", "warn_legacy_provider_names"),
    ("llm_rosetta.api", "warn_legacy_shim_names"),
    ("llm_rosetta.api", "canonical_model_type"),
    ("llm_rosetta.api", "legacy_model_type"),
    ("llm_rosetta.formats", "LEGACY_MODEL_TYPES"),
    ("llm_rosetta.formats", "canonical_model_type"),
    ("llm_rosetta.formats", "legacy_model_type"),
    ("llm_rosetta.retired_names", "ALIASES_ENABLED"),
    ("llm_rosetta.shims", "resolve_base"),
    ("llm_rosetta_gateway.maintenance.names", "correct_legacy_names"),
]


@pytest.mark.parametrize("module", REMOVED_MODULES)
def test_a_removed_module_does_not_import(module: str) -> None:
    try:
        spec = importlib.util.find_spec(module)
    except ModuleNotFoundError:
        spec = None  # its parent package is gone too
    assert spec is None, f"{module} is importable again"


@pytest.mark.parametrize(("module", "name"), REMOVED_SYMBOLS)
def test_a_removed_symbol_is_absent(module: str, name: str) -> None:
    assert not hasattr(importlib.import_module(module), name), f"{module}.{name}"


def _acknowledged_removals() -> list[tuple[str, str]]:
    document = json.loads(
        (ROOT / "scripts/dev/public_api_allowlist.json").read_text(encoding="utf-8")
    )
    removals = []
    for entry in document["acknowledged"]:
        parts = entry["key"].split("::")
        # Only the deletions the removal issue lists.  An older entry may say a
        # name left a module's surface while the module still imports it for
        # its own signatures, which a ``hasattr`` check cannot tell apart.
        if (
            len(parts) == 3
            and parts[2] == "removed"
            and entry["reason"].startswith("Removed in 0.14.0 (issue 919")
        ):
            removals.append((parts[0], parts[1]))
    return removals


ACKNOWLEDGED = _acknowledged_removals()


def test_the_allowlist_names_removals() -> None:
    assert len(ACKNOWLEDGED) >= 10


@pytest.mark.parametrize(
    ("module", "name"), ACKNOWLEDGED, ids=[f"{m}.{n}" for m, n in ACKNOWLEDGED]
)
def test_an_acknowledged_removal_stays_removed(module: str, name: str) -> None:
    """Every ``<module>::<name>::removed`` entry names something that is gone."""
    try:
        loaded = importlib.import_module(module)
    except ModuleNotFoundError:
        return
    assert not hasattr(loaded, name), f"{module}.{name} is back"
