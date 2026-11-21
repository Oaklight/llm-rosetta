"""An in-core shim reaches ``llm_rosetta`` only through the contracts (#1657, C5).

Every ``.py`` file under ``shims/providers/<vendor>/`` (the sidecars:
``transforms.py``, ``auth.py``, ``quota.py`` and their helpers) may import
``llm_rosetta`` only through ``llm_rosetta.contracts`` and its submodules, plus
relative imports inside its own vendor directory.  Not ``llm_rosetta.api``
either: sidecars execute while ``import llm_rosetta`` is still running, and the
api module needs the finished package.  Not another vendor's directory: each
vendor directory has to stand alone so it can move out of core.

The loader's own modules (``shims/providers/_*.py`` and its ``__init__``) are
internal and out of scope.  The scaffold templates, and what the scaffolder
writes, must follow the same rule, because they are what a new shim starts
from.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from _import_scan import (  # noqa: E402
    CONTRACT_PACKAGE_ONLY,
    DYNAMIC_NONLITERAL,
    FROM,
    import_sites,
)

_CORE = Path(__file__).parents[1] / "packages/llm-rosetta/src/llm_rosetta"
_PROVIDERS = _CORE / "shims" / "providers"
_TEMPLATE = _CORE / "shims" / "_shim_template"
_CONTRACTS = "llm_rosetta.contracts"


def _allowed(module: str, own_vendor: str | None) -> bool:
    if any(module.startswith(path + ".") for path in CONTRACT_PACKAGE_ONLY):
        return False
    if module == _CONTRACTS or module.startswith(_CONTRACTS + "."):
        return True
    if own_vendor is not None:
        home = "llm_rosetta.shims.providers." + own_vendor
        return module == home or module.startswith(home + ".")
    return False


def _is_core_submodule(module: str, name: str) -> bool:
    base = _CORE.joinpath(*module.split(".")[1:], name)
    return base.is_dir() or base.with_suffix(".py").is_file()


def offenders(
    source: str, package: tuple[str, ...], own_vendor: str | None
) -> list[str]:
    """Return the ``line module`` of each core import outside the rule."""
    found: list[str] = []
    for site in import_sites(source, package):
        if site.kind == DYNAMIC_NONLITERAL:
            found.append(str(site.lineno) + " <non-literal dynamic import>")
            continue
        modules = [site.module] if site.module else list(site.names)
        if site.kind == FROM and site.module in CONTRACT_PACKAGE_ONLY:
            # ``from llm_rosetta.contracts.usage import normalize`` imports
            # the submodule, which the package-only rule refuses.
            modules += [
                site.module + "." + name
                for name in site.names
                if _is_core_submodule(site.module, name)
            ]
        for module in modules:
            if module.split(".")[0] != "llm_rosetta":
                continue
            if not _allowed(module, own_vendor):
                found.append(str(site.lineno) + " " + module)
    return found


def _sidecars() -> list[Path]:
    return sorted(
        path
        for vendor in _PROVIDERS.iterdir()
        if vendor.is_dir() and not vendor.name.startswith(("_", "."))
        for path in vendor.rglob("*.py")
        if "__pycache__" not in path.parts
    )


SIDECAR_NAMES = frozenset(("transforms.py", "auth.py", "quota.py"))


def test_there_are_sidecars_to_check():
    names = frozenset(path.name for path in _sidecars())
    assert SIDECAR_NAMES <= names


@pytest.mark.parametrize(
    "path", _sidecars(), ids=lambda p: str(p.relative_to(_PROVIDERS))
)
def test_a_sidecar_imports_core_only_through_the_contracts(path: Path):
    rel = path.relative_to(_PROVIDERS)
    package = ("llm_rosetta", "shims", "providers", *rel.parent.parts)
    assert offenders(path.read_text(encoding="utf-8"), package, rel.parts[0]) == []


def _template_sources() -> list[Path]:
    return sorted(_TEMPLATE.rglob("*.py.tmpl"))


@pytest.mark.parametrize(
    "path", _template_sources(), ids=lambda p: str(p.relative_to(_TEMPLATE))
)
def test_a_template_imports_core_only_through_the_contracts(path: Path):
    assert offenders(path.read_text(encoding="utf-8"), ("template",), None) == []


def test_the_scaffolded_vendor_imports_core_only_through_the_contracts(tmp_path):
    from llm_rosetta.shims import scaffold

    root = tmp_path / "providers"
    root.mkdir()
    argv = ["acme", "--root", str(root), "--base-url", "https://example.invalid/v1"]
    argv += ["--transform", "--auth", "--quota", "--no-input"]
    argv += ["--no-registry-manifest"]
    assert scaffold.main(argv) == 0
    written = sorted(root.rglob("*.py"))
    assert SIDECAR_NAMES <= frozenset(p.name for p in written)
    for path in written:
        rel = path.relative_to(root)
        package = ("plugin", *rel.parent.parts)
        assert offenders(path.read_text(encoding="utf-8"), package, None) == [], rel


def test_the_rule():
    acme = ("llm_rosetta", "shims", "providers", "acme")

    def check(source: str) -> list[str]:
        return offenders(source, acme, "acme")

    assert check("from llm_rosetta.contracts.transform import strip_fields") == []
    assert check("from llm_rosetta.contracts import QUOTA_PROBES") == []
    assert check("from .helpers import x") == []
    assert check("import json") == []
    assert check("from llm_rosetta.api import get_shim") == ["1 llm_rosetta.api"]
    assert check("from llm_rosetta import api") == ["1 llm_rosetta"]
    assert check("import llm_rosetta.ir.primitives") == ["1 llm_rosetta.ir.primitives"]
    cross = "1 llm_rosetta.shims.providers.openai.quota"
    assert check("from llm_rosetta.shims.providers.openai.quota import X") == [cross]
    assert check("from ..openai.quota import X") == [cross]


def _body_without_docstring(path: Path) -> str:
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = tree.body[1:] if ast.get_docstring(tree) is not None else tree.body
    return "\n".join(ast.dump(node) for node in body)


def test_the_two_copies_of_the_codex_probe_stay_identical():
    # openai_chatgpt used to import openai's probe; a sidecar may not reach
    # another vendor, so it carries a copy.  Only the module docstring may
    # differ; function docstrings are compared too (comments are not in the
    # AST, so they may differ).  Comparing the whole file is deliberate: if the plan
    # lane ever needs a difference, extract the shared part instead of
    # loosening this test.
    openai = _PROVIDERS / "openai" / "quota.py"
    chatgpt = _PROVIDERS / "openai_chatgpt" / "quota.py"
    assert _body_without_docstring(openai) == _body_without_docstring(chatgpt)


def test_a_contract_package_is_imported_as_a_package_only():
    acme = ("llm_rosetta", "shims", "providers", "acme")
    assert (
        offenders("from llm_rosetta.contracts.usage import UsageReport", acme, "acme")
        == []
    )
    assert offenders("import llm_rosetta.contracts.usage", acme, "acme") == []
    assert offenders("import llm_rosetta.contracts.usage.normalize", acme, "acme") == [
        "1 llm_rosetta.contracts.usage.normalize"
    ]
    assert offenders(
        "from llm_rosetta.contracts.usage import normalize", acme, "acme"
    ) == ["1 llm_rosetta.contracts.usage.normalize"]
    assert offenders(
        "from llm_rosetta.contracts.usage.report import X", acme, "acme"
    ) == ["1 llm_rosetta.contracts.usage.report"]
