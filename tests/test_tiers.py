"""The tier table (``llm_rosetta._tiers``) lists real modules (#1657, C3)."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import warnings
from pathlib import Path

import pytest

from llm_rosetta import _tiers

_CORE = Path(_tiers.__file__).parent
LISTED = _tiers.USER + _tiers.EXTENSION + _tiers.COMPAT


@pytest.mark.parametrize("module_name", LISTED)
def test_a_listed_module_imports(module_name: str):
    # A compat module warns on import; that is its job, not a failure.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", _tiers.REMOVED)
def test_a_removed_module_stays_removed(module_name: str):
    assert importlib.util.find_spec(module_name) is None


def test_a_module_has_one_tier():
    every = LISTED + _tiers.REMOVED
    assert len(every) == len(frozenset(every))


def test_every_contract_module_is_in_the_extension_tier():
    contracts = _CORE / "contracts"
    found = ["llm_rosetta.contracts"]
    for path in sorted(contracts.iterdir()):
        if path.name.startswith("_"):
            continue  # a private module has no tier
        if path.suffix == ".py":
            found.append("llm_rosetta.contracts." + path.stem)
        elif (path / "__init__.py").is_file():
            found.append("llm_rosetta.contracts." + path.name)
    assert sorted(found) == sorted(
        name for name in _tiers.EXTENSION if name.startswith("llm_rosetta.contracts")
    )


def test_the_table_is_data_only():
    # public_api_diff loads it by path, without importing llm_rosetta.
    tree = ast.parse(Path(_tiers.__file__).read_text(encoding="utf-8"))
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert imports == []


def test_public_api_diff_reads_the_table():
    script = _CORE.parents[3] / "scripts" / "dev" / "public_api_diff.py"
    spec = importlib.util.spec_from_file_location("_tiers_public_api_diff", script)
    assert spec is not None and spec.loader is not None
    diff = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diff)
    core = [name for name in diff.PUBLIC_MODULES if not name.startswith("llm_rosetta_")]
    assert core == list(LISTED)
    assert diff.REMOVED_MODULES == _tiers.REMOVED
