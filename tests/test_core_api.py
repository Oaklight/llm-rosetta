"""``llm_rosetta.api``: the small public surface hosts import (#1630).

The module holds re-exports only.  These tests pin its shape: a literal,
grouped and sorted ``__all__``, every name resolvable, and an import that
stays light (no gateway, no provider SDK).
"""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import llm_rosetta.api as api

_SOURCE = Path(api.__file__).read_text()


def _literal_all() -> ast.List:
    for node in ast.parse(_SOURCE).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            assert isinstance(node.value, ast.List), "__all__ must be a list literal"
            return node.value
    raise AssertionError("llm_rosetta.api defines no __all__")


def _groups() -> list[list[str]]:
    """The ``__all__`` names, split at each comment line inside the list."""
    literal = _literal_all()
    lines = _SOURCE.splitlines()
    groups: list[list[str]] = []
    current: list[str] = []
    for text in lines[literal.lineno : (literal.end_lineno or 0) - 1]:
        stripped = text.strip()
        if stripped.startswith("#"):
            if current:
                groups.append(current)
            current = []
        else:
            current.append(ast.literal_eval(stripped.rstrip(",")))
    if current:
        groups.append(current)
    return groups


def test_all_is_a_literal_list_of_strings():
    literal = _literal_all()
    assert all(
        isinstance(e, ast.Constant) and isinstance(e.value, str) for e in literal.elts
    )
    assert [ast.literal_eval(e) for e in literal.elts] == api.__all__


def test_all_has_no_duplicates():
    assert len(api.__all__) == len(set(api.__all__))


@pytest.mark.parametrize("name", api.__all__)
def test_every_name_resolves(name: str):
    assert hasattr(api, name)


def test_all_is_grouped_and_each_group_is_sorted():
    groups = _groups()
    assert len(groups) > 1
    assert [n for group in groups for n in group] == api.__all__
    for group in groups:
        assert group == sorted(group), group


def test_module_holds_re_exports_only():
    """No function or class is defined here; the surface is not a second home."""
    defined = [
        node.name
        for node in ast.parse(_SOURCE).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    assert defined == []


def test_names_are_the_same_objects_as_their_home():
    import llm_rosetta
    from llm_rosetta.pipeline import ConversionPipeline
    from llm_rosetta.shims import provider_shim

    assert api.ConversionPipeline is ConversionPipeline
    assert api.ProviderShim is provider_shim.ProviderShim
    assert api.get_shim is provider_shim.get_shim
    assert api.__version__ == llm_rosetta.__version__


def test_contract_modules_are_not_re_exported():
    """Each name has one home: the four contract modules keep theirs.

    ``routing`` also lists ``ProviderType`` and ``SoftErrorPattern``; #1630
    drops them from there once their home here settles.
    """
    for module in (
        "llm_rosetta.contracts.auth",
        "llm_rosetta.contracts.quota",
        "llm_rosetta.contracts.routing",
        "llm_rosetta.contracts.usage",
    ):
        contract = importlib.import_module(module)
        # Some contract modules declare no ``__all__``: take what they define.
        names = set(getattr(contract, "__all__", ())) | {
            name
            for name, value in vars(contract).items()
            if getattr(value, "__module__", None) == contract.__name__
        }
        shared = names & set(api.__all__)
        assert shared <= {"ProviderType", "SoftErrorPattern"}, (module, shared)


def _run(code: str) -> str:
    return subprocess.check_output([sys.executable, "-c", code], text=True).strip()


def test_import_loads_no_gateway_module():
    out = _run(
        "import sys\n"
        "import llm_rosetta.api\n"
        "print('\\n'.join(sorted(m for m in sys.modules "
        "if m == 'llm_rosetta_gateway' or m.startswith('llm_rosetta_gateway.'))))"
    )
    assert out == "", out


def test_import_loads_no_provider_sdk():
    out = _run(
        "import sys\n"
        "import llm_rosetta.api\n"
        "print('\\n'.join(m for m in ('openai', 'anthropic', 'google.genai') "
        "if m in sys.modules))"
    )
    assert out == "", out
