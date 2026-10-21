"""The maintenance package's façade and its released-path compatibility shim."""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

_INIT = (
    Path(__file__).parents[2]
    / "packages/llm-rosetta-gateway/src/llm_rosetta_gateway/maintenance/__init__.py"
)


def test_facade_resolves_every_public_name():
    import llm_rosetta_gateway.maintenance as maintenance

    for name in maintenance.__all__:
        assert getattr(maintenance, name) is not None


def test_gateway_facade_resolves_every_public_name():
    import llm_rosetta_gateway as gateway

    for name in gateway.__all__:
        assert getattr(gateway, name) is not None


def test_facade_does_not_import_submodules_eagerly():
    """The package __init__ must stay import-free at module scope.

    ``gateway.keystore`` imports ``maintenance.keystore``, which executes this
    __init__; eager sibling imports would drag ``startup`` and ``dumps`` in
    with it, so a future import-time step in ``startup`` would fire from four
    modules away.  The names are resolved lazily through ``_EXPORTS``.
    """
    tree = ast.parse(_INIT.read_text())

    # Walk module-level statements, descending into `if TYPE_CHECKING` only to
    # ignore it: those imports exist for the type checker, not at runtime.
    def module_level_relative_imports(node: ast.AST) -> list[str]:
        found: list[str] = []
        body = getattr(node, "body", [])
        for stmt in body:
            if isinstance(stmt, ast.ImportFrom) and stmt.level:
                found.append("." * stmt.level + (stmt.module or ""))
            elif isinstance(stmt, ast.If):
                test = ast.unparse(stmt.test)
                if "TYPE_CHECKING" not in test:
                    found.extend(module_level_relative_imports(stmt))
        return found

    assert module_level_relative_imports(tree) == []


def test_importing_keystore_does_not_load_startup_or_dumps():
    """The claim in the PR/changelog, checked on the path it names.

    A fresh interpreter is required: pytest's collection imports most of the
    gateway already.  ``gateway.__init__`` is lazy, so importing one submodule
    must not execute the ``.app`` graph that holds these two.
    """
    code = (
        "import sys; import llm_rosetta_gateway.keystore; "
        "print(int('llm_rosetta_gateway.maintenance.startup' in sys.modules), "
        "int('llm_rosetta_gateway.maintenance.dumps' in sys.modules))"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert out == "0 0"


def test_deprecated_gateway_migrations_path_warns_and_reexports():
    """The v0.13.0 dotted path keeps resolving until 0.16.0 (#919)."""
    from llm_rosetta_gateway.maintenance.migrations import CURRENT_VERSION, migrate

    sys.modules.pop("llm_rosetta_gateway.migrations", None)
    with pytest.warns(DeprecationWarning, match="gateway.maintenance.migrations"):
        legacy = importlib.import_module("llm_rosetta_gateway.migrations")

    assert legacy.CURRENT_VERSION == CURRENT_VERSION
    assert legacy.migrate is migrate
