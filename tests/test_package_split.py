"""The gateway is its own distribution; the old import root is a shim (#1157).

`llm-rosetta` no longer ships the gateway.  ``llm_rosetta.gateway`` stays
importable until 0.16.0 (#919) but resolves to ``llm_rosetta_gateway``: the
same module objects, so state is never duplicated, and a warning so the
migration is visible.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys

from pathlib import Path


def test_importing_the_old_root_warns() -> None:
    """A fresh interpreter importing the old root is warned.

    In-process is not testable: the shim installs a meta-path finder on its
    first import, and from then on any re-import of ``llm_rosetta.gateway``
    is resolved *by that finder* — no module body, no warning.  That is the
    right behaviour (one warning per process, as an import-time warning
    should be), but it means popping ``sys.modules`` and importing again
    observes nothing, and the outcome then depends on what else the suite
    imported first.  A subprocess has no finder yet, which is exactly the
    user's situation.
    """
    code = (
        "import warnings\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    import llm_rosetta.gateway\n"
        "print(int(any('llm_rosetta_gateway' in str(w.message) for w in caught)))\n"
    )
    root = Path(__file__).parents[1]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            str(root / "packages" / pkg / "src")
            for pkg in ("llm-rosetta", "llm-rosetta-gateway")
        ),
    }
    out = subprocess.check_output(
        [sys.executable, "-c", code], text=True, env=env
    ).strip()
    assert out == "1", "importing llm_rosetta.gateway must warn about the new root"


def test_old_submodules_are_the_same_objects() -> None:
    old = importlib.import_module("llm_rosetta.gateway.config")
    new = importlib.import_module("llm_rosetta_gateway.config")

    assert old is new


def test_the_core_package_ships_only_a_shim_for_the_gateway() -> None:
    import llm_rosetta

    core = Path(llm_rosetta.__file__).parent
    gateway_dir = core / "gateway"

    assert sorted(path.name for path in gateway_dir.glob("*.py")) == ["__init__.py"]
    assert not (gateway_dir / "app").exists()


def test_a_core_only_install_names_the_missing_distribution() -> None:
    """Without the gateway installed, the shim says what to add.

    The core wheel ships the shim, so ``import llm_rosetta.gateway`` in a
    core-only environment must not leak the alias target's module name.
    """
    code = """
import sys

class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == "llm_rosetta_gateway" or name.startswith("llm_rosetta_gateway."):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return None

sys.meta_path.insert(0, Blocker())
import llm_rosetta.gateway
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode != 0
    assert "pip install llm-rosetta-gateway" in result.stderr
