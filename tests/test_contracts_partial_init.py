"""Every contract module imports while ``llm_rosetta`` is still initialising (#1657).

Plugin entry points (``llm_rosetta.shim_providers``) run during
``import llm_rosetta``: ``shims/__init__`` loads them.  A plugin whose entry
point or sidecar imports a name the half-initialised package does not have yet
(``llm_rosetta.api``, ``__version__``) fails, and the loader only logs it, so
its shims silently never register while every in-process test stays green.

This test installs a real entry point (a ``.dist-info`` on ``sys.path``) for
the plugin package in ``tests/contracts_plugin``, whose entry point and
sidecars import every contract module, and checks in a fresh interpreter that
the plugin's shim is registered with its transforms.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).resolve().parent / "contracts_plugin"
SRC_ROOTS = (
    ROOT / "packages" / "llm-rosetta" / "src",
    ROOT / "packages" / "llm-rosetta-gateway" / "src",
)


def _install_plugin(site: Path, target: str = "register_shims") -> None:
    shutil.copytree(FIXTURE / "c1657_contracts_plugin", site / "c1657_contracts_plugin")
    dist_info = site / "c1657_contracts_plugin-0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: c1657-contracts-plugin\nVersion: 0.0\n",
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        f"[llm_rosetta.shim_providers]\nc1657acme = c1657_contracts_plugin:{target}\n",
        encoding="utf-8",
    )


def _probe(tmp_path: Path, target: str = "register_shims") -> dict:
    site = tmp_path / "site"
    site.mkdir()
    _install_plugin(site, target)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in (site, *SRC_ROOTS))
    proc = subprocess.run(
        [sys.executable, str(FIXTURE / "probe.py")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_a_plugin_importing_every_contract_registers_during_import(tmp_path):
    result = _probe(tmp_path)
    assert Path(result["core"]).is_relative_to(SRC_ROOTS[0])
    assert result["warnings"] == []
    assert result["registered"]
    assert result["pre"][0] == "c1657_keep"
    assert result["ir"] == 1


def test_a_failing_entry_point_is_a_warning_naming_the_plugin_and_error(tmp_path):
    result = _probe(tmp_path, target="no_such_entry_point")
    assert not result["registered"]
    assert len(result["warnings"]) == 1
    assert "c1657acme" in result["warnings"][0]
    assert "AttributeError" in result["warnings"][0]


def test_a_shim_package_built_for_an_older_core_names_itself(tmp_path, monkeypatch):
    import logging

    from llm_rosetta.contracts.shim import load_providers_from_dir

    site = tmp_path / "site"
    leaf = site / "c1657_old_shims" / "providers" / "c1657old" / "llm"
    leaf.mkdir(parents=True)
    (site / "c1657_old_shims" / "__init__.py").write_text("", encoding="utf-8")
    yaml_text = (
        FIXTURE / "c1657_contracts_plugin/providers/c1657acme/llm/provider.yaml"
    ).read_text(encoding="utf-8")
    (leaf / "provider.yaml").write_text(
        yaml_text.replace("c1657acme", "c1657old"), encoding="utf-8"
    )
    (leaf / "transforms.py").write_text(
        "from llm_rosetta.shims.auth_exchange import AuthContext\n", encoding="utf-8"
    )
    dist_info = site / "c1657_old_shims-1.2.3.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: c1657-old-shims\nVersion: 1.2.3\n",
        encoding="utf-8",
    )
    (dist_info / "top_level.txt").write_text("c1657_old_shims\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(site))

    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Collect(logging.WARNING)
    logger = logging.getLogger("llm_rosetta.shims.providers")
    logger.addHandler(handler)
    try:
        registered = load_providers_from_dir(
            site / "c1657_old_shims" / "providers", on_error="warn"
        )
    finally:
        logger.removeHandler(handler)

    assert registered == []
    messages = [r.getMessage() for r in records if "c1657old" in r.getMessage()]
    assert len(messages) == 1
    message = messages[0]
    assert "c1657-old-shims 1.2.3" in message
    assert "built for an older llm-rosetta" in message
    assert "llm_rosetta.shims.auth_exchange" in message
    assert "upgrade the shim package" in message
