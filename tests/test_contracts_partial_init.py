"""Every contract module imports while ``llm_rosetta`` is still initialising (#1657).

Extensions load during ``import llm_rosetta``: ``shims/__init__`` registers them.
An extension whose sidecar imports a name the half-initialised package does not
have yet (``llm_rosetta.api``, ``__version__``) fails, and the loader only logs
it, so its shims silently never register while every in-process test stays
green.

This test puts a real extension folder (``llm_rosetta_ext/c1657acme`` in
``tests/contracts_plugin``) on ``sys.path``; its sidecars import every contract
module.  A fresh interpreter then checks that the shim is registered with its
transforms.
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


def _install_plugin(site: Path, broken: bool = False) -> None:
    shutil.copytree(
        FIXTURE / "ext_src" / "llm_rosetta_ext" / "c1657acme",
        site / "llm_rosetta_ext" / "c1657acme",
    )
    if broken:
        (site / "llm_rosetta_ext" / "c1657acme" / "llm" / "transforms.py").write_text(
            "raise AttributeError('no_such_name')\n", encoding="utf-8"
        )


def _probe(tmp_path: Path, broken: bool = False) -> dict:
    site = tmp_path / "site"
    site.mkdir()
    _install_plugin(site, broken)
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


def test_an_extension_importing_every_contract_registers_during_import(tmp_path):
    result = _probe(tmp_path)
    assert Path(result["core"]).is_relative_to(SRC_ROOTS[0])
    assert result["warnings"] == []
    assert result["registered"]
    assert result["pre"][0] == "c1657_keep"
    assert result["ir"] == 1


def test_a_failing_extension_is_a_warning_naming_it_and_the_error(tmp_path):
    result = _probe(tmp_path, broken=True)
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
        FIXTURE / "ext_src/llm_rosetta_ext/c1657acme/llm/provider.yaml"
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
