"""The gateway reports both versions it runs (#1127).

The gateway and the core are released together but may drift by a patch
when a gateway-only fix ships.  `--version`, `/health` and the admin payload
therefore name both, so an operator can confirm the pair.
"""

from __future__ import annotations

import json
import subprocess
import sys

import llm_rosetta
import llm_rosetta_gateway

from llm_rosetta_gateway.admin.routes.config.payload import (
    _get_core_version,
    _get_version,
)
from llm_rosetta_gateway.app.routes_health import version_info


def test_version_info_names_both_packages() -> None:
    assert version_info() == {
        "gateway": llm_rosetta_gateway.__version__,
        "core": llm_rosetta.__version__,
    }


def test_the_cli_prints_both_versions() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "llm_rosetta_gateway", "--version"],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert llm_rosetta_gateway.__version__ in result.stdout
    assert f"llm-rosetta {llm_rosetta.__version__}" in result.stdout


def test_the_admin_payload_versions_are_the_two_packages() -> None:
    assert _get_version() == llm_rosetta_gateway.__version__
    assert _get_core_version() == llm_rosetta.__version__


def test_health_carries_the_version_pair() -> None:
    from types import SimpleNamespace

    from llm_rosetta_gateway.app import handle_health

    import asyncio

    response = asyncio.run(
        handle_health(SimpleNamespace(app=SimpleNamespace(metrics=None)))
    )
    body = json.loads(response.body)

    assert body["versions"]["gateway"] == llm_rosetta_gateway.__version__
    assert body["versions"]["core"] == llm_rosetta.__version__
