"""The SDK pins and the versions stamped into the ledgers must agree.

A ledger is only comparable against the SDKs that produced it, so a bump that
lands in one place and not the other has to fail here rather than read like a
conversion regression.
"""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import pytest

from tools.conformance.pins import mismatch, read_pins

_ROOT = Path(__file__).resolve().parents[2] / "tools" / "conformance"
_LEDGERS = ("request-fields.json", "response-fields.json", "sdk-types.json")


def _installed(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


# `dev-extras-test` installs the `[dev]` extra alone, so the SDKs are absent
# there; only the tests that read the live environment are skipped.
_requires_sdks = pytest.mark.skipif(
    any(_installed(name) is None for name in read_pins()),
    reason="SDKs are not installed",
)


@pytest.mark.parametrize("ledger", _LEDGERS)
def test_ledger_stamp_matches_pins(ledger: str) -> None:
    stamped = json.loads((_ROOT / "baselines" / ledger).read_text())["sdk_versions"]
    assert stamped == read_pins(), (
        f"{ledger} was generated against {stamped}, but sdk-pins.txt pins "
        f"{read_pins()}; bump both together"
    )


def test_mismatch_reports_an_absent_distribution() -> None:
    differing = mismatch({"definitely-not-a-distribution": "1.0.0"})
    assert differing == {"definitely-not-a-distribution": ("1.0.0", "")}


def test_mismatch_reports_the_installed_version() -> None:
    pinned, installed = mismatch({"pytest": "0.0.0"})["pytest"]
    assert (pinned, installed) == ("0.0.0", importlib.metadata.version("pytest"))


@_requires_sdks
def test_mismatch_is_empty_when_versions_match() -> None:
    pins: dict[str, str] = {}
    for name in read_pins():
        version = _installed(name)
        assert version is not None
        pins[name] = version
    assert mismatch(pins) == {}
