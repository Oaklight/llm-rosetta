"""The SDK versions the committed baselines were generated against.

A baseline is only comparable against the SDKs that produced it.  CI installs
with ``-c tools/conformance/sdk-pins.txt``; a local environment that resolved
some other version is told so rather than left with a misleading failure.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

PINS_PATH = Path(__file__).with_name("sdk-pins.txt")


def read_pins(path: Path = PINS_PATH) -> dict[str, str]:
    """Return ``{distribution: version}`` from the pins file."""
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, _, version = line.partition("==")
            pins[name.strip()] = version.strip()
    return pins


def mismatch(pins: dict[str, str] | None = None) -> dict[str, tuple[str, str]]:
    """Return ``{distribution: (pinned, installed)}`` for every difference.

    A distribution that is not installed at all is reported with ``""`` as the
    installed version.
    """
    if pins is None:
        pins = read_pins()
    differing: dict[str, tuple[str, str]] = {}
    for name, pinned in pins.items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = ""
        if installed != pinned:
            differing[name] = (pinned, installed)
    return differing


def describe(differing: dict[str, tuple[str, str]]) -> str:
    """One line naming each mismatch and the install line that fixes it."""
    parts = [
        f"{name} {installed or 'not installed'} (pinned {pinned})"
        for name, (pinned, installed) in sorted(differing.items())
    ]
    return (
        "installed SDKs differ from tools/conformance/sdk-pins.txt: "
        + ", ".join(parts)
        + "; reinstall with `pip install -c tools/conformance/sdk-pins.txt ...`"
    )
