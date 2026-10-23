"""Suite registry consumed by the CLI."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..model import Probe
from . import drift, wire

SuiteFactory = Callable[[], tuple[list[Probe], dict[str, Any]]]

SUITES: dict[str, SuiteFactory] = {
    "drift": drift.build,
    "wire": wire.build,
}

__all__ = ["SUITES"]
