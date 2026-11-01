"""Suite registry consumed by the CLI."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..model import Probe
from . import drift, fields, request_fields, stream_events, wire

SuiteFactory = Callable[[], tuple[list[Probe], dict[str, Any]]]

SUITES: dict[str, SuiteFactory] = {
    "drift": drift.build,
    "fields": fields.build,
    "request-fields": request_fields.build,
    "stream-events": stream_events.build,
    "wire": wire.build,
}

__all__ = ["SUITES"]
