"""Conformance probes — run a suite, write one versioned report.

This is a developer tool, not a library dependency: it lives outside
the ``packages/`` trees so the wheel does not ship it.  The first suite wraps the
SDK type scan; wire and live-provider suites follow.

Entry point::

    python -m tools.conformance run --suite drift --out report.json
"""

from .model import SCHEMA_VERSION, ProbeFailure, ProbeResult, Status, SuiteReport
from .runner import run_suite

__all__ = [
    "SCHEMA_VERSION",
    "ProbeFailure",
    "ProbeResult",
    "Status",
    "SuiteReport",
    "run_suite",
]
