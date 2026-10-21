"""Report types shared by the runner, the CLI and the suites."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

#: Bump when a consumer would have to change to read the report.
SCHEMA_VERSION = 1


class Status(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"


class ProbeFailure(AssertionError):
    """The probe ran and the observed result is wrong.

    Any other exception means the probe could not run and is reported as
    ``error``, so a broken environment is never mistaken for a regression.
    """


class ProbeSkip(Exception):
    """The probe cannot run here (missing baseline, changed SDK version)."""


class Probe(Protocol):
    name: str

    def run(self) -> dict[str, Any] | None:
        """Run the probe, returning optional data for the report."""
        ...


@dataclass(frozen=True)
class ProbeResult:
    case: str
    status: Status
    duration_ms: int
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "detail": self.detail,
            "data": self.data,
        }


@dataclass
class SuiteReport:
    suite: str
    generated_at: str
    environment: dict[str, Any]
    results: list[ProbeResult]
    schema_version: int = SCHEMA_VERSION

    @property
    def summary(self) -> dict[str, int]:
        counts = {status.value: 0 for status in Status}
        for result in self.results:
            counts[result.status.value] += 1
        return counts

    @property
    def ok(self) -> bool:
        return not any(
            result.status in (Status.FAIL, Status.ERROR) for result in self.results
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "suite": self.suite,
            "generated_at": self.generated_at,
            "environment": self.environment,
            "results": [result.to_dict() for result in self.results],
            "summary": self.summary,
        }
