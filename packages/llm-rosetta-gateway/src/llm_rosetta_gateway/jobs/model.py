"""The job envelope: what a background job is, and the states it moves through.

One table holds every kind of background work — a data cleanup today, a Batch
API submission later (#1509) — because the lifecycle is the same, and a second
envelope would be a second set of claim, lease, cancel and recovery rules to
keep in step.

Two properties are load-bearing and worth stating here rather than in the
store:

* A job is **durable**.  It is a row before it runs, so a restart neither
  forgets a queued job nor loses the record of a finished one.
* A job is **claimed under a lease**.  One process runs one lane for now; the
  lease is what would make a second process safe without a schema change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_SUCCEEDED = "succeeded"
JOB_FAILED = "failed"
JOB_CANCELLED = "cancelled"

#: States in which a job still has work ahead of it.
ACTIVE_STATES = frozenset({JOB_QUEUED, JOB_RUNNING})

#: States a job never leaves on its own.
TERMINAL_STATES = frozenset({JOB_SUCCEEDED, JOB_FAILED, JOB_CANCELLED})

ALL_STATES = (JOB_QUEUED, JOB_RUNNING, JOB_SUCCEEDED, JOB_FAILED, JOB_CANCELLED)

#: A lane serialises the jobs in it.  ``sqlite`` work shares the one database
#: writer, so more than one worker there cannot add throughput; ``network`` is
#: reserved for Batch, whose jobs wait on an upstream rather than on the disk.
LANE_SQLITE = "sqlite"
LANE_NETWORK = "network"

#: Where an interactive admin job sits relative to background maintenance.
#: Lower runs first.  A long VACUUM is queued last so it cannot starve a clear.
PRIORITY_INTERACTIVE = 10
PRIORITY_DEFAULT = 100
PRIORITY_MAINTENANCE = 200

DEFAULT_LEASE_SECONDS = 60.0

PAYLOAD_VERSION = 1


def now_iso() -> str:
    """The current UTC time, in the format every stored timestamp uses."""
    return datetime.now(timezone.utc).isoformat()


def iso_after(seconds: float) -> str:
    """A UTC timestamp *seconds* from now."""
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


class JobCancelled(Exception):
    """Raised at a checkpoint after the operator asked the job to stop."""


class LeaseLost(Exception):
    """The job's lease is no longer ours; the handler must stop.

    Not a terminal state: the worker records nothing, because another process
    (or recovery) owns the row now.  Writing a result without the lease would
    resurrect a job that was reclaimed.
    """


@dataclass(slots=True)
class Job:
    """One row of the ``jobs`` table."""

    id: str
    kind: str
    payload: dict[str, Any]
    lane: str
    status: str
    priority: int
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    attempts: int = 0
    max_attempts: int = 3
    progress_done: int = 0
    progress_total: int | None = None
    progress_unit: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    hidden_range_ids: list[str] = field(default_factory=list)
    dedupe_key: str | None = None
    cancel_requested: bool = False

    @property
    def active(self) -> bool:
        """True while the job is still queued or running."""
        return self.status in ACTIVE_STATES

    def to_payload(self) -> dict[str, Any]:
        """The admin API's view of the job."""
        return {
            "id": self.id,
            "kind": self.kind,
            "lane": self.lane,
            "status": self.status,
            "priority": self.priority,
            "payload": self.payload,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "progress_done": self.progress_done,
            "progress_total": self.progress_total,
            "progress_unit": self.progress_unit,
            "result": self.result,
            "error": self.error,
            "cancel_requested": self.cancel_requested,
        }
