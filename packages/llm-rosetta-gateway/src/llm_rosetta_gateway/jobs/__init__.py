"""Background jobs: a durable envelope, its store, and the worker that runs it.

A destructive admin operation records what it wants removed and returns; the
worker does the work in batches, reports progress, and stops when asked.  The
same envelope will carry Batch API submissions (#1509), which is why nothing
here knows what a job *does* — that lives in the registry.
"""

from .model import (
    ACTIVE_STATES,
    ALL_STATES,
    JOB_CANCELLED,
    JOB_FAILED,
    JOB_QUEUED,
    JOB_RUNNING,
    JOB_SUCCEEDED,
    LANE_NETWORK,
    LANE_SQLITE,
    PRIORITY_DEFAULT,
    PRIORITY_INTERACTIVE,
    PRIORITY_MAINTENANCE,
    TERMINAL_STATES,
    Job,
    JobCancelled,
    LeaseLost,
)
from .registry import Handler, JobRegistry, UnknownJobKind, default_handlers
from .store import JobStore
from .worker import JobContext, JobManager

__all__ = [
    "ACTIVE_STATES",
    "ALL_STATES",
    "Handler",
    "JOB_CANCELLED",
    "JOB_FAILED",
    "JOB_QUEUED",
    "JOB_RUNNING",
    "JOB_SUCCEEDED",
    "Job",
    "JobCancelled",
    "JobContext",
    "JobManager",
    "JobRegistry",
    "LeaseLost",
    "JobStore",
    "LANE_NETWORK",
    "LANE_SQLITE",
    "PRIORITY_DEFAULT",
    "PRIORITY_INTERACTIVE",
    "PRIORITY_MAINTENANCE",
    "TERMINAL_STATES",
    "UnknownJobKind",
    "default_handlers",
]
