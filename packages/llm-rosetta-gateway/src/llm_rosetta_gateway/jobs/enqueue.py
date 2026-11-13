"""From an admin request to a job and the ranges it hides.

Two rules live here, and nowhere else:

* **Enqueue is the moment the operator's view changes.**  The hidden ranges are
  computed from the same bounds the handler will delete by, so "what I can see"
  and "what the job is about to remove" cannot drift while the job waits.
* **The payload carries those bounds.**  A cleanup computes its cutoff once,
  here, and the handler uses the payload's value; recomputing it at run time
  would hide up to the enqueue time and delete up to the run time, leaving rows
  invisible in between.

Dedupe keys are per operation: a second click while the first is queued returns
the running job instead of stacking another one.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from .model import (
    PRIORITY_DEFAULT,
    PRIORITY_INTERACTIVE,
    PRIORITY_MAINTENANCE,
    Job,
)

CLEAR_KIND = "data.clear"
CLEANUP_KIND = "data.cleanup"
TRIM_KIND = "data.trim"
VACUUM_KIND = "data.vacuum"
REBUILD_KIND = "data.rebuild"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _age_cutoff(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


async def enqueue_clear(manager: Any, table: str, *, source: str = "admin") -> Job:
    """Hide *table* completely and queue its deletion."""
    return await manager.enqueue(
        CLEAR_KIND,
        {"table": table, "source": source},
        priority=PRIORITY_INTERACTIVE,
        dedupe_key=f"clear:{table}",
        hidden_ranges=[(table, "", _now())],
    )


async def enqueue_cleanup(
    manager: Any,
    *,
    mode: str,
    tables: tuple[str, ...],
    max_age_days: int | None = None,
    before: str | None = None,
    start: str | None = None,
    end: str | None = None,
    source: str = "admin",
) -> Job:
    """Hide the rows a cleanup will remove and queue the removal.

    ``mode`` is ``age`` (older than *max_age_days*), ``before`` (older than
    *before*) or ``range`` (``[start, end)``).
    """
    payload: dict[str, Any] = {
        "mode": mode,
        "tables": list(tables),
        "source": source,
    }
    if mode == "age":
        days = max_age_days or 90
        cutoff = _age_cutoff(days)
        payload.update(max_age_days=days, cutoff=cutoff)
        ranges = [(table, "", cutoff) for table in tables]
        key = f"cleanup:age:{days}:{','.join(tables)}"
    elif mode == "before":
        payload["before"] = before
        ranges = [(table, "", before or "") for table in tables]
        key = f"cleanup:before:{before}:{','.join(tables)}"
    elif mode == "range":
        payload.update(start=start, end=end)
        ranges = [(table, start or "", end or "") for table in tables]
        key = f"cleanup:range:{start}:{end}:{','.join(tables)}"
    else:
        raise ValueError(f"unknown cleanup mode {mode!r}")

    return await manager.enqueue(
        CLEANUP_KIND,
        payload,
        priority=PRIORITY_INTERACTIVE,
        dedupe_key=key,
        hidden_ranges=ranges,
    )


async def enqueue_trim(manager: Any, *, source: str = "admin") -> Job:
    """Queue a trim to the retention caps.

    No range is hidden: the target set is not one time window — ops_log trims
    informational and warning rows to two different caps — and the operation
    is disk maintenance rather than a data loss the operator asked for.
    """
    return await manager.enqueue(
        TRIM_KIND,
        {"source": source},
        priority=PRIORITY_DEFAULT,
        dedupe_key="trim",
    )


async def enqueue_vacuum(manager: Any, *, source: str = "admin") -> Job:
    """Queue a VACUUM.  It cannot be preempted once started, so it runs last."""
    return await manager.enqueue(
        VACUUM_KIND,
        {"source": source},
        priority=PRIORITY_MAINTENANCE,
        dedupe_key="vacuum",
    )


async def enqueue_rebuild(manager: Any, *, source: str = "admin") -> Job:
    """Queue a counter rebuild from the visible log."""
    return await manager.enqueue(
        REBUILD_KIND,
        {"source": source},
        priority=PRIORITY_MAINTENANCE,
        dedupe_key="rebuild",
    )
