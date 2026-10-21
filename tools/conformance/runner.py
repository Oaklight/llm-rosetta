"""Run probes with bounded concurrency, a timeout and retries."""

from __future__ import annotations

import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any
from collections.abc import Iterable, Sequence

from .model import Probe, ProbeFailure, ProbeResult, ProbeSkip, Status, SuiteReport
from .redact import Redactor


def _run_with_timeout(probe: Probe, timeout: float | None) -> dict[str, Any] | None:
    """Run one probe, giving up after *timeout* seconds.

    A timed-out probe runs in a daemon thread that cannot be killed; the
    probe's own HTTP client should carry the real deadline.  This is a
    backstop so one hung case cannot stall the suite.
    """
    if timeout is None:
        return probe.run()

    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = probe.run()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"probe did not finish within {timeout:g}s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _run_one(
    probe: Probe, timeout: float | None, retries: int, redactor: Redactor
) -> ProbeResult:
    started = time.perf_counter()
    detail = ""
    for attempt in range(retries + 1):
        try:
            data = _run_with_timeout(probe, timeout)
        except ProbeSkip as exc:
            return ProbeResult(
                probe.name, Status.SKIP, _elapsed_ms(started), redactor.text(str(exc))
            )
        except ProbeFailure as exc:
            return ProbeResult(
                probe.name, Status.FAIL, _elapsed_ms(started), redactor.text(str(exc))
            )
        except Exception as exc:  # noqa: BLE001 - reported as an error case
            detail = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(min(2**attempt, 5))
                continue
            return ProbeResult(
                probe.name, Status.ERROR, _elapsed_ms(started), redactor.text(detail)
            )
        return ProbeResult(
            probe.name,
            Status.PASS,
            _elapsed_ms(started),
            data=redactor.value(data or {}),
        )
    raise AssertionError("unreachable")


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def run_suite(
    suite: str,
    probes: Sequence[Probe],
    *,
    timeout: float | None = 30.0,
    retries: int = 0,
    max_workers: int | None = None,
    environment: dict[str, Any] | None = None,
    secrets: Iterable[str] = (),
) -> SuiteReport:
    """Run every probe and return a report.

    One failing or erroring case never aborts the others.
    """
    redactor = Redactor(secrets)
    probes = list(probes)
    if probes:
        workers = max_workers or min(8, len(probes))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_run_one, probe, timeout, retries, redactor)
                for probe in probes
            ]
            results = [future.result() for future in as_completed(futures)]
    else:
        results = []
    results.sort(key=lambda result: result.case)

    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    env.update(environment or {})

    return SuiteReport(
        suite=suite,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        environment=redactor.value(env),
        results=results,
    )
