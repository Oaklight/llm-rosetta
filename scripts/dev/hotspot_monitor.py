#!/usr/bin/env python3
"""Host-side hotspot monitor for the LLM-Rosetta gateway container.

Samples ``/proc`` at 1 Hz, runs short ``py-spy`` windows, and reads the
gateway's request log to separate conversion CPU from upstream wait.  Results
are written as a rotating JSONL series and streamed over a Unix socket.

Run on the Docker host, not inside the container::

    python3 hotspot_monitor.py --container llm-rosetta-llm-rosetta-gateway-1

The companion ``hotspot_watch.py`` is the client.  The socket protocol is
newline-delimited JSON:

* server to client: ``hello``, ``snapshot``, ``sample``, ``profile``, ``pong``
* client to server: ``{"cmd": "snapshot", "minutes": 5}``,
  ``{"cmd": "profile", "seconds": 30}``, ``{"cmd": "ping"}``,
  ``{"cmd": "stop"}``.

The monitor never writes to the gateway.  It reads ``/proc/<pid>``, the
gateway's SQLite log in read-only mode, and attaches ``py-spy`` to the
container process for a bounded window.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import queue
import select
import shutil
import signal
import socketserver
import sqlite3
import stat
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, TextIO
from urllib.parse import quote

logger = logging.getLogger("hotspot-monitor")

CLK_TCK = os.sysconf("SC_CLK_TCK")
_MAX_CLIENT_BUFFER = 64 * 1024

# Leaves that mean "waiting", not "computing".  py-spy includes the main
# thread while it sits in the selector, and thread-pool workers while they
# wait on their queue; neither is a hotspot.
_IDLE_LEAVES = frozenset(
    {
        "EpollSelector.select",
        "select",
        "epoll_wait",
        "poll",
        "kqueue.control",
        "_worker",
        "Connection._worker",
    }
)

_DEFAULT_SOCKET = "/run/llm-rosetta-hotspot/hotspot.sock"
_DEFAULT_LOG_DIR = "/var/log/llm-rosetta-hotspot"
_DEFAULT_DB = "/dockervol/llm-rosetta/data/gateway.db"
_DEFAULT_CONTAINER = "llm-rosetta-llm-rosetta-gateway-1"


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _round(value: float | None) -> float | None:
    """Round a float to one decimal place, preserving ``None``."""
    return None if value is None else round(value, 1)


def _unlink(path: str) -> None:
    """Remove a file, ignoring a missing file."""
    try:
        os.unlink(path)
    except OSError:
        pass


def read_text(path: str) -> str:
    """Read a text file from procfs."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


# ---------------------------------------------------------------------------
# /proc parsing
# ---------------------------------------------------------------------------


def parse_proc_stat(text: str) -> dict[str, Any]:
    """Parse ``/proc/<pid>/stat``.

    The ``comm`` field may contain spaces and parentheses, so the split
    starts after the *last* ``)``.  Field numbers below are 1-based as in
    ``proc(5)``; ``fields`` is 0-based from field 3.
    """
    left = text.find("(")
    right = text.rfind(")")
    if left < 0 or right < 0:
        raise ValueError("malformed /proc stat line")
    comm = text[left + 1 : right]
    fields = text[right + 2 :].split()
    if len(fields) < 20:
        raise ValueError("short /proc stat line")
    return {
        "comm": comm,
        "state": fields[0],
        "utime": int(fields[11]),
        "stime": int(fields[12]),
        "num_threads": int(fields[17]),
        "starttime": int(fields[19]),
    }


def parse_status(text: str) -> dict[str, str]:
    """Parse the fields of ``/proc/<pid>/status`` the monitor needs."""
    wanted = {"VmRSS", "VmHWM", "VmSize", "Threads"}
    out: dict[str, str] = {}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key in wanted:
            out[key] = value.strip()
    return out


def _kb_to_mb(value: str | None) -> float | None:
    """Convert a ``"1234 kB"`` status value to MiB."""
    if not value:
        return None
    parts = value.split()
    if not parts or not parts[0].isdigit():
        return None
    return round(int(parts[0]) / 1024.0, 1)


def parse_smaps_rollup(text: str) -> dict[str, float]:
    """Parse ``/proc/<pid>/smaps_rollup`` into MiB values."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[-1] == "kB":
            try:
                out[parts[0].rstrip(":")] = int(parts[1]) / 1024.0
            except (ValueError, IndexError):
                continue
    return out


def parse_io(text: str) -> dict[str, int]:
    """Parse ``/proc/<pid>/io``."""
    out: dict[str, int] = {}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        value = value.strip()
        if value.isdigit():
            out[key.strip()] = int(value)
    return out


def read_threads(pid: int) -> dict[int, dict[str, Any]]:
    """Read the stat line of every thread under ``/proc/<pid>/task``."""
    out: dict[int, dict[str, Any]] = {}
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return out
    for tid in tids:
        try:
            out[int(tid)] = parse_proc_stat(read_text(f"/proc/{pid}/task/{tid}/stat"))
        except (OSError, ValueError):
            continue
    return out


def count_fds(pid: int) -> int | None:
    """Count open file descriptors for *pid*."""
    try:
        return len(os.listdir(f"/proc/{pid}/fd"))
    except OSError:
        return None


def discover_pid(container: str) -> int:
    """Return the host PID of the container's main process."""
    try:
        proc = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Pid}}", container],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"docker inspect failed for {container}: {exc}") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip()
        raise RuntimeError(f"docker inspect failed for {container}: {detail}")
    pid = int(proc.stdout.strip())
    if pid <= 0:
        raise RuntimeError(f"container {container} is not running")
    return pid


class PidTracker:
    """Resolve the container PID, and re-resolve it after a restart."""

    def __init__(self, container: str, pid: int | None = None) -> None:
        self._container = container
        self._pid = pid
        self._lock = threading.Lock()

    def get(self) -> int | None:
        """Return a live PID, or ``None`` while the container is down."""
        with self._lock:
            if self._pid is not None and os.path.exists(f"/proc/{self._pid}"):
                return self._pid
            self._pid = discover_pid(self._container)
            return self._pid


def _size_to_mb(value: str) -> float | None:
    """Convert a Docker size such as ``"656.1MiB"`` to MiB."""
    for unit, factor in (
        ("TiB", 1048576.0),
        ("GiB", 1024.0),
        ("MiB", 1.0),
        ("KiB", 1.0 / 1024.0),
        ("B", 1.0 / 1048576.0),
    ):
        if value.endswith(unit):
            try:
                return round(float(value[: -len(unit)]) * factor, 1)
            except ValueError:
                return None
    return None


def read_docker_stats(container: str) -> dict[str, Any] | None:
    """Read the container's CPU and memory from ``docker stats``.

    The memory figure includes page cache; the process RSS comes from
    ``/proc``.  Both are reported so they are not confused.
    """
    try:
        proc = subprocess.run(
            [
                "docker",
                "stats",
                "--no-stream",
                "--format",
                "{{.CPUPerc}}|{{.MemUsage}}|{{.MemPerc}}",
                container,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("docker stats failed: %s", exc)
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    line = proc.stdout.strip().splitlines()[0]
    try:
        cpu, memory, memory_pct = line.split("|")
        used = memory.split("/")[0].strip()
        return {
            "cpu_pct": float(cpu.rstrip("%")),
            "mem_used_mb": _size_to_mb(used),
            "mem_pct": float(memory_pct.rstrip("%")),
        }
    except (ValueError, IndexError):
        return None


def percentile(values: list[float], fraction: float) -> float | None:
    """Return a linearly interpolated percentile of *values*."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


# ---------------------------------------------------------------------------
# Request-log correlation
# ---------------------------------------------------------------------------


def read_db(
    db_path: str, window_seconds: int = 60, limit: int = 5000
) -> dict[str, Any] | None:
    """Summarize the last *window_seconds* of ``request_log`` rows.

    The profile column holds the per-request phase timings written by the
    gateway.  ``request_conversion_ms`` is conversion CPU; ``upstream_ms``
    (or the stream connect plus duration) is upstream wait and is reported
    separately, never as a hotspot.
    """
    if not db_path or not os.path.exists(db_path):
        return None
    cutoff = (
        datetime.now(timezone.utc) - timedelta(seconds=window_seconds)
    ).isoformat()
    try:
        uri = f"file:{quote(db_path, safe='/')}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=2.0)
        try:
            rows = connection.execute(
                "SELECT source_provider, target_provider, is_stream, profile "
                "FROM request_log WHERE timestamp >= ? "
                "ORDER BY timestamp DESC LIMIT ?",
                (cutoff, limit),
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        logger.debug("request_log read failed: %s", exc)
        return None

    conversion: list[float] = []
    upstream: list[float] = []
    ttfb: list[float] = []
    pairs: dict[str, list[float]] = {}
    streams = 0
    for source, target, is_stream, raw in rows:
        if is_stream:
            streams += 1
        try:
            profile = json.loads(raw) if raw else {}
        except (TypeError, ValueError):
            profile = {}
        if not isinstance(profile, dict):
            profile = {}

        value = profile.get("request_conversion_ms")
        if isinstance(value, (int, float)):
            conversion.append(float(value))
            pair = f"{source or '?'}->{target or '?'}"
            pairs.setdefault(pair, []).append(float(value))

        value = profile.get("upstream_ms")
        if not isinstance(value, (int, float)):
            connect = profile.get("stream_connect_ms")
            duration = profile.get("stream_duration_ms")
            if isinstance(connect, (int, float)) or isinstance(duration, (int, float)):
                value = (connect or 0.0) + (duration or 0.0)
        if isinstance(value, (int, float)):
            upstream.append(float(value))

        value = profile.get("ttfb_ms")
        if isinstance(value, (int, float)):
            ttfb.append(float(value))

    slow_pairs = [
        {
            "pair": pair,
            "count": len(values),
            "conversion_p95_ms": round(percentile(values, 0.95) or 0.0, 1),
        }
        for pair, values in pairs.items()
        if len(values) >= 3
    ]
    slow_pairs.sort(key=lambda item: item["conversion_p95_ms"], reverse=True)

    return {
        "window_s": window_seconds,
        "requests": len(rows),
        "streams": streams,
        "rps": round(len(rows) / window_seconds, 2),
        "stream_pct": round(streams / len(rows), 3) if rows else 0.0,
        "conversion_p50_ms": _round(percentile(conversion, 0.5)),
        "conversion_p95_ms": _round(percentile(conversion, 0.95)),
        "upstream_p95_ms": _round(percentile(upstream, 0.95)),
        "ttfb_p95_ms": _round(percentile(ttfb, 0.95)),
        "truncated": len(rows) >= limit,
        "slow_pairs": slow_pairs[:3],
    }


# ---------------------------------------------------------------------------
# py-spy / speedscope
# ---------------------------------------------------------------------------


def is_idle_leaf(name: str) -> bool:
    """Return whether a leaf frame is a wait, not CPU work."""
    return name in _IDLE_LEAVES


def _short_file(path: str) -> str:
    """Return the basename of a frame's file."""
    return path.rsplit("/", 1)[-1] if path else ""


def summarize_speedscope(data: dict[str, Any], top_n: int = 10) -> dict[str, Any]:
    """Aggregate a speedscope file into per-thread and top-function counts.

    Idle leaves are excluded from the top list and from the active count.
    When a ``MainThread`` profile exists, the top list describes that
    thread; otherwise it merges every thread's non-idle leaves.

    ``pct`` is a share of non-idle samples, not CPU time: a wait in a leaf
    that is not listed in ``_IDLE_LEAVES`` counts as active.
    """
    frames = data.get("shared", {}).get("frames") or []
    profiles = data.get("profiles") or []
    thread_stats: dict[str, dict[str, int]] = {}
    per_profile: list[tuple[str, collections.Counter]] = []

    for profile in profiles:
        name = str(profile.get("name") or "?")
        counter: collections.Counter = collections.Counter()
        total = 0
        active = 0
        for sample in profile.get("samples") or []:
            if not sample:
                continue
            try:
                frame = frames[sample[-1]]
            except (IndexError, TypeError):
                continue
            function = str(frame.get("name") or "?")
            counter[
                (
                    function,
                    _short_file(str(frame.get("file") or "")),
                    frame.get("line"),
                )
            ] += 1
            total += 1
            if not is_idle_leaf(function):
                active += 1
        thread_stats[name] = {"samples": total, "active": active}
        per_profile.append((name, counter))

    main = next((item for item in per_profile if "MainThread" in item[0]), None)
    if main is not None:
        chosen = main[1]
    else:
        chosen = collections.Counter()
        for _, counter in per_profile:
            for key, value in counter.items():
                if not is_idle_leaf(key[0]):
                    chosen[key] += value

    active_total = sum(
        value for key, value in chosen.items() if not is_idle_leaf(key[0])
    )
    total_samples = sum(item["samples"] for item in thread_stats.values())
    if main is not None:
        chosen_total = sum(chosen.values())
    else:
        chosen_total = total_samples
    top: list[dict[str, Any]] = []
    for (function, file, line), value in chosen.most_common():
        if is_idle_leaf(function):
            continue
        top.append(
            {
                "fn": function,
                "file": file,
                "line": line,
                "self": value,
                "pct": round(
                    (100.0 * value / active_total) if active_total else 0.0,
                    1,
                ),
            }
        )
        if len(top) >= top_n:
            break

    return {
        "total_samples": total_samples,
        "active_samples": active_total,
        "idle_pct": (
            round(100.0 * (1.0 - active_total / chosen_total), 1)
            if chosen_total
            else 0.0
        ),
        "threads": thread_stats,
        "top": top,
    }


# ---------------------------------------------------------------------------
# Shared state and output
# ---------------------------------------------------------------------------


class MonitorState:
    """The sample ring buffer and the connected clients."""

    def __init__(self, interval: float, max_seconds: int = 3600) -> None:
        self._interval = max(0.2, interval)
        self._samples: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=max(1, int(max_seconds / self._interval))
        )
        self._profile: dict[str, Any] | None = None
        self._subscribers: list[queue.Queue] = []
        self._lock = threading.Lock()

    def add_sample(self, sample: dict[str, Any]) -> None:
        """Append a sample and broadcast it."""
        with self._lock:
            self._samples.append(sample)
            subscribers = list(self._subscribers)
        self._broadcast(subscribers, sample)

    def set_profile(self, profile: dict[str, Any]) -> None:
        """Store and broadcast the latest profile summary."""
        with self._lock:
            self._profile = profile
            subscribers = list(self._subscribers)
        self._broadcast(subscribers, profile)

    def latest_profile(self) -> dict[str, Any] | None:
        """Return the latest profile summary."""
        with self._lock:
            return dict(self._profile) if self._profile else None

    def snapshot(self, minutes: int) -> list[dict[str, Any]]:
        """Return the last *minutes* of samples."""
        count = max(1, int(minutes * 60 / self._interval))
        with self._lock:
            return list(self._samples)[-count:]

    def subscribe(self) -> queue.Queue:
        """Register a client queue."""
        client: queue.Queue = queue.Queue(maxsize=240)
        with self._lock:
            self._subscribers.append(client)
        return client

    def unsubscribe(self, client: queue.Queue) -> None:
        """Remove a client queue."""
        with self._lock:
            if client in self._subscribers:
                self._subscribers.remove(client)

    @staticmethod
    def _broadcast(subscribers: list[queue.Queue], message: dict[str, Any]) -> None:
        for client in subscribers:
            try:
                client.put_nowait(message)
            except queue.Full:
                try:
                    client.get_nowait()
                except queue.Empty:
                    pass
                try:
                    client.put_nowait(message)
                except queue.Full:
                    pass


class JsonlWriter:
    """Append JSON lines, rotating at a size limit."""

    def __init__(self, path: str, max_bytes: int = 20 * 1024 * 1024) -> None:
        self._path = path
        self._max_bytes = max_bytes
        self._size = 0
        self._handle: TextIO | None = None
        self._lock = threading.Lock()
        self._open()

    def _open(self) -> None:
        self._handle = open(self._path, "a", encoding="utf-8")
        try:
            self._size = os.path.getsize(self._path)
        except OSError:
            self._size = 0

    def write(self, record: dict[str, Any]) -> None:
        """Write one record and flush it."""
        line = json.dumps(record, separators=(",", ":")) + "\n"
        with self._lock:
            if self._handle is None:
                return
            if self._size + len(line) > self._max_bytes:
                self._handle.close()
                os.replace(self._path, self._path + ".1")
                self._open()
            self._handle.write(line)
            self._handle.flush()
            self._size += len(line)

    def close(self) -> None:
        """Close the current file."""
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class Sampler(threading.Thread):
    """Read ``/proc`` once per interval and publish a sample."""

    def __init__(
        self,
        tracker: PidTracker,
        state: MonitorState,
        writer: JsonlWriter | None,
        interval: float,
        db_path: str,
        container: str,
        container_stats: bool,
        stop: threading.Event,
    ) -> None:
        super().__init__(name="hotspot-sampler", daemon=True)
        self._tracker = tracker
        self._state = state
        self._writer = writer
        self._interval = max(0.2, interval)
        self._db_path = db_path
        self._container = container
        self._container_stats = container_stats
        self._stop_event = stop
        self._prev: dict[str, Any] | None = None
        self._last_db = 0.0
        self._last_docker = 0.0
        self._last_pid_warning = 0.0

    def run(self) -> None:
        """Sample until stopped."""
        while not self._stop_event.wait(self._interval):
            sample = self._sample()
            if sample is None:
                continue
            self._state.add_sample(sample)
            if self._writer is not None:
                try:
                    self._writer.write(sample)
                except OSError as exc:
                    logger.warning("cannot write series: %s", exc)

    def sample_once(self) -> dict[str, Any] | None:
        """Take one sample without publishing it."""
        return self._sample()

    def _sample(self) -> dict[str, Any] | None:
        try:
            pid = self._tracker.get()
        except RuntimeError as exc:
            now = time.monotonic()
            if now - self._last_pid_warning > 60.0:
                logger.warning("%s", exc)
                self._last_pid_warning = now
            self._prev = None
            return None
        if pid is None:
            return None

        mono = time.monotonic()
        try:
            proc_stat = parse_proc_stat(read_text(f"/proc/{pid}/stat"))
            status = parse_status(read_text(f"/proc/{pid}/status"))
            smaps = parse_smaps_rollup(read_text(f"/proc/{pid}/smaps_rollup"))
            io = parse_io(read_text(f"/proc/{pid}/io"))
            threads = read_threads(pid)
        except (OSError, ValueError, IndexError) as exc:
            logger.warning("cannot read /proc/%s: %s", pid, exc)
            self._prev = None
            return None

        total_jiffies = proc_stat["utime"] + proc_stat["stime"]
        elapsed = mono - self._prev["mono"] if self._prev else 0.0
        cores = 0.0
        main_cpu: float | None = None
        thread_cpu: dict[str, float] = {}
        if self._prev is not None and elapsed > 0:
            cores = (total_jiffies - self._prev["jiffies"]) / CLK_TCK / elapsed
            previous_threads = self._prev["threads"]
            for tid, thread in threads.items():
                previous = previous_threads.get(tid)
                if previous is None:
                    continue
                delta = (thread["utime"] + thread["stime"]) - (
                    previous["utime"] + previous["stime"]
                )
                percent = delta / CLK_TCK / elapsed * 100.0
                if tid == pid:
                    main_cpu = round(percent, 1)
                if percent >= 0.1:
                    thread_cpu[f"{tid} {thread['comm']}"] = round(percent, 1)
        thread_cpu = dict(
            sorted(thread_cpu.items(), key=lambda item: item[1], reverse=True)[:5]
        )

        db = None
        if self._db_path and (self._last_db == 0.0 or mono - self._last_db >= 10.0):
            db = read_db(self._db_path)
            self._last_db = mono

        container = None
        if self._container_stats and (
            self._last_docker == 0.0 or mono - self._last_docker >= 10.0
        ):
            container = read_docker_stats(self._container)
            self._last_docker = mono

        sample = {
            "type": "sample",
            "ts": _now_iso(),
            "pid": pid,
            "cpu_pct": round(cores * 100.0, 1),
            "cpu_cores": round(cores, 3),
            "main_cpu_pct": main_cpu,
            "rss_mb": _kb_to_mb(status.get("VmRSS")),
            "hwm_mb": _kb_to_mb(status.get("VmHWM")),
            "anon_mb": _round(smaps.get("Anonymous")),
            "shared_mb": _round(
                smaps.get("Shared_Clean", 0.0) + smaps.get("Shared_Dirty", 0.0)
            ),
            "threads": proc_stat["num_threads"],
            "fds": count_fds(pid),
            "io_read_mb": _round(io.get("rchar", 0) / 1048576.0),
            "io_write_mb": _round(io.get("wchar", 0) / 1048576.0),
            "thread_cpu": thread_cpu,
            "container": container,
            "db": db,
        }
        self._prev = {
            "mono": mono,
            "jiffies": total_jiffies,
            "threads": threads,
        }
        return sample


# ---------------------------------------------------------------------------
# py-spy runner
# ---------------------------------------------------------------------------


class ProfileRunner(threading.Thread):
    """Run bounded py-spy windows and publish a summary."""

    def __init__(
        self,
        tracker: PidTracker,
        state: MonitorState,
        py_spy: str,
        seconds: int,
        every: int,
        top_n: int,
        keep: int,
        log_dir: str,
        stop: threading.Event,
        cooldown: int = 60,
    ) -> None:
        super().__init__(name="hotspot-profiler", daemon=True)
        self._tracker = tracker
        self._state = state
        self._py_spy = py_spy
        self._seconds = max(5, seconds)
        self._every = max(0, every)
        self._top_n = max(1, top_n)
        self._keep = max(0, keep)
        self._log_dir = log_dir
        self._stop_event = stop
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._pending_seconds: int | None = None
        self._kept: collections.deque[str] = collections.deque()
        self._cooldown = max(0, cooldown)
        self._last_end = 0.0
        self._running = False

    def request(self, seconds: int | None = None) -> dict[str, Any]:
        """Request an immediate profile window.

        Returns an acknowledgement.  A request is refused while a window
        is running or inside the cooldown that follows one.
        """
        now = time.monotonic()
        with self._lock:
            if self._running:
                return {"accepted": False, "reason": "busy"}
            remaining = self._cooldown - (now - self._last_end)
            if self._last_end and self._cooldown > 0 and remaining > 0:
                return {
                    "accepted": False,
                    "reason": "cooldown",
                    "retry_after_s": int(remaining) + 1,
                }
            if seconds is not None:
                self._pending_seconds = max(5, min(int(seconds), 300))
            self._wake.set()
            return {"accepted": True}

    def stop(self) -> None:
        """Stop the runner."""
        self._stop_event.set()
        self._wake.set()

    def run(self) -> None:
        """Run automatic windows, or wait for manual requests."""
        delay: float | None = float(min(10, self._every)) if self._every > 0 else None
        while not self._stop_event.is_set():
            if delay is None:
                self._wake.wait()
            else:
                self._wake.wait(timeout=delay)
            self._wake.clear()
            if self._stop_event.is_set():
                break
            with self._lock:
                seconds = self._pending_seconds or self._seconds
                self._pending_seconds = None
            self._run_once(seconds)
            delay = float(self._every) if self._every > 0 else None

    def _run_once(self, seconds: int) -> None:
        with self._lock:
            self._running = True
        try:
            self._run_window(seconds)
        finally:
            with self._lock:
                self._running = False
                self._last_end = time.monotonic()

    def _run_window(self, seconds: int) -> None:
        try:
            pid = self._tracker.get()
        except RuntimeError as exc:
            logger.warning("profile skipped: %s", exc)
            return
        if pid is None:
            return
        if not os.path.exists(self._py_spy):
            logger.warning("py-spy not found at %s", self._py_spy)
            return

        handle = tempfile.NamedTemporaryFile(
            prefix="profile-",
            suffix=".json",
            dir=self._log_dir,
            delete=False,
        )
        path = handle.name
        handle.close()
        command = [
            self._py_spy,
            "record",
            "-d",
            str(seconds),
            "-r",
            "100",
            "--nonblocking",
            "-f",
            "speedscope",
            "-o",
            path,
            "-p",
            str(pid),
        ]
        # Kill a hung spy shortly after the window; the cooldown recovers
        # the busy state when subprocess.run raises TimeoutExpired.
        timeout = seconds + 15
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("py-spy failed: %s", exc)
            _unlink(path)
            return
        if proc.returncode != 0:
            detail = (proc.stderr or "").strip()[-500:]
            logger.warning("py-spy exited %s: %s", proc.returncode, detail)
            _unlink(path)
            return
        try:
            summary = summarize_speedscope(
                json.loads(read_text(path)), top_n=self._top_n
            )
        except (OSError, ValueError) as exc:
            logger.warning("speedscope parse failed: %s", exc)
            _unlink(path)
            return
        summary.update(
            {
                "type": "profile",
                "ts": _now_iso(),
                "seconds": seconds,
                "py_spy_timeout_s": timeout,
                "pid": pid,
            }
        )
        self._state.set_profile(summary)
        if self._keep > 0:
            self._kept.append(path)
            while len(self._kept) > self._keep:
                _unlink(self._kept.popleft())
        else:
            _unlink(path)


# ---------------------------------------------------------------------------
# Unix socket server
# ---------------------------------------------------------------------------


def _bounded(value: Any, low: int, high: int) -> int:
    """Clamp *value* to an integer in ``[low, high]``."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = low
    return max(low, min(high, number))


def _split_lines(buffer: bytes, max_size: int) -> tuple[list[bytes], bytes, bool]:
    """Split complete commands from the trailing partial line.

    Returns ``(lines, remainder, overflow)``.  ``overflow`` is true only
    when a line is longer than *max_size*; a batch of complete short
    commands never trips the cap.
    """
    *lines, remainder = buffer.split(b"\n")
    if len(remainder) > max_size or any(len(line) > max_size for line in lines):
        return [], b"", True
    return lines, remainder, False


class ClientHandler(socketserver.BaseRequestHandler):
    """Serve one NDJSON client."""

    def setup(self) -> None:
        self._buffer = b""
        self._client = self.server.state.subscribe()
        try:
            self.request.settimeout(5.0)
        except OSError:
            pass

    def finish(self) -> None:
        self.server.state.unsubscribe(self._client)

    def handle(self) -> None:
        if not self._send(
            {
                "type": "hello",
                "version": 1,
                "container": self.server.container,
                "interval": self.server.interval,
            }
        ):
            return
        while True:
            ready, _, _ = select.select([self.request], [], [], 0.5)
            if ready:
                try:
                    chunk = self.request.recv(65536)
                except OSError:
                    return
                if not chunk:
                    return
                lines, self._buffer, overflow = _split_lines(
                    self._buffer + chunk, _MAX_CLIENT_BUFFER
                )
                for line in lines:
                    if line.strip() and not self._command(line):
                        return
                if overflow:
                    self._send({"type": "error", "error": "command line too long"})
                    return
            if not self._drain():
                return

    def _drain(self) -> bool:
        while True:
            try:
                message = self._client.get_nowait()
            except queue.Empty:
                return True
            if not self._send(message):
                return False

    def _send(self, message: dict[str, Any]) -> bool:
        try:
            payload = json.dumps(message, separators=(",", ":")) + "\n"
            self.request.sendall(payload.encode("utf-8"))
            return True
        except OSError:
            return False

    def _command(self, line: bytes) -> bool:
        try:
            command = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return self._send({"type": "error", "error": "invalid json"})
        if not isinstance(command, dict):
            return self._send({"type": "error", "error": "command must be an object"})
        name = command.get("cmd")
        if name == "stop":
            return False
        if name == "ping":
            return self._send({"type": "pong", "ts": _now_iso()})
        if name == "snapshot":
            minutes = _bounded(command.get("minutes", 5), 1, 60)
            return self._send(
                {
                    "type": "snapshot",
                    "samples": self.server.state.snapshot(minutes),
                    "profile": self.server.state.latest_profile(),
                }
            )
        if name == "profile":
            seconds = _bounded(command.get("seconds", 30), 5, 300)
            result = self.server.on_profile(seconds)
            return self._send(
                {
                    "type": "ack",
                    "cmd": "profile",
                    "seconds": seconds,
                    **result,
                }
            )
        return self._send({"type": "error", "error": f"unknown command: {name!r}"})


class _ThreadingUnixStreamServer(
    socketserver.ThreadingMixIn, socketserver.UnixStreamServer
):
    daemon_threads = True
    block_on_close = False


class MonitorServer(_ThreadingUnixStreamServer):
    """The Unix-socket server that fans samples out to clients."""

    def __init__(
        self,
        address: str,
        state: MonitorState,
        on_profile: Any,
        container: str,
        interval: float,
    ) -> None:
        self._remove_stale_socket(address)
        super().__init__(address, ClientHandler)
        self.state = state
        self.on_profile = on_profile
        self.container = container
        self.interval = interval
        os.chmod(address, 0o600)

    @staticmethod
    def _remove_stale_socket(address: str) -> None:
        try:
            if stat.S_ISSOCK(os.stat(address).st_mode):
                os.unlink(address)
        except FileNotFoundError:
            pass


def _noop_profile(seconds: int) -> dict[str, Any]:
    """Profile callback used when profiling is disabled."""
    del seconds
    return {"accepted": False, "reason": "disabled"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Host-side CPU/memory hotspot monitor for a gateway container."
    )
    parser.add_argument(
        "--container", default=_DEFAULT_CONTAINER, help="Docker container name"
    )
    parser.add_argument("--pid", type=int, default=None, help="Override the PID")
    parser.add_argument("--socket", default=_DEFAULT_SOCKET)
    parser.add_argument("--log-dir", default=_DEFAULT_LOG_DIR)
    parser.add_argument("--db", default=_DEFAULT_DB, help="Gateway SQLite path")
    parser.add_argument("--no-db", action="store_true")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--profile-seconds", type=int, default=60)
    parser.add_argument(
        "--profile-every",
        type=int,
        default=300,
        help="Seconds between automatic profiles; 0 means manual only",
    )
    parser.add_argument(
        "--profile-cooldown",
        type=int,
        default=60,
        help="Minimum seconds between manual profile requests; 0 disables",
    )
    parser.add_argument("--py-spy", default=None)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--keep-profiles", type=int, default=0)
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--no-file", action="store_true")
    parser.add_argument("--no-container-stats", action="store_true")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print two samples (CPU delta) and exit",
    )
    args = parser.parse_args(argv)
    if args.no_db:
        args.db = ""
    if args.py_spy is None:
        args.py_spy = shutil.which("py-spy") or "/opt/llm-rosetta-monitor/py-spy"
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the monitor until SIGTERM."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    tracker = PidTracker(args.container, args.pid)
    if args.once:
        sampler = Sampler(
            tracker,
            MonitorState(args.interval),
            None,
            args.interval,
            args.db,
            args.container,
            not args.no_container_stats,
            threading.Event(),
        )
        first = sampler.sample_once()
        if first is None:
            return 1
        time.sleep(min(max(args.interval, 0.2), 5.0))
        second = sampler.sample_once()
        sample = second if second is not None else first
        print(json.dumps(sample, indent=2, sort_keys=True))
        return 0

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.socket)), exist_ok=True)

    state = MonitorState(args.interval)
    writer = (
        None
        if args.no_file
        else JsonlWriter(os.path.join(args.log_dir, "series.jsonl"))
    )
    stop = threading.Event()
    sampler = Sampler(
        tracker,
        state,
        writer,
        args.interval,
        args.db,
        args.container,
        not args.no_container_stats,
        stop,
    )
    profiler: ProfileRunner | None = None
    if not args.no_profile:
        profiler = ProfileRunner(
            tracker,
            state,
            args.py_spy,
            args.profile_seconds,
            args.profile_every,
            args.top,
            args.keep_profiles,
            args.log_dir,
            stop,
            args.profile_cooldown,
        )
    server = MonitorServer(
        args.socket,
        state,
        profiler.request if profiler is not None else _noop_profile,
        args.container,
        args.interval,
    )

    def _shutdown(signum: int, frame: Any) -> None:
        del frame
        logger.info("stopping on signal %s", signum)
        stop.set()
        if profiler is not None:
            profiler.stop()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    sampler.start()
    if profiler is not None:
        profiler.start()
    try:
        logger.info(
            "monitoring container=%s pid=%s socket=%s",
            args.container,
            tracker.get(),
            args.socket,
        )
    except RuntimeError as exc:
        logger.warning("%s", exc)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        stop.set()
        if profiler is not None:
            profiler.stop()
        sampler.join(timeout=3)
        if profiler is not None:
            profiler.join(timeout=3)
        server.server_close()
        if writer is not None:
            writer.close()
        _unlink(args.socket)
    return 0


if __name__ == "__main__":
    sys.exit(main())
