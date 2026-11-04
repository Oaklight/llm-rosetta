"""Tests for the host-side hotspot monitor (``scripts/dev/hotspot_monitor.py``)."""

from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts/dev/hotspot_monitor.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("hotspot_monitor", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def monitor():
    return _load_script()


_WATCH = Path(__file__).resolve().parent.parent / "scripts/dev/hotspot_watch.py"


def _load_watch():
    spec = importlib.util.spec_from_file_location("hotspot_watch", _WATCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def watch():
    return _load_watch()


def _stat_line(
    *,
    comm: str = "gateway",
    utime: int = 100,
    stime: int = 50,
    threads: int = 5,
    starttime: int = 999,
) -> str:
    """Build a /proc/<pid>/stat line with the fields the monitor reads."""
    rest = ["S"] + ["0"] * 49
    rest[11] = str(utime)  # field 14
    rest[12] = str(stime)  # field 15
    rest[17] = str(threads)  # field 20
    rest[19] = str(starttime)  # field 22
    return f"123 ({comm}) " + " ".join(rest)


def test_proc_stat_handles_parentheses_in_comm(monitor):
    stat = monitor.parse_proc_stat(_stat_line(comm="llm (gateway)"))
    assert stat["comm"] == "llm (gateway)"
    assert stat["utime"] == 100
    assert stat["stime"] == 50
    assert stat["num_threads"] == 5
    assert stat["starttime"] == 999


def test_proc_stat_rejects_a_short_line(monitor):
    with pytest.raises(ValueError, match="short"):
        monitor.parse_proc_stat("1 (x) S 1 2 3")


def test_smaps_rollup_keeps_values_in_mib(monitor):
    text = (
        "Rss: 204800 kB\n"
        "Anonymous: 102400 kB\n"
        "Shared_Clean: 2048 kB\n"
        "Shared_Dirty: 1024 kB\n"
    )
    smaps = monitor.parse_smaps_rollup(text)
    assert smaps["Rss"] == 200.0
    assert smaps["Anonymous"] == 100.0
    assert smaps["Shared_Clean"] == 2.0
    assert smaps["Shared_Dirty"] == 1.0


def test_status_and_io_parsing(monitor):
    status = monitor.parse_status(
        "Name:\tgateway\nVmRSS:\t  195884 kB\nVmHWM:\t  417060 kB\n"
    )
    assert status["VmRSS"] == "195884 kB"
    assert monitor._kb_to_mb(status["VmHWM"]) == 407.3

    io = monitor.parse_io("rchar: 1048576\nwchar: 2097152\nsyscr: 10\nsyscw: 20\n")
    assert io["rchar"] == 1048576
    assert io["wchar"] == 2097152


def test_speedscope_excludes_idle_leaves(monitor):
    frames = [
        {
            "name": "EpollSelector.select",
            "file": "/usr/lib/python3.12/selectors.py",
            "line": 468,
        },
        {
            "name": "JSONEncoder.iterencode",
            "file": "/usr/lib/python3.12/json/encoder.py",
            "line": 258,
        },
    ]
    data = {
        "shared": {"frames": frames},
        "profiles": [
            {
                "name": 'Thread 1 "MainThread"',
                "samples": [[0], [0], [1], [1], [1]],
            }
        ],
    }

    summary = monitor.summarize_speedscope(data, top_n=5)

    assert summary["total_samples"] == 5
    assert summary["active_samples"] == 3
    assert summary["idle_pct"] == 40.0
    assert summary["top"][0]["fn"] == "JSONEncoder.iterencode"
    assert summary["top"][0]["pct"] == 100.0


def test_speedscope_without_a_main_thread_merges_threads(monitor):
    frames = [
        {"name": "_worker", "file": "/x/thread.py", "line": 90},
        {"name": "getaddrinfo", "file": "/x/socket.py", "line": 978},
    ]
    data = {
        "shared": {"frames": frames},
        "profiles": [
            {"name": "asyncio_0", "samples": [[0], [1], [1]]},
        ],
    }

    summary = monitor.summarize_speedscope(data, top_n=5)

    assert summary["active_samples"] == 2
    assert summary["idle_pct"] == 33.3
    assert summary["top"][0]["fn"] == "getaddrinfo"
    assert summary["top"][0]["pct"] == 100.0


def test_percentile_interpolates(monitor):
    assert monitor.percentile([1, 2, 3, 4], 0.5) == 2.5
    assert monitor.percentile([], 0.95) is None
    assert monitor.percentile([7], 0.95) == 7


def test_size_to_mb_checks_longer_units_first(monitor):
    assert monitor._size_to_mb("656.1MiB") == 656.1
    assert monitor._size_to_mb("1GiB") == 1024.0
    assert monitor._size_to_mb("512KiB") == 0.5
    assert monitor._size_to_mb("1024B") == 0.0


def test_idle_leaf_names(monitor):
    assert monitor.is_idle_leaf("EpollSelector.select")
    assert monitor.is_idle_leaf("_worker")
    assert monitor.is_idle_leaf("Connection._worker")
    assert not monitor.is_idle_leaf("JSONEncoder.iterencode")


def _make_request_log_db(path: Path) -> None:
    """Create the request_log columns read_db selects."""
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE request_log ("
        "timestamp TEXT, source_provider TEXT, target_provider TEXT, "
        "is_stream INTEGER, profile TEXT)"
    )
    connection.execute(
        "INSERT INTO request_log VALUES (?, ?, ?, ?, ?)",
        (
            "2026-11-03T00:00:00+00:00",
            "messages",
            "chat_completions",
            1,
            '{"request_conversion_ms": 3.1, "stream_connect_ms": 40, '
            '"stream_duration_ms": 900}',
        ),
    )
    connection.commit()
    connection.close()


def test_speedscope_without_main_thread_reports_idle(monitor):
    frames = [
        {"name": "JSONEncoder.iterencode", "file": "/x/encoder.py", "line": 1},
        {"name": "EpollSelector.select", "file": "/x/selectors.py", "line": 1},
    ]
    data = {
        "shared": {"frames": frames},
        "profiles": [{"name": "asyncio_0", "samples": [[0], [1], [1], [1]]}],
    }

    summary = monitor.summarize_speedscope(data, top_n=5)

    assert summary["active_samples"] == 1
    assert summary["idle_pct"] == 75.0


def test_profile_cooldown_rejects_a_second_request(monitor):
    runner = monitor.ProfileRunner(
        tracker=monitor.PidTracker("dummy", pid=os.getpid()),
        state=monitor.MonitorState(1.0),
        py_spy="",
        seconds=10,
        every=0,
        top_n=5,
        keep=0,
        log_dir="/tmp",
        stop=threading.Event(),
        cooldown=60,
    )

    assert runner.request(10)["accepted"] is True
    runner._last_end = time.monotonic()
    second = runner.request(10)

    assert second["accepted"] is False
    assert second["reason"] == "cooldown"
    assert second["retry_after_s"] >= 1


def test_read_db_connects_read_only(monitor, tmp_path, monkeypatch):
    db = tmp_path / "gateway.db"
    _make_request_log_db(db)
    seen = []
    real_connect = monitor.sqlite3.connect

    def capture(database, **kwargs):
        seen.append(database)
        return real_connect(database, **kwargs)

    monkeypatch.setattr(monitor.sqlite3, "connect", capture)
    summary = monitor.read_db(str(db), window_seconds=10**9)

    assert summary is not None
    assert summary["requests"] == 1
    assert summary["upstream_p95_ms"] == 940.0
    assert seen
    connection = sqlite3.connect(seen[0], uri=True)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("CREATE TABLE probe (x)")
    finally:
        connection.close()


@pytest.mark.skipif(os.name == "nt", reason="'?' is not valid in a Windows filename")
def test_read_db_quotes_a_path_with_a_question_mark(monitor, tmp_path):
    db = tmp_path / "gateway?x.db"
    _make_request_log_db(db)

    summary = monitor.read_db(str(db), window_seconds=10**9)

    assert summary is not None
    assert summary["requests"] == 1


@pytest.mark.skipif(not os.path.exists("/proc/self/stat"), reason="procfs required")
def test_once_does_not_create_directories(monitor, tmp_path, capsys):
    socket = tmp_path / "nested" / "hotspot.sock"
    log_dir = tmp_path / "logs"

    rc = monitor.main(
        [
            "--once",
            "--pid",
            str(os.getpid()),
            "--no-profile",
            "--no-db",
            "--no-container-stats",
            "--socket",
            str(socket),
            "--log-dir",
            str(log_dir),
            "--interval",
            "0.2",
        ]
    )

    assert rc == 0
    assert not socket.parent.exists()
    assert not log_dir.exists()
    data = json.loads(capsys.readouterr().out)
    assert data["main_cpu_pct"] is not None


def test_read_db_keeps_the_newest_rows(monitor, tmp_path):
    db = tmp_path / "gateway.db"
    connection = sqlite3.connect(db)
    connection.execute(
        "CREATE TABLE request_log ("
        "timestamp TEXT, source_provider TEXT, target_provider TEXT, "
        "is_stream INTEGER, profile TEXT)"
    )
    rows = [
        (
            f"2026-11-03T00:00:0{i}+00:00",
            "a",
            "b",
            0,
            f'{{"request_conversion_ms": {10 if i < 3 else 900}}}',
        )
        for i in range(5)
    ]
    connection.executemany("INSERT INTO request_log VALUES (?, ?, ?, ?, ?)", rows)
    connection.commit()
    connection.close()

    summary = monitor.read_db(str(db), window_seconds=10**9, limit=2)

    assert summary is not None
    assert summary["requests"] == 2
    assert summary["truncated"] is True
    assert summary["conversion_p95_ms"] == 900.0


def test_split_lines_caps_only_a_line(monitor):
    lines, remainder, overflow = monitor._split_lines(
        b'{"cmd":"ping"}\n' * 20000, 64 * 1024
    )
    assert len(lines) == 20000
    assert remainder == b""
    assert overflow is False

    lines, remainder, overflow = monitor._split_lines(b"x" * (70 * 1024), 64 * 1024)
    assert lines == []
    assert remainder == b""
    assert overflow is True

    lines, remainder, overflow = monitor._split_lines(
        b"x" * (70 * 1024) + b"\n", 64 * 1024
    )
    assert lines == []
    assert overflow is True

    lines, remainder, overflow = monitor._split_lines(b"partial", 64 * 1024)
    assert lines == []
    assert remainder == b"partial"
    assert overflow is False


def test_watch_marks_a_truncated_window(watch, capsys):
    sample = {
        "ts": "2026-11-03T00:00:00+00:00",
        "cpu_pct": 1.0,
        "cpu_cores": 0.01,
        "rss_mb": 1.0,
        "hwm_mb": 2.0,
        "threads": 1,
        "fds": 3,
        "db": {
            "rps": 83.3,
            "stream_pct": 1.0,
            "conversion_p95_ms": 1.0,
            "upstream_p95_ms": 2.0,
            "truncated": True,
        },
    }

    watch._render_sample(sample, SimpleNamespace(threads=False, pairs=False))

    assert "TRUNCATED" in capsys.readouterr().out


def _db_summary(rps: float) -> dict[str, float]:
    return {
        "rps": rps,
        "stream_pct": 0.5,
        "conversion_p95_ms": 10.0,
        "upstream_p95_ms": 20.0,
    }


def test_watch_carries_the_last_db_summary(watch, capsys):
    args = SimpleNamespace(threads=False, pairs=False)
    last: dict = {}

    watch._render_sample(
        {"ts": "2026-11-03T00:00:00+00:00", "db": _db_summary(1.5)}, args, last
    )
    capsys.readouterr()
    watch._render_sample({"ts": "2026-11-03T00:00:01+00:00", "db": None}, args, last)

    assert "rps 1.5" in capsys.readouterr().out

    # A later read replaces the carried summary; it is not set once.
    watch._render_sample(
        {"ts": "2026-11-03T00:00:02+00:00", "db": _db_summary(2.0)}, args, last
    )
    capsys.readouterr()
    watch._render_sample({"ts": "2026-11-03T00:00:03+00:00", "db": None}, args, last)

    assert "rps 2.0" in capsys.readouterr().out


def test_watch_snapshot_seeds_the_summary_from_the_window(watch, capsys):
    snapshot = {
        "samples": [
            {"ts": "2026-11-03T00:00:00+00:00", "db": _db_summary(2.0)},
            {"ts": "2026-11-03T00:00:01+00:00", "db": None},
        ]
    }

    watch._render_snapshot(
        snapshot, SimpleNamespace(threads=False, pairs=False), last={}
    )

    assert "rps 2.0" in capsys.readouterr().out
