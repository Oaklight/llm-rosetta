#!/usr/bin/env python3
"""Live client for ``hotspot_monitor.py``.

The monitor listens on a Unix socket on the Docker host.  Forward it over
SSH and connect from the workstation::

    ssh -N -L /tmp/hotspot.sock:/run/llm-rosetta-hotspot/hotspot.sock cloud.usa2
    python3 hotspot_watch.py --socket /tmp/hotspot.sock --minutes 5

Pass ``--profile`` to request an immediate py-spy window, or
``--profile 30`` for a custom duration.  ``--json`` prints the raw NDJSON
stream.
"""

from __future__ import annotations

import argparse
import json
import select
import socket
import sys
import time
from typing import Any

_DEFAULT_SOCKET = "/run/llm-rosetta-hotspot/hotspot.sock"


def _fmt(value: Any, suffix: str = "") -> str:
    """Format a number for the live line."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.1f}{suffix}"
    return f"{value}{suffix}"


def _send(sock: socket.socket, message: dict[str, Any]) -> bool:
    """Send one command line."""
    try:
        sock.sendall(
            (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
        )
        return True
    except OSError:
        return False


def _render_sample(
    sample: dict[str, Any],
    args: argparse.Namespace,
    last: dict[str, Any] | None = None,
) -> None:
    """Print one sample as a compact line.

    The monitor reads the request-log summary every 10 s, so most samples
    carry ``db: None``.  *last* keeps the previous summary so the line still
    shows the request rate and latency between reads.
    """
    db = sample.get("db") or (last or {}).get("db") or {}
    if last is not None and sample.get("db"):
        last["db"] = sample["db"]
    fields = [
        (sample.get("ts") or "")[11:19],
        f"cpu {_fmt(sample.get('cpu_pct'), '%')}",
        f"{_fmt(sample.get('cpu_cores'))}c",
        f"rss {_fmt(sample.get('rss_mb'), 'MB')}",
        f"hwm {_fmt(sample.get('hwm_mb'), 'MB')}",
        f"thr {_fmt(sample.get('threads'))}",
        f"fds {_fmt(sample.get('fds'))}",
    ]
    if db:
        fields.extend(
            [
                f"rps {_fmt(db.get('rps'))}",
                f"stream {_fmt((db.get('stream_pct') or 0.0) * 100, '%')}",
                f"conv-p95 {_fmt(db.get('conversion_p95_ms'), 'ms')}",
                f"up-p95 {_fmt(db.get('upstream_p95_ms'), 'ms')}",
            ]
        )
        if db.get("truncated"):
            fields.append("TRUNCATED")
    print("  ".join(fields), flush=True)

    if args.threads and sample.get("thread_cpu"):
        threads = "  ".join(
            f"{name}: {percent}%" for name, percent in sample["thread_cpu"].items()
        )
        print(f"    {threads}", flush=True)
    if args.pairs and db.get("slow_pairs"):
        pairs = "  ".join(
            f"{item['pair']} {item['conversion_p95_ms']}ms x{item['count']}"
            for item in db["slow_pairs"]
        )
        print(f"    slow: {pairs}", flush=True)


def _render_profile(profile: dict[str, Any]) -> None:
    """Print a py-spy summary."""
    stamp = (profile.get("ts") or "")[11:19]
    print(
        f"\n[{stamp}] profile {profile.get('seconds')}s  "
        f"samples={profile.get('total_samples')}  "
        f"active={profile.get('active_samples')}  "
        f"idle={_fmt(profile.get('idle_pct'), '%')}",
        flush=True,
    )
    for row in profile.get("top") or []:
        print(
            f"  {row.get('pct', 0):5.1f}%  {row.get('fn')}  "
            f"({row.get('file')}:{row.get('line')})",
            flush=True,
        )
    print(flush=True)


def _render_snapshot(
    snapshot: dict[str, Any],
    args: argparse.Namespace,
    last: dict[str, Any] | None = None,
) -> None:
    """Print the samples returned for a snapshot request."""
    samples = snapshot.get("samples") or []
    print(f"snapshot: {len(samples)} samples", flush=True)
    if samples:
        if last is not None:
            for sample in samples:
                if sample.get("db"):
                    last["db"] = sample["db"]
        _render_sample(samples[-1], args, last)
    profile = snapshot.get("profile")
    if profile:
        _render_profile(profile)


def _render(
    message: dict[str, Any],
    args: argparse.Namespace,
    last: dict[str, Any] | None = None,
) -> None:
    """Dispatch one server message."""
    kind = message.get("type")
    if kind == "hello":
        print(
            f"connected: container={message.get('container')} "
            f"interval={message.get('interval')}s",
            flush=True,
        )
    elif kind == "snapshot":
        _render_snapshot(message, args, last)
    elif kind == "sample":
        _render_sample(message, args, last)
    elif kind == "profile":
        _render_profile(message)
    elif kind == "ack":
        if message.get("accepted", True):
            print(f"profile requested: {message.get('seconds')}s", flush=True)
        else:
            retry = message.get("retry_after_s")
            suffix = f", retry in {retry}s" if retry else ""
            print(
                f"profile rejected: {message.get('reason')}{suffix}",
                flush=True,
            )
    elif kind == "error":
        print(f"error: {message.get('error')}", file=sys.stderr, flush=True)
    elif kind != "pong":
        print(json.dumps(message, indent=2), flush=True)


def _connect(path: str, reconnect: bool) -> socket.socket | None:
    """Connect to the monitor socket, optionally retrying."""
    while True:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(path)
            return sock
        except OSError as exc:
            sock.close()
            if not reconnect:
                print(f"cannot connect to {path}: {exc}", file=sys.stderr)
                return None
            time.sleep(2)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live client for the host-side hotspot monitor."
    )
    parser.add_argument("--socket", default=_DEFAULT_SOCKET)
    parser.add_argument("--minutes", type=int, default=5)
    parser.add_argument(
        "--profile",
        nargs="?",
        const=30,
        type=int,
        default=None,
        help="Request a profile window (default 30 s)",
    )
    parser.add_argument("--threads", action="store_true")
    parser.add_argument("--pairs", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--reconnect", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Connect, print the stream, and exit on EOF or Ctrl-C."""
    args = _parse_args(argv)
    sock = _connect(args.socket, args.reconnect)
    if sock is None:
        return 1
    _send(sock, {"cmd": "snapshot", "minutes": args.minutes})
    if args.profile is not None:
        _send(sock, {"cmd": "profile", "seconds": args.profile})

    buffer = b""
    last: dict[str, Any] = {}
    try:
        while True:
            ready, _, _ = select.select([sock], [], [], 1.0)
            if not ready:
                continue
            try:
                chunk = sock.recv(65536)
            except OSError:
                break
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line.strip():
                    continue
                if args.json:
                    print(
                        line.decode("utf-8", errors="replace"),
                        flush=True,
                    )
                    continue
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                _render(message, args, last)
    except KeyboardInterrupt:
        pass
    finally:
        _send(sock, {"cmd": "stop"})
        sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
