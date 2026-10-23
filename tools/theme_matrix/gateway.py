"""A disposable gateway for the theme matrix to point a browser at.

The matrix needs a live admin panel.  Pointing it at someone's real gateway
would mean depending on their config and their database, so by default it
starts its own: a synthetic config with two disabled providers, a fresh data
directory and an ephemeral port.

No admin password is set.  The harness checks the stylesheet, so a login would
only exercise the credential path; it is not what is under test.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

# Two providers of different shapes (one chat, one messages) and one model
# each, so the Providers tab has cards, logos and badges to draw.  Everything
# is disabled: nothing here reaches the network.
_FIXTURE_CONFIG: dict[str, object] = {
    "providers": {
        "Theme Chat": {
            "type": "chat_completions",
            "base_url": "https://example.invalid/v1",
            "api_key": "sk-theme-matrix",
            "enabled": False,
        },
        "Theme Messages": {
            "type": "messages",
            "base_url": "https://example.invalid",
            "api_key": "sk-theme-matrix",
            "enabled": False,
        },
    },
    "models": {
        "theme-chat": {
            "provider": "Theme Chat",
            "capabilities": ["text", "vision", "tools"],
        },
        "theme-messages": {
            "provider": "Theme Messages",
            "capabilities": ["text"],
        },
    },
    "server": {"host": "127.0.0.1"},
}


def _free_port() -> int:
    """Return a port the kernel says is currently free."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _healthy(base_url: str) -> bool:
    """Whether the gateway answers its health probe yet."""
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=0.5) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, OSError):
        return False


@contextmanager
def running_gateway(*, keep: bool = False, timeout: float = 30.0) -> Iterator[str]:
    """Run a throwaway gateway and yield its base URL.

    The gateway is the package in this working tree (``python -m
    llm_rosetta_gateway``), so the matrix renders the CSS of the branch being
    reviewed rather than whatever is installed globally.

    Args:
        keep: Leave the temporary config, data directory and log on disk and
            print their path, for when a cell needs debugging.
        timeout: Seconds to wait for the health endpoint.

    Yields:
        The base URL, e.g. ``http://127.0.0.1:41234``.

    Raises:
        RuntimeError: If the gateway exits early or never becomes healthy.
    """
    workdir = Path(tempfile.mkdtemp(prefix="theme-matrix-"))
    config = workdir / "config.jsonc"
    config.write_text(json.dumps(_FIXTURE_CONFIG, indent=2), encoding="utf-8")
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = workdir / "gateway.log"
    log = log_path.open("w", encoding="utf-8")

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "llm_rosetta_gateway",
            "--config",
            str(config),
            "--port",
            str(port),
            "--data-dir",
            str(workdir / "data"),
            "--no-banner",
            "--log-level",
            "warning",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                log.flush()
                raise RuntimeError(
                    f"gateway exited with {proc.returncode} before it was healthy:\n"
                    f"{log_path.read_text(encoding='utf-8', errors='replace')}"
                )
            if _healthy(base_url):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError(
                f"gateway did not answer /health within {timeout:g}s; log: {log_path}"
            )
        yield base_url
    finally:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=5)
        log.close()
        if keep:
            print(f"gateway workdir kept at {workdir}", file=sys.stderr)
        else:
            shutil.rmtree(workdir, ignore_errors=True)
