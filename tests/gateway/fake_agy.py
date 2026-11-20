"""A stand-in for the ``agy`` CLI, for tests that must not need a Google account.

It is a Python script written to disk by :func:`install_fake_agy` and run as
``agy acp ...``.  It speaks newline-delimited JSON-RPC like the ACP fake the
transport tests use, and it models the one thing the sign-in work depends on:
whether a given *profile* is signed in.

* A profile is the ``--gemini-dir`` given on the command line, or the host's
  own directory (``$FAKE_AGY_HOST``) when there is no such flag.  The profile
  is signed in when ``<profile>/signed-in`` exists.
* ``agy acp --login`` prints a sign-in URL and some noise, then — by
  ``$FAKE_AGY_LOGIN`` — finishes (writes the marker, exits 0), fails (exit 3)
  or hangs.
* ``session/new`` opens a session for a signed-in profile and is refused with
  a sign-in message otherwise; ``$FAKE_AGY_MODE=broken`` refuses with an error
  that is not about signing in.
* Every start appends one JSON object to ``$FAKE_AGY_LOG``: the pid and argv.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import textwrap
from pathlib import Path

FAKE_AGY = textwrap.dedent(
    """
    import json, os, sys, time

    args = sys.argv[1:]

    def flag(name):
        return args[args.index(name) + 1] if name in args else None

    profile = flag("--gemini-dir") or os.environ["FAKE_AGY_HOST"]
    marker = os.path.join(profile, "signed-in")
    log = os.environ.get("FAKE_AGY_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "argv": args}) + "\\n")

    if "--login" in args:
        print("Open this URL to sign in:", flush=True)
        print("https://accounts.example/o/oauth2/auth?client_id=x&state=abc", flush=True)
        print("debug ya29.A0ARrdaMsecrettokenvalue123", flush=True)
        how = os.environ.get("FAKE_AGY_LOGIN", "finish")
        if how == "hang":
            time.sleep(600)
        if how == "fail":
            sys.exit(3)
        if how == "flood":
            for _ in range(5000):
                print("x" * 300, flush=True)
        time.sleep(float(os.environ.get("FAKE_AGY_DELAY", "0.2")))
        os.makedirs(profile, exist_ok=True)
        open(marker, "w").write("signed in")
        sys.exit(0)

    def send(payload):
        sys.stdout.write(json.dumps(payload) + "\\n")
        sys.stdout.flush()

    mode = os.environ.get("FAKE_AGY_MODE", "normal")
    for line in sys.stdin:
        if not line.strip():
            continue
        msg = json.loads(line)
        method, req_id = msg.get("method"), msg.get("id")
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": req_id, "result": {
                "protocolVersion": 1, "agentCapabilities": {},
                "agentInfo": {"name": "fake-agy", "version": "0"},
                "authMethods": [{"id": "oauth-personal", "name": "Log in with Google"}]}})
        elif method == "session/new":
            if mode == "broken":
                send({"jsonrpc": "2.0", "id": req_id,
                      "error": {"code": -32603, "message": "internal explosion"}})
            elif os.path.exists(marker):
                send({"jsonrpc": "2.0", "id": req_id, "result": {"sessionId": "s-1"}})
            else:
                send({"jsonrpc": "2.0", "id": req_id, "error": {
                    "code": -32000,
                    "message": "Authentication required: please sign in first"}})
        else:
            send({"jsonrpc": "2.0", "id": req_id, "result": {}})
    """
)


def install_fake_agy(directory: Path) -> Path:
    """Write the fake as an executable ``agy`` into *directory* and return it."""
    path = directory / "agy"
    path.write_text(f"#!{sys.executable}\n{FAKE_AGY}", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def starts(log: Path) -> list[dict]:
    """Every process the fake recorded, oldest first."""
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line]


def alive(pid: int) -> bool:
    """Whether *pid* is a running process (a zombie counts as gone)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return True
    return "State:\tZ" not in status
