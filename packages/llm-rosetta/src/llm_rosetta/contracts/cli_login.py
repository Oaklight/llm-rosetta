"""Drive a vendor CLI's interactive sign-in from the admin panel.

This is a contract module: a shim's ``auth.py`` sidecar imports it as
``llm_rosetta.contracts.cli_login`` (a sidecar runs during ``import
llm_rosetta``, before :mod:`llm_rosetta.api` can load).  It uses the standard
library only and imports nothing from ``llm_rosetta`` or the gateway.


Both ACP lanes keep their credential in the vendor CLI's own store: Claude
Code writes ``$CLAUDE_CONFIG_DIR`` (``~/.claude`` when unset), and the
Antigravity CLI writes a Google account token to the desktop keyring, or to a
file under its ``--gemini-dir`` when no keyring is available.  The gateway
gives each lane a persistent directory under its data directory and passes it
to the CLI here, so an admin sign-in lands in the same profile the agent will
use and survives a restart or image upgrade.  The gateway never sees that
token, so the ``auth.py`` exchange for such a lane cannot mint one.  What it
can do is run the vendor's own sign-in command, show the operator the URL
that command prints, and feed the pasted code back to the waiting process.

That is what this module provides.  It is deliberately small and
standard-library only: the shim layer loads before the gateway and has no
dependencies.  One pending sign-in per key (the gateway provider name) is
kept; starting again replaces the old process, and an abandoned flow is
terminated when it expires or when the gateway exits.

The flow is the same for every CLI:

1. ``begin_cli_login`` spawns the command with pipes and reads its stdout on
   a worker thread until it prints an authorization URL.
2. The process is left waiting on stdin.  The admin panel opens the URL and
   asks the operator for the code (or the full callback URL) the vendor shows.
3. ``complete_cli_login`` writes that value to stdin, waits for the process to
   exit, and then lets the caller verify the result.

Nothing here parses or stores a credential.  A failure message is built from
the process's own output, with the pasted value redacted.

A CLI that opens a browser gets a recorder instead.  ``BROWSER`` points at a
small script that writes the page the CLI asked for to a file: the admin panel
already shows a page, and a tab the CLI opens on its own completes the same
sign-in behind the panel's back.  The recorded loopback address also lets a
pasted callback URL be replayed on the host that runs the CLI, which is what
makes a browser on another machine usable.
"""

from __future__ import annotations

import atexit
import http.client
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from typing import Any
from urllib.parse import parse_qs, urlparse

__all__ = [
    "CliLoginError",
    "begin_cli_login",
    "cancel_cli_login",
    "complete_cli_login",
    "run_capture",
    "subprocess_env",
]

#: How long a pending sign-in may sit unfinished before its process is killed.
PENDING_TTL = 600.0

#: How long the CLI gets to print its authorization URL.
URL_TIMEOUT = 30.0

#: How long to wait for the recorded page when the printed URL came first.
BROWSER_GRACE = 1.0

#: How long the CLI gets to finish after the pasted value is written.
COMPLETE_TIMEOUT = 120.0

#: What the admin panel calls the pasted input.
PASTE_LABEL = "Paste the code (or the callback URL) the vendor page shows"
PASTE_PLACEHOLDER = "the code shown after signing in"

#: ``BROWSER`` when the CLI must not open a tab of its own.
_BROWSER_RECORDER = """#!/bin/sh
printf '%s\\n' "$1" > "$LLM_ROSETTA_BROWSER_URL_FILE"
"""
_BROWSER_URL_ENV = "LLM_ROSETTA_BROWSER_URL_FILE"
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

_URL_RE = re.compile(r"https://[^\s\x1b\x07\"'<>]+")
#: A URL is the sign-in URL when it looks like one, not merely the first link
#: a CLI prints (a header or a documentation link is common).
_LOGIN_URL_HINTS = ("oauth", "authorize", "accounts.google.com", "claude.com")

#: Token shapes a verbose CLI may print even though the gateway asked for a
#: URL.  The pasted code is redacted exactly; these catch the vendor's own
#: echoes.  A false positive on a diagnostic word is harmless.
_TOKEN_RES = (
    re.compile(r"\bya29\.[A-Za-z0-9._\-]+"),
    re.compile(r"\b1//[A-Za-z0-9._\-]+"),
    re.compile(r"\bsk-[A-Za-z0-9._\-]{8,}"),
    re.compile(r"\bAIza[0-9A-Za-z._\-]{20,}"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=\-]{8,}"),
)

_LOCK = threading.Lock()
_PENDING: dict[str, _Pending] = {}


class CliLoginError(Exception):
    """The CLI could not start, refused the code, or did not finish."""


class _Pending:
    def __init__(
        self,
        process: subprocess.Popen[str],
        *,
        deadline: float,
        browser_file: str | None = None,
        workdir: str | None = None,
    ) -> None:
        self.process = process
        self.deadline = deadline
        self.url = ""
        self.url_event = threading.Event()
        self.first_url = ""
        self.browser_file = browser_file
        self.browser_url = ""
        self.callback_port = ""
        self.callback_path = ""
        self.callback_origin = ""
        self.busy = False
        self.workdir = workdir
        self.lines: deque[str] = deque(maxlen=200)
        self.timer: threading.Timer | None = None
        self.reader: threading.Thread | None = None

    def tail(self) -> str:
        text = "\n".join(self.lines).strip()
        return text[-1500:] if text else "(no output)"


def run_capture(
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout: float = 15.0,
) -> tuple[int, str, str] | None:
    """Run a status command and return ``(code, stdout, stderr)``.

    ``None`` means the command could not be started (a missing binary, a
    permission error).  A caller treats that as "status unavailable", not as
    "signed in" or "signed out".
    """
    try:
        proc = subprocess.run(
            argv,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.returncode, proc.stdout, proc.stderr


def begin_cli_login(
    key: str,
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
    ttl: float = PENDING_TTL,
    url_timeout: float = URL_TIMEOUT,
    paste_label: str = PASTE_LABEL,
    paste_placeholder: str = PASTE_PLACEHOLDER,
    browser: bool = False,
) -> dict[str, Any]:
    """Start one sign-in and return its authorization URL.

    Args:
        key: Names the pending sign-in; one sign-in per key.
        argv: The vendor command to run.
        env: The CLI's environment.  ``None`` means this process's.
        ttl: How long the flow may stay unfinished.
        url_timeout: How long the CLI gets to produce a page.
        paste_label: What the admin panel calls the pasted input.
        paste_placeholder: The placeholder for that input.
        browser: Point ``BROWSER`` at a recorder so the CLI opens no tab of its
            own.  When the recorded page names a loopback redirect, the result
            carries ``callback_origin`` and a pasted callback URL is replayed
            on that listener.

    Raises:
        CliLoginError: The command could not be started, exited before printing
            a URL, or printed none in time.
    """
    cancel_cli_login(key)
    browser_file = None
    workdir = None
    if browser:
        env, browser_file, workdir = _browser_env(env)
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
    except OSError as exc:
        _remove_dir(workdir)
        raise CliLoginError(f"the sign-in command could not start: {exc}") from exc
    pending = _Pending(
        process,
        deadline=time.time() + ttl,
        browser_file=browser_file,
        workdir=workdir,
    )
    reader = threading.Thread(
        target=_read_output, args=(pending,), name=f"vendor-login-{key}", daemon=True
    )
    pending.reader = reader
    reader.start()
    _wait_for_url(pending, url_timeout)
    if browser_file:
        # One signal usually precedes the other: the recorder may run before
        # the CLI prints its page or after it.  A short grace collects both,
        # so the printed page stays the primary one and the loopback address
        # is available for a pasted callback URL.
        end = time.monotonic() + BROWSER_GRACE
        while time.monotonic() < end:
            printed = pending.url or pending.first_url
            if printed and _read_browser_url(pending):
                break
            time.sleep(0.02)
    url = pending.url or pending.first_url or pending.browser_url
    if not url:
        output = pending.tail()
        _terminate(pending)
        raise CliLoginError(
            f"the sign-in command printed no authorization URL; it said: {output}"
        )
    with _LOCK:
        _PENDING[key] = pending
    timer = threading.Timer(ttl, _expire, args=(key, pending))
    timer.daemon = True
    timer.start()
    pending.timer = timer
    result: dict[str, Any] = {
        "authorization_url": url,
        "expires_at": pending.deadline,
        "paste_label": paste_label,
        "paste_placeholder": paste_placeholder,
    }
    if pending.callback_origin:
        result["callback_origin"] = pending.callback_origin
    return result


def complete_cli_login(
    key: str,
    pasted: str,
    *,
    timeout: float = COMPLETE_TIMEOUT,
) -> None:
    """Write the operator's code to the waiting CLI and wait for it to exit.

    Raises:
        CliLoginError: No sign-in is pending, the value is not from this
            sign-in, the process refused it, or it did not finish in time.
    """
    value = _first_line(pasted)
    if not value:
        raise CliLoginError("a code or callback URL is required")
    with _LOCK:
        pending = _PENDING.get(key)
        if pending is None:
            raise CliLoginError("no sign-in is pending; start again")
        if pending.busy:
            raise CliLoginError("this sign-in is already finishing; wait for it")
        pending.busy = True
    try:
        _finish_pending(key, pending, value, timeout)
    finally:
        with _LOCK:
            pending.busy = False


def _finish_pending(key: str, pending: _Pending, value: str, timeout: float) -> None:
    if time.time() >= pending.deadline:
        _take(key, pending)
        _terminate(pending)
        raise CliLoginError("the sign-in expired; start again")
    process = pending.process
    callback = _loopback_callback(value)
    replay: tuple[str, str] | None = None
    if callback is not None and pending.callback_port:
        port, target = callback
        if (
            port != pending.callback_port
            or target.split("?")[0] != pending.callback_path
        ):
            # The flow stays open: the operator can still paste its code.
            raise CliLoginError(
                "that address isn't from this sign-in: paste the code the "
                "vendor page shows, or the address this sign-in opened"
            )
        replay = (port, target)
    if process.poll() == 0:
        # The flow finished without this call: a browser reached the CLI's own
        # callback listener, or the operator submitted twice.  The CLI is the
        # authority; the exchange verifies the account after this returns.
        _take(key, pending)
        _drain_reader(pending)
        _remove_dir(pending.workdir)
        pending.workdir = None
        return
    if replay is not None:
        try:
            _replay_callback(*replay)
        except OSError as exc:
            # The CLI still waits on stdin; keep the flow so the operator can
            # paste the code instead of the address.
            raise CliLoginError(
                "the sign-in listener did not take the callback "
                f"({exc}); paste the code the page shows instead"
            ) from exc
        _take(key, pending)
    else:
        try:
            assert process.stdin is not None
            process.stdin.write(value + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            try:
                finished = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                finished = None
            if finished == 0:
                _take(key, pending)
                _drain_reader(pending)
                _remove_dir(pending.workdir)
                pending.workdir = None
                return
            _take(key, pending)
            output = pending.tail()
            _terminate(pending)
            raise CliLoginError(
                f"the sign-in command had already exited: {exc}; it said: {output}"
            ) from exc
        _take(key, pending)
    try:
        code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate(pending)
        raise CliLoginError(
            "the sign-in command did not finish after the code was entered"
        ) from exc
    _drain_reader(pending)
    _remove_dir(pending.workdir)
    pending.workdir = None
    output = pending.tail()
    if code != 0:
        raise CliLoginError(
            f"the sign-in command exited with {code}; it said: {_redact(output, value)}"
        )
    return None


def _take(key: str, pending: _Pending) -> None:
    """Remove *pending* from the map and stop its expiry timer."""
    with _LOCK:
        if _PENDING.get(key) is pending:
            _PENDING.pop(key, None)
    if pending.timer is not None:
        pending.timer.cancel()
        pending.timer = None


def cancel_cli_login(key: str) -> None:
    """Stop a pending sign-in for *key*, if there is one."""
    with _LOCK:
        pending = _PENDING.pop(key, None)
    if pending is not None:
        if pending.timer is not None:
            pending.timer.cancel()
        _terminate(pending)


def _browser_env(
    env: dict[str, str] | None,
) -> tuple[dict[str, str], str | None, str | None]:
    """Point ``BROWSER`` at a recorder; ``(env, url_file, workdir)``.

    A CLI that opens its own browser tab completes the sign-in out of band:
    the admin panel's paste box never runs, and the panel keeps showing the
    state from before.  Recording the page instead leaves exactly one tab, the
    one the panel opens, and keeps the loopback address to replay a pasted
    callback URL.  Linux and macOS only; elsewhere the CLI's own launcher is
    kept.
    """
    merged = dict(env) if env is not None else dict(os.environ)
    if os.name != "posix":
        return merged, None, None
    workdir = None
    try:
        workdir = tempfile.mkdtemp(prefix="llm-rosetta-signin-")
        command = os.path.join(workdir, "browser-open")
        with open(command, "w", encoding="utf-8") as handle:
            handle.write(_BROWSER_RECORDER)
        os.chmod(command, 0o700)
    except OSError:
        _remove_dir(workdir)
        return merged, None, None
    url_file = os.path.join(workdir, "browser-url")
    merged["BROWSER"] = command
    merged[_BROWSER_URL_ENV] = url_file
    return merged, url_file, workdir


def _read_browser_url(pending: _Pending) -> str:
    """The page the CLI asked to open, once the recorder has written it."""
    if pending.browser_url or not pending.browser_file:
        return pending.browser_url
    try:
        with open(pending.browser_file, encoding="utf-8") as handle:
            text = handle.read().strip()
    except OSError:
        return ""
    if not text:
        return ""
    url = _clean_url(text.splitlines()[0])
    if not url.startswith(("http://", "https://")):
        return ""
    pending.browser_url = url
    callback = _loopback_callback(url)
    if callback is not None:
        pending.callback_port, target = callback
        pending.callback_path = target.split("?")[0]
    else:
        target = _redirect_target(url) or _loopback_origin(url)
        if target is not None:
            pending.callback_port, pending.callback_path = target
    if pending.callback_port:
        pending.callback_origin = f"http://127.0.0.1:{pending.callback_port}"
    return url


def _loopback_origin(value: str) -> tuple[str, str] | None:
    """``(port, path)`` when *value* is itself a URL on the loopback."""
    try:
        parsed = urlparse(value)
        port = parsed.port
    except ValueError:
        return None
    if parsed.scheme not in ("http", "https"):
        return None
    if (parsed.hostname or "").lower() not in _LOOPBACK_HOSTS or not port:
        return None
    return str(port), parsed.path or "/"


def _redirect_target(value: str) -> tuple[str, str] | None:
    """``(port, path)`` of the loopback a sign-in page comes back to."""
    try:
        parsed = urlparse(value)
        redirect = parse_qs(parsed.query).get("redirect_uri", [""])[0]
        target = urlparse(redirect)
        port = target.port
    except ValueError:
        return None
    host = (target.hostname or "").lower()
    if host not in _LOOPBACK_HOSTS or not port:
        return None
    return str(port), target.path or "/"


def _loopback_callback(value: str) -> tuple[str, str] | None:
    """``(port, target)`` when *value* is a local callback URL carrying a code."""
    try:
        parsed = urlparse(value)
        port = parsed.port
    except ValueError:
        return None
    if parsed.scheme not in ("http", "https"):
        return None
    if (parsed.hostname or "").lower() not in _LOOPBACK_HOSTS:
        return None
    if not port or not parse_qs(parsed.query).get("code"):
        return None
    target = parsed.path or "/"
    if parsed.query:
        target += "?" + parsed.query
    return str(port), target


def _replay_callback(port: str, target: str) -> None:
    """Deliver a pasted callback to the CLI's listener on this host.

    The browser may run on another machine, but the CLI waits here: replaying
    the request against ``127.0.0.1`` completes the same flow that a same-host
    browser would have completed on its own.
    """
    connection = http.client.HTTPConnection("127.0.0.1", int(port), timeout=30)
    try:
        connection.request("GET", target)
        response = connection.getresponse()
        response.read()
        if response.status >= 400:
            raise OSError(f"the listener answered HTTP {response.status}")
    finally:
        connection.close()


def _remove_dir(path: str | None) -> None:
    if path:
        shutil.rmtree(path, ignore_errors=True)


def _read_output(pending: _Pending) -> None:
    process = pending.process
    assert process.stdout is not None
    for line in process.stdout:
        pending.lines.append(line.rstrip("\n"))
        match = _URL_RE.search(line)
        if not match or pending.url:
            continue
        url = _clean_url(match.group(0))
        if not pending.first_url:
            pending.first_url = url
        if any(hint in url for hint in _LOGIN_URL_HINTS):
            pending.url = url
            pending.url_event.set()


def _wait_for_url(pending: _Pending, timeout: float) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pending.url or _read_browser_url(pending):
            return
        if pending.process.poll() is not None:
            _drain_reader(pending)
            return
        time.sleep(0.05)


def _drain_reader(pending: _Pending) -> None:
    """Wait for the reader to consume what the process wrote before it exited.

    A short-lived failure mode (a bad flag, a missing key, a CLI that refuses
    to start) is the common case here.  Without this join the caller can report
    ``(no output)`` while the explanation sits unread in the pipe.
    """
    if pending.reader is not None:
        pending.reader.join(timeout=1.0)


def _clean_url(raw: str) -> str:
    return raw.rstrip(".,;:)\"'").strip()


def _first_line(text: str) -> str:
    for line in (text or "").splitlines():
        value = line.strip()
        if value:
            return value
    return ""


def _redact(text: str, value: str) -> str:
    if value:
        text = text.replace(value, "<redacted>")
    for pattern in _TOKEN_RES:
        text = pattern.sub("<redacted>", text)
    return text


def _terminate(pending: _Pending) -> None:
    process = pending.process
    try:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
    except OSError:
        pass
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:  # pragma: no cover - unkillable
                pass
    _remove_dir(pending.workdir)
    pending.workdir = None


def _expire(key: str, pending: _Pending) -> None:
    with _LOCK:
        if _PENDING.get(key) is not pending:
            return
        _PENDING.pop(key, None)
    _terminate(pending)


def _cancel_all() -> None:
    with _LOCK:
        pending = list(_PENDING.values())
        _PENDING.clear()
    for item in pending:
        if item.timer is not None:
            item.timer.cancel()
        _terminate(item)


atexit.register(_cancel_all)


def subprocess_env(
    overrides: dict[str, str] | None,
    unset: tuple[str, ...] = (),
) -> dict[str, str]:
    """The CLI's environment: the gateway's own plus the provider's switches.

    Mirrors the ACP transport's spawn rule, so the CLI signs in to the same
    profile the agent will use.  *unset* names variables the vendor must not
    inherit, such as an API key that would outrank the stored subscription.
    """
    env = dict(os.environ)
    for name in unset:
        env.pop(name, None)
    if overrides:
        env.update({str(k): str(v) for k, v in overrides.items()})
    return env
