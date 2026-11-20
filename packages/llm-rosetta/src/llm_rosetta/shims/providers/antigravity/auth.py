"""Sign in to Antigravity (``agy``) — the shim's own credential exchange.

This lane is unlike the others: the gateway never holds a credential.  It
starts the vendor CLI (``agy acp``) and speaks ACP to it, and ``agy`` owns the
Google sign-in, the token and the project.  So the "credential" this exchange
mints is not a string that goes on a request.  :meth:`AgyExchange.credential`
returns a fixed, non-secret marker when ``agy`` is signed in and raises
:class:`InteractiveSetupRequired` when it is not; the gateway's readiness,
scheduled re-check and routing machinery then treat a signed-out lane exactly
like any other provider that is waiting for a sign-in.

What "signed in" means
~~~~~~~~~~~~~~~~~~~~~~

It is asked of ``agy`` itself, never read from its files: the exchange spawns
``agy acp``, speaks ``initialize`` and ``session/new``, and classifies the
answer.  A session that opens is signed in; a refusal that names a sign-in
step is signed out; any other failure is reported as a failure, not as a
sign-out.  No credential file or keychain entry is read, printed or copied.

One identity per provider entry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default an entry uses the login of the gateway host, exactly as before
this exchange existed.  An entry that sets ``auth_fields.agy_profile`` to
``separate`` runs ``agy`` against its own profile directory
(``--gemini-dir <state_dir>/agy``), and signs in against it, so several
entries are several accounts.  That ``--gemini-dir`` relocates every file
``agy`` keeps was measured; that a login *lands* there is not (it needs a
human), which is why the mode is opt-in.

The sign-in
~~~~~~~~~~~

:meth:`begin_login` runs ``agy acp --login`` on the **gateway host** and
returns at once.  A reader thread keeps the last lines it printed (clipped,
scrubbed), picks out an ``https`` URL if it prints one, and when the process
exits it verifies as above.  The run is bounded (:data:`LOGIN_TIMEOUT`),
killed on cancel, and one per entry.  The argv is fixed: the agent command
from the provider's config, ``--login``, and the profile flag derived from the
provider name — no request-supplied string reaches it.

The standard library only: the core distribution has no dependencies.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from llm_rosetta.shims.auth_exchange import (
    AuthContext,
    AuthExchangeError,
    BaseAuthExchange,
    InteractiveSetupRequired,
)

#: The value :meth:`AgyExchange.credential` returns.  It is not a secret and
#: is sent nowhere: the ACP transport ignores the key ring.
SIGNED_IN_MARKER = "agy-signed-in"

#: The provider-entry field (``auth_fields``) that picks the identity.
PROFILE_FIELD = "agy_profile"
PROFILE_SEPARATE = "separate"

#: How long a sign-in may run before it is killed, in seconds.
LOGIN_TIMEOUT = 300.0
#: How long one handshake may take.
PROBE_TIMEOUT = 30.0
#: How long a positive answer is trusted, in seconds.
CACHE_TTL = 60.0
#: Output kept per sign-in: lines, characters per line, bytes read in all.
MAX_LINES = 40
MAX_LINE_CHARS = 400
MAX_OUTPUT_BYTES = 64 * 1024

DEFAULT_AGENT_ARGV: tuple[str, ...] = ("agy", "acp")

#: Wording of a refusal that means "sign in first".  The ACP error has no
#: code of its own, so the wording is the signal (the transport reads the
#: same markers).  ``failed to start`` alone is *not* one: a broken install
#: says that too.
_SIGNED_OUT_HINTS = (
    "not signed in",
    "not logged in",
    "sign in",
    "sign-in",
    "log in",
    "login",
    "authenticate",
    "unauthenticated",
    "unauthorized",
)

_URL_RE = re.compile(r"https://[^\s\"'<>\\]+")
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07")
#: Things that look like a token, dropped from anything shown to an operator.
_SECRET_RE = re.compile(
    r"(ya29\.[\w.-]+|1//[\w-]{10,}|eyJ[\w-]+\.[\w-]+\.[\w-]+|AIza[\w-]{20,}"
    r"|(?i:bearer)\s+\S+|(?i:(?:access|refresh|id)[_-]?token)\S*"
    r"|(?i:client[_-]?secret)\S*)"
)
#: Query parameters a URL we show must not carry.
_SECRET_PARAMS = frozenset(
    {"code", "token", "access_token", "refresh_token", "id_token", "client_secret"}
)

# ---------------------------------------------------------------------------
# Paths and argv
# ---------------------------------------------------------------------------


def _slug(provider: str) -> str:
    """A provider name as one safe path component.

    Characters a path cannot carry are replaced and leading dots are dropped,
    so a name such as ``../../etc`` cannot climb out of its directory.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", provider).lstrip(".") or "provider"


def _fields(ctx: AuthContext) -> Mapping[str, Any]:
    value = ctx.config.get("auth_fields") if ctx.config else None
    return value if isinstance(value, Mapping) else {}


def _separate(ctx: AuthContext) -> bool:
    """Whether the entry runs against its own agy profile.

    Raises:
        AuthExchangeError: ``agy_profile`` holds a value this exchange does
            not know; guessing would sign the wrong account in.
    """
    raw = _fields(ctx).get(PROFILE_FIELD)
    value = str(raw).strip().lower() if raw is not None else ""
    if value == "":
        return False
    if value == PROFILE_SEPARATE:
        return True
    raise AuthExchangeError(
        f"auth_fields.{PROFILE_FIELD} must be empty or {PROFILE_SEPARATE!r}"
    )


def _base_dir(ctx: AuthContext) -> Path:
    """The directory this entry may keep state in; always names the provider."""
    if ctx.state_dir:
        return Path(ctx.state_dir)
    return Path.home() / ".config" / "llm-rosetta" / "agy" / _slug(ctx.provider)


def profile_dir(ctx: AuthContext) -> Path:
    """The ``--gemini-dir`` of a separate-profile entry.

    Raises:
        AuthExchangeError: The resolved path leaves the entry's directory.
    """
    base = _base_dir(ctx)
    path = base / "agy"
    try:
        inside = path.resolve().is_relative_to(base.resolve())
    except OSError as exc:  # pragma: no cover - resolve() on a broken link
        raise AuthExchangeError(f"profile directory is not usable: {exc}") from exc
    if not inside:
        raise AuthExchangeError("the profile directory escapes its state directory")
    return path


def _prepare_profile(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(path, 0o700)
    except OSError:  # pragma: no cover - a filesystem that refuses chmod
        pass


def _agent_argv(ctx: AuthContext) -> list[str]:
    """The agent command: the entry's ``acp_argv``, else ``agy acp``."""
    raw = ctx.config.get("acp_argv") if ctx.config else None
    if isinstance(raw, (list, tuple)) and raw and all(isinstance(a, str) for a in raw):
        return list(raw)
    return list(DEFAULT_AGENT_ARGV)


def _key(ctx: AuthContext) -> str:
    return ctx.provider


def _workspace(ctx: AuthContext) -> str:
    return tempfile.gettempdir()


# ---------------------------------------------------------------------------
# Scrubbing
# ---------------------------------------------------------------------------


def scrub_line(line: str) -> str:
    """Clip one output line and drop anything that looks like a token."""
    text = _ANSI_RE.sub("", line).replace("\r", "").strip()
    text = _SECRET_RE.sub("[REDACTED]", text)
    return text[:MAX_LINE_CHARS]


def safe_url(raw: str) -> str | None:
    """*raw* when it is an ``https`` URL carrying no credential parameter."""
    url = raw.rstrip(".,;:)]}")
    if len(url) > 2000:
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme != "https" or not parts.netloc:
        return None
    for name, _ in parse_qsl(parts.query, keep_blank_values=True):
        if name.lower() in _SECRET_PARAMS:
            return None
    if _SECRET_RE.search(url):
        return None
    return url


# ---------------------------------------------------------------------------
# Asking agy whether it is signed in
# ---------------------------------------------------------------------------


async def _read_response(
    proc: asyncio.subprocess.Process, request_id: int, deadline: float
) -> dict[str, Any]:
    """Read newline-delimited JSON until the response to *request_id*."""
    assert proc.stdout is not None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise asyncio.TimeoutError
        line = await asyncio.wait_for(proc.stdout.readline(), remaining)
        if not line:
            raise EOFError
        try:
            message = json.loads(line)
        except ValueError:
            continue
        if (
            isinstance(message, dict)
            and message.get("id") == request_id
            and ("result" in message or "error" in message)
            and "method" not in message
        ):
            return message


async def _stop(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is None:
        try:
            proc.kill()
        except ProcessLookupError:  # pragma: no cover
            pass
    try:
        await asyncio.wait_for(proc.wait(), 5)
    except asyncio.TimeoutError:  # pragma: no cover
        pass


async def probe(argv: Sequence[str], cwd: str, timeout: float = PROBE_TIMEOUT) -> str:
    """Ask the agent whether it will open a session.

    Returns:
        ``"signed_in"`` when ``session/new`` succeeds, ``"signed_out"`` when it
        is refused with wording that names a sign-in.

    Raises:
        AuthExchangeError: ``agy`` is missing, times out, or fails in a way
            that is not a sign-in refusal.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=cwd,
        )
    except OSError as exc:
        raise AuthExchangeError(
            f"could not start {argv[0]!r}: {exc.strerror or exc}"
        ) from exc
    deadline = time.monotonic() + timeout

    async def call(request_id: int, method: str, params: dict[str, Any]) -> Any:
        assert proc.stdin is not None
        frame = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        proc.stdin.write((json.dumps(frame) + "\n").encode())
        await proc.stdin.drain()
        return await _read_response(proc, request_id, deadline)

    try:
        init = await call(
            1, "initialize", {"protocolVersion": 1, "clientCapabilities": {}}
        )
        if "error" in init:
            raise AuthExchangeError("the agent refused to initialize")
        reply = await call(2, "session/new", {"cwd": cwd, "mcpServers": []})
        if "result" in reply:
            session = (reply["result"] or {}).get("sessionId")
            if isinstance(session, str):
                try:
                    await call(3, "session/close", {"sessionId": session})
                except Exception:  # noqa: BLE001 - best effort; the kill follows
                    pass
            return "signed_in"
        message = str((reply.get("error") or {}).get("message") or "").lower()
        if any(hint in message for hint in _SIGNED_OUT_HINTS):
            return "signed_out"
        raise AuthExchangeError(
            "the agent could not open a session: " + scrub_line(message or "no reason")
        )
    except (asyncio.TimeoutError, EOFError) as exc:
        raise AuthExchangeError(
            "the agent did not answer the handshake"
            if isinstance(exc, asyncio.TimeoutError)
            else "the agent exited during the handshake"
        ) from exc
    finally:
        await _stop(proc)


# ---------------------------------------------------------------------------
# Remembered answers (memory only: nothing about a login is written to disk)
# ---------------------------------------------------------------------------

_STATE: dict[str, tuple[bool, float, str]] = {}
_STATE_LOCK = threading.Lock()


def _identity(ctx: AuthContext) -> str:
    return "profile:" + str(profile_dir(ctx)) if _separate(ctx) else "host"


def _remember(ctx: AuthContext, signed_in: bool) -> None:
    with _STATE_LOCK:
        _STATE[_key(ctx)] = (signed_in, time.time(), _identity(ctx))


def _recall(ctx: AuthContext) -> tuple[bool, float] | None:
    """The last answer for this entry, or ``None`` if it is stale or unknown.

    The identity is part of the memory: an entry switched from the host login
    to a separate profile must not inherit the answer for the other one.
    """
    with _STATE_LOCK:
        held = _STATE.get(_key(ctx))
    if held is None or held[2] != _identity(ctx):
        return None
    return held[0], held[1]


def reset_state() -> None:
    """Forget every remembered answer and sign-in run (for tests)."""
    with _STATE_LOCK:
        _STATE.clear()
    with _LOGINS_LOCK:
        runs = list(_LOGINS.values())
        _LOGINS.clear()
    for run in runs:
        run.cancel()


# ---------------------------------------------------------------------------
# The sign-in run
# ---------------------------------------------------------------------------


class _Login:
    """One ``agy acp --login`` run and what it printed."""

    def __init__(self, ctx: AuthContext, argv: list[str]) -> None:
        self.ctx = ctx
        self.argv = argv
        self.lock = threading.Lock()
        self.state = "running"
        self.message = ""
        self.url: str | None = None
        self.lines: deque[str] = deque(maxlen=MAX_LINES)
        self.started = time.time()
        self.timeout = float(LOGIN_TIMEOUT)
        self.proc: subprocess.Popen[bytes] | None = None
        self._timer: threading.Timer | None = None
        self._cancelled = False
        self._timed_out = False

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        env = dict(os.environ)
        env.setdefault("NO_COLOR", "1")
        try:
            self.proc = subprocess.Popen(  # noqa: S603 - argv is built above
                self.argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=_workspace(self.ctx),
                env=env,
                start_new_session=True,
            )
        except OSError as exc:
            raise AuthExchangeError(
                f"could not start {self.argv[0]!r}: {exc.strerror or exc}"
            ) from exc
        self._timer = threading.Timer(self.timeout, self._expire)
        self._timer.daemon = True
        self._timer.start()
        threading.Thread(target=self._pump, name="agy-login", daemon=True).start()

    def _kill(self) -> None:
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, AttributeError):
            proc.terminate()
        try:
            proc.wait(2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, AttributeError):
                proc.kill()

    def _expire(self) -> None:
        with self.lock:
            if self.state != "running":
                return
            self._timed_out = True
        self._kill()

    def cancel(self) -> bool:
        """Stop the run; ``True`` when it was still in progress."""
        with self.lock:
            if self.state not in ("running", "verifying"):
                return False
            self._cancelled = True
            self.state = "cancelled"
            self.message = "the sign-in was cancelled"
        if self._timer is not None:
            self._timer.cancel()
        self._kill()
        return True

    # -- the reader ---------------------------------------------------------

    def _pump(self) -> None:
        proc = self.proc
        assert proc is not None and proc.stdout is not None
        seen = 0
        for raw in iter(proc.stdout.readline, b""):
            seen += len(raw)
            if seen > MAX_OUTPUT_BYTES:
                # Keep draining so the child never blocks on a full pipe, but
                # stop recording: the bound is on what we hold.
                continue
            text = raw.decode("utf-8", errors="replace")
            for url in _URL_RE.findall(text):
                found = safe_url(url)
                if found:
                    with self.lock:
                        self.url = self.url or found
            line = scrub_line(text)
            if line:
                with self.lock:
                    self.lines.append(line)
        code = proc.wait()
        if self._timer is not None:
            self._timer.cancel()
        with self.lock:
            if self._cancelled:
                return
            if self._timed_out:
                self.state = "failed"
                self.message = (
                    f"the sign-in did not finish within {int(self.timeout)} s"
                )
                return
            if code != 0:
                self.state = "failed"
                self.message = f"agy exited with status {code}"
                return
            self.state = "verifying"
        self._verify()

    def _verify(self) -> None:
        try:
            outcome = asyncio.run(
                probe(
                    _agent_argv(self.ctx) + _agent_args(self.ctx),
                    _workspace(self.ctx),
                )
            )
            error = ""
        except AuthExchangeError as exc:
            outcome, error = "error", str(exc)
        except Exception as exc:  # noqa: BLE001 - a thread must not die silently
            outcome, error = "error", f"{type(exc).__name__}"
        with self.lock:
            if self._cancelled:
                return
            if outcome == "signed_in":
                _remember(self.ctx, True)
                self.state = "succeeded"
                self.message = "agy is signed in"
            elif outcome == "signed_out":
                _remember(self.ctx, False)
                self.state = "failed"
                self.message = "agy finished but is still not signed in"
            else:
                self.state = "failed"
                self.message = scrub_line(error) or "could not verify the sign-in"

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "state": self.state,
                "url": self.url,
                "lines": list(self.lines),
                "message": self.message,
                "started_at": self.started,
                "timeout": self.timeout,
            }


_LOGINS: dict[str, _Login] = {}
_LOGINS_LOCK = threading.Lock()

_IDLE: dict[str, Any] = {
    "state": "idle",
    "url": None,
    "lines": [],
    "message": "",
    "started_at": None,
    "timeout": LOGIN_TIMEOUT,
}


def _agent_args(ctx: AuthContext) -> list[str]:
    """Extra argv for this entry's agent: its own profile, or nothing."""
    if not _separate(ctx):
        return []
    path = profile_dir(ctx)
    _prepare_profile(path)
    return ["--gemini-dir", str(path)]


# ---------------------------------------------------------------------------
# The exchange
# ---------------------------------------------------------------------------


class AgyExchange(BaseAuthExchange):
    """Sign-in state for a lane served by the local ``agy`` CLI."""

    kind = "agent"
    consumes_key = False

    def agent_args(self, ctx: AuthContext) -> list[str]:
        """Extra argv for this entry's ``agy acp``: its own profile, or nothing."""
        return _agent_args(ctx)

    def needs_interactive_setup(self, ctx: AuthContext) -> bool:
        """``True`` only when the last check found ``agy`` signed out."""
        try:
            held = _recall(ctx)
        except AuthExchangeError:
            return False
        return held is not None and not held[0]

    async def credential(self, ctx: AuthContext) -> str:
        """Return a marker when ``agy`` is signed in.

        Raises:
            InteractiveSetupRequired: ``agy`` refused to open a session
                because nobody is signed in.
            AuthExchangeError: ``agy`` is missing or failed another way.
        """
        held = _recall(ctx)
        if held is not None and held[0] and time.time() - held[1] < CACHE_TTL:
            return SIGNED_IN_MARKER
        argv = _agent_argv(ctx) + _agent_args(ctx)
        outcome = await probe(argv, _workspace(ctx))
        if outcome == "signed_in":
            _remember(ctx, True)
            return SIGNED_IN_MARKER
        _remember(ctx, False)
        raise InteractiveSetupRequired(
            "agy is not signed in: use Sign in on the provider's Auth tab, "
            "or run `agy acp --login` on the gateway host"
        )

    def account_id(self, ctx: AuthContext) -> str | None:
        """One identity per profile; every host-login entry shares one."""
        try:
            return "agy:" + _identity(ctx)
        except AuthExchangeError:
            return None

    def describe(self, ctx: AuthContext) -> Mapping[str, Any] | None:
        """Say which agy login this entry uses, once it is known signed in."""
        held = _recall(ctx)
        if held is None or not held[0]:
            return None
        separate = _separate(ctx)
        return {
            "subject": (
                f"agy profile {_slug(ctx.provider)}"
                if separate
                else "the agy login of this host"
            ),
            "profile": PROFILE_SEPARATE if separate else "host",
            "checked_at": held[1],
        }

    # -- the sign-in --------------------------------------------------------

    def begin_login(self, ctx: AuthContext) -> Mapping[str, Any]:
        """Run ``agy acp --login`` on this host and return its first status."""
        key = _key(ctx)
        with _LOGINS_LOCK:
            running = _LOGINS.get(key)
            if running is not None:
                snap = running.snapshot()
                if snap["state"] in ("running", "verifying"):
                    return snap
            argv = _agent_argv(ctx) + ["--login"] + _agent_args(ctx)
            run = _Login(ctx, argv)
            run.start()
            _LOGINS[key] = run
        return run.snapshot()

    def login_status(self, ctx: AuthContext) -> Mapping[str, Any]:
        """The latest sign-in run's state, output and URL."""
        with _LOGINS_LOCK:
            run = _LOGINS.get(_key(ctx))
        return run.snapshot() if run is not None else dict(_IDLE)

    def cancel_login(self, ctx: AuthContext) -> bool:
        """Kill a running sign-in."""
        with _LOGINS_LOCK:
            run = _LOGINS.get(_key(ctx))
        return run.cancel() if run is not None else False

    def revoke(self, ctx: AuthContext) -> bool:
        """Sign out of a separate profile by removing it.

        An entry on the host's own login cannot be signed out from here: that
        login belongs to ``agy`` and to every other tool using it.

        Returns:
            ``False``: nothing is revoked upstream, the profile is only
            forgotten locally.

        Raises:
            AuthExchangeError: The entry uses the host login.
        """
        self.cancel_login(ctx)
        if not _separate(ctx):
            raise AuthExchangeError(
                "this entry uses the agy login of the gateway host; sign out "
                "with agy on that machine, or give the entry its own profile"
            )
        path = profile_dir(ctx)
        shutil.rmtree(path, ignore_errors=True)
        _remember(ctx, False)
        return False


EXCHANGE = AgyExchange()
