"""Reading a lane's allowance from the vendor CLI the shim declares.

Not every CLI reports the allowance in a turn.  A shim may instead declare
``quota.local``: a command that prints the account as JSON.  :func:`refresh_local`
runs it with the lane's profile and hands stdout to the probe's ``parse_local``;
the reading lands in the app's :class:`~.lane_allowance.LaneAllowanceStore`, so
the card cannot tell the two sources apart.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import signal
import tempfile
from datetime import datetime, timezone
from typing import Any

from llm_rosetta.contracts.quota import QuotaProbeError

from ..transport.acp.usage import ACP_QUOTA_PROBE
from .lane_allowance import LaneAllowanceStore, lane_credential
from .models import STATUS_ERROR, STATUS_OK, QuotaReading

#: How much of a local command's output is kept for decoding and errors.
_MAX_LOCAL_OUTPUT = 1 << 20

#: How long a killed command group may take to disappear before the read gives
#: up on reaping it.  Nothing waits on the result after this.
_KILL_GRACE = 5.0


def _expand_local(value: str, provider_info: Any) -> str:
    """Expand the profile placeholders a local quota command may carry."""
    home = str(getattr(provider_info, "acp_state_dir", "") or "")
    argv = tuple(getattr(provider_info, "acp_argv", ()) or ())
    executable = str(argv[0]) if argv else ""
    return (
        value.replace("{acp_home}", home)
        .replace("{acp_state_dir}", home)
        .replace("{acp_argv0}", executable)
    )


def _local_env(provider_info: Any, extra: dict[str, str] | None) -> dict[str, str]:
    """The command's environment: the lane's, then the declaration's."""
    env = dict(os.environ)
    for name in getattr(provider_info, "acp_env_unset", ()) or ():
        env.pop(str(name), None)
    for name, value in (getattr(provider_info, "acp_env", None) or {}).items():
        env[str(name)] = _expand_local(str(value), provider_info)
    for name, value in (extra or {}).items():
        env[str(name)] = _expand_local(str(value), provider_info)
    return env


def _local_workdir(provider_info: Any) -> str:
    """A writable directory for the vendor CLI."""
    home = getattr(provider_info, "acp_state_dir", None)
    if home and os.path.isdir(str(home)):
        return str(home)
    return tempfile.gettempdir()


def _json_of_output(text: str) -> Any:
    """Decode the JSON a vendor CLI printed, tolerating a banner or a footer."""
    stripped = text.strip()
    if not stripped:
        raise QuotaProbeError("the quota command printed nothing")
    try:
        return json.loads(stripped)
    except ValueError:
        start = stripped.find("{")
        if start >= 0:
            try:
                value, _end = json.JSONDecoder().raw_decode(stripped[start:])
                return value
            except ValueError:
                pass
    raise QuotaProbeError("the quota command did not print JSON: " + stripped[:300])


def _error_reading(provider_info: Any, probe_id: str, message: str) -> QuotaReading:
    """A failed local read, stored like any other reading."""
    return QuotaReading(
        probe=probe_id,
        credential=lane_credential(provider_info),
        status=STATUS_ERROR,
        error=message,
    )


def _kill_group(process: Any, pgid: int | None = None) -> None:
    """Kill the command and any helper left holding its pipes.

    *pgid* is captured at spawn: once the direct child is reaped,
    ``os.getpgid(pid)`` no longer resolves, but a helper in the same group can
    still be alive and hold the pipes.
    """
    if pgid is not None and hasattr(os, "killpg"):
        try:
            os.killpg(pgid, signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    pid = getattr(process, "pid", None)
    if pid is not None and hasattr(os, "killpg"):
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        process.kill()
    except ProcessLookupError:
        pass


def _close_transport(process: Any) -> None:
    """Close the subprocess transport while the loop that owns it still runs.

    An early exit leaves the pipe protocols pending; their ``__del__`` would
    close them after the loop is gone (a warning in tests, a late fd release
    in a long-lived gateway).
    """
    transport = getattr(process, "_transport", None)
    close = getattr(transport, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001 - teardown must not mask the reading
            pass


async def _read_capped(
    stream: Any, limit: int, process: Any, pgid: int | None
) -> tuple[bytes, bool]:
    """Read *stream* up to *limit* bytes; kill the command beyond it.

    A vendor CLI has no reason to print more than this, and a parser given a
    truncated body would blame its own format instead of the size.  Killing on
    overflow also unblocks the write side, so the read cannot outrun the cap.
    """
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            return b"".join(chunks)[:limit], False
        size += len(chunk)
        chunks.append(chunk)
        if size > limit:
            _kill_group(process, pgid)
            while True:
                rest = await stream.read(65536)
                if not rest:
                    break
            return b"".join(chunks)[:limit], True


async def refresh_local(
    store: LaneAllowanceStore,
    provider_info: Any,
    probe: Any,
    argv: tuple[str, ...],
    env: dict[str, str] | None = None,
    *,
    timeout: float = 30.0,
) -> QuotaReading:
    """Run a shim-declared local quota command and store its reading.

    The command is the lane's own vendor CLI: it sees the same profile the
    agent does, so the reading belongs to the same account.  The probe's
    ``parse_local`` decodes the JSON; a failure is stored as an error reading
    so the next Quota open does not immediately rerun it.  A failure never
    replaces a good reading from the agent's own turn.

    The command runs in its own process group, and *timeout* bounds the whole
    read: on expiry, or if the caller is cancelled, the group is killed and
    waited for at most :data:`_KILL_GRACE` seconds.  A vendor CLI that leaves
    a helper holding stdout cannot stretch the call, and the group is killed
    even when the direct child already exited.

    Args:
        store: Where the reading is kept.
        provider_info: The lane whose profile and environment the command uses.
        probe: The shim probe; it must provide ``parse_local``.
        argv: Declared command, with profile placeholders.
        env: Declaration-owned environment, applied over the lane's.
        timeout: Seconds the command may run, its output included.

    Returns:
        The reading; a cancelled call raises ``CancelledError`` instead.
    """
    probe_id = str(getattr(probe, "id", ACP_QUOTA_PROBE))
    command = [_expand_local(str(part), provider_info) for part in argv]
    if not command or not all(command):
        reading = _error_reading(provider_info, probe_id, "the quota command is empty")
        store.store_local(provider_info, reading)
        return reading
    parse_local = getattr(probe, "parse_local", None)
    if not callable(parse_local):
        reading = _error_reading(
            provider_info,
            probe_id,
            f"probe {probe_id!r} does not decode a local quota command",
        )
        store.store_local(provider_info, reading)
        return reading
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_local_env(provider_info, env),
            cwd=_local_workdir(provider_info),
            start_new_session=True,
        )
    except OSError as exc:
        reading = _error_reading(
            provider_info, probe_id, f"the quota command could not start: {exc}"
        )
        store.store_local(provider_info, reading)
        return reading

    # The command is its own session leader, so this group id outlives the
    # direct child: a helper it left behind still shares it.
    pgid = process.pid if hasattr(os, "killpg") else None
    timed_out = False
    overflowed = False
    completed = False
    out = b""
    err = b""
    try:
        (out, overflowed), (err, err_overflow), _ = await asyncio.wait_for(
            asyncio.gather(
                _read_capped(process.stdout, _MAX_LOCAL_OUTPUT, process, pgid),
                _read_capped(process.stderr, _MAX_LOCAL_OUTPUT, process, pgid),
                process.wait(),
            ),
            timeout=max(0.1, float(timeout)),
        )
        overflowed = overflowed or err_overflow
        completed = True
    except asyncio.TimeoutError:
        timed_out = True
    finally:
        if not completed:
            # Timeout, cancellation, or an overflow: kill the group even when
            # the direct child already exited, or a helper holding the pipes
            # survives the read.
            _kill_group(process, pgid)
            if process.returncode is None:
                try:
                    await asyncio.wait_for(asyncio.shield(process.wait()), _KILL_GRACE)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
            _close_transport(process)

    if timed_out:
        reading = _error_reading(
            provider_info,
            probe_id,
            f"the quota command did not finish in {timeout:g}s",
        )
        store.store_local(provider_info, reading)
        return reading
    if overflowed:
        reading = _error_reading(
            provider_info,
            probe_id,
            f"the quota command printed more than {_MAX_LOCAL_OUTPUT} bytes",
        )
        store.store_local(provider_info, reading)
        return reading
    text = out.decode("utf-8", "replace")
    error_text = err.decode("utf-8", "replace").strip()
    if process.returncode != 0:
        detail = error_text or text.strip()[:300] or f"exit code {process.returncode}"
        reading = _error_reading(
            provider_info, probe_id, f"the quota command failed: {detail}"
        )
        store.store_local(provider_info, reading)
        return reading
    try:
        payload = _json_of_output(text)
        result = parse_local(payload)
        if inspect.isawaitable(result):
            result = await result
        plan, windows = result
    except QuotaProbeError as exc:
        reading = _error_reading(provider_info, probe_id, str(exc))
    except Exception as exc:  # noqa: BLE001 - a shim parser must not break admin
        reading = _error_reading(
            provider_info, probe_id, f"the quota command's output was not read: {exc}"
        )
    else:
        reading = QuotaReading(
            probe=probe_id,
            credential=lane_credential(provider_info),
            windows=list(windows),
            plan=plan,
            status=STATUS_OK,
            read_at=datetime.now(timezone.utc),
        )
    store.store_local(provider_info, reading)
    return reading
