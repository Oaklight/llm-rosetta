"""Background credential refresh for providers that mint their own.

A provider mints its credential when it names a ``token_command`` (an argv
list) or when its shim ships an ``auth.py`` exchange.  Either way the gateway
obtains the credential at startup, re-obtains it every
``token_refresh_interval`` seconds in a background asyncio task, and retries
once on an upstream 401.

The minted credential replaces the API key (comma-separated for multi-key
round-robin).  On failure the old key is kept; after 5 consecutive failures
the log level escalates from WARNING to ERROR.  An exchange that reports it
needs an interactive sign-in is not a failure to retry on a schedule — it is
reported as such until an operator signs in.

Reactive refresh
~~~~~~~~~~~~~~~~

When the proxy receives a 401 from upstream, it can call
:func:`force_refresh` to trigger an immediate out-of-cycle refresh.
A per-provider lock prevents concurrent refreshes, and a debounce
window prevents redundant refreshes from bursts of 401s.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from llm_rosetta.shims.auth_exchange import (
    AuthContext,
    InteractiveSetupRequired,
)

if TYPE_CHECKING:
    from .provider_info import ProviderInfo

logger = logging.getLogger(__name__)

FAILURE_ESCALATION_THRESHOLD = 5
COMMAND_TIMEOUT = 30

#: The longest failure text a provider keeps in ``token_status``.
ERROR_TEXT_LIMIT = 300


def clip_error_text(error: str | None) -> str | None:
    """Bound the size of the diagnostics a mint failure stores.

    Clipping keeps a failure readable in a panel row; it does not decide
    whether the text contains a credential.  The mint path never adds a token
    to its message, but a command that prints one in its own diagnostics has
    that line quoted, and the prefix survives.  What the bound does cover is
    every surface that reads ``token_status`` — the credential endpoints,
    ``/admin/api/config``, and anything that copies the dict — so the same
    panel-sized text is what each of them renders.
    """
    if isinstance(error, str) and len(error) > ERROR_TEXT_LIMIT:
        return error[: ERROR_TEXT_LIMIT - 1] + "…"
    return error


def public_token_status(status: dict[str, Any] | None) -> dict[str, Any]:
    """A copy of *status* whose error text is bounded.

    Every surface that renders the refresh log goes through this: the
    credential endpoints and ``/admin/api/config`` alike, so a caller that
    forgets cannot be the only one that leaks.
    """
    out = dict(status or {})
    error = out.get("last_error")
    if error is not None:
        out["last_error"] = clip_error_text(error)
    return out


# ---------------------------------------------------------------------------
# Reactive refresh coordination
# ---------------------------------------------------------------------------

# Single-process state: the gateway runs one event loop, so module-level
# dicts are shared across all request handlers and the background loop.
_reactive_locks: dict[str, asyncio.Lock] = {}
_reactive_timestamps: dict[str, float] = {}  # time.monotonic()
_REACTIVE_DEBOUNCE = 5.0  # seconds

# If a refresh (reactive or scheduled) happened within this fraction of
# token_refresh_interval, the scheduled loop skips its next cycle to
# avoid redundant subprocess invocations.
_SCHEDULED_SKIP_FRACTION = 0.5


def _get_lock(name: str) -> asyncio.Lock:
    """Get or create a per-provider asyncio.Lock (lazy, event-loop-safe)."""
    if name not in _reactive_locks:
        _reactive_locks[name] = asyncio.Lock()
    return _reactive_locks[name]


def _status_keeping_enabled(prev: dict[str, Any], **fields: Any) -> dict[str, Any]:
    """A new status dict that keeps whatever ``enabled`` already said.

    Only a refresh loop knows whether one is alive.  A one-off mint — the 401
    path, the panel's refresh button — must not turn a dead loop's ``False``
    back into ``True``: the panel would stop saying the token will not renew
    while nothing has started renewing it.
    """
    status = dict(fields)
    if "enabled" in prev:
        status["enabled"] = prev["enabled"]
    return status


async def force_refresh(pinfo: ProviderInfo, *, debounce: bool = True) -> bool:
    """Reactively re-mint the credential for *pinfo*.

    Called by the proxy on upstream 401.  Returns ``True`` if a fresh
    credential was obtained (or a very recent refresh already covered it),
    ``False`` if the provider mints no credential of its own or the refresh
    failed.

    *debounce* is for the 401 path: a burst of failures must not launch a
    subprocess each.  An operator pressing "refresh now" passes ``False`` —
    the point of the button is a new credential even if one was minted seconds
    ago.
    """
    if not pinfo.mints_credential:
        return False

    lock = _get_lock(pinfo.name)
    async with lock:
        now = time.monotonic()
        last = _reactive_timestamps.get(pinfo.name, 0.0)
        if debounce and (now - last) < _REACTIVE_DEBOUNCE:
            return True

        try:
            token = await mint_credential(pinfo)
            pinfo.key_ring.refresh(token)
            _reactive_timestamps[pinfo.name] = time.monotonic()
            pinfo.token_status = _status_keeping_enabled(
                pinfo.token_status or {},
                last_refresh=datetime.now(timezone.utc).isoformat(),
                consecutive_failures=0,
                last_error=None,
            )
            logger.info("Reactive credential refresh for '%s' succeeded", pinfo.name)
            return True
        except Exception as exc:
            interactive = isinstance(exc, InteractiveSetupRequired)
            logger.warning(
                "Reactive credential refresh for '%s' failed: %s", pinfo.name, exc
            )
            prev = pinfo.token_status or {}
            pinfo.token_status = _status_keeping_enabled(
                prev,
                last_refresh=prev.get("last_refresh"),
                consecutive_failures=prev.get("consecutive_failures", 0) + 1,
                last_error=clip_error_text(str(exc)),
                # A wait for a human is still true after an unrelated failure
                # (a timeout, a 502): the panel's "a sign-in is needed" line
                # must not disappear because this attempt failed differently.
                # Only a successful mint clears it, by not writing the key.
                interactive_setup_required=(
                    interactive or prev.get("interactive_setup_required", False)
                ),
            )
            return False


def reset_reactive_state() -> None:
    """Clear all reactive-refresh state (for testing)."""
    _reactive_locks.clear()
    _reactive_timestamps.clear()
    _active_refreshers.clear()


# ---------------------------------------------------------------------------
# Token command runners
# ---------------------------------------------------------------------------


async def mint_credential(pinfo: ProviderInfo) -> str:
    """Obtain a usable credential for *pinfo*.

    The single seam every caller uses — startup seeding, the scheduled
    refresher and the reactive 401 retry — so a provider's credential form
    never leaks into the caller.  A provider-level ``token_command`` and a
    shim-shipped ``auth.py`` are mutually exclusive by construction; the
    builder attaches exactly one.

    Raises:
        RuntimeError: No credential source, or the command failed.
        InteractiveSetupRequired: The exchange needs a human to sign in.
        AuthExchangeError: The exchange failed for another reason.
    """
    if pinfo.auth_exchange is not None:
        ctx = pinfo.auth_context or AuthContext(provider=pinfo.name)
        return await pinfo.auth_exchange.credential(ctx)
    if pinfo.token_command is not None:
        return await run_token_command(pinfo.token_command, env=command_env(pinfo))
    raise RuntimeError(f"Provider '{pinfo.name}' mints no credential")


def command_env(pinfo: ProviderInfo) -> dict[str, str] | None:
    """Return the environment to run *pinfo*'s command with, or ``None``.

    A shim whose exchange takes the operator's key passes it under the declared
    variable; everything else inherits the gateway's environment unchanged.
    The key is handed over in the environment rather than argv, where the
    process table would expose it.
    """
    if not pinfo.token_command_env or not pinfo.token_input:
        return None
    return {**os.environ, pinfo.token_command_env: pinfo.token_input}


def run_token_command_sync(
    argv: list[str],
    *,
    timeout: float = COMMAND_TIMEOUT,
    env: dict[str, str] | None = None,
) -> str:
    """Run *argv* synchronously and return stripped stdout.

    Used at startup (before the event loop is running) to seed the
    initial API key.  Raises on non-zero exit, empty output, or timeout.
    """
    result = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"token_command {argv} exited with {result.returncode}: "
            f"{result.stderr.strip()}"
        )
    token = result.stdout.strip()
    if not token:
        raise RuntimeError(f"token_command {argv} produced empty output")
    return token


async def run_token_command(
    argv: list[str],
    *,
    timeout: float = COMMAND_TIMEOUT,
    env: dict[str, str] | None = None,
) -> str:
    """Run *argv* asynchronously and return stripped stdout."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise RuntimeError(f"token_command {argv} timed out after {timeout}s") from None
    if proc.returncode != 0:
        raise RuntimeError(
            f"token_command {argv} exited with {proc.returncode}: "
            f"{stderr.decode(errors='replace').strip()}"
        )
    token = stdout.decode().strip()
    if not token:
        raise RuntimeError(f"token_command {argv} produced empty output")
    return token


# ---------------------------------------------------------------------------
# Background refresh loop
# ---------------------------------------------------------------------------


#: The loop each provider's ``token_status['enabled']`` speaks for.
#: ``start_token_refreshers`` can run twice for one provider (a reload, or a
#: caller with its own bookkeeping); the newest task is the one its callers
#: keep, and only that task finishing may clear the flag.
_active_refreshers: dict[str, asyncio.Task] = {}


def _mark_refresher(pinfo: Any, running: bool) -> None:
    """Record whether a refresh loop is alive for *pinfo*.

    ``token_status['enabled']`` is what the panel reads to say whether the
    token will renew.  Before this, every writer set it to ``True`` and nothing
    ever cleared it, so a loop that a reload cancelled — or one that died —
    still read as renewing the token.
    """
    status = dict(pinfo.token_status or {})
    status["enabled"] = running
    pinfo.token_status = status


def _refresher_ended(name: str, task: asyncio.Task, pinfo: Any) -> None:
    """Clear *pinfo*'s flag only when the loop that ended is the current one.

    A second loop for the same provider does not make the first one's exit
    meaningful: the flag tracks the newest task, which is the one its callers
    keep.  Without this check, cancelling the older of two loops would report
    "will not renew" while a live task still refreshes the ring.
    """
    if _active_refreshers.get(name) is task:
        _active_refreshers.pop(name, None)
        _mark_refresher(pinfo, False)


async def start_token_refreshers(
    providers: dict[str, ProviderInfo],
) -> list[asyncio.Task]:
    """Create background refresh tasks for every provider that mints its own."""
    tasks: list[asyncio.Task] = []
    for name, pinfo in providers.items():
        if pinfo.mints_credential:
            task = asyncio.create_task(
                _refresh_loop(pinfo), name=f"token-refresh-{name}"
            )
            _active_refreshers[name] = task
            _mark_refresher(pinfo, True)
            # A cancelled or crashed loop leaves the last token in the ring;
            # the status must stop claiming it will be renewed — but only when
            # the task that ended is the one this provider's flag speaks for.
            task.add_done_callback(lambda t, n=name, p=pinfo: _refresher_ended(n, t, p))
            tasks.append(task)
            logger.info(
                "Started token refresh for '%s' (interval=%ds)",
                name,
                pinfo.token_refresh_interval,
            )
    return tasks


async def _refresh_loop(pinfo: ProviderInfo) -> None:
    if not pinfo.mints_credential:
        return
    consecutive_failures = 0
    while True:
        await asyncio.sleep(pinfo.token_refresh_interval)

        # Skip if a refresh (reactive or scheduled) happened recently
        last = _reactive_timestamps.get(pinfo.name, 0.0)
        skip_threshold = pinfo.token_refresh_interval * _SCHEDULED_SKIP_FRACTION
        if last and (time.monotonic() - last) < skip_threshold:
            logger.debug(
                "Skipping scheduled refresh for '%s' — reactive refresh was recent",
                pinfo.name,
            )
            continue

        try:
            token = await mint_credential(pinfo)
            old_count = pinfo.key_ring.refresh(token)
            if old_count is not None:
                logger.warning(
                    "credential source for '%s': key count changed %d -> %d, "
                    "affinity mappings may shift",
                    pinfo.name,
                    old_count,
                    len(pinfo.key_ring),
                )
            consecutive_failures = 0
            now_iso = datetime.now(timezone.utc).isoformat()
            _reactive_timestamps[pinfo.name] = time.monotonic()
            pinfo.token_status = {
                "enabled": True,
                "last_refresh": now_iso,
                "consecutive_failures": 0,
                "last_error": None,
            }
            logger.info("Refreshed credential for provider '%s'", pinfo.name)
        except Exception as exc:
            interactive = isinstance(exc, InteractiveSetupRequired)
            consecutive_failures += 1
            # Waiting for a human is not a failure that escalates: the loop
            # keeps its interval, the status says what is needed, and the
            # operator sees it in the panel rather than an ERROR per cycle.
            escalated = (
                not interactive and consecutive_failures >= FAILURE_ESCALATION_THRESHOLD
            )
            level = logging.ERROR if escalated else logging.WARNING
            logger.log(
                level,
                "credential source for '%s' failed (attempt %d): %s",
                pinfo.name,
                consecutive_failures,
                exc,
            )
            prev = pinfo.token_status or {}
            pinfo.token_status = {
                "enabled": True,
                "last_refresh": prev.get("last_refresh"),
                "consecutive_failures": consecutive_failures,
                "last_error": clip_error_text(str(exc)),
                # Same rule as force_refresh: an unrelated failure (a
                # timeout, a 502) does not cancel a sign-in that is still
                # needed, and the panel's "sign-in needed" line is the only
                # place this key renders.  A successful cycle clears it by
                # not writing the key.
                "interactive_setup_required": (
                    interactive or prev.get("interactive_setup_required", False)
                ),
            }
