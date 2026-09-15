"""Background token refresh for providers with ``token_command``.

When a provider config specifies ``token_command`` (an argv list), the
gateway shells out to it at startup to seed the API key, then re-runs it
every ``token_refresh_interval`` seconds in a background asyncio task.

The command's stdout is captured as the new API key (comma-separated for
multi-key round-robin).  On failure the old key is kept; after 5
consecutive failures the log level escalates from WARNING to ERROR.

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
import subprocess
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .provider_info import ProviderInfo

logger = logging.getLogger(__name__)

FAILURE_ESCALATION_THRESHOLD = 5
COMMAND_TIMEOUT = 30

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


async def force_refresh(pinfo: ProviderInfo) -> bool:
    """Reactively refresh the token for *pinfo* via its ``token_command``.

    Called by the proxy on upstream 401.  Returns ``True`` if a fresh
    token was obtained (or a very recent refresh already covered it),
    ``False`` if no ``token_command`` is configured or the refresh failed.
    """
    if pinfo.token_command is None:
        return False

    lock = _get_lock(pinfo.name)
    async with lock:
        now = time.monotonic()
        last = _reactive_timestamps.get(pinfo.name, 0.0)
        if (now - last) < _REACTIVE_DEBOUNCE:
            return True

        try:
            token = await run_token_command(pinfo.token_command)
            pinfo.key_ring.refresh(token)
            _reactive_timestamps[pinfo.name] = time.monotonic()
            pinfo.token_status = {
                "enabled": True,
                "last_refresh": datetime.now(timezone.utc).isoformat(),
                "consecutive_failures": 0,
                "last_error": None,
            }
            logger.info("Reactive token refresh for '%s' succeeded", pinfo.name)
            return True
        except Exception as exc:
            logger.warning(
                "Reactive token refresh for '%s' failed: %s", pinfo.name, exc
            )
            prev = pinfo.token_status or {}
            pinfo.token_status = {
                "enabled": True,
                "last_refresh": prev.get("last_refresh"),
                "consecutive_failures": prev.get("consecutive_failures", 0) + 1,
                "last_error": str(exc),
            }
            return False


def reset_reactive_state() -> None:
    """Clear all reactive-refresh state (for testing)."""
    _reactive_locks.clear()
    _reactive_timestamps.clear()


# ---------------------------------------------------------------------------
# Token command runners
# ---------------------------------------------------------------------------


def run_token_command_sync(argv: list[str], *, timeout: float = COMMAND_TIMEOUT) -> str:
    """Run *argv* synchronously and return stripped stdout.

    Used at startup (before the event loop is running) to seed the
    initial API key.  Raises on non-zero exit, empty output, or timeout.
    """
    result = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
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
    argv: list[str], *, timeout: float = COMMAND_TIMEOUT
) -> str:
    """Run *argv* asynchronously and return stripped stdout."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
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


async def start_token_refreshers(
    providers: dict[str, ProviderInfo],
) -> list[asyncio.Task]:
    """Create background refresh tasks for all providers with ``token_command``."""
    tasks: list[asyncio.Task] = []
    for name, pinfo in providers.items():
        if pinfo.token_command:
            task = asyncio.create_task(
                _refresh_loop(pinfo), name=f"token-refresh-{name}"
            )
            tasks.append(task)
            logger.info(
                "Started token refresh for '%s' (interval=%ds)",
                name,
                pinfo.token_refresh_interval,
            )
    return tasks


async def _refresh_loop(pinfo: ProviderInfo) -> None:
    if pinfo.token_command is None:
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
            token = await run_token_command(pinfo.token_command)
            old_count = pinfo.key_ring.refresh(token)
            if old_count is not None:
                logger.warning(
                    "token_command for '%s': key count changed %d -> %d, "
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
            logger.info("Refreshed token for provider '%s'", pinfo.name)
        except Exception as exc:
            consecutive_failures += 1
            level = (
                logging.ERROR
                if consecutive_failures >= FAILURE_ESCALATION_THRESHOLD
                else logging.WARNING
            )
            logger.log(
                level,
                "token_command for '%s' failed (attempt %d): %s",
                pinfo.name,
                consecutive_failures,
                exc,
            )
            pinfo.token_status = {
                "enabled": True,
                "last_refresh": (pinfo.token_status or {}).get("last_refresh"),
                "consecutive_failures": consecutive_failures,
                "last_error": str(exc),
            }
