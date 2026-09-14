"""Background token refresh for providers with ``token_command``.

When a provider config specifies ``token_command`` (an argv list), the
gateway shells out to it at startup to seed the API key, then re-runs it
every ``token_refresh_interval`` seconds in a background asyncio task.

The command's stdout is captured as the new API key (comma-separated for
multi-key round-robin).  On failure the old key is kept; after 5
consecutive failures the log level escalates from WARNING to ERROR.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .provider_info import ProviderInfo

logger = logging.getLogger(__name__)

FAILURE_ESCALATION_THRESHOLD = 5
COMMAND_TIMEOUT = 30


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
            now = datetime.now(timezone.utc).isoformat()
            pinfo.token_status = {
                "enabled": True,
                "last_refresh": now,
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
