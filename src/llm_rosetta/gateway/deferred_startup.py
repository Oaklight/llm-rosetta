"""Deferred startup manager for the gateway.

Moves expensive initialization tasks — token-command execution, counter
rebuilds, and database backfills — out of the synchronous startup path
so the server socket starts accepting connections immediately.

All deferred work runs as ``asyncio.Task``s created by :meth:`start`
and cancelled by :meth:`shutdown`.  Heavy synchronous I/O (SQLite
queries) is dispatched to the default thread executor via
``run_in_executor``.

Provider readiness is tracked per-provider: the proxy routing layer
skips providers whose initial token fetch has not yet completed (or
has failed), and the ``/health/ready`` endpoint reflects overall
initialization progress.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider initialization state
# ---------------------------------------------------------------------------


class ProviderInitState(Enum):
    """Lifecycle state of a provider's deferred token initialization."""

    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class ProviderNotReady(Exception):
    """Raised when all providers for a model are still initializing or have failed."""

    def __init__(self, model: str, *, failed: bool = False) -> None:
        self.model = model
        self.failed = failed
        state = "failed" if failed else "initializing"
        super().__init__(
            f"All providers for model '{model}' are {state}; please retry shortly"
        )


# ---------------------------------------------------------------------------
# DeferredStartup manager
# ---------------------------------------------------------------------------


class DeferredStartup:
    """Coordinates background initialization tasks after the server starts.

    Instantiated in :func:`~gateway.app.run_gateway` and attached to the
    app as ``app.deferred_startup``.
    """

    def __init__(self, config: Any, app: Any) -> None:
        self._config = config
        self._app = app
        self._tasks: list[asyncio.Task] = []

        # Provider readiness: only providers with token_command need tracking
        self._provider_states: dict[str, ProviderInitState] = {}
        for name, pinfo in config.providers.items():
            if pinfo.token_command is not None:
                self._provider_states[name] = ProviderInitState.PENDING

        # Task-level completion tracking
        self._pending_tasks: set[str] = set()

    # -- Provider readiness -------------------------------------------------

    def provider_state(self, name: str) -> ProviderInitState:
        """Return the init state for *name*, defaulting to READY."""
        return self._provider_states.get(name, ProviderInitState.READY)

    # -- Overall readiness --------------------------------------------------

    def is_fully_ready(self) -> bool:
        """True when all deferred work has completed."""
        if self._pending_tasks:
            return False
        return all(s == ProviderInitState.READY for s in self._provider_states.values())

    def status(self) -> dict[str, Any]:
        """Structured status dict for the ``/health/ready`` endpoint."""
        providers: dict[str, str] = {}
        for name, state in self._provider_states.items():
            providers[name] = state.value
        return {
            "ready": self.is_fully_ready(),
            "pending_tasks": sorted(self._pending_tasks),
            "providers": providers,
        }

    # -- Lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Launch all deferred tasks.  Returns immediately."""
        # Token seeding
        if self._provider_states:
            self._pending_tasks.add("token_seed")
            self._tasks.append(
                asyncio.create_task(self._seed_tokens(), name="deferred-token-seed")
            )

        # Counter rebuild
        if getattr(self._app, "_counter_rebuild_needed", False):
            self._pending_tasks.add("counter_rebuild")
            self._tasks.append(
                asyncio.create_task(
                    self._rebuild_counters(), name="deferred-counter-rebuild"
                )
            )

        # Database backfills
        persistence = getattr(self._app, "persistence", None)
        if persistence is not None:
            self._pending_tasks.add("backfills")
            self._tasks.append(
                asyncio.create_task(self._run_backfills(), name="deferred-backfills")
            )

    async def shutdown(self) -> None:
        """Cancel any in-flight deferred tasks."""
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()

    # -- Token seeding (parallel async) -------------------------------------

    async def _seed_one_provider(self, name: str) -> None:
        """Fetch the initial token for a single provider."""
        from .transport.token_refresh import run_token_command

        pinfo = self._config.providers[name]
        assert pinfo.token_command is not None

        try:
            token = await run_token_command(pinfo.token_command)
            pinfo.key_ring.refresh(token)
            pinfo.token_status = {
                "enabled": True,
                "last_refresh": datetime.now(timezone.utc).isoformat(),
                "consecutive_failures": 0,
                "last_error": None,
            }
            self._provider_states[name] = ProviderInitState.READY
            logger.info("Seeded API key for '%s' via token_command", name)
        except Exception as exc:
            self._provider_states[name] = ProviderInitState.FAILED
            pinfo.token_status = {
                "enabled": True,
                "last_refresh": None,
                "consecutive_failures": 1,
                "last_error": str(exc),
            }
            logger.error("Failed to seed API key for '%s': %s", name, exc)

    async def _seed_tokens(self) -> None:
        """Seed all token_command providers in parallel, then start refresh loops."""
        from .transport.token_refresh import start_token_refreshers

        pending = [
            name
            for name, state in self._provider_states.items()
            if state == ProviderInitState.PENDING
        ]

        # Parallel initial fetch
        await asyncio.gather(
            *(self._seed_one_provider(name) for name in pending),
            return_exceptions=True,
        )

        # Start background refresh loops only for providers that succeeded
        ready_providers = {
            name: pinfo
            for name, pinfo in self._config.providers.items()
            if (
                pinfo.token_command is not None
                and self._provider_states.get(name) == ProviderInitState.READY
            )
        }
        if ready_providers:
            refresh_tasks = await start_token_refreshers(ready_providers)
            # Store on app so run_gateway's finally block can cancel them
            existing = getattr(self._app, "_token_refresh_tasks", [])
            existing.extend(refresh_tasks)
            setattr(self._app, "_token_refresh_tasks", existing)

        self._pending_tasks.discard("token_seed")
        logger.info(
            "Token seeding complete: %d ready, %d failed",
            sum(
                1
                for s in self._provider_states.values()
                if s == ProviderInitState.READY
            ),
            sum(
                1
                for s in self._provider_states.values()
                if s == ProviderInitState.FAILED
            ),
        )

    # -- Counter rebuild (thread executor) ----------------------------------

    async def _rebuild_counters(self) -> None:
        """Rebuild metrics counters in a background thread, then merge."""
        from llm_rosetta.observability import MetricsCollector

        metrics = getattr(self._app, "metrics", None)
        persistence = getattr(self._app, "persistence", None)
        if metrics is None or persistence is None:
            self._pending_tasks.discard("counter_rebuild")
            return

        # Snapshot current state *before* the rebuild starts (on event loop)
        pre_snapshot = metrics.export_counters()

        loop = asyncio.get_running_loop()

        def _sync_rebuild() -> dict:
            tmp = MetricsCollector()
            count = tmp.rebuild_counters(persistence.iter_log_rows_for_rebuild())
            logger.info("Background counter rebuild processed %d rows", count)
            return tmp.export_counters()

        logger.info("Starting background counter rebuild")
        baseline = await loop.run_in_executor(None, _sync_rebuild)

        # Merge back on the event-loop thread
        metrics.merge_rebuild(baseline, pre_snapshot)
        persistence.save_metrics(metrics.export_counters())

        self._pending_tasks.discard("counter_rebuild")
        logger.info("Counter rebuild complete, metrics merged")

    # -- Backfills (thread executor) ----------------------------------------

    async def _run_backfills(self) -> None:
        """Run all database backfills sequentially in a background thread."""
        persistence = getattr(self._app, "persistence", None)
        if persistence is None:
            self._pending_tasks.discard("backfills")
            return

        config = self._config
        keystore = getattr(self._app, "keystore", None)

        def _sync_backfills() -> dict[str, int]:
            results: dict[str, int] = {}
            logger.info("Starting deferred database backfills")

            # 1. Backfill provider names
            model_to_provider = {
                model: route.provider_names[0] for model, route in config.models.items()
            }
            results["provider_names"] = (
                persistence.backfill_provider_names(model_to_provider) or 0
            )

            # 2. Backfill API key last_used
            if keystore is not None:
                results["key_last_used"] = (
                    keystore.backfill_last_used(persistence.db_path) or 0
                )

            # 3. Backfill error dump log IDs
            aliases = config.model_upstream_names if config else {}
            results["error_dump_log_ids"] = (
                persistence.backfill_error_dump_log_ids(model_aliases=aliases) or 0
            )

            logger.info("Deferred backfills complete: %s", results)
            return results

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _sync_backfills)

        self._pending_tasks.discard("backfills")
