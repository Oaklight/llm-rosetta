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

    _RETRY_DELAY = 30  # seconds before retrying a failed token_command

    def __init__(self, config: Any, app: Any) -> None:
        self._config = config
        self._app = app
        self._tasks: list[asyncio.Task] = []

        # The seeding pass, held in its own slot so the reload path can
        # supersede it without touching the other deferred tasks.  The
        # generation lets a pass that is already running notice it has been
        # replaced and stop before it registers refresh loops or clears the
        # readiness flag its replacement owns.
        self._seed_task: asyncio.Task[None] | None = None
        self._seed_generation = 0

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
            self._seed_task = asyncio.create_task(
                self._seed_tokens(), name="deferred-token-seed"
            )
            self._tasks.append(self._seed_task)

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

    # -- Config reload ------------------------------------------------------

    async def apply_config_reload(self, new_config: Any) -> None:
        """Re-attach to a reloaded config and re-seed ``token_command`` providers.

        A hot config reload rebuilds :class:`GatewayConfig` and every
        :class:`ProviderInfo`, so each ``token_command`` provider comes back
        holding the pending sentinel with no refresh loop.  Seeding otherwise
        runs once, at process start, so without this the provider never
        obtains a token and every request to it is rejected until the process
        restarts.

        Refresh tasks bound to the superseded ``ProviderInfo`` objects are
        cancelled first: they would otherwise keep refreshing objects that the
        new config no longer references.
        """
        old_tasks = list(getattr(self._app, "_token_refresh_tasks", []))
        for task in old_tasks:
            task.cancel()
        for task in old_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001 - refreshers must not break reload
                logger.warning("Token refresh task failed during reload: %s", exc)
        setattr(self._app, "_token_refresh_tasks", [])

        # A seeding pass started before this reload would otherwise finish
        # against the config it is about to be replaced by: it reads
        # ``self._config`` and ``self._provider_states`` after the swap and
        # would register a second set of refresh loops over the same
        # providers.  Supersede it — the bump makes a pass that is already
        # running bail out at its next checkpoint — and wait for the old one
        # to stop before scheduling its replacement.
        self._seed_generation += 1
        stale_seed = self._seed_task
        self._seed_task = None
        if stale_seed is not None and not stale_seed.done():
            stale_seed.cancel()
            try:
                await stale_seed
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001 - must not break the reload
                logger.warning("Token seeding failed during reload: %s", exc)

        self._config = new_config
        # ``resolve()`` reads provider states off the config it routes through,
        # so the replacement config needs the same back-reference.
        setattr(new_config, "_deferred_startup", self)

        self._provider_states = {
            name: ProviderInitState.PENDING
            for name, pinfo in new_config.providers.items()
            if pinfo.token_command is not None
        }
        self._pending_tasks.discard("token_seed")
        if not self._provider_states:
            return

        # Deliberately not awaited: ``_seed_tokens`` sleeps up to
        # ``_RETRY_DELAY`` when a provider fails, and the reload handler must
        # not block on that.
        self._pending_tasks.add("token_seed")
        self._seed_task = asyncio.create_task(
            self._seed_tokens(), name="deferred-token-seed"
        )
        self._tasks.append(self._seed_task)

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
        """Seed all token_command providers in parallel, then start refresh loops.

        Failed providers get a single delayed retry before being marked
        permanently FAILED — this handles transient failures (network
        blips, slow auth services) without blocking startup.

        The generation captured here is re-checked at each checkpoint: a
        reload that supersedes this pass bumps it, and the stale pass then
        returns without registering refresh loops or clearing the readiness
        flag that now belongs to its replacement.
        """
        from .transport.token_refresh import start_token_refreshers

        generation = self._seed_generation
        pending = [
            name
            for name, state in self._provider_states.items()
            if state == ProviderInitState.PENDING
        ]

        # Parallel initial fetch
        await asyncio.gather(
            *(self._seed_one_provider(name) for name in pending),
        )
        if generation != self._seed_generation:
            return

        # Retry failed providers once after a delay
        failed = [
            name
            for name in pending
            if self._provider_states.get(name) == ProviderInitState.FAILED
        ]
        if failed:
            logger.info(
                "Retrying %d failed provider(s) in %ds: %s",
                len(failed),
                self._RETRY_DELAY,
                ", ".join(failed),
            )
            await asyncio.sleep(self._RETRY_DELAY)
            if generation != self._seed_generation:
                return
            for name in failed:
                self._provider_states[name] = ProviderInitState.PENDING
            await asyncio.gather(
                *(self._seed_one_provider(name) for name in failed),
            )
            if generation != self._seed_generation:
                return

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
            if generation != self._seed_generation:
                # Superseded while registering: stop the loops just started —
                # the superseding pass owns its own set.
                for task in refresh_tasks:
                    task.cancel()
                return
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
        """Rebuild metrics counters from the async persistence layer."""
        from llm_rosetta.observability import MetricsCollector

        metrics = getattr(self._app, "metrics", None)
        persistence = getattr(self._app, "persistence", None)
        if metrics is None or persistence is None:
            self._pending_tasks.discard("counter_rebuild")
            return

        try:
            # Snapshot current state *before* the rebuild starts
            pre_snapshot = metrics.export_counters()

            logger.info("Starting background counter rebuild")
            tmp = MetricsCollector()
            rows = [row async for row in persistence.iter_log_rows_for_rebuild()]
            count = tmp.rebuild_counters(iter(rows))
            logger.info("Background counter rebuild processed %d rows", count)
            baseline = tmp.export_counters()

            # Merge back
            metrics.merge_rebuild(baseline, pre_snapshot)
            await persistence.save_metrics(metrics.export_counters())
            logger.info("Counter rebuild complete, metrics merged")
        except Exception:
            logger.exception("Background counter rebuild failed")
        finally:
            self._pending_tasks.discard("counter_rebuild")

    # -- Backfills (thread executor) ----------------------------------------

    async def _run_backfills(self) -> None:
        """Run all database backfills sequentially via the async persistence layer."""
        persistence = getattr(self._app, "persistence", None)
        if persistence is None:
            self._pending_tasks.discard("backfills")
            return

        config = self._config
        keystore = getattr(self._app, "keystore", None)

        try:
            results: dict[str, int] = {}
            logger.info("Starting deferred database backfills")

            # 1. Backfill provider names
            model_to_provider = {
                model: route.provider_names[0] for model, route in config.models.items()
            }
            results["provider_names"] = (
                await persistence.backfill_provider_names(model_to_provider) or 0
            )

            # 2. Backfill API key last_used
            if keystore is not None:
                results["key_last_used"] = (
                    await keystore.backfill_last_used(persistence.db_path) or 0
                )

            # 3. Backfill total_tokens (cache tokens were excluded)
            results["total_tokens"] = await persistence.backfill_total_tokens() or 0

            # 4. Backfill error dump log IDs
            aliases = config.model_upstream_names if config else {}
            results["error_dump_log_ids"] = (
                await persistence.backfill_error_dump_log_ids(model_aliases=aliases)
                or 0
            )

            logger.info("Deferred backfills complete: %s", results)
        except Exception:
            logger.exception("Deferred backfills failed")
        finally:
            self._pending_tasks.discard("backfills")
