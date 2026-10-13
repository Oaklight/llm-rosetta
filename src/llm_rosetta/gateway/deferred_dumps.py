"""Defer error-dump writes until the request-log row they name exists.

An error dump names the request's log entry in ``error_dumps.request_log_id``,
but that row is written when the request finishes — after the error handling
that captures the dump has run.  With the foreign key on
``error_dumps.request_log_id`` enforced (``PRAGMA foreign_keys=ON``, added in
#848), writing the dump on the spot fails the constraint; ``dump_error``
swallows the exception, so the dump is silently lost.  A database created
before that constraint existed has no FK and keeps its dumps, which is why the
loss only shows up on newer deployments.

:class:`DeferredDumpStore` wraps the persistence manager handed to the proxy
handlers and queues the two writes ``dump_error`` makes.  The gateway replays
them once the request-log row has landed.  Writes issued after that — the
mid-stream dumps, taken while the streaming response drains, long after the log
row was written — pass straight through.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("llm-rosetta.gateway")

_Pending = list[tuple[str, tuple[Any, ...], dict[str, Any]]]


class DeferredDumpStore:
    """Queue dump writes, then replay them once the log row exists.

    Forwards only the two methods ``dump_error`` uses on a persistence manager
    (``insert_dump_body`` and ``insert_error_dump``) — that is the whole surface
    the gateway's error paths touch, so nothing else needs to be proxied.

    Replay is fire-and-forget, like ``dump_error`` itself: a failing write is
    logged and the rest still run, so a bad dump can never break the response
    whose handling triggered it.
    """

    def __init__(self, store: Any) -> None:
        self._store = store
        self._pending: _Pending = []
        self._flushed = False

    async def insert_dump_body(self, *args: Any, **kwargs: Any) -> None:
        """Queue (or forward, after :meth:`flush`) a body insert."""
        if self._flushed:
            await self._store.insert_dump_body(*args, **kwargs)
            return
        self._pending.append(("insert_dump_body", args, kwargs))

    async def insert_error_dump(self, *args: Any, **kwargs: Any) -> None:
        """Queue (or forward, after :meth:`flush`) a dump record insert."""
        if self._flushed:
            await self._store.insert_error_dump(*args, **kwargs)
            return
        self._pending.append(("insert_error_dump", args, kwargs))

    @property
    def pending(self) -> int:
        """Number of writes still queued."""
        return len(self._pending)

    async def flush(self) -> None:
        """Replay queued writes in order, then stop deferring.

        Every write is attempted even if one fails; a failure is logged at
        ``WARNING`` with its traceback, since a swallowed one is what made the
        original loss invisible.  Never raises.
        """
        self._flushed = True
        pending, self._pending = self._pending, []
        for name, args, kwargs in pending:
            try:
                await getattr(self._store, name)(*args, **kwargs)
            except Exception:
                logger.warning(
                    "Deferred dump write %s failed; the dump is lost",
                    name,
                    exc_info=True,
                )
