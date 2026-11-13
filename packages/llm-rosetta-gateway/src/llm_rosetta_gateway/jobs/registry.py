"""Kind name to handler: the one place that knows what a job actually does.

A handler receives a :class:`~llm_rosetta_gateway.jobs.worker.JobContext` and
returns a JSON-serialisable summary (or ``None``).  A kind with no handler
fails cleanly instead of wedging the queue: the worker records the unknown kind
as the job's error and moves on.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .worker import JobContext

Handler = Callable[["JobContext"], Any]


class UnknownJobKind(Exception):
    """No handler is registered for a job's kind."""

    def __init__(self, kind: str) -> None:
        super().__init__(f"no handler registered for job kind {kind!r}")
        self.kind = kind


class JobRegistry:
    """The handlers this process can run."""

    def __init__(self, handlers: Mapping[str, Handler] | None = None) -> None:
        self._handlers: dict[str, Handler] = dict(handlers or {})

    def register(self, kind: str, handler: Handler) -> Handler:
        """Bind *kind* to *handler*, returning the handler for decorator use."""
        self._handlers[kind] = handler
        return handler

    def get(self, kind: str) -> Handler:
        """Return the handler for *kind*, or raise :class:`UnknownJobKind`."""
        try:
            return self._handlers[kind]
        except KeyError:
            raise UnknownJobKind(kind) from None

    def kinds(self) -> list[str]:
        """Every kind this registry knows, sorted."""
        return sorted(self._handlers)


def default_handlers() -> dict[str, Handler]:
    """The handlers this build ships.

    Empty until the data operations migrate onto the envelope; a queued job of
    any other kind fails with the unknown-kind error rather than sitting in the
    queue forever.
    """
    return {}
