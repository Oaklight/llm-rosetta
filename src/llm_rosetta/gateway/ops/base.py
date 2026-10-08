"""Base classes for the unified gateway operations layer.

Every gateway action — proxy requests, admin cleanup, key management,
config changes — is an :class:`OpsBase` subclass.  Calling
:meth:`execute` runs the operation and automatically records it.

The default :meth:`_record` writes to ``ops_log``.
:class:`~llm_rosetta.gateway.ops.proxy.OpsProxyRequest` overrides this
to write to ``request_log`` + ``metrics`` instead (hot path).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from llm_rosetta.observability.ops_log import (
    SEVERITY_INFO,
    SOURCE_ADMIN,
    OpsLogEntry,
)


class OpsContext:
    """Lightweight bag of services, created once at app init.

    Attributes are optional — ``None`` means the subsystem is not
    configured (e.g. no persistence in in-memory mode).
    """

    __slots__ = ("ops_log", "request_log", "metrics", "persistence")

    def __init__(
        self,
        *,
        ops_log: Any = None,
        request_log: Any = None,
        metrics: Any = None,
        persistence: Any = None,
    ) -> None:
        self.ops_log = ops_log
        self.request_log = request_log
        self.metrics = metrics
        self.persistence = persistence


class OpsBase(ABC):
    """Abstract base for all gateway operations.

    Subclasses declare ``event_type``, ``severity``, and ``source`` as
    class variables, then implement :meth:`_run`, :meth:`_message`, and
    :meth:`_details`.  The base :meth:`_record` writes an
    :class:`OpsLogEntry` to ``ops_log`` automatically.

    For operations that should **not** write to ``ops_log`` (e.g. the
    hot-path proxy request), override :meth:`_record`.
    """

    __slots__ = ("_ctx",)

    event_type: ClassVar[str]
    severity: ClassVar[str] = SEVERITY_INFO
    source: ClassVar[str] = SOURCE_ADMIN

    def __init__(self, ctx: OpsContext) -> None:
        self._ctx = ctx

    async def execute(self) -> Any:
        """Run the operation and record it.  Returns the result of :meth:`_run`."""
        result = await self._run()
        await self._record(result)
        return result

    @abstractmethod
    async def _run(self) -> Any:
        """Execute the operation logic.  Return value is passed to :meth:`_record`."""

    @abstractmethod
    def _message(self, result: Any) -> str:
        """Human-readable summary for the ops_log entry."""

    @abstractmethod
    def _details(self, result: Any) -> dict[str, Any]:
        """Structured details dict for the ops_log entry."""

    async def _record(self, result: Any) -> None:
        """Write an audit record to ops_log.

        Override in subclasses that record elsewhere (e.g.
        ``OpsProxyRequest`` writes to ``request_log`` + ``metrics``).
        """
        if self._ctx.ops_log is None:
            return
        await self._ctx.ops_log.add(
            OpsLogEntry.create(
                event_type=self.event_type,
                severity=self.severity,
                message=self._message(result),
                details=self._details(result),
                source=self.source,
            )
        )
