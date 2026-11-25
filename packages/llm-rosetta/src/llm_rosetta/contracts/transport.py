"""The transport contract: how a host reaches one upstream lane.

A host such as the gateway programs against :class:`UpstreamTransport`; a
lane implementation may live in the host itself (HTTP, ACP) or in an
installed package (the official Claude Code CLI).  This module holds only
stdlib types, so an extension can import it without importing the host.

The provider surface is a Protocol, not a class: the host passes its own
runtime provider object, and the transport reads the fields it needs.  A
lane's own options (a CLI command, its environment) are opaque to the
contract: the host hands them over as ``transport_options`` and the transport
validates them.

:class:`LaneProbe` and :meth:`UpstreamStream.lane_profile` are optional
capabilities a host may use when present.  A transport that does not answer a
probe or report accounting simply omits them.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True)
class UpstreamResponse:
    """Result from a non-streaming upstream call.

    Attributes:
        status_code: HTTP status code (or equivalent for other transports).
        body: Parsed JSON response body, or ``None`` on error.
        raw_content: Raw response bytes for error passthrough.
        meta: Facts the transport alone knows, for the request-log profile.
            A lane that reports its own accounting (native usage, cost and
            context window) puts it here; an HTTP transport leaves it empty.
    """

    status_code: int
    body: dict[str, Any] | None
    raw_content: bytes
    meta: dict[str, Any] | None = None

    @property
    def is_error(self) -> bool:
        """True if the upstream returned an error status."""
        return self.status_code >= 400

    @property
    def error_text(self) -> str:
        """Decode *raw_content* as UTF-8 for logging/passthrough."""
        return self.raw_content.decode("utf-8", errors="replace")

    @property
    def error_json(self) -> str:
        """Best-effort JSON string of the error body for SSE passthrough."""
        try:
            return json.dumps(json.loads(self.error_text))
        except (json.JSONDecodeError, ValueError):
            return self.error_text


class UpstreamStream(ABC):
    """Async context manager + iterator for a streaming upstream response.

    Usage::

        stream = await transport.send_streaming(...)
        async with stream:
            if stream.is_error:
                error_text = await stream.read_error()
                ...
                return
            async for chunk in stream:
                process(chunk)   # chunk is a parsed dict

    Subclasses must implement :meth:`read_error`, :meth:`__aiter__`,
    and :meth:`close`.  A lane that reports its own accounting also defines
    ``lane_profile() -> dict[str, Any]``.
    """

    status_code: int

    @property
    def is_error(self) -> bool:
        """True if the upstream returned an error status."""
        return self.status_code >= 400

    @abstractmethod
    async def read_error(self) -> str:
        """Read the error body as a string.

        Only valid when :attr:`is_error` is ``True``.
        """

    @abstractmethod
    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        """Yield parsed response chunks (e.g. JSON objects from SSE data)."""

    async def __aenter__(self) -> UpstreamStream:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    @abstractmethod
    async def close(self) -> None:
        """Release underlying transport resources."""


class UpstreamConnectionError(Exception):
    """Raised when the transport cannot reach the upstream provider."""


class UpstreamTimeoutError(UpstreamConnectionError):
    """Raised when the upstream provider does not respond in time."""


class TransportProvider(Protocol):
    """The provider view a transport may read.

    The host's runtime provider satisfies this structurally; a transport
    imports the Protocol for typing only.
    """

    config_name: str
    """The configured provider entry, distinct from the base format name."""

    base_url: str
    """The upstream base URL.  A process lane carries a placeholder."""

    timeout: float | None
    """The host's upstream timeout, or ``None`` for the transport's own."""

    keyless: bool
    """Whether the lane sends no credential of its own."""

    transport: str
    """The lane id that selected this transport."""

    transport_options: Mapping[str, Any]
    """The lane's options after the provider entry is overlaid on the shim's.

    Opaque to the host; the transport reads and validates it.  Empty when the
    lane has none.
    """

    def auth_headers(self) -> dict[str, str]:
        """Build the auth headers for one upstream request."""


class UpstreamTransport(Protocol):
    """Interface for sending requests to an upstream LLM provider.

    The proxy pipeline programs against this protocol.  The host ships HTTP
    and ACP; an installed package may supply another lane.
    """

    async def send(
        self,
        provider_info: TransportProvider,
        url: str,
        body: dict[str, Any],
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> UpstreamResponse:
        """Send a non-streaming request and return the full response.

        The caller is responsible for constructing the upstream URL and
        injecting any provider-specific body modifications (e.g. stream
        flags) before calling this method.
        """
        ...

    async def send_streaming(
        self,
        provider_info: TransportProvider,
        url: str,
        body: dict[str, Any],
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> UpstreamStream:
        """Send a streaming request and return an async chunk iterator."""
        ...

    async def close(self) -> None:
        """Release all transport resources (connection pools, etc.)."""
        ...


class LaneProbe(Protocol):
    """Optional capability: answer a connectivity test without an HTTP dial.

    The admin panel calls this when the provider's lane is not HTTP.  A
    transport that does not implement it is reported as untestable.
    """

    async def probe_provider(self, provider_info: TransportProvider) -> dict[str, Any]:
        """Run a short turn or equivalent check and report the outcome."""
        ...


class LaneReadingSink(Protocol):
    """Optional capability: persist and read a lane's allowance.

    A process lane can read the upstream's rate-limit headers from its own
    turn.  The host owns the lane's profile and its in-memory reading; a
    transport that has one calls these methods on the provider object when
    the host exposes them.  A host without them simply drops the reading.
    """

    def record_lane_reading(self, update: dict[str, Any]) -> bool:
        """Store one lane reading; return whether it carried an allowance."""
        ...

    def lane_reading(self) -> Any | None:
        """Return the lane's last stored reading, if there is one."""
        ...


__all__ = [
    "LaneProbe",
    "LaneReadingSink",
    "TransportProvider",
    "UpstreamConnectionError",
    "UpstreamResponse",
    "UpstreamStream",
    "UpstreamTimeoutError",
    "UpstreamTransport",
]
