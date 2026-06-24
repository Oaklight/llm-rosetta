"""Transport layer abstractions — protocol-agnostic interface.

Defines the :class:`UpstreamTransport` protocol and response types that
the proxy pipeline programs against.  Concrete implementations (HTTP/SSE,
gRPC, WebSocket) live in sub-packages (e.g. ``transport.http``).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol

from llm_rosetta.auto_detect import ProviderType

from .provider_info import ProviderInfo


# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class UpstreamResponse:
    """Result from a non-streaming upstream call.

    Attributes:
        status_code: HTTP status code (or equivalent for other transports).
        body: Parsed JSON response body, or ``None`` on error.
        raw_content: Raw response bytes for error passthrough.
    """

    status_code: int
    body: dict[str, Any] | None
    raw_content: bytes

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
    and :meth:`close`.
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


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class UpstreamConnectionError(Exception):
    """Raised when the transport cannot reach the upstream provider."""


# ---------------------------------------------------------------------------
# Transport protocol
# ---------------------------------------------------------------------------


class UpstreamTransport(Protocol):
    """Interface for sending requests to an upstream LLM provider.

    The proxy pipeline programs against this protocol.  Concrete
    implementations live in sub-packages (``transport.http``, etc.).
    """

    async def send_request(
        self,
        provider_info: ProviderInfo,
        target_provider: ProviderType,
        body: dict[str, Any],
        model: str,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> UpstreamResponse:
        """Send a non-streaming request and return the full response."""
        ...

    async def send_streaming(
        self,
        provider_info: ProviderInfo,
        target_provider: ProviderType,
        body: dict[str, Any],
        model: str,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> UpstreamStream:
        """Send a streaming request and return an async chunk iterator."""
        ...

    async def close(self) -> None:
        """Release all transport resources (connection pools, etc.)."""
        ...
