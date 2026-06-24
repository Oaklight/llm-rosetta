"""Transport layer — protocol-agnostic interface and implementations.

This package provides a clean boundary between the proxy pipeline and
the underlying communication protocol.  The proxy programs against the
:class:`UpstreamTransport` protocol; concrete implementations live in
sub-packages:

* :mod:`transport.http` — HTTP REST + SSE streaming (current).
* ``transport.grpc`` — gRPC (future).
* ``transport.websocket`` — WebSocket (future).
"""

from ._base import (
    UpstreamConnectionError,
    UpstreamResponse,
    UpstreamStream,
    UpstreamTransport,
)
from .http import HttpTransport
from .provider_info import AuthHeaderFn, KeyRing, ProviderInfo
from .sse_format import SSE_FORMATTERS, format_sse_done

__all__ = [
    # Protocol + response types
    "UpstreamConnectionError",
    "UpstreamResponse",
    "UpstreamStream",
    "UpstreamTransport",
    # HTTP implementation
    "HttpTransport",
    # Provider config
    "AuthHeaderFn",
    "KeyRing",
    "ProviderInfo",
    # Downstream SSE formatting
    "SSE_FORMATTERS",
    "format_sse_done",
]
