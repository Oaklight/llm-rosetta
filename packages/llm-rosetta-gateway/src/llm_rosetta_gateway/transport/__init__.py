"""Transport layer — protocol-agnostic interface and implementations.

This package provides a clean boundary between the proxy pipeline and
the underlying communication protocol.  The proxy programs against the
:class:`UpstreamTransport` protocol; concrete implementations live in
sub-packages:

* :mod:`transport.http` — HTTP REST + SSE streaming (current).
* :mod:`transport.acp` — a local agent CLI over the Agent Client Protocol.
* :mod:`transport.claude_cli` — the Claude Code CLI as a model client.
* ``transport.grpc`` — gRPC (future).
* ``transport.websocket`` — WebSocket (future).
"""

from ._base import (
    UpstreamConnectionError,
    UpstreamTimeoutError,
    UpstreamResponse,
    UpstreamStream,
    UpstreamTransport,
)
from .acp import AcpTransport, AcpTurnError
from .claude_cli import ClaudeCliTransport
from .http import HttpTransport
from .provider_info import (
    TOKEN_PENDING_SENTINEL,
    AuthHeaderFn,
    KeyRing,
    ProviderInfo,
    anthropic_auth,
    google_auth,
    openai_auth,
)
from .sse_format import SSE_FORMATTERS, format_sse_done

#: Transport names a provider may declare in ``connection.transport`` (or its
#: own ``transport`` key).  The gateway builds one instance per name; a name
#: outside this tuple fails at config load instead of on the first request.
TRANSPORT_NAMES: tuple[str, ...] = ("http", "acp", "claude_cli")

__all__ = [
    # Transport names a provider may declare
    "TRANSPORT_NAMES",
    # Protocol + response types
    "UpstreamConnectionError",
    "UpstreamTimeoutError",
    "UpstreamResponse",
    "UpstreamStream",
    "UpstreamTransport",
    # HTTP implementation
    "HttpTransport",
    # Local agent CLI over ACP
    "AcpTransport",
    "AcpTurnError",
    # The Claude Code CLI as a model client
    "ClaudeCliTransport",
    # Provider config
    "TOKEN_PENDING_SENTINEL",
    "AuthHeaderFn",
    "KeyRing",
    "ProviderInfo",
    "openai_auth",
    "anthropic_auth",
    "google_auth",
    # Downstream SSE formatting
    "SSE_FORMATTERS",
    "format_sse_done",
]
