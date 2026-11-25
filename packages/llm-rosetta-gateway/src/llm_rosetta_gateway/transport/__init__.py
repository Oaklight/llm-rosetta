"""Transport layer — protocol-agnostic interface and implementations.

This package provides a clean boundary between the proxy pipeline and
the underlying communication protocol.  The proxy programs against the
:class:`UpstreamTransport` protocol; the transports this distribution ships
live in sub-packages:

* :mod:`transport.http` — HTTP REST + SSE streaming (current).
* :mod:`transport.acp` — a local agent CLI over the Agent Client Protocol.

A lane this distribution does not ship comes from an installed package that
publishes an entry point in :data:`TRANSPORT_ENTRY_POINT_GROUP`; see
:mod:`transport.registry`.  The official Claude Code CLI is one: its shim
ships the transport in ``llm-rosetta-shim-contrib``.
"""

from ._base import (
    LaneProbe,
    LaneReadingSink,
    TransportProvider,
    UpstreamConnectionError,
    UpstreamTimeoutError,
    UpstreamResponse,
    UpstreamStream,
    UpstreamTransport,
)
from .acp import AcpTransport, AcpTurnError
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
from .registry import (
    BUILTIN_TRANSPORTS,
    TRANSPORT_ENTRY_POINT_GROUP,
    TransportPluginError,
    build_lane_transports,
    clear_transport_cache,
    installed_transport_names,
)
from .sse_format import SSE_FORMATTERS, format_sse_done

#: Transport names this distribution ships.  An installed package adds more
#: through :data:`TRANSPORT_ENTRY_POINT_GROUP`; use
#: :func:`installed_transport_names` to see every id a provider may name.
TRANSPORT_NAMES: tuple[str, ...] = tuple(BUILTIN_TRANSPORTS)

__all__ = [
    # Transport names
    "TRANSPORT_NAMES",
    "BUILTIN_TRANSPORTS",
    "TRANSPORT_ENTRY_POINT_GROUP",
    "TransportPluginError",
    "build_lane_transports",
    "clear_transport_cache",
    "installed_transport_names",
    # Protocol + response types
    "LaneProbe",
    "LaneReadingSink",
    "TransportProvider",
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
