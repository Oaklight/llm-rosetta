"""Transport layer — protocol-agnostic interface and implementations.

This package provides a clean boundary between the proxy pipeline and
the underlying communication protocol.  The proxy programs against the
:class:`UpstreamTransport` protocol; the transports this distribution ships
live in sub-packages:

* :mod:`transport.http` — HTTP REST + SSE streaming (current).
* :mod:`transport.acp` — a local agent CLI over the Agent Client Protocol.

A lane this distribution does not ship comes from an installed extension: its
manifest declares the transport and ``llm_rosetta_ext.<id>.transport`` builds
it, scoped to the extension that declares it; see :mod:`transport.registry`.
The official Claude Code CLI is one, shipped by the ``claude`` extension.
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
    TRANSPORTS_ATTRIBUTE,
    LaneRef,
    TransportPluginError,
    build_lane_transports,
    lane_key,
    reachable_transport_names,
    resolve_lane,
)
from .sse_format import SSE_FORMATTERS, format_sse_done

#: Transport names this distribution ships.  An installed extension adds more,
#: scoped to itself; use :func:`reachable_transport_names` to see the ids a
#: shim may name.
TRANSPORT_NAMES: tuple[str, ...] = tuple(BUILTIN_TRANSPORTS)

__all__ = [
    # Transport names
    "TRANSPORT_NAMES",
    "BUILTIN_TRANSPORTS",
    "TRANSPORTS_ATTRIBUTE",
    "LaneRef",
    "TransportPluginError",
    "build_lane_transports",
    "lane_key",
    "reachable_transport_names",
    "resolve_lane",
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
