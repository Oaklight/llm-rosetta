"""Transport layer abstractions, re-exported from the extension contract.

The types and the protocol live in :mod:`llm_rosetta.contracts.transport`, so
a transport package can implement them without importing the gateway.  This
module stays as the gateway's internal import point.
"""

from __future__ import annotations

from llm_rosetta.contracts.transport import (
    LaneProbe,
    LaneReadingSink,
    TransportProvider,
    UpstreamConnectionError,
    UpstreamResponse,
    UpstreamStream,
    UpstreamTimeoutError,
    UpstreamTransport,
)

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
