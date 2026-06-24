"""Transport layer — SSE, HTTP client, and provider connection primitives.

This package isolates all protocol-specific code (HTTP, SSE) from the
higher-level proxy pipeline.  Future protocol backends (gRPC, WebSocket)
can be added as sibling modules.
"""

from .client import close_clients, get_client, prepare_upstream
from .provider_info import AuthHeaderFn, KeyRing, ProviderInfo
from .sse import (
    SENTINEL_DONE,
    SSE_FORMATTERS,
    is_openai_done,
    parse_sse_data,
    parse_sse_line,
)

__all__ = [
    # client
    "close_clients",
    "get_client",
    "prepare_upstream",
    # provider_info
    "AuthHeaderFn",
    "KeyRing",
    "ProviderInfo",
    # sse
    "SENTINEL_DONE",
    "SSE_FORMATTERS",
    "is_openai_done",
    "parse_sse_data",
    "parse_sse_line",
]
