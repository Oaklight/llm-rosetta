"""Compatibility re-exports — use :mod:`llm_rosetta_gateway.observability.sanitize`.

Credential scrubbing is a cross-cutting concern: error dumps, request logs and
the gateway's upstream error path all need it.  The implementation moved to the
framework-agnostic observability layer; this module keeps the old import path
resolving for the gateway's own callers and downstream code.
"""

from __future__ import annotations

from llm_rosetta_gateway.observability.sanitize import (
    sanitize_headers,
    sanitize_upstream_error,
    scrub_credential_patterns,
)

__all__ = [
    "sanitize_headers",
    "sanitize_upstream_error",
    "scrub_credential_patterns",
]
