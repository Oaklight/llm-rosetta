"""Reusable observability layer for LLM proxy servers.

This package provides framework-agnostic building blocks for metrics
collection, request logging, SQLite persistence, and on-demand
profiling.  Any HTTP proxy built on top of ``llm-rosetta`` can import
from here — no dependency on the gateway's config system or HTTP
server.

Typical usage::

    from llm_rosetta.observability import (
        MetricsCollector,
        PersistenceManager,
        ProfilerState,
        RequestLog,
        RequestLogEntry,
    )

    metrics = MetricsCollector()
    persistence = PersistenceManager("/var/data/myproxy")
    request_log = RequestLog(persistence=persistence)
    profiler = ProfilerState()
"""

from __future__ import annotations

from .error_dump import (
    compress_body,
    compute_body_hash,
    decompress_body,
    dump_error,
    offload_images,
)
from .metrics import MetricsCollector
from .persistence import DEFAULT_ERROR_MAX, DEFAULT_SUCCESS_MAX, PersistenceManager
from .profiling import ProfilerState
from .request_log import RequestLog, RequestLogEntry

__all__ = [
    "DEFAULT_ERROR_MAX",
    "DEFAULT_SUCCESS_MAX",
    "MetricsCollector",
    "PersistenceManager",
    "ProfilerState",
    "RequestLog",
    "RequestLogEntry",
    "compress_body",
    "compute_body_hash",
    "decompress_body",
    "dump_error",
    "offload_images",
]
