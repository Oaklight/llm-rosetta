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
from .capture import CapturedRequest, CaptureState
from .metrics import MetricsCollector
from .ops_log import (
    ALL_EVENT_TYPES,
    ALL_SEVERITIES,
    ALL_SOURCES,
    EVENT_ADMIN_SETUP,
    EVENT_CONFIG_RELOAD,
    EVENT_HEALTH_CHANGE,
    EVENT_KEY_CREATE,
    EVENT_KEY_DELETE,
    EVENT_KEY_ROTATE,
    EVENT_KEY_UPDATE,
    EVENT_OPS_LOG_CLEARED,
    EVENT_SHUTDOWN,
    EVENT_STARTUP,
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    SOURCE_ADMIN,
    SOURCE_CONFIG,
    SOURCE_GATEWAY,
    SOURCE_KEYS,
    SOURCE_PERSISTENCE,
    OpsLog,
    OpsLogEntry,
)
from .persistence import (
    DEFAULT_ERROR_MAX,
    DEFAULT_MAX_AGE_DAYS,
    DEFAULT_OPS_INFO_MAX,
    DEFAULT_OPS_WARN_MAX,
    DEFAULT_SUCCESS_MAX,
    PersistenceManager,
)
from .profiling import ProfilerState
from .request_log import RequestLog, RequestLogEntry

__all__ = [
    "ALL_EVENT_TYPES",
    "ALL_SEVERITIES",
    "ALL_SOURCES",
    "CapturedRequest",
    "CaptureState",
    "DEFAULT_ERROR_MAX",
    "DEFAULT_MAX_AGE_DAYS",
    "DEFAULT_OPS_INFO_MAX",
    "DEFAULT_OPS_WARN_MAX",
    "DEFAULT_SUCCESS_MAX",
    "EVENT_ADMIN_SETUP",
    "EVENT_CONFIG_RELOAD",
    "EVENT_HEALTH_CHANGE",
    "EVENT_KEY_CREATE",
    "EVENT_KEY_DELETE",
    "EVENT_KEY_ROTATE",
    "EVENT_KEY_UPDATE",
    "EVENT_OPS_LOG_CLEARED",
    "EVENT_SHUTDOWN",
    "EVENT_STARTUP",
    "SEVERITY_ERROR",
    "SEVERITY_INFO",
    "SEVERITY_WARNING",
    "SOURCE_ADMIN",
    "SOURCE_CONFIG",
    "SOURCE_GATEWAY",
    "SOURCE_KEYS",
    "SOURCE_PERSISTENCE",
    "MetricsCollector",
    "OpsLog",
    "OpsLogEntry",
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
