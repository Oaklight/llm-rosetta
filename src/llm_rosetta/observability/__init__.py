"""Reusable observability layer for LLM proxy servers.

This package provides framework-agnostic building blocks for metrics
collection, request logging, SQLite persistence, usage metering, and
on-demand profiling.  Any HTTP proxy built on top of ``llm-rosetta`` can
import from here — no dependency on the gateway's config system or HTTP
server.

The package root resolves its names lazily (PEP 562): importing a
submodule, or one of the names below, initializes only the modules that
name belongs to.  A library user that touches
:mod:`llm_rosetta.observability.usage` therefore does not pull in the
SQLite store, the request log, metrics or the error-dump machinery.

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

Usage — what the tokens add up to, and for whom — is a subpackage of its
own, :mod:`llm_rosetta.observability.usage`, and needs none of the above::

    from llm_rosetta.observability import UsageLedger, token_total

    ledger = UsageLedger()
    ledger.record(response["usage"], model="gpt-5-nano")
    ledger.total("24h")
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .capture import CapturedRequest, CaptureState
    from .error_dump import (
        compress_body,
        compute_body_hash,
        decompress_body,
        dump_error,
        offload_images,
    )
    from .maintenance import Policy, Registry, Step, StepContext, StepResult
    from .metrics import MetricsCollector
    from .ops_log import (
        ALL_EVENT_TYPES,
        ALL_SEVERITIES,
        ALL_SOURCES,
        EVENT_ADMIN_SETUP,
        EVENT_CONFIG_RELOAD,
        EVENT_DATA_CLEARED,
        EVENT_DATA_CLEANUP,
        EVENT_DATA_REBUILT,
        EVENT_DATA_TRIMMED,
        EVENT_DATA_VACUUMED,
        EVENT_HEALTH_CHANGE,
        EVENT_KEY_CREATE,
        EVENT_KEY_DELETE,
        EVENT_KEY_ROTATE,
        EVENT_KEY_UPDATE,
        EVENT_OPS_LOG_CLEARED,
        EVENT_PASSWORD_CHANGED,
        EVENT_SESSION_LOGOUT_ALL,
        EVENT_SHUTDOWN,
        EVENT_STARTUP,
        EVENT_TOKEN_ROTATED,
        SEVERITY_ERROR,
        SEVERITY_INFO,
        SEVERITY_WARNING,
        SOURCE_ADMIN,
        SOURCE_AUTH,
        SOURCE_CONFIG,
        SOURCE_GATEWAY,
        SOURCE_KEYS,
        SOURCE_PERSISTENCE,
        OpsLog,
        OpsLogEntry,
    )
    from .persistence import (
        DEFAULT_MAX_AGE_DAYS,
        DEFAULT_OPS_INFO_MAX,
        DEFAULT_OPS_WARN_MAX,
        DEFAULT_SUCCESS_MAX,
        PersistenceManager,
    )
    from .profiling import ProfilerState
    from .request_log import RequestLog, RequestLogEntry
    from .retention import RetentionPolicy, RetentionTracker
    from .sanitize import (
        sanitize_headers,
        sanitize_upstream_error,
        scrub_credential_patterns,
    )
    from .usage import (
        LEDGER_COLUMNS,
        USAGE_WINDOW_LABELS,
        USAGE_WINDOWS,
        UsageLedger,
        UsageReport,
        cache_is_additive,
        token_total,
        usage_columns,
    )

#: Public name -> module that defines it.  Keeping this map explicit means a
#: typo at import time is an AttributeError rather than an empty module.
_LAZY: dict[str, str] = {
    # capture
    "CapturedRequest": ".capture",
    "CaptureState": ".capture",
    # error_dump
    "compress_body": ".error_dump",
    "compute_body_hash": ".error_dump",
    "decompress_body": ".error_dump",
    "dump_error": ".error_dump",
    "offload_images": ".error_dump",
    # maintenance
    "Policy": ".maintenance",
    "Registry": ".maintenance",
    "Step": ".maintenance",
    "StepContext": ".maintenance",
    "StepResult": ".maintenance",
    # metrics
    "MetricsCollector": ".metrics",
    # ops_log
    "ALL_EVENT_TYPES": ".ops_log",
    "ALL_SEVERITIES": ".ops_log",
    "ALL_SOURCES": ".ops_log",
    "EVENT_ADMIN_SETUP": ".ops_log",
    "EVENT_CONFIG_RELOAD": ".ops_log",
    "EVENT_DATA_CLEARED": ".ops_log",
    "EVENT_DATA_CLEANUP": ".ops_log",
    "EVENT_DATA_REBUILT": ".ops_log",
    "EVENT_DATA_TRIMMED": ".ops_log",
    "EVENT_DATA_VACUUMED": ".ops_log",
    "EVENT_HEALTH_CHANGE": ".ops_log",
    "EVENT_KEY_CREATE": ".ops_log",
    "EVENT_KEY_DELETE": ".ops_log",
    "EVENT_KEY_ROTATE": ".ops_log",
    "EVENT_KEY_UPDATE": ".ops_log",
    "EVENT_OPS_LOG_CLEARED": ".ops_log",
    "EVENT_PASSWORD_CHANGED": ".ops_log",
    "EVENT_SESSION_LOGOUT_ALL": ".ops_log",
    "EVENT_SHUTDOWN": ".ops_log",
    "EVENT_STARTUP": ".ops_log",
    "EVENT_TOKEN_ROTATED": ".ops_log",
    "OpsLog": ".ops_log",
    "OpsLogEntry": ".ops_log",
    "SEVERITY_ERROR": ".ops_log",
    "SEVERITY_INFO": ".ops_log",
    "SEVERITY_WARNING": ".ops_log",
    "SOURCE_ADMIN": ".ops_log",
    "SOURCE_AUTH": ".ops_log",
    "SOURCE_CONFIG": ".ops_log",
    "SOURCE_GATEWAY": ".ops_log",
    "SOURCE_KEYS": ".ops_log",
    "SOURCE_PERSISTENCE": ".ops_log",
    # persistence
    "DEFAULT_MAX_AGE_DAYS": ".persistence",
    "DEFAULT_OPS_INFO_MAX": ".persistence",
    "DEFAULT_OPS_WARN_MAX": ".persistence",
    "DEFAULT_SUCCESS_MAX": ".persistence",
    "PersistenceManager": ".persistence",
    # profiling
    "ProfilerState": ".profiling",
    # request_log
    "RequestLog": ".request_log",
    "RequestLogEntry": ".request_log",
    # retention
    "RetentionPolicy": ".retention",
    "RetentionTracker": ".retention",
    # sanitize
    "sanitize_headers": ".sanitize",
    "sanitize_upstream_error": ".sanitize",
    "scrub_credential_patterns": ".sanitize",
    # usage
    "LEDGER_COLUMNS": ".usage",
    "USAGE_WINDOW_LABELS": ".usage",
    "USAGE_WINDOWS": ".usage",
    "UsageLedger": ".usage",
    "UsageReport": ".usage",
    "cache_is_additive": ".usage",
    "token_total": ".usage",
    "usage_columns": ".usage",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "ALL_EVENT_TYPES",
    "ALL_SEVERITIES",
    "ALL_SOURCES",
    "CapturedRequest",
    "CaptureState",
    "DEFAULT_MAX_AGE_DAYS",
    "DEFAULT_OPS_INFO_MAX",
    "DEFAULT_OPS_WARN_MAX",
    "DEFAULT_SUCCESS_MAX",
    "EVENT_ADMIN_SETUP",
    "EVENT_CONFIG_RELOAD",
    "EVENT_DATA_CLEARED",
    "EVENT_DATA_CLEANUP",
    "EVENT_DATA_REBUILT",
    "EVENT_DATA_TRIMMED",
    "EVENT_DATA_VACUUMED",
    "EVENT_HEALTH_CHANGE",
    "EVENT_KEY_CREATE",
    "EVENT_KEY_DELETE",
    "EVENT_KEY_ROTATE",
    "EVENT_KEY_UPDATE",
    "EVENT_OPS_LOG_CLEARED",
    "EVENT_PASSWORD_CHANGED",
    "EVENT_SESSION_LOGOUT_ALL",
    "EVENT_SHUTDOWN",
    "EVENT_STARTUP",
    "EVENT_TOKEN_ROTATED",
    "SEVERITY_ERROR",
    "SEVERITY_INFO",
    "SEVERITY_WARNING",
    "SOURCE_ADMIN",
    "SOURCE_AUTH",
    "SOURCE_CONFIG",
    "SOURCE_GATEWAY",
    "SOURCE_KEYS",
    "SOURCE_PERSISTENCE",
    "MetricsCollector",
    "OpsLog",
    "OpsLogEntry",
    "PersistenceManager",
    "Policy",
    "ProfilerState",
    "LEDGER_COLUMNS",
    "Registry",
    "RequestLog",
    "RequestLogEntry",
    "RetentionPolicy",
    "RetentionTracker",
    "Step",
    "StepContext",
    "StepResult",
    "USAGE_WINDOW_LABELS",
    "USAGE_WINDOWS",
    "UsageLedger",
    "UsageReport",
    "cache_is_additive",
    "token_total",
    "usage_columns",
    "compress_body",
    "compute_body_hash",
    "decompress_body",
    "dump_error",
    "offload_images",
    "sanitize_headers",
    "sanitize_upstream_error",
    "scrub_credential_patterns",
]
