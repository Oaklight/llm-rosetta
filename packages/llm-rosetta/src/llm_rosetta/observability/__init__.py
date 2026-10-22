"""Usage metering for the translation core.

What the tokens add up to, and for whom, is a property of a conversion:
:mod:`llm_rosetta.observability.usage` holds the semantics, and the pipeline
reads them when it normalizes a response's usage block.

The rest of the old ``observability`` package — metrics, the request log,
SQLite persistence, the ops log, error dumps, capture, maintenance, profiling,
retention and header sanitizing — moved to the gateway distribution as
:mod:`llm_rosetta_gateway.observability` (epic #1127, issue #1161).  Every
consumer of those modules is a proxy server; the core neither writes a
database nor counts requests.

The old paths keep working until the 0.16.0 removal window (#919): importing
``llm_rosetta.observability.metrics`` warns and resolves to the *same* module
object as its gateway counterpart, so a store is never opened twice.

Typical usage::

    from llm_rosetta.observability import UsageLedger, token_total

    ledger = UsageLedger()
    ledger.record(response["usage"], model="gpt-5-nano")
    ledger.total("24h")
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
import warnings
from importlib import import_module
from typing import Any


_ALIAS = "llm_rosetta.observability"
_TARGET = "llm_rosetta_gateway.observability"

#: Submodules that moved with the telemetry stack.  The alias finder below
#: matches exactly these names, so ``llm_rosetta.observability.usage`` is still
#: served from this package.
_MOVED_SUBMODULES = frozenset(
    {
        "capture",
        "error_dump",
        "maintenance",
        "metrics",
        "ops_log",
        "persistence",
        "profiling",
        "request_log",
        "retention",
        "sanitize",
    }
)


class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve the moved submodule paths to the gateway's module objects."""

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: Any = None,
    ) -> importlib.machinery.ModuleSpec | None:
        prefix = _ALIAS + "."
        if fullname.startswith(prefix) and fullname[len(prefix) :] in _MOVED_SUBMODULES:
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> Any:
        try:
            return importlib.import_module(_TARGET + spec.name[len(_ALIAS) :])
        except ModuleNotFoundError as exc:
            raise ImportError(
                f"{spec.name} moved to the gateway distribution; "
                "install it with: pip install llm-rosetta-gateway"
            ) from exc

    def exec_module(self, module: Any) -> None:
        """The aliased module is already executed."""
        # ``module.__name__`` is the real name, because ``create_module``
        # returned the real object; recover the spelling the caller used.
        source = module.__name__.replace(_TARGET, _ALIAS, 1)
        warnings.warn(
            f"importing {source} is deprecated; use {module.__name__} "
            f"({_ALIAS}.* will be removed in 0.16.0)",
            DeprecationWarning,
            stacklevel=2,
        )


if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

#: Public name -> module that defines it, for the part that stayed here.
_LAZY: dict[str, str] = {
    "LEDGER_COLUMNS": ".usage",
    "USAGE_WINDOW_LABELS": ".usage",
    "USAGE_WINDOWS": ".usage",
    "UsageLedger": ".usage",
    "UsageReport": ".usage",
    "cache_is_additive": ".usage",
    "token_total": ".usage",
    "usage_columns": ".usage",
}


#: Public name -> gateway submodule, for the telemetry that moved.
_MOVED: dict[str, str] = {
    "ALL_EVENT_TYPES": "ops_log",
    "ALL_SEVERITIES": "ops_log",
    "ALL_SOURCES": "ops_log",
    "CaptureState": "capture",
    "CapturedRequest": "capture",
    "DEFAULT_MAX_AGE_DAYS": "persistence",
    "DEFAULT_OPS_INFO_MAX": "persistence",
    "DEFAULT_OPS_WARN_MAX": "persistence",
    "DEFAULT_SUCCESS_MAX": "persistence",
    "EVENT_ADMIN_SETUP": "ops_log",
    "EVENT_CONFIG_RELOAD": "ops_log",
    "EVENT_DATA_CLEANUP": "ops_log",
    "EVENT_DATA_CLEARED": "ops_log",
    "EVENT_DATA_REBUILT": "ops_log",
    "EVENT_DATA_TRIMMED": "ops_log",
    "EVENT_DATA_VACUUMED": "ops_log",
    "EVENT_HEALTH_CHANGE": "ops_log",
    "EVENT_KEY_CREATE": "ops_log",
    "EVENT_KEY_DELETE": "ops_log",
    "EVENT_KEY_ROTATE": "ops_log",
    "EVENT_KEY_UPDATE": "ops_log",
    "EVENT_OPS_LOG_CLEARED": "ops_log",
    "EVENT_PASSWORD_CHANGED": "ops_log",
    "EVENT_SESSION_LOGOUT_ALL": "ops_log",
    "EVENT_SHUTDOWN": "ops_log",
    "EVENT_STARTUP": "ops_log",
    "EVENT_TOKEN_ROTATED": "ops_log",
    "MetricsCollector": "metrics",
    "OpsLog": "ops_log",
    "OpsLogEntry": "ops_log",
    "PersistenceManager": "persistence",
    "Policy": "maintenance",
    "ProfilerState": "profiling",
    "Registry": "maintenance",
    "RequestLog": "request_log",
    "RequestLogEntry": "request_log",
    "RetentionPolicy": "retention",
    "RetentionTracker": "retention",
    "SEVERITY_ERROR": "ops_log",
    "SEVERITY_INFO": "ops_log",
    "SEVERITY_WARNING": "ops_log",
    "SOURCE_ADMIN": "ops_log",
    "SOURCE_AUTH": "ops_log",
    "SOURCE_CONFIG": "ops_log",
    "SOURCE_GATEWAY": "ops_log",
    "SOURCE_KEYS": "ops_log",
    "SOURCE_PERSISTENCE": "ops_log",
    "Step": "maintenance",
    "StepContext": "maintenance",
    "StepResult": "maintenance",
    "compress_body": "error_dump",
    "compute_body_hash": "error_dump",
    "decompress_body": "error_dump",
    "dump_error": "error_dump",
    "offload_images": "error_dump",
    "sanitize_headers": "sanitize",
    "sanitize_upstream_error": "sanitize",
    "scrub_credential_patterns": "sanitize",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is not None:
        value = getattr(import_module(module_name, __name__), name)
        globals()[name] = value
        return value

    moved = _MOVED.get(name)
    if moved is not None:
        # Same object as the gateway's; the import warns and names the new home.
        module = importlib.import_module(f"{_ALIAS}.{moved}")
        value = getattr(module, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY) | set(_MOVED))


#: Everything the facade has ever advertised.  The moved names resolve through
#: the alias above (warning, same object); keeping them listed means the
#: deprecation window does not change the surface a caller can introspect.
__all__ = [
    "ALL_EVENT_TYPES",
    "ALL_SEVERITIES",
    "ALL_SOURCES",
    "CaptureState",
    "CapturedRequest",
    "DEFAULT_MAX_AGE_DAYS",
    "DEFAULT_OPS_INFO_MAX",
    "DEFAULT_OPS_WARN_MAX",
    "DEFAULT_SUCCESS_MAX",
    "EVENT_ADMIN_SETUP",
    "EVENT_CONFIG_RELOAD",
    "EVENT_DATA_CLEANUP",
    "EVENT_DATA_CLEARED",
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
    "LEDGER_COLUMNS",
    "MetricsCollector",
    "OpsLog",
    "OpsLogEntry",
    "PersistenceManager",
    "Policy",
    "ProfilerState",
    "Registry",
    "RequestLog",
    "RequestLogEntry",
    "RetentionPolicy",
    "RetentionTracker",
    "SEVERITY_ERROR",
    "SEVERITY_INFO",
    "SEVERITY_WARNING",
    "SOURCE_ADMIN",
    "SOURCE_AUTH",
    "SOURCE_CONFIG",
    "SOURCE_GATEWAY",
    "SOURCE_KEYS",
    "SOURCE_PERSISTENCE",
    "Step",
    "StepContext",
    "StepResult",
    "USAGE_WINDOWS",
    "USAGE_WINDOW_LABELS",
    "UsageLedger",
    "UsageReport",
    "cache_is_additive",
    "compress_body",
    "compute_body_hash",
    "decompress_body",
    "dump_error",
    "offload_images",
    "sanitize_headers",
    "sanitize_upstream_error",
    "scrub_credential_patterns",
    "token_total",
    "usage_columns",
]
