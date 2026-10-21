"""The 0.10.0-era ``gateway.admin`` re-export paths warn, then go in 0.16.0."""

import importlib
import sys

import pytest

import llm_rosetta.observability as obs
from llm_rosetta_gateway.admin.routes.profiling import ProfilerState

_ADMIN_REEXPORTS = [
    (
        "llm_rosetta_gateway.admin.metrics",
        "llm_rosetta.observability.metrics",
        ["MetricsCollector", "_RollingWindow"],
    ),
    (
        "llm_rosetta_gateway.admin.persistence",
        "llm_rosetta.observability.persistence",
        ["PersistenceManager", "DEFAULT_SUCCESS_MAX"],
    ),
    (
        "llm_rosetta_gateway.admin.request_log",
        "llm_rosetta.observability.request_log",
        ["RequestLog", "RequestLogEntry"],
    ),
]


def _import_fresh(module_path: str):
    """Import *module_path* from scratch so its import-time warning fires."""
    sys.modules.pop(module_path, None)
    return importlib.import_module(module_path)


@pytest.mark.parametrize(("module_path", "canonical_path", "names"), _ADMIN_REEXPORTS)
def test_admin_reexport_paths_warn(module_path, canonical_path, names):
    with pytest.warns(DeprecationWarning, match=f"{module_path} is deprecated"):
        module = _import_fresh(module_path)
    canonical = importlib.import_module(canonical_path)
    for name in names:
        assert getattr(module, name) is getattr(canonical, name)


def test_ops_log_path_removed() -> None:
    """``ops_log`` was only ever a dev convenience, not a released path."""
    sys.modules.pop("llm_rosetta_gateway.admin.ops_log", None)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("llm_rosetta_gateway.admin.ops_log")


def test_profiler_state_identity() -> None:
    assert ProfilerState is obs.ProfilerState
