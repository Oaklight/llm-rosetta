"""The 0.10.0-era ``gateway.admin`` re-export paths are gone."""

import importlib
import sys

import pytest

import llm_rosetta_gateway.observability as obs
from llm_rosetta_gateway.observability.profiling import ProfilerState

_REMOVED_ADMIN_PATHS = [
    "llm_rosetta_gateway.admin.metrics",
    "llm_rosetta_gateway.admin.persistence",
    "llm_rosetta_gateway.admin.request_log",
]


@pytest.mark.parametrize("module_path", _REMOVED_ADMIN_PATHS)
def test_admin_reexport_paths_removed(module_path):
    """The 0.10.0-era re-export paths are gone; use ``observability.*``."""
    sys.modules.pop(module_path, None)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_path)


def test_ops_log_path_removed() -> None:
    """``ops_log`` was only ever a dev convenience, not a released path."""
    sys.modules.pop("llm_rosetta_gateway.admin.ops_log", None)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("llm_rosetta_gateway.admin.ops_log")


def test_profiler_state_identity() -> None:
    assert ProfilerState is obs.ProfilerState
