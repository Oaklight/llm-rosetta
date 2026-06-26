"""Tests verifying backward-compatible imports from gateway.admin still work."""

from llm_rosetta.gateway.admin.metrics import MetricsCollector, _RollingWindow
from llm_rosetta.gateway.admin.persistence import (
    DEFAULT_ERROR_MAX,
    DEFAULT_SUCCESS_MAX,
    PersistenceManager,
)
from llm_rosetta.gateway.admin.request_log import RequestLog, RequestLogEntry
from llm_rosetta.gateway.admin.routes.profiling import ProfilerState

# Also verify they are the same objects (not copies)
import llm_rosetta.observability as obs


class TestBackwardCompatImports:
    """Verify that the old import paths still resolve to the canonical classes."""

    def test_metrics_collector_identity(self):
        assert MetricsCollector is obs.MetricsCollector

    def test_rolling_window_identity(self):
        from llm_rosetta.observability.metrics import _RollingWindow as _RW

        assert _RollingWindow is _RW

    def test_persistence_manager_identity(self):
        assert PersistenceManager is obs.PersistenceManager

    def test_persistence_defaults_identity(self):
        assert DEFAULT_SUCCESS_MAX is obs.DEFAULT_SUCCESS_MAX
        assert DEFAULT_ERROR_MAX is obs.DEFAULT_ERROR_MAX

    def test_request_log_identity(self):
        assert RequestLog is obs.RequestLog

    def test_request_log_entry_identity(self):
        assert RequestLogEntry is obs.RequestLogEntry

    def test_profiler_state_identity(self):
        assert ProfilerState is obs.ProfilerState
