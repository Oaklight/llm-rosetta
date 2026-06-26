"""Tests for ProfilerState and profiling admin routes."""

from llm_rosetta.gateway.admin.routes.profiling import ProfilerState


class TestProfilerState:
    """Unit tests for the ProfilerState class."""

    def test_initial_state(self):
        state = ProfilerState()
        assert not state.enabled
        assert state.remaining == 0
        assert state.results == []
        status = state.status()
        assert status["enabled"] is False
        assert status["remaining"] == 0
        assert status["results_count"] == 0

    def test_enable(self):
        state = ProfilerState()
        result = state.enable(requests=3)
        assert state.enabled
        assert state.remaining == 3
        assert result["enabled"] is True
        assert result["remaining"] == 3

    def test_enable_clamps_minimum(self):
        state = ProfilerState()
        state.enable(requests=0)
        assert state.remaining == 1  # clamped to >= 1

    def test_disable(self):
        state = ProfilerState()
        state.enable(5)
        result = state.disable()
        assert not state.enabled
        assert state.remaining == 0
        assert result["enabled"] is False

    def test_should_profile_decrements(self):
        state = ProfilerState()
        state.enable(requests=2)

        assert state.should_profile()  # remaining: 2 -> 1
        assert state.remaining == 1
        assert state.enabled  # still enabled

        assert state.should_profile()  # remaining: 1 -> 0
        assert state.remaining == 0
        assert not state.enabled  # auto-disabled

    def test_should_profile_returns_false_when_disabled(self):
        state = ProfilerState()
        assert not state.should_profile()

    def test_should_profile_auto_disables(self):
        state = ProfilerState()
        state.enable(requests=1)
        assert state.should_profile()
        assert not state.enabled
        assert not state.should_profile()

    def test_store_result(self):
        state = ProfilerState()

        # Create a mock profiler
        class MockProfiler:
            def output_html(self):
                return "<html>flamegraph</html>"

            def output_text(self):
                return "call tree text"

        state.store_result(
            MockProfiler(),
            request_id="test-123",
            model="gpt-4o",
            source="openai_chat",
            target="anthropic",
            is_stream=False,
            duration_ms=150.3,
        )
        assert len(state.results) == 1
        result = state.results[0]
        assert result["request_id"] == "test-123"
        assert result["model"] == "gpt-4o"
        assert result["html"] == "<html>flamegraph</html>"
        assert result["text"] == "call tree text"
        assert result["duration_ms"] == 150.3

    def test_store_result_caps_at_max(self):
        state = ProfilerState(max_results=3)

        class MockProfiler:
            def output_html(self):
                return ""

            def output_text(self):
                return ""

        for i in range(5):
            state.store_result(MockProfiler(), model=f"model-{i}")

        assert len(state.results) == 3
        # Should keep the latest 3
        assert state.results[0]["model"] == "model-2"
        assert state.results[2]["model"] == "model-4"

    def test_clear_results(self):
        state = ProfilerState()

        class MockProfiler:
            def output_html(self):
                return ""

            def output_text(self):
                return ""

        state.store_result(MockProfiler())
        assert len(state.results) == 1
        state.clear_results()
        assert len(state.results) == 0


class TestRequestLogProfile:
    """Test profile field in RequestLogEntry."""

    def test_create_with_profile(self):
        from llm_rosetta.gateway.admin.request_log import RequestLogEntry

        profile = {"request_conversion_ms": 2.45, "upstream_ms": 150.3}
        entry = RequestLogEntry.create(
            model="gpt-4o",
            source_provider="openai_chat",
            target_provider="anthropic",
            is_stream=False,
            status_code=200,
            duration_ms=155.0,
            profile=profile,
        )
        assert entry.profile == profile

    def test_create_without_profile(self):
        from llm_rosetta.gateway.admin.request_log import RequestLogEntry

        entry = RequestLogEntry.create(
            model="gpt-4o",
            source_provider="openai_chat",
            target_provider="anthropic",
            is_stream=False,
            status_code=200,
            duration_ms=155.0,
        )
        assert entry.profile is None

    def test_to_dict_includes_profile(self):
        from llm_rosetta.gateway.admin.request_log import RequestLogEntry

        profile = {"request_conversion_ms": 2.45}
        entry = RequestLogEntry.create(
            model="gpt-4o",
            source_provider="openai_chat",
            target_provider="anthropic",
            is_stream=False,
            status_code=200,
            duration_ms=155.0,
            profile=profile,
        )
        d = entry.to_dict()
        assert d["profile"] == profile

    def test_to_dict_omits_none_profile(self):
        from llm_rosetta.gateway.admin.request_log import RequestLogEntry

        entry = RequestLogEntry.create(
            model="gpt-4o",
            source_provider="openai_chat",
            target_provider="anthropic",
            is_stream=False,
            status_code=200,
            duration_ms=155.0,
        )
        d = entry.to_dict()
        assert "profile" not in d

    def test_update_profile_in_memory(self):
        from llm_rosetta.gateway.admin.request_log import RequestLog, RequestLogEntry

        log = RequestLog()
        entry = RequestLogEntry.create(
            model="gpt-4o",
            source_provider="openai_chat",
            target_provider="anthropic",
            is_stream=True,
            status_code=200,
            duration_ms=100.0,
            profile={"request_conversion_ms": 2.0},
        )
        log.add(entry)

        # Update with stream metrics
        log.update_profile(
            entry.id,
            {"stream_ttfb_ms": 120.5, "stream_complete": True},
        )

        # Verify merged
        result = log.get_entry(entry.id)
        assert result is not None
        assert result["profile"]["request_conversion_ms"] == 2.0
        assert result["profile"]["stream_ttfb_ms"] == 120.5
        assert result["profile"]["stream_complete"] is True


class TestPersistenceProfile:
    """Test profile column in SQLite persistence."""

    def test_insert_and_query_with_profile(self, tmp_path):
        from llm_rosetta.gateway.admin.persistence import PersistenceManager

        pm = PersistenceManager(str(tmp_path))
        profile = {"request_conversion_ms": 2.45, "upstream_ms": 150.3}
        pm.insert_log_entries(
            [
                {
                    "id": "test-1",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "model": "gpt-4o",
                    "source_provider": "openai_chat",
                    "target_provider": "anthropic",
                    "is_stream": False,
                    "status_code": 200,
                    "duration_ms": 155.0,
                    "profile": profile,
                }
            ]
        )

        entry = pm.get_log_entry("test-1")
        assert entry is not None
        assert entry["profile"] == profile
        pm.close()

    def test_insert_without_profile(self, tmp_path):
        from llm_rosetta.gateway.admin.persistence import PersistenceManager

        pm = PersistenceManager(str(tmp_path))
        pm.insert_log_entries(
            [
                {
                    "id": "test-2",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "model": "gpt-4o",
                    "source_provider": "openai_chat",
                    "target_provider": "anthropic",
                    "is_stream": False,
                    "status_code": 200,
                    "duration_ms": 155.0,
                }
            ]
        )

        entry = pm.get_log_entry("test-2")
        assert entry is not None
        assert "profile" not in entry  # omitted when None
        pm.close()

    def test_update_entry_profile(self, tmp_path):
        from llm_rosetta.gateway.admin.persistence import PersistenceManager

        pm = PersistenceManager(str(tmp_path))
        pm.insert_log_entries(
            [
                {
                    "id": "test-3",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "model": "gpt-4o",
                    "source_provider": "openai_chat",
                    "target_provider": "anthropic",
                    "is_stream": True,
                    "status_code": 200,
                    "duration_ms": 155.0,
                    "profile": {"request_conversion_ms": 2.0},
                }
            ]
        )

        # Update with stream metrics
        pm.update_entry_profile(
            "test-3",
            {"stream_ttfb_ms": 120.5, "stream_complete": True},
        )

        entry = pm.get_log_entry("test-3")
        assert entry is not None
        assert entry["profile"]["request_conversion_ms"] == 2.0
        assert entry["profile"]["stream_ttfb_ms"] == 120.5
        assert entry["profile"]["stream_complete"] is True
        pm.close()

    def test_update_entry_profile_nonexistent(self, tmp_path):
        """Updating a non-existent entry is a no-op."""
        from llm_rosetta.gateway.admin.persistence import PersistenceManager

        pm = PersistenceManager(str(tmp_path))
        pm.update_entry_profile("nonexistent", {"foo": 1})
        pm.close()

    def test_migration_adds_profile_column(self, tmp_path):
        """Verify the profile column is added by migration."""
        import sqlite3

        db_path = tmp_path / "gateway.db"
        conn = sqlite3.connect(str(db_path))
        # Create table WITHOUT profile column (old schema)
        conn.executescript("""
            CREATE TABLE request_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                model TEXT NOT NULL,
                source_provider TEXT NOT NULL,
                target_provider TEXT NOT NULL,
                is_stream INTEGER NOT NULL,
                status_code INTEGER NOT NULL,
                duration_ms REAL NOT NULL,
                error_detail TEXT,
                api_key_label TEXT,
                target_provider_name TEXT,
                client_ip TEXT
            );
            CREATE TABLE metrics (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        conn.close()

        # PersistenceManager should add the profile column via migration
        from llm_rosetta.gateway.admin.persistence import PersistenceManager

        pm = PersistenceManager(str(tmp_path))
        cursor = pm._conn.execute("PRAGMA table_info(request_log)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "profile" in columns
        pm.close()
