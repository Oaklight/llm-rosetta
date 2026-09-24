"""Tests for the observability ProfilerState (standalone, no gateway)."""

from llm_rosetta.observability import ProfilerState


class TestProfilerState:
    def test_initial_state(self):
        state = ProfilerState()
        assert not state.enabled
        assert state.remaining == 0
        assert not state.tracing
        assert state.results == []
        status = state.status()
        assert status["enabled"] is False
        assert status["remaining"] == 0
        assert status["tracing"] is False
        assert status["results_count"] == 0

    def test_enable(self):
        state = ProfilerState()
        result = state.enable(requests=3)
        assert state.enabled
        assert state.remaining == 3
        assert not state.tracing
        assert result["enabled"] is True
        assert result["remaining"] == 3
        assert result["tracing"] is False

    def test_enable_with_tracing(self):
        state = ProfilerState()
        result = state.enable(requests=3, tracing=True)
        assert state.tracing
        assert result["tracing"] is True

    def test_enable_clamps_minimum(self):
        state = ProfilerState()
        state.enable(requests=0)
        assert state.remaining == 1

    def test_disable_resets_tracing(self):
        state = ProfilerState()
        state.enable(requests=5, tracing=True)
        assert state.tracing
        result = state.disable()
        assert not state.enabled
        assert state.remaining == 0
        assert not state.tracing
        assert result["enabled"] is False
        assert result["tracing"] is False

    def test_should_profile_countdown(self):
        state = ProfilerState()
        state.enable(requests=2)
        assert state.should_profile() is True
        assert state.remaining == 1
        assert state.should_profile() is True
        assert state.remaining == 0
        assert not state.enabled
        assert state.should_profile() is False

    def test_create_profiler_default(self):
        state = ProfilerState()
        state.enable(requests=1)
        profiler = state.create_profiler()
        assert not profiler.is_tracing

    def test_create_profiler_tracing(self):
        state = ProfilerState()
        state.enable(requests=1, tracing=True)
        profiler = state.create_profiler()
        assert profiler.is_tracing

    def test_store_result_records_tracing_flag(self):
        state = ProfilerState()

        class _FakeProfiler:
            is_tracing = True

            def output_html(self):
                return "<html/>"

            def output_text(self):
                return "text"

        state.store_result(_FakeProfiler(), model="test")
        assert state.results[0]["tracing"] is True

    def test_store_result_records_non_tracing(self):
        state = ProfilerState()

        class _FakeProfiler:
            is_tracing = False

            def output_html(self):
                return "<html/>"

            def output_text(self):
                return "text"

        state.store_result(_FakeProfiler(), model="test")
        assert state.results[0]["tracing"] is False

    def test_clear_results(self):
        state = ProfilerState()
        state.results.append({"test": True})
        assert len(state.results) == 1
        state.clear_results()
        assert len(state.results) == 0

    def test_max_results_trim(self):
        state = ProfilerState(max_results=3)

        class _FakeProfiler:
            is_tracing = False

            def output_html(self):
                return "<html/>"

            def output_text(self):
                return "text"

        for i in range(5):
            state.store_result(_FakeProfiler(), model=f"model-{i}")
        assert len(state.results) == 3
        assert state.results[0]["model"] == "model-2"
