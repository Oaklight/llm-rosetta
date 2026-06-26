"""Tests for the observability ProfilerState (standalone, no gateway)."""

from llm_rosetta.observability import ProfilerState


class TestProfilerState:
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
        assert state.remaining == 1

    def test_disable(self):
        state = ProfilerState()
        state.enable(requests=5)
        result = state.disable()
        assert not state.enabled
        assert state.remaining == 0
        assert result["enabled"] is False

    def test_should_profile_countdown(self):
        state = ProfilerState()
        state.enable(requests=2)
        assert state.should_profile() is True
        assert state.remaining == 1
        assert state.should_profile() is True
        assert state.remaining == 0
        assert not state.enabled
        assert state.should_profile() is False

    def test_clear_results(self):
        state = ProfilerState()
        state.results.append({"test": True})
        assert len(state.results) == 1
        state.clear_results()
        assert len(state.results) == 0

    def test_max_results_trim(self):
        state = ProfilerState(max_results=3)

        class _FakeProfiler:
            def output_html(self):
                return "<html/>"

            def output_text(self):
                return "text"

        for i in range(5):
            state.store_result(_FakeProfiler(), model=f"model-{i}")
        assert len(state.results) == 3
        assert state.results[0]["model"] == "model-2"
