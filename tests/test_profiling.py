"""Tests for the core DeepProfiler (zerodep-based, always available)."""

import asyncio

import pytest

from llm_rosetta.profiling import DeepProfiler


class TestDeepProfiler:
    """Tests for DeepProfiler wrapping the vendored zerodep Profiler."""

    def test_class_importable(self):
        dp = DeepProfiler()
        assert not dp.is_running
        assert not dp.is_tracing

    def test_output_before_start_raises(self):
        dp = DeepProfiler()
        with pytest.raises(RuntimeError, match="never started"):
            dp.output_text()

    def test_stop_before_start_raises(self):
        dp = DeepProfiler()
        with pytest.raises(RuntimeError, match="not running"):
            dp.stop()

    def test_sync_context_manager(self):
        with DeepProfiler(async_mode=False) as dp:
            total = sum(range(1000))
            assert total == 499500
        text = dp.output_text()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_explicit_start_stop(self):
        dp = DeepProfiler(async_mode=False)
        dp.start()
        assert dp.is_running
        _ = sum(range(1000))
        dp.stop()
        assert not dp.is_running
        html = dp.output_html()
        assert "<html" in html.lower()

    def test_double_start_raises(self):
        dp = DeepProfiler(async_mode=False)
        dp.start()
        with pytest.raises(RuntimeError, match="already running"):
            dp.start()
        dp.stop()

    def test_output_while_running_raises(self):
        dp = DeepProfiler(async_mode=False)
        dp.start()
        with pytest.raises(RuntimeError, match="still running"):
            dp.output_text()
        dp.stop()

    def test_save_html(self, tmp_path):
        with DeepProfiler(async_mode=False) as dp:
            _ = sum(range(100))
        outfile = tmp_path / "profile.html"
        dp.save_html(str(outfile))
        assert outfile.exists()
        content = outfile.read_text()
        assert len(content) > 0

    def test_async_context_manager(self):
        async def _run():
            async with DeepProfiler(async_mode=True) as dp:
                await asyncio.sleep(0.01)
            return dp

        dp = asyncio.run(_run())
        text = dp.output_text()
        assert isinstance(text, str)

    def test_context_manager_stops_on_exception(self):
        dp = DeepProfiler(async_mode=False)
        with pytest.raises(ValueError):
            with dp:
                raise ValueError("test error")
        assert not dp.is_running
        text = dp.output_text()
        assert isinstance(text, str)


class TestDeepProfilerTracing:
    """Tests for DeepProfiler with tracing=True."""

    def test_tracing_flag(self):
        dp = DeepProfiler(tracing=True, async_mode=False)
        assert dp.is_tracing

    def test_tracing_context_manager(self):
        with DeepProfiler(tracing=True, async_mode=False) as dp:
            total = sum(range(1000))
            assert total == 499500
        text = dp.output_text()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_tracing_output_html(self):
        with DeepProfiler(tracing=True, async_mode=False) as dp:
            _ = sum(range(100))
        html = dp.output_html()
        assert "<html" in html.lower()

    def test_tracing_traces(self):
        with DeepProfiler(tracing=True, async_mode=False) as dp:
            _ = sum(range(100))
        records = dp.traces()
        assert isinstance(records, list)
        assert len(records) > 0
        rec = records[0]
        assert hasattr(rec, "func")
        assert hasattr(rec, "start_ns")
        assert hasattr(rec, "end_ns")

    def test_traces_raises_without_tracing(self):
        with DeepProfiler(tracing=False, async_mode=False) as dp:
            _ = sum(range(100))
        with pytest.raises(RuntimeError, match="tracing=True"):
            dp.traces()

    def test_tracing_async_context_manager(self):
        async def _run():
            async with DeepProfiler(tracing=True, async_mode=True) as dp:
                await asyncio.sleep(0.01)
            return dp

        dp = asyncio.run(_run())
        text = dp.output_text()
        assert isinstance(text, str)
