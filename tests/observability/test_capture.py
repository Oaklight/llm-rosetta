"""Tests for CaptureState internal gating logic."""

from __future__ import annotations

from llm_rosetta.observability.capture import CapturedRequest, CaptureState


def _dummy(label: str = "") -> CapturedRequest:
    """Create a minimal CapturedRequest for testing."""
    return CapturedRequest(request_id=label)


class TestCaptureStateGating:
    """Verify that record() gates internally on enabled state."""

    def test_disabled_by_default(self) -> None:
        cs = CaptureState()
        assert not cs.enabled
        cs.record(_dummy())
        assert cs.results == []

    def test_enable_captures_exactly_n(self) -> None:
        cs = CaptureState()
        cs.enable(3)
        assert cs.enabled

        for i in range(5):
            cs.record(_dummy(f"req-{i}"))

        assert len(cs.results) == 3
        assert [r.request_id for r in cs.results] == ["req-0", "req-1", "req-2"]

    def test_auto_disables_after_n(self) -> None:
        cs = CaptureState()
        cs.enable(2)

        cs.record(_dummy("a"))
        cs.record(_dummy("b"))
        assert not cs.enabled

        # Further calls are no-ops
        cs.record(_dummy("c"))
        assert len(cs.results) == 2

    def test_disable_mid_capture(self) -> None:
        cs = CaptureState()
        cs.enable(10)

        cs.record(_dummy("first"))
        assert len(cs.results) == 1

        cs.disable()
        assert not cs.enabled

        cs.record(_dummy("after-disable"))
        assert len(cs.results) == 1
        assert cs.results[0].request_id == "first"

    def test_re_enable_after_disable(self) -> None:
        cs = CaptureState()
        cs.enable(1)
        cs.record(_dummy("a"))
        assert not cs.enabled

        cs.enable(2)
        assert cs.enabled
        cs.record(_dummy("b"))
        cs.record(_dummy("c"))
        assert len(cs.results) == 3

    def test_enable_minimum_one(self) -> None:
        cs = CaptureState()
        cs.enable(0)  # should clamp to 1
        cs.record(_dummy("only"))
        cs.record(_dummy("extra"))
        assert len(cs.results) == 1
