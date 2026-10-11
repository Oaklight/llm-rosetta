"""Tests for decision source-format detection in the gateway handler."""

from types import SimpleNamespace

from llm_rosetta.gateway.config import GatewayConfig
from llm_rosetta.gateway.pipelines.decision import _detect_decision_source_format


def _req(path: str = "") -> SimpleNamespace:
    return SimpleNamespace(path=path)


_CFG = GatewayConfig({"default_decision_format": "typesafe"})


class TestDetectDecisionSourceFormat:
    def test_path_v1_decisions_is_openai(self):
        assert (
            _detect_decision_source_format(_req("/v1/decisions"), {}, _CFG)
            == "openai_decisions"
        )

    def test_path_v1_systemone_is_typesafe(self):
        assert (
            _detect_decision_source_format(_req("/v1/systemone"), {}, _CFG)
            == "typesafe"
        )

    def test_openai_body_on_canonical_route(self):
        body = {
            "input": "x",
            "questions": [{"type": "predicate", "name": "q", "instructions": "i"}],
        }
        assert (
            _detect_decision_source_format(_req("/v1/decision"), body, _CFG)
            == "openai_decisions"
        )

    def test_systemone_body_on_canonical_route(self):
        body = {"state": "x", "questions": {"q": {"type": "noul", "instructions": "i"}}}
        assert (
            _detect_decision_source_format(_req("/v1/decision"), body, _CFG)
            == "typesafe"
        )

    def test_falls_back_to_configured_default(self):
        cfg = GatewayConfig({"default_decision_format": "openai_decisions"})
        assert (
            _detect_decision_source_format(_req("/v1/decision"), {}, cfg)
            == "openai_decisions"
        )

    def test_related_prefix_is_not_matched(self):
        assert (
            _detect_decision_source_format(_req("/v1/decisionsXYZ"), {}, _CFG)
            == "typesafe"
        )
