"""The non-chat model types reach the usage ledger.

Chat has always recorded what it moved; ``/v1/embeddings``, ``/v1/rerank`` and
``/v1/decision`` bill the same upstreams and recorded nothing.  Each handler now
records one row through
:class:`~llm_rosetta.gateway.ops.proxy.OpsProxyRequest` — the chat routes' own
write path — carrying the model type and the ledger columns its IR usage
normalizes to.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from llm_rosetta.gateway.config import GatewayConfig
from llm_rosetta.gateway.ops.base import OpsContext
from llm_rosetta.gateway.pipelines.decision import handle_decision
from llm_rosetta.gateway.pipelines.embeddings import handle_embeddings
from llm_rosetta.gateway.pipelines.rerank import handle_rerank
from llm_rosetta.gateway.transport._base import UpstreamResponse
from llm_rosetta.observability import PersistenceManager, RequestLog


class _FakeTransport:
    """Stands in for the real transport: one canned upstream response."""

    def __init__(self, response: UpstreamResponse) -> None:
        self._response = response
        self.calls = 0

    async def send(self, provider_info, url, body, *, extra_headers=None):
        self.calls += 1
        return self._response


def _upstream(body: dict[str, Any], status_code: int = 200) -> UpstreamResponse:
    payload = json.dumps(body).encode()
    return UpstreamResponse(
        status_code=status_code,
        body=body if status_code < 400 else None,
        raw_content=payload,
    )


@pytest.fixture()
async def pm(tmp_path):
    manager = await PersistenceManager.create(str(tmp_path / "gateway.db"))
    yield manager
    await manager.close()


@pytest.fixture()
def ctx(pm) -> OpsContext:
    return OpsContext(request_log=RequestLog(persistence=pm))


def _request(body: dict[str, Any], transport, ctx: OpsContext, path: str) -> MagicMock:
    request = MagicMock()
    request.json.return_value = body
    request.headers = {}
    request.path = path
    request.app = SimpleNamespace(transport=transport, ops_ctx=ctx)
    return request


async def _the_row(pm) -> dict[str, Any]:
    """The single row this request should have written."""
    rows, total = await pm.query_log_entries()
    assert total == 1, f"expected one ledger row, got {total}"
    return rows[0]


def _embedding_config() -> GatewayConfig:
    return GatewayConfig(
        {
            "providers": {
                "emb": {
                    "api_key": "k",
                    "base_url": "https://emb.invalid",
                    "type": "openai",
                }
            },
            "models": {
                "my-embed": {"provider": "emb", "capabilities": ["embedding"]},
            },
        }
    )


def _rerank_config() -> GatewayConfig:
    """A Cohere rerank provider, whose tokens carry a cache counter."""
    return GatewayConfig(
        {
            "providers": {
                "cohere": {
                    "api_key": "k",
                    "base_url": "https://rerank.invalid",
                    "rerank_format": "cohere",
                }
            },
            "models": {"rerank-model": {"provider": "cohere", "type": "rerank"}},
            "default_rerank_format": "cohere",
        }
    )


def _decision_config() -> GatewayConfig:
    return GatewayConfig(
        {
            "providers": {
                "ts": {
                    "api_key": "k",
                    "base_url": "https://decision.invalid",
                    "decision_format": "systemone",
                }
            },
            "models": {"sys-one": {"provider": "ts", "type": "decision"}},
            "default_decision_format": "systemone",
        }
    )


class TestTheEmbeddingHandler:
    async def test_a_successful_call_records_its_usage(self, pm, ctx):
        transport = _FakeTransport(
            _upstream(
                {
                    "object": "list",
                    "data": [{"object": "embedding", "embedding": [0.1], "index": 0}],
                    "model": "my-embed",
                    "usage": {"prompt_tokens": 5, "total_tokens": 5},
                }
            )
        )
        response = await handle_embeddings(
            _request(
                {"model": "my-embed", "input": ["hello"]},
                transport,
                ctx,
                "/v1/embeddings",
            ),
            _embedding_config(),
        )
        assert response.status_code == 200

        row = await _the_row(pm)
        assert row["model_type"] == "embedding"
        assert row["model"] == "my-embed"
        assert row["status_code"] == 200
        assert row["is_stream"] is False
        assert row["input_tokens"] == 5
        assert row["total_tokens"] == 5
        # Embedding has no output side, so the column stays absent rather than
        # reading zero.
        assert row.get("output_tokens") is None

    async def test_an_unknown_model_is_not_billed_and_not_logged(self, pm, ctx):
        transport = _FakeTransport(_upstream({"error": "never asked"}))
        response = await handle_embeddings(
            _request(
                {"model": "nope", "input": ["hi"]}, transport, ctx, "/v1/embeddings"
            ),
            _embedding_config(),
        )
        assert response.status_code == 404
        assert transport.calls == 0
        rows, total = await pm.query_log_entries()
        assert (rows, total) == ([], 0)


class TestTheRerankHandler:
    async def test_a_successful_call_records_its_usage(self, pm, ctx):
        transport = _FakeTransport(
            _upstream(
                {
                    "model": "rerank-model",
                    "results": [{"index": 0, "relevance_score": 0.9}],
                    "meta": {"tokens": {"input_tokens": 54, "cached_tokens": 20}},
                }
            )
        )
        response = await handle_rerank(
            _request(
                {"model": "rerank-model", "query": "q", "documents": ["a"]},
                transport,
                ctx,
                "/v1/rerank",
            ),
            _rerank_config(),
        )
        assert response.status_code == 200

        row = await _the_row(pm)
        assert row["model_type"] == "rerank"
        assert row["model"] == "rerank-model"
        assert row["input_tokens"] == 54
        # Cohere sends no total; the canonical additive rule fills one from the
        # input side.
        assert row["total_tokens"] == 54
        # RerankUsageInfo's cache counter is the ledger's cache-read column.
        assert row["cache_read_tokens"] == 20

    async def test_an_upstream_error_still_records_the_status(self, pm, ctx):
        transport = _FakeTransport(
            _upstream({"error": {"message": "rate limited"}}, status_code=429)
        )
        response = await handle_rerank(
            _request(
                {"model": "rerank-model", "query": "q", "documents": ["a"]},
                transport,
                ctx,
                "/v1/rerank",
            ),
            _rerank_config(),
        )
        assert response.status_code == 429

        row = await _the_row(pm)
        assert row["model_type"] == "rerank"
        assert row["status_code"] == 429
        assert "rate limited" in (row.get("error_detail") or "")
        assert row.get("total_tokens") is None


class TestTheDecisionHandler:
    async def test_a_successful_call_records_its_usage(self, pm, ctx):
        transport = _FakeTransport(
            _upstream(
                {
                    "model": "sys-one",
                    "answers": {"q1": {"type": "assertion", "probability": 0.7}},
                    "usage": {"input_tokens": 30, "output_tokens": 4},
                }
            )
        )
        response = await handle_decision(
            _request(
                {
                    "model": "sys-one",
                    "state": "some evidence",
                    "questions": {
                        "q1": {"type": "assertion", "instructions": "is it?"}
                    },
                },
                transport,
                ctx,
                "/v1/decision",
            ),
            _decision_config(),
        )
        assert response.status_code == 200

        row = await _the_row(pm)
        assert row["model_type"] == "decision"
        assert row["input_tokens"] == 30
        assert row["output_tokens"] == 4
        # The decision IR declares no total, so the canonical additive rule
        # fills one.
        assert row["total_tokens"] == 34


class TestTheDailyTable:
    """The dashboard's per-day table carries the same split, not a second
    reading of the same rows.

    Its per-day counters keep the meaning they had — every model type summed —
    so the dashboard renders them unchanged, and ``by_model_type`` rides
    alongside for a consumer that asks what kind of call spent the tokens.
    """

    @staticmethod
    def _row(entry_id: str, total: int, model_type: str | None) -> dict[str, Any]:
        """One logged row, as a non-chat handler writes it."""
        entry: dict[str, Any] = {
            "id": entry_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": "m",
            "source_provider": "openai",
            "target_provider": "openai",
            "is_stream": False,
            "status_code": 200,
            "duration_ms": 1.0,
            "input_tokens": total,
            "total_tokens": total,
        }
        if model_type is not None:
            entry["model_type"] = model_type
        return entry

    async def test_a_day_is_every_type_summed_and_also_split_by_type(self, pm):
        await pm.insert_log_entries(
            [
                self._row("a", 30, "embedding"),
                self._row("b", 20, "embedding"),
                self._row("c", 5, "rerank"),
                # Written by an older writer, or by a caller that never names
                # a type: chat.
                self._row("d", 7, None),
            ]
        )

        table = await pm.query_token_usage_by_day(days=7)
        (day,) = table["days"]
        assert day["request_count"] == 4
        assert day["total_tokens"] == 62
        assert {
            name: split["total_tokens"] for name, split in day["by_model_type"].items()
        } == {"embedding": 50, "rerank": 5, "llm": 7}
        # The split is the day's own counters divided, not a peer reading of
        # them: every counter has to add up to what the dashboard is shown.
        for counter, value in day.items():
            if counter in ("day", "by_model_type"):
                continue
            assert (
                sum(split[counter] for split in day["by_model_type"].values()) == value
            ), counter
        assert table["totals"]["total_tokens"] == 62
        assert table["totals"]["by_model_type"]["embedding"]["total_tokens"] == 50

    async def test_the_type_filter_narrows_the_whole_table(self, pm):
        await pm.insert_log_entries(
            [
                self._row("a", 30, "embedding"),
                self._row("b", 20, "embedding"),
                self._row("c", 5, "rerank"),
            ]
        )

        table = await pm.query_token_usage_by_day(days=7, model_type="embedding")
        (day,) = table["days"]
        assert day["request_count"] == 2
        assert day["total_tokens"] == 50
        assert set(day["by_model_type"]) == {"embedding"}
        assert table["totals"]["total_tokens"] == 50
        assert table["totals"]["by_model_type"]["embedding"]["request_count"] == 2

    async def test_an_empty_table_keeps_its_shape(self, pm):
        """The dashboard reads these keys with no traffic at all."""
        table = await pm.query_token_usage_by_day(days=7)
        assert table["days"] == []
        assert table["totals"]["total_tokens"] == 0
        assert table["totals"]["request_count"] == 0
        assert table["totals"]["by_model_type"] == {}
