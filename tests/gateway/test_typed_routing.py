"""Typed endpoints resolve through the selector, not a dictionary lookup.

An embedding, rerank or decision model may name several providers.  Before
this slice the pool kept only the first, so weights, quota and a spent first
lane meant nothing to those routes.  These tests pin the three answers that
matter: a typed model's candidates carry their weights, a first provider a
reading removed is passed over, and a route with nothing left reports the same
"not ready" the chat route does.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_rosetta_gateway.config import GatewayConfig
from llm_rosetta_gateway.maintenance.startup import ProviderNotReady
from llm_rosetta_gateway.quota.models import (
    QuotaReading,
    QuotaWindow,
    credential_fingerprint,
)
from llm_rosetta_gateway.quota.state import QUOTA_STATE
from llm_rosetta_gateway.pipelines.embeddings import handle_embeddings
from llm_rosetta_gateway.routing import ProviderEntry
from llm_rosetta_gateway.transport._base import UpstreamResponse


def _raw(*, model: dict | None = None) -> dict:
    """Two providers that both declare a Jina embedding endpoint."""
    return {
        "providers": {
            "jina-a": {
                "api_key": "key-a",
                "base_url": "https://a.example.test",
                "type": "openai_chat",
                "embeddings_format": "jina",
            },
            "jina-b": {
                "api_key": "key-b",
                "base_url": "https://b.example.test",
                "type": "openai_chat",
                "embeddings_format": "jina",
            },
        },
        "models": {
            "emb": model
            if model is not None
            else {"providers": ["jina-a", "jina-b"], "type": "embeddings"},
        },
    }


def _spend(config: GatewayConfig, provider: str, probe: str = "p-typed") -> None:
    """Give *provider* a fresh reading that says its window is spent."""
    pinfo = config.embeddings_provider_infos[provider]
    pinfo.quota_probe = probe
    key = pinfo.key_ring.keys()[0]
    QUOTA_STATE.put_reading(
        QuotaReading(
            probe=probe,
            credential=credential_fingerprint(key),
            windows=[
                QuotaWindow(
                    name="5h",
                    used_percent=100,
                    span_seconds=18000,
                    resets_at=datetime.now(timezone.utc) + timedelta(hours=1),
                )
            ],
            read_at=datetime.now(timezone.utc) - timedelta(seconds=5),
        )
    )


@pytest.fixture(autouse=True)
def _clean_quota_state():
    QUOTA_STATE.clear()
    yield
    QUOTA_STATE.clear()


class TestTheCandidatePool:
    """The pool is the declared list, not its first element."""

    def test_both_providers_are_candidates(self):
        config = GatewayConfig(_raw())

        entries = config.typed_model_candidates["embeddings"]["emb"]

        assert [entry.name for entry in entries] == ["jina-a", "jina-b"]
        assert all(entry.weight == 1.0 for entry in entries)

    def test_declared_weights_survive(self):
        raw = _raw(
            model={
                "providers": [{"name": "jina-a", "weight": 3}, {"name": "jina-b"}],
                "type": "embeddings",
            }
        )

        entries = GatewayConfig(raw).typed_model_candidates["embeddings"]["emb"]

        assert entries == [
            ProviderEntry(name="jina-a", weight=3.0),
            ProviderEntry(name="jina-b", weight=1.0),
        ]

    def test_a_provider_without_the_endpoint_is_not_a_candidate(self):
        raw = _raw(model={"providers": ["chat-only", "jina-b"], "type": "embeddings"})
        raw["providers"]["chat-only"] = {
            "api_key": "key-c",
            "base_url": "https://c.example.test",
            "type": "openai_chat",
        }

        entries = GatewayConfig(raw).typed_model_candidates["embeddings"]["emb"]

        assert [entry.name for entry in entries] == ["jina-b"]

    def test_a_single_provider_model_still_resolves(self):
        raw = _raw(model={"provider": "jina-a", "type": "embeddings"})

        route = GatewayConfig(raw).resolve_typed("embeddings", "emb")

        assert route.provider_name == "jina-a"
        assert route.format == "jina"

    def test_the_first_provider_serves_an_untouched_route(self):
        route = GatewayConfig(_raw()).resolve_typed("embeddings", "emb")

        assert route.provider_name == "jina-a"


class TestEligibilityComesFirst:
    """A lane the state removed is passed over, not served."""

    def test_a_spent_first_provider_is_passed_over(self):
        config = GatewayConfig(_raw())
        _spend(config, "jina-a")

        route = config.resolve_typed("embeddings", "emb")

        assert route.provider_name == "jina-b"
        assert route.format == "jina"

    def test_a_disabled_first_provider_is_passed_over(self):
        config = GatewayConfig(_raw())
        QUOTA_STATE.rate_limits.note("jina-a", None, _policy())
        QUOTA_STATE.rate_limits.note("jina-a", None, _policy())
        QUOTA_STATE.rate_limits.note("jina-a", None, _policy())

        route = config.resolve_typed("embeddings", "emb")

        assert route.provider_name == "jina-b"

    def test_nothing_left_is_not_ready_not_unknown(self):
        config = GatewayConfig(_raw())
        _spend(config, "jina-a")
        _spend(config, "jina-b", probe="p-typed-b")

        with pytest.raises(ProviderNotReady) as caught:
            config.resolve_typed("embeddings", "emb")

        assert caught.value.model == "emb"
        assert caught.value.exhausted is True
        assert caught.value.retry_after is not None

    def test_an_excluded_provider_is_not_chosen_again(self):
        config = GatewayConfig(_raw())

        route = config.resolve_typed(
            "embeddings", "emb", excluded=frozenset({"jina-a"})
        )

        assert route.provider_name == "jina-b"

    def test_an_unknown_model_is_still_a_key_error(self):
        config = GatewayConfig(_raw())

        with pytest.raises(KeyError):
            config.resolve_typed("embeddings", "nope")


def _policy():
    """The default continuous-429 policy: the third refusal disables."""
    from llm_rosetta_gateway.quota.rate_limit import RateLimitPolicy

    return RateLimitPolicy()


def _openai_raw(*, failover: bool, opt_out: bool = False) -> dict:
    """Two providers on the OpenAI embedding dialect, one response shape."""
    model: dict = {
        "providers": ["jina-a", "jina-b"],
        "type": "embeddings",
    }
    if failover:
        model["failover_on_quota"] = True
    if opt_out:
        model["failover_on_error"] = False
    raw = _raw(model=model)
    for name in ("jina-a", "jina-b"):
        raw["providers"][name]["embeddings_format"] = "openai"
    return raw


def _request() -> MagicMock:
    request = MagicMock()
    request.json.return_value = {"model": "emb", "input": "hi"}
    request.headers = {}
    request.app.metrics = None
    request.app.request_log = None
    request.app.transport = MagicMock()
    return request


def _embedding_ok() -> UpstreamResponse:
    body = {
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
        "model": "m",
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }
    return UpstreamResponse(status_code=200, body=body, raw_content=b"{}")


def _rate_limited() -> UpstreamResponse:
    return UpstreamResponse(
        status_code=429,
        body={"error": {"message": "slow down"}},
        raw_content=b'{"error":{"message":"slow down"}}',
    )


class TestFailoverAfterARefusal:
    """A 429 tries the next provider, when the model opted in."""

    @pytest.mark.asyncio
    async def test_the_refused_provider_is_not_asked_twice(self):
        config = GatewayConfig(_openai_raw(failover=True))
        request = _request()
        request.app.transport.send = AsyncMock(
            side_effect=[_rate_limited(), _embedding_ok()]
        )

        response = await handle_embeddings(request, config)

        assert response.status_code == 200
        assert request.app.transport.send.call_count == 2
        urls = [call.args[1] for call in request.app.transport.send.call_args_list]
        assert urls[0].startswith("https://a.example.test")
        assert urls[1].startswith("https://b.example.test")

    @pytest.mark.asyncio
    async def test_no_candidate_left_is_a_503_not_a_404(self):
        config = GatewayConfig(_openai_raw(failover=False))
        _spend(config, "jina-a")
        _spend(config, "jina-b", probe="p-typed-b")
        request = _request()

        response = await handle_embeddings(request, config)

        assert response.status_code == 503
        # A spent allowance has a known end, so the client is told when to come
        # back rather than "unknown model".
        assert response.headers.get("Retry-After")

    @pytest.mark.asyncio
    async def test_an_opted_out_model_passes_the_refusal_through(self):
        config = GatewayConfig(_openai_raw(failover=False, opt_out=True))
        request = _request()
        request.app.transport.send = AsyncMock(return_value=_rate_limited())

        response = await handle_embeddings(request, config)

        assert response.status_code == 429
        assert request.app.transport.send.call_count == 1
