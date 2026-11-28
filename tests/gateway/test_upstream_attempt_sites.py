"""Each chat path counts its upstream attempt on the lane that made it (#1830 A4).

The JSON path, the stream-only bridge and the streaming path each file a
refusal or a transport failure under the lane the selection picked and clear
that lane's record when it serves.  These pins drive the three paths that no
other test reached for the served clear or the lane key, so a change to where
the bookkeeping lives cannot drop one unseen.  The lane is ``fast``: the
candidate's upstream model, which is not the model the client asked for (``m``).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from llm_rosetta_gateway.config import GatewayConfig
from llm_rosetta_gateway.proxy import (
    ProviderMetadataStore,
    handle_non_streaming,
    handle_streaming,
)
from llm_rosetta_gateway.quota.rate_limit import RateLimitPolicy
from llm_rosetta_gateway.quota.state import QUOTA_STATE
from llm_rosetta_gateway.transport._base import UpstreamConnectionError

_REQUEST: dict[str, Any] = {
    "model": "m",
    "messages": [{"role": "user", "content": "hello"}],
}

_CHUNKS: list[dict[str, Any]] = [
    {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 1763000000,
        "model": "fast",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}}],
    },
    {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 1763000000,
        "model": "fast",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    },
]


@pytest.fixture(autouse=True)
def _clean_quota_state():
    QUOTA_STATE.clear()
    yield
    QUOTA_STATE.clear()


class _Refused:
    """An upstream that refused before a single event."""

    status_code = 503

    @property
    def is_error(self) -> bool:
        return True

    async def read_error(self) -> str:
        return '{"error": {"message": "not now", "type": "server_error"}}'

    async def close(self) -> None:
        pass


class _Served:
    """An upstream that streams a short answer."""

    status_code = 200
    is_error = False

    async def __aenter__(self) -> _Served:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def close(self) -> None:
        pass

    async def read_error(self) -> str:  # pragma: no cover - never an error
        return ""

    def __aiter__(self) -> Any:
        async def gen():
            for chunk in _CHUNKS:
                yield chunk

        return gen()


class _Transport:
    """Answers each stream from a script; ``None`` is a host that is down."""

    def __init__(self, *script: Any) -> None:
        self._script = list(script)
        self.calls = 0

    async def send(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("these paths always stream")

    async def send_streaming(
        self, provider_info: Any, url: str, body: Any, **kwargs: Any
    ) -> Any:
        provider_info.auth_headers()
        self.calls += 1
        step = self._script.pop(0)
        if step is None:
            raise UpstreamConnectionError("refused")
        return step

    async def close(self) -> None:
        pass


def _config(*, stream_required: bool = False) -> GatewayConfig:
    """One provider, two lanes under the model ``m``, failover on.

    ``fast`` is picked first; a refusal there moves to ``eco``, which is the
    same provider, so the refusal's record is what tells the paths apart.
    ``sink`` is off and the threshold high: a record is a count, and ``fast``
    stays the first pick for the next request.
    """
    config = GatewayConfig(
        {
            "providers": {
                "a": {
                    "type": "chat_completions",
                    "api_key": "sk-a",
                    "base_url": "https://a.test",
                }
            },
            "models": {
                "m": {
                    "providers": [
                        {"name": "a", "upstream_model": "fast", "weight": 100},
                        {"name": "a", "upstream_model": "eco"},
                    ]
                }
            },
        }
    )
    config.model_failover = {"m": True}
    config.routing_rate_limit = RateLimitPolicy(sink=False, after=9)
    config.providers["a"].stream_required = stream_required
    return config


def _json(config: GatewayConfig, transport: _Transport) -> Any:
    selection = config.selection_context("m", "chat_completions")
    route, pinfo = config.resolve(selection)
    body = dict(_REQUEST, model=route.upstream_model or "m")

    async def go():
        return await handle_non_streaming(
            route,
            pinfo,
            body,
            selection=selection,
            transport=transport,
            metadata_store=ProviderMetadataStore(),
            config=config,
        )

    response, _ = asyncio.run(go())
    return response


def _stream(config: GatewayConfig, transport: _Transport) -> Any:
    selection = config.selection_context("m", "chat_completions")
    route, pinfo = config.resolve(selection)
    body = dict(_REQUEST, model=route.upstream_model or "m", stream=True)

    async def go():
        return await handle_streaming(
            route,
            pinfo,
            body,
            reported_model="m",
            selection=selection,
            transport=transport,
            metadata_store=ProviderMetadataStore(),
            config=config,
        )

    response, _ = asyncio.run(go())
    return response


def _refusals(lane: str = "fast") -> int:
    record = QUOTA_STATE.rate_limits.upstream_record("a", lane)
    return 0 if record is None else record.refusals


class TestTheStreamOnlyBridge:
    """A provider that refuses ``stream: false`` is asked for a stream."""

    def test_a_refusal_is_counted_on_the_lane(self):
        response = _json(
            _config(stream_required=True), _Transport(_Refused(), _Served())
        )

        assert response.status_code == 200
        assert _refusals() == 1
        assert _refusals("m") == 0

    def test_a_served_answer_clears_the_lane(self):
        config = _config(stream_required=True)
        _json(config, _Transport(_Refused(), _Served()))
        assert _refusals() == 1

        response = _json(config, _Transport(_Served()))

        assert response.status_code == 200
        assert _refusals() == 0


class TestTheStreamingPath:
    def test_a_refusal_is_counted_on_the_lane(self):
        response = _stream(_config(), _Transport(_Refused(), _Served()))

        assert response.status_code == 200
        assert _refusals() == 1
        assert _refusals("m") == 0

    def test_a_dead_host_is_counted_on_the_lane(self):
        response = _stream(_config(), _Transport(None, _Served()))

        assert response.status_code == 200
        assert _refusals() == 1
        assert _refusals("m") == 0

    def test_a_served_stream_clears_the_lane(self):
        config = _config()
        _stream(config, _Transport(_Refused(), _Served()))
        assert _refusals() == 1

        response = _stream(config, _Transport(_Served()))

        assert response.status_code == 200
        assert _refusals() == 0
