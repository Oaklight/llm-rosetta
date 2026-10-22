"""Cross-provider failover when the serving provider's allowance is spent.

A 429 demotes the credential that earned it.  When that leaves the provider
with nothing to retry on, the request usually still has somewhere to go — the
model's other providers.  That is opt-in per model, spent once per request, and
recorded so a reader of the request log can see it happened.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from llm_rosetta_gateway.config import GatewayConfig
from llm_rosetta_gateway.proxy import ProviderMetadataStore, handle_non_streaming
from llm_rosetta_gateway.transport._base import UpstreamResponse

_OK_BODY: dict[str, Any] = {
    "id": "resp-1",
    "object": "chat.completion",
    "choices": [{"message": {"role": "assistant", "content": "hi"}, "index": 0}],
}

_REQUEST: dict[str, Any] = {
    "model": "m",
    "messages": [{"role": "user", "content": "hello"}],
}

_RATE_LIMITED = b'{"error": {"message": "usage limit exceeded", "type": "rate_limit"}}'


class _Probe:
    """A vendor probe whose only job here is to classify the refusal."""

    def inband(self, status_code: int, body: Any) -> Any:
        from llm_rosetta.shims.quota_probe import InBandSignal

        if status_code != 429:
            return None
        return InBandSignal(cooldown_seconds=60, reason="allowance spent")


class _Transport:
    """An upstream that answers with the scripted responses, in order."""

    def __init__(self, *responses: UpstreamResponse) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    async def send_streaming(
        self, provider_info: Any, url: str, body: Any, **kw: Any
    ) -> Any:  # pragma: no cover - the non-streaming path never asks
        raise AssertionError("this test never streams")

    async def close(self) -> None:
        """Nothing to close: no sockets are opened."""

    async def send(self, provider_info: Any, url: str, body: Any, **kw: Any) -> Any:
        # The real transport picks a key here, which is what makes the
        # credential the refusal belongs to known to the quota layer.
        provider_info.auth_headers()
        self.calls.append(provider_info.base_url)
        return self._responses.pop(0)


def _ok() -> UpstreamResponse:
    return UpstreamResponse(
        status_code=200, body=_OK_BODY, raw_content=json.dumps(_OK_BODY).encode()
    )


def _refused() -> UpstreamResponse:
    return UpstreamResponse(status_code=429, body=None, raw_content=_RATE_LIMITED)


def _config(
    *,
    providers: tuple[str, ...] = ("a", "b"),
    opt_in: bool = True,
    probe: bool = True,
) -> GatewayConfig:
    config = GatewayConfig(
        {
            "providers": {
                name: {
                    "type": "chat_completions",
                    "api_key": f"sk-{name}",
                    "base_url": f"https://{name}.test",
                }
                for name in providers
            },
            "models": {"m": {"providers": list(providers)}},
        }
    )
    if probe:
        for name in providers:
            pinfo = config.providers[name]
            pinfo.quota_probe = "p-failover"
            pinfo.quota_probe_impl = _Probe()
    if opt_in:
        config.model_failover_on_quota = {"m": True}
    return config


def _run(config: GatewayConfig, transport: _Transport):
    selection = config.selection_context("m", "chat_completions")
    route, pinfo = config.resolve(selection)

    async def _go():
        return await handle_non_streaming(
            route,
            pinfo,
            dict(_REQUEST),
            selection=selection,
            transport=transport,
            metadata_store=ProviderMetadataStore(),
            config=config,
        )

    return asyncio.run(_go())


class TestTheFailover:
    def test_a_refusal_on_one_provider_is_served_by_the_other(self):
        config = _config()
        transport = _Transport(_refused(), _ok())

        response, profile = _run(config, transport)

        assert response.status_code == 200
        assert transport.calls == ["https://a.test", "https://b.test"]
        assert profile["failover"]["from"] == "a"
        assert profile["failover"]["to"] == "b"
        assert profile["failover"]["reason"] == "allowance spent"

    def test_an_opted_out_model_keeps_the_refusal(self):
        config = _config(opt_in=False)
        transport = _Transport(_refused(), _ok())

        response, profile = _run(config, transport)

        assert response.status_code == 429
        assert transport.calls == ["https://a.test"]
        assert "failover" not in profile

    def test_one_retry_budget(self):
        """Two providers tried, then the upstream's own refusal is reported."""
        config = _config()
        transport = _Transport(_refused(), _refused(), _ok())

        response, profile = _run(config, transport)

        assert response.status_code == 429
        assert transport.calls == ["https://a.test", "https://b.test"]
        # The second attempt happened, and the record says where it came from.
        assert profile["failover"]["to"] == "b"

    def test_a_single_provider_has_nowhere_to_go(self):
        config = _config(providers=("a",), opt_in=True)
        transport = _Transport(_refused())

        response, profile = _run(config, transport)

        assert response.status_code == 429
        assert transport.calls == ["https://a.test"]
        assert "failover" not in profile

    def test_an_unclassified_refusal_is_not_a_quota_refusal(self):
        """No probe, no signal: the gateway will not guess that a 429 is quota."""
        config = _config(probe=False, opt_in=True)
        transport = _Transport(_refused(), _ok())

        response, profile = _run(config, transport)

        assert response.status_code == 429
        assert transport.calls == ["https://a.test"]
        assert "failover" not in profile
