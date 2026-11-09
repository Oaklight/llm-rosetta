"""Cross-provider failover when the serving provider cannot answer.

Two kinds of refusal qualify: an allowance that ran out (a 429 the quota probe
can explain, or a 402) and a transient upstream failure (5xx, 408, 529).  When
one leaves the provider with nothing to retry on, the request usually still has
somewhere to go — the model's other providers.  That is opt-in per model, spent
once per request, and recorded so a reader of the request log can see it
happened.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import pytest

from llm_rosetta_gateway._vendor.httpserver import JSONResponse
from llm_rosetta_gateway.config import GatewayConfig
from llm_rosetta_gateway.proxy import (
    ProviderMetadataStore,
    handle_non_streaming,
    handle_streaming,
)
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


@pytest.fixture(autouse=True)
def _clean_quota_state():
    """The quota state is process-wide: a 60 s block must not leak forward."""
    from llm_rosetta_gateway.quota.state import QUOTA_STATE

    QUOTA_STATE.clear()
    yield
    QUOTA_STATE.clear()


class _Probe:
    """A vendor probe whose only job here is to classify the refusal."""

    def inband(self, status_code: int, body: Any) -> Any:
        from llm_rosetta.shims.quota_probe import InBandSignal

        if status_code != 429:
            return None
        return InBandSignal(cooldown_seconds=60, reason="allowance spent")


class _Transport:
    """An upstream that answers with the scripted responses, in order."""

    def __init__(self, *responses: Any) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []
        self.models: list[Any] = []

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
        self.models.append(body.get("model"))
        return self._responses.pop(0)


def _ok() -> UpstreamResponse:
    return UpstreamResponse(
        status_code=200, body=_OK_BODY, raw_content=json.dumps(_OK_BODY).encode()
    )


def _refused() -> UpstreamResponse:
    return UpstreamResponse(status_code=429, body=None, raw_content=_RATE_LIMITED)


def _server_error() -> UpstreamResponse:
    """A plain 500: no quota signal, but a transient refusal."""
    raw = b'{"error": {"message": "internal error", "type": "server_error"}}'
    return UpstreamResponse(status_code=500, body=None, raw_content=raw)


def _unavailable() -> UpstreamResponse:
    """A 503: the provider is up but cannot serve right now."""
    raw = b'{"error": {"message": "Service unavailable", "type": "server_error"}}'
    return UpstreamResponse(status_code=503, body=None, raw_content=raw)


def _transient(status_code: int) -> UpstreamResponse:
    """Any transient status; the body is never read for the decision."""
    raw = b'{"error": {"message": "not now", "type": "server_error"}}'
    return UpstreamResponse(status_code=status_code, body=None, raw_content=raw)


def _config(
    *,
    providers: tuple[str, ...] = ("a", "b"),
    opt_in: bool = True,
    probe: bool = True,
    lanes: list[dict[str, Any]] | None = None,
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
            "models": {"m": {"providers": lanes or list(providers)}},
        }
    )
    if probe:
        for name in providers:
            pinfo = config.providers[name]
            pinfo.quota_probe = "p-failover"
            pinfo.quota_probe_impl = _Probe()
    # The config defaults failover on, so an opted-out test says so explicitly.
    config.model_failover = {"m": True} if opt_in else {}
    return config


def _run(config: GatewayConfig, transport: _Transport):
    selection = config.selection_context("m", "chat_completions")
    route, pinfo = config.resolve(selection)
    body = dict(_REQUEST)
    # The route handler rewrites the model to the provider's upstream alias
    # before the proxy runs; a handler called directly must do the same.
    if route.upstream_model:
        body["model"] = route.upstream_model

    async def _go():
        return await handle_non_streaming(
            route,
            pinfo,
            body,
            selection=selection,
            transport=transport,
            metadata_store=ProviderMetadataStore(),
            config=config,
        )

    return asyncio.run(_go())


def _stream_required_config() -> GatewayConfig:
    """Provider ``a`` refuses ``stream: false``; ``b`` answers either way."""
    config = _config()
    config.providers["a"].stream_required = True
    return config


class TestTheStreamRequiredBridge:
    def test_a_refusal_is_classified_and_failed_over(self):
        """A client asking for one body must not lose the failover.

        The bridge used to handle only 401, so a 429 on a stream-required
        provider went straight back to the client even when the model had
        another provider and had opted in.
        """
        config = _stream_required_config()
        transport = _StreamTransport(_stream_refused(), _ok())

        response, profile = _run(config, transport)

        assert response.status_code == 200
        assert transport.calls == ["https://a.test", "https://b.test"]
        assert profile["failover"]["from"] == "a"
        assert profile["failover"]["to"] == "b"

    def test_an_opted_out_model_keeps_the_refusal(self):
        config = _stream_required_config()
        config.model_failover = {}
        transport = _StreamTransport(_stream_refused())

        response, profile = _run(config, transport)

        assert response.status_code == 429
        assert transport.calls == ["https://a.test"]
        assert "failover" not in profile


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

    def test_a_lane_with_no_probe_still_fails_over(self):
        """A 429 needs no probe to be a rate limit.

        This used to be "no probe, no signal: do not guess".  The guess was
        wrong for the lane it mattered on: a provider with no quota probe
        answered 429, the request was answered with that refusal, and the next
        request went to the same lane as if nothing had happened.
        """
        config = _config(probe=False, opt_in=True)
        transport = _Transport(_refused(), _ok())

        response, profile = _run(config, transport)

        assert response.status_code == 200
        assert transport.calls == ["https://a.test", "https://b.test"]
        assert profile["failover"]["reason"] == "upstream answered 429"

    def test_a_transient_upstream_error_is_failed_over(self):
        """A 500 says the provider cannot answer; the other one is tried."""
        config = _config()
        transport = _Transport(_server_error(), _ok())

        response, profile = _run(config, transport)

        assert response.status_code == 200
        assert transport.calls == ["https://a.test", "https://b.test"]
        assert profile["failover"]["reason"] == "upstream 500"

    def test_a_503_is_failed_over(self):
        """The case that motivated this: DeepSeek started answering 503."""
        config = _config()
        transport = _Transport(_unavailable(), _ok())

        response, profile = _run(config, transport)

        assert response.status_code == 200
        assert transport.calls == ["https://a.test", "https://b.test"]
        assert profile["failover"]["reason"] == "upstream 503"

    @pytest.mark.parametrize(
        "status", (408, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529)
    )
    def test_every_transient_status_fails_over(self, status):
        """The set, end to end: narrowing it must red a test, not a log."""
        config = _config()
        transport = _Transport(_transient(status), _ok())

        response, profile = _run(config, transport)

        assert response.status_code == 200
        assert transport.calls == ["https://a.test", "https://b.test"]
        assert profile["failover"]["reason"] == f"upstream {status}"

    def test_a_disabled_provider_is_routed_away(self):
        """Three capacity refusals in the window take the provider out.

        Each refusal is still served by the failover; the point of the count
        is the next request, which should not try the provider at all.  The
        first refusal also sinks it, so this drives the count directly rather
        than through three requests whose order the sink decides.
        """
        from llm_rosetta_gateway.quota.service import (
            note_transient_refusal,
            provider_cooldown,
        )
        from llm_rosetta_gateway.quota.state import QUOTA_STATE

        config = _config()
        pinfo = config.providers["a"]
        for _ in range(3):
            note_transient_refusal(pinfo, 503)

        assert QUOTA_STATE.rate_limits.latest_disable("a") is not None
        cooldown = provider_cooldown("a", pinfo)
        assert cooldown is not None
        assert cooldown["reason"] == "repeated refusals"

        transport = _Transport(_ok())
        response, _ = _run(config, transport)

        assert response.status_code == 200
        assert transport.calls == ["https://b.test"]

    def test_an_opted_out_model_keeps_a_transient_error(self):
        """Failover is opt-in: without the flag the 503 is the answer."""
        config = _config(opt_in=False)
        transport = _Transport(_unavailable())

        response, profile = _run(config, transport)

        assert response.status_code == 503
        assert transport.calls == ["https://a.test"]
        assert "failover" not in profile


class _ErrorStream:
    """An upstream that refused before sending a single event."""

    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self._text = text

    @property
    def is_error(self) -> bool:
        return True

    async def read_error(self) -> str:
        return self._text

    async def close(self) -> None:
        pass

    def __aiter__(self) -> Any:  # pragma: no cover - never iterated
        return self


class _EmptyStream:
    """An upstream that accepted the request; the test never drains it."""

    status_code = 200

    @property
    def is_error(self) -> bool:
        return False

    async def read_error(self) -> str:  # pragma: no cover - never an error
        return ""

    async def close(self) -> None:
        pass

    def __aiter__(self) -> Any:
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration


class _StreamTransport(_Transport):
    """A transport whose streaming calls answer from the same script."""

    async def send_streaming(
        self, provider_info: Any, url: str, body: Any, **kw: Any
    ) -> Any:
        provider_info.auth_headers()
        self.calls.append(provider_info.base_url)
        self.models.append(body.get("model"))
        return self._responses.pop(0)


def _stream_refused() -> Any:
    return _ErrorStream(429, _RATE_LIMITED.decode())


def _run_stream(config: GatewayConfig, transport: _StreamTransport):
    selection = config.selection_context("m", "chat_completions")
    route, pinfo = config.resolve(selection)
    body = {**_REQUEST, "stream": True}
    # Same rewrite the route handler does; a lane's alias must reach the wire.
    if route.upstream_model:
        body["model"] = route.upstream_model

    async def _go():
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

    return asyncio.run(_go())


class TestTheStreamFailover:
    """Before the first event, the answer can still come from elsewhere."""

    def test_a_refusal_before_the_first_event_is_served_by_another(self):
        from llm_rosetta_gateway._vendor.httpserver import StreamingResponse

        config = _config()
        transport = _StreamTransport(_stream_refused(), _EmptyStream())

        response, profile = _run_stream(config, transport)

        assert isinstance(response, StreamingResponse)
        assert response.status_code == 200
        assert transport.calls == ["https://a.test", "https://b.test"]
        assert profile["failover"]["from"] == "a"
        assert profile["failover"]["to"] == "b"

    def test_a_transient_refusal_moves_to_the_sibling_lane(self):
        transport = _StreamTransport(_ErrorStream(500, "boom"), _EmptyStream())

        response, profile = _run_stream(_two_lane_config(), transport)

        assert response.status_code == 200
        assert transport.models == ["fast", "eco"]
        assert profile["failover"]["scope"] == "lane"
        assert profile["failover"]["to_lane"] == "eco"

    def test_an_opted_out_model_keeps_the_refusal(self):
        config = _config(opt_in=False)
        transport = _StreamTransport(_stream_refused(), _EmptyStream())

        response, profile = _run_stream(config, transport)

        assert response.status_code == 429
        assert transport.calls == ["https://a.test"]
        assert "failover" not in profile


class TestTheKeylessLane:
    """A keyless 429 has no key to demote, and still belongs to the quota class.

    The one-line change that makes it return a signal is what makes such a
    lane eligible for failover at all; without this test nothing keeps it.
    """

    def _keyless_config(self) -> GatewayConfig:
        config = _config(providers=("a", "b"), opt_in=True, probe=False)
        for name in ("a", "b"):
            pinfo = config.providers[name]
            pinfo.keyless = True
            pinfo.quota_probe = "p-keyless"
            pinfo.quota_probe_impl = _Probe()
        config.model_failover = {"m": True}
        return config

    def test_a_keyless_refusal_can_fail_over(self):
        config = self._keyless_config()
        transport = _Transport(_refused(), _ok())

        response, profile = _run(config, transport)

        assert response.status_code == 200
        assert transport.calls == ["https://a.test", "https://b.test"]
        # The keyless branch classifies before any probe is consulted.
        assert profile["failover"]["reason"] == "keyless lane rate limited"

    def test_the_refusal_is_counted_against_the_provider(self):
        """A keyless lane has no credential, so the provider itself is counted.

        This used to set a 60-second block for keyless lanes only; every other
        lane answered 429 and was retried as if nothing had happened.  The
        continuous-429 rule replaces that private case: one refusal sinks the
        provider, and only a run of them takes it out.
        """
        from llm_rosetta_gateway.quota.rate_limit import RateLimitPolicy
        from llm_rosetta_gateway.quota.service import (
            note_upstream_quota_error,
            note_upstream_rate_limit,
        )
        from llm_rosetta_gateway.quota.state import QUOTA_STATE

        config = self._keyless_config()
        pinfo = config.providers["a"]
        pinfo.quota_identity = "acct:keyless-test"

        assert note_upstream_quota_error(pinfo, 429, "{}") is not None
        assert QUOTA_STATE.provider_block("acct:keyless-test") is None

        assert note_upstream_rate_limit(pinfo, RateLimitPolicy()) == "sink"
        assert QUOTA_STATE.rate_limits.sunk("a") is True


class TestTheRequestLog:
    """The row must name the provider whose bytes the client received.

    The lifecycle records the first choice before the proxy runs; nothing but
    the failover itself updates it, so this is exactly the wiring that would
    rot silently.
    """

    def test_the_logged_provider_becomes_the_one_that_served(self):
        from llm_rosetta_gateway import lifecycle

        config = _config()
        selection = config.selection_context("m", "chat_completions")
        route, pinfo = config.resolve(selection)
        ctx = lifecycle.LifecycleContext(
            config=config,
            app=None,
            request_id="req-1",
            entry_id="entry-1",
            model_type="llm",
            body=dict(_REQUEST),
        )
        ctx.note_route(provider_name=route.provider_name)
        token = lifecycle._lifecycle_var.set(ctx)
        try:
            transport = _Transport(_refused(), _ok())

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

            asyncio.run(_go())
        finally:
            lifecycle._lifecycle_var.reset(token)

        assert ctx.provider_name == "b"


class _PartialStream:
    """An upstream that starts answering, then dies."""

    status_code = 200

    def __init__(self, chunks: list[dict[str, Any]], exc: BaseException) -> None:
        self._chunks = list(chunks)
        self._exc = exc

    @property
    def is_error(self) -> bool:
        return False

    async def read_error(self) -> str:  # pragma: no cover - never an error status
        return ""

    async def close(self) -> None:
        pass

    async def __aenter__(self) -> Any:
        return self

    async def __aexit__(self, *exc: object) -> None:
        pass

    def __aiter__(self) -> Any:
        return self

    async def __anext__(self) -> Any:
        if self._chunks:
            return self._chunks.pop(0)
        raise self._exc


def _first_chunk() -> dict[str, Any]:
    return {
        "id": "c1",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}}],
    }


async def _drain(response: Any) -> str:
    """Read the SSE bytes a client would receive.

    The generator writes its terminal error events and then re-raises for the
    server layer, so the bytes a client got are collected either way.
    """
    out: list[str] = []
    try:
        async for chunk in response._generator:  # noqa: SLF001 - the only reader
            out.append(chunk if isinstance(chunk, str) else chunk.decode())
    except Exception:  # noqa: BLE001 - the terminal events are already collected
        pass
    return "".join(out)


class TestTheMidStreamHalf:
    """A failure after the first event must never move provider.

    Nothing can be taken back once bytes are on the wire, so this holds only
    because the failover is decided above the ``StreamingResponse``.  Moving
    that call into the generator later would duplicate output rather than
    change a status code, which is why it gets its own guard.
    """

    def test_a_failure_after_the_first_event_ends_the_stream(self):
        from llm_rosetta_gateway._vendor.httpserver import StreamingResponse

        config = _config(providers=("a", "b", "c"), opt_in=True)
        transport = _StreamTransport(
            _stream_refused(),
            _PartialStream([_first_chunk()], RuntimeError("connection reset")),
            _ok(),
        )

        response, profile = _run_stream(config, transport)
        assert isinstance(response, StreamingResponse)
        body = asyncio.run(_drain(response))

        # The refused provider failed over once; the mid-stream failure did
        # not fail over again, so the third provider was never contacted.
        assert transport.calls == ["https://a.test", "https://b.test"]
        assert profile["failover"]["to"] == "b"
        # The client got what B produced, then a terminal error event.
        assert "hi" in body
        assert "error" in body.lower()


class TestTheConfigFile:
    """The flag has to come from the file an operator edits.

    Every other test assigns ``config.model_failover`` by hand, which
    leaves the parsing path — the only one production uses — untested.
    """

    def test_a_model_entry_can_opt_in(self):
        config = GatewayConfig(
            {
                "providers": {
                    "a": {
                        "type": "chat_completions",
                        "api_key": "sk-a",
                        "base_url": "https://a.test",
                    }
                },
                "models": {"m": {"provider": "a", "failover_on_error": True}},
            }
        )

        assert config.model_failover == {"m": True}

    def test_the_old_quota_key_still_opts_in(self):
        """``failover_on_quota`` is the older spelling; existing configs keep working."""
        config = GatewayConfig(
            {
                "providers": {
                    "a": {
                        "type": "chat_completions",
                        "api_key": "sk-a",
                        "base_url": "https://a.test",
                    }
                },
                "models": {"m": {"provider": "a", "failover_on_quota": True}},
            }
        )

        assert config.model_failover == {"m": True}

    def test_an_explicit_false_opts_out(self):
        """A model turns failover off with either spelling of the key."""
        for key in ("failover_on_error", "failover_on_quota"):
            config = GatewayConfig(
                {
                    "providers": {
                        "a": {
                            "type": "chat_completions",
                            "api_key": "sk-a",
                            "base_url": "https://a.test",
                        }
                    },
                    "models": {"m": {"provider": "a", key: False}},
                }
            )
            assert config.model_failover == {}

    def test_the_default_is_on(self):
        config = GatewayConfig(
            {
                "providers": {
                    "a": {
                        "type": "chat_completions",
                        "api_key": "sk-a",
                        "base_url": "https://a.test",
                    }
                },
                "models": {"m": "a"},
            }
        )

        assert config.model_failover == {"m": True}

    def test_a_conflicting_pair_is_refused(self):
        """Naming both spellings with different values is an error, not a coin flip."""
        with pytest.raises(ValueError, match="failover_on_error=True"):
            GatewayConfig(
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
                            "provider": "a",
                            "failover_on_error": True,
                            "failover_on_quota": False,
                        }
                    },
                }
            )

    @pytest.mark.parametrize("bad", ("false", 0, 1, "true"))
    def test_a_non_boolean_is_refused(self, bad):
        """A quoted or numeric value must not silently enable or disable it."""
        with pytest.raises(ValueError, match="non-boolean"):
            GatewayConfig(
                {
                    "providers": {
                        "a": {
                            "type": "chat_completions",
                            "api_key": "sk-a",
                            "base_url": "https://a.test",
                        }
                    },
                    "models": {"m": {"provider": "a", "failover_on_error": bad}},
                }
            )


class TestTheBudgetIsOneFailover:
    """The exclusion alone cannot bound the budget: with a third provider the
    round-robin would happily try it.  Measuring with three providers is what
    makes the ``attempted`` guard load-bearing."""

    def test_a_second_refusal_does_not_reach_a_third_provider(self):
        config = _config(providers=("a", "b", "c"), opt_in=True)
        transport = _Transport(_refused(), _refused(), _ok())

        response, profile = _run(config, transport)

        assert response.status_code == 429
        assert transport.calls == ["https://a.test", "https://b.test"]
        assert profile["failover"]["to"] == "b"


def _two_lane_config() -> GatewayConfig:
    """One provider with two lanes, and a third provider that would outbid one.

    ``a/fast`` carries the weight that makes it the first pick; ``b`` outbids
    ``a/eco`` in any plain re-selection, so a test that sees ``eco`` served has
    proven the sibling was tried first, not that the weights happened to agree.
    """
    return _config(
        providers=("a", "b"),
        lanes=[
            {"name": "a", "upstream_model": "fast", "weight": 100},
            {"name": "a", "upstream_model": "eco"},
            {"name": "b", "weight": 9},
        ],
    )


class TestTheSiblingLane:
    """A content-level refusal is about one upstream model, not its key (#1430).

    The lane that answered leaves the route; the provider's other lanes stay
    in it, and they are tried before another provider — the same account, the
    same base URL, so the cheapest retry is also the most likely to work.
    """

    def test_a_5xx_is_retried_on_the_sibling_lane(self):
        transport = _Transport(_server_error(), _ok())

        response, profile = _run(_two_lane_config(), transport)

        assert response.status_code == 200
        # ``b`` outweighs ``eco``: only the sibling pin can have chosen it.
        assert transport.models == ["fast", "eco"]
        failover = profile["failover"]
        assert (failover["from"], failover["from_lane"]) == ("a", "fast")
        assert (failover["to"], failover["to_lane"]) == ("a", "eco")
        assert failover["scope"] == "lane"

    def test_a_429_takes_both_lanes_out_and_another_provider_serves(self):
        """A quota refusal is one credential's state: the sibling shares it."""
        transport = _Transport(_refused(), _ok())

        response, profile = _run(_two_lane_config(), transport)

        assert response.status_code == 200
        assert transport.models == ["fast", "m"]
        assert profile["failover"]["to"] == "b"
        assert profile["failover"]["scope"] == "credential"

    def test_both_lanes_refusing_reaches_another_provider(self):
        transport = _Transport(_server_error(), _server_error(), _ok())

        response, profile = _run(_two_lane_config(), transport)

        assert response.status_code == 200
        assert transport.models == ["fast", "eco", "m"]
        outer = profile["failover"]
        assert (outer["from_lane"], outer["to_lane"]) == ("fast", "eco")
        inner = outer["profile"]["failover"]
        assert (inner["from_lane"], inner["to"]) == ("eco", "b")

    def test_the_content_level_budget_stops_at_three_attempts(self):
        transport = _Transport(_server_error(), _server_error(), _server_error(), _ok())

        response, _profile = _run(_two_lane_config(), transport)

        assert response.status_code == 500
        assert transport.models == ["fast", "eco", "m"]

    def test_the_refused_lane_is_the_one_selection_chose_not_the_route_alias(self):
        """The lane key comes from the selection, not from the route.

        A model-level ``upstream_model`` alias is what a request is sent as, not
        which candidate was picked: a key read off the route would name a lane
        no candidate has, exclude nothing, and retry the lane that just
        refused.
        """
        config = GatewayConfig(
            {
                "providers": {
                    name: {
                        "type": "chat_completions",
                        "api_key": f"sk-{name}",
                        "base_url": f"https://{name}.test",
                    }
                    for name in ("a", "b")
                },
                "models": {
                    "m": {
                        "providers": [{"name": "a", "weight": 100}, {"name": "b"}],
                        "upstream_model": "alias",
                    }
                },
            }
        )
        config.model_failover = {"m": True}
        transport = _Transport(_server_error(), _ok())

        response, profile = _run(config, transport)

        assert response.status_code == 200
        # ``a`` was tried once: the lane key matched the candidate that
        # refused, so the retry moved to ``b`` instead of repeating ``a``.
        assert transport.calls == ["https://a.test", "https://b.test"]
        assert profile["failover"]["to"] == "b"

    def test_a_route_of_single_lane_providers_is_bounded(self):
        """Every provider has one lane, so each sibling step lands elsewhere.

        Counting kinds rather than steps let this shape walk the whole route:
        the first provider had no sibling, so ``lane_retried`` never became
        true and every refusal bought another hop.
        """
        config = _config(providers=("a", "b", "c", "d"))
        transport = _Transport(_server_error(), _server_error(), _server_error())

        response, profile = _run(config, transport)

        assert response.status_code == 500
        assert transport.calls == [
            "https://a.test",
            "https://b.test",
            "https://c.test",
        ]
        assert profile["failover"]["profile"]["failover"]["to"] == "c"

    def test_the_same_bound_holds_when_the_client_asked_for_a_stream(self):
        """One decision serves both transports, so the bound is shared."""
        config = _config(providers=("a", "b", "c", "d"))
        transport = _StreamTransport(
            _ErrorStream(500, "boom"),
            _ErrorStream(500, "boom"),
            _ErrorStream(500, "boom"),
        )

        response, _profile = _run_stream(config, transport)

        assert response.status_code == 500
        assert transport.calls == [
            "https://a.test",
            "https://b.test",
            "https://c.test",
        ]

    def test_a_provider_with_one_lane_falls_through_to_another(self):
        config = _config(
            providers=("a", "b"),
            lanes=[{"name": "a", "upstream_model": "fast"}, {"name": "b"}],
        )
        transport = _Transport(_server_error(), _ok())

        response, profile = _run(config, transport)

        assert response.status_code == 200
        assert transport.models == ["fast", "m"]
        # The refusal was lane-scoped; the provider simply had no sibling.
        assert profile["failover"]["scope"] == "lane"
        assert profile["failover"]["to"] == "b"


class TestTheRetryBody:
    """The second provider gets its own upstream model name."""

    def test_the_alias_of_the_serving_provider_is_used(self):
        config = GatewayConfig(
            {
                "providers": {
                    name: {
                        "type": "chat_completions",
                        "api_key": f"sk-{name}",
                        "base_url": f"https://{name}.test",
                    }
                    for name in ("a", "b")
                },
                "models": {
                    "m": {
                        "providers": [
                            {"name": "a", "upstream_model": "alias-a"},
                            {"name": "b", "upstream_model": "alias-b"},
                        ],
                        "failover_on_quota": True,
                    }
                },
            }
        )
        for name in ("a", "b"):
            pinfo = config.providers[name]
            pinfo.quota_probe = "p-alias"
            pinfo.quota_probe_impl = _Probe()
        transport = _Transport(_refused(), _ok())

        response, _profile = _run(config, transport)

        assert response.status_code == 200
        assert transport.models == ["alias-a", "alias-b"]


class TestTheRouteWiring:
    """Both handlers only learn to fail over if the route hands them the
    config.  Nothing else in the suite drives a request through
    ``_proxy_handler``, so dropping either argument would turn the feature off
    in production and leave every test green."""

    @staticmethod
    def _drive(monkeypatch: Any, stream: bool) -> dict[str, Any]:
        from unittest.mock import MagicMock

        import llm_rosetta_gateway.app as app_module
        from llm_rosetta_gateway import lifecycle
        from llm_rosetta_gateway._vendor.httpserver import StreamingResponse
        from llm_rosetta_gateway.app import _state

        config = _config()
        captured: dict[str, Any] = {}
        body = {
            "model": "m",
            "stream": stream,
            "messages": [{"role": "user", "content": "hi"}],
        }

        async def _fake_stream_handler(*args: Any, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return StreamingResponse(
                _empty_body(), content_type="text/event-stream"
            ), {}

        async def _fake_plain_handler(*args: Any, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return JSONResponse({"ok": True}), {}

        monkeypatch.setattr(
            app_module.routes_llm, "handle_streaming", _fake_stream_handler
        )
        monkeypatch.setattr(
            app_module.routes_llm, "handle_non_streaming", _fake_plain_handler
        )
        monkeypatch.setattr(_state, "_config", config)

        ctx = lifecycle.LifecycleContext(
            config=config,
            app=None,
            request_id="req-wiring",
            entry_id="entry-wiring",
            model_type="llm",
            body=dict(body),
        )
        token = lifecycle._lifecycle_var.set(ctx)
        try:
            request = MagicMock()
            request.headers = {}
            request.json.return_value = dict(body)
            request.app.gateway_config = config
            request.app.metadata_store = MagicMock()
            request.app.metrics = None
            request.app.request_log = None
            request.app.persistence = None
            request.app.capture_state = None
            request.app.transport = MagicMock()

            asyncio.run(app_module._proxy_handler(request, "chat_completions"))
        finally:
            lifecycle._lifecycle_var.reset(token)
        return captured

    def test_the_streaming_handler_receives_the_config(self, monkeypatch):
        assert self._drive(monkeypatch, stream=True)["config"] is not None

    def test_the_plain_handler_receives_the_config(self, monkeypatch):
        assert self._drive(monkeypatch, stream=False)["config"] is not None


async def _empty_body() -> Any:
    """An SSE body that yields nothing."""
    return
    yield  # pragma: no cover - makes this an async generator


class _ExplodingTransport(_Transport):
    """Fails the second attempt the way an unexpected error would."""

    def __init__(self, *responses: Any) -> None:
        super().__init__(*responses)
        self._remaining = len(responses)

    async def send(self, provider_info: Any, url: str, body: Any, **kw: Any) -> Any:
        if self.calls:
            provider_info.auth_headers()
            self.calls.append(provider_info.base_url)
            raise RuntimeError("upstream exploded")
        return await super().send(provider_info, url, body, **kw)


class _ExplodingStreamTransport(_StreamTransport):
    """Fails the second streaming attempt the way an unexpected error would."""

    async def send_streaming(
        self, provider_info: Any, url: str, body: Any, **kw: Any
    ) -> Any:
        if self.calls:
            provider_info.auth_headers()
            self.calls.append(provider_info.base_url)
            raise RuntimeError("upstream exploded")
        return await super().send_streaming(provider_info, url, body, **kw)


class TestTheFailurePath:
    """A failure produced by the provider that ran belongs to it.

    The row used to keep the refused provider when the second attempt raised,
    so a 500 from B was attributed to A — the one place the "row names the
    server" invariant did not hold.
    """

    def test_a_raising_second_attempt_still_moves_the_log(self):
        from llm_rosetta_gateway import lifecycle

        config = _config()
        selection = config.selection_context("m", "chat_completions")
        route, pinfo = config.resolve(selection)
        ctx = lifecycle.LifecycleContext(
            config=config,
            app=None,
            request_id="req-raise",
            entry_id="entry-raise",
            model_type="llm",
            body=dict(_REQUEST),
        )
        ctx.note_route(provider_name=route.provider_name)
        token = lifecycle._lifecycle_var.set(ctx)
        transport = _ExplodingTransport(_refused())
        try:

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

            try:
                asyncio.run(_go())
            except RuntimeError:
                pass
            else:  # pragma: no cover - the failure this test exists for
                raise AssertionError("the second attempt was expected to raise")
        finally:
            lifecycle._lifecycle_var.reset(token)

        assert transport.calls[-1] == "https://b.test"
        assert ctx.provider_name == "b"

    def test_a_raising_stream_attempt_still_moves_the_log(self):
        """The streaming twin: same failure, same obligation."""
        from llm_rosetta_gateway import lifecycle

        config = _config()
        selection = config.selection_context("m", "chat_completions")
        route, pinfo = config.resolve(selection)
        ctx = lifecycle.LifecycleContext(
            config=config,
            app=None,
            request_id="req-stream-raise",
            entry_id="entry-stream-raise",
            model_type="llm",
            body=dict(_REQUEST),
        )
        ctx.note_route(provider_name=route.provider_name)
        token = lifecycle._lifecycle_var.set(ctx)
        transport = _ExplodingStreamTransport(_stream_refused())
        try:

            async def _go():
                return await handle_streaming(
                    route,
                    pinfo,
                    {**_REQUEST, "stream": True},
                    reported_model="m",
                    selection=selection,
                    transport=transport,
                    metadata_store=ProviderMetadataStore(),
                    config=config,
                )

            try:
                asyncio.run(_go())
            except RuntimeError:
                pass
            else:  # pragma: no cover - the failure this test exists for
                raise AssertionError("the second attempt was expected to raise")
        finally:
            lifecycle._lifecycle_var.reset(token)

        assert transport.calls[-1] == "https://b.test"
        assert ctx.provider_name == "b"

    def test_a_bookkeeping_failure_does_not_replace_the_client_error(self):
        """The record is best-effort: the answer always wins.

        `record_failover` runs from a `finally` on the failure path, so an
        exception of its own would otherwise become the error the client sees.
        """
        from llm_rosetta_gateway import lifecycle

        config = _config()
        selection = config.selection_context("m", "chat_completions")
        route, pinfo = config.resolve(selection)
        ctx = lifecycle.LifecycleContext(
            config=config,
            app=None,
            request_id="req-bookkeeping",
            entry_id="entry-bookkeeping",
            model_type="llm",
            body=dict(_REQUEST),
        )

        def _boom(**kwargs: Any) -> None:
            raise ValueError("bookkeeping exploded")

        # An instance attribute shadows the method: only the failover's own
        # note is made to fail, not the selection that ran before it.  Cast
        # because the attribute's declared type is the bound method.
        cast(Any, ctx).note_route = _boom
        token = lifecycle._lifecycle_var.set(ctx)
        transport = _ExplodingTransport(_refused())
        try:

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

            with pytest.raises(RuntimeError, match="upstream exploded"):
                asyncio.run(_go())
        finally:
            lifecycle._lifecycle_var.reset(token)

        assert transport.calls[-1] == "https://b.test"
