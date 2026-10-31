"""Cross-provider failover when the serving provider's allowance is spent.

A 429 demotes the credential that earned it.  When that leaves the provider
with nothing to retry on, the request usually still has somewhere to go — the
model's other providers.  That is opt-in per model, spent once per request, and
recorded so a reader of the request log can see it happened.
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
    """A plain upstream failure: no quota signal, nothing to retry."""
    raw = b'{"error": {"message": "internal error", "type": "server_error"}}'
    return UpstreamResponse(status_code=500, body=None, raw_content=raw)


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

    def test_a_plain_upstream_error_is_the_answer_after_one_attempt(self):
        """``BREAK``: report this attempt's error, and do not try again.

        A decision-level test cannot tell ``BREAK`` from a handler that falls
        out of the loop by accident; what tells them apart is that the client
        sees the upstream's own 500 and the upstream was asked exactly once.
        """
        config = _config()
        transport = _Transport(_server_error())

        response, profile = _run(config, transport)

        assert response.status_code == 500
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
        return self._responses.pop(0)


def _stream_refused() -> Any:
    return _ErrorStream(429, _RATE_LIMITED.decode())


def _run_stream(config: GatewayConfig, transport: _StreamTransport):
    selection = config.selection_context("m", "chat_completions")
    route, pinfo = config.resolve(selection)
    body = {**_REQUEST, "stream": True}

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
        config.model_failover_on_quota = {"m": True}
        return config

    def test_a_keyless_refusal_can_fail_over(self):
        config = self._keyless_config()
        transport = _Transport(_refused(), _ok())

        response, profile = _run(config, transport)

        assert response.status_code == 200
        assert transport.calls == ["https://a.test", "https://b.test"]
        # The keyless branch classifies before any probe is consulted.
        assert profile["failover"]["reason"] == "keyless lane rate limited"

    def test_it_blocks_the_lane_as_before(self):
        """Classification must not cost the cooldown the 429 used to set."""
        from llm_rosetta_gateway.quota.service import note_upstream_quota_error
        from llm_rosetta_gateway.quota.state import QUOTA_STATE

        config = self._keyless_config()
        pinfo = config.providers["a"]
        pinfo.quota_identity = "acct:keyless-test"

        signal = note_upstream_quota_error(pinfo, 429, "{}")

        assert signal is not None
        assert QUOTA_STATE.provider_block("acct:keyless-test") is not None


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

    Every other test assigns ``config.model_failover_on_quota`` by hand, which
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
                "models": {"m": {"provider": "a", "failover_on_quota": True}},
            }
        )

        assert config.model_failover_on_quota == {"m": True}

    def test_the_default_is_off(self):
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

        assert config.model_failover_on_quota == {}


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
