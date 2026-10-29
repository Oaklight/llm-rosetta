"""A terminal upstream state the client's dialect cannot express.

Chat Completions has no finish reason for a generation that failed, the
Messages API's vocabulary describes only *finished* turns, and Google's
``OTHER`` is its catch-all for a turn the model did produce — none of the three
can say "the upstream failed".  The writer raises instead of telling the client
the turn finished; the gateway answers 502 (the upstream produced something
this gateway cannot represent) and keeps the payload for its error dump,
whatever dialect the client speaks (#1234).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

import llm_rosetta_gateway.proxy.engine as engine_module
from llm_rosetta_gateway.config import GatewayConfig
from llm_rosetta_gateway.proxy import ProviderMetadataStore, handle_non_streaming
from llm_rosetta_gateway.transport._base import UpstreamResponse

_REQUESTS: dict[str, dict[str, Any]] = {
    "chat_completions": {
        "model": "m",
        "messages": [{"role": "user", "content": "hello"}],
    },
    "messages": {
        "model": "m",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hello"}],
    },
    "google_generate": {
        "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
    },
}

#: The one field each client dialect puts the error's type/status in.
_ENVELOPE: dict[str, dict[str, Any]] = {
    "chat_completions": {"path": ("error", "type"), "value": "api_error"},
    "messages": {"path": ("error", "type"), "value": "api_error"},
    "google_generate": {"path": ("error", "status"), "value": "UNAVAILABLE"},
}

#: An upstream that reports the failure on the Responses wire.
_FAILED_RESPONSE = {
    "id": "resp_x",
    "object": "response",
    "created_at": 1,
    "status": "failed",
    "error": {"type": "server_error", "code": "boom", "message": "upstream failed"},
    "output": [],
}

#: A Google candidate whose call the model itself malformed.
_MALFORMED = {
    "candidates": [
        {
            "content": {
                "role": "model",
                "parts": [{"functionCall": {"name": "f", "args": {}}}],
            },
            "finishReason": "MALFORMED_FUNCTION_CALL",
            "index": 0,
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 1,
        "candidatesTokenCount": 2,
        "totalTokenCount": 3,
    },
    "modelVersion": "gemini-2.5-flash",
    "responseId": "r1",
}


class _Transport:
    """An upstream that answers with one scripted response."""

    def __init__(self, response: UpstreamResponse) -> None:
        self._response = response
        self.calls = 0

    async def send_streaming(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("this test never streams")

    async def close(self) -> None:
        """Nothing to close: no sockets are opened."""

    async def send(self, provider_info: Any, url: str, body: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return self._response


_CONFIG: dict[str, Any] = {
    "providers": {
        "g": {
            "type": "google--google_generate",
            "api_key": "k",
            "base_url": "https://g.test",
        },
        "r": {
            "type": "open_responses",
            "api_key": "k",
            "base_url": "https://r.test",
        },
    },
    # One model per upstream, so the route picks the dialect under test.
    "models": {"gm": {"providers": ["g"]}, "rm": {"providers": ["r"]}},
}


def _run(source: str, upstream: dict[str, Any], model: str = "gm"):
    config = GatewayConfig(_CONFIG)
    transport = _Transport(
        UpstreamResponse(
            status_code=200,
            body=upstream,
            raw_content=json.dumps(upstream).encode(),
        )
    )
    selection = config.selection_context(model, source)
    route, pinfo = config.resolve(selection)
    body = dict(_REQUESTS[source])
    body["model"] = route.upstream_model or model

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


@pytest.mark.parametrize("source", sorted(_REQUESTS))
def test_a_failed_upstream_is_a_502_whatever_the_client_speaks(source):
    """One failed upstream, one outcome: the client is told the turn failed."""
    (response, _profile) = _run(source, _FAILED_RESPONSE, model="rm")

    assert response.status_code == 502
    body = json.loads(response.body)
    envelope = _ENVELOPE[source]
    value = body
    for key in envelope["path"]:
        value = value[key]
    assert value == envelope["value"]
    assert "cannot represent" in body["error"]["message"]


@pytest.mark.parametrize("source", ["chat_completions", "messages"])
def test_a_google_word_the_client_cannot_say_is_a_502(source):
    """The upstream's own odd stop is still not something these dialects say."""
    (response, _profile) = _run(source, _MALFORMED)

    assert response.status_code == 502
    body = json.loads(response.body)
    assert "cannot represent" in body["error"]["message"]


def test_the_reason_and_response_id_reach_the_error_dump(monkeypatch):
    """The dump names the turn rather than only the exception's message."""
    dumps: list[dict[str, Any]] = []

    async def _record(_persistence: Any, **kwargs: Any) -> None:
        dumps.append(kwargs)

    monkeypatch.setattr(engine_module, "dump_error", _record)

    (response, _profile) = _run("google_generate", _FAILED_RESPONSE, model="rm")

    assert response.status_code == 502
    assert len(dumps) == 1
    assert dumps[0]["status_code"] == 502
    assert "reason='error'" in dumps[0]["response_text"]
    assert "'resp_x'" in dumps[0]["response_text"]
    assert dumps[0]["source_provider"] == "google_generate"


def test_the_same_dialect_round_trips_its_own_odd_stop():
    """Google keeps saying what Google said — it is not a foreign failure."""
    (response, _profile) = _run("google_generate", _MALFORMED)

    assert response.status_code == 200
    body = json.loads(response.body)
    assert body["candidates"][0]["finishReason"] == "MALFORMED_FUNCTION_CALL"
