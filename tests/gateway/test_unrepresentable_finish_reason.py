"""A terminal upstream state the client's dialect cannot express.

Chat Completions has no finish reason for a generation that failed, so the
writer raises instead of telling the client the turn finished.  The gateway
answers 502 — the upstream produced something this gateway cannot represent —
and keeps the payload for its error dump.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from llm_rosetta_gateway.config import GatewayConfig
from llm_rosetta_gateway.proxy import ProviderMetadataStore, handle_non_streaming
from llm_rosetta_gateway.transport._base import UpstreamResponse

_REQUEST: dict[str, Any] = {
    "model": "m",
    "messages": [{"role": "user", "content": "hello"}],
}

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
    """An upstream that answers with one scripted Google response."""

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


def _run():
    config = GatewayConfig(
        {
            "providers": {
                "g": {
                    "type": "google--google_generate",
                    "api_key": "k",
                    "base_url": "https://g.test",
                }
            },
            "models": {"m": {"providers": ["g"]}},
        }
    )
    transport = _Transport(
        UpstreamResponse(
            status_code=200,
            body=_MALFORMED,
            raw_content=json.dumps(_MALFORMED).encode(),
        )
    )
    selection = config.selection_context("m", "chat_completions")
    route, pinfo = config.resolve(selection)
    body = dict(_REQUEST)
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

    return asyncio.run(_go()), transport


def test_a_terminal_failure_the_client_cannot_express_answers_502():
    (response, _profile), transport = _run()

    assert transport.calls == 1
    assert response.status_code == 502
    body = json.loads(response.body)
    assert "cannot represent" in body["error"]["message"]
    # The client's request was fine; the upstream produced the bad turn.
    assert body["error"]["type"] == "api_error"
