"""The ``/v1/responses/compact`` route.

Compaction has no neutral form: the request is a Responses request and the
answer carries an opaque ``compaction`` item only its upstream can read.  The
route therefore relays both verbatim, and refuses outright when the resolved
target is not a Responses upstream.  Both halves are pinned here, including
the endpoint it is sent to.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from llm_rosetta_gateway.app.routes_compact import handle_responses_compact
from llm_rosetta_gateway.config import GatewayConfig
from llm_rosetta_gateway.transport._base import UpstreamResponse

COMPACTION_BODY: dict[str, Any] = {
    "model": "gpt-5-nano",
    "input": [
        {"type": "message", "role": "user", "content": "we agreed on Tuesday"},
        {"type": "message", "role": "assistant", "content": "understood"},
    ],
}

COMPACTED: dict[str, Any] = {
    "id": "resp_1",
    "object": "response.compaction",
    "created_at": 1,
    "output": [
        {
            "id": "cmp_1",
            "type": "compaction",
            "encrypted_content": "gAAAA-opaque",
        }
    ],
    "usage": {"input_tokens": 11, "output_tokens": 3, "total_tokens": 14},
}


def _config(
    target_format: str = "open_responses",
    upstream_model: str | None = None,
    *,
    breaker: int | None = None,
) -> GatewayConfig:
    model_entry: dict[str, Any] = {"provider": "p", "type": target_format}
    if upstream_model:
        model_entry["upstream_model"] = upstream_model
    raw: dict[str, Any] = {
        "providers": {
            "p": {
                "api_key": "k",
                "base_url": "https://api.example.com/v1",
                "type": target_format,
            }
        },
        "models": {"gpt-5-nano": model_entry},
    }
    if breaker is not None:
        raw["server"] = {
            "circuit_breaker": {
                "enabled": True,
                "error_threshold": breaker,
                "cooldown_seconds": 60.0,
            }
        }
    return GatewayConfig(raw)


def _request(body: dict[str, Any], transport: Any) -> MagicMock:
    request = MagicMock()
    request.json.return_value = body
    request.headers = {}
    request.app.metrics = None
    request.app.request_log = None
    # A bare handler call has no ops context, so it records nothing (the
    # lifecycle tests below cover the ledger).
    request.app.ops_ctx = None
    request.app.transport = transport
    return request


def _transport(response: UpstreamResponse) -> MagicMock:
    transport = MagicMock()
    transport.send = AsyncMock(return_value=response)
    return transport


def _sent(transport: MagicMock) -> tuple[Any, str, dict[str, Any]]:
    args, kwargs = transport.send.call_args
    return args[0], args[1], args[2]


class TestPassThrough:
    def test_the_answer_is_relayed_unchanged(self):
        raw = json.dumps(COMPACTED).encode()
        transport = _transport(
            UpstreamResponse(status_code=200, body=COMPACTED, raw_content=raw)
        )

        response = asyncio.run(
            handle_responses_compact(_request(COMPACTION_BODY, transport), _config())
        )

        assert response.status_code == 200
        assert response.body == raw
        assert response.headers["x-rosetta-conversion"] == "passthrough"

    def test_it_is_sent_to_the_compact_sibling_of_the_responses_endpoint(self):
        transport = _transport(
            UpstreamResponse(status_code=200, body=COMPACTED, raw_content=b"{}")
        )

        asyncio.run(
            handle_responses_compact(_request(COMPACTION_BODY, transport), _config())
        )

        _, url, _ = _sent(transport)
        assert url == "https://api.example.com/v1/responses/compact"

    def test_the_body_is_forwarded_verbatim(self):
        transport = _transport(
            UpstreamResponse(status_code=200, body=COMPACTED, raw_content=b"{}")
        )

        asyncio.run(
            handle_responses_compact(_request(COMPACTION_BODY, transport), _config())
        )

        _, _, body = _sent(transport)
        assert body == COMPACTION_BODY

    def test_the_upstream_model_alias_replaces_the_client_name(self):
        transport = _transport(
            UpstreamResponse(status_code=200, body=COMPACTED, raw_content=b"{}")
        )

        asyncio.run(
            handle_responses_compact(
                _request(COMPACTION_BODY, transport),
                _config(upstream_model="gpt-5-nano-2025-08-07"),
            )
        )

        _, _, body = _sent(transport)
        assert body["model"] == "gpt-5-nano-2025-08-07"
        assert body["input"] == COMPACTION_BODY["input"]

    def test_a_non_json_error_body_is_not_relabelled_as_json(self):
        """The transport keeps no headers, so the relay infers the type."""
        transport = _transport(
            UpstreamResponse(
                status_code=502,
                body=None,
                raw_content=b"<html>bad gateway</html>",
            )
        )

        response = asyncio.run(
            handle_responses_compact(_request(COMPACTION_BODY, transport), _config())
        )

        assert response.status_code == 502
        assert response.headers["Content-Type"].startswith("text/plain")

    def test_an_upstream_error_is_relayed_with_its_status(self):
        error = {"error": {"message": "nope", "type": "invalid_request_error"}}
        transport = _transport(
            UpstreamResponse(
                status_code=400,
                body=error,
                raw_content=json.dumps(error).encode(),
            )
        )

        response = asyncio.run(
            handle_responses_compact(_request(COMPACTION_BODY, transport), _config())
        )

        assert response.status_code == 400
        assert json.loads(response.body) == error


class TestRefusals:
    def test_a_target_that_is_not_a_responses_upstream_gets_501(self):
        transport = _transport(
            UpstreamResponse(status_code=200, body={}, raw_content=b"{}")
        )

        response = asyncio.run(
            handle_responses_compact(
                _request(COMPACTION_BODY, transport), _config("chat_completions")
            )
        )

        assert response.status_code == 501
        assert "open_responses" in json.loads(response.body)["error"]["message"]
        transport.send.assert_not_called()

    def test_a_missing_model_is_rejected_before_any_upstream_call(self):
        transport = _transport(
            UpstreamResponse(status_code=200, body={}, raw_content=b"{}")
        )
        body = {k: v for k, v in COMPACTION_BODY.items() if k != "model"}

        response = asyncio.run(
            handle_responses_compact(_request(body, transport), _config())
        )

        assert response.status_code == 400
        assert "model" in json.loads(response.body)["error"]["message"]
        transport.send.assert_not_called()

    def test_an_unknown_model_is_a_404(self):
        transport = _transport(
            UpstreamResponse(status_code=200, body={}, raw_content=b"{}")
        )

        response = asyncio.run(
            handle_responses_compact(
                _request({**COMPACTION_BODY, "model": "nope"}, transport), _config()
            )
        )

        assert response.status_code == 404
        transport.send.assert_not_called()


class TestTheLedger:
    def test_the_compaction_counters_are_recorded(self, monkeypatch):
        recorded: list[dict[str, Any]] = []

        async def _capture(_request_, **kwargs):
            recorded.append(kwargs)

        monkeypatch.setattr(
            "llm_rosetta_gateway.app.routes_compact.finish_typed", _capture
        )
        transport = _transport(
            UpstreamResponse(status_code=200, body=COMPACTED, raw_content=b"{}")
        )

        asyncio.run(
            handle_responses_compact(_request(COMPACTION_BODY, transport), _config())
        )

        assert recorded[0]["usage"] == {
            "prompt_tokens": 11,
            "completion_tokens": 3,
            "total_tokens": 14,
        }
        assert recorded[0]["model_type"] == "llm"
        assert recorded[0]["status_code"] == 200


class TestThroughTheLifecycle:
    """The route as the router runs it: lifecycle, breaker and ledger.

    A unit call cannot see the breaker, and the breaker is where a gateway-side
    refusal can be mistaken for a provider failure.
    """

    def _lifecycle_request(self, body: dict[str, Any], transport: Any) -> MagicMock:
        request = _request(body, transport)
        request.app.persistence = None
        request.app.profiler_state = None
        request.app.ops_ctx = None
        return request

    def test_a_501_is_not_charged_to_the_provider(self, monkeypatch):
        from llm_rosetta_gateway.app import _state
        from llm_rosetta_gateway.app.routes_compact import route_responses_compact
        from llm_rosetta_gateway.middleware.circuit_breaker import CircuitState

        config = _config("chat_completions", breaker=2)
        monkeypatch.setattr(_state, "_config", config)
        transport = _transport(
            UpstreamResponse(status_code=200, body={}, raw_content=b"{}")
        )
        request = self._lifecycle_request(COMPACTION_BODY, transport)

        for _ in range(3):
            response = asyncio.run(route_responses_compact(request))
            assert response.status_code == 501

        assert transport.send.await_count == 0
        breaker = config.circuit_breaker_registry.get_or_create("p")
        assert breaker.state == CircuitState.CLOSED

    def test_an_upstream_failure_still_counts(self, monkeypatch):
        """The breaker keeps doing its job for failures that are the provider's."""
        from llm_rosetta_gateway.app import _state
        from llm_rosetta_gateway.app.routes_compact import route_responses_compact
        from llm_rosetta_gateway.middleware.circuit_breaker import CircuitState

        config = _config(breaker=2)
        monkeypatch.setattr(_state, "_config", config)
        transport = _transport(
            UpstreamResponse(status_code=500, body=None, raw_content=b"boom")
        )
        request = self._lifecycle_request(COMPACTION_BODY, transport)

        for _ in range(2):
            response = asyncio.run(route_responses_compact(request))
            assert response.status_code == 500

        breaker = config.circuit_breaker_registry.get_or_create("p")
        assert breaker.state == CircuitState.OPEN
