"""The wire samples the API Keys page shows.

A sample is only useful if it is a call this gateway accepts, so these cases
pin the parts a converter would not tell the panel: the client-facing path
(a model that travels in the URL must appear there, not in the body), the
auth header each wire family reads, and one sample per format of the model's
type.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from llm_rosetta_gateway.admin.routes.examples import get_key_examples


def _config(**pools: dict) -> SimpleNamespace:
    base = {
        "models": {"gpt-5-nano": object()},
        "embeddings_models": {},
        "rerank_models": {},
        "decisions_models": {},
    }
    base.update(pools)
    return SimpleNamespace(**base)


def _req(**query: str) -> SimpleNamespace:
    return SimpleNamespace(
        query_params={k: [v] for k, v in query.items()},
        app=SimpleNamespace(gateway_config=_config()),
    )


def _payload(resp) -> dict:
    return json.loads(resp.body)


def _by_format(payload: dict) -> dict[str, dict]:
    return {ex["format"]: ex for ex in payload["examples"]}


@pytest.mark.asyncio
async def test_a_chat_model_gets_one_sample_per_chat_format() -> None:
    payload = _payload(await get_key_examples(_req(type="llm")))
    assert set(_by_format(payload)) == {
        "chat_completions",
        "open_responses",
        "messages",
        "google_generate",
        "google_interactions",
    }


@pytest.mark.asyncio
async def test_the_path_is_the_gateway_route() -> None:
    got = _by_format(_payload(await get_key_examples(_req(type="llm"))))
    assert got["chat_completions"]["path"] == "/v1/chat/completions"
    assert got["messages"]["path"] == "/v1/messages"
    assert got["open_responses"]["path"] == "/v1/responses"


@pytest.mark.asyncio
async def test_a_google_model_travels_in_the_path_not_the_body() -> None:
    got = _by_format(
        _payload(await get_key_examples(_req(type="llm", model="gemini-2.5-flash")))
    )
    google = got["google_generate"]
    assert google["path"] == "/v1beta/models/gemini-2.5-flash:generateContent"
    assert "model" not in google["body"]


@pytest.mark.asyncio
async def test_each_wire_family_gets_the_header_it_reads() -> None:
    got = _by_format(_payload(await get_key_examples(_req(type="llm"))))
    assert got["chat_completions"]["auth"] == {
        "name": "Authorization",
        "value": "Bearer {key}",
    }
    assert got["messages"]["auth"] == {"name": "x-api-key", "value": "{key}"}
    assert got["google_generate"]["auth"] == {
        "name": "x-goog-api-key",
        "value": "{key}",
    }


@pytest.mark.asyncio
async def test_the_body_comes_from_the_converter() -> None:
    """The chat body is what the chat converter writes for a plain turn."""
    got = _by_format(
        _payload(await get_key_examples(_req(type="llm", model="gpt-5-nano")))
    )
    body = got["chat_completions"]["body"]
    assert body["model"] == "gpt-5-nano"
    assert body["messages"] == [{"role": "user", "content": "Hello"}]


@pytest.mark.asyncio
async def test_a_typed_model_gets_its_own_formats() -> None:
    payload = _payload(await get_key_examples(_req(type="embeddings")))
    got = _by_format(payload)
    assert "openai" in got
    assert got["openai"]["path"] == "/v1/embeddings"


@pytest.mark.asyncio
async def test_with_no_query_every_type_is_served_with_a_placeholder() -> None:
    """The panel is static: it must read the same on a deployment with no
    model of some type configured."""
    payload = _payload(await get_key_examples(_req()))
    assert payload["model"] == "<model-id>"
    types = {ex["model_type"] for ex in payload["examples"]}
    assert types == {"llm", "embeddings", "rerank", "decisions"}
    chat = _by_format(
        {"examples": [e for e in payload["examples"] if e["model_type"] == "llm"]}
    )
    assert chat["chat_completions"]["body"]["model"] == "<model-id>"
    assert (
        chat["google_generate"]["path"] == "/v1beta/models/<model-id>:generateContent"
    )


@pytest.mark.asyncio
async def test_a_named_model_fills_the_samples() -> None:
    payload = _payload(await get_key_examples(_req(type="llm", model="gpt-5-nano")))
    assert payload["model"] == "gpt-5-nano"
    assert _by_format(payload)["chat_completions"]["body"]["model"] == "gpt-5-nano"


@pytest.mark.asyncio
async def test_a_format_the_converter_cannot_build_is_left_out(monkeypatch) -> None:
    """A sample that would be wrong is worse than no sample."""
    from llm_rosetta_gateway.admin.routes import examples as mod

    real = mod._body_for

    def fake(spec, model):
        if spec.format == "messages":
            raise ValueError("no body")
        return real(spec, model)

    monkeypatch.setattr(mod, "_body_for", fake)
    got = _by_format(_payload(await get_key_examples(_req(type="llm"))))
    assert "messages" not in got
    assert "chat_completions" in got
