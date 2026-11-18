"""The connectivity probe reads an endpoint's face (#1595 review).

The probe iterated the flat ``embeddings_format`` / ``embeddings_path`` keys.
The v5 fold moves those into the endpoint's block, so after one load a
migrated provider was probed at the chat host for the chat endpoints only —
the embedding and rerank rows vanished from the response with nothing said,
in exactly the feature whose subject is which host a request reaches.
"""

from __future__ import annotations

import json
from typing import Any
from types import SimpleNamespace

import pytest

from llm_rosetta_gateway.admin.routes import connectivity


class _Response:
    status_code = 200


class _Client:
    """Records the URLs the probe asks for; answers 200 to everything."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def get(self, url: str, headers: Any = None) -> _Response:
        self.calls.append(("GET", url))
        return _Response()

    async def post(self, url: str, headers: Any = None, json: Any = None) -> _Response:
        self.calls.append(("POST", url))
        return _Response()


@pytest.fixture
def probe(monkeypatch):
    client = _Client()

    class _Factory:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> _Client:
            return client

        async def __aexit__(self, *exc: Any) -> None:
            return None

    import llm_rosetta_gateway._vendor.httpclient as httpclient

    monkeypatch.setattr(httpclient, "AsyncClient", _Factory)
    return client


def _request(tmp_path, entry: dict, models: dict | None = None) -> SimpleNamespace:
    from llm_rosetta_gateway.config import load_gateway_config

    path = tmp_path / "config.jsonc"
    path.write_text(json.dumps({"providers": {"mix": entry}, "models": models or {}}))
    config = load_gateway_config(str(path))
    app = SimpleNamespace(gateway_config=config, config_path=str(path))
    return SimpleNamespace(app=app, path_params={"name": "mix"})


class TestTheProbeSeesTheFace:
    @pytest.mark.asyncio
    async def test_a_migrated_provider_is_probed_at_its_face(self, tmp_path, probe):
        """The probe asks for the URL the runtime calls, not a rebuild of it.

        The face's host and the face's *path* both come from the pool the
        request is routed through.  The probe used to compose the type's
        default path onto the face's host, so a Cohere face was probed at
        ``/v2/v1/embeddings`` while the gateway called ``/v2/embed`` — a 404
        against the real vendor, reported as a broken endpoint (#1595 review).
        """
        request = _request(
            tmp_path,
            {
                "type": "openai--chat_completions",
                "base_url": "https://chat.example/v1",
                "api_key": "k",
                "embeddings": {
                    "shim": "cohere--embedding",
                    "base_url": "https://api.cohere.com/v2",
                },
            },
        )
        runtime = request.app.gateway_config.embeddings_provider_infos["mix"]

        response = await connectivity.test_provider_connectivity(request, "mix")
        body = json.loads(response.body)

        assert "embeddings" in body["endpoints"]
        posted = [url for verb, url in probe.calls if verb == "POST"]
        assert runtime.upstream_url("") in posted, (runtime.upstream_url(""), posted)
        assert body["endpoints"]["embeddings"]["url"] == runtime.upstream_url("")
        assert not any(
            url.startswith("https://chat.example/v1") and "embed" in url
            for url in posted
        ), posted

    @pytest.mark.asyncio
    async def test_a_flat_entry_is_probed_the_same_way(self, tmp_path, probe):
        """The read-side fallback keeps a read-only mount's document working."""
        request = _request(
            tmp_path,
            {
                "type": "openai--chat_completions",
                "base_url": "https://chat.example/v1",
                "api_key": "k",
                "embeddings_shim": "cohere--embedding",
            },
        )

        response = await connectivity.test_provider_connectivity(request, "mix")

        assert "embeddings" in json.loads(response.body)["endpoints"]

    @pytest.mark.asyncio
    async def test_a_face_on_its_own_host_keeps_the_formats_path(self, tmp_path, probe):
        """A same-vendor face: OpenAI's own path, not Cohere's."""
        request = _request(
            tmp_path,
            {
                "type": "openai--chat_completions",
                "base_url": "https://chat.example/v1",
                "api_key": "k",
                "embeddings": {"format": "openai"},
            },
        )
        runtime = request.app.gateway_config.embeddings_provider_infos["mix"]

        await connectivity.test_provider_connectivity(request, "mix")

        posted = [url for verb, url in probe.calls if verb == "POST"]
        assert runtime.upstream_url("") in posted, (runtime.upstream_url(""), posted)
        assert runtime.upstream_url("").endswith("/v1/embeddings")

    @pytest.mark.asyncio
    async def test_a_path_that_carries_the_model_is_probed_with_one(
        self, tmp_path, probe
    ):
        """Google puts the model in the path; an empty one asks a URL nobody uses.

        Probing with ``""`` produced ``…/models/:embedContent`` for a gateway
        that knows the model, and the row read ``ok: false`` on a URL no
        request would ever reach (#1595 review).
        """
        request = _request(
            tmp_path,
            {
                "type": "openai--chat_completions",
                "base_url": "https://chat.example/v1",
                "api_key": "k",
                "embeddings": {"format": "google_generate"},
            },
            models={"text-embedding-004": {"provider": "mix", "type": "embeddings"}},
        )

        await connectivity.test_provider_connectivity(request, "mix")

        posted = [url for verb, url in probe.calls if verb == "POST"]
        embedded = [url for url in posted if "embedContent" in url]
        assert embedded, posted
        assert "models/:embedContent" not in embedded[0], embedded
        assert "text-embedding-004" in embedded[0], embedded
