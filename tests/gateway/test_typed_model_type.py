"""A fourth model type, registered by a test, needs no edits outside itself.

The model-type registry exists so a new type (embedding, rerank and decision
were the first three) is declared once.  This test registers a fake type
through the same public entry points a real one uses — a descriptor, a
``TypedHandler``, and config entries — and proves it is parsed, resolved and
served end to end without a single edit in ``config.py``, ``app.py`` or the
pipelines.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_rosetta_gateway.config import GatewayConfig
from llm_rosetta_gateway.model_types import (
    MODEL_TYPE_REGISTRY,
    ModelTypeDescriptor,
    RouteSpec,
    register_model_type,
)
from llm_rosetta_gateway.pipelines.typed import TypedHandler, handle_typed_request
from llm_rosetta_gateway.transport._base import UpstreamResponse

TYPE_NAME = "sentiment"
FORMAT = "acme"
DEFAULT_PATH = "/v1/sentiment"


class _FakePipeline:
    """Stands in for a converter pipeline: it marks what it converted."""

    warnings: list[str] = []

    def __init__(self, source_format: str, target_format: str) -> None:
        self.source_format = source_format
        self.target_format = target_format

    def convert_request(self, body: dict[str, Any]) -> dict[str, Any]:
        return {**body, "converted": self.target_format}

    def convert_response(self, body: dict[str, Any]) -> dict[str, Any]:
        return {**body, "converted_back": self.source_format}


def _fake_response_usage(_format: str, _body: Any) -> dict[str, Any]:
    return {}


@pytest.fixture
def sentiment_type():
    """Register the fake type for one test, then take it back out."""
    register_model_type(
        ModelTypeDescriptor(
            name=TYPE_NAME,
            routes=[RouteSpec(DEFAULT_PATH)],
            formats=[FORMAT],
            pipeline=None,
            supports_streaming=False,
            badge_class="cap-badge-sentiment",
            config_format_key="sentiment_format",
            config_path_key="sentiment_path",
            config_shim_key="sentiment_shim",
            default_format=FORMAT,
            default_path=DEFAULT_PATH,
            detect_source_format=lambda request, body, config: FORMAT,
        )
    )
    yield TYPE_NAME
    del MODEL_TYPE_REGISTRY[TYPE_NAME]


def _config(base_url: str) -> GatewayConfig:
    """A config that declares the fake type only through its own keys."""
    return GatewayConfig(
        {
            "providers": {
                "acme": {
                    "api_key": "k",
                    "base_url": base_url,
                    "type": "chat_completions",
                    "sentiment_format": FORMAT,
                }
            },
            "models": {
                "acme-sentiment": {"provider": "acme", "type": TYPE_NAME},
            },
        }
    )


class TestFourthType:
    def test_config_parses_and_resolves_it(self, sentiment_type: str) -> None:
        config = _config("https://acme.test")
        providers, _infos, models = config.typed_pool(TYPE_NAME)
        assert providers == {"acme": {"format": FORMAT, "sentiment_path": DEFAULT_PATH}}
        assert getattr(config, f"default_{TYPE_NAME}_format") == FORMAT
        # The typed model moved out of the chat pool into the type's pool.
        assert models == {"acme-sentiment": "acme"}
        assert "acme-sentiment" not in config.models

        route = config.resolve_typed(TYPE_NAME, "acme-sentiment")
        assert route.provider_name == "acme"
        assert route.format == FORMAT
        assert route.path == DEFAULT_PATH
        assert route.provider_info.upstream_url("") == (
            "https://acme.test" + DEFAULT_PATH
        )

    def test_explicit_path_wins(self, sentiment_type: str) -> None:
        config = GatewayConfig(
            {
                "providers": {
                    "acme": {
                        "api_key": "k",
                        "base_url": "https://acme.test",
                        "sentiment_format": FORMAT,
                        "sentiment_path": "/v2/sentiment",
                    }
                },
                "models": {"m": {"provider": "acme", "type": TYPE_NAME}},
            }
        )
        assert config.resolve_typed(TYPE_NAME, "m").path == "/v2/sentiment"

    def test_legacy_section_is_parsed_too(self, sentiment_type: str) -> None:
        config = GatewayConfig(
            {
                "sentiment_providers": {
                    "acme": {"api_key": "k", "base_url": "https://acme.test"}
                },
                "sentiment_models": {"m": "acme"},
            }
        )
        providers, _infos, models = config.typed_pool(TYPE_NAME)
        assert providers == {"acme": {"format": FORMAT, "sentiment_path": DEFAULT_PATH}}
        assert models == {"m": "acme"}

    def test_the_shared_handler_serves_it(self, sentiment_type: str) -> None:
        """One TypedHandler is the whole handler for a new type."""
        config = _config("https://acme.test")
        handler = TypedHandler(
            model_type=TYPE_NAME,
            build_pipeline=_FakePipeline,
            resolve=lambda cfg, model: _resolve(cfg, model),
            configured_models=_configured_models,
            response_usage=_fake_response_usage,
        )

        request = MagicMock()
        request.json.return_value = {"model": "acme-sentiment", "input": "hi"}
        request.headers = {}
        request.app.metrics = None
        request.app.request_log = None
        request.app.transport = MagicMock()
        request.app.transport.send = AsyncMock(
            return_value=UpstreamResponse(
                status_code=200,
                body={"label": "positive"},
                raw_content=b'{"label": "positive"}',
            )
        )

        response = asyncio.run(handle_typed_request(request, config, handler))

        assert response.status_code == 200
        assert json.loads(response.body) == {
            "label": "positive",
            "converted_back": FORMAT,
        }
        # It reached the descriptor's default path.
        _, url, sent_body = request.app.transport.send.call_args.args
        assert url == "https://acme.test" + DEFAULT_PATH
        assert sent_body["converted"] == FORMAT

    def test_unknown_model_is_a_404(self, sentiment_type: str) -> None:
        config = _config("https://acme.test")
        handler = TypedHandler(
            model_type=TYPE_NAME,
            build_pipeline=_FakePipeline,
            resolve=lambda cfg, model: _resolve(cfg, model),
            configured_models=_configured_models,
            response_usage=_fake_response_usage,
        )
        request = MagicMock()
        request.json.return_value = {"model": "nope"}
        request.headers = {}
        request.app.metrics = None
        request.app.request_log = None
        request.app.transport = MagicMock()

        response = asyncio.run(handle_typed_request(request, config, handler))

        assert response.status_code == 404
        assert "Unknown model: 'nope'" in json.loads(response.body)["error"]["message"]


def _configured_models(config: GatewayConfig):
    """The models of a type that is not declared on ``GatewayConfig``."""
    return config.typed_pool(TYPE_NAME)[2].keys()


def _resolve(config: GatewayConfig, model: str):
    """The resolver a new type writes once, over the generic one."""
    try:
        return config.resolve_typed(TYPE_NAME, model)
    except KeyError:
        return None
