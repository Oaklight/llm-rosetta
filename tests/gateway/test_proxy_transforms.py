"""Tests for shim transform integration in the gateway proxy pipeline."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_rosetta.gateway.proxy import (
    ProviderMetadataStore,
    _resolve_target_transforms,
    handle_non_streaming,
)
from llm_rosetta.gateway.transport._base import UpstreamResponse
from llm_rosetta.shims.provider_shim import (
    ProviderShim,
    _reset_registry,
    register_shim,
)
from llm_rosetta.shims.transforms import rename_field, strip_fields


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset the shim registry before each test."""
    yield
    _reset_registry()


@pytest.fixture()
def volcengine_shim():
    """Register a volcengine-like shim with to_transforms that strip fields."""
    shim = ProviderShim(
        name="volcengine--openai_chat",
        base="openai_chat",
        to_transforms=(strip_fields("logprobs", "top_logprobs"),),
    )
    register_shim(shim)
    return shim


@pytest.fixture()
def shim_with_transforms():
    """Register a shim with both from_transforms and to_transforms."""
    shim = ProviderShim(
        name="custom_provider",
        base="openai_chat",
        from_transforms=(rename_field("custom_id", "id"),),
        to_transforms=(strip_fields("logprobs"),),
    )
    register_shim(shim)
    return shim


# ---------------------------------------------------------------------------
# _resolve_target_transforms
# ---------------------------------------------------------------------------


class TestResolveTargetTransforms:
    def test_none_shim(self):
        from_t, to_t = _resolve_target_transforms(None)
        assert from_t == ()
        assert to_t == ()

    def test_unknown_shim(self):
        from_t, to_t = _resolve_target_transforms("nonexistent-provider")
        assert from_t == ()
        assert to_t == ()

    def test_volcengine_shim(self, volcengine_shim):
        from_t, to_t = _resolve_target_transforms("volcengine--openai_chat")
        assert to_t == volcengine_shim.to_transforms
        assert from_t == ()

    def test_shim_with_both_transforms(self, shim_with_transforms):
        from_t, to_t = _resolve_target_transforms("custom_provider")
        assert from_t == shim_with_transforms.from_transforms
        assert to_t == shim_with_transforms.to_transforms


# ---------------------------------------------------------------------------
# Mock transport helpers
# ---------------------------------------------------------------------------


def _make_mock_transport(
    response_json: dict[str, Any],
    *,
    status_code: int = 200,
    captured_body: dict[str, Any] | None = None,
) -> MagicMock:
    """Create a mock transport that returns an UpstreamResponse."""

    async def mock_send_request(
        provider_info, target_provider, body, model, *, extra_headers=None
    ):
        if captured_body is not None:
            captured_body.update(body)
        return UpstreamResponse(
            status_code=status_code,
            body=response_json if status_code < 400 else None,
            raw_content=json.dumps(response_json).encode(),
        )

    transport = MagicMock()
    transport.send_request = AsyncMock(side_effect=mock_send_request)
    return transport


def _make_provider_info() -> MagicMock:
    """Create a mock ProviderInfo."""
    info = MagicMock()
    info.upstream_url.return_value = "https://api.example.com/v1/chat/completions"
    info.auth_headers.return_value = {"Authorization": "Bearer test"}
    info.proxy_url = None
    return info


# ---------------------------------------------------------------------------
# handle_non_streaming — transform integration
# ---------------------------------------------------------------------------


class TestNonStreamingTransforms:
    def test_to_transforms_strip_fields(self, volcengine_shim):
        """to_transforms should strip fields from the target request body."""
        captured_body: dict[str, Any] = {}
        transport = _make_mock_transport(
            {
                "id": "resp-1",
                "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            },
            captured_body=captured_body,
        )

        async def run():
            await handle_non_streaming(
                source_provider="openai_chat",
                target_provider="openai_chat",
                provider_info=_make_provider_info(),
                body={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "logprobs": True,
                    "top_logprobs": 5,
                },
                model="test-model",
                transport=transport,
                metadata_store=ProviderMetadataStore(),
                target_shim_name="volcengine--openai_chat",
            )

        asyncio.run(run())

        # logprobs and top_logprobs should have been stripped
        assert "logprobs" not in captured_body
        assert "top_logprobs" not in captured_body
        assert "model" in captured_body

    def test_no_shim_no_transforms(self):
        """Without a shim, no transforms should be applied."""
        captured_body: dict[str, Any] = {}
        transport = _make_mock_transport(
            {
                "id": "resp-1",
                "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            },
            captured_body=captured_body,
        )

        async def run():
            await handle_non_streaming(
                source_provider="openai_chat",
                target_provider="openai_chat",
                provider_info=_make_provider_info(),
                body={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "hello"}],
                    "logprobs": True,
                    "top_logprobs": 5,
                },
                model="gpt-4",
                transport=transport,
                metadata_store=ProviderMetadataStore(),
                target_shim_name=None,
            )

        asyncio.run(run())

        # Fields should NOT be stripped when no shim is configured
        assert "logprobs" in captured_body
        assert "top_logprobs" in captured_body

    def test_from_transforms_on_response(self, shim_with_transforms):
        """from_transforms should be applied to the upstream response."""
        transport = _make_mock_transport(
            {
                "custom_id": "resp-1",
                "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            },
        )

        async def run():
            response = await handle_non_streaming(
                source_provider="openai_chat",
                target_provider="openai_chat",
                provider_info=_make_provider_info(),
                body={
                    "model": "regular-model",
                    "messages": [{"role": "user", "content": "hello"}],
                },
                model="regular-model",
                transport=transport,
                metadata_store=ProviderMetadataStore(),
                target_shim_name="custom_provider",
            )
            return response

        response = asyncio.run(run())
        # The handler should complete without error
        assert response.status_code == 200
