"""Tests for shim transform integration in the gateway proxy pipeline."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_rosetta._vendor.httpclient import Response as HttpResponse
from llm_rosetta.gateway.proxy import (
    _resolve_target_transforms,
    handle_non_streaming,
    ProviderMetadataStore,
)
from llm_rosetta.shims.provider_shim import (
    ProviderShim,
    _reset_registry,
    register_shim,
)
from llm_rosetta.shims.transforms import strip_fields, rename_field


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset the shim registry before and after each test."""
    _reset_registry()
    yield
    _reset_registry()


@pytest.fixture()
def volcengine_shim():
    """Register a volcengine-like shim with to_transforms."""
    shim = ProviderShim(
        name="volcengine",
        base="openai_chat",
        to_transforms=(strip_fields("logprobs", "top_logprobs"),),
    )
    register_shim(shim)
    return shim


@pytest.fixture()
def shim_with_transforms():
    """Register a shim with provider-level transforms."""
    shim = ProviderShim(
        name="custom_provider",
        base="openai_chat",
        to_transforms=(strip_fields("unsupported_field"),),
        from_transforms=(rename_field("custom_id", "id"),),
    )
    register_shim(shim)
    return shim


# ---------------------------------------------------------------------------
# _resolve_target_transforms
# ---------------------------------------------------------------------------


class TestResolveTargetTransforms:
    def test_none_shim_returns_empty(self):
        from_t, to_t = _resolve_target_transforms(None, "any-model")
        assert from_t == ()
        assert to_t == ()

    def test_unknown_shim_returns_empty(self):
        from_t, to_t = _resolve_target_transforms("nonexistent", "any-model")
        assert from_t == ()
        assert to_t == ()

    def test_shim_without_transforms_returns_empty(self):
        register_shim(ProviderShim(name="plain", base="openai_chat"))
        from_t, to_t = _resolve_target_transforms("plain", "any-model")
        assert from_t == ()
        assert to_t == ()

    def test_shim_with_to_transforms(self, volcengine_shim):
        from_t, to_t = _resolve_target_transforms("volcengine", "some-model")
        assert from_t == ()
        assert len(to_t) == 1
        # Verify the transform actually strips the right fields
        body = {"logprobs": True, "top_logprobs": 5, "model": "test"}
        result = to_t[0](body)
        assert "logprobs" not in result
        assert "top_logprobs" not in result
        assert result["model"] == "test"

    def test_shim_provider_level_only(self, shim_with_transforms):
        """Only provider-level transforms are returned."""
        from_t, to_t = _resolve_target_transforms("custom_provider", "special-v1")
        assert len(from_t) == 1
        assert len(to_t) == 1

    def test_shim_same_regardless_of_model(self, shim_with_transforms):
        """Transforms are the same no matter which model name is passed."""
        from_t1, to_t1 = _resolve_target_transforms("custom_provider", "special-v1")
        from_t2, to_t2 = _resolve_target_transforms("custom_provider", "regular-model")
        assert from_t1 == from_t2
        assert to_t1 == to_t2


# ---------------------------------------------------------------------------
# handle_non_streaming — transform integration
# ---------------------------------------------------------------------------


def _make_mock_response(status_code: int, json_data: dict) -> MagicMock:
    """Create a mock HTTP response that passes isinstance checks."""
    resp = MagicMock(spec=HttpResponse)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.content = b"{}"
    resp.text = "{}"
    return resp


def _make_provider_info() -> MagicMock:
    """Create a mock ProviderInfo."""
    info = MagicMock()
    info.upstream_url.return_value = "https://api.example.com/v1/chat/completions"
    info.auth_headers.return_value = {"Authorization": "Bearer test"}
    info.proxy_url = None
    return info


class TestNonStreamingTransforms:
    def test_to_transforms_strip_fields(self, volcengine_shim):
        """to_transforms should strip fields from the target request body."""
        captured_body = {}

        async def mock_post(url, json=None, headers=None, **kwargs):
            captured_body.update(json or {})
            return _make_mock_response(
                200,
                {
                    "id": "resp-1",
                    "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                },
            )

        mock_client = AsyncMock()
        mock_client.post = mock_post

        async def run():
            with patch(
                "llm_rosetta.gateway.proxy.get_client",
                return_value=mock_client,
            ):
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
                    metadata_store=ProviderMetadataStore(),
                    target_shim_name="volcengine",
                )

        asyncio.run(run())

        # logprobs and top_logprobs should have been stripped
        assert "logprobs" not in captured_body
        assert "top_logprobs" not in captured_body
        assert "model" in captured_body

    def test_no_shim_no_transforms(self):
        """Without a shim, no transforms should be applied."""
        captured_body = {}

        async def mock_post(url, json=None, headers=None, **kwargs):
            captured_body.update(json or {})
            return _make_mock_response(
                200,
                {
                    "id": "resp-1",
                    "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                },
            )

        mock_client = AsyncMock()
        mock_client.post = mock_post

        async def run():
            with patch(
                "llm_rosetta.gateway.proxy.get_client",
                return_value=mock_client,
            ):
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
                    metadata_store=ProviderMetadataStore(),
                    target_shim_name=None,
                )

        asyncio.run(run())

        # Fields should NOT be stripped when no shim is configured
        assert "logprobs" in captured_body
        assert "top_logprobs" in captured_body

    def test_from_transforms_on_response(self, shim_with_transforms):
        """from_transforms should be applied to the upstream response."""

        async def mock_post(url, json=None, headers=None, **kwargs):
            return _make_mock_response(
                200,
                {
                    "custom_id": "resp-1",
                    "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                },
            )

        mock_client = AsyncMock()
        mock_client.post = mock_post

        async def run():
            with patch(
                "llm_rosetta.gateway.proxy.get_client",
                return_value=mock_client,
            ):
                response = await handle_non_streaming(
                    source_provider="openai_chat",
                    target_provider="openai_chat",
                    provider_info=_make_provider_info(),
                    body={
                        "model": "regular-model",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                    model="regular-model",
                    metadata_store=ProviderMetadataStore(),
                    target_shim_name="custom_provider",
                )
                return response

        response = asyncio.run(run())
        # The handler should complete without error
        assert response.status_code == 200
