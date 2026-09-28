"""Tests for routing loop detection (hop-count header + config validation)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from llm_rosetta.gateway.config import detect_routing_loops
from llm_rosetta.gateway.middleware.headers import (
    HOP_COUNT_HEADER,
    MAX_HOPS,
    build_upstream_extra_headers,
    get_hop_count,
)
from llm_rosetta.gateway.middleware.hop_limit import create_hop_limit_hook
from llm_rosetta.gateway.routing_strategy import ModelRoute, ProviderEntry


# ---------------------------------------------------------------------------
# Hop-count header tests
# ---------------------------------------------------------------------------


def test_get_hop_count_absent():
    request = MagicMock()
    request.headers = {}
    assert get_hop_count(request) == 0


def test_get_hop_count_present():
    request = MagicMock()
    request.headers = {HOP_COUNT_HEADER: "2"}
    assert get_hop_count(request) == 2


def test_get_hop_count_invalid():
    request = MagicMock()
    request.headers = {HOP_COUNT_HEADER: "garbage"}
    assert get_hop_count(request) == 0


def test_get_hop_count_negative():
    request = MagicMock()
    request.headers = {HOP_COUNT_HEADER: "-5"}
    assert get_hop_count(request) == 0


def test_build_upstream_extra_headers_increments_hop_count():
    request = MagicMock()
    request.headers = {}
    headers = build_upstream_extra_headers(request, "req-1")
    assert headers[HOP_COUNT_HEADER] == "1"


def test_build_upstream_extra_headers_increments_existing_hop_count():
    request = MagicMock()
    request.headers = {HOP_COUNT_HEADER: "2"}
    headers = build_upstream_extra_headers(request, "req-1")
    assert headers[HOP_COUNT_HEADER] == "3"


# ---------------------------------------------------------------------------
# Config-time loop detection tests
# ---------------------------------------------------------------------------


def _route(provider: str, upstream: str | None = None) -> ModelRoute:
    return ModelRoute([ProviderEntry(provider, upstream_model=upstream)])


def test_no_loops():
    models = {
        "argo:gpt-5.6-luna": _route("argo"),
        "claude-opus": _route("anthropic"),
    }
    upstream = {"argo:gpt-5.6-luna": "gpt56luna"}
    assert detect_routing_loops(models, upstream) == []


def test_upstream_collision():
    """upstream_model collides with another model key (non-cyclic chain)."""
    models = {
        "argo:gpt-5.6-luna": _route("argo"),
        "gpt56luna": _route("argo"),
    }
    upstream = {"argo:gpt-5.6-luna": "gpt56luna"}
    warnings = detect_routing_loops(models, upstream)
    assert len(warnings) == 1
    assert "argo:gpt-5.6-luna" in warnings[0]
    assert "gpt56luna" in warnings[0]
    assert "collision" in warnings[0]


def test_transitive_cycle():
    models = {
        "model-a": _route("p1"),
        "model-b": _route("p2"),
        "model-c": _route("p3"),
    }
    upstream = {
        "model-a": "model-b",
        "model-b": "model-c",
        "model-c": "model-a",
    }
    warnings = detect_routing_loops(models, upstream)
    assert len(warnings) == 1
    assert "routing loop" in warnings[0]
    assert "model-a" in warnings[0]


def test_per_entry_upstream_collision():
    """Per-provider upstream_model collides with another model key."""
    models = {
        "display-name": ModelRoute(
            [ProviderEntry("p1", upstream_model="internal-name")]
        ),
        "internal-name": _route("p1"),
    }
    warnings = detect_routing_loops(models, {})
    assert len(warnings) == 1
    assert "display-name" in warnings[0]
    assert "internal-name" in warnings[0]


def test_no_loop_upstream_not_in_models():
    models = {
        "argo:gpt-5.6-luna": _route("argo"),
    }
    upstream = {"argo:gpt-5.6-luna": "gpt56luna"}
    assert detect_routing_loops(models, upstream) == []


def test_max_hops_constant():
    assert MAX_HOPS >= 2


# ---------------------------------------------------------------------------
# Hop-limit middleware tests
# ---------------------------------------------------------------------------


def _make_request(path: str = "/v1/chat/completions", hop_count: int | None = None):
    req = MagicMock()
    req.path = path
    req.headers = {}
    if hop_count is not None:
        req.headers[HOP_COUNT_HEADER] = str(hop_count)
    return req


def test_middleware_allows_normal_request():
    hook = create_hop_limit_hook()
    result = asyncio.run(hook(_make_request()))
    assert result is None


def test_middleware_allows_below_limit():
    hook = create_hop_limit_hook()
    result = asyncio.run(hook(_make_request(hop_count=MAX_HOPS - 1)))
    assert result is None


def test_middleware_rejects_at_limit():
    hook = create_hop_limit_hook()
    result = asyncio.run(hook(_make_request(hop_count=MAX_HOPS)))
    assert result is not None
    assert result.status_code == 508


def test_middleware_rejects_above_limit():
    hook = create_hop_limit_hook()
    result = asyncio.run(hook(_make_request(hop_count=MAX_HOPS + 5)))
    assert result is not None
    assert result.status_code == 508


def test_middleware_skips_admin_paths():
    hook = create_hop_limit_hook()
    result = asyncio.run(
        hook(_make_request(path="/admin/api/config", hop_count=MAX_HOPS + 1))
    )
    assert result is None


def test_middleware_skips_health_paths():
    hook = create_hop_limit_hook()
    result = asyncio.run(hook(_make_request(path="/health", hop_count=MAX_HOPS + 1)))
    assert result is None
