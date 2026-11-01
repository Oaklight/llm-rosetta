"""The admin config payload reports per-endpoint provider failures (#1106).

A provider can serve several model types.  When one endpoint's shim is wrong
the endpoint is skipped and the provider keeps its other endpoints; the reason
travels to the panel through ``_auto_disabled_endpoints`` on the provider
entry, with the same sanitised shape as ``_auto_disabled``.
"""

from __future__ import annotations

from llm_rosetta_gateway.admin.routes.config.payload import (
    _mask_provider_entry,
    _with_live_endpoint_markers,
)


def test_endpoint_diagnostics_keep_only_reason_and_category():
    masked = _mask_provider_entry(
        "p",
        {
            "type": "openai",
            "base_url": "https://a.example/v1",
            "api_key": "sk-secret-value",
            "embeddings_format": "openai",
            "_auto_disabled_endpoints": {
                "rerank": {
                    "reason": "config: rerank_shim names unknown shim 'nope'",
                    "category": "unknown_shim",
                    "api_key": "should-not-leak",
                }
            },
        },
    )

    assert masked["_auto_disabled_endpoints"]["rerank"] == {
        "reason": "config: rerank_shim names unknown shim 'nope'",
        "category": "unknown_shim",
    }
    assert masked["api_key"] != "sk-secret-value"


def test_a_provider_without_endpoint_failures_has_no_marker():
    masked = _mask_provider_entry(
        "p", {"type": "openai", "base_url": "https://a.example/v1"}
    )

    assert "_auto_disabled_endpoints" not in masked


def test_a_legacy_endpoint_flag_is_spelled_the_way_the_panel_reads_it():
    """The parser honours `embedding_enabled`; the panel reads the canonical key.

    Without the normalisation the card shows an endpoint as enabled while the
    gateway has already stopped routing to it.
    """
    masked = _mask_provider_entry(
        "p",
        {
            "type": "openai",
            "base_url": "https://a.example/v1",
            "embedding_enabled": False,
        },
    )

    assert masked["embeddings_enabled"] is False


def test_the_canonical_flag_wins_over_the_legacy_one():
    masked = _mask_provider_entry(
        "p",
        {
            "type": "openai",
            "base_url": "https://a.example/v1",
            "embeddings_enabled": True,
            "embedding_enabled": False,
        },
    )

    assert masked["embeddings_enabled"] is True


def test_a_non_mapping_marker_is_dropped():
    masked = _mask_provider_entry("p", {"_auto_disabled_endpoints": "not-a-mapping"})

    assert "_auto_disabled_endpoints" not in masked


def test_the_whole_provider_marker_still_works():
    masked = _mask_provider_entry(
        "p",
        {
            "_auto_disabled": {
                "reason": "no base_url",
                "category": "missing_base_url",
                "shim": "argo--chat_completions",
                "token": "should-not-leak",
            }
        },
    )

    assert masked["_auto_disabled"] == {
        "reason": "no base_url",
        "category": "missing_base_url",
        "shim": "argo--chat_completions",
    }


class _LiveConfig:
    """The bit of ``GatewayConfig`` the payload overlay reads."""

    def __init__(self, providers):
        self.raw_providers = providers


def test_the_live_parse_marker_reaches_the_payload():
    """The marker is written on the running parse, not on disk.

    ``get_config`` reads the document from disk, so without the overlay a
    degraded endpoint would never reach the panel (#1106).
    """
    masked = {"type": "openai--chat_completions"}
    live = _LiveConfig(
        {
            "p": {
                "_auto_disabled_endpoints": {
                    "embedding": {
                        "reason": "config: embeddings_shim names unknown shim 'nope'",
                        "category": "unknown_shim",
                        "api_key": "should-not-leak",
                    }
                }
            }
        }
    )

    _with_live_endpoint_markers(masked, "p", live)

    assert masked["_auto_disabled_endpoints"] == {
        "embedding": {
            "reason": "config: embeddings_shim names unknown shim 'nope'",
            "category": "unknown_shim",
        }
    }


def test_a_healthy_provider_gets_no_live_marker():
    masked = {"type": "openai--chat_completions"}
    _with_live_endpoint_markers(
        masked, "p", _LiveConfig({"p": {"base_url": "https://x"}})
    )
    assert "_auto_disabled_endpoints" not in masked


def test_the_overlay_tolerates_a_missing_or_odd_live_config():
    masked = {"type": "openai--chat_completions"}
    _with_live_endpoint_markers(masked, "p", None)
    _with_live_endpoint_markers(masked, "p", object())
    _with_live_endpoint_markers(masked, "p", _LiveConfig({"p": "not-a-mapping"}))
    _with_live_endpoint_markers(
        masked, "p", _LiveConfig({"p": {"_auto_disabled_endpoints": "nope"}})
    )
    assert "_auto_disabled_endpoints" not in masked
