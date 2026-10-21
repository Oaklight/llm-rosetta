"""The admin config payload reports per-endpoint provider failures (#1106).

A provider can serve several model types.  When one endpoint's shim is wrong
the endpoint is skipped and the provider keeps its other endpoints; the reason
travels to the panel through ``_auto_disabled_endpoints`` on the provider
entry, with the same sanitised shape as ``_auto_disabled``.
"""

from __future__ import annotations

from llm_rosetta.gateway.admin.routes.config.payload import _mask_provider_entry


def test_endpoint_diagnostics_keep_only_reason_and_category():
    masked = _mask_provider_entry(
        "p",
        {
            "type": "openai",
            "base_url": "https://a.example/v1",
            "api_key": "sk-secret-value",
            "embedding_format": "openai",
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
