"""Tests for audited external-provider recovery."""

from __future__ import annotations

import asyncio

from llm_rosetta.gateway.ops import OpsContext, OpsProviderValidation


def test_missing_external_shim_disables_only_affected_provider():
    raw = {
        "providers": {
            "argo": {
                "type": "argo--chat_completions",
                "api_key": "k",
                "base_url": "https://argo.test",
            },
            "healthy": {
                "type": "chat_completions",
                "api_key": "k",
                "base_url": "https://healthy.test",
            },
        },
        "models": {"healthy-model": "healthy"},
    }

    result = asyncio.run(OpsProviderValidation(OpsContext(), raw).execute())

    assert [item["provider"] for item in result["disabled"]] == ["argo"]
    assert raw["providers"]["argo"]["enabled"] is False
    assert raw["providers"]["healthy"].get("enabled", True) is True
    assert (
        "required external shim" in raw["providers"]["argo"]["_auto_disabled"]["reason"]
    )


def test_already_disabled_provider_is_not_modified():
    raw = {
        "providers": {
            "argo": {
                "type": "argo--chat_completions",
                "enabled": False,
            }
        },
        "models": {},
    }

    result = asyncio.run(OpsProviderValidation(OpsContext(), raw).execute())

    assert result == {"disabled": []}
    assert "_auto_disabled" not in raw["providers"]["argo"]
