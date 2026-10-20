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


def test_invalid_provider_config_isolated_from_healthy_provider():
    raw = {
        "providers": {
            "broken": {"type": "chat_completions"},
            "bad_token": {
                "type": "chat_completions",
                "base_url": "https://token.test",
                "token_command": "python",
            },
            "healthy": {
                "type": "chat_completions",
                "base_url": "https://healthy.test",
                "api_key": "${HEALTHY_KEY}",
            },
        }
    }

    result = asyncio.run(OpsProviderValidation(OpsContext(), raw).execute())

    assert {item["provider"] for item in result["disabled"]} == {
        "broken",
        "bad_token",
    }
    assert raw["providers"]["healthy"].get("enabled", True) is True
    assert raw["providers"]["broken"]["_auto_disabled"]["category"] == (
        "missing_base_url"
    )
    assert raw["providers"]["bad_token"]["_auto_disabled"]["category"] == (
        "invalid_provider_info"
    )


def test_malformed_provider_entry_is_disabled_with_structured_diagnostic():
    raw = {
        "providers": {
            "broken": None,
            "healthy": {
                "type": "chat_completions",
                "base_url": "https://healthy.test",
            },
        }
    }

    result = asyncio.run(OpsProviderValidation(OpsContext(), raw).execute())

    assert result["disabled"][0]["category"] == "invalid_provider_config"
    assert raw["providers"]["broken"]["enabled"] is False
    assert raw["providers"]["healthy"].get("enabled", True) is True


def test_legacy_alcf_name_gets_actionable_diagnostic():
    raw = {
        "providers": {
            "alcf": {
                "type": "alcf--sophia",
                "token_command": ["python3", "token.py"],
            }
        }
    }

    result = asyncio.run(OpsProviderValidation(OpsContext(), raw).execute())

    reason = result["disabled"][0]["reason"]
    assert "alcf_sophia--chat_completions" in reason
    assert "shims.presets: ['alcf']" in reason
