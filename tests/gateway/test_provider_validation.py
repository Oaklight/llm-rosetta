"""Tests for audited external-provider recovery."""

from __future__ import annotations

import asyncio
import json

from llm_rosetta_gateway.ops import OpsContext, OpsProviderValidation


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

    assert result == {"disabled": [], "recovered": []}
    assert "_auto_disabled" not in raw["providers"]["argo"]


def test_an_auto_disabled_provider_recovers_when_its_fault_is_gone():
    """The marker is the gateway's state, not the operator's switch."""
    raw = {
        "providers": {
            "argo": {
                "type": "chat_completions",
                "api_key": "k",
                "base_url": "https://argo.test",
                "enabled": False,
                "_auto_disabled": {"reason": "required external shim missing"},
            }
        },
        "models": {},
    }

    result = asyncio.run(OpsProviderValidation(OpsContext(), raw).execute())

    assert result["recovered"] == ["argo"]
    assert result["disabled"] == []
    assert raw["providers"]["argo"]["enabled"] is True
    assert "_auto_disabled" not in raw["providers"]["argo"]


def test_an_auto_disabled_provider_that_still_fails_stays_disabled():
    raw = {
        "providers": {
            "argo": {
                "type": "argo--chat_completions",
                "api_key": "k",
                "base_url": "https://argo.test",
                "enabled": False,
                "_auto_disabled": {"reason": "required external shim missing"},
            }
        },
        "models": {},
    }

    result = asyncio.run(OpsProviderValidation(OpsContext(), raw).execute())

    assert result["recovered"] == []
    assert [item["provider"] for item in result["disabled"]] == ["argo"]
    assert raw["providers"]["argo"]["enabled"] is False
    assert raw["providers"]["argo"]["_auto_disabled"]["reason"]


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
    broken_marker = raw["providers"]["broken"].get("_auto_disabled")
    bad_token_marker = raw["providers"]["bad_token"].get("_auto_disabled")
    assert isinstance(broken_marker, dict)
    assert isinstance(bad_token_marker, dict)
    assert broken_marker["category"] == "missing_base_url"
    assert bad_token_marker["category"] == "invalid_provider_info"


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
    broken = raw["providers"]["broken"]
    assert isinstance(broken, dict)
    assert broken["enabled"] is False
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


def test_a_recovery_is_written_to_the_document(tmp_path):
    """The in-memory fix is not enough: the next load reads the file."""
    from llm_rosetta_gateway.config import JsoncConfigIO

    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "providers": {
                    "argo": {
                        "type": "chat_completions",
                        "api_key": "k",
                        "base_url": "https://argo.test",
                        "enabled": False,
                        "_auto_disabled": {
                            "reason": "required external shim missing",
                            "category": "missing_external_shim",
                        },
                    },
                    "hand": {"type": "argo--chat_completions", "enabled": False},
                },
                "models": {},
            }
        ),
        encoding="utf-8",
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    ctx = OpsContext(config_io=JsoncConfigIO(), config_path=str(path))

    result = asyncio.run(OpsProviderValidation(ctx, raw, persist=True).execute())

    assert result["recovered"] == ["argo"]
    disk = json.loads(path.read_text(encoding="utf-8"))
    assert disk["providers"]["argo"]["enabled"] is True
    assert "_auto_disabled" not in disk["providers"]["argo"]
    # The hand-disabled entry is not this operation's to change.
    assert disk["providers"]["hand"] == {
        "type": "argo--chat_completions",
        "enabled": False,
    }


def test_a_still_broken_lane_is_written_once(tmp_path):
    """A fault that has not changed must not rewrite the document on every load."""
    from llm_rosetta_gateway.config import JsoncConfigIO

    class _CountingIO(JsoncConfigIO):
        def __init__(self) -> None:
            self.saves = 0

        def save(self, path, data):
            self.saves += 1
            return super().save(path, data)

    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {"providers": {"broken": {"type": "chat_completions"}}, "models": {}}
        ),
        encoding="utf-8",
    )
    io = _CountingIO()
    ctx = OpsContext(config_io=io, config_path=str(path))

    first_raw = json.loads(path.read_text(encoding="utf-8"))
    first = asyncio.run(OpsProviderValidation(ctx, first_raw, persist=True).execute())
    assert [item["provider"] for item in first["disabled"]] == ["broken"]
    assert io.saves == 1

    second_raw = json.loads(path.read_text(encoding="utf-8"))
    second = asyncio.run(OpsProviderValidation(ctx, second_raw, persist=True).execute())
    assert [item["provider"] for item in second["disabled"]] == ["broken"]
    assert io.saves == 1, "an unchanged fault must not rewrite the document"


def test_a_recovery_only_run_says_so_in_its_message():
    """The audit line must not read "passed" while the document changed."""
    op = OpsProviderValidation(OpsContext(), {"providers": {}})

    message = op._message({"disabled": [], "recovered": ["argo"]})

    assert "Recovered automatically disabled providers: argo" in message


def test_a_startup_run_leaves_an_audit_row():
    """The load-time run happens before the log exists; its row is written later."""
    from llm_rosetta_gateway.ops import startup_validation_entry

    entry = startup_validation_entry(
        {
            "disabled": [{"provider": "argo", "reason": "r", "category": "c"}],
            "recovered": [],
        }
    )

    assert entry is not None
    assert entry.event_type == "provider_validation"
    assert "Automatically disabled invalid providers: argo" in entry.message
    assert entry.details is not None
    assert entry.details["disabled"][0]["provider"] == "argo"


def test_a_quiet_startup_writes_no_row():
    from llm_rosetta_gateway.ops import startup_validation_entry

    assert startup_validation_entry(None) is None
    assert startup_validation_entry({"disabled": [], "recovered": []}) is None


def test_a_recovery_only_startup_writes_a_row():
    """Re-enabling a provider is a decision an operator looks for in the log.

    The guard accepts a run that recovered something and disabled nothing;
    dropping that half leaves the recovery out of the trail silently.
    """
    from llm_rosetta_gateway.ops import startup_validation_entry

    entry = startup_validation_entry({"disabled": [], "recovered": ["argo"]})

    assert entry is not None
    assert "Recovered automatically disabled providers: argo" in entry.message


class _StubLog:
    def __init__(self):
        self.entries = []

    async def add(self, entry):
        self.entries.append(entry)


def _startup_app(result):
    from types import SimpleNamespace

    return SimpleNamespace(
        ops_log=_StubLog(),
        gateway_config=SimpleNamespace(
            providers={},
            models={},
            provider_validation_result=result,
        ),
    )


def test_the_startup_rows_land_in_validations_order():
    """The load-time decision is written before the boot line, not after."""

    from llm_rosetta_gateway.app.factory import _record_startup_entries

    app = _startup_app(
        {
            "disabled": [{"provider": "argo", "reason": "r", "category": "c"}],
            "recovered": [],
        }
    )
    asyncio.run(_record_startup_entries(app, "127.0.0.1", 8080))

    assert [entry.event_type for entry in app.ops_log.entries] == [
        "provider_validation",
        "startup",
    ]
    assert "argo" in app.ops_log.entries[0].message


def test_a_quiet_load_writes_only_the_startup_row():

    from llm_rosetta_gateway.app.factory import _record_startup_entries

    app = _startup_app(None)
    asyncio.run(_record_startup_entries(app, "127.0.0.1", 8080))

    assert [entry.event_type for entry in app.ops_log.entries] == ["startup"]


def test_an_app_without_a_log_is_left_alone():
    from types import SimpleNamespace

    from llm_rosetta_gateway.app.factory import _record_startup_entries

    asyncio.run(_record_startup_entries(SimpleNamespace(), "127.0.0.1", 8080))


def test_the_load_hands_the_validation_result_to_the_config(tmp_path):
    """The CLI's run cannot record itself; the config carries the result.

    This is the link between the load-time decision and the row the factory
    writes: without the assignment the field stays ``None`` and the disable is
    silently unrecorded.
    """
    import argparse

    from llm_rosetta_gateway.cli import _load_startup_config

    path = tmp_path / "gateway.jsonc"
    path.write_text(
        json.dumps(
            {
                "providers": {
                    "broken": {"type": "chat_completions", "base_url": ""},
                },
                "models": {},
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(proxy=None, data_dir=None, tls_cert=None, tls_key=None)

    config, _ = _load_startup_config(str(path), args)

    result = config.provider_validation_result
    assert result is not None
    assert result["disabled"][0]["provider"] == "broken"


def test_a_config_that_never_ran_validation_reads_as_none():
    """The factory reads the attribute unconditionally, so it always exists."""
    from llm_rosetta_gateway.config import GatewayConfig

    config = GatewayConfig({"providers": {}, "models": {}})

    assert config.provider_validation_result is None
