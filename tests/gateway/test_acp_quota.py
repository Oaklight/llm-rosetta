"""The allowance an ACP agent reports during a turn.

Claude's adapter sends the account's rolling windows as ``usage_update``
metadata; there is no key to probe.  These tests pin the translation, the
place the reading is stored, and the payload the Quota card reads.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from llm_rosetta_gateway.quota.service import read_provider_quota
from llm_rosetta_gateway.quota.state import QUOTA_STATE
from llm_rosetta_gateway.transport.acp_chunks import _chunk_of
from llm_rosetta_gateway.transport.acp_quota import (
    ACP_QUOTA_PROBE,
    lane_credential,
    lanes_reading,
    observation_of,
    reading_of,
    windows_of,
)

RATE_LIMIT = {
    "status": "allowed",
    "rateLimitType": "five_hour",
    "resetsAt": 1_800_000_000,
    "unifiedWindows": {
        "five_hour": {"utilization": 0.02},
        "seven_day": {"utilization": 0.31, "resetsAt": 1_800_600_000},
    },
}

UPDATE = {
    "sessionUpdate": "usage_update",
    "used": 42,
    "size": 200000,
    "cost": {"amount": 0.01, "currency": "USD"},
    "_meta": {"_claude/rateLimit": RATE_LIMIT},
}


def _provider(tmp_path):
    return SimpleNamespace(
        config_name="claude",
        keyless=True,
        transport="acp",
        acp_state_dir=str(tmp_path / "data" / "acp" / "claude"),
        quota_identity=None,
        quota_probe=None,
        key_ring=None,
    )


@pytest.fixture(autouse=True)
def _clean_quota_state():
    QUOTA_STATE.clear()
    yield
    QUOTA_STATE.clear()


def test_the_usage_chunk_keeps_the_metadata_and_cost():
    chunk = _chunk_of(UPDATE)

    assert chunk is not None
    assert chunk["type"] == "usage"
    assert chunk["used"] == 42
    assert chunk["meta"]["_claude/rateLimit"]["unifiedWindows"]


def test_a_usage_update_without_metadata_is_only_a_token_count():
    chunk = _chunk_of({"sessionUpdate": "usage_update", "used": 7})

    assert chunk is not None
    assert "meta" not in chunk


def test_the_rate_limit_is_found_under_the_adapter_meta_key():
    assert observation_of({"update": UPDATE}) is RATE_LIMIT


def test_utilization_is_a_fraction_and_resets_are_unix_seconds():
    windows = {window.name: window for window in windows_of(RATE_LIMIT)}

    five_hour = windows["5 hours"]
    assert five_hour.used_percent == pytest.approx(2.0)
    assert five_hour.resets_at == datetime.fromtimestamp(1_800_000_000, tz=timezone.utc)
    assert five_hour.span_seconds == 5 * 3600
    assert five_hour.status == "allowed"
    assert windows["7 days"].used_percent == pytest.approx(31.0)


def test_a_reading_is_keyed_by_the_lane_profile(tmp_path):
    provider = _provider(tmp_path)

    reading = reading_of(UPDATE, provider)

    assert reading is not None
    assert reading.probe == ACP_QUOTA_PROBE
    assert reading.credential == lane_credential(provider)
    assert reading.credential != lane_credential(
        SimpleNamespace(
            config_name="claude-2",
            acp_state_dir=str(tmp_path / "data" / "acp" / "claude-2"),
        )
    )


def test_the_quota_payload_says_how_to_get_a_first_reading(tmp_path):
    provider = _provider(tmp_path)
    config = SimpleNamespace(providers={"claude": provider})

    payload = asyncio.run(read_provider_quota(config, "claude"))

    assert payload["supported"] is True
    assert payload["credentials"] == []
    assert "press Test" in payload["reason"]


def test_the_quota_payload_returns_the_recorded_windows(tmp_path):
    provider = _provider(tmp_path)
    config = SimpleNamespace(providers={"claude": provider})
    reading = reading_of(UPDATE, provider)
    assert reading is not None
    QUOTA_STATE.put_reading(reading)

    payload = asyncio.run(read_provider_quota(config, "claude"))

    assert payload["supported"] is True
    assert payload["status"] == "ok"
    entry = payload["credentials"][0]
    assert entry["credential"] == reading.credential
    assert [window["name"] for window in entry["reading"]["windows"]] == [
        "5 hours",
        "7 days",
    ]


def test_two_entries_with_different_profiles_do_not_share_a_reading(tmp_path):
    first = _provider(tmp_path)
    second = SimpleNamespace(
        config_name="claude-work",
        keyless=True,
        transport="acp",
        acp_state_dir=str(tmp_path / "acp" / "claude-work"),
        quota_identity=None,
        quota_probe=None,
        key_ring=None,
    )
    reading = reading_of(UPDATE, first)
    assert reading is not None
    QUOTA_STATE.put_reading(reading)

    first_reading = lanes_reading(first)
    assert first_reading is not None
    assert first_reading.to_dict()["windows"]
    assert lanes_reading(second) is None
