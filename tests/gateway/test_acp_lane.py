"""``resolve_acp_lane``: one provider entry in, its resolved lane out (#1830 A8)."""

from __future__ import annotations

import dataclasses

import pytest

from llm_rosetta_gateway.transport.acp.lane import ResolvedLane, resolve_acp_lane


def _entry(**more):
    return {"base_url": "http://lane.local", "transport": "acp", **more}


def test_an_entry_resolves_to_a_frozen_record(tmp_path):
    lane = resolve_acp_lane(_entry(), "claude", None, str(tmp_path))

    assert isinstance(lane, ResolvedLane)
    assert (lane.transport, lane.transport_owner) == ("acp", None)
    assert lane.acp_state_dir == str(tmp_path / "acp" / "claude")
    assert lane.managed_acp_state_dir == lane.acp_state_dir
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(lane, "transport", "http")


def test_the_limits_are_the_providerinfo_keywords(tmp_path):
    entry = _entry(acp={"processes": 2, "max_sessions": 5, "session_ttl": 30})

    lane = resolve_acp_lane(entry, "claude", None, str(tmp_path))

    assert lane.limits == {
        "acp_processes": 2,
        "acp_max_sessions": 5,
        "acp_session_ttl": 30.0,
    }
    assert resolve_acp_lane(_entry(), "claude", None, str(tmp_path)).limits == {}


def test_an_operator_home_is_not_the_gateways_to_move(tmp_path):
    entry = _entry(acp={"home": str(tmp_path / "mine")})

    lane = resolve_acp_lane(entry, "claude", None, str(tmp_path))

    assert lane.acp_state_dir == str(tmp_path / "mine")
    assert lane.managed_acp_state_dir is None


def test_an_http_lane_owns_no_agent_profile(tmp_path):
    lane = resolve_acp_lane(
        {"base_url": "http://x", "acp": {"home": "profiles/x"}},
        "claude",
        None,
        str(tmp_path),
    )

    assert lane.transport == "http"
    assert lane.managed_acp_state_dir is None
    assert lane.acp_env is None


def test_a_malformed_setting_names_the_provider(tmp_path):
    with pytest.raises(ValueError, match="Provider 'claude': acp.harden must be"):
        resolve_acp_lane(_entry(acp={"harden": "yes"}), "claude", None, str(tmp_path))
