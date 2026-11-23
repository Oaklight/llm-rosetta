"""One login profile per provider entry.

Two entries that resolve to one directory make a sign-in for one serve the
other.  The gateway refuses that at config build, moves a managed profile when
an entry is renamed, and removes it when the entry is deleted.  An operator's
own ``acp.home`` is never moved or removed.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from llm_rosetta_gateway.admin.routes._shared import _config_preflight_error
from llm_rosetta_gateway.admin.routes.config.providers import (
    _owned_profile_paths,
    _relocate_profiles,
)
from llm_rosetta_gateway.config.providers import (
    lane_profile_is_managed,
    managed_acp_state_dir,
    validate_provider_profiles,
)


def _pinfo(home: str, *, transport: str = "acp", auth_dir: str | None = None):
    return SimpleNamespace(
        transport=transport,
        acp_state_dir=home,
        auth_context=SimpleNamespace(state_dir=auth_dir) if auth_dir else None,
        auth_exchange=object() if auth_dir else None,
    )


def test_two_entries_may_not_share_an_acp_home():
    providers = {
        "claude": _pinfo("/srv/profiles/one"),
        "claude-work": _pinfo("/srv/profiles/one"),
    }

    with pytest.raises(ValueError, match="share the ACP profile"):
        validate_provider_profiles(providers)


def test_two_entries_may_not_share_a_shim_auth_directory():
    providers = {
        "chatgpt": _pinfo("/srv/acp/a", transport="http", auth_dir="/srv/auth/one"),
        "chatgpt-work": _pinfo(
            "/srv/acp/b", transport="http", auth_dir="/srv/auth/one"
        ),
    }

    with pytest.raises(ValueError, match="credential record directory"):
        validate_provider_profiles(providers)


def test_distinct_profiles_are_accepted():
    validate_provider_profiles(
        {
            "claude": _pinfo("/srv/acp/claude"),
            "claude-work": _pinfo("/srv/acp/claude-work"),
        }
    )


def test_an_http_lane_does_not_own_the_acp_home_it_carries():
    validate_provider_profiles(
        {
            "a": _pinfo("/srv/acp/shared", transport="http"),
            "b": _pinfo("/srv/acp/shared", transport="http"),
        }
    )


@pytest.mark.parametrize(
    "cfg, managed",
    [
        ({}, True),
        ({"acp": {}}, True),
        ({"acp": {"home": "/srv/profile"}}, False),
        ({"acp_argv": ["agy", "acp", "--gemini-dir", "/srv/profile"]}, False),
    ],
)
def test_managed_means_the_gateway_chose_the_path(cfg, managed):
    assert lane_profile_is_managed("claude", cfg) is managed


def test_no_data_dir_means_no_managed_profile(tmp_path):
    assert managed_acp_state_dir("claude", {}, None) is None
    assert (
        managed_acp_state_dir("claude", {"acp": {"home": str(tmp_path)}}, str(tmp_path))
        is None
    )


def test_a_rename_moves_the_managed_profile(tmp_path):
    data_dir = tmp_path / "data"
    old = data_dir / "acp" / "claude"
    old.mkdir(parents=True)
    (old / "claude" / ".credentials.json").parent.mkdir()
    (old / "claude" / ".credentials.json").write_text("{}", encoding="utf-8")

    report = _relocate_profiles(
        old_name="claude",
        old_entry={},
        old_transport="acp",
        new_name="claude-work",
        new_entry={},
        new_transport="acp",
        data_dir=str(data_dir),
    )

    new = data_dir / "acp" / "claude-work"
    assert not old.exists()
    assert (new / "claude" / ".credentials.json").is_file()
    assert report["moved"] == [{"from": str(old), "to": str(new)}]
    assert report["warnings"] == []


def test_a_delete_removes_the_managed_profile(tmp_path):
    data_dir = tmp_path / "data"
    home = data_dir / "acp" / "claude"
    home.mkdir(parents=True)
    shim_auth = data_dir / "shim-auth" / "claude"
    shim_auth.mkdir(parents=True)

    report = _relocate_profiles(
        old_name="claude",
        old_entry={},
        old_transport="acp",
        new_name=None,
        new_entry=None,
        new_transport="http",
        data_dir=str(data_dir),
    )

    assert not home.exists()
    assert not shim_auth.exists()
    assert sorted(report["removed"]) == sorted([str(home), str(shim_auth)])


def test_an_operators_own_home_is_never_touched(tmp_path):
    data_dir = tmp_path / "data"
    own = tmp_path / "my-claude-profile"
    own.mkdir()
    entry = {"acp": {"home": str(own)}}

    paths = _owned_profile_paths("claude", entry, str(data_dir), "acp")

    assert "acp" not in paths
    report = _relocate_profiles(
        old_name="claude",
        old_entry=entry,
        old_transport="acp",
        new_name=None,
        new_entry=None,
        new_transport="http",
        data_dir=str(data_dir),
    )
    assert own.is_dir()
    assert report["removed"] == []


def test_a_rename_that_finds_the_new_profile_already_present_warns(tmp_path):
    data_dir = tmp_path / "data"
    old = data_dir / "acp" / "claude"
    new = data_dir / "acp" / "claude-work"
    old.mkdir(parents=True)
    new.mkdir(parents=True)

    report = _relocate_profiles(
        old_name="claude",
        old_entry={},
        old_transport="acp",
        new_name="claude-work",
        new_entry={},
        new_transport="acp",
        data_dir=str(data_dir),
    )

    assert old.is_dir()
    assert report["moved"] == []
    assert "already exists" in report["warnings"][0]


def test_the_admin_preflight_refuses_a_shared_home(tmp_path):
    data = {
        "providers": {
            "one": {
                "type": "acp",
                "base_url": "http://one.local",
                "api_key": "k",
                "transport": "acp",
                "acp": {"home": str(tmp_path / "shared")},
            },
            "two": {
                "type": "acp",
                "base_url": "http://two.local",
                "api_key": "k",
                "transport": "acp",
                "acp": {"home": str(tmp_path / "shared")},
            },
        }
    }

    error = _config_preflight_error(data, str(tmp_path / "data"))

    assert error is not None
    assert "share the ACP profile" in error
    assert not os.path.exists(tmp_path / "shared")
