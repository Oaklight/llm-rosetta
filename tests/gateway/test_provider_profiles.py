"""One login profile per provider entry.

Two entries that resolve to one directory make a sign-in for one serve the
other.  The gateway refuses that at config build, moves a managed profile when
an entry is renamed, and removes it when the entry is deleted.  An operator's
own ``acp.home`` is never moved or removed.
"""

from __future__ import annotations

import os
import shutil
import time
from types import SimpleNamespace

import pytest

from llm_rosetta_gateway.admin.routes._shared import _config_preflight_error
from llm_rosetta_gateway.admin.routes.config.providers import _relocate_profiles
from llm_rosetta_gateway.config import GatewayConfig
from llm_rosetta_gateway.config.providers import (
    build_provider_info,
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


def test_a_symlinked_profile_is_the_same_profile(tmp_path):
    real = tmp_path / "real-profile"
    real.mkdir()
    link = tmp_path / "linked-profile"
    link.symlink_to(real, target_is_directory=True)

    providers = {
        "claude": _pinfo(str(real)),
        "claude-work": _pinfo(str(link)),
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
        old_info=SimpleNamespace(
            managed_acp_state_dir=str(old), managed_auth_state_dir=None
        ),
        new_info=SimpleNamespace(
            managed_acp_state_dir=str(data_dir / "acp" / "claude-work"),
            managed_auth_state_dir=None,
        ),
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
        old_info=SimpleNamespace(
            managed_acp_state_dir=str(home), managed_auth_state_dir=str(shim_auth)
        ),
        new_info=None,
        remove=True,
    )

    assert not home.exists()
    assert not shim_auth.exists()
    assert sorted(report["removed"]) == sorted([str(home), str(shim_auth)])


def test_a_rename_that_disables_the_entry_keeps_the_profile(tmp_path):
    old = tmp_path / "data" / "acp" / "claude"
    old.mkdir(parents=True)

    report = _relocate_profiles(
        old_info=SimpleNamespace(
            managed_acp_state_dir=str(old), managed_auth_state_dir=None
        ),
        new_info=None,
    )

    assert old.is_dir()
    assert report["moved"] == []
    assert report["removed"] == []
    assert "was kept" in report["warnings"][0]


def test_a_rename_away_from_the_lane_keeps_the_profile(tmp_path):
    old = tmp_path / "data" / "acp" / "claude"
    old.mkdir(parents=True)

    report = _relocate_profiles(
        old_info=SimpleNamespace(
            managed_acp_state_dir=str(old), managed_auth_state_dir=None
        ),
        new_info=SimpleNamespace(
            managed_acp_state_dir=None, managed_auth_state_dir=None
        ),
    )

    assert old.is_dir()
    assert report["moved"] == []
    assert report["removed"] == []
    assert "no longer owns" in report["warnings"][0]


def test_an_operators_own_home_is_never_touched(tmp_path):
    data_dir = tmp_path / "data"
    own = tmp_path / "my-claude-profile"
    own.mkdir()

    pinfo = build_provider_info(
        "acp",
        {
            "base_url": "http://lane.local",
            "transport": "acp",
            "acp": {"home": str(own)},
        },
        config_name="claude",
        data_dir=str(data_dir),
        quiet=True,
    )

    assert pinfo.managed_acp_state_dir is None
    report = _relocate_profiles(old_info=pinfo, new_info=None)
    assert own.is_dir()
    assert report["removed"] == []


def test_the_builder_names_the_gateway_managed_profile(tmp_path):
    pinfo = build_provider_info(
        "acp",
        {"base_url": "http://lane.local", "transport": "acp"},
        config_name="claude",
        data_dir=str(tmp_path),
        quiet=True,
    )

    assert pinfo.managed_acp_state_dir == str(tmp_path / "acp" / "claude")
    assert pinfo.managed_auth_state_dir is None


def test_a_rename_into_an_existing_profile_keeps_the_sign_in(tmp_path):
    data_dir = tmp_path / "data"
    old = data_dir / "acp" / "claude"
    new = data_dir / "acp" / "claude-work"
    (old / "claude").mkdir(parents=True)
    (old / "claude" / ".credentials.json").write_text(
        '{"token": "old"}', encoding="utf-8"
    )
    (new / "claude").mkdir(parents=True)
    (new / "claude" / ".claude.json").write_text('{"stale": true}', encoding="utf-8")

    report = _relocate_profiles(
        old_info=SimpleNamespace(
            managed_acp_state_dir=str(old), managed_auth_state_dir=None
        ),
        new_info=SimpleNamespace(
            managed_acp_state_dir=str(new), managed_auth_state_dir=None
        ),
    )

    # The renamed entry carries its own sign-in; the occupant is not lost.
    assert not old.exists()
    assert (new / "claude" / ".credentials.json").is_file()
    assert not (new / "claude" / ".claude.json").exists()
    assert report["moved"] == [{"from": str(old), "to": str(new)}]
    assert len(report["superseded"]) == 1
    kept_at = report["superseded"][0]["kept_at"]
    assert kept_at.startswith(str(new) + ".superseded-")
    assert os.path.isfile(os.path.join(kept_at, "claude", ".claude.json"))
    assert "already existed" in report["warnings"][0]


def test_a_rename_into_an_empty_profile_replaces_it_without_a_warning(tmp_path):
    data_dir = tmp_path / "data"
    old = data_dir / "acp" / "claude"
    new = data_dir / "acp" / "claude-work"
    (old / "claude").mkdir(parents=True)
    (old / "claude" / ".credentials.json").write_text("{}", encoding="utf-8")
    new.mkdir(parents=True)

    report = _relocate_profiles(
        old_info=SimpleNamespace(
            managed_acp_state_dir=str(old), managed_auth_state_dir=None
        ),
        new_info=SimpleNamespace(
            managed_acp_state_dir=str(new), managed_auth_state_dir=None
        ),
    )

    assert (new / "claude" / ".credentials.json").is_file()
    assert report["superseded"] == []
    assert report["warnings"] == []


def test_a_second_set_aside_in_one_second_gets_a_suffix(tmp_path, monkeypatch):
    stamp = "20260101T000000Z"
    monkeypatch.setattr(time, "strftime", lambda *args, **kwargs: stamp)
    data_dir = tmp_path / "data"
    old = data_dir / "acp" / "claude"
    new = data_dir / "acp" / "claude-work"
    (old / "claude").mkdir(parents=True)
    (old / "claude" / ".credentials.json").write_text("{}", encoding="utf-8")
    (new / "claude").mkdir(parents=True)
    (new / "claude" / ".claude.json").write_text("{}", encoding="utf-8")
    # A sibling from an earlier set-aside in the same second, as a coarse
    # clock would produce: the second one must not clobber it.
    earlier = new.parent / f"{new.name}.superseded-{stamp}"
    earlier.mkdir()
    (earlier / "occupied.txt").write_text("keep me", encoding="utf-8")

    report = _relocate_profiles(
        old_info=SimpleNamespace(
            managed_acp_state_dir=str(old), managed_auth_state_dir=None
        ),
        new_info=SimpleNamespace(
            managed_acp_state_dir=str(new), managed_auth_state_dir=None
        ),
    )

    kept_at = report["superseded"][0]["kept_at"]
    assert kept_at == f"{new}.superseded-{stamp}-1"
    assert (earlier / "occupied.txt").read_text(encoding="utf-8") == "keep me"
    assert os.path.isfile(os.path.join(kept_at, "claude", ".claude.json"))


def test_a_symlinked_empty_target_is_set_aside_not_refused(tmp_path):
    data_dir = tmp_path / "data"
    old = data_dir / "acp" / "claude"
    new = data_dir / "acp" / "claude-work"
    (old / "claude").mkdir(parents=True)
    (old / "claude" / ".credentials.json").write_text("{}", encoding="utf-8")
    empty = tmp_path / "empty-target"
    empty.mkdir()
    new.parent.mkdir(parents=True, exist_ok=True)
    new.symlink_to(empty, target_is_directory=True)

    report = _relocate_profiles(
        old_info=SimpleNamespace(
            managed_acp_state_dir=str(old), managed_auth_state_dir=None
        ),
        new_info=SimpleNamespace(
            managed_acp_state_dir=str(new), managed_auth_state_dir=None
        ),
    )

    assert (new / "claude" / ".credentials.json").is_file()
    assert not new.is_symlink()
    assert len(report["superseded"]) == 1
    assert os.path.islink(report["superseded"][0]["kept_at"])


def test_a_failed_move_restores_the_occupant(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    old = data_dir / "acp" / "claude"
    new = data_dir / "acp" / "claude-work"
    (old / "claude").mkdir(parents=True)
    (old / "claude" / ".credentials.json").write_text("{}", encoding="utf-8")
    (new / "claude").mkdir(parents=True)
    (new / "claude" / ".claude.json").write_text("{}", encoding="utf-8")

    real_move = shutil.move
    failed = False

    def flaky_move(src, dst):
        nonlocal failed
        if not failed and os.path.abspath(dst) == os.path.abspath(str(new)):
            failed = True
            raise OSError("disk full")
        return real_move(src, dst)

    monkeypatch.setattr(shutil, "move", flaky_move)
    report = _relocate_profiles(
        old_info=SimpleNamespace(
            managed_acp_state_dir=str(old), managed_auth_state_dir=None
        ),
        new_info=SimpleNamespace(
            managed_acp_state_dir=str(new), managed_auth_state_dir=None
        ),
    )

    # The occupant is back where it was, the old login is untouched, and no
    # half-finished sibling is left behind.
    assert (new / "claude" / ".claude.json").is_file()
    assert (old / "claude" / ".credentials.json").is_file()
    assert report["moved"] == []
    assert report["superseded"] == []
    assert not list(new.parent.glob(f"{new.name}.superseded-*"))
    assert report["warnings"]


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


def test_the_loader_itself_refuses_a_shared_home(tmp_path):
    """The build path refuses too, not only the admin preflight."""
    shared = str(tmp_path / "shared")
    raw = {
        "providers": {
            "one": {
                "type": "acp",
                "base_url": "http://one.local",
                "api_key": "k",
                "transport": "acp",
                "acp": {"home": shared},
            },
            "two": {
                "type": "acp",
                "base_url": "http://two.local",
                "api_key": "k",
                "transport": "acp",
                "acp": {"home": shared},
            },
        },
        "models": {},
    }

    with pytest.raises(ValueError, match="share the ACP profile"):
        GatewayConfig(raw, config_path=str(tmp_path / "config.json"))
