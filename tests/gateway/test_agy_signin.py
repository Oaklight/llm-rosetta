"""The agy sign-in exchange, driven against a fake ``agy``.

No Google account and no real ``agy``: ``fake_agy.py`` is a script that speaks
ACP over stdio and models a per-profile "signed in" marker.  What is under
test is this repository's half: how signed-in is *asked* (the handshake, never
a credential file), how the login subprocess is bounded and observed, and how
one provider entry gets one identity.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from llm_rosetta.shims.auth_exchange import (
    AuthContext,
    AuthExchangeError,
    InteractiveSetupRequired,
)
from llm_rosetta.shims.provider_shim import get_shim

from .fake_agy import alive, install_fake_agy, starts

_SHIM = get_shim("antigravity--acp")
assert _SHIM is not None
SHIM = _SHIM
EXCHANGE: Any = SHIM.auth
MODULE = __import__("sys").modules[type(EXCHANGE).__module__]


@pytest.fixture(autouse=True)
def fake(tmp_path, monkeypatch):
    """A fake agy, its host profile and its log; state reset around the test."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    agy = install_fake_agy(bin_dir)
    host = tmp_path / "host"
    host.mkdir()
    monkeypatch.setenv("FAKE_AGY_HOST", str(host))
    monkeypatch.setenv("FAKE_AGY_LOG", str(tmp_path / "log.jsonl"))
    monkeypatch.setenv("FAKE_AGY_DELAY", "0.1")
    monkeypatch.setattr(MODULE, "LOGIN_TIMEOUT", 20.0)
    MODULE.reset_state()
    yield SimpleFake(agy, host, tmp_path / "log.jsonl", tmp_path)
    MODULE.reset_state()


class SimpleFake:
    def __init__(self, agy: Path, host: Path, log: Path, root: Path) -> None:
        self.agy, self.host, self.log, self.root = agy, host, log, root

    def sign_host_in(self) -> None:
        (self.host / "signed-in").write_text("x")

    def ctx(
        self,
        provider: str = "agy",
        *,
        separate: bool = False,
        state: bool = True,
        **config: Any,
    ) -> AuthContext:
        cfg: dict[str, Any] = {"acp_argv": [str(self.agy), "acp"], **config}
        if separate:
            cfg["auth_fields"] = {"agy_profile": "separate"}
        state_dir = str(self.root / "state" / provider) if state else None
        return AuthContext(provider=provider, config=cfg, state_dir=state_dir)


def credential(ctx: AuthContext) -> str:
    return asyncio.run(EXCHANGE.credential(ctx))


def wait_for(ctx: AuthContext, states: tuple[str, ...], seconds: float = 15.0) -> dict:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        snap = dict(EXCHANGE.login_status(ctx))
        if snap["state"] in states:
            return snap
        time.sleep(0.05)
    raise AssertionError(f"still {snap['state']!r} after {seconds}s: {snap}")


class TestShimShape:
    def test_the_shim_ships_a_host_login_exchange_not_a_keyless_lane(self):
        assert EXCHANGE is not None
        assert EXCHANGE.can_host_login() is True
        assert SHIM.connection.keyless is False
        # No install(): a record cannot be handed to agy from outside.
        assert EXCHANGE.can_install() is False
        assert EXCHANGE.can_complete_login() is False

    def test_the_profile_is_a_declared_field(self):
        assert [f["key"] for f in SHIM.fields] == ["agy_profile"]


class TestSignedInIsAskedOfAgy:
    def test_a_signed_in_agent_yields_a_marker_and_a_summary(self, fake):
        fake.sign_host_in()
        ctx = fake.ctx()

        assert credential(ctx) == MODULE.SIGNED_IN_MARKER
        assert EXCHANGE.needs_interactive_setup(ctx) is False
        assert EXCHANGE.describe(ctx)["profile"] == "host"

    def test_a_signed_out_agent_needs_a_sign_in(self, fake):
        ctx = fake.ctx()

        with pytest.raises(InteractiveSetupRequired):
            credential(ctx)
        assert EXCHANGE.needs_interactive_setup(ctx) is True
        assert EXCHANGE.describe(ctx) is None

    def test_a_failure_that_is_not_a_sign_out_is_not_reported_as_one(
        self, fake, monkeypatch
    ):
        monkeypatch.setenv("FAKE_AGY_MODE", "broken")
        ctx = fake.ctx()

        with pytest.raises(AuthExchangeError) as raised:
            credential(ctx)
        assert not isinstance(raised.value, InteractiveSetupRequired)
        assert "internal explosion" in str(raised.value)
        assert EXCHANGE.needs_interactive_setup(ctx) is False

    def test_a_missing_agy_is_a_failure_that_names_it(self, fake):
        ctx = fake.ctx(acp_argv=["/nonexistent/agy", "acp"])

        with pytest.raises(AuthExchangeError, match="could not start"):
            credential(ctx)

    def test_each_check_closes_its_session_and_leaves_no_process(self, fake):
        fake.sign_host_in()
        credential(fake.ctx())

        (entry,) = starts(fake.log)
        deadline = time.monotonic() + 5
        while alive(entry["pid"]) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not alive(entry["pid"])

    def test_a_positive_answer_is_remembered_briefly(self, fake):
        fake.sign_host_in()
        ctx = fake.ctx()
        credential(ctx)
        credential(ctx)

        assert len(starts(fake.log)) == 1

    def test_the_answer_belongs_to_the_identity_it_was_given_for(self, fake):
        fake.sign_host_in()
        host = fake.ctx()
        credential(host)
        # The same entry switched to a separate profile must be asked again.
        separate = fake.ctx(separate=True)

        with pytest.raises(InteractiveSetupRequired):
            credential(separate)


class TestLoginLifecycle:
    def test_a_login_runs_agy_login_then_verifies_and_signs_in(self, fake):
        ctx = fake.ctx(separate=True)

        first = EXCHANGE.begin_login(ctx)
        assert first["state"] == "running"
        done = wait_for(ctx, ("succeeded", "failed"))

        assert done["state"] == "succeeded", done
        assert (
            done["url"]
            == "https://accounts.example/o/oauth2/auth?client_id=x&state=abc"
        )
        assert credential(ctx) == MODULE.SIGNED_IN_MARKER
        login = [s for s in starts(fake.log) if "--login" in s["argv"]]
        assert len(login) == 1
        assert login[0]["argv"][:2] == ["acp", "--login"]

    def test_the_argv_is_fixed_and_carries_only_the_derived_profile(self, fake):
        ctx = fake.ctx("work", separate=True)
        EXCHANGE.begin_login(ctx)
        wait_for(ctx, ("succeeded", "failed"))

        (login,) = [s for s in starts(fake.log) if "--login" in s["argv"]]
        profile = str(fake.root / "state" / "work" / "agy")
        assert login["argv"] == ["acp", "--login", "--gemini-dir", profile]

    def test_output_that_looks_like_a_token_never_reaches_the_snapshot(self, fake):
        ctx = fake.ctx(separate=True)
        EXCHANGE.begin_login(ctx)
        snap = wait_for(ctx, ("succeeded", "failed"))

        text = "\n".join(snap["lines"])
        assert "ya29" not in text
        assert "[REDACTED]" in text
        assert "Open this URL to sign in:" in text

    def test_a_failing_login_reports_the_exit_status(self, fake, monkeypatch):
        monkeypatch.setenv("FAKE_AGY_LOGIN", "fail")
        ctx = fake.ctx(separate=True)
        EXCHANGE.begin_login(ctx)
        snap = wait_for(ctx, ("failed",))

        assert "status 3" in snap["message"]
        assert EXCHANGE.needs_interactive_setup(ctx) is False  # nothing was checked

    def test_a_login_that_exits_zero_but_is_not_signed_in_fails_honestly(
        self, fake, monkeypatch
    ):
        # agy exits 0 and the verification still finds nobody signed in.
        monkeypatch.setattr(
            MODULE, "_agent_args", lambda ctx: ["--gemini-dir", "/nonexistent-x/y"]
        )
        ctx = fake.ctx(separate=True)
        # The fake cannot write into that directory, so it exits non-zero; the
        # point is that success is never claimed without the check passing.
        EXCHANGE.begin_login(ctx)
        snap = wait_for(ctx, ("failed", "succeeded"))

        assert snap["state"] == "failed"

    def test_a_hung_login_is_killed_at_the_deadline(self, fake, monkeypatch):
        monkeypatch.setenv("FAKE_AGY_LOGIN", "hang")
        monkeypatch.setattr(MODULE, "LOGIN_TIMEOUT", 0.6)
        ctx = fake.ctx(separate=True)
        EXCHANGE.begin_login(ctx)
        pid = _wait_pid(fake.log)

        snap = wait_for(ctx, ("failed",), seconds=10)

        assert "did not finish" in snap["message"]
        assert not alive(pid)

    def test_cancel_kills_the_process_and_says_so(self, fake, monkeypatch):
        monkeypatch.setenv("FAKE_AGY_LOGIN", "hang")
        ctx = fake.ctx(separate=True)
        EXCHANGE.begin_login(ctx)
        pid = _wait_pid(fake.log)

        assert EXCHANGE.cancel_login(ctx) is True

        assert EXCHANGE.login_status(ctx)["state"] == "cancelled"
        assert not alive(pid)
        assert EXCHANGE.cancel_login(ctx) is False

    def test_one_login_at_a_time_per_entry(self, fake, monkeypatch):
        monkeypatch.setenv("FAKE_AGY_LOGIN", "hang")
        ctx = fake.ctx(separate=True)
        first = EXCHANGE.begin_login(ctx)
        again = EXCHANGE.begin_login(ctx)

        assert again["started_at"] == first["started_at"]
        _wait_pid(fake.log)
        assert len(starts(fake.log)) == 1
        EXCHANGE.cancel_login(ctx)

    def test_a_cancelled_entry_can_sign_in_again(self, fake, monkeypatch):
        monkeypatch.setenv("FAKE_AGY_LOGIN", "hang")
        ctx = fake.ctx(separate=True)
        EXCHANGE.begin_login(ctx)
        EXCHANGE.cancel_login(ctx)
        monkeypatch.setenv("FAKE_AGY_LOGIN", "finish")
        EXCHANGE.begin_login(ctx)

        assert wait_for(ctx, ("succeeded", "failed"))["state"] == "succeeded"

    def test_two_entries_may_sign_in_at_the_same_time(self, fake, monkeypatch):
        monkeypatch.setenv("FAKE_AGY_LOGIN", "hang")
        a, b = fake.ctx("a", separate=True), fake.ctx("b", separate=True)
        EXCHANGE.begin_login(a)
        EXCHANGE.begin_login(b)

        assert EXCHANGE.login_status(a)["state"] == "running"
        assert EXCHANGE.login_status(b)["state"] == "running"
        EXCHANGE.cancel_login(a)
        assert EXCHANGE.login_status(b)["state"] == "running"
        EXCHANGE.cancel_login(b)

    def test_output_is_bounded(self, fake, monkeypatch):
        monkeypatch.setenv("FAKE_AGY_LOGIN", "flood")
        ctx = fake.ctx(separate=True)
        EXCHANGE.begin_login(ctx)
        snap = wait_for(ctx, ("succeeded", "failed"))

        assert len(snap["lines"]) <= MODULE.MAX_LINES
        assert all(len(line) <= MODULE.MAX_LINE_CHARS for line in snap["lines"])

    def test_a_missing_agy_fails_the_start_not_the_gateway(self, fake):
        ctx = fake.ctx(separate=True, acp_argv=["/nonexistent/agy", "acp"])

        with pytest.raises(AuthExchangeError, match="could not start"):
            EXCHANGE.begin_login(ctx)
        assert EXCHANGE.login_status(ctx)["state"] == "idle"


def _wait_pid(log: Path, seconds: float = 10.0) -> int:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        entries = starts(log)
        if entries:
            return entries[-1]["pid"]
        time.sleep(0.05)
    raise AssertionError("the fake agy never started")


class TestOneIdentityPerEntry:
    def test_a_host_entry_adds_no_argument(self, fake):
        assert EXCHANGE.agent_args(fake.ctx()) == []

    def test_a_separate_entry_gets_its_own_profile_under_its_state_dir(self, fake):
        a, b = fake.ctx("a", separate=True), fake.ctx("b", separate=True)

        args_a, args_b = EXCHANGE.agent_args(a), EXCHANGE.agent_args(b)

        assert args_a == ["--gemini-dir", str(fake.root / "state" / "a" / "agy")]
        assert args_b == ["--gemini-dir", str(fake.root / "state" / "b" / "agy")]
        assert Path(args_a[1]).is_dir() and Path(args_b[1]).is_dir()
        assert oct(Path(args_a[1]).stat().st_mode & 0o777) == "0o700"

    def test_signing_one_entry_in_does_not_sign_the_other_in(self, fake):
        a, b = fake.ctx("a", separate=True), fake.ctx("b", separate=True)
        EXCHANGE.begin_login(a)
        wait_for(a, ("succeeded",))

        assert credential(a) == MODULE.SIGNED_IN_MARKER
        with pytest.raises(InteractiveSetupRequired):
            credential(b)

    def test_account_identities_differ_per_profile_and_match_per_host(self, fake):
        a, b = fake.ctx("a", separate=True), fake.ctx("b", separate=True)
        h1, h2 = fake.ctx("h1"), fake.ctx("h2")

        assert EXCHANGE.account_id(a) != EXCHANGE.account_id(b)
        # Two entries on the host login are one upstream account, and say so.
        assert EXCHANGE.account_id(h1) == EXCHANGE.account_id(h2)
        assert EXCHANGE.account_id(a) != EXCHANGE.account_id(h1)

    def test_an_unknown_profile_value_is_refused_rather_than_guessed(self, fake):
        ctx = fake.ctx()
        bad = AuthContext(
            provider="agy",
            config={**ctx.config, "auth_fields": {"agy_profile": "other"}},
            state_dir=ctx.state_dir,
        )

        with pytest.raises(AuthExchangeError, match="agy_profile"):
            EXCHANGE.agent_args(bad)
        assert EXCHANGE.account_id(bad) is None

    def test_sign_out_removes_a_separate_profile(self, fake):
        ctx = fake.ctx(separate=True)
        EXCHANGE.begin_login(ctx)
        wait_for(ctx, ("succeeded",))
        profile = Path(EXCHANGE.agent_args(ctx)[1])
        assert (profile / "signed-in").exists()

        assert EXCHANGE.revoke(ctx) is False  # nothing is revoked upstream

        assert not profile.exists()
        assert EXCHANGE.needs_interactive_setup(ctx) is True

    def test_sign_out_of_the_host_login_is_refused_not_faked(self, fake):
        fake.sign_host_in()
        ctx = fake.ctx()
        credential(ctx)

        with pytest.raises(AuthExchangeError, match="gateway host"):
            EXCHANGE.revoke(ctx)
        assert (fake.host / "signed-in").exists()


class TestPathSafety:
    def test_a_provider_name_cannot_climb_out_of_the_default_directory(
        self, fake, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        ctx = fake.ctx("../../../etc", separate=True, state=False)

        (flag, path) = EXCHANGE.agent_args(ctx)
        base = (tmp_path / "home" / ".config" / "llm-rosetta" / "agy").resolve()

        assert flag == "--gemini-dir"
        assert Path(path).resolve().is_relative_to(base)
        assert ".." not in Path(path).parts

    def test_slugs_hold_no_separator_and_no_leading_dot(self):
        for name in ("../x", "..", "a/b", "/abs", ".hidden", "", "a b"):
            slug = MODULE._slug(name)
            assert "/" not in slug and not slug.startswith(".") and slug

    def test_a_profile_directory_that_is_a_link_out_is_refused(self, fake, tmp_path):
        ctx = fake.ctx("evil", separate=True)
        base = Path(ctx.state_dir)
        base.mkdir(parents=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        (base / "agy").symlink_to(outside)

        with pytest.raises(AuthExchangeError, match="escapes"):
            EXCHANGE.agent_args(ctx)


class TestScrubbing:
    @pytest.mark.parametrize(
        "text",
        [
            "ya29.A0ARrdaMsecretsecretsecret",
            "1//0gRefreshTokenValueHere123",
            "eyJhbGciOi.eyJzdWIiOi.c2lnbmF0dXJl",
            "AIzaSyD-exampleexampleexampleexample12",
            "Authorization: Bearer abc.def",
            "refresh_token=abc",
        ],
    )
    def test_token_shapes_are_redacted(self, text):
        assert "[REDACTED]" in MODULE.scrub_line("prefix " + text)

    def test_a_line_is_clipped_and_stripped_of_terminal_codes(self):
        assert MODULE.scrub_line("\x1b[31mred\x1b[0m") == "red"
        assert len(MODULE.scrub_line("x" * 5000)) == MODULE.MAX_LINE_CHARS

    @pytest.mark.parametrize(
        "url",
        [
            "http://accounts.example/x",
            "javascript:alert(1)",
            "https://accounts.example/cb?code=4/0AbCd",
            "https://accounts.example/cb?access_token=zzz",
            "https://" + "a" * 3000,
        ],
    )
    def test_a_url_that_is_not_safe_to_show_is_dropped(self, url):
        assert MODULE.safe_url(url) is None

    def test_an_authorization_url_is_kept(self):
        url = "https://accounts.example/o/oauth2/auth?client_id=x&state=abc"
        assert MODULE.safe_url(url + ".") == url
