"""``llm-rosetta-gateway credential``: produce a record where a browser is.

The admin API can install a shim-declared credential record, but a record has
to be *made* first — and the gateway host is often a machine nobody signs in
on.  This command is the other end: it runs the exchange's interactive login on
the operator's own machine and prints what the admin API accepts.
"""

from __future__ import annotations

import json
import os
import stat
from types import SimpleNamespace
from typing import Any

import pytest

from llm_rosetta_gateway import cli

RECORD = {
    "access_token": "tok-a",
    "refresh_token": "tok-r",
    "id_token": "tok-i",
    "client_id": "client-1",
}


class _Exchange:
    def __init__(self, *, can_login: bool = True, fails: str | None = None) -> None:
        self._can_login = can_login
        self.fails = fails
        self.calls = 0

    def can_login(self) -> bool:
        return self._can_login

    def login(self, ctx: Any) -> dict[str, Any]:
        self.calls += 1
        if self.fails:
            raise RuntimeError(self.fails)
        return dict(RECORD)


def _config(exchange: Any) -> Any:
    pinfo = SimpleNamespace(
        auth_exchange=exchange,
        auth_context=SimpleNamespace(provider="chatgpt"),
    )
    return SimpleNamespace(providers={"chatgpt": pinfo})


@pytest.fixture
def configured(monkeypatch):
    """Point the command at a config that exists, without reading a file."""
    exchange = _Exchange()

    def _install(cfg: Any = None) -> _Exchange:
        monkeypatch.setattr(cli, "discover_config", lambda path: "/tmp/c.jsonc")
        monkeypatch.setattr(
            cli,
            "_load_startup_config",
            lambda path, args: (cfg or _config(exchange), None),
        )
        monkeypatch.setattr(cli, "_report_shim_sources", lambda report: None)
        return exchange

    return _install


def _args(**kw: Any) -> Any:
    base = {"config": None, "provider": "chatgpt", "out": None}
    base.update(kw)
    return SimpleNamespace(**base)


class TestTheRecord:
    def test_it_goes_to_stdout(self, configured, capsys):
        exchange = configured()

        cli._cmd_credential(_args())

        out = capsys.readouterr()
        assert json.loads(out.out) == RECORD
        assert exchange.calls == 1
        # stdout is the record alone, so it can be piped; the advice is stderr.
        assert "PUT /admin/api/config/providers/chatgpt/credential" in out.err
        assert "do not paste it anywhere else" in out.err

    def test_out_writes_a_private_file(self, configured, capsys, tmp_path):
        configured()
        target = tmp_path / "record.json"

        cli._cmd_credential(_args(out=str(target)))

        assert json.loads(target.read_text()) == RECORD
        mode = stat.S_IMODE(os.stat(target).st_mode)
        assert mode == 0o600
        out = capsys.readouterr()
        # The record must not also land in the terminal.
        assert "tok-a" not in out.out
        assert "tok-a" not in out.err
        assert str(target) in out.out


class TestTheWriterSurvivesAPlantedPath:
    """``--out /tmp/rec.json`` must not let the directory decide the mode.

    The temporary name is derivable from the path the operator typed, so on a
    shared host someone with write access there can plant a file or a symlink
    at it.  ``0o600`` is the mode *if the file is created*, and ``O_CREAT``
    keeps an existing mode and follows a link.
    """

    def test_a_planted_temp_file_neither_keeps_its_mode_nor_is_overwritten(
        self, configured, tmp_path
    ):
        configured()
        target = tmp_path / "rec.json"
        planted = tmp_path / ".rec.json.tmp"
        planted.write_text("someone else's file")
        os.chmod(planted, 0o644)

        cli._cmd_credential(_args(out=str(target)))

        assert stat.S_IMODE(os.stat(target).st_mode) == 0o600
        assert json.loads(target.read_text()) == RECORD
        assert planted.read_text() == "someone else's file"

    def test_a_symlink_at_the_temp_path_is_not_written_through(
        self, configured, tmp_path
    ):
        configured()
        victim = tmp_path / "victim.txt"
        victim.write_text("original")
        os.chmod(victim, 0o644)
        target = tmp_path / "rec.json"
        os.symlink(victim, tmp_path / ".rec.json.tmp")

        cli._cmd_credential(_args(out=str(target)))

        assert victim.read_text() == "original"
        assert not target.is_symlink()
        assert stat.S_IMODE(os.stat(target).st_mode) == 0o600


class TestRefusals:
    def test_an_unknown_provider(self, configured, capsys):
        configured()

        with pytest.raises(SystemExit) as exit_info:
            cli._cmd_credential(_args(provider="nope"))

        assert exit_info.value.code == 1
        assert "Unknown provider 'nope'. Configured providers: chatgpt" in (
            capsys.readouterr().err
        )

    def test_a_provider_without_an_exchange(self, configured, capsys):
        configured(_config(None))

        with pytest.raises(SystemExit) as exit_info:
            cli._cmd_credential(_args())

        assert exit_info.value.code == 1
        assert "mints no credential of its own" in capsys.readouterr().err

    def test_an_exchange_without_a_browser_flow(self, configured, capsys):
        exchange = configured()
        exchange._can_login = False

        with pytest.raises(SystemExit) as exit_info:
            cli._cmd_credential(_args())

        assert exit_info.value.code == 1
        assert "does not implement login()" in capsys.readouterr().err
        assert exchange.calls == 0

    def test_a_failed_sign_in_is_reported_in_its_own_words(self, configured, capsys):
        exchange = configured()
        exchange.fails = "the sign-in timed out"

        with pytest.raises(SystemExit) as exit_info:
            cli._cmd_credential(_args())

        assert exit_info.value.code == 1
        assert "Sign-in failed: the sign-in timed out" in capsys.readouterr().err

    def test_a_message_that_carries_a_credential_is_scrubbed(self, configured, capsys):
        """The contract says it must not, but a vendor body is not ours.

        The canary is a JWT rather than an ``sk-`` key on purpose: the record
        this flow produces is made of JWTs, so that is the shape the scrub has
        to catch.
        """
        exchange = configured()
        # Assembled from its segments so no scanner mistakes the fixture for a
        # real token; the shape is what is under test, not the value.
        token = ".".join(
            ("eyJhbGciOiJub25lIn0", "eyJzdWIiOiJjYW5hcnkifQ", "c2lnbmF0dXJl")
        )
        exchange.fails = f"the issuer refused: id_token={token}"

        with pytest.raises(SystemExit):
            cli._cmd_credential(_args())

        err = capsys.readouterr().err
        assert token not in err
        assert "[REDACTED]" in err


class TestTheConfigPath:
    def test_a_nonexistent_config_is_reported_not_raised(self, monkeypatch, capsys):
        """A typo'd path is an operator mistake, not a traceback."""
        monkeypatch.setattr(cli, "discover_config", lambda path: "/tmp/nope.jsonc")

        with pytest.raises(SystemExit) as exit_info:
            cli._cmd_credential(_args())

        assert exit_info.value.code == 1
        err = capsys.readouterr().err
        assert "Cannot read config file /tmp/nope.jsonc" in err


class TestTheConfigFlag:
    """``--config`` must work both ways round, as it does elsewhere."""

    def _run(self, monkeypatch, argv: list[str]) -> dict[str, Any]:
        seen: dict[str, Any] = {}
        monkeypatch.setattr(
            cli,
            "_cmd_credential",
            lambda args: seen.update(provider=args.provider, config=args.config),
        )
        monkeypatch.setattr(cli.sys, "argv", ["llm-rosetta-gateway", *argv])
        cli.main()
        return seen

    def test_after_the_subcommand(self, monkeypatch):
        seen = self._run(monkeypatch, ["credential", "probe", "--config", "cfg.jsonc"])

        assert seen == {"provider": "probe", "config": "cfg.jsonc"}

    def test_before_the_subcommand(self, monkeypatch):
        """A subparser default would silently drop the global one."""
        seen = self._run(monkeypatch, ["--config", "cfg.jsonc", "credential", "probe"])

        assert seen == {"provider": "probe", "config": "cfg.jsonc"}


class TestTheLocalCopy:
    def test_the_note_names_the_record_directory(self, configured, capsys):
        """The sign-in stores a copy here; a remote operator must know."""
        pinfo_exchange = configured()

        def with_state_dir():
            pinfo = SimpleNamespace(
                auth_exchange=pinfo_exchange,
                auth_context=SimpleNamespace(provider="chatgpt"),
                auth_state_dir="/srv/gw/data/shim-auth/chatgpt",
            )
            return SimpleNamespace(providers={"chatgpt": pinfo})

        configured(with_state_dir())

        cli._cmd_credential(_args())

        assert "/srv/gw/data/shim-auth/chatgpt" in capsys.readouterr().err
