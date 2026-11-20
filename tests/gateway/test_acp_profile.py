"""The gateway-owned agent profile and the tool-frame guard.

The agent CLI behind an ACP lane is an upstream, not the gateway's client: it
must never run a tool on the gateway host.  These tests drive a fake agent
that is *named* ``agy`` (the profile is applied to an ``agy`` command only) and
assert what the gateway wrote to disk, what it put on the command line, and
what it does when the agent reports a tool.
"""

from __future__ import annotations

import json
import logging
import os
import stat
import sys
from pathlib import Path

import pytest

from llm_rosetta_gateway.transport.acp import AcpTransport
from llm_rosetta_gateway.transport.acp_ledger import SessionKey
from llm_rosetta_gateway.transport.acp_profile import (
    AGENT_NAME,
    PERMISSION_FLAG,
    ensure_profile,
    flag_value,
    harden_argv,
    is_agy,
    state_dir_for,
    tool_name_of,
)
from llm_rosetta_gateway.transport.provider_info import ProviderInfo, openai_auth

from .test_acp_transport import FAKE_AGENT, TURN, _kinds, _log, _turn


def _agy_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
    *,
    name: str = "antigravity",
    custom: bool = False,
    argv_extra: tuple[str, ...] = (),
) -> ProviderInfo:
    """A provider whose agent executable is called ``agy``."""
    bin_dir = tmp_path / f"bin-{name}"
    bin_dir.mkdir(exist_ok=True)
    agy = bin_dir / "agy"
    agy.write_text(f"#!{sys.executable}\n{FAKE_AGENT}", encoding="utf-8")
    agy.chmod(agy.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("FAKE_ACP_MODE", mode)
    monkeypatch.setenv("FAKE_ACP_LOG", str(tmp_path / "agent.log"))
    return ProviderInfo(
        "acp",
        config_name=name,
        api_key="",
        base_url="http://agy.local",
        auth_header_fn=openai_auth,
        url_template="{base_url}/acp-turn",
        keyless=True,
        transport="acp",
        acp_argv=(str(agy), "acp", *argv_extra),
        acp_argv_custom=custom,
        acp_state_dir=str(tmp_path / "acp" / name),
    )


def _root(provider: ProviderInfo) -> Path:
    assert provider.acp_state_dir
    return Path(provider.acp_state_dir)


def _started_argv(tmp_path: Path) -> list[list[str]]:
    return [e["argv"] for e in _log(tmp_path) if e["kind"] == "start"]


class _Warnings(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


@pytest.fixture
def warnings_seen():
    handler = _Warnings()
    log = logging.getLogger("llm-rosetta-gateway")
    log.addHandler(handler)
    try:
        yield handler.lines
    finally:
        log.removeHandler(handler)


async def _chunks(transport: AcpTransport, provider: ProviderInfo, body=TURN):
    stream = await transport.send_streaming(provider, "", body)
    return [chunk async for chunk in stream]


class TestTheProfileOnDisk:
    async def test_the_profile_is_written_before_the_agent_starts(
        self, monkeypatch, tmp_path
    ):
        provider = _agy_provider(monkeypatch, tmp_path, "ok")
        transport = AcpTransport(timeout=5)
        try:
            await _chunks(transport, provider)
        finally:
            await transport.close()
        root = _root(provider)
        settings = json.loads((root / "antigravity-cli" / "settings.json").read_text())
        assert settings["toolPermission"] == "strict"
        hooks = json.loads((root / "config" / "hooks.json").read_text())
        hook = hooks["gateway-deny"]["PreToolUse"][0]
        assert hook["matcher"] == "*"
        assert '"decision":"deny"' in hook["hooks"][0]["command"]
        agent = (root / "config" / "agents" / AGENT_NAME / "agent.md").read_text()
        assert "tools: []" in agent and "excludeDefaultComponents: true" in agent
        assert stat.S_IMODE(os.stat(root).st_mode) == 0o700
        assert (
            stat.S_IMODE(os.stat(root / "antigravity-cli" / "settings.json").st_mode)
            == 0o600
        )

    async def test_the_argv_names_the_absolute_profile_and_the_agent(
        self, monkeypatch, tmp_path
    ):
        provider = _agy_provider(monkeypatch, tmp_path, "ok")
        transport = AcpTransport(timeout=5)
        try:
            await _chunks(transport, provider)
        finally:
            await transport.close()
        (argv,) = _started_argv(tmp_path)
        assert PERMISSION_FLAG in argv
        directory = argv[argv.index("--gemini-dir") + 1]
        assert os.path.isabs(directory) and directory == provider.acp_state_dir
        assert argv[argv.index("--agent") + 1] == AGENT_NAME

    async def test_a_hardened_session_is_not_put_in_plan_mode(
        self, monkeypatch, tmp_path
    ):
        provider = _agy_provider(monkeypatch, tmp_path, "ok")
        transport = AcpTransport(timeout=5)
        try:
            await _chunks(transport, provider)
        finally:
            await transport.close()
        assert "mode" not in _kinds(tmp_path)

    async def test_the_session_runs_in_an_empty_workspace_beside_the_profile(
        self, monkeypatch, tmp_path
    ):
        provider = _agy_provider(monkeypatch, tmp_path, "ok")
        transport = AcpTransport(timeout=5)
        try:
            await _chunks(transport, provider)
        finally:
            await transport.close()
        workspace = _root(provider) / "workspace"
        assert workspace.is_dir() and list(workspace.iterdir()) == []
        assert stat.S_IMODE(os.stat(workspace).st_mode) == 0o700

    async def test_a_tampered_profile_is_restored_before_the_next_turn(
        self, monkeypatch, tmp_path, warnings_seen
    ):
        provider = _agy_provider(monkeypatch, tmp_path, "ok")
        root = _root(provider)
        transport = AcpTransport(timeout=5)
        try:
            await _chunks(transport, provider)
            settings = root / "antigravity-cli" / "settings.json"
            settings.write_text('{"toolPermission": "always-proceed"}')
            (root / "config" / "hooks.json").unlink()
            (root / "config" / "agents" / AGENT_NAME / "agent.md").write_text(
                "tools: [all]"
            )
            await _chunks(transport, provider)
        finally:
            await transport.close()
        assert json.loads(settings.read_text())["toolPermission"] == "strict"
        assert (root / "config" / "hooks.json").is_file()
        assert (
            "tools: []"
            in (root / "config" / "agents" / AGENT_NAME / "agent.md").read_text()
        )
        changed = [line for line in warnings_seen if "was changed" in line]
        assert len(changed) == 1
        assert "settings.json" in changed[0] and "agent.md" in changed[0]

    def test_other_settings_keys_survive_the_re_assertion(self, tmp_path):
        settings = tmp_path / "antigravity-cli" / "settings.json"
        settings.parent.mkdir(parents=True)
        settings.write_text('{"modelProvider": "gemini", "toolPermission": "x"}')
        ensure_profile(str(tmp_path))
        data = json.loads(settings.read_text())
        assert data == {"modelProvider": "gemini", "toolPermission": "strict"}

    def test_an_untouched_profile_is_not_rewritten(self, tmp_path):
        assert ensure_profile(str(tmp_path)) == []
        before = (tmp_path / "config" / "hooks.json").stat().st_mtime_ns
        assert ensure_profile(str(tmp_path)) == []
        assert (tmp_path / "config" / "hooks.json").stat().st_mtime_ns == before

    def test_a_symlink_planted_at_a_profile_file_is_replaced_not_followed(
        self, tmp_path
    ):
        ensure_profile(str(tmp_path))
        victim = tmp_path / "victim.txt"
        victim.write_text("keep")
        hooks = tmp_path / "config" / "hooks.json"
        hooks.unlink()
        hooks.symlink_to(victim)
        ensure_profile(str(tmp_path))
        assert victim.read_text() == "keep"
        assert not hooks.is_symlink()


class TestTheGuard:
    async def test_a_builtin_tool_frame_cancels_and_fails_the_turn(
        self, monkeypatch, tmp_path, warnings_seen
    ):
        provider = _agy_provider(monkeypatch, tmp_path, "tool")
        transport = AcpTransport(timeout=5)
        try:
            chunks = await _chunks(transport, provider)
        finally:
            await transport.close()
        assert chunks[-1]["type"] == "error"
        assert "search_web" in chunks[-1]["message"]
        assert "agent tools are disabled" in chunks[-1]["message"]
        assert "stop" not in [c["type"] for c in chunks]
        kinds = _kinds(tmp_path)
        assert "cancel" in kinds and "close" in kinds
        lines = [x for x in warnings_seen if "built-in tool" in x]
        assert lines == [
            "ACP provider 'antigravity': the agent started its built-in tool "
            "'search_web'; agent tools are disabled, so the turn was stopped"
        ]
        everything = " ".join(warnings_seen) + chunks[-1]["message"]
        assert "SECRET" not in everything

    async def test_a_non_streaming_request_gets_a_502_not_an_empty_200(
        self, monkeypatch, tmp_path
    ):
        provider = _agy_provider(monkeypatch, tmp_path, "tool")
        transport = AcpTransport(timeout=5)
        try:
            response = await transport.send(provider, "", TURN)
        finally:
            await transport.close()
        assert response.status_code == 502
        assert b"agent tools are disabled" in response.raw_content

    async def test_an_mcp_named_frame_is_blocked_too_in_phase_one(
        self, monkeypatch, tmp_path
    ):
        # Phase 2 names its bridge tools ``mcp_<server>_<tool>`` and will
        # relax this; phase 1 has no bridge, so any tool frame is refused.
        provider = _agy_provider(monkeypatch, tmp_path, "mcp-tool")
        transport = AcpTransport(timeout=5)
        try:
            chunks = await _chunks(transport, provider)
        finally:
            await transport.close()
        assert chunks[-1]["type"] == "error"
        assert "mcp_bridge_echo" in chunks[-1]["message"]

    async def test_the_session_is_not_reused_after_the_guard_fires(
        self, monkeypatch, tmp_path
    ):
        provider = _agy_provider(monkeypatch, tmp_path, "tool").with_acp_conversation(
            "conv-a"
        )
        transport = AcpTransport(timeout=5)
        try:
            await _chunks(transport, provider, _turn("[User]\none"))
            assert transport.ledger.get(SessionKey("antigravity", "conv-a")) is None
        finally:
            await transport.close()

    def test_tool_name_of_reads_only_tool_frames_and_only_the_name(self):
        frame = {
            "update": {
                "sessionUpdate": "tool_call_update",
                "name": "read_file",
                "title": "Read /etc/passwd",
                "rawInput": {"Path": "/etc/passwd"},
            }
        }
        assert tool_name_of(frame) == "read_file"
        assert (
            tool_name_of({"update": {"sessionUpdate": "agent_message_chunk"}}) is None
        )
        no_name = {"update": {"sessionUpdate": "tool_call", "kind": "execute"}}
        assert tool_name_of(no_name) == "kind:execute"
        hostile = {
            "update": {"sessionUpdate": "tool_call", "name": "a b\n" + "x" * 200}
        }
        name = tool_name_of(hostile)
        assert name is not None and len(name) <= 64 and "\n" not in name


class TestOverridesAndSeparateProfiles:
    async def test_an_acp_argv_override_keeps_its_override_and_gets_no_profile(
        self, monkeypatch, tmp_path
    ):
        provider = _agy_provider(
            monkeypatch,
            tmp_path,
            "ok",
            custom=True,
            argv_extra=("--dangerously-skip-permissions=true",),
        )
        transport = AcpTransport(timeout=5)
        try:
            await _chunks(transport, provider)
        finally:
            await transport.close()
        (argv,) = _started_argv(tmp_path)
        assert argv == ["acp", "--dangerously-skip-permissions=true"]
        assert not _root(provider).exists()
        # The old best-effort gating stays for an agent without the profile.
        assert "mode" in _kinds(tmp_path)

    async def test_two_entries_get_two_profiles_and_two_processes(
        self, monkeypatch, tmp_path
    ):
        one = _agy_provider(monkeypatch, tmp_path, "ok", name="agy-one")
        two = _agy_provider(monkeypatch, tmp_path, "ok", name="agy-two")
        transport = AcpTransport(timeout=5)
        try:
            await _chunks(transport, one)
            await _chunks(transport, two)
        finally:
            await transport.close()
        argvs = _started_argv(tmp_path)
        assert len(argvs) == 2
        dirs = {a[a.index("--gemini-dir") + 1] for a in argvs}
        assert dirs == {one.acp_state_dir, two.acp_state_dir}
        assert len(dirs) == 2
        pids = {e["pid"] for e in _log(tmp_path) if e["kind"] == "start"}
        assert len(pids) == 2

    async def test_a_gemini_dir_already_in_the_argv_gets_the_profile(
        self, monkeypatch, tmp_path
    ):
        # An entry with its own login profile names that directory; the strict
        # settings are written there and no second --gemini-dir is added.
        own = tmp_path / "entry-profile"
        provider = _agy_provider(
            monkeypatch, tmp_path, "ok", argv_extra=("--gemini-dir", str(own))
        )
        transport = AcpTransport(timeout=5)
        try:
            await _chunks(transport, provider)
        finally:
            await transport.close()
        (argv,) = _started_argv(tmp_path)
        assert argv.count("--gemini-dir") == 1
        assert flag_value(argv, ("--gemini-dir",)) == str(own)
        assert (
            json.loads((own / "antigravity-cli" / "settings.json").read_text())[
                "toolPermission"
            ]
            == "strict"
        )


class TestArgvHelpers:
    def test_is_agy_matches_the_executable_name_only(self):
        assert is_agy(["/usr/local/bin/agy", "acp"])
        assert not is_agy(["python", "agy"])
        assert not is_agy([])

    def test_harden_argv_adds_missing_flags_once(self):
        out = harden_argv(["agy", "acp"], "/state/x")
        assert out == [
            "agy",
            "acp",
            PERMISSION_FLAG,
            "--gemini-dir",
            "/state/x",
            "--agent",
            AGENT_NAME,
        ]
        assert harden_argv(out, "/state/x").count("--agent") == 1

    def test_harden_argv_replaces_a_foreign_agent_and_keeps_a_profile(self):
        out = harden_argv(["agy", "acp", "--agent=other", "--gemini_dir=/mine"], "/s")
        assert "--agent=other" not in out
        assert flag_value(out, ("--gemini-dir", "--gemini_dir")) == "/mine"
        assert out[-2:] == ["--agent", AGENT_NAME]

    def test_state_dir_is_per_provider_under_the_data_dir(self):
        assert state_dir_for("/data", "agy one") == "/data/acp/agy_one"
        assert state_dir_for("/data", "../x") == "/data/acp/_x"


class TestPerEntryLoginProfile:
    """An ``agy_profile: separate`` entry: its own dir holds login and profile."""

    @staticmethod
    def _entry(monkeypatch, tmp_path, name="agy-sep", *, separate=True):
        from llm_rosetta.shims.auth_exchange import AuthContext
        from llm_rosetta.shims.provider_shim import get_shim

        provider = _agy_provider(monkeypatch, tmp_path, "ok", name=name)
        shim = get_shim("antigravity--acp")
        assert shim is not None
        cfg: dict = {"auth_fields": {"agy_profile": "separate"}} if separate else {}
        provider.auth_exchange = shim.auth
        provider.auth_context = AuthContext(
            provider=name, config=cfg, state_dir=str(tmp_path / "shim-auth" / name)
        )
        return provider

    async def test_agent_args_come_before_hardening_and_hold_the_strict_profile(
        self, monkeypatch, tmp_path
    ):
        provider = self._entry(monkeypatch, tmp_path)
        own = tmp_path / "shim-auth" / "agy-sep" / "agy"
        transport = AcpTransport(timeout=5)
        try:
            await _chunks(transport, provider)
        finally:
            await transport.close()
        (argv,) = _started_argv(tmp_path)
        assert argv.count("--gemini-dir") == 1
        assert flag_value(argv, ("--gemini-dir",)) == str(own)
        assert argv.count(PERMISSION_FLAG) == 1
        assert flag_value(argv, ("--agent",)) == AGENT_NAME
        settings = json.loads((own / "antigravity-cli" / "settings.json").read_text())
        assert settings["toolPermission"] == "strict"
        assert (own / "config" / "hooks.json").is_file()
        assert (own / "config" / "agents" / AGENT_NAME / "agent.md").is_file()
        # The default profile directory of the entry is not used at all.
        assert not (_root(provider) / "antigravity-cli").exists()
        assert oct(own.stat().st_mode & 0o777) == "0o700"

    async def test_the_agent_args_precede_the_hardening_flags(
        self, monkeypatch, tmp_path
    ):
        provider = self._entry(monkeypatch, tmp_path)
        argv = AcpTransport(timeout=5)._argv(provider)
        own = str(tmp_path / "shim-auth" / "agy-sep" / "agy")
        assert argv[argv.index("--gemini-dir") + 1] == own
        assert argv.index("--gemini-dir") < argv.index("--agent")
        assert argv.count("--gemini-dir") == 1

    async def test_a_login_made_under_the_dir_survives_every_re_assertion(
        self, monkeypatch, tmp_path
    ):
        provider = self._entry(monkeypatch, tmp_path)
        own = tmp_path / "shim-auth" / "agy-sep" / "agy"
        # The "login" writes a marker file, a credential-like blob in a nested
        # dir, and an auth method into settings.json, as agy's own login would.
        (own / "antigravity-cli").mkdir(parents=True)
        (own / "login-marker").write_text("signed in")
        (own / "oauth").mkdir()
        (own / "oauth" / "token.json").write_text("{}")
        (own / "antigravity-cli" / "settings.json").write_text(
            json.dumps({"selectedAuthType": "oauth-personal", "toolPermission": "off"})
        )
        transport = AcpTransport(timeout=5)
        try:
            await _chunks(transport, provider)
            drift = ensure_profile(str(own))
            ensure_profile(str(own))
        finally:
            await transport.close()
        assert (own / "login-marker").read_text() == "signed in"
        assert (own / "oauth" / "token.json").read_text() == "{}"
        settings = json.loads((own / "antigravity-cli" / "settings.json").read_text())
        assert settings["selectedAuthType"] == "oauth-personal"
        assert settings["toolPermission"] == "strict"
        assert drift == []
        # Only the three profile files and their directories are ours.
        names = {p.name for p in own.rglob("*") if p.is_file()}
        assert names == {
            "login-marker",
            "token.json",
            "settings.json",
            "hooks.json",
            "agent.md",
        }

    async def test_a_marker_written_by_the_agent_before_a_turn_survives(
        self, monkeypatch, tmp_path
    ):
        provider = self._entry(monkeypatch, tmp_path)
        own = tmp_path / "shim-auth" / "agy-sep" / "agy"
        agy = Path(provider.acp_argv[0])
        body = agy.read_text(encoding="utf-8").split("\n", 1)[1]
        agy.write_text(
            f"#!{sys.executable}\n"
            "import os, sys\n"
            "d = sys.argv[sys.argv.index('--gemini-dir') + 1]\n"
            "open(os.path.join(d, 'login'), 'w').write('x')\n" + body,
            encoding="utf-8",
        )
        transport = AcpTransport(timeout=5)
        try:
            await _chunks(transport, provider)
            transport._profile(provider, transport._argv(provider))
        finally:
            await transport.close()
        assert (own / "login").read_text() == "x"

    async def test_a_host_entry_is_unchanged(self, monkeypatch, tmp_path):
        provider = self._entry(monkeypatch, tmp_path, "agy-host", separate=False)
        transport = AcpTransport(timeout=5)
        try:
            await _chunks(transport, provider)
        finally:
            await transport.close()
        (argv,) = _started_argv(tmp_path)
        assert flag_value(argv, ("--gemini-dir",)) == provider.acp_state_dir
        assert argv.count("--gemini-dir") == 1
        assert not (tmp_path / "shim-auth" / "agy-host" / "agy").exists()

    async def test_the_status_probe_works_with_the_profile_files_present(
        self, monkeypatch, tmp_path
    ):
        from tests.gateway.fake_agy import install_fake_agy, starts

        provider = self._entry(monkeypatch, tmp_path, "agy-probe")
        own = tmp_path / "shim-auth" / "agy-probe" / "agy"
        ensure_profile(str(own))
        (own / "signed-in").write_text("x")
        bin_dir = tmp_path / "fakebin"
        bin_dir.mkdir()
        agy = install_fake_agy(bin_dir)
        monkeypatch.setenv("FAKE_AGY_HOST", str(tmp_path / "host"))
        monkeypatch.setenv("FAKE_AGY_LOG", str(tmp_path / "probe.jsonl"))
        ctx = provider.auth_context
        ctx.config["acp_argv"] = [str(agy), "acp"]
        assert await provider.auth_exchange.credential(ctx)
        (start,) = starts(tmp_path / "probe.jsonl")
        # Un-hardened: no agent flag, no permission flag, and no prompt sent.
        assert "--agent" not in start["argv"]
        assert PERMISSION_FLAG not in start["argv"]
        assert start["argv"][-2:] == ["--gemini-dir", str(own)]
        # The probe leaves the profile as it found it.
        assert ensure_profile(str(own)) == []
