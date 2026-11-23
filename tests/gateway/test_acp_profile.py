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
import subprocess
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
    harden: bool = True,
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
        acp_harden=harden,
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
        assert settings["permissions"]["allow"] == ["mcp(*)"]
        hooks = json.loads((root / "config" / "hooks.json").read_text())
        hook = hooks["gateway-client-tools"]["PreToolUse"][0]
        assert hook["matcher"] == "*"
        command = hook["hooks"][0]["command"]
        assert "allow_client_tools.py" in command
        script = root / "config" / "hooks" / "allow_client_tools.py"
        text = script.read_text()
        assert "call_mcp_tool" in text and "llm_rosetta_tools" in text
        assert stat.S_IMODE(os.stat(script).st_mode) == 0o600
        agent = (root / "config" / "agents" / AGENT_NAME / "agent.md").read_text()
        assert "tools: []" in agent and "excludeDefaultComponents: true" in agent
        assert "llm_rosetta_tools" in agent
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
        assert data == {
            "modelProvider": "gemini",
            "toolPermission": "strict",
            "permissions": {"allow": ["mcp(*)"]},
        }

    def test_the_hook_allows_only_the_client_server(self, tmp_path):
        ensure_profile(str(tmp_path))
        script = tmp_path / "config" / "hooks" / "allow_client_tools.py"

        def decide(payload):
            proc = subprocess.run(
                [sys.executable, str(script)],
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                check=True,
            )
            return json.loads(proc.stdout)["decision"]

        assert (
            decide(
                {
                    "toolCall": {
                        "name": "call_mcp_tool",
                        "args": {
                            "ServerName": "llm_rosetta_tools",
                            "ToolName": "echo",
                        },
                    }
                }
            )
            == "allow"
        )
        assert (
            decide(
                {
                    "toolCall": {
                        "name": "call_mcp_tool",
                        "args": {"ServerName": "other", "ToolName": "echo"},
                    }
                }
            )
            == "deny"
        )
        # agy 1.5.14 and later expose each MCP tool under the server prefix
        # instead of routing it through `call_mcp_tool` (measured live).
        assert (
            decide(
                {
                    "toolCall": {
                        "name": "mcp_llm_rosetta_tools_echo",
                        "args": {"text": "hello", "toolSummary": "Echo text"},
                    }
                }
            )
            == "allow"
        )
        assert (
            decide({"toolCall": {"name": "mcp_other_tools_echo", "args": {}}}) == "deny"
        )
        # A bare prefix names no tool.
        assert (
            decide({"toolCall": {"name": "mcp_llm_rosetta_tools_", "args": {}}})
            == "deny"
        )
        assert decide({"toolCall": {"name": "run_command", "args": {}}}) == "deny"
        assert (
            decide(
                {
                    "toolCall": {
                        "name": "list_resources",
                        "args": {"ServerName": "llm_rosetta_tools"},
                    }
                }
            )
            == "allow"
        )

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

    async def test_an_undeclared_mcp_named_frame_is_blocked(
        self, monkeypatch, tmp_path
    ):
        # A request with no declared tools has no bridge, so a frame that
        # merely looks like an MCP tool is still an agent tool and is refused.
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
    async def test_an_acp_argv_override_still_gets_the_profile(
        self, monkeypatch, tmp_path
    ):
        """Another install is not another profile: the sign-in must survive."""
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
        assert argv[:3] == [
            "acp",
            "--dangerously-skip-permissions=true",
            "--process-per-session",
        ]
        assert argv[argv.index("--gemini-dir") + 1] == str(_root(provider))
        assert argv[-2:] == ["--agent", AGENT_NAME]
        assert (_root(provider) / "config" / "hooks.json").is_file()
        # The profile asks for every tool; the plan mode is not needed.
        assert "mode" not in _kinds(tmp_path)

    async def test_harden_false_runs_the_command_as_written(
        self, monkeypatch, tmp_path
    ):
        provider = _agy_provider(
            monkeypatch,
            tmp_path,
            "ok",
            custom=True,
            harden=False,
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
            "--process-per-session",
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
