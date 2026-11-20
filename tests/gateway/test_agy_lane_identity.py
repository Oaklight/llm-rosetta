"""One provider entry is one agy identity, end to end through the config builder.

Entries are built the way the gateway builds them (``build_provider_info`` with
the shim and a data directory), then started through the real
:class:`AcpTransport` against the fake ``agy``.  What is asserted is what the
processes were actually launched with.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_rosetta_gateway.providers import build_provider_info
from llm_rosetta_gateway.transport.acp import AcpTransport

from .fake_agy import alive, install_fake_agy, starts


@pytest.fixture
def lane(tmp_path, monkeypatch):
    agy = install_fake_agy(tmp_path)
    host = tmp_path / "host"
    host.mkdir()
    monkeypatch.setenv("FAKE_AGY_HOST", str(host))
    monkeypatch.setenv("FAKE_AGY_LOG", str(tmp_path / "log.jsonl"))
    data = tmp_path / "data"
    data.mkdir()

    def build(name: str, **extra):
        cfg = {"acp_argv": [str(agy), "acp"], **extra}
        return build_provider_info(
            "acp",
            cfg,
            shim_name="antigravity--acp",
            data_dir=str(data),
            config_name=name,
            quiet=True,
        )

    return build, tmp_path, data


class TestTheEntryBuilder:
    def test_an_entry_has_an_exchange_and_is_not_keyless(self, lane):
        build, _, _ = lane
        pinfo = build("agy")

        assert pinfo.auth_exchange is not None
        assert pinfo.keyless is False
        assert pinfo.mints_credential is True
        assert pinfo.credential_source == "auth_exchange"
        # Not ready until agy has been asked: a signed-out lane is not routed to.
        assert pinfo.credentials_pending is True

    def test_two_entries_get_two_state_directories(self, lane):
        build, _, data = lane
        a, b = build("agy"), build("agy2")

        assert a.auth_context.state_dir != b.auth_context.state_dir
        assert Path(a.auth_context.state_dir).parent == data / "shim-auth"

    def test_the_entry_cannot_carry_a_static_key(self, lane):
        build, _, _ = lane

        with pytest.raises(ValueError, match="mutually exclusive"):
            build("agy", api_key="sk-own")


class TestTheLaunchArgv:
    def test_a_host_entry_is_launched_exactly_as_before(self, lane):
        build, tmp_path, _ = lane
        pinfo = build("agy")

        assert AcpTransport()._argv(pinfo) == [str(tmp_path / "agy"), "acp"]

    def test_a_separate_entry_is_launched_with_its_own_profile(self, lane):
        build, tmp_path, _ = lane
        pinfo = build("agy2", auth_fields={"agy_profile": "separate"})
        argv = AcpTransport()._argv(pinfo)

        assert argv[:2] == [str(tmp_path / "agy"), "acp"]
        assert argv[2] == "--gemini-dir"
        assert argv[3] == str(Path(pinfo.auth_context.state_dir) / "agy")

    def test_an_exchange_less_provider_gets_only_the_hardening(self):
        from llm_rosetta_gateway.transport.provider_info import (
            ProviderInfo,
            openai_auth,
        )

        pinfo = ProviderInfo(
            "acp",
            config_name="x",
            api_key="",
            base_url="http://agy.local",
            auth_header_fn=openai_auth,
            url_template="{base_url}/",
            transport="acp",
            acp_argv=("agy", "acp"),
        )

        argv = AcpTransport()._argv(pinfo)

        # No identity argument from an exchange: the argv is the declared one,
        # then the gateway's own profile flags, each exactly once.
        assert argv[:2] == ["agy", "acp"]
        assert argv.count("--gemini-dir") == 1
        assert argv.count("--dangerously-skip-permissions=false") == 1
        assert argv[-2:] == ["--agent", "gateway-text-only"]


class TestSeparateProcesses:
    async def test_two_entries_run_two_agents_with_two_profiles(self, lane):
        build, tmp_path, _ = lane
        host_entry = build("agy")
        separate_entry = build("agy2", auth_fields={"agy_profile": "separate"})
        transport = AcpTransport()
        try:
            await transport._agent(host_entry)
            await transport._agent(separate_entry)
        finally:
            started = starts(tmp_path / "log.jsonl")
            await transport.close()

        assert len({s["pid"] for s in started}) == 2
        by_flag = {("--gemini-dir" in s["argv"]): s for s in started}
        assert by_flag[False]["argv"] == ["acp"]
        profile = by_flag[True]["argv"][by_flag[True]["argv"].index("--gemini-dir") + 1]
        assert profile == str(Path(separate_entry.auth_context.state_dir) / "agy")

    async def test_closing_one_entry_leaves_the_other_running(self, lane):
        build, tmp_path, _ = lane
        a = build("agy", auth_fields={"agy_profile": "separate"})
        b = build("agy2", auth_fields={"agy_profile": "separate"})
        transport = AcpTransport()
        try:
            await transport._agent(a)
            await transport._agent(b)
            pids = {
                s["argv"][-1].split("/")[-2]: s["pid"]
                for s in starts(tmp_path / "log.jsonl")
            }
            await transport.aclose_provider("agy2")

            import asyncio

            await asyncio.sleep(0.3)
            assert alive(pids["agy"])
            assert not alive(pids["agy2"])
        finally:
            await transport.close()
