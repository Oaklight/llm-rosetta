"""The persistent profile one ACP lane's agent and sign-in share.

A vendor CLI keeps its own login store.  The gateway must point that store at
a directory under its data directory, and it must point the admin sign-in at
the same directory as the agent it will later start.  These tests pin the
path resolution, the ``{acp_home}`` expansion in a shim's environment, and
the environment a credential exchange receives.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_rosetta_gateway.config.providers import (
    _acp_env,
    _acp_state_dir,
    _auth_context,
    _bind_auth_exchange,
)
from llm_rosetta_gateway.transport.acp_profile import state_dir_for


def _shim(env: dict[str, str]) -> SimpleNamespace:
    return SimpleNamespace(connection=SimpleNamespace(acp_env=env))


class TestTheHomePath:
    def test_the_default_is_one_directory_per_provider_under_the_data_dir(self):
        assert _acp_state_dir("/data", "agy one", {}) == "/data/acp/agy_one"
        assert _acp_state_dir(None, "../x", {}) == state_dir_for(None, "../x")

    def test_a_relative_home_resolves_under_the_data_dir(self):
        cfg = {"acp": {"home": "saved/claude-pro"}}

        assert _acp_state_dir("/data", "claude", cfg) == "/data/saved/claude-pro"

    def test_an_absolute_home_is_used_as_written(self, tmp_path):
        wanted = tmp_path / "volume" / "acp-home"
        cfg = {"acp": {"home": str(wanted)}}

        assert _acp_state_dir("/data", "claude", cfg) == str(wanted)

    def test_a_tilde_is_expanded(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        cfg = {"acp": {"home": "~/saved/claude"}}

        assert _acp_state_dir("/data", "claude", cfg) == str(tmp_path / "saved/claude")

    def test_two_providers_can_share_a_switched_home(self, tmp_path):
        cfg = {"acp": {"home": str(tmp_path / "shared")}}

        assert _acp_state_dir("/data", "one", cfg) == _acp_state_dir(
            "/data", "two", cfg
        )

    @pytest.mark.parametrize("bad", ["", "   ", 7, ["/x"]])
    def test_a_malformed_home_is_refused(self, bad):
        with pytest.raises(ValueError, match="acp.home must be a non-empty path"):
            _acp_state_dir("/data", "claude", {"acp": {"home": bad}})


class TestTheAgentEnvironment:
    def test_the_shims_placeholder_is_expanded(self):
        shim = _shim(
            {
                "ENABLE_TOOL_SEARCH": "false",
                "CLAUDE_CONFIG_DIR": "{acp_home}/claude",
            }
        )

        env = _acp_env({}, shim, "claude", "/data/acp/claude", inject_home=True)

        assert env == {
            "ENABLE_TOOL_SEARCH": "false",
            "CLAUDE_CONFIG_DIR": "/data/acp/claude/claude",
            "LLM_ROSETTA_ACP_HOME": "/data/acp/claude",
        }
        assert "{acp_home}" not in repr(env)

    def test_the_state_dir_spelling_is_an_alias(self):
        shim = _shim({"P": "{acp_state_dir}/x"})

        env = _acp_env({}, shim, "p", "/home/p", inject_home=True)

        assert env is not None
        assert env["P"] == "/home/p/x"

    def test_the_operators_value_wins_over_the_shims(self):
        shim = _shim({"CLAUDE_CONFIG_DIR": "{acp_home}/claude"})

        env = _acp_env(
            {"acp_env": {"CLAUDE_CONFIG_DIR": "/mine"}},
            shim,
            "claude",
            "/home/lane",
            inject_home=True,
        )

        assert env is not None
        assert env["CLAUDE_CONFIG_DIR"] == "/mine"
        assert env["LLM_ROSETTA_ACP_HOME"] == "/home/lane"

    def test_the_operator_may_override_the_published_home(self):
        env = _acp_env(
            {"acp_env": {"LLM_ROSETTA_ACP_HOME": "/mine"}},
            None,
            "claude",
            "/home/lane",
            inject_home=True,
        )

        assert env == {"LLM_ROSETTA_ACP_HOME": "/mine"}

    def test_an_http_lane_does_not_publish_the_acp_home(self):
        assert _acp_env({}, None, "openai", "/home/lane") is None

    def test_the_operator_environment_is_still_type_checked(self):
        with pytest.raises(ValueError, match="acp_env must map strings"):
            _acp_env({"acp_env": {"X": 1}}, None, "claude", "/home/lane")


class TestTheExchangeContext:
    def test_the_exchange_sees_the_effective_agent_environment(self):
        effective = {
            "ENABLE_TOOL_SEARCH": "false",
            "CLAUDE_CONFIG_DIR": "/data/acp/claude/claude",
            "LLM_ROSETTA_ACP_HOME": "/data/acp/claude",
        }

        ctx = _auth_context("claude", {"acp_env": {"IGNORED": "1"}}, "/auth", effective)

        assert ctx.config["acp_env"] == effective

    def test_binding_an_exchange_forwards_the_effective_environment(self):
        seen = {}

        class Exchange:
            vendor_managed = True

            def account_id(self, ctx):
                seen.update(ctx.config)
                return None

        context, account_id = _bind_auth_exchange(
            "claude",
            {"acp_env": {"CLAUDE_CONFIG_DIR": "/old"}},
            Exchange(),
            "/data",
            {"CLAUDE_CONFIG_DIR": "/data/acp/claude/claude"},
        )

        assert account_id is None
        assert context is not None
        assert seen["acp_env"]["CLAUDE_CONFIG_DIR"] == "/data/acp/claude/claude"

    def test_the_exchange_never_receives_the_operators_key(self):
        ctx = _auth_context("claude", {"api_key": "sk-secret"}, "/auth", None)

        assert "api_key" not in ctx.config
        assert ctx.input_key == "sk-secret"

    def test_the_default_home_still_comes_from_the_fallback_root(self):
        assert _acp_state_dir(None, "x", {}) == state_dir_for(None, "x")
