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
    build_provider_info,
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

    @pytest.mark.parametrize("bad", ["x", True, ["home"]])
    def test_a_non_mapping_acp_block_is_refused(self, bad):
        """An operator who wrote an ``acp:`` block meant to set a home."""
        with pytest.raises(ValueError, match="acp must be a mapping"):
            _acp_state_dir("/data", "claude", {"acp": bad})

    @pytest.mark.parametrize("home", ["../outside", "saved/../../outside"])
    def test_a_relative_home_may_not_escape_the_data_dir(self, home):
        """A relative home stays on the volume the login has to survive."""
        with pytest.raises(ValueError, match="must stay under the data directory"):
            _acp_state_dir("/tmp/x/data", "claude", {"acp": {"home": home}})

    def test_an_absolute_home_is_the_explicit_escape_hatch(self, tmp_path):
        wanted = str(tmp_path / "outside")
        assert (
            _acp_state_dir("/tmp/x/data", "claude", {"acp": {"home": wanted}}) == wanted
        )

    def test_a_home_that_names_a_file_is_refused(self, tmp_path):
        target = tmp_path / "not-a-directory"
        target.write_text("x", encoding="utf-8")

        with pytest.raises(ValueError, match="acp.home is not a directory"):
            _acp_state_dir("/data", "claude", {"acp": {"home": str(target)}})


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

    def test_the_published_home_is_gateway_owned(self):
        """An operator moves the profile with ``acp.home``, never this map."""
        env = _acp_env(
            {"acp_env": {"LLM_ROSETTA_ACP_HOME": "/mine"}},
            None,
            "claude",
            "/home/lane",
            inject_home=True,
        )

        assert env == {"LLM_ROSETTA_ACP_HOME": "/home/lane"}

    def test_an_http_lane_ignores_the_acp_home(self):
        """A lane that starts no agent has no profile to point at."""
        env = _acp_env(
            {"acp_env": {"PLACEHOLDER": "{acp_home}/x"}},
            None,
            "openai",
            None,
        )

        assert env == {"PLACEHOLDER": "{acp_home}/x"}

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


class TestTheBuilderWiresTheExchange:
    def test_the_sign_in_driver_receives_the_agents_environment(self, tmp_path):
        """The admin sign-in must write the profile the agent will read."""
        from llm_rosetta.contracts.shim import ConnectionConfig, ProviderShim
        from llm_rosetta.shims.provider_shim import register_shim, unregister_shim

        seen: dict[str, str] = {}

        class Exchange:
            kind = "vendor-cli"
            consumes_key = False
            vendor_managed = True

            def needs_interactive_setup(self, ctx):
                return True

            async def credential(self, ctx):
                raise AssertionError("a vendor-managed exchange mints nothing")

            def account_id(self, ctx):
                seen.update(ctx.config.get("acp_env") or {})
                return None

        name = "__test_acp_home_exchange__"
        register_shim(
            ProviderShim(
                name=name,
                format="acp",
                connection=ConnectionConfig(
                    base_url="http://agent.local",
                    transport="acp",
                    keyless=True,
                    acp_env={"CLAUDE_CONFIG_DIR": "{acp_home}/claude"},
                ),
                auth=Exchange(),
            )
        )
        try:
            info = build_provider_info(
                "acp",
                {"acp_env": {"LLM_ROSETTA_ACP_HOME": "/elsewhere"}},
                shim_name=name,
                config_name="claude",
                data_dir=str(tmp_path),
            )
        finally:
            unregister_shim(name)

        assert info.auth_context is not None
        assert seen == info.acp_env
        assert seen["LLM_ROSETTA_ACP_HOME"] == info.acp_state_dir
        assert seen["CLAUDE_CONFIG_DIR"] == f"{info.acp_state_dir}/claude"
