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


class TestTheInstallAndTheProfile:
    """The command names an install; ``acp.home`` names the profile."""

    def _lane(self, tmp_path, **cfg):
        config = {"base_url": "http://agent.local", "transport": "acp"}
        config.update(cfg)
        return build_provider_info(
            "acp", config, config_name="agy", data_dir=str(tmp_path)
        )

    def test_acp_argv_names_the_install_and_acp_home_the_profile(self, tmp_path):
        info = self._lane(
            tmp_path,
            acp={"argv": ["/opt/agy-1.7/bin/agy", "acp"], "home": "saved"},
        )

        assert info.acp_argv == ("/opt/agy-1.7/bin/agy", "acp")
        assert info.acp_state_dir == str(tmp_path / "saved")
        assert info.acp_env["LLM_ROSETTA_ACP_HOME"] == info.acp_state_dir

    def test_the_top_level_acp_argv_is_still_an_alias(self, tmp_path):
        info = self._lane(tmp_path, acp_argv=["agy", "acp"])

        assert info.acp_argv == ("agy", "acp")
        assert info.acp_argv_custom is True

    def test_acp_argv_wins_over_the_alias(self, tmp_path):
        info = self._lane(
            tmp_path,
            acp={"argv": ["/new/agy", "acp"]},
            acp_argv=["/old/agy", "acp"],
        )

        assert info.acp_argv == ("/new/agy", "acp")

    def test_two_installs_share_one_profile(self, tmp_path):
        """A version switch moves the command, never the login store."""
        one = self._lane(
            tmp_path,
            acp={"argv": ["/opt/agy-1.6/bin/agy", "acp"], "home": "saved"},
        )
        two = self._lane(
            tmp_path,
            acp={"argv": ["/opt/agy-1.7/bin/agy", "acp"], "home": "saved"},
        )

        assert one.acp_state_dir == two.acp_state_dir
        assert one.acp_env == two.acp_env

    def test_a_gemini_dir_in_the_command_is_the_home(self, tmp_path):
        """The profile the transport writes into is the one the sign-in names."""
        profile = tmp_path / "profile"
        info = self._lane(
            tmp_path, acp={"argv": ["agy", "acp", "--gemini-dir", str(profile)]}
        )

        assert info.acp_state_dir == str(profile)
        assert info.acp_env["LLM_ROSETTA_ACP_HOME"] == str(profile)

    def test_a_gemini_dir_in_a_wrapper_is_the_home(self, tmp_path):
        """A wrapper that reads a named profile keeps it as the lane's home."""
        profile = tmp_path / "profile"
        info = self._lane(
            tmp_path,
            acp={
                "argv": ["/opt/run-agy.sh", "acp", "--gemini-dir", str(profile)],
                "harden": False,
            },
        )

        assert info.acp_state_dir == str(profile)
        assert info.acp_env["LLM_ROSETTA_ACP_HOME"] == str(profile)

    def test_a_relative_gemini_dir_is_refused(self, tmp_path):
        """``agy`` resolves a relative path against its own cwd, not ours."""
        with pytest.raises(ValueError, match="must be an absolute path"):
            self._lane(
                tmp_path, acp={"argv": ["agy", "acp", "--gemini-dir", "profile"]}
            )

    def test_a_conflicting_home_and_gemini_dir_are_refused(self, tmp_path):
        with pytest.raises(ValueError, match="name different directories"):
            self._lane(
                tmp_path,
                acp={
                    "argv": ["agy", "acp", "--gemini-dir", str(tmp_path / "a")],
                    "home": "b",
                },
            )

    def test_harden_follows_the_executable_name_by_default(self, tmp_path):
        assert self._lane(tmp_path).acp_harden is True
        named = self._lane(tmp_path, acp={"argv": ["/opt/agy-1.7/bin/agy", "acp"]})
        assert named.acp_harden is True
        wrapper = self._lane(tmp_path, acp={"argv": ["/opt/run-agy.sh", "acp"]})
        assert wrapper.acp_harden is False

    def test_harden_true_forces_the_profile_on_a_renamed_command(self, tmp_path):
        info = self._lane(
            tmp_path, acp={"argv": ["/opt/agy-1.7", "acp"], "harden": True}
        )

        assert info.acp_harden is True

    def test_harden_false_is_the_opt_out(self, tmp_path):
        profile = tmp_path / "profile"
        info = self._lane(
            tmp_path,
            acp={
                "argv": ["agy", "acp", "--gemini-dir", str(profile)],
                "harden": False,
            },
        )

        assert info.acp_harden is False
        assert info.acp_state_dir == str(profile)

    def test_harden_false_needs_a_profile_the_agent_will_read(self, tmp_path):
        with pytest.raises(ValueError, match="runs agy as written"):
            self._lane(tmp_path, acp={"argv": ["agy", "acp"], "harden": False})

    @pytest.mark.parametrize(
        "argv",
        [
            ["agy", "acp", "--gemini-dir="],
            ["agy", "acp", "--gemini-dir"],
            ["agy", "acp", "--gemini-dir", "/a", "--gemini-dir", "/b"],
        ],
    )
    def test_a_malformed_gemini_dir_is_refused(self, tmp_path, argv):
        with pytest.raises(ValueError, match="gemini-dir"):
            self._lane(tmp_path, acp={"argv": argv})

    def test_the_underscore_alias_names_the_same_profile(self, tmp_path):
        profile = tmp_path / "profile"
        info = self._lane(
            tmp_path, acp={"argv": ["agy", "acp", "--gemini_dir", str(profile)]}
        )

        assert info.acp_state_dir == str(profile)

    def test_a_non_boolean_harden_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="acp.harden must be true or false"):
            self._lane(tmp_path, acp={"harden": "no"})

    @pytest.mark.parametrize("bad", ["agy acp", [], ["agy", ""], [1, "acp"]])
    def test_a_malformed_command_is_refused(self, tmp_path, bad):
        with pytest.raises(ValueError, match="must be a non-empty list of strings"):
            self._lane(tmp_path, acp={"argv": bad})


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

    def test_a_cli_lane_wires_its_own_login_profile_into_the_exchange(
        self, tmp_path, monkeypatch
    ):
        """The direct CLI lane must not read the gateway process's ~/.claude."""
        from llm_rosetta.contracts.shim import ConnectionConfig, ProviderShim
        from llm_rosetta.shims.provider_shim import register_shim, unregister_shim
        from llm_rosetta_gateway.transport import registry

        # The lane id belongs to an installed extension's transport; this test
        # exercises profile wiring, not scope resolution.
        monkeypatch.setattr(
            registry,
            "resolve_lane",
            lambda name, shim_name=None: registry.LaneRef(name, "claude"),
        )

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
                seen.update(ctx.config.get("transport_options", {}).get("env") or {})
                return None

        name = "__test_cli_home_exchange__"
        register_shim(
            ProviderShim(
                name=name,
                format="messages",
                connection=ConnectionConfig(
                    base_url="http://claude.local",
                    transport="claude_cli",
                    keyless=True,
                    transport_options={
                        "env": {"CLAUDE_CONFIG_DIR": "{acp_home}/claude"}
                    },
                ),
                auth=Exchange(),
            )
        )
        try:
            info = build_provider_info(
                "messages",
                {},
                shim_name=name,
                config_name="claude",
                data_dir=str(tmp_path),
            )
        finally:
            unregister_shim(name)

        assert info.auth_context is not None
        assert seen == info.transport_options["env"]
        assert seen["CLAUDE_CONFIG_DIR"] == f"{info.acp_state_dir}/claude"


class TestTheLaneTransport:
    def test_an_http_lane_ignores_the_acp_home(self):
        """``acp.home`` describes a profile only an ACP lane opens."""
        info = build_provider_info(
            "chat_completions",
            {
                "base_url": "https://x/v1",
                "acp": {"home": "saved"},
                "acp_env": {"P": "{acp_home}/x"},
            },
            config_name="http-lane",
            data_dir="/data",
        )

        assert info.acp_state_dir == "/data/acp/http-lane"
        assert info.acp_env == {"P": "{acp_home}/x"}
