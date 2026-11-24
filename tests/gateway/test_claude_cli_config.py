"""Provider-level overrides for a ``claude_cli`` lane.

The shim carries the vendor facts (the command, the caller name, the auth
variables a gateway must not let outrank the CLI's login).  A provider entry
carries deployment facts (another install, an organization gateway, one more
environment name).  These tests pin which side wins and how the two merge.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_rosetta_gateway.config.providers import (
    _cli_args,
    _cli_argv,
    _cli_env,
    _cli_env_unset,
    _cli_upstream,
)


def _shim(**connection) -> SimpleNamespace:
    defaults = {
        "cli_argv": ("claude",),
        "cli_env": {"CLAUDE_CODE_ENTRYPOINT": "llm-rosetta"},
        "cli_env_unset": ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"),
        "cli_args": ("--setting-sources", "user"),
        "cli_upstream": None,
    }
    defaults.update(connection)
    return SimpleNamespace(connection=SimpleNamespace(**defaults))


class TestTheCommand:
    def test_the_shim_supplies_the_default(self):
        assert _cli_argv({}, _shim(), "claude") == ("claude",)

    def test_a_provider_may_point_at_another_install(self):
        cfg = {"cli_argv": ["/opt/claude/bin/claude"]}

        assert _cli_argv(cfg, _shim(), "claude") == ("/opt/claude/bin/claude",)

    def test_no_shim_leaves_the_transport_default(self):
        assert _cli_argv({}, None, "claude") is None

    @pytest.mark.parametrize("bad", [[], ["claude", 7], "claude", {"argv": ["x"]}])
    def test_a_malformed_command_is_refused(self, bad):
        with pytest.raises(ValueError, match="cli_argv must be a non-empty list"):
            _cli_argv({"cli_argv": bad}, _shim(), "claude")


class TestTheArguments:
    def test_the_shim_supplies_the_default(self):
        assert _cli_args({}, _shim(), "claude") == ("--setting-sources", "user")

    def test_the_provider_list_replaces_the_shim_list(self):
        """The flags are one command line; a second flag would silently win."""
        cfg = {"cli_args": ["--setting-sources", ""]}

        assert _cli_args(cfg, _shim(), "claude") == ("--setting-sources", "")

    def test_an_empty_list_clears_the_shim_arguments(self):
        assert _cli_args({"cli_args": []}, _shim(), "claude") == ()

    def test_no_shim_means_no_arguments(self):
        assert _cli_args({}, None, "claude") == ()

    @pytest.mark.parametrize("bad", ["x", ["ok", 7], {"args": ["x"]}])
    def test_a_malformed_list_is_refused(self, bad):
        with pytest.raises(ValueError, match="cli_args must be a list"):
            _cli_args({"cli_args": bad}, _shim(), "claude")


class TestTheEnvironment:
    def test_the_shim_supplies_the_default(self):
        assert _cli_env({}, _shim(), "claude") == {
            "CLAUDE_CODE_ENTRYPOINT": "llm-rosetta"
        }

    def test_the_provider_wins_and_keeps_the_shim_keys(self):
        cfg = {"cli_env": {"CLAUDE_CODE_ENTRYPOINT": "my-lane", "A": "1"}}

        assert _cli_env(cfg, _shim(), "claude") == {
            "CLAUDE_CODE_ENTRYPOINT": "my-lane",
            "A": "1",
        }

    def test_neither_side_declares_one(self):
        assert _cli_env({}, _shim(cli_env=None), "claude") is None
        assert _cli_env({}, None, "claude") is None

    @pytest.mark.parametrize("bad", ["x", ["A=1"], {"A": 7}, {7: "1"}])
    def test_a_malformed_mapping_is_refused(self, bad):
        with pytest.raises(ValueError, match="cli_env must"):
            _cli_env({"cli_env": bad}, _shim(), "claude")


class TestTheUnsetList:
    def test_the_shim_supplies_the_default(self):
        assert _cli_env_unset({}, _shim(), "claude") == (
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_BASE_URL",
        )

    def test_the_provider_adds_names_and_keeps_the_shim_first(self):
        cfg = {"cli_env_unset": ["CLAUDE_CODE_USE_BEDROCK", "ANTHROPIC_API_KEY"]}

        assert _cli_env_unset(cfg, _shim(), "claude") == (
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_BASE_URL",
            "CLAUDE_CODE_USE_BEDROCK",
        )

    def test_no_shim_leaves_the_provider_list(self):
        assert _cli_env_unset({"cli_env_unset": ["X"]}, None, "claude") == ("X",)

    @pytest.mark.parametrize("bad", ["X", ["ok", 7], {"names": ["X"]}])
    def test_a_malformed_list_is_refused(self, bad):
        with pytest.raises(ValueError, match="cli_env_unset must be a list"):
            _cli_env_unset({"cli_env_unset": bad}, _shim(), "claude")


class TestTheUpstream:
    def test_the_shim_default_is_anthropics_api(self):
        assert _cli_upstream({}, _shim(), "claude") is None

    def test_a_provider_may_name_an_organization_gateway(self):
        cfg = {"cli_upstream": "https://rosetta.example"}

        assert _cli_upstream(cfg, _shim(), "claude") == "https://rosetta.example"

    def test_the_provider_wins_over_the_shim(self):
        shim = _shim(cli_upstream="https://shim.example")

        assert _cli_upstream(
            {"cli_upstream": "https://lane.example"}, shim, "claude"
        ) == ("https://lane.example")
        assert _cli_upstream({}, shim, "claude") == "https://shim.example"

    @pytest.mark.parametrize("bad", ["", "   ", 7, ["https://x"]])
    def test_a_malformed_upstream_is_refused(self, bad):
        with pytest.raises(ValueError, match="cli_upstream must be a non-empty"):
            _cli_upstream({"cli_upstream": bad}, _shim(), "claude")
