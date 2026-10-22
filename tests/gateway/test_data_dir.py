"""Tests for data_dir resolution order: CLI > config > default."""

from __future__ import annotations

import pytest

import argparse
import json
import os
from llm_rosetta_gateway.cli import _resolve_data_dir


from typing import Any


class _MockApp:
    persistence: Any = None


def _write_config(path: str, server: dict | None = None) -> None:
    cfg = {
        "providers": {"test": {"api_key": "k", "base_url": "http://x"}},
        "models": {"m": "test"},
        "server": server or {},
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f)


class TestResolveDataDir:
    """_resolve_data_dir: CLI --data-dir > config server.data_dir > default."""

    @pytest.mark.asyncio
    async def test_cli_flag_wins(self, tmp_path):
        config_path = str(tmp_path / "config.jsonc")
        _write_config(config_path, {"data_dir": "/from-config"})
        args = argparse.Namespace(data_dir="/from-cli")
        assert _resolve_data_dir(config_path, args) == "/from-cli"

    @pytest.mark.asyncio
    async def test_config_absolute(self, tmp_path):
        config_path = str(tmp_path / "config.jsonc")
        _write_config(config_path, {"data_dir": "/absolute/data"})
        args = argparse.Namespace(data_dir=None)
        assert _resolve_data_dir(config_path, args) == "/absolute/data"

    @pytest.mark.asyncio
    async def test_config_relative_resolved_against_config_dir(self, tmp_path):
        config_path = str(tmp_path / "config.jsonc")
        _write_config(config_path, {"data_dir": "my-data"})
        args = argparse.Namespace(data_dir=None)
        expected = str(tmp_path / "my-data")
        assert _resolve_data_dir(config_path, args) == expected

    @pytest.mark.asyncio
    async def test_config_relative_dot_prefix(self, tmp_path):
        config_path = str(tmp_path / "config.jsonc")
        _write_config(config_path, {"data_dir": "./subdir"})
        args = argparse.Namespace(data_dir=None)
        expected = str(tmp_path / "./subdir")
        assert os.path.normpath(
            _resolve_data_dir(config_path, args)
        ) == os.path.normpath(expected)

    @pytest.mark.asyncio
    async def test_default_fallback(self, tmp_path):
        config_path = str(tmp_path / "config.jsonc")
        _write_config(config_path)
        args = argparse.Namespace(data_dir=None)
        expected = str(tmp_path / "data")
        assert _resolve_data_dir(config_path, args) == expected

    @pytest.mark.asyncio
    async def test_no_data_dir_attr_on_args(self, tmp_path):
        config_path = str(tmp_path / "config.jsonc")
        _write_config(config_path)
        args = argparse.Namespace()  # no data_dir attr at all (SUPPRESS)
        expected = str(tmp_path / "data")
        assert _resolve_data_dir(config_path, args) == expected


class TestSetupAdminDataDir:
    """setup_admin data_dir resolution mirrors _resolve_data_dir."""

    @pytest.mark.asyncio
    async def test_explicit_data_dir_wins(self, tmp_path):
        from llm_rosetta_gateway.config import GatewayConfig

        explicit_dir = str(tmp_path / "explicit")
        raw = {
            "providers": {"t": {"api_key": "k", "base_url": "http://x"}},
            "models": {"m": "t"},
            "server": {"data_dir": str(tmp_path / "from-config")},
        }
        config = GatewayConfig(raw)
        config_path = str(tmp_path / "config.jsonc")
        _write_config(config_path)

        app = _MockApp()
        from llm_rosetta_gateway.admin import setup_admin

        await setup_admin(app, config, config_path, data_dir=explicit_dir)
        assert app.persistence is not None
        assert "explicit" in str(app.persistence._data_dir)

    @pytest.mark.asyncio
    async def test_config_relative_resolved(self, tmp_path):
        from llm_rosetta_gateway.config import GatewayConfig

        raw = {
            "providers": {"t": {"api_key": "k", "base_url": "http://x"}},
            "models": {"m": "t"},
            "server": {"data_dir": "my-data"},
        }
        config = GatewayConfig(raw)
        config_path = str(tmp_path / "config.jsonc")
        _write_config(config_path)

        app = _MockApp()
        from llm_rosetta_gateway.admin import setup_admin

        await setup_admin(app, config, config_path)
        assert app.persistence is not None
        assert os.path.normpath(app.persistence._data_dir) == os.path.normpath(
            str(tmp_path / "my-data")
        )


class TestArgparseSubparserClobber:
    """Verify --data-dir on parent isn't clobbered by db subparser."""

    @pytest.mark.asyncio
    async def test_parent_data_dir_survives_subcommand(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--data-dir", default=None)
        sub = parser.add_subparsers(dest="command")
        db_parser = sub.add_parser("db")
        db_parser.add_argument("--data-dir", default=argparse.SUPPRESS)
        db_sub = db_parser.add_subparsers(dest="db_type")
        db_sub.add_parser("cleanup")

        args = parser.parse_args(["--data-dir", "/foo", "db", "cleanup"])
        assert args.data_dir == "/foo"

    @pytest.mark.asyncio
    async def test_subparser_data_dir_explicit(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--data-dir", default=None)
        sub = parser.add_subparsers(dest="command")
        db_parser = sub.add_parser("db")
        db_parser.add_argument("--data-dir", default=argparse.SUPPRESS)
        db_sub = db_parser.add_subparsers(dest="db_type")
        db_sub.add_parser("cleanup")

        args = parser.parse_args(["db", "--data-dir", "/bar", "cleanup"])
        assert args.data_dir == "/bar"

    @pytest.mark.asyncio
    async def test_neither_level_sets_data_dir(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--data-dir", default=None)
        sub = parser.add_subparsers(dest="command")
        db_parser = sub.add_parser("db")
        db_parser.add_argument("--data-dir", default=argparse.SUPPRESS)
        db_sub = db_parser.add_subparsers(dest="db_type")
        db_sub.add_parser("cleanup")

        args = parser.parse_args(["db", "cleanup"])
        assert args.data_dir is None


class TestKeysDbDataDir:
    """keys.db should default to data_dir when available."""

    def _make_config(self, tmp_path, server_extra=None):
        from llm_rosetta_gateway.config import GatewayConfig

        config_path = str(tmp_path / "config.jsonc")
        server = {}
        if server_extra:
            server.update(server_extra)
        raw = {
            "providers": {"t": {"api_key": "k", "base_url": "http://x"}},
            "models": {"m": "t"},
            "server": server,
        }
        _write_config(config_path, server)
        config = GatewayConfig(raw)
        return config, config_path

    @pytest.mark.asyncio
    async def test_keys_db_defaults_to_data_dir(self, tmp_path):
        """Fresh start, no existing keys.db → created in data_dir."""
        from llm_rosetta_gateway.app import _setup_auth

        config, config_path = self._make_config(tmp_path)
        data_dir = str(tmp_path / "data")
        os.makedirs(data_dir, exist_ok=True)

        _, keystore, _ = await _setup_auth(config, config_path, data_dir=data_dir)
        try:
            assert str(keystore._db_path) == os.path.join(data_dir, "keys.db")
        finally:
            await keystore.close()

    @pytest.mark.asyncio
    async def test_keys_db_legacy_fallback(self, tmp_path):
        """keys.db exists at old config-sibling location → uses old path."""
        import sqlite3

        from llm_rosetta_gateway.app import _setup_auth

        config, config_path = self._make_config(tmp_path)
        data_dir = str(tmp_path / "data")
        os.makedirs(data_dir, exist_ok=True)

        # Create a legacy keys.db next to config
        old_path = str(tmp_path / "keys.db")
        conn = sqlite3.connect(old_path)
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.close()

        _, keystore, _ = await _setup_auth(config, config_path, data_dir=data_dir)
        try:
            assert str(keystore._db_path) == old_path
        finally:
            await keystore.close()

    @pytest.mark.asyncio
    async def test_keys_db_explicit_api_keys_db_wins(self, tmp_path):
        """config.api_keys_db set → uses that, ignores data_dir."""
        from llm_rosetta_gateway.app import _setup_auth
        from llm_rosetta_gateway.config import GatewayConfig

        explicit = str(tmp_path / "custom" / "my-keys.db")
        raw = {
            "providers": {"t": {"api_key": "k", "base_url": "http://x"}},
            "models": {"m": "t"},
            "server": {"api_keys_db": explicit},
        }
        config = GatewayConfig(raw)
        config_path = str(tmp_path / "config.jsonc")
        _write_config(config_path)

        data_dir = str(tmp_path / "data")
        _, keystore, _ = await _setup_auth(config, config_path, data_dir=data_dir)
        try:
            assert str(keystore._db_path) == explicit
        finally:
            await keystore.close()

    @pytest.mark.asyncio
    async def test_keys_db_new_location_preferred(self, tmp_path):
        """keys.db exists in both old and data_dir → uses data_dir."""
        import sqlite3

        from llm_rosetta_gateway.app import _setup_auth

        config, config_path = self._make_config(tmp_path)
        data_dir = str(tmp_path / "data")
        os.makedirs(data_dir, exist_ok=True)

        # Create keys.db in both locations
        for path in [str(tmp_path / "keys.db"), os.path.join(data_dir, "keys.db")]:
            conn = sqlite3.connect(path)
            conn.execute("CREATE TABLE test (id INTEGER)")
            conn.close()

        _, keystore, _ = await _setup_auth(config, config_path, data_dir=data_dir)
        try:
            assert str(keystore._db_path) == os.path.join(data_dir, "keys.db")
        finally:
            await keystore.close()

    @pytest.mark.asyncio
    async def test_resolve_data_dir_for_app(self, tmp_path):
        """_resolve_data_dir_for_app mirrors _resolve_data_dir behavior."""
        from llm_rosetta_gateway.app import _resolve_data_dir_for_app
        from llm_rosetta_gateway.config import GatewayConfig

        config_path = str(tmp_path / "config.jsonc")
        _write_config(config_path)

        # No data_dir in config → default
        raw = {
            "providers": {"t": {"api_key": "k", "base_url": "http://x"}},
            "models": {"m": "t"},
            "server": {},
        }
        config = GatewayConfig(raw)
        result = _resolve_data_dir_for_app(config, config_path)
        assert result == str(tmp_path / "data")

        # Relative data_dir → resolved against config dir
        raw["server"]["data_dir"] = "my-data"
        config = GatewayConfig(raw)
        result = _resolve_data_dir_for_app(config, config_path)
        assert result == str(tmp_path / "my-data")

        # Absolute data_dir → used as-is
        raw["server"]["data_dir"] = "/absolute/path"
        config = GatewayConfig(raw)
        result = _resolve_data_dir_for_app(config, config_path)
        assert result == "/absolute/path"


class TestTheExchangeRecordDirectory:
    """A shim that mints its credential keeps its record under the data dir.

    The directory is resolved while the config is built — before the app
    factory knows where the config file lives — so ``server.data_dir`` has to
    be anchored at that point, not later.  A relative value resolves beside
    the config file, which is what the persistence layer has always done.
    """

    @pytest.fixture()
    def shim(self):
        from llm_rosetta.shims.provider_shim import (
            ConnectionConfig,
            ProviderShim,
            register_shim,
            unregister_shim,
        )

        class Exchange:
            kind = "oauth"
            consumes_key = False

            async def credential(self, ctx):
                return "access-token"

            def account_id(self, ctx):
                return ctx.provider

            def needs_interactive_setup(self, ctx):
                return False

        register_shim(
            ProviderShim(
                name="__test_data_dir__",
                format="chat_completions",
                connection=ConnectionConfig(base_url="https://vendor.example/v1"),
                auth=Exchange(),
            )
        )
        yield "__test_data_dir__"
        unregister_shim("__test_data_dir__")

    def _config(self, shim: str, server: dict | None = None, **kw) -> Any:
        from llm_rosetta_gateway.config import GatewayConfig

        return GatewayConfig(
            {
                "providers": {"team-a": {"type": shim}},
                "models": {},
                "server": server or {},
            },
            **kw,
        )

    def test_a_relative_data_dir_resolves_beside_the_config(self, shim, tmp_path):
        config_path = str(tmp_path / "gateway.jsonc")

        config = self._config(shim, {"data_dir": "my-data"}, config_path=config_path)

        assert config.data_dir == str(tmp_path / "my-data")
        assert config.providers["team-a"].auth_state_dir == str(
            tmp_path / "my-data" / "shim-auth" / "team-a"
        )

    def test_the_default_data_dir_is_used(self, shim, tmp_path):
        config_path = str(tmp_path / "gateway.jsonc")

        config = self._config(shim, config_path=config_path)

        assert config.providers["team-a"].auth_state_dir == str(
            tmp_path / "data" / "shim-auth" / "team-a"
        )

    def test_without_a_config_path_the_shim_chooses(self, shim):
        config = self._config(shim)

        assert config.data_dir is None
        assert config.providers["team-a"].auth_state_dir is None

    def test_an_absolute_value_is_left_alone(self, shim, tmp_path):
        config_path = str(tmp_path / "gateway.jsonc")

        config = self._config(shim, {"data_dir": "/srv/data"}, config_path=config_path)

        assert config.providers["team-a"].auth_state_dir == "/srv/data/shim-auth/team-a"

    def test_the_reader_carries_the_path_it_read_from(self, shim, tmp_path):
        """The shipped path: the CLI's reader, not a hand-built config."""
        from llm_rosetta_gateway.config import load_gateway_config

        path = tmp_path / "gateway.jsonc"
        path.write_text(
            json.dumps(
                {
                    "providers": {"team-a": {"type": shim}},
                    "models": {},
                    "server": {},
                }
            )
        )

        config = load_gateway_config(str(path))

        assert config.providers["team-a"].auth_state_dir == str(
            tmp_path / "data" / "shim-auth" / "team-a"
        )
