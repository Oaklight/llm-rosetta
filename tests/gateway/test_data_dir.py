"""Tests for data_dir resolution order: CLI > config > default."""

from __future__ import annotations

import argparse
import json
import os
from llm_rosetta.gateway.cli import _resolve_data_dir


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

    def test_cli_flag_wins(self, tmp_path):
        config_path = str(tmp_path / "config.jsonc")
        _write_config(config_path, {"data_dir": "/from-config"})
        args = argparse.Namespace(data_dir="/from-cli")
        assert _resolve_data_dir(config_path, args) == "/from-cli"

    def test_config_absolute(self, tmp_path):
        config_path = str(tmp_path / "config.jsonc")
        _write_config(config_path, {"data_dir": "/absolute/data"})
        args = argparse.Namespace(data_dir=None)
        assert _resolve_data_dir(config_path, args) == "/absolute/data"

    def test_config_relative_resolved_against_config_dir(self, tmp_path):
        config_path = str(tmp_path / "config.jsonc")
        _write_config(config_path, {"data_dir": "my-data"})
        args = argparse.Namespace(data_dir=None)
        expected = str(tmp_path / "my-data")
        assert _resolve_data_dir(config_path, args) == expected

    def test_config_relative_dot_prefix(self, tmp_path):
        config_path = str(tmp_path / "config.jsonc")
        _write_config(config_path, {"data_dir": "./subdir"})
        args = argparse.Namespace(data_dir=None)
        expected = str(tmp_path / "./subdir")
        assert os.path.normpath(
            _resolve_data_dir(config_path, args)
        ) == os.path.normpath(expected)

    def test_default_fallback(self, tmp_path):
        config_path = str(tmp_path / "config.jsonc")
        _write_config(config_path)
        args = argparse.Namespace(data_dir=None)
        expected = str(tmp_path / "data")
        assert _resolve_data_dir(config_path, args) == expected

    def test_no_data_dir_attr_on_args(self, tmp_path):
        config_path = str(tmp_path / "config.jsonc")
        _write_config(config_path)
        args = argparse.Namespace()  # no data_dir attr at all (SUPPRESS)
        expected = str(tmp_path / "data")
        assert _resolve_data_dir(config_path, args) == expected


class TestSetupAdminDataDir:
    """setup_admin data_dir resolution mirrors _resolve_data_dir."""

    def test_explicit_data_dir_wins(self, tmp_path):
        from llm_rosetta.gateway.config import GatewayConfig

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
        from llm_rosetta.gateway.admin import setup_admin

        setup_admin(app, config, config_path, data_dir=explicit_dir)
        assert app.persistence is not None
        assert "explicit" in str(app.persistence._data_dir)

    def test_config_relative_resolved(self, tmp_path):
        from llm_rosetta.gateway.config import GatewayConfig

        raw = {
            "providers": {"t": {"api_key": "k", "base_url": "http://x"}},
            "models": {"m": "t"},
            "server": {"data_dir": "my-data"},
        }
        config = GatewayConfig(raw)
        config_path = str(tmp_path / "config.jsonc")
        _write_config(config_path)

        app = _MockApp()
        from llm_rosetta.gateway.admin import setup_admin

        setup_admin(app, config, config_path)
        assert app.persistence is not None
        assert os.path.normpath(app.persistence._data_dir) == os.path.normpath(
            str(tmp_path / "my-data")
        )


class TestArgparseSubparserClobber:
    """Verify --data-dir on parent isn't clobbered by db subparser."""

    def test_parent_data_dir_survives_subcommand(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--data-dir", default=None)
        sub = parser.add_subparsers(dest="command")
        db_parser = sub.add_parser("db")
        db_parser.add_argument("--data-dir", default=argparse.SUPPRESS)
        db_sub = db_parser.add_subparsers(dest="db_type")
        db_sub.add_parser("cleanup")

        args = parser.parse_args(["--data-dir", "/foo", "db", "cleanup"])
        assert args.data_dir == "/foo"

    def test_subparser_data_dir_explicit(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--data-dir", default=None)
        sub = parser.add_subparsers(dest="command")
        db_parser = sub.add_parser("db")
        db_parser.add_argument("--data-dir", default=argparse.SUPPRESS)
        db_sub = db_parser.add_subparsers(dest="db_type")
        db_sub.add_parser("cleanup")

        args = parser.parse_args(["db", "--data-dir", "/bar", "cleanup"])
        assert args.data_dir == "/bar"

    def test_neither_level_sets_data_dir(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--data-dir", default=None)
        sub = parser.add_subparsers(dest="command")
        db_parser = sub.add_parser("db")
        db_parser.add_argument("--data-dir", default=argparse.SUPPRESS)
        db_sub = db_parser.add_subparsers(dest="db_type")
        db_sub.add_parser("cleanup")

        args = parser.parse_args(["db", "cleanup"])
        assert args.data_dir is None
