"""Tests for the standalone ALCF token helper."""

from __future__ import annotations

import importlib.util
import json
import stat
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "alcf-token.py"


@pytest.fixture
def alcf_token():
    """Load the hyphenated helper as a test module."""
    spec = importlib.util.spec_from_file_location("alcf_token", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _token_data(module, *, access_token="access", refresh_token="refresh", expires=0):
    return {
        "data": {
            "DEFAULT": {
                module.GATEWAY_RESOURCE_SERVER: {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "expires_at_seconds": expires,
                }
            }
        }
    }


def test_write_tokens_uses_private_atomic_file(alcf_token, tmp_path):
    path = tmp_path / "nested" / "tokens.json"

    alcf_token._write_tokens(str(path), {"ok": True})

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text()) == {"ok": True}
    assert not list(path.parent.glob("*.tmp"))


def test_ensure_fresh_returns_existing_token_without_refresh(
    alcf_token, tmp_path, monkeypatch
):
    path = tmp_path / "tokens.json"
    alcf_token._write_tokens(
        str(path), _token_data(alcf_token, access_token="fresh", expires=9999999999)
    )
    monkeypatch.setattr(alcf_token, "_refresh_token", pytest.fail)

    assert alcf_token._ensure_fresh(str(path)) == "fresh"


def test_ensure_fresh_preserves_refresh_token_when_omitted(
    alcf_token, tmp_path, monkeypatch
):
    path = tmp_path / "tokens.json"
    alcf_token._write_tokens(str(path), _token_data(alcf_token))
    monkeypatch.setattr(
        alcf_token,
        "_refresh_token",
        lambda entry: {"access_token": "new-access", "expires_in": 3600},
    )

    assert alcf_token._ensure_fresh(str(path), force=True) == "new-access"
    entry = alcf_token._get_gateway_entry(alcf_token._read_tokens(str(path)))
    assert entry["refresh_token"] == "refresh"
    assert entry["access_token"] == "new-access"


def test_ensure_fresh_returns_existing_token_on_refresh_failure(
    alcf_token, tmp_path, monkeypatch
):
    path = tmp_path / "tokens.json"
    alcf_token._write_tokens(
        str(path), _token_data(alcf_token, access_token="fallback")
    )
    monkeypatch.setattr(
        alcf_token,
        "_refresh_token",
        lambda entry: (_ for _ in ()).throw(
            alcf_token.urllib.error.URLError("offline")
        ),
    )

    assert alcf_token._ensure_fresh(str(path), force=True) == "fallback"


@pytest.mark.parametrize("tokens_dir_exists", [False, True])
def test_status_tokens_dir_requires_json_file(
    alcf_token, tmp_path, monkeypatch, tokens_dir_exists
):
    tokens_dir = tmp_path / "tokens"
    if tokens_dir_exists:
        tokens_dir.mkdir()
    monkeypatch.setattr(
        sys, "argv", ["alcf-token.py", "--status", "--tokens-dir", str(tokens_dir)]
    )

    with pytest.raises(SystemExit) as exc_info:
        alcf_token.main()

    assert exc_info.value.code == 1


def test_collect_tokens_dir_returns_tokens(alcf_token, tmp_path):
    path = tmp_path / "alice.json"
    alcf_token._write_tokens(
        str(path), _token_data(alcf_token, access_token="alice", expires=9999999999)
    )

    assert alcf_token._collect_tokens_dir(str(tmp_path)) == ["alice"]
