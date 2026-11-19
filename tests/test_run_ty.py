"""Tests for scripts/dev/run_ty.py (ty itself is not run)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "run_ty", Path(__file__).parent.parent / "scripts" / "dev" / "run_ty.py"
)
assert _SPEC is not None and _SPEC.loader is not None
rt = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rt)


@pytest.fixture
def fake_repo(tmp_path):
    for sp in rt.SEARCH_PATHS:
        (tmp_path / sp).mkdir(parents=True)
    sub = tmp_path / "tests" / "deep"
    sub.mkdir(parents=True)
    return tmp_path, sub


def test_find_repo_root_walks_up(fake_repo):
    root, sub = fake_repo
    assert rt.find_repo_root(sub) == root


def test_find_repo_root_none(tmp_path):
    assert rt.find_repo_root(tmp_path) is None


def test_pick_python_precedence(monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", "/v")
    monkeypatch.setenv("CONDA_PREFIX", "/c")
    assert rt.pick_python() == "/v"
    monkeypatch.delenv("VIRTUAL_ENV")
    assert rt.pick_python() == "/c"
    monkeypatch.delenv("CONDA_PREFIX")
    assert rt.pick_python() == rt.sys.prefix


def test_command_puts_checkout_sources_first(fake_repo, monkeypatch):
    root, sub = fake_repo
    seen = {}

    def fake_call(cmd, cwd):
        seen["cmd"], seen["cwd"] = cmd, cwd
        return 7

    monkeypatch.chdir(sub)
    monkeypatch.setenv("VIRTUAL_ENV", "/v")
    monkeypatch.setattr(rt, "find_ty", lambda: ["ty"])
    monkeypatch.setattr(rt.subprocess, "call", fake_call)
    assert rt.main(["--output-format", "concise"]) == 7
    cmd = seen["cmd"]
    assert cmd[:4] == ["ty", "check", "--python", "/v"]
    for sp in rt.SEARCH_PATHS:
        i = cmd.index(sp)
        assert cmd[i - 1] == "--extra-search-path"
    assert cmd[-2:] == ["--output-format", "concise"]
    assert seen["cwd"] == root


def test_missing_ty_is_an_error(fake_repo, monkeypatch, capsys):
    _, sub = fake_repo
    monkeypatch.chdir(sub)
    monkeypatch.setattr(rt, "find_ty", lambda: None)
    assert rt.main([]) == 2
    assert "not installed" in capsys.readouterr().err


def test_main_warns_when_no_environment_is_active(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr(rt, "find_repo_root", lambda start: tmp_path)
    monkeypatch.setattr(rt, "find_ty", lambda: ["ty"])
    monkeypatch.setattr(rt.subprocess, "call", lambda cmd, cwd=None: 0)
    assert rt.main([]) == 0
    assert "no virtualenv or conda env is active" in capsys.readouterr().err
