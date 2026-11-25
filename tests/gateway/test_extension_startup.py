"""Startup points core at the gateway's extensions directory (epic #1703 E6)."""

from __future__ import annotations

from pathlib import Path

from _ext_wheels import build_wheel
from llm_rosetta.api import extensions_dir, get_shim, install_wheel
from llm_rosetta_gateway import cli


def _raw(tmp_path: Path) -> dict:
    return {"providers": {}, "server": {"data_dir": str(tmp_path / "state")}}


def test_startup_chooses_the_data_directorys_extensions_folder(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_ROSETTA_EXTENSIONS_DIR", raising=False)
    seen: list[object] = []
    monkeypatch.setattr(
        cli, "apply_shim_sources", lambda raw, path: seen.append(raw) or "report"
    )

    result = cli._apply_shim_sources_or_exit(_raw(tmp_path), str(tmp_path / "c.jsonc"))

    assert extensions_dir() == tmp_path / "state" / "extensions"
    # Nothing is installed there and no source is configured: no rebuild.
    assert result is None and seen == []


def test_an_extension_already_in_that_folder_is_loaded_without_a_shims_section(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("LLM_ROSETTA_EXTENSIONS_DIR", raising=False)
    target = tmp_path / "state" / "extensions"
    target.mkdir(parents=True)
    install_wheel(build_wheel("acme", "0.1.0", ["acme--chat_completions"]), dest=target)
    assert get_shim("acme--chat_completions") is None

    report = cli._apply_shim_sources_or_exit(_raw(tmp_path), str(tmp_path / "c.jsonc"))

    assert report is not None
    assert get_shim("acme--chat_completions") is not None
