"""The gateway's install leg reports an unwritable extensions directory (#1704).

The same message the CLI gives, as the source's error, and the failure leg
keeps what #1759's does: the cached index and the versions already installed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from llm_rosetta.api import configure_extensions_dir, get_shim
from llm_rosetta.testing import restore_registry, snapshot_registry
from llm_rosetta_gateway.shims import sources as sources_mod
from llm_rosetta_gateway.shims.shim_sources import apply_shim_sources
from gateway.test_shim_index_sources import INDEX, FakeRelease


@pytest.fixture(autouse=True)
def _clean(tmp_path):
    snapshot = snapshot_registry()
    sources_mod._source_shims.clear()
    extensions = tmp_path / "extensions"
    extensions.mkdir()
    configure_extensions_dir(extensions)
    yield extensions
    configure_extensions_dir(None)
    restore_registry(snapshot)
    sources_mod._source_shims.clear()
    for name in [n for n in sys.modules if n.split(".")[0] == "llm_rosetta_ext"]:
        del sys.modules[name]


def _apply(tmp_path: Path, http: FakeRelease, **kwargs):
    raw = {"shims": {"sources": [INDEX], "install": ["acme"]}}
    return apply_shim_sources(
        raw, str(tmp_path / "config.jsonc"), http_get=http, **kwargs
    )


def _status(report):
    return next(s for s in report.sources if s.vendor == "acme")


def test_an_unwritable_directory_fails_the_upgrade_and_keeps_what_was_installed(
    tmp_path, _clean, monkeypatch
):
    first = FakeRelease("2026-11-26T08:00:00Z")
    first.publish("acme", "0.1.0")
    ok = _status(_apply(tmp_path, first))
    assert ok.ok and ok.release["version"] == "0.1.0"
    cached = list(tmp_path.rglob("index.json"))
    assert cached, "the first run caches the index"

    newer = FakeRelease("2026-11-27T08:00:00Z")
    newer.publish("acme", "0.2.0")
    real_open = os.open

    def open_(file, flags, *args, **kwargs):
        if str(file).startswith(str(_clean)):
            raise PermissionError(13, "Permission denied", str(file))
        return real_open(file, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", open_)

    report = _apply(tmp_path, newer, force_refresh=True)

    status = _status(report)
    error = status.error or ""
    assert f"the extensions directory {_clean} is not writable" in error
    assert "set LLM_ROSETTA_EXTENSIONS_DIR to a writable directory" in error
    assert "cannot read" not in error
    # The upgrade failed, so the source keeps the release it had.
    assert status.ok and status.release["version"] == "0.1.0"
    # What was there stays: the 0.1.0 shim is still registered and the cache
    # that the first run wrote is still on disk.
    assert get_shim("acme--chat_completions") is not None
    assert all(path.is_file() for path in cached)
