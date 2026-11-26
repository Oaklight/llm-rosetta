"""An extensions directory that cannot be written is reported as that (#1704).

``ext install`` used to say "cannot read <wheel>: Permission denied: <dir>" when
the directory was the problem.  Every command that writes the directory
(``install`` from a file, from an id, ``remove``, ``update``) names the
directory and says what is wrong with it; "cannot read" is kept for a wheel that
really cannot be read.  Four failures differ in their errno and so in their
text: the path is a file, a parent of it is a file, its parent is read-only (so
it cannot be created), and it exists but is not writable.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
from pathlib import Path

import pytest
from _ext_wheels import build_wheel

from llm_rosetta import cli
from llm_rosetta.contracts import IR_VERSION, SURFACE_VERSION
from llm_rosetta.extensions import (
    ExtensionInstallError,
    configure_extensions_dir,
    install_wheel,
)

FIX = "set LLM_ROSETTA_EXTENSIONS_DIR to a writable directory"


@pytest.fixture(autouse=True)
def _no_configured_dir(monkeypatch):
    monkeypatch.delenv("LLM_ROSETTA_EXTENSIONS_DIR", raising=False)
    configure_extensions_dir(None)
    yield
    configure_extensions_dir(None)


def _wheel(version: str = "0.1.0") -> bytes:
    return build_wheel("acme", version, ["acme--chat_completions"])


def _index(root: Path, *versions: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for version in versions:
        data = _wheel(version)
        name = f"llm_rosetta_ext_acme-{version}-py3-none-any.whl"
        (root / name).write_bytes(data)
        entries.append(
            {
                "version": version,
                "url": name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "surface": SURFACE_VERSION,
                "ir": IR_VERSION,
            }
        )
    path = root / "index.json"
    path.write_text(
        json.dumps({"index_version": 1, "extensions": {"acme": entries}}),
        encoding="utf-8",
    )
    return path


def _bad_path(kind: str, tmp_path: Path) -> tuple[Path, str]:
    """A path that cannot be the extensions directory, and what is said of it."""
    if kind == "is a file":
        path = tmp_path / "a-file"
        path.write_text("x")
        return path, "is a file, not a directory"
    if kind == "under a file":
        blocker = tmp_path / "a-file"
        blocker.write_text("x")
        return blocker / "exts", "a parent of it is a file"
    path = tmp_path / "locked" / "exts"
    return path, "Permission denied"


def _deny(monkeypatch, kind: str, path: Path) -> None:
    """Make *path* fail the way a read-only parent or directory does (also as root)."""
    real_mkdir = pathlib.Path.mkdir
    real_open = os.open

    def mkdir(self, *args, **kwargs):
        if kind == "read-only parent" and self == path:
            raise PermissionError(13, "Permission denied", str(self))
        return real_mkdir(self, *args, **kwargs)

    def open_(file, flags, *args, **kwargs):
        if kind == "read-only directory" and str(file).startswith(str(path)):
            raise PermissionError(13, "Permission denied", str(file))
        return real_open(file, flags, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "mkdir", mkdir)
    monkeypatch.setattr(os, "open", open_)


KINDS = ["is a file", "under a file", "read-only parent", "read-only directory"]


def _target(kind: str, tmp_path: Path, monkeypatch) -> tuple[Path, str]:
    if kind in ("is a file", "under a file"):
        return _bad_path(kind, tmp_path)
    path = tmp_path / "exts"
    if kind == "read-only directory":
        path.mkdir()
    _deny(monkeypatch, kind, path)
    return path, "Permission denied"


def _assert_names_the_directory(err: str, path: Path, reason: str) -> None:
    assert f"the extensions directory {path} is not writable" in err, err
    assert reason in err, err
    assert FIX in err, err
    assert "cannot read" not in err, err


@pytest.mark.parametrize("kind", KINDS)
def test_installing_a_wheel_file_names_the_directory(
    kind, tmp_path, monkeypatch, capsys
):
    wheel = tmp_path / "acme.whl"
    wheel.write_bytes(_wheel())
    path, reason = _target(kind, tmp_path, monkeypatch)

    code = cli.main(["ext", "install", str(wheel), "--dir", str(path)])

    assert code == 1
    _assert_names_the_directory(capsys.readouterr().err, path, reason)


@pytest.mark.parametrize("kind", KINDS)
def test_installing_an_id_from_an_index_names_the_directory(
    kind, tmp_path, monkeypatch, capsys
):
    index = _index(tmp_path / "pub", "0.1.0")
    path, reason = _target(kind, tmp_path, monkeypatch)

    code = cli.main(
        ["ext", "install", "acme", "--index", str(index), "--dir", str(path)]
    )

    assert code == 1
    _assert_names_the_directory(capsys.readouterr().err, path, reason)


@pytest.mark.parametrize("kind", ["is a file", "read-only directory"])
def test_removing_names_the_directory(kind, tmp_path, monkeypatch, capsys):
    good = tmp_path / "exts"
    install_wheel(_wheel(), dest=good)
    if kind == "is a file":
        path, reason = _bad_path(kind, tmp_path)
    else:
        path, reason = good, "Permission denied"
        _deny(monkeypatch, kind, path)

    code = cli.main(["ext", "remove", "acme", "--dir", str(path)])

    assert code == 1
    _assert_names_the_directory(capsys.readouterr().err, path, reason)


@pytest.mark.parametrize("kind", ["is a file", "read-only directory"])
def test_updating_names_the_directory(kind, tmp_path, monkeypatch, capsys):
    index = _index(tmp_path / "pub", "0.1.0", "0.2.0")
    good = tmp_path / "exts"
    install_wheel(_wheel("0.1.0"), dest=good)
    if kind == "is a file":
        path, reason = _bad_path(kind, tmp_path)
    else:
        path, reason = good, "Permission denied"
        _deny(monkeypatch, kind, path)

    code = cli.main(
        ["ext", "update", "acme", "--index", str(index), "--dir", str(path)]
    )

    assert code == 1
    _assert_names_the_directory(capsys.readouterr().err, path, reason)


def test_the_library_raises_the_same_message(tmp_path):
    path, reason = _bad_path("is a file", tmp_path)

    with pytest.raises(ExtensionInstallError) as info:
        install_wheel(_wheel(), dest=path)

    _assert_names_the_directory(str(info.value), path, reason)


def test_a_wheel_that_cannot_be_read_is_still_cannot_read(tmp_path, capsys):
    good = tmp_path / "exts"
    missing = tmp_path / "nope.whl"

    code = cli.main(["ext", "install", str(missing), "--dir", str(good)])

    err = capsys.readouterr().err
    assert code == 1
    assert f"cannot read {missing}" in err
    assert "is not writable" not in err
