"""Tests for declared shim manifests and the scan they bound (#1557).

Two layers are pinned separately: the manifest reader (what a package may
declare) and the scan (what a source actually loads).  The ``shims.include`` /
``shims.exclude`` selection that once narrowed a scan was removed in 0.15.0;
``tests/test_shim_section.py`` pins its refusal.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from llm_rosetta.api import (
    load_providers_from_dir,
    restore_registry,
    snapshot_registry,
    staging_registry,
)
from llm_rosetta_gateway.shims import sources as shim_sources
from llm_rosetta_gateway.shims.manifest import (
    MANIFEST_FILENAME,
    ShimManifestError,
    read_shim_manifest,
)
from llm_rosetta_gateway.shims._scan import _scan_dir


@pytest.fixture(autouse=True)
def _clean_registry():
    """Start and end every test with no source shims registered."""
    snapshot = snapshot_registry()
    with staging_registry():
        pass
    yield
    restore_registry(snapshot)


def _leaf(root: Path, slug: str, name: str | None = None) -> Path:
    """Write one chat shim under ``root/<slug>`` and return its directory."""
    name = name or f"{slug}--chat_completions"
    directory = root / slug
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "provider.yaml").write_text(
        f"name: {name}\n"
        "format: chat_completions\n"
        "model_type: llm\n"
        "connection:\n"
        f"  base_url: https://{slug}.example/v1\n"
        "  api_key_env: ACME_API_KEY\n",
        encoding="utf-8",
    )
    return directory


def _manifest(root: Path, body: str) -> None:
    (root / MANIFEST_FILENAME).write_text(body, encoding="utf-8")


def _three_shims(root: Path) -> None:
    for slug in ("a", "b", "c"):
        _leaf(root, slug)


MANIFEST = """version: 1
shims:
  - name: a--chat_completions
    path: a
  - name: b--chat_completions
    path: b
  - name: c--chat_completions
    path: c
"""


class _Capture(logging.Handler):
    """Collect the gateway logger's messages; it does not propagate."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def captured():
    """Collect gateway warnings; the two modules log under two names.

    ``shims/sources.py`` uses its module logger and ``shim_sources.py`` uses the
    package one, so a handler on either alone would miss half the warnings.
    """
    handler = _Capture()
    loggers = [
        shim_sources.logger,
        logging.getLogger("llm-rosetta-gateway"),
    ]
    for logger in loggers:
        logger.addHandler(handler)
    try:
        yield handler.messages
    finally:
        for logger in loggers:
            logger.removeHandler(handler)


class TestManifestReader:
    """What a package is allowed to declare."""

    def test_absent_returns_none(self, tmp_path: Path):
        assert read_shim_manifest(tmp_path) is None

    def test_declared_entries_keep_their_order(self, tmp_path: Path):
        _three_shims(tmp_path)
        _manifest(tmp_path, MANIFEST)

        manifest = read_shim_manifest(tmp_path)

        assert manifest is not None
        assert manifest.names == (
            "a--chat_completions",
            "b--chat_completions",
            "c--chat_completions",
        )
        assert [entry.path for entry in manifest.entries] == ["a", "b", "c"]

    def test_an_empty_list_declares_nothing(self, tmp_path: Path):
        """A package may ship a manifest that selects none of its own leaves."""
        _three_shims(tmp_path)
        _manifest(tmp_path, "version: 1\nshims: []\n")

        manifest = read_shim_manifest(tmp_path)

        assert manifest is not None
        assert manifest.names == ()

    @pytest.mark.parametrize(
        "body",
        [
            "version: 2\nshims: []\n",
            "- not a mapping\n",
            "version: 1\nshims:\n  - name: a--chat_completions\n  - name: a--chat_completions\n",
            "version: 1\nshims:\n  - name: x\n    path: ../outside\n",
            "version: 1\nshims:\n  - name: x\n    path: /abs\n",
            "version: 1\nshims:\n  - name: x\n    path: https://example.com/x\n",
            "version: 1\nshims:\n  - name: x\n    path: missing\n",
            "version: 1\nshims:\n  - path: a\n",
            "version: 1\nshims:\n  - name: x\n",
            "version: 1\nshims: not-a-list\n",
        ],
    )
    def test_an_unusable_manifest_is_refused(self, tmp_path: Path, body: str):
        _leaf(tmp_path, "a")
        _manifest(tmp_path, body)

        with pytest.raises(ShimManifestError):
            read_shim_manifest(tmp_path)


class TestScanSelection:
    """What a selection actually loads."""

    def test_a_full_load_follows_the_manifest(self, tmp_path: Path):
        _three_shims(tmp_path)
        _manifest(tmp_path, MANIFEST)

        status = _scan_dir(tmp_path, kind="path", location="src")

        assert status.ok
        assert status.shim_names == (
            "a--chat_completions",
            "b--chat_completions",
            "c--chat_completions",
        )
        assert status.manifest_shims == status.shim_names

    def test_a_directory_outside_the_manifest_is_never_read(self, tmp_path: Path):
        """Authority means authority: an undeclared leaf is not even parsed."""
        _three_shims(tmp_path)
        stray = _leaf(tmp_path, "d")
        (stray / "provider.yaml").write_text("this: is not a shim\n", encoding="utf-8")
        _manifest(tmp_path, MANIFEST)

        status = _scan_dir(tmp_path, kind="path", location="src")

        assert status.ok
        assert "d--chat_completions" not in status.shim_names

    def test_a_broken_manifest_fails_the_source(self, tmp_path: Path):
        _three_shims(tmp_path)
        _manifest(
            tmp_path,
            "version: 1\nshims:\n  - name: a--chat_completions\n    path: nope\n",
        )

        status = _scan_dir(tmp_path, kind="path", location="src")

        assert not status.ok
        assert "shims.yaml" in (status.error or "")

    def test_a_manifest_entry_loading_another_name_is_warned(
        self, tmp_path: Path, captured
    ):
        _leaf(tmp_path, "a", name="renamed--chat_completions")
        _manifest(
            tmp_path, "version: 1\nshims:\n  - name: a--chat_completions\n    path: a\n"
        )

        status = _scan_dir(tmp_path, kind="path", location="src")

        assert status.shim_names == ("renamed--chat_completions",)
        assert any("declares no shim named" in message for message in captured)

    def test_a_manifest_less_source_loads_every_leaf(self, tmp_path: Path):
        """The plain layout keeps working: every leaf the scan finds registers."""
        _three_shims(tmp_path)

        status = _scan_dir(tmp_path, kind="path", location="src")

        assert status.shim_names == (
            "a--chat_completions",
            "b--chat_completions",
            "c--chat_completions",
        )
        assert status.manifest_shims == ()

    def test_a_release_record_refuses_a_name_it_does_not_list(
        self, tmp_path: Path, captured
    ):
        """``only`` is the complete set a manifest install may register."""
        _three_shims(tmp_path)

        status = _scan_dir(
            tmp_path, kind="path", location="src", only=("a--chat_completions",)
        )

        assert status.shim_names == ("a--chat_completions",)
        assert any("not in the release record" in message for message in captured)

    def test_a_refused_name_is_not_registered(self, tmp_path: Path):
        """``refuse`` names what core already provides; core wins a clash."""
        _three_shims(tmp_path)

        status = _scan_dir(
            tmp_path, kind="path", location="src", refuse=("b--chat_completions",)
        )

        assert status.shim_names == ("a--chat_completions", "c--chat_completions")


class TestCoreLoaderSelection:
    """The two hooks the gateway uses, exercised without the gateway."""

    def test_only_paths_selects_one_leaf(self, tmp_path: Path):
        _three_shims(tmp_path)

        shims = load_providers_from_dir(tmp_path, only_paths={"b"})

        assert [s.name for s in shims] == ["b--chat_completions"]

    def test_only_paths_selects_a_subtree(self, tmp_path: Path):
        _leaf(tmp_path / "vendor" / "llm", "x")
        _leaf(tmp_path, "a")

        shims = load_providers_from_dir(tmp_path, only_paths={"vendor"})

        assert [s.name for s in shims] == ["x--chat_completions"]

    def test_an_empty_selection_loads_nothing(self, tmp_path: Path):
        _three_shims(tmp_path)

        assert load_providers_from_dir(tmp_path, only_paths=set()) == []

    def test_a_declined_leaf_is_not_imported(self, tmp_path: Path):
        """Selection happens before the module import, so a broken one is moot."""
        _three_shims(tmp_path)
        (tmp_path / "b" / "transforms.py").write_text(
            "this is not python(\n", encoding="utf-8"
        )

        shims = load_providers_from_dir(
            tmp_path, on_error="raise", keep_names=lambda name: "b--" not in name
        )

        assert [s.name for s in shims] == [
            "a--chat_completions",
            "c--chat_completions",
        ]


class TestCoreFloor:
    """A package that tracks its branch states the oldest core it needs."""

    def test_a_floor_the_running_core_meets_loads(self, tmp_path: Path):
        _leaf(tmp_path, "a")
        _manifest(
            tmp_path,
            "version: 1\ncore: '>=0.1'\nshims:\n"
            "  - name: a--chat_completions\n    path: a\n",
        )

        status = _scan_dir(tmp_path, kind="path", location="src")

        assert status.ok
        assert status.shim_names == ("a--chat_completions",)

    def test_a_floor_above_the_running_core_is_refused(self, tmp_path: Path):
        _leaf(tmp_path, "a")
        _manifest(
            tmp_path,
            "version: 1\ncore: '>=99.0.0'\nshims:\n"
            "  - name: a--chat_completions\n    path: a\n",
        )

        status = _scan_dir(tmp_path, kind="path", location="src")

        assert not status.ok
        error = status.error or ""
        assert "needs llm-rosetta >=" in error
        assert "upgrade the gateway" in error

    @pytest.mark.parametrize("body", ["core: '>='\n", "core: '^0.15'\n", "core: 15\n"])
    def test_an_unusable_floor_is_refused(self, tmp_path: Path, body: str):
        _leaf(tmp_path, "a")
        _manifest(tmp_path, f"version: 1\n{body}shims: []\n")

        with pytest.raises(ShimManifestError):
            read_shim_manifest(tmp_path)

    def test_the_floor_is_recorded_on_the_manifest(self, tmp_path: Path):
        _leaf(tmp_path, "a")
        _manifest(
            tmp_path,
            "version: 1\ncore: '>=0.15.0'\nshims:\n"
            "  - name: a--chat_completions\n    path: a\n",
        )

        manifest = read_shim_manifest(tmp_path)

        assert manifest is not None
        assert manifest.core == ">=0.15.0"
