"""The two selection hooks the core directory loader offers.

``only_paths`` and ``keep_names`` are how the extension loader and the scaffold
load one folder out of a tree.  The ``shims.yaml`` manifest that once drove them
from a source was removed with the local-directory sources (#1704).
"""

from __future__ import annotations

from pathlib import Path

from llm_rosetta.api import load_providers_from_dir


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


def _three_shims(root: Path) -> None:
    for slug in ("a", "b", "c"):
        _leaf(root, slug)


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
