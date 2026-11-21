"""``llm-rosetta-migrate --scan`` names the replacement of a removed import (#919).

The scan reads the import statements of ``.py`` files against
``retired_names.RETIRED_MODULES``.  A listed path is reported only when it does
not resolve in the installed core: until the #1657 move lands, its old paths
(``llm_rosetta.routing`` and the rest) still exist and must not be reported.
"""

from __future__ import annotations

from pathlib import Path

from llm_rosetta.api import retired_names

from llm_rosetta_gateway.maintenance import migrate_0140 as mig


def _imports(tmp_path: Path, source: str) -> list[tuple]:
    (tmp_path / "app.py").write_text(source)
    return sorted(
        (item.line, item.old, item.new)
        for item in mig.scan_path(tmp_path)
        if item.action == "import"
    )


def test_a_removed_package_import_names_its_replacement(tmp_path, capsys) -> None:
    found = _imports(
        tmp_path,
        "import llm_rosetta.converters.openai_chat\n"
        "from llm_rosetta.converters.anthropic.converter import X\n",
    )

    assert found == [
        (
            1,
            "llm_rosetta.converters.openai_chat",
            "llm_rosetta.converters.llm.chat_completions",
        ),
        (
            2,
            "llm_rosetta.converters.anthropic.converter",
            "llm_rosetta.converters.llm.messages",
        ),
    ]
    assert mig.main(["--scan", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "imports removed module 'llm_rosetta.converters.openai_chat'" in out
    assert "import 'llm_rosetta.converters.llm.chat_completions' instead" in out
    assert "2 removed import(s)" in out


def test_a_removed_submodule_imported_from_its_package(tmp_path) -> None:
    """``from pkg import mod`` is the only spelling that shows a removed submodule."""
    found = _imports(
        tmp_path,
        "from llm_rosetta.converters.base.tools import (\n    batch,\n    ToolChoiceSpec,\n)\n",
    )

    assert found == [
        (1, "llm_rosetta.converters.base.tools.batch", "llm_rosetta.ir.tools.batch")
    ]


def test_a_path_without_replacement_says_so(tmp_path) -> None:
    (tmp_path / "app.py").write_text("from llm_rosetta.provider_names import x\n")

    [item] = [i for i in mig.scan_path(tmp_path) if i.action == "import"]

    assert item.new is None
    assert "has no public replacement" in item.detail


def test_a_listed_path_that_still_resolves_is_not_reported(
    tmp_path, monkeypatch
) -> None:
    """A row whose path still exists is silent, whatever the table says."""
    rows = (
        retired_names.RetiredModule("llm_rosetta.formats", "llm_rosetta.api"),
        *retired_names.RETIRED_MODULES,
    )
    monkeypatch.setattr(retired_names, "RETIRED_MODULES", rows)

    found = _imports(
        tmp_path,
        "import llm_rosetta.formats\n"
        "from llm_rosetta.formats import catalog\n"
        "import llm_rosetta.converters.openai_chat\n",
    )

    assert [old for _, old, _ in found] == ["llm_rosetta.converters.openai_chat"]


def test_every_row_that_still_resolves_is_silent(tmp_path) -> None:
    """The real table: only rows whose path is gone produce an item."""
    lines = [f"import {row.path}" for row in retired_names.RETIRED_MODULES]
    found = _imports(tmp_path, "\n".join(lines) + "\n")

    reported = {old for _, old, _ in found}
    for row in retired_names.RETIRED_MODULES:
        assert (row.path in reported) is (not mig._resolves(row.path)), row.path


def test_relative_imports_and_unparsable_files_are_skipped(tmp_path) -> None:
    (tmp_path / "rel.py").write_text("from .openai_chat import x\n")
    (tmp_path / "broken.py").write_text(
        "import llm_rosetta.converters.openai_chat\ndef (\n"
    )

    assert [i for i in mig.scan_path(tmp_path) if i.action == "import"] == []
