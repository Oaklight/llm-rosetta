"""The removed-import-path table in ``retired_names`` (#919).

``RETIRED_MODULES`` is data: a removed module path and what to import
instead.  ``llm-rosetta gateway migrate --scan`` reads it (see
``tests/gateway/test_migrate_scan_imports.py``).  These tests pin the table's
shape and the message.
"""

from __future__ import annotations

import importlib.util

import pytest

from llm_rosetta.retired_names import (
    RETIRED_MODULES,
    retired_module,
    retired_module_message,
)

ROWS = [pytest.param(row, id=row.path) for row in RETIRED_MODULES]

#: The contract homes of #1657.  Their package lands in a parallel change, so a
#: replacement under it may not resolve yet on this tree.
_CONTRACTS = "llm_rosetta.contracts."


def test_paths_are_unique() -> None:
    paths = [row.path for row in RETIRED_MODULES]
    assert len(paths) == len(set(paths))


@pytest.mark.parametrize("row", ROWS)
def test_no_replacement_is_itself_retired(row) -> None:
    if row.replacement is not None:
        assert retired_module(row.replacement) is None, row


@pytest.mark.parametrize("row", ROWS)
def test_a_row_without_replacement_says_what_to_do(row) -> None:
    assert (row.replacement is None) == bool(row.note), row


@pytest.mark.parametrize("row", ROWS)
def test_every_replacement_resolves_or_is_a_contract_home(row) -> None:
    if row.replacement is None or row.replacement.startswith(_CONTRACTS):
        return
    assert importlib.util.find_spec(row.replacement) is not None, row


def test_the_import_paths_919_removed_do_not_resolve() -> None:
    """Rows from the #919 removals name paths that are gone today."""
    for path in (
        "llm_rosetta.converters.openai_chat",
        "llm_rosetta.converters.base.tools.batch",
        "llm_rosetta.provider_names",
        "llm_rosetta.shims.transforms",
    ):
        assert retired_module(path) is not None, path
        assert importlib.util.find_spec(path) is None, path


def test_a_submodule_is_covered_by_its_package_row() -> None:
    row = retired_module("llm_rosetta.converters.openai_chat.converter")
    assert row is not None
    assert row.path == "llm_rosetta.converters.openai_chat"


def test_a_sibling_with_a_common_prefix_is_not_covered() -> None:
    assert retired_module("llm_rosetta.transforms_extra") is None
    assert retired_module("llm_rosetta.converters.llm.chat_completions") is None


def test_the_message_names_the_replacement() -> None:
    message = retired_module_message("llm_rosetta.converters.anthropic.converter")
    assert message == (
        "'llm_rosetta.converters.anthropic' was removed in 0.14; "
        "import 'llm_rosetta.converters.llm.messages' instead; "
        "see the migration guide"
    )


def test_the_message_for_a_path_without_replacement_says_so() -> None:
    message = retired_module_message("llm_rosetta.capabilities")
    assert message is not None
    assert "has no public replacement" in message


def test_the_contract_moves_name_their_new_homes() -> None:
    expected = {
        "llm_rosetta.transforms": "llm_rosetta.contracts.transform",
        "llm_rosetta.shims.auth_exchange": "llm_rosetta.contracts.auth",
        "llm_rosetta.shims.quota_probe": "llm_rosetta.contracts.quota",
        "llm_rosetta.routing": "llm_rosetta.contracts.routing",
        "llm_rosetta.observability.usage": "llm_rosetta.contracts.usage",
        "llm_rosetta.capabilities": None,
    }
    for path, replacement in expected.items():
        row = retired_module(path)
        assert row is not None and row.replacement == replacement, path


def test_an_unknown_path_has_no_message() -> None:
    assert retired_module_message("llm_rosetta.formats") is None
    assert retired_module_message("requests") is None
