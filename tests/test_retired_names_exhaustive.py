"""Whole-table properties of the retired-names tables.

``test_retired_names.py`` pins the table with one example test per row.  The
checks here read the tables themselves and assert a law over every row, so a
row added later is covered without a new test.  Each law is stated once, over
the entire domain (``RETIRED_NAMES``, ``RETIRED_MODULES``, ``MOVED_VENDORS``).
"""

from __future__ import annotations

import importlib.util
import itertools

from llm_rosetta import retired_names
from llm_rosetta.converters.registry import formats_for
from llm_rosetta.model_vocabulary import MODEL_TYPES
from llm_rosetta.retired_names import (
    MODEL_TYPE,
    MOVED_VENDORS,
    PROVIDER_TYPE,
    RETIRED_MODULES,
    RETIRED_NAMES,
    SHIM,
    check_not_retired,
    lookup,
    moved_vendor,
    retired_name_message,
)
from llm_rosetta.shims import get_shim

import pytest

#: Bare chat formats that are also retired *shim* names (formats in a shim slot).
FORMAT_SHIM_NAMES = frozenset(["google_generate", "google_interactions"])

FORMATS = frozenset(f for t in MODEL_TYPES for f in formats_for(t))


def _shim_exists(name: str) -> bool:
    try:
        return get_shim(name) is not None
    except ValueError:
        return False


def _is_live(row: retired_names.RetiredName) -> bool:
    """Whether *row*'s replacement resolves in a core install."""
    if row.kind == MODEL_TYPE:
        return row.replacement in MODEL_TYPES
    if row.kind == PROVIDER_TYPE:
        return row.replacement in FORMATS or _shim_exists(row.replacement)
    return _shim_exists(row.replacement)


def _is_moved(row: retired_names.RetiredName) -> bool:
    return moved_vendor(row.replacement) is not None


def test_the_tables_are_not_empty() -> None:
    assert RETIRED_NAMES and RETIRED_MODULES and MOVED_VENDORS


def test_every_row_has_a_known_kind_and_distinct_key() -> None:
    keys = [(r.kind, r.name) for r in RETIRED_NAMES]
    assert len(keys) == len(set(keys))
    assert {r.kind for r in RETIRED_NAMES} <= set(retired_names.KINDS)


def test_every_replacement_resolves_somewhere() -> None:
    """A replacement is live in core, moved to contrib, or installable."""
    for row in RETIRED_NAMES:
        assert _is_live(row) or _is_moved(row) or row.install, row


def test_install_is_declared_exactly_for_an_external_replacement() -> None:
    """``install`` names the package that supplies a replacement core lacks."""
    for row in RETIRED_NAMES:
        if row.install:
            assert not _is_live(row) and not _is_moved(row), row
        else:
            assert _is_live(row) or _is_moved(row), row


def test_a_retired_name_is_not_live_in_its_own_slot() -> None:
    for row in RETIRED_NAMES:
        if row.kind == MODEL_TYPE:
            assert row.name not in MODEL_TYPES, row
        elif row.kind == PROVIDER_TYPE:
            assert row.name not in FORMATS, row
            assert not _shim_exists(row.name), row
        elif row.name not in FORMAT_SHIM_NAMES:
            assert not _shim_exists(row.name), row


def test_a_format_named_as_a_shim_is_a_live_format() -> None:
    shim_rows = {r.name for r in RETIRED_NAMES if r.kind == SHIM}
    for name in FORMAT_SHIM_NAMES:
        assert name in shim_rows and name in FORMATS


def test_no_row_chains_or_renames_to_itself() -> None:
    for row in RETIRED_NAMES:
        assert row.replacement != row.name, row
        assert lookup(row.replacement, row.kind) is None, row


def test_no_replacement_is_retired_under_a_slot_it_can_be_read_from() -> None:
    """A provider-type replacement is also read in the shim slot: not retired."""
    for row in RETIRED_NAMES:
        if row.kind == PROVIDER_TYPE and row.replacement not in FORMATS:
            assert lookup(row.replacement, SHIM) is None, row


def test_the_message_is_complete_for_every_row() -> None:
    for row in RETIRED_NAMES:
        message = retired_name_message(row.name, row.kind)
        assert message is not None, row
        assert f"{row.name!r} was renamed to {row.replacement!r} in 0.14" in message
        assert "no longer accepted" in message
        assert message.endswith(retired_names.MIGRATE_HINT)
        assert (f"shims.install: ['{row.install}']" in message) is bool(row.install)


def test_check_follows_the_kind_order_for_every_row() -> None:
    """For any ordering of kinds, the first kind that retires the name wins."""
    for row in RETIRED_NAMES:
        for size in range(1, len(retired_names.KINDS) + 1):
            for kinds in itertools.permutations(retired_names.KINDS, size):
                first = next((k for k in kinds if lookup(row.name, k)), None)
                if first is None:
                    check_not_retired(row.name, *kinds)
                    continue
                expected = lookup(row.name, first)
                assert expected is not None
                with pytest.raises(ValueError) as info:
                    check_not_retired(row.name, *kinds)
                assert repr(expected.replacement) in str(info.value), (row, kinds)
                assert str(info.value).endswith(retired_names.MIGRATE_HINT), kinds


def test_a_name_in_no_slot_is_never_retired() -> None:
    known = {r.name for r in RETIRED_NAMES}
    for name in itertools.chain(FORMATS, MODEL_TYPES, ["", " ", "x" * 300]):
        if name in known:
            continue
        for kind in retired_names.KINDS:
            assert retired_name_message(name, kind) is None
        check_not_retired(name, *retired_names.KINDS)


def test_no_moved_shim_is_retired_and_every_vendor_owns_its_shims() -> None:
    for vendor in MOVED_VENDORS:
        for name in vendor.shims:
            assert vendor.owns(name), name
            assert lookup(name, SHIM) is None, name


def test_moved_vendors_are_distinct_and_own_distinct_shims() -> None:
    vendors = [v.vendor for v in MOVED_VENDORS]
    assert len(vendors) == len(set(vendors))
    shims = [s for v in MOVED_VENDORS for s in v.shims]
    assert len(shims) == len(set(shims))


def test_every_retired_module_replacement_resolves_and_is_not_retired() -> None:
    paths = [m.path for m in RETIRED_MODULES]
    assert len(paths) == len(set(paths))
    for module in RETIRED_MODULES:
        if module.replacement is None:
            assert module.note, module
            continue
        assert importlib.util.find_spec(module.replacement) is not None, module
        assert retired_names.retired_module(module.replacement) is None, module
