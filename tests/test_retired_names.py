"""The retired-names table: coverage, shape, switch and default behaviour."""

from __future__ import annotations

import json
import warnings
from typing import Any
from pathlib import Path

import pytest

from llm_rosetta import provider_names as pn
from llm_rosetta import retired_names
from llm_rosetta.auto_detect import get_converter_for_provider
from llm_rosetta.formats import (
    LEGACY_MODEL_TYPES,
    canonical_model_type,
)
from llm_rosetta.pipeline import ConversionPipeline
from llm_rosetta.retired_names import (
    MODEL_TYPE,
    PROVIDER_TYPE,
    RETIRED_NAMES,
    SHIM,
    check_not_retired,
    lookup,
    retired_name_message,
)
from llm_rosetta.shims import resolve_format, resolve_model_type

GOLDEN = Path(__file__).parent / "data" / "retired_names_golden.json"

ROWS = [pytest.param(row, id=f"{row.kind}:{row.name}") for row in RETIRED_NAMES]


@pytest.fixture
def aliases_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retired_names, "ALIASES_ENABLED", False)


# ---------------------------------------------------------------------------
# The table covers the live alias tables
# ---------------------------------------------------------------------------


def test_table_covers_every_live_alias_entry() -> None:
    for old, new in pn.LEGACY_PROVIDER_ALIASES.items():
        row = lookup(old, PROVIDER_TYPE)
        assert row is not None, f"provider alias {old!r} is not in RETIRED_NAMES"
        assert row.replacement == new
    for old, new in pn.LEGACY_SHIM_ALIASES.items():
        row = lookup(old, SHIM)
        assert row is not None, f"shim alias {old!r} is not in RETIRED_NAMES"
        assert row.replacement == new
    for old, new in LEGACY_MODEL_TYPES.items():
        row = lookup(old, MODEL_TYPE)
        assert row is not None, f"model type {old!r} is not in RETIRED_NAMES"
        assert row.replacement == new


def test_table_has_no_entry_beyond_the_live_alias_tables() -> None:
    """A row nobody aliases any more is only legal once its table is gone."""
    live = {
        PROVIDER_TYPE: pn.LEGACY_PROVIDER_ALIASES,
        SHIM: pn.LEGACY_SHIM_ALIASES,
        MODEL_TYPE: LEGACY_MODEL_TYPES,
    }
    for row in RETIRED_NAMES:
        assert row.name in live[row.kind], row


def test_table_size_by_kind() -> None:
    sizes = {
        kind: sum(1 for row in RETIRED_NAMES if row.kind == kind)
        for kind in retired_names.KINDS
    }
    assert sizes == {PROVIDER_TYPE: 9, SHIM: 20, MODEL_TYPE: 2}


def test_no_duplicate_rows() -> None:
    keys = [(row.kind, row.name) for row in RETIRED_NAMES]
    assert len(keys) == len(set(keys))


def test_no_chains() -> None:
    """A replacement is never itself retired under the same kind."""
    for row in RETIRED_NAMES:
        assert lookup(row.replacement, row.kind) is None, row
        assert row.replacement != row.name


def test_kinds_are_known() -> None:
    assert {row.kind for row in RETIRED_NAMES} <= set(retired_names.KINDS)


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row", ROWS)
def test_message_names_the_replacement(row: retired_names.RetiredName) -> None:
    message = retired_name_message(row.name, row.kind)
    assert message is not None
    assert f"{row.name!r} was renamed to {row.replacement!r} in 0.14" in message
    assert "no longer accepted" in message
    if row.preset:
        assert f"shims.presets: ['{row.preset}']" in message
    else:
        assert "shims.presets" not in message


@pytest.mark.parametrize("name", ["openai_chatt", "chat_completions", "", "Embedding"])
def test_unretired_names_get_no_hint(name: str) -> None:
    for kind in retired_names.KINDS:
        assert retired_name_message(name, kind) is None


# ---------------------------------------------------------------------------
# The switch: on is a no-op, off raises at every boundary
# ---------------------------------------------------------------------------


def test_switch_defaults_on() -> None:
    assert retired_names.ALIASES_ENABLED is True


@pytest.mark.parametrize("row", ROWS)
def test_check_is_silent_while_aliases_are_on(
    row: retired_names.RetiredName,
) -> None:
    check_not_retired(row.name, row.kind)


@pytest.mark.parametrize("row", ROWS)
def test_core_boundaries_raise_with_aliases_off(
    row: retired_names.RetiredName, aliases_off: None
) -> None:
    expected = row.replacement
    with pytest.raises(ValueError, match="was renamed") as info:
        if row.kind == PROVIDER_TYPE:
            pn.normalize_provider_name(row.name)
        elif row.kind == SHIM:
            pn.warn_legacy_shim_names(row.name)
        else:
            canonical_model_type(row.name)
    assert expected in str(info.value)
    if row.kind == PROVIDER_TYPE:
        with pytest.raises(ValueError, match=expected):
            pn.warn_legacy_provider_names(row.name)
        with pytest.raises(ValueError, match=expected):
            get_converter_for_provider(row.name)
    if (
        row.kind == SHIM
        and row.name not in {"google_generate", "google_interactions"}
        # ``google`` is retired under both kinds; the provider reading wins.
        and lookup(row.name, PROVIDER_TYPE) is None
    ):
        with pytest.raises(ValueError, match=expected):
            get_converter_for_provider(row.name)
    if row.kind == MODEL_TYPE:
        with pytest.raises(ValueError, match=expected):
            resolve_model_type(row.name)


def test_pipeline_constructor_raises_with_aliases_off(aliases_off: None) -> None:
    with pytest.raises(ValueError, match="'chat_completions'"):
        ConversionPipeline("openai_chat", "messages")


def test_alias_tables_behave_as_empty_with_aliases_off(aliases_off: None) -> None:
    for old in pn.LEGACY_PROVIDER_ALIASES:
        assert pn.canonical_provider_name(old) == old
    for old in pn.LEGACY_SHIM_ALIASES:
        assert pn.canonical_shim_name(old) == old


def test_typos_keep_todays_message_with_aliases_off(aliases_off: None) -> None:
    with pytest.raises(ValueError, match="Unsupported provider: openai_chatt"):
        get_converter_for_provider("openai_chatt")
    assert canonical_model_type("embeddingz") == "embeddingz"
    assert pn.normalize_provider_name("openai_chatt") == "openai_chatt"
    assert resolve_format("chat_completions") == "chat_completions"


# ---------------------------------------------------------------------------
# Default behaviour is unchanged: differential against a recording of master
# ---------------------------------------------------------------------------


def _call(fn, *args):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out: dict[str, Any] = {"ok": repr(fn(*args))}
        except Exception as exc:  # noqa: BLE001 - recorded, compared verbatim
            out = {"err": f"{type(exc).__name__}: {exc}"}
        out["warnings"] = [f"{w.category.__name__}: {w.message}" for w in caught]
    return out


def test_golden_covers_every_retired_name() -> None:
    golden = json.loads(GOLDEN.read_text())
    assert set(golden) == {row.name for row in RETIRED_NAMES}


@pytest.mark.parametrize("name", sorted({row.name for row in RETIRED_NAMES}))
def test_default_behaviour_matches_the_recording(name: str) -> None:
    """Core half of the differential test (the gateway half lives beside it).

    ``tests/data/retired_names_golden.json`` was recorded on master before
    this change; every core call below must reproduce it byte for byte.
    """
    recorded = json.loads(GOLDEN.read_text())[name]
    live = {
        "canonical_provider_name": _call(pn.canonical_provider_name, name),
        "canonical_shim_name": _call(pn.canonical_shim_name, name),
        "normalize_provider_name": _call(pn.normalize_provider_name, name),
        "warn_legacy_provider_names": _call(pn.warn_legacy_provider_names, name),
        "warn_legacy_shim_names": _call(pn.warn_legacy_shim_names, name),
        "canonical_model_type": _call(canonical_model_type, name),
        "get_converter_for_provider": _call(
            lambda x: type(get_converter_for_provider(x)).__name__, name
        ),
        "resolve_format": _call(resolve_format, name),
        "resolve_model_type": _call(resolve_model_type, name),
    }
    for key, value in live.items():
        assert value == recorded[key], key
