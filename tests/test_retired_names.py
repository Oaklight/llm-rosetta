"""The retired-names table: shape, messages, switch and the boundaries it guards.

The provider-type and shim alias tables are gone (#919).  A retired spelling is
no longer translated; it fails at every boundary with a message that names its
replacement.  The two model-type spellings (``embedding``, ``decision``) still
translate while :data:`~llm_rosetta.retired_names.ALIASES_ENABLED` is ``True``.
"""

from __future__ import annotations

import importlib.util
import warnings

import pytest

from llm_rosetta import ConversionPipeline, convert, convert_response, retired_names
from llm_rosetta.auto_detect import get_converter_for_provider
from llm_rosetta.converters.embeddings.pipeline import get_embedding_converter
from llm_rosetta.formats import LEGACY_MODEL_TYPES, canonical_model_type
from llm_rosetta.retired_names import (
    MODEL_TYPE,
    PROVIDER_TYPE,
    RETIRED_NAMES,
    SHIM,
    check_not_retired,
    lookup,
    retired_name_message,
)
from llm_rosetta.shims import ProviderShim, get_shim, resolve_model_type

BODY = dict(
    model="gpt-4o",
    messages=[dict(role="user", content="hello")],
)
RESPONSE = dict(
    id="chatcmpl-1",
    object="chat.completion",
    model="gpt-4o",
    choices=[
        dict(
            index=0,
            message=dict(role="assistant", content="hi"),
            finish_reason="stop",
        )
    ],
)

#: Bare chat formats that are also listed as retired *shim* names.  In a shim
#: slot they are formats, not shims, and must not be rejected.
FORMAT_SHIM_NAMES = frozenset(["google_generate", "google_interactions"])

ROWS = [pytest.param(row, id=f"{row.kind}:{row.name}") for row in RETIRED_NAMES]
PROVIDER_ROWS = [r for r in RETIRED_NAMES if r.kind == PROVIDER_TYPE]
SHIM_ROWS = [r for r in RETIRED_NAMES if r.kind == SHIM]
SHIM_SLOT_ROWS = [r for r in SHIM_ROWS if r.name not in FORMAT_SHIM_NAMES]
MODEL_TYPE_ROWS = [r for r in RETIRED_NAMES if r.kind == MODEL_TYPE]


def _ids(rows):
    return [pytest.param(r, id=r.name) for r in rows]


def _replacement(name: str, *kinds: str) -> str:
    """The replacement a boundary reading *name* under *kinds* names."""
    for kind in kinds:
        row = lookup(name, kind)
        if row is not None:
            return row.replacement
    raise AssertionError(f"{name!r} is not retired under {kinds}")


# ---------------------------------------------------------------------------
# The alias tables are gone
# ---------------------------------------------------------------------------


def test_provider_names_module_is_gone() -> None:
    assert importlib.util.find_spec("llm_rosetta.provider_names") is None


def test_alias_functions_are_not_exported() -> None:
    import llm_rosetta
    from llm_rosetta import api

    for name in (
        "canonical_provider_name",
        "canonical_shim_name",
        "legacy_names",
        "normalize_provider_name",
        "warn_legacy_provider_names",
        "warn_legacy_shim_names",
    ):
        assert not hasattr(api, name), name
        assert not hasattr(llm_rosetta, name), name
        assert name not in api.__all__
        assert name not in llm_rosetta.__all__


def test_provider_type_literal_has_no_legacy_members() -> None:
    from typing import get_args

    from llm_rosetta import ProviderType

    members = set(get_args(ProviderType))
    assert not members.intersection(row.name for row in PROVIDER_ROWS)
    assert members.issuperset(["chat_completions", "messages", "google_generate"])


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


def test_every_live_model_type_alias_has_a_row() -> None:
    for old, new in LEGACY_MODEL_TYPES.items():
        row = lookup(old, MODEL_TYPE)
        assert row is not None, f"model type {old!r} is not in RETIRED_NAMES"
        assert row.replacement == new


def test_table_size_by_kind() -> None:
    sizes = dict(
        (kind, sum(1 for row in RETIRED_NAMES if row.kind == kind))
        for kind in retired_names.KINDS
    )
    assert sizes == dict([(PROVIDER_TYPE, 9), (SHIM, 20), (MODEL_TYPE, 2)])


def test_no_duplicate_rows() -> None:
    keys = [(row.kind, row.name) for row in RETIRED_NAMES]
    assert len(keys) == len(set(keys))


def test_no_chains() -> None:
    """A replacement is never itself retired under the same kind."""
    for row in RETIRED_NAMES:
        assert lookup(row.replacement, row.kind) is None, row
        assert row.replacement != row.name


def test_kinds_are_known() -> None:
    assert set(retired_names.KINDS).issuperset(row.kind for row in RETIRED_NAMES)


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
# The switch governs model types only
# ---------------------------------------------------------------------------


def test_switch_defaults_on() -> None:
    assert retired_names.ALIASES_ENABLED is True


@pytest.mark.parametrize("row", ROWS)
def test_check_with_the_switch_on(row: retired_names.RetiredName) -> None:
    """Provider-type and shim rows raise; model-type rows still translate."""
    if row.kind == MODEL_TYPE:
        check_not_retired(row.name, row.kind)
        return
    with pytest.raises(ValueError, match="was renamed") as info:
        check_not_retired(row.name, row.kind)
    assert row.replacement in str(info.value)


@pytest.mark.parametrize("row", _ids(MODEL_TYPE_ROWS))
def test_model_type_rows_raise_with_the_switch_off(
    row: retired_names.RetiredName, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(retired_names, "ALIASES_ENABLED", False)
    with pytest.raises(ValueError, match=row.replacement):
        canonical_model_type(row.name)
    with pytest.raises(ValueError, match=row.replacement):
        resolve_model_type(row.name)


def test_model_types_translate_with_the_switch_on() -> None:
    assert canonical_model_type("embedding") == "embeddings"
    assert canonical_model_type("decision") == "decisions"


def test_non_string_is_ignored() -> None:
    check_not_retired(None, PROVIDER_TYPE, SHIM)
    check_not_retired(3, PROVIDER_TYPE, SHIM)


# ---------------------------------------------------------------------------
# Retired provider types fail at every library boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row", _ids(PROVIDER_ROWS))
def test_get_converter_names_the_replacement(row: retired_names.RetiredName) -> None:
    with pytest.raises(ValueError, match="was renamed") as info:
        get_converter_for_provider(row.name)
    assert repr(row.replacement) in str(info.value)


@pytest.mark.parametrize("row", _ids(PROVIDER_ROWS))
def test_pipeline_rejects_a_retired_source_and_target(
    row: retired_names.RetiredName,
) -> None:
    with pytest.raises(ValueError, match=repr(row.replacement)):
        ConversionPipeline(row.name, "messages")
    with pytest.raises(ValueError, match=repr(row.replacement)):
        ConversionPipeline("messages", row.name)


@pytest.mark.parametrize("row", _ids(PROVIDER_ROWS))
def test_convert_rejects_a_retired_target_and_source(
    row: retired_names.RetiredName,
) -> None:
    with pytest.raises(ValueError, match=repr(row.replacement)):
        convert(BODY, row.name)
    with pytest.raises(ValueError, match=repr(row.replacement)):
        convert(BODY, "messages", source_provider=row.name)
    with pytest.raises(ValueError, match=repr(row.replacement)):
        convert_response(RESPONSE, BODY, row.name, "chat_completions")
    with pytest.raises(ValueError, match=repr(row.replacement)):
        convert_response(RESPONSE, BODY, "chat_completions", row.name)


@pytest.mark.parametrize("row", _ids(PROVIDER_ROWS))
def test_embedding_converter_rejects_a_retired_format(
    row: retired_names.RetiredName,
) -> None:
    with pytest.raises(ValueError, match=repr(row.replacement)):
        get_embedding_converter(row.name)


@pytest.mark.parametrize("row", _ids(PROVIDER_ROWS))
def test_provider_shim_rejects_a_retired_format(
    row: retired_names.RetiredName,
) -> None:
    with pytest.raises(ValueError, match=repr(row.replacement)):
        ProviderShim(name="x", format=row.name)
    with pytest.warns(DeprecationWarning, match="base="):
        with pytest.raises(ValueError, match=repr(row.replacement)):
            ProviderShim(name="x", base=row.name)


# ---------------------------------------------------------------------------
# Retired shim names fail where a shim name enters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row", _ids(SHIM_SLOT_ROWS))
def test_pipeline_rejects_a_retired_shim_string(
    row: retired_names.RetiredName,
) -> None:
    attempts = [
        lambda: ConversionPipeline(
            "chat_completions", "chat_completions", source_shim=row.name
        ),
        lambda: ConversionPipeline(
            "chat_completions", "chat_completions", target_shim=row.name
        ),
    ]
    for attempt in attempts:
        with pytest.raises(ValueError, match=repr(row.replacement)) as info:
            attempt()
        if row.preset:
            assert f"shims.presets: ['{row.preset}']" in str(info.value)


@pytest.mark.parametrize("row", _ids(SHIM_SLOT_ROWS))
def test_convert_rejects_a_retired_shim_as_provider(
    row: retired_names.RetiredName,
) -> None:
    expected = _replacement(row.name, PROVIDER_TYPE, SHIM)
    with pytest.raises(ValueError, match=repr(expected)):
        convert(BODY, row.name)


@pytest.mark.parametrize("row", _ids(SHIM_SLOT_ROWS))
def test_a_retired_shim_name_is_not_a_registered_shim(
    row: retired_names.RetiredName,
) -> None:
    assert get_shim(row.name) is None


@pytest.mark.parametrize("name", sorted(FORMAT_SHIM_NAMES))
def test_a_base_format_is_not_rejected_in_a_shim_slot(name: str) -> None:
    """``google_generate`` is retired as a shim name but is a format."""
    ConversionPipeline("chat_completions", "chat_completions", target_shim=name)
    convert(BODY, name)


def test_replacement_shim_names_still_work() -> None:
    assert get_shim("openai--chat_completions") is not None
    assert get_shim("openai--open_responses") is not None
    ConversionPipeline(
        "chat_completions",
        "chat_completions",
        target_shim="volcengine--chat_completions",
    )


# ---------------------------------------------------------------------------
# Typos and canonical names
# ---------------------------------------------------------------------------


def test_typos_keep_the_plain_message() -> None:
    with pytest.raises(ValueError, match="Unsupported provider: openai_chatt") as info:
        get_converter_for_provider("openai_chatt")
    assert "was renamed" not in str(info.value)
    with pytest.raises(ValueError) as info:
        get_embedding_converter("googl")
    assert "was renamed" not in str(info.value)


def test_canonical_names_never_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        convert(BODY, "chat_completions")
        ConversionPipeline("chat_completions", "messages")
        get_converter_for_provider("open_responses")
        get_converter_for_provider("openai--open_responses")


# ---------------------------------------------------------------------------
# Lookups no longer translate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row", _ids(PROVIDER_ROWS))
def test_resolve_format_leaves_a_retired_name_alone(
    row: retired_names.RetiredName,
) -> None:
    from llm_rosetta.shims import resolve_format

    assert resolve_format(row.name) == row.name


def test_register_shim_keys_on_the_exact_name() -> None:
    """A shim named like a retired spelling is reachable under that name only."""
    import llm_rosetta.shims.provider_shim as ps

    saved = dict(ps._SHIM_REGISTRY)
    try:
        ps._SHIM_REGISTRY.clear()
        neutral = ProviderShim(name="open_responses", format="open_responses")
        profile = ProviderShim(name="openai_responses", format="open_responses")
        ps.register_shim(neutral)
        ps.register_shim(profile)
        assert ps.get_shim("open_responses") is neutral
        assert ps.get_shim("openai_responses") is profile
        assert ps.get_shim("openai--open_responses") is None
        assert ps.resolve_format("openai_responses") == "open_responses"
    finally:
        ps._SHIM_REGISTRY.clear()
        ps._SHIM_REGISTRY.update(saved)
