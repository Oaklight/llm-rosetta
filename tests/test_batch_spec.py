"""The batch record: one per ``(model_type, format)``, derived from the catalog.

Two contracts are pinned here:

* every measured pair resolves to a complete record, and the format with no
  native batch surface (``google_interactions``) has none;
* a record can only name a pair the format catalog knows, which is the same
  gate the converter registry applies to formats — so a batch record for a
  format with no converter cannot exist.

The dialect table is checked row by row against the values reviewed for epic
#1509, because a later codec reads them and a silent edit would change what a
batch request means.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from llm_rosetta.formats import BY_PAIR, FORMAT_SPECS
from llm_rosetta.formats.batch import (
    BATCH_BY_PAIR,
    BATCH_SPECS,
    BY_DIALECT,
    DECLARABLE_DIALECTS,
    DIALECTS,
    BatchSpec,
    batch_spec_for,
    batch_spec_for_dialect,
    batch_specs_for_dialect,
    check_batch_specs,
)

#: ``(model_type, format) → (dialect, container, item_key, create_path)``.
#: The one table a codec author reads before writing a container.
EXPECTED = {
    ("llm", "chat_completions"): (
        "openai_file",
        "file",
        "custom_id",
        "/v1/batches",
    ),
    ("llm", "open_responses"): (
        "openai_file",
        "file",
        "custom_id",
        "/v1/batches",
    ),
    ("llm", "messages"): (
        "anthropic_inline",
        "inline",
        "custom_id",
        "/v1/messages/batches",
    ),
    ("llm", "google_generate"): (
        "gemini_inline",
        "inline",
        "index",
        "/v1beta/models/{model}:batchGenerateContent",
    ),
}


@pytest.mark.parametrize(("pair", "expected"), EXPECTED.items())
def test_every_measured_pair_has_the_documented_record(pair, expected):
    spec = batch_spec_for(*pair)

    assert spec is not None, pair
    assert (spec.dialect, spec.container, spec.item_key, spec.create_path) == expected


def test_the_catalog_pairs_split_into_records_and_absence():
    """Every catalog pair resolves to either a record or "no native batch"."""
    for spec in FORMAT_SPECS:
        pair = (spec.model_type, spec.format)
        assert (pair in BATCH_BY_PAIR) == (pair in EXPECTED), pair


def test_a_format_without_a_batch_surface_has_no_record():
    """Interactions is generateContent-only; it must not claim a batch API."""
    assert batch_spec_for("llm", "google_interactions") is None


def test_pairs_are_unique():
    pairs = [(spec.model_type, spec.format) for spec in BATCH_SPECS]
    assert len(pairs) == len(set(pairs))
    assert set(pairs) == set(BATCH_BY_PAIR)


def test_every_record_names_a_catalog_pair():
    for spec in BATCH_SPECS:
        assert (spec.model_type, spec.format) in BY_PAIR, spec.format


def test_a_record_reaches_its_format_spec_by_value():
    """The pair is the join: a codec reads the chat-side facts, not a copy."""
    for spec in BATCH_SPECS:
        assert spec.format_spec is BY_PAIR[(spec.model_type, spec.format)]


def test_the_gate_refuses_a_pair_the_catalog_does_not_know(monkeypatch):
    """A batch record with no catalog entry has no converter binding."""
    ghost = BatchSpec(
        model_type="llm",
        format="ghost_wire",
        dialect="openai_file",
        container="file",
        create_path="/v1/batches",
        retrieve_path="/v1/batches/{batch_id}",
        cancel_path="/v1/batches/{batch_id}/cancel",
        item_key="custom_id",
        state_map={},
    )
    monkeypatch.setattr("llm_rosetta.formats.batch.BATCH_SPECS", (*BATCH_SPECS, ghost))

    with pytest.raises(RuntimeError, match="no converter binding"):
        check_batch_specs()


def test_the_gate_refuses_a_duplicate_pair(monkeypatch):
    first = BATCH_SPECS[0]
    monkeypatch.setattr("llm_rosetta.formats.batch.BATCH_SPECS", (*BATCH_SPECS, first))

    with pytest.raises(RuntimeError, match="Duplicate batch spec"):
        check_batch_specs()


def test_the_state_maps_land_in_the_normalized_vocabulary():
    from typing import get_args

    from llm_rosetta.ir.model.batch import BatchJobState

    normalized = set(get_args(BatchJobState))
    for spec in BATCH_SPECS:
        assert spec.state_map, spec.format
        unknown = set(spec.state_map.values()) - normalized
        assert not unknown, f"{spec.format}: {unknown}"


def test_the_dialect_table_lists_every_owner_per_id():
    """Two formats share ``openai_file``; the file variant owns its own id."""
    assert [s.format for s in BY_DIALECT["openai_file"]] == [
        "chat_completions",
        "open_responses",
    ]
    assert [s.format for s in BY_DIALECT["anthropic_inline"]] == ["messages"]
    assert [s.format for s in BY_DIALECT["gemini_inline"]] == ["google_generate"]
    assert [s.format for s in BY_DIALECT["gemini_file"]] == ["google_generate"]
    assert set(BY_DIALECT) == set(DIALECTS)


def test_the_declarable_dialects_are_the_codecs_plus_emulated():
    assert DECLARABLE_DIALECTS == (*BY_DIALECT, "emulated")
    assert batch_spec_for_dialect("emulated") is None
    assert batch_spec_for_dialect("nonsense") is None
    assert batch_specs_for_dialect("emulated") == ()


def test_an_ambiguous_dialect_lookup_refuses_instead_of_picking_one():
    """``openai_file`` is owned by two records; the first must not win."""
    with pytest.raises(ValueError, match="shared by.*chat_completions.*open_responses"):
        batch_spec_for_dialect("openai_file")
    owners = {s.format: s for s in batch_specs_for_dialect("openai_file")}
    assert owners["open_responses"].supported_endpoints == ("/v1/responses",)
    assert owners["chat_completions"].supported_endpoints == ("/v1/chat/completions",)


def test_the_gemini_file_dialect_resolves_to_file_variant_facts():
    spec = batch_spec_for_dialect("gemini_file")
    assert spec is not None
    assert (spec.dialect, spec.container) == ("gemini_file", "file")
    assert spec.limits.max_bytes == 2 * 1024**3
    assert spec.file_results_path is not None
    inline = batch_spec_for("llm", "google_generate")
    assert inline is not None
    assert inline.container == "inline"
    assert inline.limits.max_bytes == 20 * 1024 * 1024
    assert inline.file_variant is spec
    assert spec.file_variant is None


def _bad(monkeypatch, **changes):
    import dataclasses

    broken = dataclasses.replace(BATCH_SPECS[0], **changes)
    monkeypatch.setattr(
        "llm_rosetta.formats.batch.BATCH_SPECS", (broken, *BATCH_SPECS[1:])
    )


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"dialect": "openai_fil"}, "dialect is not one of"),
        ({"container": "none"}, "container 'none'"),
        ({"item_key": "customid"}, "item_key 'customid'"),
        ({"supported_endpoints": ()}, "supported_endpoints is empty"),
        ({"state_map": {"x": "runnin"}}, "state_map values runnin"),
    ],
)
def test_the_gate_refuses_a_malformed_record(monkeypatch, changes, match):
    _bad(monkeypatch, **changes)
    with pytest.raises(RuntimeError, match=match):
        check_batch_specs()


def test_the_gate_refuses_a_malformed_file_variant(monkeypatch):
    import dataclasses

    gemini = BATCH_SPECS[-1]
    assert gemini.file_variant is not None
    variant = dataclasses.replace(gemini.file_variant, dialect="gemini_fil")
    broken = dataclasses.replace(gemini, file_variant=variant)
    monkeypatch.setattr(
        "llm_rosetta.formats.batch.BATCH_SPECS", (*BATCH_SPECS[:-1], broken)
    )
    with pytest.raises(RuntimeError, match="gemini_fil"):
        check_batch_specs()


def test_the_container_vocabulary_has_no_none():
    from typing import get_args

    from llm_rosetta.formats.batch import BatchContainer

    assert set(get_args(BatchContainer)) == {"file", "inline"}


def test_the_openai_records_differ_only_in_endpoint():
    chat = batch_spec_for("llm", "chat_completions")
    responses = batch_spec_for("llm", "open_responses")
    assert chat is not None and responses is not None

    assert chat.supported_endpoints == ("/v1/chat/completions",)
    assert responses.supported_endpoints == ("/v1/responses",)
    assert chat.state_map == responses.state_map
    assert chat.file_results_path == responses.file_results_path


def test_messages_carries_its_own_retention_and_result_path():
    spec = batch_spec_for("llm", "messages")
    assert spec is not None

    assert spec.results_path == "/v1/messages/batches/{batch_id}/results"
    assert spec.file_results_path is None
    assert spec.result_retention_days == 29
    assert spec.limits.max_items == 100_000


def test_gemini_records_its_file_variant_and_has_no_item_cap():
    spec = batch_spec_for("llm", "google_generate")
    assert spec is not None

    assert spec.file_dialect == "gemini_file"
    # The inline ceiling is the 20 MB body; no item count is
    # published, so none is recorded.
    assert spec.limits.max_items is None
    assert spec.limits.max_bytes == 20 * 1024 * 1024


def test_records_are_immutable():
    """A record is a fact; a per-request read must not be able to edit it."""
    spec = BATCH_BY_PAIR[("llm", "chat_completions")]

    # ``FrozenInstanceError`` is an ``AttributeError``; the cast only keeps the
    # checker from rejecting the deliberate write.
    writable = cast("Any", spec)
    with pytest.raises(AttributeError):
        writable.create_path = "/elsewhere"

    # The state map is a read-only proxy, so a writer that tries anyway fails
    # here rather than in another request's view of the record.
    mutable = cast("dict[str, str]", spec.state_map)
    with pytest.raises(TypeError):
        mutable["completed"] = "failed"
