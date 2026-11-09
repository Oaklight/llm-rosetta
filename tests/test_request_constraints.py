"""Request-field constraints: defaults, null normalization, ranges (#1460).

A dialect declares what it requires of its own request fields
(``BaseConverter.request_constraints``) and a shim may declare or override the
same paths for its vendor (``request:`` in ``provider.yaml``).  The pass applies
the merged object to the IR right before a writer reads it.
"""

from __future__ import annotations

from typing import Any

import pytest

from llm_rosetta.conversion_warnings import DOWNGRADED, DROPPED, WarningSink
from llm_rosetta.converters.llm.messages import MessagesConverter
from llm_rosetta.ir.passes import (
    RequestConstraintViolation,
    apply_request_constraints,
)
from llm_rosetta.types.ir.configs import GenerationConfig
from llm_rosetta.types.ir.constraints import Range, RequestConstraints


def _ir(**generation: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"model": "m"}
    if generation:
        body["generation"] = dict(generation)
    return body


class TestNullNormalization:
    """An explicit null means "the client named no value", in IR terms."""

    def test_every_generation_key_is_normalized(self):
        """The scope is the IR type, not a list kept by hand."""
        sink = WarningSink()
        keys = sorted(GenerationConfig.__annotations__)
        ir = _ir(**{key: None for key in keys})
        apply_request_constraints(ir, warnings=sink)
        assert not ir.get("generation"), ir["generation"]
        assert len(sink) == len(keys)
        assert all(w.code == DROPPED for w in sink)
        assert {w.subject for w in sink} == {f"generation.{k}" for k in keys}

    def test_a_value_is_untouched(self):
        ir = _ir(max_tokens=123)
        apply_request_constraints(ir)
        assert ir["generation"]["max_tokens"] == 123

    def test_a_null_outside_generation_is_left_alone(self):
        """Only the generation block is normalized in this batch."""
        ir = {"model": "m", "stream": None}
        apply_request_constraints(ir)
        assert ir["stream"] is None

    def test_a_null_is_normalized_without_any_declaration(self):
        """IR semantics, not a dialect quirk: no constraints object needed."""
        ir = _ir(max_tokens=None)
        apply_request_constraints(ir, constraints=None)
        assert "generation" in ir and not ir["generation"]


class TestDefaults:
    def test_an_absent_block_is_created(self):
        ir: dict[str, Any] = {"model": "m"}
        constraints = RequestConstraints(defaults={"generation.max_tokens": 4096})
        apply_request_constraints(ir, constraints=constraints)
        assert ir["generation"]["max_tokens"] == 4096

    def test_a_client_value_wins(self):
        ir = _ir(max_tokens=11)
        apply_request_constraints(
            ir, constraints=RequestConstraints(defaults={"generation.max_tokens": 4096})
        )
        assert ir["generation"]["max_tokens"] == 11

    def test_a_null_counts_as_absent(self):
        ir = _ir(max_tokens=None)
        apply_request_constraints(
            ir, constraints=RequestConstraints(defaults={"generation.max_tokens": 4096})
        )
        assert ir["generation"]["max_tokens"] == 4096

    def test_the_filled_value_is_recorded_as_injected(self):
        metadata: dict[str, Any] = {}
        apply_request_constraints(
            _ir(),
            constraints=RequestConstraints(defaults={"generation.max_tokens": 4096}),
            metadata=metadata,
        )
        assert metadata == {"injected": {"max_tokens": 4096}}

    def test_a_client_value_is_not_recorded(self):
        metadata: dict[str, Any] = {}
        apply_request_constraints(
            _ir(max_tokens=11),
            constraints=RequestConstraints(defaults={"generation.max_tokens": 4096}),
            metadata=metadata,
        )
        assert metadata == {}


class TestRanges:
    def test_clamp_writes_the_nearest_bound_and_reports(self):
        sink = WarningSink()
        ir = _ir(temperature=1.8)
        apply_request_constraints(
            ir,
            constraints=RequestConstraints(
                ranges={"generation.temperature": Range(0.0, 1.0, "clamp")}
            ),
            warnings=sink,
        )
        assert ir["generation"]["temperature"] == 1.0
        assert len(sink) == 1
        assert sink[0].code == DOWNGRADED
        assert sink[0].subject == "generation.temperature"

    def test_drop_removes_the_field_and_reports(self):
        sink = WarningSink()
        ir = _ir(top_p=0.001)
        apply_request_constraints(
            ir,
            constraints=RequestConstraints(
                ranges={"generation.top_p": Range(0.01, None, "drop")}
            ),
            warnings=sink,
        )
        assert "top_p" not in ir["generation"]
        assert sink[0].code == DROPPED
        assert sink[0].subject == "generation.top_p"

    def test_reject_raises_with_the_path_and_the_range(self):
        ir = _ir(temperature=1.8)
        with pytest.raises(RequestConstraintViolation) as caught:
            apply_request_constraints(
                ir,
                constraints=RequestConstraints(
                    ranges={"generation.temperature": Range(0.0, 1.0, "reject")}
                ),
            )
        assert caught.value.path == "generation.temperature"
        assert "1.8" in str(caught.value)

    def test_a_value_inside_the_range_is_untouched(self):
        sink = WarningSink()
        ir = _ir(temperature=0.4)
        apply_request_constraints(
            ir,
            constraints=RequestConstraints(
                ranges={"generation.temperature": Range(0.0, 1.0, "clamp")}
            ),
            warnings=sink,
        )
        assert ir["generation"]["temperature"] == 0.4
        assert not sink

    def test_a_non_number_is_left_to_the_writer(self):
        """Constraint checking does not become a type checker."""
        ir = _ir(temperature="hot")
        apply_request_constraints(
            ir,
            constraints=RequestConstraints(
                ranges={"generation.temperature": Range(0.0, 1.0, "clamp")}
            ),
        )
        assert ir["generation"]["temperature"] == "hot"


class TestMerge:
    def test_a_shim_overrides_the_dialect_per_path(self):
        dialect = RequestConstraints(
            defaults={"generation.max_tokens": 4096, "generation.top_k": 40},
            ranges={"generation.temperature": Range(0.0, 1.0, "clamp")},
        )
        shim = RequestConstraints(
            defaults={"generation.max_tokens": 2048},
            ranges={"generation.temperature": Range(0.0, 2.0, "clamp")},
        )
        merged = dialect.merged_with(shim)
        assert merged.defaults == {
            "generation.max_tokens": 2048,
            "generation.top_k": 40,
        }
        assert merged.ranges["generation.temperature"].max == 2.0

    def test_an_empty_overlay_returns_the_dialect_object(self):
        dialect = RequestConstraints(defaults={"generation.max_tokens": 4096})
        assert dialect.merged_with(None) is dialect
        assert dialect.merged_with(RequestConstraints()) is dialect


class TestMessagesDialectDeclaration:
    """The two facts the Messages dialect used to hardcode in its writers."""

    def test_the_dialect_declares_its_required_field_and_range(self):
        declared = MessagesConverter.request_constraints
        assert declared.defaults == {"generation.max_tokens": 4096}
        temperature = declared.ranges["generation.temperature"]
        assert (temperature.min, temperature.max) == (0.0, 1.0)
        assert temperature.on_out_of_range == "clamp"

    def test_a_shim_value_replaces_the_dialect_default(self):
        """The vendor speaks for its own upstream, per path."""
        constraints = MessagesConverter.request_constraints.merged_with(
            RequestConstraints(defaults={"generation.max_tokens": 2048})
        )
        ir = _ir()
        apply_request_constraints(ir, constraints=constraints)
        assert ir["generation"]["max_tokens"] == 2048
