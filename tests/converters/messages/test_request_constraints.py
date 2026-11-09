"""The Messages dialect's required field, exercised through its writer (#1460).

`max_tokens` is required and has no API default, so the value lives in the
dialect's `request_constraints`, applied by `BaseConverter` before the writer
runs.  A shim may declare a different value; the pipeline merges it in and
hands it over through the context.  These cases pin the three states the
review asked for: no declaration, an empty one, and one that overrides.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.converters.llm.messages import MessagesConverter
from llm_rosetta.types.ir import IRRequest
from llm_rosetta.types.ir.constraints import Range, RequestConstraints

IR: dict[str, Any] = {
    "model": "claude-test",
    "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
}


def _request(**generation: Any) -> IRRequest:
    body = {**IR}
    if generation:
        body["generation"] = generation
    return cast(IRRequest, body)


def _write(
    converter: MessagesConverter | None = None,
    *,
    generation: dict[str, Any] | None = None,
    constraints: RequestConstraints | None = None,
) -> tuple[dict[str, Any], ConversionContext]:
    converter = converter or MessagesConverter()
    ctx = ConversionContext()
    if constraints is not None:
        ctx.options["request_constraints"] = constraints
    ir = _request(**(generation or {}))
    body, _ = converter.request_to_provider(ir, context=ctx)
    return body, ctx


class TestRequiredField:
    def test_a_request_that_names_no_limit_gets_the_declared_value(self):
        body, ctx = _write()
        assert body["max_tokens"] == 4096
        assert ctx.metadata["injected"] == {"max_tokens": 4096}

    def test_an_empty_generation_block_gets_it_too(self):
        body, _ = _write(generation={"temperature": 0.5})
        assert body["max_tokens"] == 4096

    def test_an_explicit_null_is_filled_rather_than_written(self):
        """#1459: a null never reaches the wire."""
        body, ctx = _write(generation={"max_tokens": None})
        assert body["max_tokens"] == 4096
        assert ctx.metadata["injected"] == {"max_tokens": 4096}
        assert any("explicit null" in str(w) for w in ctx.warnings)

    def test_a_client_value_is_kept_and_not_recorded(self):
        body, ctx = _write(generation={"max_tokens": 123})
        assert body["max_tokens"] == 123
        assert "injected" not in ctx.metadata

    def test_a_removed_declaration_fails_loudly(self):
        """A converter with no default cannot produce a valid Messages body."""

        class Undeclared(MessagesConverter):
            request_constraints = RequestConstraints()

        with pytest.raises(ValueError, match="declares no default"):
            _write(Undeclared())


class TestShimOverride:
    def test_an_empty_shim_declaration_is_byte_identical_to_no_shim(self):
        without, _ = _write()
        empty, _ = _write(
            constraints=MessagesConverter.request_constraints.merged_with(
                RequestConstraints()
            )
        )
        assert empty == without

    def test_a_shim_value_replaces_the_dialect_default(self):
        constraints = MessagesConverter.request_constraints.merged_with(
            RequestConstraints(defaults={"generation.max_tokens": 2048})
        )
        body, _ = _write(constraints=constraints)
        assert body["max_tokens"] == 2048

    def test_an_empty_shim_range_leaves_the_dialect_range_in_place(self):
        """Merging is per path: a shim that declares nothing changes nothing."""
        constraints = MessagesConverter.request_constraints.merged_with(
            RequestConstraints(ranges={})
        )
        body, _ = _write(
            generation={"temperature": 1.8},
            constraints=constraints,
        )
        assert body["temperature"] == 1.0

    def test_a_shim_range_replaces_the_dialect_range(self):
        """A vendor that accepts 2.0 keeps 1.8; the dialect's ceiling does not win."""
        constraints = MessagesConverter.request_constraints.merged_with(
            RequestConstraints(
                ranges={"generation.temperature": Range(0.0, 2.0, "clamp")}
            )
        )
        body, ctx = _write(
            generation={"temperature": 1.8},
            constraints=constraints,
        )
        assert body["temperature"] == 1.8
        assert not [w for w in ctx.warnings if "outside the range" in str(w)]


class TestTemperatureRange:
    def test_a_value_above_the_ceiling_is_clamped_and_reported(self):
        body, ctx = _write(generation={"temperature": 1.8})
        assert body["temperature"] == 1.0
        assert any("outside the range" in str(w) for w in ctx.warnings)
        assert any(w.subject == "generation.temperature" for w in ctx.warnings)

    def test_a_value_inside_the_range_is_unchanged(self):
        body, ctx = _write(generation={"temperature": 0.3})
        assert body["temperature"] == 0.3
        assert not [w for w in ctx.warnings if "outside the range" in str(w)]
