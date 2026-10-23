"""Tests for the ``wire`` conformance suite.

The suite's own contract is two-sided: a loss the conversion reports is data,
and a loss nobody reports is a failure.  Both sides are pinned here, plus the
comparison helpers the report's `lost` list is built from and the ambient
wiring that lets a converter report at all.
"""

from __future__ import annotations

from typing import Any

import pytest

from tools.conformance import ProbeFailure
from tools.conformance.suites import SUITES
from tools.conformance.suites.wire import (
    CANONICAL_REQUEST,
    FORMATS,
    SAMPLING_REQUEST,
    _RoundTripProbe,
    build,
    differences,
    explained,
    normalize,
    project,
)


def _warning(subject: str, message: str) -> Any:
    """A stand-in for a ``ConversionWarning`` with a given subject."""
    from llm_rosetta.conversion_warnings import ConversionWarning

    return ConversionWarning(message, subject=subject)


class TestNormalize:
    def test_empty_defaults_are_dropped_at_every_depth(self):
        value = {"a": "", "b": None, "c": [], "d": {}, "e": [{"f": ""}], "g": 0}

        assert normalize(value) == {"g": 0}

    def test_real_values_survive(self):
        value = {"text": "hi", "items": ["a"], "flag": False}

        assert normalize(value) == value


class TestProject:
    def test_only_the_shapes_fields_survive(self):
        shape = {"model": "", "generation": {"temperature": 0}}
        value = {
            "model": "m",
            "generation": {"temperature": 0.2, "seed": 7},
            "metadata": {"responses_status": "completed"},
        }

        assert project(shape, value) == {
            "model": "m",
            "generation": {"temperature": 0.2},
        }

    def test_a_list_is_projected_with_its_first_element(self):
        shape = {"tools": [{"name": ""}]}

        assert project(shape, {"tools": [{"name": "a", "strict": False}]}) == {
            "tools": [{"name": "a"}]
        }

    def test_a_field_the_value_does_not_have_is_left_to_the_diff(self):
        assert project({"model": "", "tools": []}, {"model": "m"}) == {"model": "m"}


class TestDifferences:
    def test_a_lost_value_is_named_by_its_path(self):
        assert differences(
            {"generation": {"temperature": 0.2}}, {"generation": {}}
        ) == ["generation.temperature: lost"]

    def test_a_changed_value_is_shown_on_both_sides(self):
        assert differences({"model": "a"}, {"model": "b"}) == ["model: 'a' → 'b'"]

    def test_a_shorter_list_is_reported(self):
        found = differences({"tools": [{"name": "a"}, {"name": "b"}]}, {"tools": []})

        assert found == ["tools: 2 → 0 items"]

    def test_equal_values_have_no_differences(self):
        assert differences(CANONICAL_REQUEST, CANONICAL_REQUEST) == []


class TestExplained:
    """A warning excuses a loss only when it names the field."""

    def test_a_subject_naming_the_field_explains_the_loss(self):
        warning = _warning("top_k", "OpenAI Chat does not support top_k; ignored")

        assert explained("generation.top_k: lost", [warning])

    def test_a_subject_naming_the_container_explains_the_loss(self):
        warning = _warning("generation", "the generation config was cleared")

        assert explained("generation.top_p: lost", [warning])

    def test_prose_explains_a_loss_when_the_warning_has_no_subject(self):
        assert explained(
            "generation.top_p: lost",
            ["Google Interactions does not support top_p; ignored"],
        )

    def test_a_subject_about_something_else_does_not(self):
        warning = _warning(
            "response_format",
            "Anthropic does not support response_format; use system "
            "instructions or tools instead",
        )

        assert not explained("tools: 1 → 0 items", [warning])
        assert not explained("tools[0].name: lost", [warning])
        assert not explained("messages[0].content[0].text: lost", [warning])

    def test_a_warning_with_a_subject_never_falls_back_to_prose(self):
        """The words may appear in the message; the subject is the evidence."""
        warning = _warning("refusal", "converting tools to a text block")

        assert not explained("tools[0].name: lost", [warning])

    def test_a_changed_value_needs_a_subject_naming_it(self):
        prose = "temperature is clamped by this route"

        assert not explained("generation.temperature: 0.2 → 0.4", [prose])
        assert explained(
            "generation.temperature: 0.2 → 0.4", [_warning("temperature", "clamped")]
        )


class TestTheSuite:
    def test_every_case_runs_in_every_format(self):
        probes, environment = build()

        assert len(probes) == 2 * len(FORMATS)
        assert environment["formats"] == list(FORMATS)
        assert environment["cases"] == ["request", "sampling"]

    def test_the_registry_exposes_it(self):
        assert SUITES["wire"] is build

    @pytest.mark.parametrize("provider", FORMATS)
    def test_the_canonical_request_survives_every_format(self, provider):
        data = _RoundTripProbe(provider, CANONICAL_REQUEST).run()

        assert data["lost"] == []

    @pytest.mark.parametrize("provider", FORMATS)
    def test_every_sampling_loss_is_reported(self, provider):
        data = _RoundTripProbe(provider, SAMPLING_REQUEST, "sampling").run()

        reported = [_warning(w["subject"], w["message"]) for w in data["warnings"]]
        assert all(explained(loss, reported) for loss in data["lost"])

    def test_an_ambient_report_is_what_explains_a_loss(self):
        """The probe must install the sink the way the pipeline does.

        The chat writer reports ``top_k`` through ``report_drop`` with no
        ``warnings=`` argument, so a probe that called the converter bare
        would report a reported loss as unreported.
        """
        request = {**CANONICAL_REQUEST, "generation": {"top_k": 20}}

        data = _RoundTripProbe("chat_completions", request).run()

        assert data["lost"] == ["generation.top_k: lost"]
        assert [w["subject"] for w in data["warnings"]] == ["top_k"]
        assert data["warnings"][0]["leg"] == "write"

    def test_a_loss_no_warning_names_fails_the_case(self):
        request: dict[str, Any] = {**CANONICAL_REQUEST, "sentinel": "kept?"}

        with pytest.raises(ProbeFailure, match="without a warning"):
            _RoundTripProbe("messages", request).run()

    def test_a_reported_loss_is_data_not_a_failure(self, monkeypatch):
        from llm_rosetta.converters.llm import messages as messages_module

        original = messages_module.MessagesConverter.request_from_provider

        def restore_without_temperature(self, body, context=None):
            restored = original(self, body, context=context)
            restored.get("generation", {}).pop("temperature", None)
            if context is not None:
                context.warnings.append("temperature was not carried back")
            return restored

        monkeypatch.setattr(
            messages_module.MessagesConverter,
            "request_from_provider",
            restore_without_temperature,
        )

        data = _RoundTripProbe("messages", CANONICAL_REQUEST).run()

        assert data["lost"] == ["generation.temperature: lost"]
        assert [w["leg"] for w in data["warnings"]] == ["read"]
