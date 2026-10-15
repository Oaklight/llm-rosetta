"""Tests for the intrinsic-kind helpers.

``intrinsic_kind`` is first-class on every carrier (part, event, definition);
``intrinsic_kind_of`` reads it first and still accepts the legacy metadata
spelling, and the constructors write only the first-class field.
"""

from llm_rosetta.converters.base.tools.intrinsic import (
    get_definition_kind,
    get_intrinsic_kind,
    intrinsic_kind_of,
    is_intrinsic,
    is_intrinsic_part,
    make_intrinsic_tool_call,
    make_intrinsic_tool_definition,
    make_intrinsic_tool_result,
)


# ---------------------------------------------------------------------------
# intrinsic_kind_of
# ---------------------------------------------------------------------------


def test_reads_the_first_class_field_on_a_part():
    part = {
        "type": "tool_call",
        "tool_type": "intrinsic",
        "intrinsic_kind": "web_search",
    }
    assert intrinsic_kind_of(part) == "web_search"


def test_falls_back_to_the_legacy_part_provider_metadata():
    part = {
        "type": "tool_result",
        "tool_type": "intrinsic",
        "provider_metadata": {"intrinsic_kind": "code_execution"},
    }
    assert intrinsic_kind_of(part) == "code_execution"


def test_first_class_field_wins_over_the_legacy_spelling():
    part = {
        "type": "tool_call",
        "tool_type": "intrinsic",
        "intrinsic_kind": "web_search",
        "provider_metadata": {"intrinsic_kind": "code_execution"},
    }
    assert intrinsic_kind_of(part) == "web_search"


def test_falls_back_to_the_legacy_definition_metadata():
    definition = {
        "type": "intrinsic",
        "name": "web_search",
        "metadata": {"intrinsic_kind": "web_search"},
    }
    assert intrinsic_kind_of(definition) == "web_search"


def test_reads_the_first_class_field_on_a_definition():
    definition = {
        "type": "intrinsic",
        "name": "web_search",
        "intrinsic_kind": "web_search",
    }
    assert intrinsic_kind_of(definition) == "web_search"


def test_missing_kind_reads_as_empty():
    assert intrinsic_kind_of({"type": "tool_call", "tool_type": "intrinsic"}) == ""
    assert (
        intrinsic_kind_of({"type": "intrinsic", "name": "web_search", "metadata": {}})
        == ""
    )
    assert intrinsic_kind_of({"provider_metadata": {"intrinsic_kind": ""}}) == ""
    assert intrinsic_kind_of("not a dict") == ""
    assert intrinsic_kind_of(None) == ""


def test_never_infers_the_kind_from_the_tool_name():
    part = {"type": "tool_call", "tool_type": "intrinsic", "tool_name": "web_search"}
    assert intrinsic_kind_of(part) == ""
    assert get_intrinsic_kind(part) == ""


def test_deprecated_aliases_read_the_same_thing():
    part = {
        "type": "tool_call",
        "tool_type": "intrinsic",
        "intrinsic_kind": "web_search",
    }
    definition = {
        "type": "intrinsic",
        "name": "web_search",
        "intrinsic_kind": "web_search",
    }
    assert get_intrinsic_kind(part) == intrinsic_kind_of(part) == "web_search"
    assert get_definition_kind(definition) == intrinsic_kind_of(definition)


# ---------------------------------------------------------------------------
# is_intrinsic / is_intrinsic_part
# ---------------------------------------------------------------------------


def test_is_intrinsic_covers_parts_events_and_definitions():
    assert is_intrinsic({"type": "tool_call", "tool_type": "intrinsic"})
    assert is_intrinsic({"type": "tool_result", "tool_type": "intrinsic"})
    assert is_intrinsic({"type": "tool_call_start", "tool_type": "intrinsic"})
    assert is_intrinsic({"type": "intrinsic", "name": "web_search"})


def test_is_intrinsic_rejects_client_tools_and_non_dicts():
    assert not is_intrinsic({"type": "tool_call", "tool_type": "function"})
    assert not is_intrinsic({"type": "function", "name": "get_weather"})
    assert not is_intrinsic(None)


def test_is_intrinsic_part_stays_parts_only():
    assert is_intrinsic_part({"type": "tool_call", "tool_type": "intrinsic"})
    assert is_intrinsic_part({"type": "tool_result", "tool_type": "intrinsic"})
    # Events and definitions are not "parts".
    assert not is_intrinsic_part({"type": "tool_call_start", "tool_type": "intrinsic"})
    assert not is_intrinsic_part({"type": "intrinsic", "name": "web_search"})


# ---------------------------------------------------------------------------
# Constructors never write the legacy key
# ---------------------------------------------------------------------------


def test_call_constructor_writes_the_field_not_the_legacy_key():
    part = make_intrinsic_tool_call(
        tool_call_id="c1",
        tool_name="web_search",
        tool_input={"query": "x"},
        intrinsic_kind="web_search",
    )
    assert part["intrinsic_kind"] == "web_search"
    assert "provider_metadata" not in part


def test_call_constructor_keeps_extra_metadata_apart_from_the_kind():
    part = make_intrinsic_tool_call(
        tool_call_id="c1",
        tool_name="web_search",
        tool_input={"query": "x"},
        intrinsic_kind="web_search",
        extra_metadata={"responses_status": "completed"},
    )
    assert part["intrinsic_kind"] == "web_search"
    assert part["provider_metadata"] == {"responses_status": "completed"}
    assert "intrinsic_kind" not in part["provider_metadata"]


def test_result_constructor_writes_the_field_not_the_legacy_key():
    part = make_intrinsic_tool_result(
        tool_call_id="c1",
        result="R",
        intrinsic_kind="code_execution",
        is_error=True,
        extra_metadata={"responses_status": "completed"},
    )
    assert part["intrinsic_kind"] == "code_execution"
    assert part["is_error"] is True
    assert part["provider_metadata"] == {"responses_status": "completed"}
    assert "intrinsic_kind" not in part["provider_metadata"]


def test_definition_constructor_writes_the_field_not_the_legacy_key():
    definition = make_intrinsic_tool_definition("web_search", description="d")
    assert definition["intrinsic_kind"] == "web_search"
    assert "metadata" not in definition


def test_definition_constructor_keeps_only_native_in_metadata():
    definition = make_intrinsic_tool_definition(
        "web_search",
        native={"type": "web_search"},
        native_base="open_responses",
    )
    assert definition["intrinsic_kind"] == "web_search"
    assert definition["metadata"] == {
        "_native": {"base": "open_responses", "tool": {"type": "web_search"}}
    }
    assert "intrinsic_kind" not in definition["metadata"]
