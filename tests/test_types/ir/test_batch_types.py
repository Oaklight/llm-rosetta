"""The batch IR envelope: shape, and that it survives a JSON round trip.

The state and mode vocabularies are pinned here because every dialect's
``state_map`` translates into them; a renamed member silently re-labels a whole
dialect's lifecycle.
"""

from __future__ import annotations

import json
from typing import cast, get_args

import pytest

from llm_rosetta._vendor.validate import ValidationError, json_schema, validate
from llm_rosetta.types.ir import (
    IRRequest,
    IRResponse,
    BatchItem,
    BatchItemCounts,
    BatchItemError,
    BatchJob,
    BatchJobState,
    BatchMode,
    BatchResultItem,
)
from llm_rosetta.types.ir import batch as batch_module

_REQUEST = cast(
    IRRequest,
    {
        "model": "acme-1",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    },
)

_RESPONSE = cast(
    IRResponse,
    {
        "id": "resp-1",
        "object": "response",
        "created": 1_700_000_000,
        "model": "acme-1",
        "choices": [],
    },
)


def test_the_normalized_state_vocabulary_is_frozen():
    """Three vendors' lifecycles reduce to exactly these words."""
    assert get_args(BatchJobState) == (
        "validating",
        "queued",
        "running",
        "finalizing",
        "completed",
        "partially_completed",
        "cancelled",
        "expired",
        "failed",
    )


def test_mode_has_exactly_two_values():
    assert get_args(BatchMode) == ("native", "emulated")


def test_a_job_round_trips_through_json():
    job: BatchJob = {
        "id": "batch-1",
        "state": "partially_completed",
        "mode": "native",
        "counts": {"total": 3, "succeeded": 2, "expired": 1},
        "created_at": 1_700_000_000,
        "expires_at": 1_700_086_400,
        "results": [
            {"client_ref": "a", "response": _RESPONSE},
            {
                "client_ref": "b",
                "error": {
                    "code": "batch_expired",
                    "message": "expired",
                    "retryable": False,
                },
            },
            {"client_ref": "c", "response": None, "error": None},
        ],
    }

    assert json.loads(json.dumps(job)) == job


def test_expires_at_may_be_absent_or_null():
    """Gemini documents a 24-hour target, not a deadline."""
    job: BatchJob = {"id": "b", "state": "running", "mode": "native"}
    assert json.loads(json.dumps(job)) == job

    job["expires_at"] = None
    assert json.loads(json.dumps(job)) == job


def test_an_item_carries_the_converter_ir_unchanged():
    item: BatchItem = {"client_ref": "a", "request": _REQUEST}
    assert json.loads(json.dumps(item))["request"] == _REQUEST


def test_an_error_carries_code_message_and_optional_retryability():
    error: BatchItemError = {"code": "invalid_request", "message": "bad model"}
    assert json.loads(json.dumps(error)) == error

    # Read the annotation wrappers, not ``__optional_keys__``: on 3.10 the
    # stdlib TypedDict ignores them (see test_typeddict_key_sets.py).
    assert json_schema(BatchItemError)["required"] == ["code", "message"]
    with pytest.raises(ValidationError):
        validate({"code": "x"}, BatchItemError)


def test_counts_are_all_optional():
    """An absent key is "not reported", never zero."""
    schema = json_schema(BatchItemCounts)
    assert schema.get("required", []) == []
    assert set(schema["properties"]) == {
        "total",
        "processing",
        "succeeded",
        "errored",
        "cancelled",
        "expired",
    }
    # ``total=False`` means an empty mapping is a legal report.
    validate({}, BatchItemCounts)


def test_result_items_join_on_the_client_reference_only():
    result: BatchResultItem = {"client_ref": "a", "response": _RESPONSE}
    assert result["client_ref"] == "a"


def test_the_module_exports_its_whole_surface():
    exported = set(batch_module.__all__)
    assert exported == {
        "BatchItem",
        "BatchItemCounts",
        "BatchItemError",
        "BatchJob",
        "BatchJobState",
        "BatchMode",
        "BatchResultItem",
    }
    for name in exported:
        assert hasattr(batch_module, name), name
