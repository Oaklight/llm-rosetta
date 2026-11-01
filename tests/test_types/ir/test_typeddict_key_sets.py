"""Key-set behaviour of the IR TypedDicts across supported Pythons.

``typing.TypedDict`` on Python 3.10 ignores the ``Required``/``NotRequired``
wrappers, so ``__required_keys__``/``__optional_keys__`` disagree with PEP 655
there.  The runtime validator reads the annotation-level wrapper first, so
``validate()`` and ``json_schema()`` behave the same on every version.  When the
floor moves to 3.11 the 3.10 test below is deleted and the key sets become
correct by themselves.

``BaseMessage`` is a sentinel, not the full set: many IR TypedDicts change key
sets on 3.10.  The wrapper-agreement sweep covers every exported IR TypedDict,
so a new wrapped type is checked without editing this file.
"""

import sys
from typing import get_origin, get_type_hints

import pytest

from llm_rosetta._vendor.validate import (
    ValidationError,
    _typeddict_fields,
    json_schema,
    validate,
)
from llm_rosetta.types.ir import BaseMessage


def _ir_typeddicts():
    import llm_rosetta.types.ir as ir

    return {
        name: obj
        for name, obj in vars(ir).items()
        if isinstance(obj, type) and hasattr(obj, "__required_keys__")
    }


@pytest.mark.skipif(sys.version_info < (3, 11), reason="3.10 deviates, see below")
def test_key_sets_are_pep655_accurate_on_311():
    assert set(BaseMessage.__required_keys__) == {"content"}
    assert set(BaseMessage.__optional_keys__) == {"metadata", "provider_metadata"}


@pytest.mark.skipif(sys.version_info >= (3, 11), reason="3.11+ honours the wrappers")
def test_key_sets_deviate_on_310():
    # typing.TypedDict on 3.10 does not understand the wrappers, so every key
    # is reported as required.
    assert set(BaseMessage.__required_keys__) == {
        "content",
        "metadata",
        "provider_metadata",
    }
    assert set(BaseMessage.__optional_keys__) == set()


def test_validator_requiredness_agrees_with_the_wrappers():
    # The sweep: every exported IR TypedDict that carries a wrapper must be
    # classified by the annotation, never by the (3.10-incorrect) key sets.
    for name, td in _ir_typeddicts().items():
        hints = get_type_hints(td, include_extras=True)
        for field, (_, is_required) in _typeddict_fields(td).items():
            explicit = getattr(get_origin(hints[field]), "_name", None)
            if explicit in ("Required", "NotRequired"):
                assert is_required is (explicit == "Required"), (name, field)


def test_json_schema_reads_the_wrappers_not_the_key_sets():
    assert json_schema(BaseMessage)["required"] == ["content"]


def test_validate_rejects_a_missing_required_key():
    # The runtime half of the same sentence; the schema check alone would not
    # catch a change to the annotation-first rule.
    with pytest.raises(ValidationError):
        validate({}, BaseMessage)
