"""``staging_registry`` failure path: a raising block leaves no trace (#1630).

Since #1646 the context manager clears the derived caches on entry and, when
the block raises, puts the convention model-list transforms back.  These tests
pin the whole failure contract: the conversion table is restored to exactly
the state it had before the block (entries the block added are gone, entries
the clear removed are back), the live registry is unchanged on this thread,
and the block's own exception reaches the caller unchanged.
"""

from __future__ import annotations

import pytest

import llm_rosetta.api as api
from llm_rosetta.shims import ProviderShim
from llm_rosetta.shims.providers import _model_list_transforms


class _StagedBlockFailed(Exception):
    """A failure class no library code raises, so a match proves propagation."""


def _kept(models):  # type: ignore[no-untyped-def]
    return models, {}


def _added(models):  # type: ignore[no-untyped-def]
    return [], {}


def _shim(name: str) -> ProviderShim:
    return ProviderShim(name=name, format="chat_completions")


def _state() -> tuple[dict[str, str], dict[str, object]]:
    registry = {s.name: s.format for s in api.list_shims()}
    return registry, dict(_model_list_transforms)


def test_a_raising_block_restores_the_conversion_table_and_the_registry():
    api.register_shim(_shim("t-staging-kept"))
    _model_list_transforms["t-staging-kept"] = _kept  # type: ignore[assignment]
    before = _state()

    error = _StagedBlockFailed("staged block failed")
    with pytest.raises(_StagedBlockFailed) as caught:
        with api.staging_registry():
            # The block sees a clean slate, then writes into it.
            assert "t-staging-kept" not in _model_list_transforms
            api.register_shim(_shim("t-staging-added"))
            _model_list_transforms["t-staging-added"] = _added  # type: ignore[assignment]
            raise error

    assert caught.value is error
    assert _state() == before
    assert api.get_shim("t-staging-added") is None
    assert "t-staging-added" not in _model_list_transforms
    assert _model_list_transforms["t-staging-kept"] is _kept


def test_a_base_exception_in_the_block_also_restores_the_state():
    _model_list_transforms["t-staging-kept"] = _kept  # type: ignore[assignment]
    before = _state()

    with pytest.raises(KeyboardInterrupt):
        with api.staging_registry():
            api.register_shim(_shim("t-staging-added"))
            raise KeyboardInterrupt

    assert _state() == before
    assert api.get_shim("t-staging-added") is None
