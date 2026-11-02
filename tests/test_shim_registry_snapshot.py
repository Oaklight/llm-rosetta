"""The registry snapshot the suite restores between tests (#1332).

A unit suite that rebuilds the shim registry with its own fixtures used to leak
that registry into whatever ran next, so two files passed alone and failed
together.  The root conftest restores a snapshot around every test; these tests
pin both the helpers and the fixture.
"""

from __future__ import annotations

from llm_rosetta.shims import (
    get_shim,
    list_shims,
    register_shim,
    restore_registry,
    snapshot_registry,
)
from llm_rosetta.shims.provider_shim import ProviderShim
from llm_rosetta.shims.providers import (
    _model_list_transforms,
    register_model_list_transform,
)


def _fake(name: str) -> ProviderShim:
    return ProviderShim(name=name, format="chat_completions")


class TestSnapshotAndRestore:
    def test_restore_removes_a_registration_made_after_the_snapshot(self):
        saved = snapshot_registry()
        try:
            register_shim(_fake("acme--snapshot"))
            assert get_shim("acme--snapshot") is not None
        finally:
            restore_registry(saved)
        assert get_shim("acme--snapshot") is None
        assert {shim.name for shim in list_shims()} == set(saved.registry)

    def test_a_snapshot_is_a_copy(self):
        saved = snapshot_registry()
        saved.registry["acme--copy"] = _fake("acme--copy")
        assert get_shim("acme--copy") is None

    def test_restore_puts_the_model_list_transforms_back(self):
        # Transforms are registered while shims load, not rebuilt on demand,
        # so a restore has to carry them: clearing them would leave the
        # built-ins without theirs for the rest of the process.
        saved = snapshot_registry()
        try:
            register_model_list_transform("acme--snapshot", lambda entries: ([], {}))
            assert "acme--snapshot" in _model_list_transforms
        finally:
            restore_registry(saved)
        assert "acme--snapshot" not in _model_list_transforms
        assert set(_model_list_transforms) == set(saved.model_list_transforms)

    def test_restore_drops_a_converter_cached_for_a_removed_shim(self):
        from llm_rosetta.auto_detect import _converter_cache

        saved = snapshot_registry()
        _converter_cache["acme--snapshot"] = object()
        try:
            restore_registry(saved)
            assert "acme--snapshot" not in _converter_cache
        finally:
            _converter_cache.clear()


#: Names registered by the tests below, in the order they ran.  Each test must
#: find every earlier name gone: whichever runs second is the witness, so a
#: random test order cannot turn the pair into two no-ops.
_LEAKED: list[str] = []


def _register_and_forget() -> None:
    """Register a shim, expecting the previous test's one to be gone."""
    survivors = [name for name in _LEAKED if get_shim(name) is not None]
    assert survivors == [], f"a registration survived its test: {survivors}"
    name = f"acme--leak-{len(_LEAKED)}"
    _LEAKED.append(name)
    register_shim(_fake(name))
    assert get_shim(name) is not None


class TestTheFixtureRestores:
    def test_a_registration_does_not_outlive_its_test(self):
        _register_and_forget()

    def test_a_later_test_starts_from_the_registry_it_was_given(self):
        _register_and_forget()
        assert list_shims()
