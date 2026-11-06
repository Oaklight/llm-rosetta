"""Root conftest — shared fixtures for the entire test suite."""

import pytest

# The teardown check below is only as good as the fingerprints it compares,
# and recording them is opt-in: production pays a canonical serialization per
# insert for a check it never runs (#1420).  The suite needs them, so the
# switch is turned on for the whole session before any test runs.
from llm_rosetta.converters.base.helpers import cache as _cache_module

_cache_module.RECORD_FINGERPRINTS = True


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item, nextitem):
    """Give every test the shim registry it started with.

    The registry is process-wide, and a suite that rebuilds it with its own
    fixtures (``tests/test_shim_sources.py``) would otherwise leak those shims
    into whatever runs next — the app-wiring tests, typically — so the same
    two files pass alone and fail together (#1332).

    A hook wrapper rather than an autouse fixture: a function-scoped fixture is
    set up *after* module-scoped ones, so a module that loads preset shims in
    its own ``sources`` fixture (``tests/integration/test_shim_presets_live.py``)
    would have its shims captured as the baseline and restored as if they were
    built-ins.  The protocol hook runs before any fixture and after every
    finalizer, so the snapshot is always the registry the previous test left.

    The restore is what makes file order irrelevant for every suite, not just
    the one that remembered to clean up.  ``tests/test_shim_registry_snapshot``
    witnesses that a registration does not outlive its test; the hook-versus-
    fixture ordering above is argued rather than witnessed, because the
    module-scoped fixture that distinguishes the two lives in
    ``tests/integration/``, which a default run skips.
    """
    from llm_rosetta.shims import restore_registry, snapshot_registry

    saved = snapshot_registry()
    try:
        return (yield)
    finally:
        restore_registry(saved)


@pytest.fixture(autouse=True)
def _clear_tool_conversion_caches():
    """Ensure each test starts and ends with clean conversion caches.

    On teardown, verifies that no test mutated a cached value (which
    would silently corrupt the cache in production).  Then clears all
    caches for the next test.
    """
    from llm_rosetta.converters.base.helpers.cache import (
        clear_all_caches,
        ir_validation_cache,
        sanitize_cache,
        tool_entry_cache,
    )

    clear_all_caches()
    yield

    # Mutation is a code bug — catch it here so it doesn't slip into prod.
    for name, cache in [
        ("tool_entry", tool_entry_cache),
        ("sanitize", sanitize_cache),
        ("ir_validation", ir_validation_cache),
    ]:
        corrupted = cache.check_integrity()
        if corrupted:
            msg = (
                f"Cache mutation detected in {name}_cache: "
                f"keys {corrupted} were modified after caching. "
                f"Cached values must not be mutated — see cache.py docstring."
            )
            pytest.fail(msg)  # ty: ignore[invalid-argument-type]

    clear_all_caches()
