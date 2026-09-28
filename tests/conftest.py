"""Root conftest — shared fixtures for the entire test suite."""

import pytest


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
            pytest.fail(
                f"Cache mutation detected in {name}_cache: "
                f"keys {corrupted} were modified after caching. "
                f"Cached values must not be mutated — see cache.py docstring."
            )

    clear_all_caches()
