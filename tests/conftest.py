"""Root conftest — shared fixtures for the entire test suite."""

import pytest


@pytest.fixture(autouse=True)
def _clear_tool_conversion_caches():
    """Ensure each test starts and ends with clean conversion caches.

    On teardown, verifies that no test mutated a cached value (which
    would silently corrupt the cache in production).  Then clears all
    caches for the next test.
    """
    from llm_rosetta.converters.base.cache import (
        clear_all_caches,
        sanitize_cache,
        tool_entry_cache,
        validated_msg_cache,
    )

    clear_all_caches()
    yield

    # Mutation is a code bug — catch it here so it doesn't slip into prod.
    for name, cache in [
        ("tool_entry", tool_entry_cache),
        ("sanitize", sanitize_cache),
        ("validated_msg", validated_msg_cache),
    ]:
        corrupted = cache.check_integrity()
        if corrupted:
            pytest.fail(
                f"Cache mutation detected in {name}_cache: "
                f"keys {corrupted} were modified after caching. "
                f"Cached values must not be mutated — see cache.py docstring."
            )

    clear_all_caches()
