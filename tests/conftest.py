"""Root conftest — shared fixtures for the entire test suite."""

import pytest


@pytest.fixture(autouse=True)
def _clear_tool_conversion_caches():
    """Ensure each test starts and ends with clean tool conversion caches.

    Without this, a cached tool list from one test could leak into another
    and hide bugs (or cause false negatives when mutation occurs).
    """
    from llm_rosetta.converters.base.cache import clear_all_caches

    clear_all_caches()
    yield
    clear_all_caches()
