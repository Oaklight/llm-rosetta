"""Backward-compatibility shim for ``llm_rosetta.converters.base.cache``.

This module moved to ``llm_rosetta.converters.base.helpers.cache`` in v0.6.11.
It is re-exported here so the old import path keeps working.  The cache
singletons (``tool_entry_cache``, ``sanitize_cache``, ``ir_validation_cache``)
are the same objects as in the canonical module — importing from either path
shares one cache.

New code should import from ``llm_rosetta.converters.base.helpers`` instead.
"""

from .helpers.cache import (
    LRUCache,
    cache_info,
    clear_all_caches,
    entry_cache_key,
    get_cached_tool,
    ir_validation_cache,
    is_ir_validated,
    mark_ir_validated,
    put_cached_tool,
    sanitize_cache,
    schema_cache_key,
    tool_entry_cache,
)

__all__ = [
    "LRUCache",
    "cache_info",
    "clear_all_caches",
    "entry_cache_key",
    "get_cached_tool",
    "ir_validation_cache",
    "is_ir_validated",
    "mark_ir_validated",
    "put_cached_tool",
    "sanitize_cache",
    "schema_cache_key",
    "tool_entry_cache",
]
