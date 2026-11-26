"""Helpers for tests that change the shim registry.

A test that registers shims or loads extensions changes process-wide state.
:func:`snapshot_registry` takes a copy of the registry and what is derived from
it, and :func:`restore_registry` puts it back, so the next test starts from
what it expects.  The extension repositories use these two in their
``conftest.py``.

Like :mod:`llm_rosetta.api`, this module is for callers of core.  Core never
imports it (``tests/test_core_import_boundary.py`` guards this).
"""

from __future__ import annotations

from llm_rosetta.shims.provider_shim import (
    RegistrySnapshot,
    restore_registry,
    snapshot_registry,
)

__all__ = ["RegistrySnapshot", "restore_registry", "snapshot_registry"]
