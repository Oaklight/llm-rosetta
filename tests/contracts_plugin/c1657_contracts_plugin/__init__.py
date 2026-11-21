"""A shim plugin package used by ``tests/test_contracts_partial_init.py`` (#1657).

Plugin entry points run while ``import llm_rosetta`` is still executing, so
this package imports the contract at module scope on purpose: if a contract
module needed a name from a half-initialised ``llm_rosetta``, the entry point
would fail and the loader would only log it.
"""

from __future__ import annotations

from pathlib import Path

from llm_rosetta.contracts.shim import load_providers_from_dir

PROVIDERS_DIR = Path(__file__).parent / "providers"


def register_shims() -> list:
    """Register this package's shims and return what was registered."""
    return load_providers_from_dir(PROVIDERS_DIR)
