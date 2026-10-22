"""The moved telemetry paths warn, and resolve to the gateway's own modules.

Two properties carry the deprecation window (#1161), and neither is visible
from a passing suite unless it is asserted directly:

* an alias import emits ``DeprecationWarning`` naming the new home — delete the
  ``warnings.warn`` and every other test still passes;
* the alias returns the *same* module object, not a twin with a copied
  ``__dict__`` — a twin would duplicate exactly the state (the connection, the
  registry) the design exists to keep single, and no behavioural test would
  notice.

The tests unload only the *alias* name.  Evicting the real
``llm_rosetta_gateway.observability.<name>`` module would hand a second
import to every module that already holds a reference to the first, which
would make unrelated identity assertions in this suite fail for no reason.
"""

from __future__ import annotations

import importlib
import sys

import pytest

MOVED = [
    "capture",
    "error_dump",
    "maintenance",
    "metrics",
    "ops_log",
    "persistence",
    "profiling",
    "request_log",
    "retention",
    "sanitize",
]

ALIAS = "llm_rosetta.observability"
TARGET = "llm_rosetta_gateway.observability"


def _import_alias_fresh(name: str):
    """Import ``ALIAS.name`` with its alias key evicted, so the hook runs."""
    real = importlib.import_module(f"{TARGET}.{name}")
    sys.modules.pop(f"{ALIAS}.{name}", None)
    return real


@pytest.mark.parametrize("name", MOVED)
def test_an_alias_import_warns_and_names_the_new_home(name):
    alias_name = f"{ALIAS}.{name}"
    target_name = f"{TARGET}.{name}"
    _import_alias_fresh(name)

    with pytest.warns(DeprecationWarning) as record:
        aliased = importlib.import_module(alias_name)

    messages = [str(w.message) for w in record]
    assert any(alias_name in m and "is deprecated" in m for m in messages), messages
    # The warning is only useful if it says where the module went.
    assert any(f"use {target_name}" in m for m in messages), messages
    assert aliased is importlib.import_module(target_name)


@pytest.mark.parametrize("name", MOVED)
def test_the_alias_is_the_same_module_object(name):
    alias_name = f"{ALIAS}.{name}"
    target_name = f"{TARGET}.{name}"
    real = _import_alias_fresh(name)

    with pytest.warns(DeprecationWarning):
        aliased = importlib.import_module(alias_name)

    # A twin built from a copied ``__dict__`` would duplicate the store; the
    # identity is the contract, not the import succeeding.
    assert aliased is real
    assert aliased is sys.modules[target_name]


def test_the_package_root_forwards_moved_names_with_the_same_warning():
    facade = importlib.import_module(ALIAS)
    from llm_rosetta_gateway.observability import MetricsCollector as real
    from llm_rosetta_gateway.observability import metrics as real_module

    # ``__getattr__`` caches what it resolved; a cached name would skip the
    # import (and the warning) this test is about.
    facade.__dict__.pop("MetricsCollector", None)
    sys.modules.pop(f"{ALIAS}.metrics", None)

    with pytest.warns(DeprecationWarning, match="is deprecated"):
        collector = facade.MetricsCollector

    assert collector is real
    assert importlib.import_module(f"{ALIAS}.metrics") is real_module
