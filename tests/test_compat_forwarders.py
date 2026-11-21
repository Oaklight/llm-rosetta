"""The 0.13.0 flat modules forward for one minor (#1657 compat decision).

``llm_rosetta.routing`` and ``llm_rosetta.capabilities`` shipped in 0.13.0 as
flat modules.  Each now forwards exactly its 0.13.0 public names, as the same
objects as their current homes, and warns once on import.  Both are removed in
0.15.0; the gateway and the shims may not import them (the boundary tests).
"""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest

ROUTING_NAMES = ["ProviderType", "ResolvedRoute", "Router"]
CAPABILITIES_NAMES = [
    "ConversionContext",
    "ProviderShim",
    "ReasoningCapability",
    "enforce_custom_tools",
    "enforce_reasoning",
    "enforce_vision",
    "get_custom_tool_names",
    "relocate_oversized_tool_descriptions",
    "resolve_shim",
    "restore_custom_tool_calls",
    "strip_reasoning_for_non_reasoning",
    "unwrap_custom_tool_input",
]


def _fresh_import(name: str):
    sys.modules.pop(name, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module(name)
    return module, [w for w in caught if issubclass(w.category, DeprecationWarning)]


@pytest.mark.parametrize(
    ("name", "names"),
    [
        ("llm_rosetta.routing", ROUTING_NAMES),
        ("llm_rosetta.capabilities", CAPABILITIES_NAMES),
    ],
)
def test_a_forwarder_warns_once_and_names_its_removal(name, names):
    module, caught = _fresh_import(name)
    assert len(caught) == 1
    message = str(caught[0].message)
    assert name in message
    assert "0.15.0" in message
    # stacklevel points at the importer, not at the forwarder itself.
    assert not caught[0].filename.endswith(name.rsplit(".", 1)[-1] + ".py")
    assert sorted(module.__all__) == sorted(names)
    assert "0.15.0" in (module.__doc__ or "")


def test_routing_forwards_the_contract_objects():
    module, _ = _fresh_import("llm_rosetta.routing")
    import llm_rosetta.contracts.routing as home

    for name in ROUTING_NAMES:
        assert getattr(module, name) is getattr(home, name), name


def test_capabilities_forwards_the_current_objects():
    module, _ = _fresh_import("llm_rosetta.capabilities")
    from llm_rosetta.contracts import shim
    from llm_rosetta.converters import base
    from llm_rosetta.ir import passes
    from llm_rosetta.shims import provider_shim

    assert module.ConversionContext is base.ConversionContext
    assert module.ProviderShim is shim.ProviderShim
    assert module.ReasoningCapability is shim.ReasoningCapability
    assert module.resolve_shim is provider_shim.resolve_shim
    for name in CAPABILITIES_NAMES[3:]:
        if name == "resolve_shim":
            continue
        assert getattr(module, name) is getattr(passes, name), name
