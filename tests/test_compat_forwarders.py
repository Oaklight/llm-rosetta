"""The 0.13.0 compat modules forward for one minor (#1657, #919 decisions).

``llm_rosetta.routing`` and ``llm_rosetta.capabilities`` shipped in 0.13.0 as
flat modules, and ``llm_rosetta.types.ir.extensions`` as a deprecated
forwarder.  Each now forwards exactly its 0.13.0 public names, as the same
objects as their current homes, and warns once on import.  All three are
removed in 0.15.0; the gateway and the shims may not import them (the boundary
tests).
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


#: ``llm_rosetta.types.ir.extensions.__all__`` on the v0.13.0 tag.
EXTENSIONS_NAMES = [
    "SystemEvent",
    "BatchMarker",
    "SessionControl",
    "ToolChainNode",
    "ExtensionItem",
    "is_extension_item",
    "is_extension_type",
    "get_extension_type",
    "isinstance_extension",
    "EXTENSION_TYPE_MAP",
]


def _fresh_import(name: str):
    parent = name.rpartition(".")[0]
    if parent == "llm_rosetta.types.ir":
        # A submodule of the forwarder package imports the package first, which
        # warns for itself; import it quietly so only the module's own warning
        # is counted.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            importlib.import_module(parent)
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
        ("llm_rosetta.types.ir.extensions", EXTENSIONS_NAMES),
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


def test_extensions_forwards_the_experimental_objects():
    module, _ = _fresh_import("llm_rosetta.types.ir.extensions")
    import llm_rosetta.ir.model.extensions_experimental as home

    for name in EXTENSIONS_NAMES:
        assert getattr(module, name) is getattr(home, name), name


def test_extensions_is_in_the_compat_tier():
    from llm_rosetta import _tiers

    assert "llm_rosetta.types.ir.extensions" in _tiers.COMPAT


def test_capabilities_forwards_the_current_objects():
    module, _ = _fresh_import("llm_rosetta.capabilities")
    from llm_rosetta.contracts import shim
    from llm_rosetta.converters import base
    from llm_rosetta.ir import passes
    from llm_rosetta.contracts import resolver

    assert module.ConversionContext is base.ConversionContext
    assert module.ProviderShim is shim.ProviderShim
    assert module.ReasoningCapability is shim.ReasoningCapability
    assert module.resolve_shim is resolver.resolve_shim
    for name in CAPABILITIES_NAMES[3:]:
        if name == "resolve_shim":
            continue
        assert getattr(module, name) is getattr(passes, name), name


#: ``llm_rosetta.types.ir.__all__`` on the v0.13.0 tag: the IR merged into
#: ``llm_rosetta.ir.model`` (#1706), and the old path forwards these names only.
TYPES_IR_NAMES_013 = [
    "ContentPart",
    "TextPart",
    "ImagePart",
    "ImageData",
    "FilePart",
    "FileData",
    "ToolCallPart",
    "ToolResultPart",
    "ReasoningPart",
    "RefusalPart",
    "CitationPart",
    "AudioPart",
    "AudioData",
    "SystemContentPart",
    "UserContentPart",
    "AssistantContentPart",
    "ToolContentPart",
    "Message",
    "BaseMessage",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolMessage",
    "LegacyMessage",
    "MessageMetadata",
    "StreamingMetadata",
    "ToolDefinition",
    "ToolChoice",
    "ToolCallConfig",
    "GenerationConfig",
    "ResponseFormatConfig",
    "StreamConfig",
    "ReasoningConfig",
    "ReasoningEffortLevel",
    "CacheConfig",
    "IRInputItem",
    "IRRequest",
    "IRResponse",
    "ExtensionItem",
    "ProviderPassthroughItem",
    "ProviderPassthroughEvent",
    "UsageInfo",
    "FinishReason",
    "ChoiceInfo",
    "IRStreamEvent",
    "is_provider_passthrough_event",
    "is_provider_passthrough_item",
    "StreamStartEvent",
    "StreamEndEvent",
    "ContentBlockStartEvent",
    "ContentBlockEndEvent",
    "TextDeltaEvent",
    "ReasoningDeltaEvent",
    "RefusalDeltaEvent",
    "ToolCallStartEvent",
    "ToolCallDeltaEvent",
    "FinishEvent",
    "UsageEvent",
    "EmbeddingTaskType",
    "EmbeddingEncodingFormat",
    "EmbeddingUsageInfo",
    "EmbeddingItem",
    "IREmbeddingRequest",
    "IREmbeddingResponse",
    "RerankDocument",
    "RerankUsageInfo",
    "RerankResultItem",
    "IRRerankRequest",
    "IRRerankResponse",
    "IRInput",
    "IRInputSimple",
    "experimental",
]

#: The submodules ``llm_rosetta.types.ir`` had on the v0.13.0 tag (``extensions``
#: is a forwarder of its own, tested above).
TYPES_IR_SUBMODULES_013 = [
    "configs",
    "embedding",
    "extensions_experimental",
    "helpers",
    "messages",
    "parts",
    "passthrough",
    "reasoning",
    "request",
    "rerank",
    "response",
    "stream",
    "tools",
    "type_guards",
    "validation",
]


def test_types_ir_forwards_its_013_names_as_the_same_objects():
    module, caught = _fresh_import("llm_rosetta.types.ir")
    model = importlib.import_module("llm_rosetta.ir.model")
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "llm_rosetta.types.ir" in message
    assert "llm_rosetta.ir.model" in message
    assert "0.15.0" in message
    assert sorted(module.__all__) == sorted(TYPES_IR_NAMES_013)
    for name in TYPES_IR_NAMES_013:
        assert getattr(module, name) is getattr(model, name), name
    assert "0.15.0" in (module.__doc__ or "")


def test_types_ir_does_not_forward_names_that_first_shipped_in_0_14_0():
    module, _ = _fresh_import("llm_rosetta.types.ir")
    model = importlib.import_module("llm_rosetta.ir.model")
    new_only = set(model.__all__) - set(TYPES_IR_NAMES_013)
    assert new_only, "the model has no 0.14.0-only name left to check"
    for name in new_only:
        assert not hasattr(module, name), name


@pytest.mark.parametrize("sub", TYPES_IR_SUBMODULES_013)
def test_the_013_types_ir_submodules_forward_the_model_objects(sub):
    for name in [n for n in sys.modules if n.startswith("llm_rosetta.types.ir")]:
        del sys.modules[name]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old = importlib.import_module(f"llm_rosetta.types.ir.{sub}")
    new = importlib.import_module(f"llm_rosetta.ir.model.{sub}")
    assert old.__all__, sub
    for name in old.__all__:
        assert getattr(old, name) is getattr(new, name), (sub, name)
    assert any(
        issubclass(w.category, DeprecationWarning) and "0.15.0" in str(w.message)
        for w in caught
    )


def test_types_ir_is_in_the_compat_tier():
    from llm_rosetta import _tiers

    assert "llm_rosetta.types.ir" in _tiers.COMPAT
    assert "llm_rosetta.types.ir" not in _tiers.USER


#: ``llm_rosetta.converters.base.context`` on the v0.13.0 tag: the three names it
#: held.  The context moved to ``llm_rosetta.ir.context``; ``ImageFetcher`` and
#: the rest first shipped in 0.14.0 and are not forwarded.
CONTEXT_NAMES_013 = ["ConversionContext", "MetadataMode", "StreamContext"]


def test_the_old_context_path_forwards_its_013_names_and_warns():
    module, caught = _fresh_import("llm_rosetta.converters.base.context")
    canonical = importlib.import_module("llm_rosetta.ir.context")
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "llm_rosetta.converters.base.context" in message
    assert "llm_rosetta.ir.context" in message
    assert "0.15.0" in message
    assert sorted(module.__all__) == sorted(CONTEXT_NAMES_013)
    for name in CONTEXT_NAMES_013:
        assert getattr(module, name) is getattr(canonical, name), name
    assert not hasattr(module, "ImageFetcher")
    assert "0.15.0" in (module.__doc__ or "")


def test_the_old_context_path_is_in_the_compat_tier():
    from llm_rosetta import _tiers

    assert "llm_rosetta.converters.base.context" in _tiers.COMPAT


def test_importing_converters_base_does_not_warn():
    """The package keeps its names; only the old submodule path is deprecated."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::DeprecationWarning",
            "-c",
            "import llm_rosetta; import llm_rosetta.converters.base as b; "
            "assert b.ConversionContext and b.StreamContext and b.MetadataMode",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


#: Names ``llm_rosetta.shims.provider_shim`` held on the v0.13.0 tag that have a
#: new home, with that home.  The module is still the live registry, so it cannot
#: warn on import; asking for one of these warns.
PROVIDER_SHIM_MOVED_013 = {
    "IREffort": "llm_rosetta.ir.model.configs",
    "IRMode": "llm_rosetta.ir.model.configs",
    "IRTransform": "llm_rosetta.contracts.transform",
    "ProviderShim": "llm_rosetta.contracts.shim",
    "ReasoningCapability": "llm_rosetta.contracts.shim",
    "ToolSearchMode": "llm_rosetta.contracts.shim",
    "Transform": "llm_rosetta.contracts.transform",
    "UnsignedBlocks": "llm_rosetta.contracts.shim",
    "resolve_shim": "llm_rosetta.contracts.resolver",
}

#: Names first shipped in 0.14.0 that used to be re-exported by this module
#: during development; they are not forwarded.
PROVIDER_SHIM_NEW_IN_014 = [
    "BatchDeclaration",
    "ConnectionConfig",
    "QuotaDeclaration",
    "ReplayMissing",
    "SoftErrorPattern",
    "TYPE_PARAM_KEYS",
    "ToolsConfig",
    "VendorManifest",
    "match_model_reasoning",
    "resolve_format",
]


@pytest.mark.parametrize(("name", "home"), sorted(PROVIDER_SHIM_MOVED_013.items()))
def test_provider_shim_forwards_a_moved_013_name_and_warns(name, home):
    from llm_rosetta.shims import provider_shim

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = getattr(provider_shim, name)
    assert got is getattr(importlib.import_module(home), name)
    messages = [
        str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert len(messages) == 1
    assert name in messages[0]
    assert home in messages[0]
    assert "0.15.0" in messages[0]


@pytest.mark.parametrize("name", PROVIDER_SHIM_NEW_IN_014)
def test_provider_shim_does_not_forward_0_14_0_names(name):
    from llm_rosetta.shims import provider_shim

    with pytest.raises(AttributeError):
        getattr(provider_shim, name)


def test_provider_shim_keeps_the_registry_without_a_warning():
    from llm_rosetta.shims import provider_shim

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        for name in (
            "get_shim",
            "list_shims",
            "register_shim",
            "unregister_shim",
            "resolve_model_type",
            "staging_registry",
        ):
            assert callable(getattr(provider_shim, name))
