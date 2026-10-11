"""Backward-compatibility tests for the deprecated ``openai_chat`` package.

The canonical module is ``llm_rosetta.converters.chat_completions``; the
``llm_rosetta.converters.openai_chat`` package (and the ``OpenAIChat*`` class
names) are retained as deprecated aliases.
"""

import importlib
import sys
import warnings

LEGACY = "llm_rosetta.converters.openai_chat"
CANONICAL = "llm_rosetta.converters.chat_completions"
_LEGACY_SUBMODULES = (
    "_constants",
    "config_ops",
    "content_ops",
    "converter",
    "message_ops",
    "tool_ops",
)


def _import_legacy_fresh():
    """Import the legacy package from scratch and capture its warnings."""
    for name in (LEGACY, *(f"{LEGACY}.{sub}" for sub in _LEGACY_SUBMODULES)):
        sys.modules.pop(name, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module(LEGACY)
    return module, caught


def test_legacy_package_emits_deprecation_warning():
    _module, caught = _import_legacy_fresh()
    messages = [
        str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert any("converters.openai_chat is deprecated" in m for m in messages)


def test_legacy_package_reexports_canonical_names():
    module, _ = _import_legacy_fresh()
    canonical = importlib.import_module(CANONICAL)
    assert module.ChatCompletionsConverter is canonical.ChatCompletionsConverter
    assert module.ChatCompletionsToolOps is canonical.ChatCompletionsToolOps
    assert module.ChatCompletionsContentOps is canonical.ChatCompletionsContentOps
    assert module.ChatCompletionsMessageOps is canonical.ChatCompletionsMessageOps
    assert module.ChatCompletionsConfigOps is canonical.ChatCompletionsConfigOps


def test_legacy_package_reexports_alias_names():
    module, _ = _import_legacy_fresh()
    canonical = importlib.import_module(CANONICAL)
    assert module.OpenAIChatConverter is canonical.ChatCompletionsConverter
    assert module.OpenAIChatToolOps is canonical.ChatCompletionsToolOps
    assert module.OpenAIChatContentOps is canonical.ChatCompletionsContentOps
    assert module.OpenAIChatMessageOps is canonical.ChatCompletionsMessageOps
    assert module.OpenAIChatConfigOps is canonical.ChatCompletionsConfigOps


def test_legacy_submodule_path_resolves():
    _import_legacy_fresh()
    # ``from llm_rosetta.converters.openai_chat.tool_ops import OpenAIChatToolOps``
    # resolves via the sys.modules alias registered by the compat package.
    legacy_tool_ops = importlib.import_module(f"{LEGACY}.tool_ops")
    canonical_tool_ops = importlib.import_module(f"{CANONICAL}.tool_ops")

    assert legacy_tool_ops is canonical_tool_ops
    assert (
        legacy_tool_ops.OpenAIChatToolOps is canonical_tool_ops.ChatCompletionsToolOps
    )


def test_legacy_submodule_objects_are_shared():
    _import_legacy_fresh()
    for sub in _LEGACY_SUBMODULES:
        legacy = importlib.import_module(f"{LEGACY}.{sub}")
        canonical = importlib.import_module(f"{CANONICAL}.{sub}")
        assert legacy is canonical


def test_ops_aliases_are_identical_objects():
    canonical = importlib.import_module(CANONICAL)
    assert canonical.OpenAIChatConverter is canonical.ChatCompletionsConverter
    assert canonical.OpenAIChatContentOps is canonical.ChatCompletionsContentOps
    assert canonical.OpenAIChatToolOps is canonical.ChatCompletionsToolOps
    assert canonical.OpenAIChatMessageOps is canonical.ChatCompletionsMessageOps
    assert canonical.OpenAIChatConfigOps is canonical.ChatCompletionsConfigOps
