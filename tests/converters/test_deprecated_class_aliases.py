"""Deprecated 0.13.0 class aliases warn from the caller's line, exactly once."""

import warnings

import pytest

import llm_rosetta
import llm_rosetta.converters as converters

_ALIASES = {
    "OpenAIChatConverter": "ChatCompletionsConverter",
    "AnthropicConverter": "MessagesConverter",
    "GoogleGenAIConverter": "GoogleGenerateConverter",
    "GoogleConverter": "GoogleGenerateConverter",
}


def _deprecations(caught):
    return [w for w in caught if issubclass(w.category, DeprecationWarning)]


def test_from_root_import_warns_once_at_caller():
    # The alias may already be bound by an earlier test.
    llm_rosetta.__dict__.pop("OpenAIChatConverter", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from llm_rosetta import OpenAIChatConverter

    assert OpenAIChatConverter is llm_rosetta.ChatCompletionsConverter
    dep = _deprecations(caught)
    assert len(dep) == 1
    assert dep[0].filename == __file__
    assert "llm_rosetta.OpenAIChatConverter is deprecated" in str(dep[0].message)


def test_from_converters_import_warns_once_at_caller():
    converters.__dict__.pop("AnthropicConverter", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from llm_rosetta.converters import AnthropicConverter

    assert AnthropicConverter is converters.MessagesConverter
    dep = _deprecations(caught)
    assert len(dep) == 1
    assert dep[0].filename == __file__
    assert "llm_rosetta.converters.AnthropicConverter is deprecated" in str(
        dep[0].message
    )


@pytest.mark.parametrize("alias,canonical", sorted(_ALIASES.items()))
def test_alias_resolves_to_canonical(alias, canonical):
    converters.__dict__.pop(alias, None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert getattr(converters, alias) is getattr(converters, canonical)
