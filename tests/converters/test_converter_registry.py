"""The one converter registry: ``(model_type, format) → converter``."""

from __future__ import annotations

import pytest

from llm_rosetta.converters.registry import (
    CONVERTER_REGISTRY,
    FORMATS_BY_TYPE,
    MODEL_TYPES,
    formats_for,
    get_converter_class,
)


class TestRegistryShape:
    def test_the_four_categories(self) -> None:
        assert MODEL_TYPES == ("llm", "embedding", "rerank", "decision")

    def test_every_category_has_formats(self) -> None:
        for model_type in MODEL_TYPES:
            assert CONVERTER_REGISTRY[model_type], model_type

    def test_formats_by_type_is_derived(self) -> None:
        for model_type in MODEL_TYPES:
            assert FORMATS_BY_TYPE[model_type] == frozenset(
                CONVERTER_REGISTRY[model_type]
            )

    def test_format_lists_keep_registry_order(self) -> None:
        for model_type in MODEL_TYPES:
            assert formats_for(model_type) == tuple(CONVERTER_REGISTRY[model_type])


class TestLookup:
    def test_known_pair(self) -> None:
        from llm_rosetta.converters.chat_completions.converter import (
            ChatCompletionsConverter,
        )

        assert (
            get_converter_class("llm", "chat_completions") is ChatCompletionsConverter
        )

    def test_unknown_model_type_names_the_categories(self) -> None:
        with pytest.raises(ValueError, match="Unknown model type"):
            get_converter_class("speech", "whisper")

    def test_unknown_format_names_the_category_and_formats(self) -> None:
        with pytest.raises(ValueError, match="Unknown embedding format.*openai"):
            get_converter_class("embedding", "openai_embeddings")


class TestTheDescriptorsAgree:
    def test_each_descriptor_lists_its_registry(self) -> None:
        from llm_rosetta.gateway.model_types import all_model_types

        descriptors = {d.name: d for d in all_model_types()}
        assert set(descriptors) == set(MODEL_TYPES)
        for model_type, descriptor in descriptors.items():
            assert descriptor.formats == list(CONVERTER_REGISTRY[model_type]), (
                model_type
            )

    def test_chat_converter_lookup_uses_the_registry(self) -> None:
        from llm_rosetta.auto_detect import get_converter_for_provider
        from llm_rosetta.converters.chat_completions.converter import (
            ChatCompletionsConverter,
        )

        assert isinstance(
            get_converter_for_provider("chat_completions"), ChatCompletionsConverter
        )
