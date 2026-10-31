"""The one converter registry: ``(model_type, format) → converter``."""

from __future__ import annotations

import pytest

from llm_rosetta.converters.registry import (
    CONVERTER_REGISTRY,
    FORMATS_BY_TYPE,
    MODEL_TYPES,
    converter_map,
    formats_for,
    get_converter_class,
)


class TestRegistryShape:
    def test_the_four_categories(self) -> None:
        assert MODEL_TYPES == ("llm", "embeddings", "rerank", "decisions")

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
        from llm_rosetta.converters.llm.chat_completions.converter import (
            ChatCompletionsConverter,
        )

        assert (
            get_converter_class("llm", "chat_completions") is ChatCompletionsConverter
        )

    def test_unknown_model_type_names_the_categories(self) -> None:
        with pytest.raises(ValueError, match="Unknown model type"):
            get_converter_class("speech", "whisper")

    def test_unknown_format_names_the_category_and_formats(self) -> None:
        with pytest.raises(ValueError, match="Unknown embeddings format.*openai"):
            get_converter_class("embeddings", "openai_embeddings")

    def test_the_legacy_category_spelling_resolves(self) -> None:
        """A caller that still says ``embedding`` gets the same converters."""
        from llm_rosetta.converters.embeddings.openai import OpenAIEmbeddingConverter
        from llm_rosetta.formats import LEGACY_MODEL_TYPES

        assert get_converter_class("embedding", "openai") is OpenAIEmbeddingConverter
        for legacy, canonical in LEGACY_MODEL_TYPES.items():
            assert converter_map(legacy) is converter_map(canonical), legacy


class TestTheDescriptorsAgree:
    def test_each_descriptor_lists_its_registry(self) -> None:
        from llm_rosetta_gateway.model_types import all_model_types

        descriptors = {d.name: d for d in all_model_types()}
        assert set(descriptors) == set(MODEL_TYPES)
        for model_type, descriptor in descriptors.items():
            assert descriptor.formats == list(CONVERTER_REGISTRY[model_type]), (
                model_type
            )

    def test_chat_converter_lookup_uses_the_registry(self) -> None:
        from llm_rosetta.auto_detect import get_converter_for_provider
        from llm_rosetta.converters.llm.chat_completions.converter import (
            ChatCompletionsConverter,
        )

        assert isinstance(
            get_converter_for_provider("chat_completions"), ChatCompletionsConverter
        )


class TestModuleLayout:
    """Where each family's packages live, and which old paths still resolve."""

    def test_llm_formats_live_under_the_llm_package(self) -> None:
        import importlib

        for fmt in FORMATS_BY_TYPE["llm"]:
            module = importlib.import_module(f"llm_rosetta.converters.llm.{fmt}")
            assert module.__name__ == f"llm_rosetta.converters.llm.{fmt}"

    def test_google_packages_keep_their_0130_paths(self) -> None:
        """``converters.google_generate`` was importable in 0.13.0.

        The implementations moved under ``converters.llm``; the old path is a
        stub package that warns on import and registers the new submodules
        under the old dotted names, so a class fetched by the old path is the
        one the registry holds, not a second copy.
        """
        import importlib
        import sys

        for name in ("google_generate", "google_interactions"):
            sys.modules.pop(f"llm_rosetta.converters.{name}", None)
            with pytest.warns(
                DeprecationWarning, match=f"converters.{name} is deprecated"
            ):
                old = importlib.import_module(f"llm_rosetta.converters.{name}")
            new = importlib.import_module(f"llm_rosetta.converters.llm.{name}")
            assert old.__all__ == new.__all__
            old_sub = importlib.import_module(
                f"llm_rosetta.converters.{name}.converter"
            )
            new_sub = importlib.import_module(
                f"llm_rosetta.converters.llm.{name}.converter"
            )
            assert old_sub is new_sub
            # The 0.13.0 attribute form (``gg.converter.X``) must resolve too,
            # not just the dotted import form registered in ``sys.modules``.
            assert old.converter is new.converter
        assert old_sub.GoogleInteractionsConverter is new.GoogleInteractionsConverter

    def test_other_llm_formats_have_no_legacy_top_level_path(self) -> None:
        """Those three names did not exist in 0.13.0, so they keep no alias."""
        import importlib.util

        for fmt in ("chat_completions", "messages", "open_responses"):
            assert importlib.util.find_spec(f"llm_rosetta.converters.{fmt}") is None, (
                fmt
            )


class TestBasePackage:
    """The shared base package: its public surface and its layering."""

    _PACKAGES = (
        "llm_rosetta.converters.base",
        "llm_rosetta.converters.base.helpers",
        "llm_rosetta.converters.base.tools",
    )

    def test_every_all_name_exists(self) -> None:
        """``__all__`` may not promise a name the package does not export."""
        import importlib

        for name in self._PACKAGES:
            module = importlib.import_module(name)
            missing = [n for n in module.__all__ if not hasattr(module, n)]
            assert missing == [], f"{name}: {missing}"

    def test_star_import_resolves(self) -> None:
        namespace: dict[str, object] = {}
        exec(  # noqa: S102 - the contract is the import itself
            "from llm_rosetta.converters.base import *\n"
            "from llm_rosetta.converters.base.helpers import *\n"
            "from llm_rosetta.converters.base.tools import *",
            namespace,
        )

    def test_base_does_not_import_a_family_package(self) -> None:
        """Families depend on ``base``; ``base`` must not depend on them.

        The direction is what keeps the shared layer free of per-type
        knowledge (and keeps the import graph acyclic); a helper that needs a
        family's code belongs in that family.
        """
        import ast
        import importlib.util
        from pathlib import Path

        spec = importlib.util.find_spec("llm_rosetta.converters.base")
        assert spec is not None and spec.origin is not None
        root = Path(spec.origin).parent
        families = (
            "llm_rosetta.converters.llm",
            "llm_rosetta.converters.embeddings",
            "llm_rosetta.converters.rerank",
            "llm_rosetta.converters.decisions",
            "llm_rosetta.converters.registry",
        )

        offences: list[str] = []
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            package = _module_name(path, root)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    target = _resolve_import(node, package)
                    if target and any(
                        target == family or target.startswith(family + ".")
                        for family in families
                    ):
                        offences.append(f"{path}: {target}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if any(
                            alias.name == family or alias.name.startswith(family + ".")
                            for family in families
                        ):
                            offences.append(f"{path}: {alias.name}")
        assert offences == [], offences


def _module_name(path, root) -> str:
    """``.../converters/base/helpers/x.py`` → ``...base.helpers``."""
    rel = path.relative_to(root)
    parts = ["llm_rosetta", "converters", "base", *rel.parts[:-1]]
    return ".".join(parts)


def _resolve_import(node, package: str) -> str | None:
    """Resolve an ``ImportFrom`` to an absolute module name."""
    if node.level == 0:
        return node.module
    parts = package.split(".")
    keep = len(parts) - (node.level - 1)
    if keep < 0:
        return None
    resolved = parts[:keep]
    if node.module:
        resolved += node.module.split(".")
    return ".".join(resolved)
