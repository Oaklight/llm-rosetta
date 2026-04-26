"""Tests for the shim transform layer."""

from __future__ import annotations

import pytest

from llm_rosetta.shims.provider_shim import (
    ModelShim,
    ProviderShim,
    _reset_registry,
    get_shim,
    register_shim,
)
from llm_rosetta.shims.transforms import (
    Transformable,
    apply_transforms,
    rename_field,
    resolve_transforms,
    set_defaults,
    strip_fields,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset the shim registry before and after each test."""
    _reset_registry()
    yield
    _reset_registry()


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


class TestStripFields:
    def test_removes_existing_fields(self):
        t = strip_fields("a", "b")
        result = t({"a": 1, "b": 2, "c": 3})
        assert result == {"c": 3}

    def test_noop_for_missing_fields(self):
        t = strip_fields("x", "y")
        body = {"a": 1}
        result = t(body)
        assert result == {"a": 1}

    def test_idempotent(self):
        t = strip_fields("a")
        body = {"a": 1, "b": 2}
        result1 = t(body)
        result2 = t(result1)
        assert result1 == result2 == {"b": 2}

    def test_empty_keys(self):
        t = strip_fields()
        body = {"a": 1}
        assert t(body) == {"a": 1}


class TestRenameField:
    def test_renames_existing_field(self):
        t = rename_field("old", "new")
        result = t({"old": 42, "other": 1})
        assert result == {"new": 42, "other": 1}

    def test_noop_for_missing_field(self):
        t = rename_field("old", "new")
        body = {"other": 1}
        result = t(body)
        assert result == {"other": 1}

    def test_idempotent(self):
        t = rename_field("a", "b")
        body = {"a": 1}
        result1 = t(body)
        result2 = t(result1)
        assert result1 == result2 == {"b": 1}


class TestSetDefaults:
    def test_sets_missing_fields(self):
        t = set_defaults(x=10, y=20)
        result = t({"z": 30})
        assert result == {"z": 30, "x": 10, "y": 20}

    def test_does_not_overwrite_existing(self):
        t = set_defaults(x=10)
        body = {"x": 99}
        result = t(body)
        assert result == {"x": 99}

    def test_idempotent(self):
        t = set_defaults(x=10)
        body = {}
        result1 = t(body)
        result2 = t(result1)
        assert result1 == result2 == {"x": 10}

    def test_empty_defaults(self):
        t = set_defaults()
        body = {"a": 1}
        assert t(body) == {"a": 1}


# ---------------------------------------------------------------------------
# Inspectability (repr)
# ---------------------------------------------------------------------------


class TestInspectability:
    def test_strip_fields_repr(self):
        t = strip_fields("a", "b")
        assert repr(t) == "strip_fields('a', 'b')"

    def test_rename_field_repr(self):
        t = rename_field("old", "new")
        assert repr(t) == "rename_field('old', 'new')"

    def test_set_defaults_repr(self):
        t = set_defaults(x=10, y="hello")
        assert repr(t) == "set_defaults(x=10, y='hello')"


# ---------------------------------------------------------------------------
# apply_transforms
# ---------------------------------------------------------------------------


class TestApplyTransforms:
    def test_empty_transforms(self):
        body = {"a": 1}
        assert apply_transforms((), body) == {"a": 1}

    def test_single_transform(self):
        result = apply_transforms((strip_fields("a"),), {"a": 1, "b": 2})
        assert result == {"b": 2}

    def test_ordered_composition(self):
        transforms = (
            rename_field("x", "y"),
            strip_fields("y"),
        )
        result = apply_transforms(transforms, {"x": 1, "z": 2})
        # rename x→y first, then strip y
        assert result == {"z": 2}

    def test_reverse_order_different_result(self):
        transforms = (
            strip_fields("x"),
            rename_field("x", "y"),
        )
        result = apply_transforms(transforms, {"x": 1, "z": 2})
        # strip x first (removes it), then rename x→y (noop)
        assert result == {"z": 2}

    def test_custom_callable(self):
        def double_value(body: dict) -> dict:
            if "n" in body:
                body["n"] *= 2
            return body

        result = apply_transforms((double_value,), {"n": 5})
        assert result == {"n": 10}


# ---------------------------------------------------------------------------
# resolve_transforms
# ---------------------------------------------------------------------------


class TestResolveTransforms:
    def test_provider_only(self):
        t1 = strip_fields("a")
        t2 = strip_fields("b")
        provider = ProviderShim(
            name="test",
            base="openai_chat",
            from_transforms=(t1,),
            to_transforms=(t2,),
        )
        from_t, to_t = resolve_transforms(provider, None)
        assert from_t == (t1,)
        assert to_t == (t2,)

    def test_merge_provider_and_model(self):
        pt = strip_fields("a")
        mt = strip_fields("b")
        provider = ProviderShim(name="test", base="openai_chat", from_transforms=(pt,))
        model = ModelShim("test-*", from_transforms=(mt,))
        from_t, to_t = resolve_transforms(provider, model)
        # provider first, model after
        assert from_t == (pt, mt)
        assert to_t == ()

    def test_merge_both_directions(self):
        p_from = strip_fields("a")
        p_to = strip_fields("b")
        m_from = rename_field("x", "y")
        m_to = set_defaults(z=1)
        provider = ProviderShim(
            name="test",
            base="openai_chat",
            from_transforms=(p_from,),
            to_transforms=(p_to,),
        )
        model = ModelShim(
            "test-*",
            from_transforms=(m_from,),
            to_transforms=(m_to,),
        )
        from_t, to_t = resolve_transforms(provider, model)
        assert from_t == (p_from, m_from)
        assert to_t == (p_to, m_to)

    def test_model_empty_transforms(self):
        pt = strip_fields("a")
        provider = ProviderShim(name="test", base="openai_chat", from_transforms=(pt,))
        model = ModelShim("test-*")  # no transforms
        from_t, to_t = resolve_transforms(provider, model)
        assert from_t == (pt,)
        assert to_t == ()

    def test_provider_empty_transforms(self):
        mt = strip_fields("b")
        provider = ProviderShim(name="test", base="openai_chat")
        model = ModelShim("test-*", from_transforms=(mt,))
        from_t, _ = resolve_transforms(provider, model)
        assert from_t == (mt,)


# ---------------------------------------------------------------------------
# Transformable protocol
# ---------------------------------------------------------------------------


class TestTransformable:
    def test_provider_shim_is_transformable(self):
        s = ProviderShim(name="test", base="openai_chat")
        assert isinstance(s, Transformable)

    def test_model_shim_is_transformable(self):
        m = ModelShim("test-*")
        assert isinstance(m, Transformable)


# ---------------------------------------------------------------------------
# ProviderShim / ModelShim with transforms
# ---------------------------------------------------------------------------


class TestShimWithTransforms:
    def test_provider_shim_stores_transforms(self):
        t1 = strip_fields("a")
        t2 = rename_field("b", "c")
        s = ProviderShim(
            name="test",
            base="openai_chat",
            from_transforms=(t1,),
            to_transforms=(t2,),
        )
        assert s.from_transforms == (t1,)
        assert s.to_transforms == (t2,)

    def test_model_shim_stores_transforms(self):
        t = set_defaults(x=1)
        m = ModelShim("test-*", from_transforms=(t,))
        assert m.from_transforms == (t,)
        assert m.to_transforms == ()

    def test_provider_shim_default_empty(self):
        s = ProviderShim(name="test", base="openai_chat")
        assert s.from_transforms == ()
        assert s.to_transforms == ()

    def test_model_shim_default_empty(self):
        m = ModelShim("test-*")
        assert m.from_transforms == ()
        assert m.to_transforms == ()


# ---------------------------------------------------------------------------
# Built-in shim transforms
# ---------------------------------------------------------------------------


class TestBuiltinTransforms:
    @pytest.fixture(autouse=True)
    def _load_builtins(self):
        from llm_rosetta.shims.builtins import _register_builtins

        _register_builtins()

    def test_volcengine_has_to_transforms(self):
        shim = get_shim("volcengine")
        assert shim is not None
        assert len(shim.to_transforms) > 0

    def test_volcengine_strips_logprobs(self):
        shim = get_shim("volcengine")
        assert shim is not None
        body = {"model": "test", "logprobs": True, "top_logprobs": 5, "messages": []}
        result = apply_transforms(shim.to_transforms, body)
        assert "logprobs" not in result
        assert "top_logprobs" not in result
        assert result["model"] == "test"


# ---------------------------------------------------------------------------
# Integration: convert() with transforms
# ---------------------------------------------------------------------------


class TestConvertWithTransforms:
    @pytest.fixture(autouse=True)
    def _load_builtins(self):
        from llm_rosetta.shims.builtins import _register_builtins

        _register_builtins()

    def test_convert_applies_source_from_transforms(self):
        """Source shim's from_transforms should normalise before conversion."""
        # Register a custom shim with a from_transform
        custom = ProviderShim(
            name="custom-oai",
            base="openai_chat",
            from_transforms=(rename_field("custom_field", "model"),),
        )
        register_shim(custom)

        from llm_rosetta import convert

        body = {
            "custom_field": "gpt-4",
            "messages": [{"role": "user", "content": "hello"}],
        }
        # Convert to anthropic — the rename should happen before converter reads it
        result = convert(body, "anthropic", source_provider="custom-oai")
        assert "model" in result

    def test_convert_applies_target_to_transforms(self):
        """Target shim's to_transforms should adapt after conversion."""
        # Register a custom target shim that strips a field
        custom = ProviderShim(
            name="custom-target",
            base="openai_chat",
            to_transforms=(strip_fields("logprobs"),),
        )
        register_shim(custom)

        from llm_rosetta import convert

        body = {
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
        }
        result = convert(body, "custom-target", source_provider="openai_chat")
        assert "logprobs" not in result

    def test_convert_with_model_transforms(self):
        """Model-level transforms should be merged with provider transforms."""
        model_t = set_defaults(extra="added_by_model")
        custom = ProviderShim(
            name="custom-model-test",
            base="openai_chat",
            to_transforms=(strip_fields("logprobs"),),
            models=(ModelShim("test-model-*", to_transforms=(model_t,)),),
        )
        register_shim(custom)

        from llm_rosetta import convert

        body = {
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
        }
        result = convert(
            body,
            "custom-model-test",
            source_provider="openai_chat",
            model="test-model-1",
        )
        # Provider transform strips logprobs, model transform adds extra
        assert "logprobs" not in result
        assert result.get("extra") == "added_by_model"

    def test_convert_without_shim_still_works(self):
        """Base type conversion without shim should work as before."""
        from llm_rosetta import convert

        body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hello"}],
        }
        result = convert(body, "anthropic", source_provider="openai_chat")
        assert "messages" in result

    def test_convert_idempotent_transforms(self):
        """Duplicate transforms between provider and model should be harmless."""
        t = strip_fields("logprobs")
        custom = ProviderShim(
            name="idem-test",
            base="openai_chat",
            to_transforms=(t,),
            models=(ModelShim("test-*", to_transforms=(t,)),),
        )
        register_shim(custom)

        from llm_rosetta import convert

        body = {
            "model": "test",
            "logprobs": True,
            "messages": [{"role": "user", "content": "hello"}],
        }
        result = convert(
            body,
            "idem-test",
            source_provider="openai_chat",
            model="test-model",
        )
        assert "logprobs" not in result
