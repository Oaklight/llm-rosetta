"""Tests for the shim transform layer."""

from __future__ import annotations

import pytest

from llm_rosetta.shims.provider_shim import (
    ProviderShim,
    _reset_registry,
    get_shim,
    register_shim,
)
from llm_rosetta.shims.transforms import (
    apply_transforms,
    default_message_field,
    rename_field,
    replace_message_field,
    set_defaults,
    strip_fields,
    strip_fields_for_model,
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
# ProviderShim with transforms
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

    def test_provider_shim_default_empty(self):
        s = ProviderShim(name="test", base="openai_chat")
        assert s.from_transforms == ()
        assert s.to_transforms == ()


# ---------------------------------------------------------------------------
# Built-in shim transforms
# ---------------------------------------------------------------------------


class TestBuiltinTransforms:
    @pytest.fixture(autouse=True)
    def _load_builtins(self):
        from llm_rosetta.shims.providers import load_providers

        load_providers()

    def test_volcengine_has_to_transforms(self):
        shim = get_shim("volcengine--openai_chat")
        assert shim is not None
        assert len(shim.to_transforms) > 0

    def test_volcengine_strips_logprobs(self):
        shim = get_shim("volcengine--openai_chat")
        assert shim is not None
        body = {"model": "test", "logprobs": True, "top_logprobs": 5, "messages": []}
        result = apply_transforms(shim.to_transforms, body)
        assert "logprobs" not in result
        assert "top_logprobs" not in result
        assert result["model"] == "test"

    def test_deepseek_strips_unsupported(self):
        shim = get_shim("deepseek")
        assert shim is not None
        body = {
            "model": "deepseek-chat",
            "n": 2,
            "logit_bias": {"50256": -100},
            "seed": 42,
            "temperature": 0.7,
            "messages": [],
        }
        result = apply_transforms(shim.to_transforms, body)
        assert "n" not in result
        assert "logit_bias" not in result
        assert "seed" not in result
        assert result["model"] == "deepseek-chat"
        assert result["temperature"] == 0.7

    def test_xai_strips_logit_bias(self):
        shim = get_shim("xai")
        assert shim is not None
        body = {
            "model": "grok-3",
            "logit_bias": {"50256": -100},
            "temperature": 1.0,
            "messages": [],
        }
        result = apply_transforms(shim.to_transforms, body)
        assert "logit_bias" not in result
        assert result["model"] == "grok-3"
        assert result["temperature"] == 1.0

    def test_moonshot_strips_unsupported(self):
        shim = get_shim("moonshot")
        assert shim is not None
        body = {
            "model": "moonshot-v1-8k",
            "logprobs": True,
            "top_logprobs": 3,
            "logit_bias": {},
            "seed": 99,
            "temperature": 0.5,
            "messages": [],
        }
        result = apply_transforms(shim.to_transforms, body)
        assert "logprobs" not in result
        assert "top_logprobs" not in result
        assert "logit_bias" not in result
        assert "seed" not in result
        assert result["model"] == "moonshot-v1-8k"
        assert result["temperature"] == 0.5

    def test_qwen_strips_unsupported(self):
        shim = get_shim("qwen")
        assert shim is not None
        body = {
            "model": "qwen-max",
            "frequency_penalty": 0.5,
            "logit_bias": {"50256": -100},
            "temperature": 0.7,
            "messages": [],
        }
        result = apply_transforms(shim.to_transforms, body)
        assert "frequency_penalty" not in result
        assert "logit_bias" not in result
        assert result["model"] == "qwen-max"
        assert result["temperature"] == 0.7

    def test_minimax_strips_unsupported(self):
        shim = get_shim("minimax--openai_chat")
        assert shim is not None
        body = {
            "model": "MiniMax-M2",
            "logprobs": True,
            "top_logprobs": 3,
            "seed": 42,
            "stop": ["END"],
            "temperature": 0.8,
            "messages": [],
        }
        result = apply_transforms(shim.to_transforms, body)
        assert "logprobs" not in result
        assert "top_logprobs" not in result
        assert "seed" not in result
        assert "stop" not in result
        assert result["model"] == "MiniMax-M2"
        assert result["temperature"] == 0.8

    def test_zhipu_strips_unsupported(self):
        shim = get_shim("zhipu")
        assert shim is not None
        body = {
            "model": "glm-4",
            "n": 2,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.5,
            "logprobs": True,
            "top_logprobs": 5,
            "logit_bias": {},
            "seed": 99,
            "temperature": 0.8,
            "messages": [],
        }
        result = apply_transforms(shim.to_transforms, body)
        assert "n" not in result
        assert "presence_penalty" not in result
        assert "frequency_penalty" not in result
        assert "logprobs" not in result
        assert "top_logprobs" not in result
        assert "logit_bias" not in result
        assert "seed" not in result
        assert result["model"] == "glm-4"
        assert result["temperature"] == 0.8


# ---------------------------------------------------------------------------
# Integration: convert() with transforms
# ---------------------------------------------------------------------------


class TestConvertWithTransforms:
    @pytest.fixture(autouse=True)
    def _load_builtins(self):
        from llm_rosetta.shims.providers import load_providers

        load_providers()

    def test_convert_applies_source_from_transforms(self):
        """Source shim's from_transforms should normalise before conversion."""
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
        result = convert(body, "anthropic", source_provider="custom-oai")
        assert "model" in result

    def test_convert_applies_target_to_transforms(self):
        """Target shim's to_transforms should adapt after conversion."""
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
        """Duplicate transforms should be harmless."""
        t = strip_fields("logprobs")
        custom = ProviderShim(
            name="idem-test",
            base="openai_chat",
            to_transforms=(t, t),
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
        )
        assert "logprobs" not in result


# ---------------------------------------------------------------------------
# Message-level transforms
# ---------------------------------------------------------------------------


class TestReplaceMessageField:
    def test_replaces_matching_value(self):
        t = replace_message_field("role", "developer", "system")
        body = {"messages": [{"role": "developer", "content": "hi"}]}
        result = t(body)
        assert result["messages"][0]["role"] == "system"

    def test_noop_on_non_matching_value(self):
        t = replace_message_field("role", "developer", "system")
        body = {"messages": [{"role": "user", "content": "hi"}]}
        result = t(body)
        assert result["messages"][0]["role"] == "user"

    def test_noop_on_no_messages(self):
        t = replace_message_field("role", "developer", "system")
        body = {"model": "gpt-4"}
        result = t(body)
        assert result == {"model": "gpt-4"}

    def test_multiple_messages(self):
        t = replace_message_field("role", "developer", "system")
        body = {
            "messages": [
                {"role": "developer", "content": "sys"},
                {"role": "user", "content": "hi"},
                {"role": "developer", "content": "more"},
            ]
        }
        result = t(body)
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][1]["role"] == "user"
        assert result["messages"][2]["role"] == "system"

    def test_idempotent(self):
        t = replace_message_field("role", "developer", "system")
        body = {"messages": [{"role": "developer", "content": "hi"}]}
        result1 = t(body)
        result2 = t(result1)
        assert result1 == result2

    def test_repr(self):
        t = replace_message_field("role", "developer", "system")
        assert repr(t) == "replace_message_field('role', 'developer', 'system')"


class TestDefaultMessageField:
    def test_sets_none_to_default(self):
        t = default_message_field("content", "")
        body = {"messages": [{"role": "assistant", "content": None}]}
        result = t(body)
        assert result["messages"][0]["content"] == ""

    def test_noop_on_existing_value(self):
        t = default_message_field("content", "")
        body = {"messages": [{"role": "user", "content": "hello"}]}
        result = t(body)
        assert result["messages"][0]["content"] == "hello"

    def test_sets_default_on_missing_field(self):
        t = default_message_field("content", "")
        body = {"messages": [{"role": "user"}]}
        result = t(body)
        # field absent, get() returns None → sets default
        assert result["messages"][0]["content"] == ""

    def test_noop_on_no_messages(self):
        t = default_message_field("content", "")
        body = {"model": "gpt-4"}
        result = t(body)
        assert result == {"model": "gpt-4"}

    def test_idempotent(self):
        t = default_message_field("content", "")
        body = {"messages": [{"role": "assistant", "content": None}]}
        result1 = t(body)
        result2 = t(result1)
        assert result1 == result2

    def test_repr(self):
        t = default_message_field("content", "")
        assert repr(t) == "default_message_field('content', '')"


class TestStripFieldsForModel:
    def test_strips_when_model_matches(self):
        t = strip_fields_for_model(r"^claudeopus47", "temperature")
        body = {"model": "claudeopus47", "temperature": 0.7, "messages": []}
        result = t(body)
        assert "temperature" not in result
        assert result["model"] == "claudeopus47"

    def test_strips_with_normalised_model(self):
        """Model name is normalised (lowercase, non-alnum stripped)."""
        t = strip_fields_for_model(r"^claudeopus47", "temperature")
        body = {"model": "Claude-Opus-4.7", "temperature": 0.7, "messages": []}
        result = t(body)
        assert "temperature" not in result

    def test_noop_when_model_no_match(self):
        t = strip_fields_for_model(r"^claudeopus47", "temperature")
        body = {"model": "gpt-4o", "temperature": 0.7, "messages": []}
        result = t(body)
        assert result["temperature"] == 0.7

    def test_noop_when_no_model(self):
        t = strip_fields_for_model(r"^claudeopus47", "temperature")
        body = {"temperature": 0.7, "messages": []}
        result = t(body)
        assert result["temperature"] == 0.7

    def test_multiple_keys(self):
        t = strip_fields_for_model(r"^claudeopus47", "temperature", "top_p")
        body = {"model": "claudeopus47", "temperature": 0.7, "top_p": 0.9}
        result = t(body)
        assert "temperature" not in result
        assert "top_p" not in result

    def test_idempotent(self):
        t = strip_fields_for_model(r"^claudeopus47", "temperature")
        body = {"model": "claudeopus47", "temperature": 0.7}
        result1 = t(body)
        result2 = t(result1)
        assert result1 == result2

    def test_repr(self):
        t = strip_fields_for_model(r"^claudeopus47", "temperature")
        assert repr(t) == "strip_fields_for_model('^claudeopus47', 'temperature')"


# ---------------------------------------------------------------------------
# Argo OpenAI Chat shim integration
# ---------------------------------------------------------------------------


class TestArgoOpenaiChatTransforms:
    @pytest.fixture(autouse=True)
    def _load_builtins(self):
        from llm_rosetta.shims.providers import load_providers

        load_providers()

    def test_argo_downgrades_developer_role(self):
        shim = get_shim("argo--openai_chat")
        assert shim is not None
        body = {
            "model": "gpt-4",
            "messages": [{"role": "developer", "content": "system prompt"}],
        }
        result = apply_transforms(shim.to_transforms, body)
        assert result["messages"][0]["role"] == "system"

    def test_argo_normalizes_null_content(self):
        shim = get_shim("argo--openai_chat")
        assert shim is not None
        body = {
            "model": "gpt-4",
            "messages": [
                {"role": "assistant", "content": None, "tool_calls": []},
            ],
        }
        result = apply_transforms(shim.to_transforms, body)
        assert result["messages"][0]["content"] == ""

    def test_argo_strips_temperature_for_opus47(self):
        shim = get_shim("argo--openai_chat")
        assert shim is not None
        body = {
            "model": "claudeopus47",
            "temperature": 0.7,
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = apply_transforms(shim.to_transforms, body)
        assert "temperature" not in result

    def test_argo_keeps_temperature_for_other_models(self):
        shim = get_shim("argo--openai_chat")
        assert shim is not None
        body = {
            "model": "gpt-4o",
            "temperature": 0.7,
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = apply_transforms(shim.to_transforms, body)
        assert result["temperature"] == 0.7

    def test_argo_renames_max_tokens(self):
        shim = get_shim("argo--openai_chat")
        assert shim is not None
        body = {
            "model": "gpt-4",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = apply_transforms(shim.to_transforms, body)
        assert "max_tokens" not in result
        assert result["max_completion_tokens"] == 100
