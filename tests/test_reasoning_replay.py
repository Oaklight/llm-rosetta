"""Reasoning replay is a model capability, not a vendor transform (#1143).

A thinking-mode target may require the model's reasoning to be passed back on
every assistant turn once tools are in play.  The requirement is declared on
the target model's ``ReasoningCapability`` and enforced by an IR pass, so it
applies whichever endpoint fronts the model.  The pass is format-neutral: it
adds an empty reasoning part and the target converter renders it in its own
dialect.
"""

from __future__ import annotations

from llm_rosetta.capabilities import backfill_reasoning_replay
from llm_rosetta.shims.provider_shim import (
    ProviderShim,
    ReasoningCapability,
    ReplayMissing,
)


def _cap(policy: ReplayMissing = "fill_empty") -> ReasoningCapability:
    return ReasoningCapability(replay_missing=policy)


def _text(value: str = "hi") -> dict:
    return {"type": "text", "text": value}


def _ir(messages: list[dict], *, tools: bool = True) -> dict:
    request: dict = {"messages": messages}
    if tools:
        request["tools"] = [{"type": "function", "name": "read"}]
    return request


def _reasoning_parts(message: dict) -> list[dict]:
    return [p for p in message["content"] if p.get("type") == "reasoning"]


class TestBackfillReasoningReplay:
    def test_fills_every_assistant_turn_ending_on_tool(self):
        request = _ir(
            [
                {"role": "user", "content": [_text()]},
                {"role": "assistant", "content": [_text("first")]},
                {"role": "tool", "content": []},
                {"role": "assistant", "content": [_text("second")]},
                {"role": "tool", "content": []},
            ]
        )

        out = backfill_reasoning_replay(request, reasoning_cap=_cap())

        assistants = [m for m in out["messages"] if m["role"] == "assistant"]
        assert len(assistants) == 2
        for message in assistants:
            parts = _reasoning_parts(message)
            assert len(parts) == 1
            assert parts[0]["reasoning"] == ""

    def test_keeps_existing_reasoning_and_does_not_duplicate(self):
        request = _ir(
            [
                {"role": "user", "content": [_text()]},
                {
                    "role": "assistant",
                    "content": [{"type": "reasoning", "reasoning": "kept"}, _text()],
                },
                {"role": "tool", "content": []},
            ]
        )

        out = backfill_reasoning_replay(request, reasoning_cap=_cap())

        parts = _reasoning_parts(out["messages"][1])
        assert [p["reasoning"] for p in parts] == ["kept"]

    def test_idempotent(self):
        request = _ir(
            [
                {"role": "user", "content": [_text()]},
                {"role": "assistant", "content": [_text()]},
                {"role": "tool", "content": []},
            ]
        )

        once = backfill_reasoning_replay(request, reasoning_cap=_cap())
        twice = backfill_reasoning_replay(once, reasoning_cap=_cap())

        assert _reasoning_parts(twice["messages"][1]) == [
            {"type": "reasoning", "reasoning": ""}
        ]

    def test_noop_when_history_ends_on_user(self):
        request = _ir(
            [
                {"role": "user", "content": [_text()]},
                {"role": "assistant", "content": [_text()]},
                {"role": "user", "content": [_text("next")]},
            ]
        )

        out = backfill_reasoning_replay(request, reasoning_cap=_cap())

        assert _reasoning_parts(out["messages"][1]) == []

    def test_fills_when_history_ends_on_assistant_with_tools(self):
        request = _ir(
            [
                {"role": "user", "content": [_text()]},
                {"role": "assistant", "content": [_text()]},
            ]
        )

        out = backfill_reasoning_replay(request, reasoning_cap=_cap())

        assert len(_reasoning_parts(out["messages"][1])) == 1

    def test_noop_when_history_ends_on_assistant_without_tools(self):
        request = _ir(
            [
                {"role": "user", "content": [_text()]},
                {"role": "assistant", "content": [_text()]},
            ],
            tools=False,
        )

        out = backfill_reasoning_replay(request, reasoning_cap=_cap())

        assert _reasoning_parts(out["messages"][1]) == []

    def test_noop_for_as_is_policy(self):
        request = _ir(
            [
                {"role": "user", "content": [_text()]},
                {"role": "assistant", "content": [_text()]},
                {"role": "tool", "content": []},
            ]
        )

        for cap in (None, _cap("as_is")):
            out = backfill_reasoning_replay(request, reasoning_cap=cap)
            assert _reasoning_parts(out["messages"][1]) == []

    def test_capability_is_the_only_gate(self):
        """A defaulted capability list must not suppress a declared replay.

        The gateway passes ``route.model_capabilities`` on every request and
        its default is ``["text"]``, so gating on that list would disable the
        fix for every model whose config entry does not spell out
        ``capabilities`` — including the DeepSeek lane #948 had fixed.  The
        pass therefore reads only ``replay_missing``: the declaration is the
        model-level fact, and the capability list says what a client may ask
        for, not what the upstream demands.
        """
        request = _ir(
            [
                {"role": "user", "content": [_text()]},
                {"role": "assistant", "content": [_text()]},
                {"role": "tool", "content": []},
            ]
        )

        out = backfill_reasoning_replay(request, reasoning_cap=_cap())

        assert len(_reasoning_parts(out["messages"][1])) == 1


class TestReplayCapabilityParsing:
    def test_defaults_to_as_is(self):
        assert ReasoningCapability().replay_missing == "as_is"

    def test_deepseek_and_opencode_declare_the_capability(self):
        from llm_rosetta.shims.provider_shim import get_shim, match_model_reasoning
        from llm_rosetta.shims.providers import load_providers

        load_providers()

        deepseek = get_shim("deepseek--chat_completions")
        assert deepseek is not None and deepseek.reasoning is not None
        assert deepseek.reasoning.replay_missing == "fill_empty"

        opencode = get_shim("opencode--chat_completions")
        assert opencode is not None and opencode.model_reasoning is not None
        override = match_model_reasoning(opencode, "deepseek-v4.1-flash")
        assert override is not None
        assert override.replay_missing == "fill_empty"
        # Family-scoped, not shim-wide: the same shim fronts Claude/GPT/Gemini.
        assert opencode.reasoning is not None
        assert opencode.reasoning.replay_missing == "as_is"
        assert match_model_reasoning(opencode, "claude-sonnet-4") is None


class TestModelFamilyMatching:
    """``model_overrides`` keys match exact IDs first, then glob patterns."""

    @staticmethod
    def _shim() -> ProviderShim:
        return ProviderShim(
            name="acme--chat_completions",
            format="chat_completions",
            model_reasoning={
                "deepseek-v4.1-flash": ReasoningCapability(replay_missing="as_is"),
                "deepseek-*": ReasoningCapability(replay_missing="fill_empty"),
            },
        )

    def test_exact_key_beats_a_pattern(self):
        from llm_rosetta.shims.provider_shim import match_model_reasoning

        cap = match_model_reasoning(self._shim(), "deepseek-v4.1-flash")
        assert cap is not None and cap.replay_missing == "as_is"

    def test_pattern_covers_the_family(self):
        from llm_rosetta.shims.provider_shim import match_model_reasoning

        for model in ("deepseek-v9", "deepseek-", "deepseek-v4.1-flash-free"):
            cap = match_model_reasoning(self._shim(), model)
            assert cap is not None, model
            assert cap.replay_missing == "fill_empty", model

    def test_non_matching_model_falls_through(self):
        from llm_rosetta.shims.provider_shim import match_model_reasoning

        assert match_model_reasoning(self._shim(), "claude-sonnet-4") is None
        assert match_model_reasoning(self._shim(), None) is None
        assert match_model_reasoning(self._shim(), "") is None
