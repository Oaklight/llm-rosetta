"""Tests for the selection context and the stage chain."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from llm_rosetta_gateway.routing import (
    Identity,
    ModelRoute,
    ProviderEntry,
    RoutingChain,
    RoutingStage,
    SelectionContext,
    StageOrderError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from llm_rosetta_gateway.transport import ProviderInfo


def _ctx() -> SelectionContext:
    ctx = SelectionContext("gpt-4o", "chat_completions")
    ctx.install_route(
        model_route=ModelRoute([ProviderEntry("a"), ProviderEntry("b")], "gpt-4o"),
        providers={},
        provider_types={},
        provider_shim_names={},
        provider_account_tokens={},
    )
    return ctx


class _Recorder(RoutingStage):
    """A stage that records that it ran, for ordering tests."""

    name = "recorder"

    def __init__(self, log: list[str], label: str) -> None:
        self._log = log
        self._label = label

    def run(self, ctx: SelectionContext) -> None:
        self._log.append(self._label)


class TestIdentity:
    def test_defaults_are_empty(self):
        ident = Identity()
        assert ident.client is None
        assert ident.conversation is None

    def test_is_frozen(self):
        with pytest.raises(Exception):
            Identity(client="x").client = "y"  # ty: ignore[invalid-assignment]


class TestSelectionContext:
    def test_starts_with_only_its_inputs(self):
        ctx = SelectionContext("m", "messages")
        assert ctx.model == "m"
        assert ctx.source_format == "messages"
        assert ctx.identity == Identity()
        assert ctx.chosen is None
        assert ctx.upstream_session_id is None

    def test_records_the_client(self):
        ctx = _ctx()
        ctx.with_client("hash-1")
        assert ctx.identity.client == "hash-1"
        assert ctx.identity.conversation is None

    def test_records_the_conversation_after_the_client(self):
        ctx = _ctx()
        ctx.with_client("hash-1")
        ctx.with_conversation("first message")
        assert ctx.identity == Identity(client="hash-1", conversation="first message")

    def test_conversation_can_be_absent(self):
        ctx = _ctx()
        ctx.with_client(None)
        ctx.with_conversation(None)
        assert ctx.identity == Identity()

    def test_choosing_keeps_the_entry(self):
        ctx = _ctx()
        entry = ProviderEntry("b", upstream_model="up")
        ctx.choose(entry)
        assert ctx.chosen is entry

    def test_candidates_are_the_route_providers(self):
        assert [p.name for p in _ctx().candidates] == ["a", "b"]

    def test_candidates_without_a_route(self):
        assert SelectionContext("m", "messages").candidates == []

    def test_readiness_is_read_from_the_providers_table(self):
        ctx = _ctx()

        class _Info:
            def __init__(self, ready: bool) -> None:
                self.ready = ready

        ctx.providers = cast(
            "Mapping[str, ProviderInfo]", {"a": _Info(False), "b": _Info(True)}
        )
        assert ctx.is_ready("a") is False
        assert ctx.is_ready("b") is True


class TestRoutingChain:
    def test_runs_stages_in_order(self):
        log: list[str] = []
        chain = RoutingChain([_Recorder(log, "first"), _Recorder(log, "second")])
        chain.run(_ctx())
        assert log == ["first", "second"]

    def test_exposes_its_stages(self):
        log: list[str] = []
        stages = [_Recorder(log, "only")]
        assert RoutingChain(stages).stages == tuple(stages)

    def test_refuses_a_stage_that_needs_what_nothing_provides(self):
        class _Needy(RoutingStage):
            name = "needy"
            requires = frozenset({"conversation"})

            def run(self, ctx: SelectionContext) -> None: ...

        with pytest.raises(StageOrderError, match="needy.*conversation"):
            RoutingChain([_Needy()])

    def test_accepts_a_stage_whose_requirement_an_earlier_stage_provides(self):
        class _First(RoutingStage):
            name = "first"
            provides = frozenset({"conversation"})

            def run(self, ctx: SelectionContext) -> None:
                ctx.with_conversation("c")

        class _Second(RoutingStage):
            name = "second"
            requires = frozenset({"conversation"})

            def run(self, ctx: SelectionContext) -> None:
                assert ctx.identity.conversation == "c"

        RoutingChain([_First(), _Second()]).run(_ctx())

    def test_seeded_fields_need_no_producer(self):
        class _ReadsTheModel(RoutingStage):
            name = "reads-model"
            requires = frozenset({"model", "model_route"})

            def run(self, ctx: SelectionContext) -> None:
                assert ctx.model_route is not None

        RoutingChain([_ReadsTheModel()]).run(_ctx())

    def test_an_empty_chain_runs(self):
        RoutingChain([]).run(_ctx())
