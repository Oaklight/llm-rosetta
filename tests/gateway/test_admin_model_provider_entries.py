"""Multi-provider model entries: merge, per-provider switches, and pinning.

These cover the three things the admin panel needs to show one model under
every provider that serves it: a second save merges instead of replacing, a
provider entry can be turned off, and a test request can be pinned to one
provider.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from llm_rosetta_gateway.admin.routes.config.model_providers import (
    _set_provider_attrs,
)
from llm_rosetta_gateway.admin.routes.config.models import (
    _declared_providers,
    _to_multi_provider_entry,
    _update_multi_provider_entry,
)
from llm_rosetta_gateway.config.parse import parse_multi_provider
from llm_rosetta_gateway.routing import ModelRoute, ProviderEntry
from llm_rosetta_gateway.routing.context import SelectionContext

if TYPE_CHECKING:
    from collections.abc import Mapping

    from llm_rosetta_gateway.transport.provider_info import ProviderInfo


class TestDeclaredProviders:
    """``_declared_providers`` reads both entry shapes."""

    def test_single_provider(self):
        assert _declared_providers({"provider": "a"}) == ["a"]

    def test_providers_list_of_names(self):
        assert _declared_providers({"providers": ["a", "b"]}) == ["a", "b"]

    def test_providers_list_of_objects(self):
        entry = {"providers": [{"name": "a"}, {"name": "b", "weight": 3}]}
        assert _declared_providers(entry) == ["a", "b"]

    def test_empty_entry(self):
        assert _declared_providers({}) == []


class TestMergeSecondProvider:
    """A second provider joins an existing model instead of replacing it."""

    def test_converts_single_provider_entry(self):
        existing = {"provider": "a", "capabilities": ["text", "vision"]}
        entry = _to_multi_provider_entry(existing, "b", {"capabilities": ["text"]})

        assert _declared_providers(entry) == ["a", "b"]
        assert "provider" not in entry
        assert entry["capabilities"] == ["text"]

    def test_moves_upstream_model_onto_its_provider(self):
        existing = {
            "provider": "a",
            "upstream_model": "a-upstream",
            "capabilities": ["text"],
        }
        entry = _to_multi_provider_entry(existing, "b", {"capabilities": ["text"]})

        assert entry["providers"][0] == {
            "name": "a",
            "weight": 1,
            "upstream_model": "a-upstream",
        }
        assert "upstream_model" not in entry

    def test_carries_the_model_enabled_flag(self):
        existing = {"provider": "a", "capabilities": ["text"], "enabled": False}
        entry = _to_multi_provider_entry(existing, "b", {"capabilities": ["text"]})

        assert entry["enabled"] is False


class TestUpdateMultiProviderEntry:
    """The merge flag adds the provider; a plain edit leaves the list alone."""

    def test_merge_appends_provider(self):
        entry = {"providers": [{"name": "a"}], "capabilities": ["text"]}
        _update_multi_provider_entry(entry, {"merge": True}, "b")

        assert _declared_providers(entry) == ["a", "b"]

    def test_merge_is_idempotent(self):
        entry = {"providers": [{"name": "a"}, {"name": "b"}]}
        _update_multi_provider_entry(entry, {"merge": True}, "b")

        assert _declared_providers(entry) == ["a", "b"]

    def test_edit_without_merge_keeps_the_list(self):
        entry = {"providers": [{"name": "a"}, {"name": "b"}]}
        _update_multi_provider_entry(entry, {"capabilities": ["text"]}, "b")

        assert _declared_providers(entry) == ["a", "b"]
        assert entry["capabilities"] == ["text"]


class TestProviderEntrySwitches:
    """``_set_provider_attrs`` turns one provider of a model on or off."""

    def test_disables_one_entry(self):
        plist = [{"name": "a"}, {"name": "b"}]
        assert _set_provider_attrs(plist, "b", None, False) is True

        assert plist[0] == {"name": "a"}
        assert plist[1] == {"name": "b", "enabled": False}

    def test_enabling_removes_the_key(self):
        plist = [{"name": "a"}, {"name": "b", "enabled": False}]
        assert _set_provider_attrs(plist, "b", None, True) is True

        assert plist[1] == {"name": "b"}

    def test_weight_and_enabled_together(self):
        plist = [{"name": "a"}]
        assert _set_provider_attrs(plist, "a", 3, False) is True

        assert plist[0] == {"name": "a", "weight": 3, "enabled": False}

    def test_weight_one_is_the_default_and_drops_the_key(self):
        plist = [{"name": "a", "weight": 3}]
        _set_provider_attrs(plist, "a", 1, None)

        assert plist[0] == {"name": "a"}

    def test_unknown_provider(self):
        plist = [{"name": "a"}]
        assert _set_provider_attrs(plist, "z", None, False) is False


class TestParseSkipsDisabledEntry:
    """A provider entry turned off takes no traffic."""

    PROVS = {"a": {}, "b": {}}

    def test_disabled_entry_is_skipped(self):
        route = parse_multi_provider(
            "m",
            {"providers": [{"name": "a"}, {"name": "b", "enabled": False}]},
            self.PROVS,
        )

        assert route is not None
        assert [p.name for p in route.providers] == ["a"]

    def test_all_entries_disabled_leaves_no_route(self):
        route = parse_multi_provider(
            "m",
            {
                "providers": [
                    {"name": "a", "enabled": False},
                    {"name": "b", "enabled": False},
                ]
            },
            self.PROVS,
        )

        assert route is None


class TestPinnedProvider:
    """A pinned request may only be served by the provider it names."""

    @staticmethod
    def _context(pin: str | None) -> SelectionContext:
        ctx = SelectionContext("m", "chat_completions")
        ctx.install_route(
            model_route=ModelRoute([ProviderEntry("a"), ProviderEntry("b")], "m"),
            # The pin logic reads the route, never the provider table.
            providers=cast("Mapping[str, ProviderInfo]", {"a": None, "b": None}),
            provider_types={},
            provider_shim_names={},
            provider_account_tokens={},
        )
        ctx.with_pinned_provider(pin)
        return ctx

    def test_unpinned_context_excludes_nothing(self):
        assert self._context(None).excluded_providers() == frozenset()

    def test_pin_excludes_every_other_provider(self):
        assert self._context("b").excluded_providers() == frozenset({"a"})

    def test_pin_stacks_with_failover_exclusions(self):
        ctx = self._context("b")
        ctx.with_failover_excluded({"a"})

        assert ctx.excluded_providers() == frozenset({"a"})
