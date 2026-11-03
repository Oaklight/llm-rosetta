"""Multi-provider model entries: merge, per-provider switches, and pinning.

These cover the three things the admin panel needs to show one model under
every provider that serves it: a second save merges instead of replacing, a
provider entry can be turned off, and a test request can be pinned to one
provider.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from llm_rosetta_gateway.admin.routes.config.model_providers import (
    _reorder_providers,
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

    def test_the_old_quota_key_is_normalised_away(self):
        """On is the default, so an old opt-in leaves no key behind."""
        entry = {"providers": [{"name": "a"}], "failover_on_quota": True}

        _update_multi_provider_entry(entry, {}, "a")

        assert "failover_on_error" not in entry
        assert "failover_on_quota" not in entry

    def test_a_body_that_turns_it_off_writes_false(self):
        entry = {"providers": [{"name": "a"}]}

        _update_multi_provider_entry(entry, {"failover_on_error": False}, "a")

        assert entry["failover_on_error"] is False

    def test_a_body_that_turns_it_on_removes_the_key(self):
        entry = {"providers": [{"name": "a"}], "failover_on_error": False}

        _update_multi_provider_entry(entry, {"failover_on_error": True}, "a")

        assert "failover_on_error" not in entry

    def test_an_existing_opt_out_survives_a_body_that_omits_it(self):
        entry = {"providers": [{"name": "a"}], "failover_on_error": False}

        _update_multi_provider_entry(entry, {}, "a")

        assert entry["failover_on_error"] is False

    def test_a_body_that_clears_the_flag_removes_it(self):
        entry = {"providers": [{"name": "a"}], "failover_on_error": False}

        _update_multi_provider_entry(entry, {"failover_on_error": None}, "a")

        assert "failover_on_error" not in entry


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


class TestReorderProviders:
    """The provider list is the route order, so a drag writes it in place."""

    def test_moves_an_entry_with_its_own_attributes(self):
        models = {
            "m": {
                "providers": [
                    {"name": "a", "weight": 3},
                    "b",
                    {"name": "c", "capabilities": ["vision"]},
                ]
            }
        }

        assert _reorder_providers(models, "m", ["c", "a", "b"]) is None
        assert models["m"]["providers"] == [
            {"name": "c", "capabilities": ["vision"]},
            {"name": "a", "weight": 3},
            "b",
        ]

    def test_a_disabled_lane_keeps_its_place(self):
        # The board does not list a lane the operator turned off, so a drag
        # cannot name it; it stays at its index while the others move.
        models = {
            "m": {
                "providers": [
                    "a",
                    {"name": "off", "enabled": False},
                    "b",
                    "c",
                ]
            }
        }

        assert _reorder_providers(models, "m", ["c", "a", "b"]) is None
        assert models["m"]["providers"] == [
            "c",
            {"name": "off", "enabled": False},
            "a",
            "b",
        ]

    def test_a_provider_that_is_not_in_the_document_keeps_its_place(self):
        # A disabled provider is dropped before the route is built, so the
        # lane is not on the board even though the entry still names it.
        models = {"m": {"providers": ["a", "gone", "b"]}}

        assert (
            _reorder_providers(
                models,
                "m",
                ["b", "a"],
                raw_providers={"a": {}, "b": {}, "gone": {"enabled": False}},
            )
            is None
        )
        assert models["m"]["providers"] == ["b", "gone", "a"]

    def test_a_full_list_may_name_the_hidden_entries_too(self):
        # The Models page's provider menu shows a switched-off provider, so
        # its drag sends every entry; that list is written as given.
        models = {"m": {"providers": ["a", {"name": "off", "enabled": False}, "b"]}}

        assert _reorder_providers(models, "m", ["b", "off", "a"]) is None
        assert models["m"]["providers"] == [
            "b",
            {"name": "off", "enabled": False},
            "a",
        ]

    def test_refuses_a_name_the_model_does_not_list(self):
        models = {"m": {"providers": ["a", "b"]}}

        error = _reorder_providers(models, "m", ["a", "z"])

        assert error is not None and error.status_code == 400
        assert "'z'" in error.body.decode()
        assert models["m"]["providers"] == ["a", "b"]

    def test_refuses_a_list_that_drops_a_provider(self):
        models = {"m": {"providers": ["a", "b"]}}

        error = _reorder_providers(models, "m", ["a"])

        assert error is not None and error.status_code == 400
        assert models["m"]["providers"] == ["a", "b"]

    def test_refuses_a_repeated_name(self):
        models = {"m": {"providers": ["a", "b"]}}

        error = _reorder_providers(models, "m", ["a", "a"])

        assert error is not None and error.status_code == 400

    def test_refuses_a_single_provider_entry(self):
        models = {"m": {"provider": "a"}}

        error = _reorder_providers(models, "m", ["a"])

        assert error is not None and error.status_code == 400
        assert models["m"] == {"provider": "a"}

    def test_refuses_a_body_that_is_not_a_list_of_names(self):
        models = {"m": {"providers": ["a", "b"]}}

        error = _reorder_providers(models, "m", [{"name": "b"}, "a"])

        assert error is not None and error.status_code == 400


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
