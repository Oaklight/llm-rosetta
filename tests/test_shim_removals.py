"""Tests for the scheduled shim removals (epic #927).

The manifest drives three consumers — the startup advisory, the admin preset and
the CI gate — so it is pinned from both ends here: the entries themselves, and
the pure detection the consumers share.
"""

from __future__ import annotations

import os

import pytest

import llm_rosetta
from llm_rosetta.provider_names import LEGACY_SHIM_ALIASES, canonical_shim_name
from llm_rosetta.shims import (
    SHIM_REMOVALS,
    detect_shim_removals,
    get_shim,
    parse_version,
    removals_due,
    scan_removal_references,
)
from llm_rosetta.shims.provider_shim import _reset_registry
from llm_rosetta.shims.providers import load_providers

#: Pinned so a silently dropped entry fails here rather than quietly reducing
#: what the advisory covers.
EXPECTED_SHIMS = {
    "alcf_metis--chat_completions",
    "alcf_minerva--chat_completions",
    "alcf_sophia--chat_completions",
    "argo--chat_completions",
    "argo--messages",
    "argo--open_responses",
    "asksage--chat_completions",
    "asksage--messages",
    "asksage--open_responses",
    "asksage--google_generate",
}


@pytest.fixture
def empty_registry():
    """Clear the shim registry for one test, then put the built-ins back.

    Restoring matters: leaving it empty would make every later ``get_shim`` in
    the same process return ``None``, which is how ``tests/test_shim_loader.py``
    already breaks any file run after it.
    """
    _reset_registry()
    try:
        yield
    finally:
        _reset_registry()
        load_providers()


class TestManifest:
    """The entries themselves."""

    def test_manifest_is_exactly_the_expected_set(self):
        assert {entry.shim for entry in SHIM_REMOVALS} == EXPECTED_SHIMS

    def test_every_entry_is_complete(self):
        for entry in SHIM_REMOVALS:
            assert entry.shim
            assert parse_version(entry.removed_in) == (0, 16, 0)
            assert entry.source_repo.startswith("https://github.com/")
            assert entry.remote["url"] == entry.source_repo
            assert entry.remote["path"].endswith("/providers")
            # No `ref` while none of the three repos is tagged: an omitted ref is
            # resolved to a commit and pinned in the cache stamp instead.
            assert "ref" not in entry.remote

    def test_every_removed_shim_is_currently_registered(self):
        """The pre-removal half of the gate.

        Deleting a shim without deleting its manifest entry fails here.
        """
        load_providers()
        unregistered = [e.shim for e in SHIM_REMOVALS if get_shim(e.shim) is None]
        assert unregistered == []

    def test_every_legacy_name_still_resolves_to_its_shim(self):
        """The old spellings are the migration path, so they have to work."""
        load_providers()
        for entry in SHIM_REMOVALS:
            for legacy in entry.legacy_names:
                assert canonical_shim_name(legacy) == entry.shim, legacy
                assert legacy in LEGACY_SHIM_ALIASES, legacy
                assert get_shim(legacy) is get_shim(entry.shim), legacy

    def test_every_remote_round_trips_through_the_source_resolver(self):
        """A preset is only useful if the gateway would accept it."""
        from llm_rosetta.gateway.shim_sources import resolve_source_config

        for entry in SHIM_REMOVALS:
            _dirs, remotes, _cache = resolve_source_config(
                {"remotes": [dict(entry.remote)]}, None
            )
            assert remotes and remotes[0]["url"] == entry.remote["url"]

    def test_to_dict_keys(self):
        assert set(SHIM_REMOVALS[0].to_dict()) == {
            "shim",
            "removed_in",
            "remote",
            "source_repo",
            "legacy_names",
        }


class TestDetectShimRemovals:
    """The pure detection shared by the warning, the panel and the gate."""

    def test_current_provider_on_a_removed_shim_is_reported(self):
        advisories = detect_shim_removals(
            {"alcf": {"shim": "alcf_sophia--chat_completions"}}
        )
        assert len(advisories) == 1
        advisory = advisories[0]
        assert advisory.provider == "alcf"
        assert advisory.slot == "shim"
        assert advisory.shim == "alcf_sophia--chat_completions"
        assert advisory.removed_in == "0.16.0"
        assert advisory.remote["url"].endswith("llm-rosetta-shim-alcf")
        assert set(advisory.to_dict()) == {
            "provider",
            "slot",
            "shim",
            "removed_in",
            "remote",
            "source_repo",
        }

    def test_legacy_spelling_in_the_type_slot_is_reported(self):
        advisories = detect_shim_removals({"argo": {"type": "argo--openai_chat"}})
        assert [a.shim for a in advisories] == ["argo--chat_completions"]
        assert advisories[0].slot == "type"

    def test_a_provider_keyed_by_the_shim_is_reported(self):
        advisories = detect_shim_removals({"alcf--metis": {"api_key": "k"}})
        assert [a.shim for a in advisories] == ["alcf_metis--chat_completions"]
        assert advisories[0].slot == "name"

    def test_unrelated_and_base_format_providers_are_silent(self):
        assert (
            detect_shim_removals(
                {
                    "o": {"shim": "openai--chat_completions"},
                    "b": {"type": "chat_completions"},
                    "n": {"api_key": "k"},
                    "d": {"shim": "deepseek--chat_completions"},
                }
            )
            == []
        )

    def test_a_disabled_provider_still_counts(self):
        """Its config breaks the moment it is switched back on."""
        advisories = detect_shim_removals(
            {"argo": {"shim": "argo--messages", "enabled": False}}
        )
        assert len(advisories) == 1

    def test_a_declared_external_source_silences_its_own_shim(self):
        providers = {
            "alcf": {"shim": "alcf_sophia--chat_completions"},
            "argo": {"shim": "argo--messages"},
        }
        advisories = detect_shim_removals(
            providers, external_shims={"alcf_sophia--chat_completions"}
        )
        assert [a.shim for a in advisories] == ["argo--messages"]

    def test_is_sorted_by_provider(self):
        advisories = detect_shim_removals(
            {
                "zeta": {"shim": "asksage--messages"},
                "alpha": {"shim": "argo--messages"},
                "mid": {"shim": "alcf_metis--chat_completions"},
            }
        )
        assert [a.provider for a in advisories] == ["alpha", "mid", "zeta"]

    def test_empty_and_malformed_inputs(self):
        assert detect_shim_removals(None) == []
        assert detect_shim_removals({}) == []
        assert detect_shim_removals({"p": "not-a-mapping"}) == []

    def test_is_pure_and_needs_no_registry(self, empty_registry):
        """Resolution works with no registry at all.

        That is the property which lets this be a maintenance step (#1030): the
        answer comes from the config mapping alone, not from what happens to be
        loaded.
        """
        assert get_shim("alcf_sophia--chat_completions") is None
        advisories = detect_shim_removals({"alcf": {"shim": "alcf--sophia"}})
        assert [a.shim for a in advisories] == ["alcf_sophia--chat_completions"]

    def test_the_legacy_spelling_reports_the_canonical_shim(self):
        advisories = detect_shim_removals({"alcf": {"shim": "alcf--sophia"}})
        assert [a.shim for a in advisories] == ["alcf_sophia--chat_completions"]


class TestVersionGate:
    """When a removal becomes due — the self-arming half."""

    def test_parse_version(self):
        assert parse_version("0.14.0.dev17") == (0, 14, 0)
        assert parse_version("0.16.0") == (0, 16, 0)
        assert parse_version("0.16.0.dev1") == (0, 16, 0)
        assert parse_version("1.0") == (1, 0)
        assert parse_version("") == ()
        assert parse_version("not-a-version") == ()

    def test_nothing_is_due_before_the_removal(self):
        assert removals_due("0.14.0.dev17") == []
        assert removals_due("0.15.9") == []

    def test_everything_is_due_at_the_removal(self):
        assert len(removals_due("0.16.0")) == len(SHIM_REMOVALS)
        # A pre-release of the removal release is the first build that could
        # ship the deletion, so it arms the gate too.
        assert len(removals_due("0.16.0.dev1")) == len(SHIM_REMOVALS)

    def test_an_unparseable_version_never_arms(self):
        assert removals_due("") == []

    def test_the_gate_is_green_on_this_checkout(self):
        """The real-repo property: today the scan must find nothing.

        This is what makes the check self-arming — it flips on its own the day
        the version reaches the removal, with nobody remembering to enable it.
        """
        due = removals_due(llm_rosetta.__version__)
        assert due == []
        assert (
            scan_removal_references(
                ("examples", ".github/ci", "docker", "docs_en/docs", "docs_zh/docs"),
                removals=due,
            )
            == []
        )


class TestScanRemovalReferences:
    """The scan the gate is built on."""

    @staticmethod
    def _hits(found) -> list[tuple[str, int, str]]:
        return [(os.path.basename(f.path), f.line, f.shim) for f in found]

    @staticmethod
    def _examples(tmp_path):
        examples = tmp_path / "examples"
        examples.mkdir()
        return examples

    def test_finds_a_reference_in_a_config(self, tmp_path):
        examples = self._examples(tmp_path)
        (examples / "config.jsonc").write_text('{"shim": "alcf--sophia"}\n')
        assert self._hits(scan_removal_references((examples,))) == [
            ("config.jsonc", 1, "alcf--sophia")
        ]

    def test_canonical_names_count_too(self, tmp_path):
        examples = self._examples(tmp_path)
        (examples / "config.jsonc").write_text(
            '"shim": "alcf_sophia--chat_completions"\n'
        )
        assert len(scan_removal_references((examples,))) == 1

    def test_marker_lets_a_deliberate_mention_through(self, tmp_path):
        examples = self._examples(tmp_path)
        (examples / "migration.md").write_text(
            "| `alcf--sophia` | `alcf_sophia--chat_completions` |  # shim-removal-ok\n"
        )
        assert scan_removal_references((examples,)) == []

    def test_exempt_file_is_skipped_entirely(self, tmp_path):
        changelog = tmp_path / "changelog.md"
        changelog.write_text("`alcf--sophia` was removed in 0.16.0\n")
        assert len(scan_removal_references((changelog,))) == 1
        assert scan_removal_references((changelog,), exempt_files=(changelog,)) == []

    def test_exemption_survives_relocating_the_scanned_tree(self, tmp_path):
        """The exemption is about the file, not about where the caller put it.

        The gate's default exemption names the changelogs by their path from the
        repository root; a run whose roots point somewhere else — a relocated
        docs tree, a test's ``tmp_path`` — must not silently lose it, or the
        changelog's historical mentions become failures.
        """
        relocated = tmp_path / "elsewhere" / "en" / "docs" / "changelog.md"
        relocated.parent.mkdir(parents=True)
        relocated.write_text("`alcf--sophia` was removed in 0.16.0\n")

        assert len(scan_removal_references((tmp_path,))) == 1
        assert (
            scan_removal_references(
                (tmp_path,), exempt_files=("docs_en/docs/changelog.md",)
            )
            == []
        )

    def test_neighbours_and_substrings_are_not_hits(self, tmp_path):
        examples = self._examples(tmp_path)
        (examples / "notes.txt").write_text(
            "alcf--sophia-extra\n"
            "xalcf--sophia\n"
            "alcf_sophia--chat_completions_now\n"
            "gpt-4.1-nano\n"
            "xai\n"
        )
        assert scan_removal_references((examples,)) == []

    def test_missing_roots_and_skipped_file_types(self, tmp_path):
        examples = self._examples(tmp_path)
        (examples / "logo.svg").write_text("alcf--sophia")
        assert scan_removal_references((tmp_path / "nope", examples)) == []

    def test_results_are_sorted_by_path_then_line(self, tmp_path):
        examples = self._examples(tmp_path)
        (examples / "b.txt").write_text("alcf--metis\n")
        (examples / "a.txt").write_text("x\nalcf--sophia\n")
        assert [
            (os.path.basename(f.path), f.line)
            for f in scan_removal_references((examples,))
        ] == [("a.txt", 2), ("b.txt", 1)]

    def test_nested_directories_are_walked(self, tmp_path):
        examples = self._examples(tmp_path)
        nested = examples / "gateway"
        nested.mkdir()
        (nested / "alcf.jsonc").write_text(
            '{"shim": "alcf_sophia--chat_completions"}\n'
        )
        assert len(scan_removal_references((examples,))) == 1
