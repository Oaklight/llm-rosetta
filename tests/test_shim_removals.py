"""Tests for the scheduled shim removals (epic #927).

The manifest drives three consumers — the startup advisory, the admin preset and
the CI gate — so it is pinned from both ends here: the entries themselves, and
the pure detection the consumers share.

Nothing is scheduled at the moment: ALCF, AskSage and Argo all ship standalone
from 0.14.0 and their names are served by ``shims.presets``.  The machinery is
kept self-arming, so the detection and gate tests use a synthetic manifest of
shims that are still built in.
"""

from __future__ import annotations

import os

import pytest

import llm_rosetta
from llm_rosetta.provider_names import LEGACY_SHIM_ALIASES, canonical_shim_name
from llm_rosetta.shims import (
    SHIM_REMOVALS,
    SHIM_SOURCE_PACKAGES,
    ShimRemoval,
    detect_shim_removals,
    get_shim,
    get_source_package,
    parse_version,
    removals_due,
    scan_removal_references,
)
from llm_rosetta.shims.provider_shim import _reset_registry
from llm_rosetta.shims.providers import load_providers

#: Synthetic manifest entries for the still-built-in ``minimax`` and ``deepseek``
#: shims.  Real entries would be deleted from the manifest the day the shims go
#: away, so the tests that pin the shared machinery cannot depend on one.
SAMPLE = ShimRemoval(
    shim="minimax--messages",
    removed_in="0.16.0",
    remote={
        "url": "https://github.com/Oaklight/llm-rosetta-shim-minimax",
        "path": "src/llm_rosetta_shim_minimax/providers",
    },
    source_repo="https://github.com/Oaklight/llm-rosetta-shim-minimax",
    legacy_names=("minimax--anthropic",),
)
SAMPLE_TWO = ShimRemoval(
    shim="deepseek--chat_completions",
    removed_in="0.16.0",
    remote={
        "url": "https://github.com/Oaklight/llm-rosetta-shim-deepseek",
        "path": "src/llm_rosetta_shim_deepseek/providers",
    },
    source_repo="https://github.com/Oaklight/llm-rosetta-shim-deepseek",
)
SAMPLES = (SAMPLE, SAMPLE_TWO)


def _detect(providers, **kwargs):
    """Call the detector against the synthetic manifest unless told otherwise."""
    kwargs.setdefault("removals", SAMPLES)
    return detect_shim_removals(providers, **kwargs)


def _scan(roots, **kwargs):
    """Scan against the synthetic manifest."""
    kwargs.setdefault("removals", SAMPLES)
    return scan_removal_references(roots, **kwargs)


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

    def test_no_removals_are_scheduled(self):
        """Pinned so a newly scheduled removal is a deliberate, reviewed edit.

        The previous three entries (ALCF, AskSage, Argo) were all due to the
        move to standalone packages in 0.14.0; a future removal adds an entry
        here and the advisory, panel and gate pick it up automatically.
        """
        assert SHIM_REMOVALS == ()

    def test_sample_entries_are_complete(self):
        for entry in SAMPLES:
            assert entry.shim
            assert parse_version(entry.removed_in) == (0, 16, 0)
            assert entry.source_repo.startswith("https://github.com/")
            assert entry.remote["url"] == entry.source_repo
            assert entry.remote["path"].endswith("/providers")
            # The published standalone repositories are public, so a preset
            # carries no forge token; a private mirror can add `token_env`.
            assert "token_env" not in entry.remote
            # No `ref` while the standalone repos are untagged: an omitted ref
            # is resolved to a commit and pinned in the cache stamp instead.
            assert "ref" not in entry.remote

    def test_every_removed_shim_is_currently_registered(self):
        """The pre-removal half of the gate.

        Deleting a shim without deleting its manifest entry fails here.
        """
        load_providers()
        unregistered = [e.shim for e in SAMPLES if get_shim(e.shim) is None]
        assert unregistered == []

    def test_every_legacy_name_still_resolves_to_its_shim(self):
        """The old spellings are the migration path, so they have to work."""
        load_providers()
        for entry in SAMPLES:
            for legacy in entry.legacy_names:
                assert canonical_shim_name(legacy) == entry.shim, legacy
                assert legacy in LEGACY_SHIM_ALIASES, legacy
                assert get_shim(legacy) is get_shim(entry.shim), legacy

    def test_every_remote_round_trips_through_the_source_resolver(self):
        """A preset is only useful if the gateway would accept it."""
        from llm_rosetta.gateway.shim_sources import resolve_source_config

        for entry in SAMPLES:
            _dirs, remotes, _cache = resolve_source_config(
                {"remotes": [dict(entry.remote)]}, None
            )
            assert remotes and remotes[0]["url"] == entry.remote["url"]

    def test_to_dict_keys(self):
        assert set(SAMPLE.to_dict()) == {
            "shim",
            "removed_in",
            "remote",
            "source_repo",
            "legacy_names",
        }


class TestDetectShimRemovals:
    """The pure detection shared by the warning, the panel and the gate."""

    def test_current_provider_on_a_removed_shim_is_reported(self):
        advisories = _detect({"mini": {"shim": "minimax--messages"}})
        assert len(advisories) == 1
        advisory = advisories[0]
        assert advisory.provider == "mini"
        assert advisory.slot == "shim"
        assert advisory.shim == "minimax--messages"
        assert advisory.removed_in == "0.16.0"
        assert advisory.remote["url"].endswith("llm-rosetta-shim-minimax")
        assert set(advisory.to_dict()) == {
            "provider",
            "slot",
            "shim",
            "removed_in",
            "remote",
            "source_repo",
        }

    def test_legacy_spelling_in_the_type_slot_is_reported(self):
        advisories = _detect({"mini": {"type": "minimax--anthropic"}})
        assert [a.shim for a in advisories] == ["minimax--messages"]
        assert advisories[0].slot == "type"

    def test_a_provider_keyed_by_the_shim_is_reported(self):
        advisories = _detect({"minimax--anthropic": {"api_key": "k"}})
        assert [a.shim for a in advisories] == ["minimax--messages"]
        assert advisories[0].slot == "name"

    def test_unrelated_and_base_format_providers_are_silent(self):
        assert (
            _detect(
                {
                    "o": {"shim": "openai--chat_completions"},
                    "b": {"type": "chat_completions"},
                    "n": {"api_key": "k"},
                    "x": {"shim": "xai--chat_completions"},
                }
            )
            == []
        )

    def test_a_disabled_provider_still_counts(self):
        """Its config breaks the moment it is switched back on."""
        advisories = _detect({"mini": {"shim": "minimax--messages", "enabled": False}})
        assert len(advisories) == 1

    def test_a_declared_external_source_silences_its_own_shim(self):
        providers = {
            "mini-chat": {"shim": "minimax--messages"},
            "ds": {"shim": "deepseek--chat_completions"},
        }
        advisories = _detect(providers, external_shims={"minimax--messages"})
        assert [a.shim for a in advisories] == ["deepseek--chat_completions"]

    def test_is_sorted_by_provider(self):
        advisories = _detect(
            {
                "zeta": {"shim": "minimax--messages"},
                "alpha": {"shim": "deepseek--chat_completions"},
                "mid": {"shim": "minimax--anthropic"},
            }
        )
        assert [a.provider for a in advisories] == ["alpha", "mid", "zeta"]

    def test_empty_and_malformed_inputs(self):
        assert _detect(None) == []
        assert _detect({}) == []
        assert _detect({"p": "not-a-mapping"}) == []

    def test_is_pure_and_needs_no_registry(self, empty_registry):
        """Resolution works with no registry at all.

        That is the property which lets this be a maintenance step (#1030): the
        answer comes from the config mapping alone, not from what happens to be
        loaded.
        """
        assert get_shim("minimax--messages") is None
        advisories = _detect({"mini": {"shim": "minimax--anthropic"}})
        assert [a.shim for a in advisories] == ["minimax--messages"]

    def test_the_legacy_spelling_reports_the_canonical_shim(self):
        advisories = _detect({"mini": {"shim": "minimax--anthropic"}})
        assert [a.shim for a in advisories] == ["minimax--messages"]


class TestSourceCatalog:
    """The standalone packages the Settings panel offers as presets."""

    def test_catalog_covers_the_three_standalone_repos(self):
        assert [entry.key for entry in SHIM_SOURCE_PACKAGES] == [
            "alcf",
            "argo",
            "asksage",
        ]

    def test_every_package_lists_its_shims_and_remote(self):
        for entry in SHIM_SOURCE_PACKAGES:
            assert entry.label
            assert entry.repo.startswith("https://github.com/")
            assert entry.shims
            assert entry.to_dict()["remote"] == {
                "url": entry.repo,
                "path": f"src/{entry.package}/providers",
            }
            assert "token_env" not in entry.to_dict()["remote"]

    def test_every_preset_round_trips_through_the_source_resolver(self):
        from llm_rosetta.gateway.shim_sources import resolve_source_config

        for entry in SHIM_SOURCE_PACKAGES:
            _dirs, remotes, _cache = resolve_source_config(
                {"remotes": [entry.remote]}, None
            )
            assert remotes and remotes[0]["url"] == entry.repo

    def test_get_source_package_resolves_a_preset_key(self):
        package = get_source_package("argo")
        assert package is not None
        assert package.repo.endswith("llm-rosetta-shim-argo")
        assert get_source_package("nope") is None

    def test_find_source_package_maps_a_shim_to_its_package(self):
        from llm_rosetta.shims import find_source_package

        package = find_source_package("argo--messages")
        assert package is not None
        assert package.key == "argo"
        assert find_source_package("openai--chat_completions") is None

    def test_the_catalog_does_not_imply_a_removal(self):
        """Standalone packages are presets; only the manifest schedules exits."""
        assert SHIM_REMOVALS == ()

    def test_to_dict_keys(self):
        assert set(SHIM_SOURCE_PACKAGES[0].to_dict()) == {
            "key",
            "label",
            "source_repo",
            "remote",
            "shims",
        }


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
        assert removals_due("0.14.0.dev17", removals=SAMPLES) == []
        assert removals_due("0.15.9", removals=SAMPLES) == []

    def test_everything_is_due_at_the_removal(self):
        assert len(removals_due("0.16.0", removals=SAMPLES)) == len(SAMPLES)
        # A pre-release of the removal release is the first build that could
        # ship the deletion, so it arms the gate too.
        assert len(removals_due("0.16.0.dev1", removals=SAMPLES)) == len(SAMPLES)

    def test_an_unparseable_version_never_arms(self):
        assert removals_due("", removals=SAMPLES) == []

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
        (examples / "config.jsonc").write_text('{"shim": "minimax--anthropic"}\n')
        assert self._hits(_scan((examples,))) == [
            ("config.jsonc", 1, "minimax--anthropic")
        ]

    def test_canonical_names_count_too(self, tmp_path):
        examples = self._examples(tmp_path)
        (examples / "config.jsonc").write_text('"shim": "minimax--messages"\n')
        assert len(_scan((examples,))) == 1

    def test_marker_lets_a_deliberate_mention_through(self, tmp_path):
        examples = self._examples(tmp_path)
        (examples / "migration.md").write_text(
            "| `minimax--anthropic` | `minimax--messages` |  # shim-removal-ok\n"
        )
        assert _scan((examples,)) == []

    def test_exempt_file_is_skipped_entirely(self, tmp_path):
        changelog = tmp_path / "changelog.md"
        changelog.write_text("`minimax--anthropic` was removed in 0.16.0\n")
        assert len(_scan((changelog,))) == 1
        assert _scan((changelog,), exempt_files=(changelog,)) == []

    def test_exemption_survives_relocating_the_scanned_tree(self, tmp_path):
        """The exemption is about the file, not about where the caller put it.

        The gate's default exemption names the changelogs by their path from the
        repository root; a run whose roots point somewhere else — a relocated
        docs tree, a test's ``tmp_path`` — must not silently lose it, or the
        changelog's historical mentions become failures.
        """
        relocated = tmp_path / "elsewhere" / "en" / "docs" / "changelog.md"
        relocated.parent.mkdir(parents=True)
        relocated.write_text("`minimax--anthropic` was removed in 0.16.0\n")

        assert len(_scan((tmp_path,))) == 1
        assert _scan((tmp_path,), exempt_files=("docs_en/docs/changelog.md",)) == []

    def test_neighbours_and_substrings_are_not_hits(self, tmp_path):
        examples = self._examples(tmp_path)
        (examples / "notes.txt").write_text(
            "minimax--messages-extra\n"
            "xminimax--messages\n"
            "minimax--messages_now\n"
            "gpt-4.1-nano\n"
            "xai\n"
        )
        assert _scan((examples,)) == []

    def test_missing_roots_and_skipped_file_types(self, tmp_path):
        examples = self._examples(tmp_path)
        (examples / "logo.svg").write_text("minimax--messages")
        assert _scan((tmp_path / "nope", examples)) == []

    def test_results_are_sorted_by_path_then_line(self, tmp_path):
        examples = self._examples(tmp_path)
        (examples / "b.txt").write_text("minimax--messages\n")
        (examples / "a.txt").write_text("x\nminimax--anthropic\n")
        assert [(os.path.basename(f.path), f.line) for f in _scan((examples,))] == [
            ("a.txt", 2),
            ("b.txt", 1),
        ]

    def test_nested_directories_are_walked(self, tmp_path):
        examples = self._examples(tmp_path)
        nested = examples / "gateway"
        nested.mkdir()
        (nested / "mini.jsonc").write_text('{"shim": "minimax--messages"}\n')
        assert len(_scan((examples,))) == 1
