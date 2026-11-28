"""Choosing among the releases an extension index lists for one extension.

Ported from the shim-manifest contract tests, whose release-selection rules
moved to ``manifest_sources/releases.py`` when the manifest format went away.
"""

from __future__ import annotations

from llm_rosetta.api import IndexEntry, choose_release
from llm_rosetta.contracts import IR_VERSION, SURFACE_VERSION
from llm_rosetta_gateway.shims.manifest_sources.releases import (
    release_from_dict,
    release_to_dict,
    requirement_problem,
)


def select_release(releases):
    """The gateway's reading of core's one install policy: stable first."""
    return choose_release(releases, prefer_stable=True)


def _entry(
    version: str,
    *,
    surface: int | None = SURFACE_VERSION,
    ir: int | None = IR_VERSION,
    shims: tuple[str, ...] = ("acme--chat_completions",),
) -> IndexEntry:
    return IndexEntry(
        id="acme",
        version=version,
        url=f"https://x.example/acme-{version}.whl",
        sha256="0" * 64,
        surface=surface,
        ir=ir,
        shims=shims,
        tags=("tier:free",),
        maturity="beta",
        name="Acme",
        logo="https://x.example/acme.svg",
        commit="c" * 40,
    )


class TestRequirementProblem:
    def test_this_cores_surface_and_ir_are_installable(self):
        assert requirement_problem(_entry("1.0.0")) is None

    def test_an_older_ir_is_installable(self):
        assert requirement_problem(_entry("1.0.0", ir=IR_VERSION - 1)) is None

    def test_any_other_surface_is_refused(self):
        problem = requirement_problem(_entry("1.0.0", surface=SURFACE_VERSION + 1))
        assert problem is not None and "surface" in problem
        assert requirement_problem(_entry("1.0.0", surface=SURFACE_VERSION - 1))

    def test_a_newer_ir_is_refused(self):
        problem = requirement_problem(_entry("1.0.0", ir=IR_VERSION + 1))
        assert problem is not None and "IR" in problem

    def test_a_level_the_index_does_not_state_is_not_checked(self):
        assert requirement_problem(_entry("1.0.0", surface=None, ir=None)) is None


class TestSelectRelease:
    def test_the_newest_installable_release_wins(self):
        chosen, skipped = select_release(
            [_entry("1.0.0"), _entry("1.2.0"), _entry("1.1.0")]
        )
        assert chosen is not None and chosen.version == "1.2.0"
        assert skipped == ()

    def test_older_incompatible_releases_are_not_reported(self):
        chosen, skipped = select_release(
            [_entry("0.9.0", surface=SURFACE_VERSION + 1), _entry("1.0.0")]
        )
        assert chosen is not None and chosen.version == "1.0.0"
        assert skipped == ()

    def test_nothing_installable_returns_none_with_every_reason(self):
        chosen, skipped = select_release(
            [_entry("1.0.0", surface=SURFACE_VERSION + 1), _entry("1.1.0", ir=99)]
        )
        assert chosen is None
        assert sorted(r.version for r, _ in skipped) == ["1.0.0", "1.1.0"]

    def test_a_stable_release_wins_over_a_newer_prerelease(self):
        chosen, _ = select_release([_entry("1.0.0"), _entry("1.1.0-rc.1")])
        assert chosen is not None and chosen.version == "1.0.0"

    def test_a_prerelease_is_chosen_only_when_no_stable_is_installable(self):
        chosen, _ = select_release(
            [_entry("1.0.0", surface=SURFACE_VERSION + 1), _entry("1.1.0-rc.1")]
        )
        assert chosen is not None and chosen.version == "1.1.0-rc.1"

    def test_an_incompatible_newer_release_is_reported_beside_a_stable_choice(self):
        chosen, skipped = select_release(
            [_entry("1.0.0"), _entry("2.0.0", surface=SURFACE_VERSION + 1)]
        )
        assert chosen is not None and chosen.version == "1.0.0"
        assert [release.version for release, _ in skipped] == ["2.0.0"]
        assert "surface" in skipped[0][1]

    def test_releases_are_ordered_by_version_not_position(self):
        chosen, _ = select_release([_entry("1.10.0"), _entry("1.9.0")])
        assert chosen is not None and chosen.version == "1.10.0"

    def test_prerelease_numbers_are_ordered_as_numbers(self):
        chosen, _ = select_release(
            [_entry("1.0.0-rc.10"), _entry("1.0.0-rc.2"), _entry("1.0.0-rc.9")]
        )
        assert chosen is not None and chosen.version == "1.0.0-rc.10"


class TestReleaseRecord:
    def test_the_record_round_trips(self):
        entry = _entry("1.2.3")
        assert release_from_dict("acme", release_to_dict(entry)) == entry

    def test_a_record_carries_its_name_and_logo(self):
        record = release_to_dict(_entry("1.2.3"))
        assert record["name"] == "Acme"
        assert record["logo"] == "https://x.example/acme.svg"

    def test_an_unreadable_record_is_none(self):
        assert release_from_dict("acme", {}) is None
        assert release_from_dict("acme", {"version": "1.0.0"}) is None
