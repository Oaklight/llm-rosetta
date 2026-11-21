"""The shim source manifest contract (#1660 S1): schema, validator, selection."""

from __future__ import annotations

import copy

import pytest

from llm_rosetta.contracts import IR_VERSION, SURFACE_VERSION
from llm_rosetta.contracts.shim_manifest import (
    ANNOTATION_KEYWORDS,
    SUPPORTED_KEYWORDS,
    ShimManifestInvalid,
    ShimRequirement,
    load_manifest_schema,
    parse_manifest,
    promoted_core_version,
    requirement_problem,
    select_release,
    validate_manifest,
    version_key,
)

SHA = "a" * 64
COMMIT = "b" * 40


def _release(version: str, *, surface: int = SURFACE_VERSION, ir: int = IR_VERSION):
    return {
        "version": version,
        "commit": COMMIT,
        "archive": f"{version}/acme.tar.gz",
        "sha256": SHA,
        "requires": {"surface": surface, "ir": ir},
        "shims": ["acme--chat_completions"],
        "tags": ["tier:free"],
        "maturity": "beta",
        "since": "2026-11-21",
        "conformance": {"date": "2026-11-20", "result": "pass"},
        "promoted_to": None,
    }


def _doc(*releases):
    return {
        "manifest_version": 1,
        "source": "contrib",
        "repo": "https://github.com/Oaklight/llm-rosetta-shim-contrib",
        "generated_at": "2026-11-21T00:00:00Z",
        "vendors": {"acme": {"versions": list(releases)}},
    }


def _keywords(node, out):
    if isinstance(node, dict):
        for key, value in node.items():
            out.add(key)
            if key in ("properties", "$defs"):
                for sub in value.values():
                    _keywords(sub, out)
            elif key in ("items", "additionalProperties", "propertyNames"):
                _keywords(value, out)
    return out


class TestSchema:
    def test_the_schema_uses_only_keywords_the_validator_reads(self):
        used = _keywords(load_manifest_schema(), set())
        assert used <= SUPPORTED_KEYWORDS | ANNOTATION_KEYWORDS, sorted(
            used - SUPPORTED_KEYWORDS - ANNOTATION_KEYWORDS
        )

    def test_a_valid_manifest_has_no_problems(self):
        assert validate_manifest(_doc(_release("0.1.0"))) == []

    def test_unknown_keys_are_allowed_for_forward_compatibility(self):
        doc = _doc(dict(_release("0.1.0"), future_field=1))
        doc["future_top"] = True
        assert validate_manifest(doc) == []

    @pytest.mark.parametrize(
        ("mutate", "fragment"),
        [
            (lambda d: d.update(manifest_version=2), "manifest_version"),
            (lambda d: d.pop("repo"), "'repo'"),
            (lambda d: d.update(source="tag"), "source"),
            (lambda d: d["vendors"].update({"Bad-Name": {"versions": []}}), "Bad-Name"),
            (lambda d: _r(d).update(sha256="x"), "sha256"),
            (lambda d: _r(d).update(archive="../escape.tar.gz"), "archive"),
            (lambda d: _r(d).update(archive="/abs/acme.tar.gz"), "archive"),
            (lambda d: _r(d).update(version="1.0"), "version"),
            (lambda d: _r(d).update(maturity="gold"), "maturity"),
            (lambda d: _r(d)["requires"].update(surface=True), "surface"),
            (lambda d: _r(d)["requires"].pop("ir"), "'ir'"),
            (lambda d: _r(d).update(promoted_to="0.15.0"), "promoted_to"),
            (lambda d: _r(d).update(shims=["a--b", "a--b"]), "duplicate"),
            (lambda d: _r(d).update(shims=[]), "at least"),
        ],
    )
    def test_each_violation_is_named(self, mutate, fragment):
        doc = _doc(_release("0.1.0"))
        mutate(doc)
        problems = validate_manifest(doc)
        assert problems and any(fragment in p for p in problems), problems

    def test_a_version_listed_twice_is_refused(self):
        problems = validate_manifest(_doc(_release("0.1.0"), _release("0.1.0")))
        assert problems == ["$.vendors.acme: version 0.1.0 is listed twice"]

    def test_parse_raises_with_every_problem(self):
        doc = _doc(_release("0.1.0"))
        doc.pop("source")
        doc["vendors"]["acme"]["versions"][0]["sha256"] = "nope"
        with pytest.raises(ShimManifestInvalid) as caught:
            parse_manifest(doc)
        assert len(caught.value.problems) == 2


def _r(doc):
    return doc["vendors"]["acme"]["versions"][0]


class TestParse:
    def test_releases_are_ordered_by_semver_not_position(self):
        manifest = parse_manifest(
            _doc(_release("0.2.0"), _release("0.10.0"), _release("0.10.0-rc1"))
        )
        assert [r.version for r in manifest.vendors["acme"]] == [
            "0.10.0",
            "0.10.0-rc1",
            "0.2.0",
        ]
        assert manifest.vendors["acme"][0].tags == ("tier:free",)

    def test_the_record_round_trips(self):
        raw = _release("0.1.0")
        record = parse_manifest(_doc(copy.deepcopy(raw))).vendors["acme"][0]
        assert record.to_dict() == raw


class TestCompatibility:
    def test_same_surface_and_equal_ir_is_installable(self):
        assert requirement_problem(ShimRequirement(1, 1), surface=1, ir=1) is None

    def test_an_older_ir_is_installable(self):
        assert requirement_problem(ShimRequirement(1, 1), surface=1, ir=3) is None

    @pytest.mark.parametrize("surface", [0, 2])
    def test_any_other_surface_is_refused(self, surface):
        problem = requirement_problem(ShimRequirement(surface, 1), surface=1, ir=1)
        assert problem is not None and "surface" in problem

    def test_a_newer_ir_is_refused(self):
        problem = requirement_problem(ShimRequirement(1, 2), surface=1, ir=1)
        assert problem is not None and "IR 2" in problem

    def test_the_defaults_are_this_cores_levels(self):
        assert requirement_problem(ShimRequirement(SURFACE_VERSION, IR_VERSION)) is None


class TestSelect:
    def _releases(self, *specs):
        return parse_manifest(_doc(*[_release(v, **kw) for v, kw in specs])).vendors[
            "acme"
        ]

    def test_the_newest_installable_release_wins(self):
        releases = self._releases(
            ("0.3.0", {"surface": 2}), ("0.2.0", {"ir": 9}), ("0.1.0", {})
        )
        chosen, skipped = select_release(releases, surface=1, ir=1)
        assert chosen is not None and chosen.version == "0.1.0"
        assert [(r.version, "surface" in why) for r, why in skipped] == [
            ("0.3.0", True),
            ("0.2.0", False),
        ]

    def test_older_incompatible_releases_are_not_reported(self):
        releases = self._releases(("0.2.0", {}), ("0.1.0", {"surface": 2}))
        chosen, skipped = select_release(releases, surface=1, ir=1)
        assert chosen is not None and chosen.version == "0.2.0"
        assert skipped == ()

    def test_nothing_installable_returns_none(self):
        releases = self._releases(("0.1.0", {"surface": 2}))
        chosen, skipped = select_release(releases, surface=1, ir=1)
        assert chosen is None and len(skipped) == 1

    def test_a_stable_release_wins_over_a_newer_prerelease(self):
        """The prerelease policy (#1662 review): stable first, pre-release last."""
        releases = self._releases(("0.5.0-rc.1", {}), ("0.4.0", {}))
        chosen, skipped = select_release(releases, surface=1, ir=1)
        assert chosen is not None and chosen.version == "0.4.0"
        assert skipped == ()

    def test_a_prerelease_is_chosen_only_when_no_stable_is_installable(self):
        releases = self._releases(
            ("0.6.0-rc.2", {}), ("0.6.0-rc.1", {}), ("0.5.0", {"surface": 2})
        )
        chosen, skipped = select_release(releases, surface=1, ir=1)
        assert chosen is not None and chosen.version == "0.6.0-rc.2"
        assert skipped == ()

    def test_an_incompatible_newer_release_is_reported_beside_a_stable_choice(self):
        releases = self._releases(
            ("0.7.0", {"surface": 2}), ("0.6.0-rc.1", {}), ("0.5.0", {})
        )
        chosen, skipped = select_release(releases, surface=1, ir=1)
        assert chosen is not None and chosen.version == "0.5.0"
        assert [r.version for r, _ in skipped] == ["0.7.0"]

    def test_build_metadata_is_refused(self):
        problems = validate_manifest(_doc(_release("0.1.0+build.5")))
        assert any("version" in p for p in problems)

    def test_semver_ordering(self):
        assert version_key("1.0.0-rc.2") < version_key("1.0.0-rc.10")
        assert version_key("1.0.0-alpha") < version_key("1.0.0")
        assert version_key("0.9.9") < version_key("0.10.0")
        with pytest.raises(ValueError):
            version_key("1.0")


def test_promoted_core_version():
    record = parse_manifest(
        _doc(dict(_release("0.1.0"), promoted_to="core@0.15"))
    ).vendors["acme"][0]
    assert promoted_core_version(record) == (0, 15, 0)
    plain = parse_manifest(_doc(_release("0.1.0"))).vendors["acme"][0]
    assert promoted_core_version(plain) is None
