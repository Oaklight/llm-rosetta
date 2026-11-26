"""The extension manifest contract (#1703 E1): schema, validator, compatibility, transport scope."""

from __future__ import annotations

import copy

import pytest

from llm_rosetta.contracts import IR_VERSION, SURFACE_VERSION
from llm_rosetta.contracts.extension import (
    MATURITY_LEVELS,
    ExtensionManifest,
    ExtensionManifestInvalid,
    UnresolvedTransport,
    compatibility_problems,
    load_extension_schema,
    parse_extension_manifest,
    resolve_transport,
    validate_extension_manifest,
)
from llm_rosetta.contracts._schema import (
    ANNOTATION_KEYWORDS,
    SUPPORTED_KEYWORDS,
)


def _doc(**override):
    doc = {
        "id": "claude",
        "name": "Anthropic Claude (official CLI)",
        "version": "0.1.0",
        "maturity": "experimental",
        "requires": {"surface": SURFACE_VERSION, "ir": IR_VERSION},
        "provides": {
            "shims": ["claude--acp", "claude--messages"],
            "transports": ["claude_cli"],
        },
    }
    doc.update(override)
    return doc


def _manifest(**override) -> ExtensionManifest:
    return parse_extension_manifest(_doc(**override))


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
        used = _keywords(load_extension_schema(), set())
        assert used <= SUPPORTED_KEYWORDS | ANNOTATION_KEYWORDS, sorted(
            used - SUPPORTED_KEYWORDS - ANNOTATION_KEYWORDS
        )

    def test_the_maturity_enum_is_the_ladder(self):
        enum = load_extension_schema()["properties"]["maturity"]["enum"]
        assert tuple(enum) == MATURITY_LEVELS

    def test_a_valid_manifest_has_no_problems(self):
        assert validate_extension_manifest(_doc()) == []

    @pytest.mark.parametrize(
        "key", ["id", "name", "version", "maturity", "requires", "provides"]
    )
    def test_a_missing_required_key_is_reported(self, key):
        doc = _doc()
        del doc[key]
        problems = validate_extension_manifest(doc)
        assert any(key in p for p in problems), problems

    def test_an_unknown_top_level_key_is_refused(self):
        assert validate_extension_manifest(_doc(vendor="x"))

    @pytest.mark.parametrize("bad", ["Claude", "1claude", "kilo-free", "", "a b"])
    def test_a_bad_id_is_refused(self, bad):
        assert validate_extension_manifest(_doc(id=bad, provides={"shims": [bad]}))

    @pytest.mark.parametrize("bad", ["1.0", "v1.0.0", "1.0.0+build", "01.0.0"])
    def test_a_bad_version_is_refused(self, bad):
        assert validate_extension_manifest(_doc(version=bad))

    def test_an_unknown_maturity_is_refused(self):
        assert validate_extension_manifest(_doc(maturity="core"))

    def test_a_platform_outside_the_vocabulary_is_refused(self):
        assert validate_extension_manifest(_doc(platforms=["linux", "beos"]))

    def test_summary_needs_english(self):
        assert validate_extension_manifest(_doc(summary={"zh": "中文"}))

    def test_an_external_program_needs_a_check_command(self):
        assert validate_extension_manifest(_doc(external=[{"name": "claude"}]))


class TestRules:
    def test_provides_cannot_be_empty(self):
        problems = validate_extension_manifest(_doc(provides={}))
        assert any("no shim, no transport and no command" in p for p in problems), (
            problems
        )

    def test_a_transport_only_extension_is_valid(self):
        doc = _doc(id="acp_pty", provides={"transports": ["acp_pty"]})
        assert validate_extension_manifest(doc) == []

    @pytest.mark.parametrize("name", ["claude", "claude--acp", "claude_work--messages"])
    def test_shim_names_that_belong_to_the_id_pass(self, name):
        assert validate_extension_manifest(_doc(provides={"shims": [name]})) == []

    @pytest.mark.parametrize("name", ["openai--chat", "claudex--acp", "claude-acp"])
    def test_a_shim_that_belongs_to_another_vendor_is_refused(self, name):
        problems = validate_extension_manifest(_doc(provides={"shims": [name]}))
        assert any("provides.shims" in p for p in problems), problems

    def test_an_extension_cannot_require_itself(self):
        doc = _doc()
        doc["requires"]["extensions"] = ["claude"]
        problems = validate_extension_manifest(doc)
        assert any("cannot require itself" in p for p in problems), problems

    def test_duplicate_shims_are_refused(self):
        assert validate_extension_manifest(
            _doc(provides={"shims": ["claude--acp", "claude--acp"]})
        )

    def test_parse_raises_with_every_problem(self):
        with pytest.raises(ExtensionManifestInvalid) as caught:
            parse_extension_manifest(_doc(version="x", maturity="core"))
        assert len(caught.value.problems) >= 2

    def test_parse_builds_the_manifest(self):
        doc = _doc(
            tags=["tier:subscription"],
            platforms=["linux", "darwin"],
            external=[
                {
                    "name": "claude",
                    "check": ["claude", "--version"],
                    "minimum": "2.1.299",
                }
            ],
            summary={"en": "x", "zh": "y"},
        )
        doc["requires"]["extensions"] = ["acp_pty"]
        manifest = parse_extension_manifest(copy.deepcopy(doc))
        assert manifest.id == "claude"
        assert manifest.shims == ("claude--acp", "claude--messages")
        assert manifest.transports == ("claude_cli",)
        assert manifest.requires.extensions == ("acp_pty",)
        assert manifest.external[0].check == ("claude", "--version")
        assert manifest.external[0].minimum == "2.1.299"
        assert manifest.platforms == ("linux", "darwin")
        assert manifest.summary == {"en": "x", "zh": "y"}


class TestCompatibility:
    def test_a_matching_manifest_runs(self):
        assert compatibility_problems(_manifest(), platform="linux") == []

    def test_a_different_surface_is_refused(self):
        doc = _doc()
        doc["requires"]["surface"] = SURFACE_VERSION + 1
        problems = compatibility_problems(parse_extension_manifest(doc))
        assert any("surface" in p for p in problems), problems

    def test_a_newer_ir_is_refused(self):
        doc = _doc()
        doc["requires"]["ir"] = IR_VERSION + 1
        problems = compatibility_problems(parse_extension_manifest(doc))
        assert any("IR" in p for p in problems), problems

    def test_another_platform_is_refused(self):
        manifest = _manifest(platforms=["linux", "darwin"])
        problems = compatibility_problems(manifest, platform="windows")
        assert any("windows" in p for p in problems), problems

    def test_no_platforms_means_any(self):
        assert compatibility_problems(_manifest(), platform="windows") == []

    def test_every_reason_is_reported(self):
        doc = _doc(platforms=["linux"])
        doc["requires"]["surface"] = SURFACE_VERSION + 1
        problems = compatibility_problems(
            parse_extension_manifest(doc), platform="windows"
        )
        assert len(problems) == 2


class TestTransportScope:
    def _installed(self, *manifests):
        return {m.id: m for m in manifests}

    def test_own_transport_wins_over_builtin(self):
        own = _manifest(provides={"transports": ["claude", "acp"]})
        owner = resolve_transport("acp", own, self._installed(own), {"acp", "http"})
        assert (owner.kind, owner.extension) == ("own", "claude")

    def test_a_builtin_is_reachable(self):
        own = _manifest()
        owner = resolve_transport("acp", own, self._installed(own), {"acp", "http"})
        assert (owner.kind, owner.extension) == ("builtin", None)

    def test_a_required_extensions_transport_is_reachable(self):
        shared = parse_extension_manifest(
            _doc(id="acp_pty", provides={"transports": ["acp_pty"]})
        )
        doc = _doc(provides={"shims": ["claude--acp"]})
        doc["requires"]["extensions"] = ["acp_pty"]
        user = parse_extension_manifest(doc)
        owner = resolve_transport(
            "acp_pty", user, self._installed(user, shared), {"http"}
        )
        assert (owner.kind, owner.extension) == ("dependency", "acp_pty")

    def test_an_unrequired_extensions_transport_is_refused_and_named(self):
        other = parse_extension_manifest(
            _doc(id="acp_pty", provides={"transports": ["acp_pty"]})
        )
        user = _manifest(provides={"shims": ["claude--acp"]})
        with pytest.raises(UnresolvedTransport) as caught:
            resolve_transport("acp_pty", user, self._installed(user, other), {"http"})
        assert "acp_pty" in str(caught.value)
        assert "requires.extensions" in str(caught.value)

    def test_a_required_but_not_installed_extension_is_refused(self):
        doc = _doc(provides={"shims": ["claude--acp"]})
        doc["requires"]["extensions"] = ["acp_pty"]
        user = parse_extension_manifest(doc)
        with pytest.raises(UnresolvedTransport):
            resolve_transport("acp_pty", user, self._installed(user), {"http"})

    def test_two_extensions_with_the_same_transport_id_each_resolve_their_own(self):
        a = parse_extension_manifest(_doc(id="a", provides={"transports": ["lane"]}))
        b = parse_extension_manifest(_doc(id="b", provides={"transports": ["lane"]}))
        installed = self._installed(a, b)
        assert resolve_transport("lane", a, installed, {"http"}).extension == "a"
        assert resolve_transport("lane", b, installed, {"http"}).extension == "b"

    def test_an_unknown_transport_is_refused(self):
        own = _manifest()
        with pytest.raises(UnresolvedTransport):
            resolve_transport("nope", own, self._installed(own), {"http"})
