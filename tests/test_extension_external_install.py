"""The ``install`` spec of an ``external:`` program in ``plugin.yaml`` (#1820).

``llm-rosetta ext install-cli`` installs a program an extension drives, at an
exact version, from an npm package or a release archive.  The manifest says
how; these tests pin what a manifest may say and what ``ext check`` refuses.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from llm_rosetta.contracts.extension import (
    ExtensionManifestInvalid,
    ExternalInstall,
    ReleaseArchive,
    parse_extension_manifest,
    validate_extension_manifest,
)

SHA = "a12514fb33e8835186082a00b60725f6fb15941ba061f0403347b86c4dbd8f46"
SHA_ARM = "455d20cc37507058bd9287d47d51aa7e6110351eb717fbf17e19619a7dc9d32e"

NPM: dict[str, Any] = {"npm": "@anthropic-ai/claude-code", "version": "2.1.299"}
RELEASE: dict[str, Any] = {
    "version": "1.5.15",
    "release": {
        "url": "https://github.com/acme/agy/releases/download/{version}/agy_{platform}.tar.gz",
        "sha256": {"linux_x64_musl": SHA, "linux_arm64_musl": SHA_ARM},
        "binary": "antigravity",
        "link": "agy",
    },
}


def _doc(program: dict) -> dict:
    return {
        "id": "demo",
        "name": "Demo",
        "version": "0.1.0",
        "maturity": "experimental",
        "requires": {"surface": 1, "ir": 1},
        "provides": {"shims": ["demo--acp"]},
        "external": [program],
    }


def _program(install: dict, *, name: str = "claude", check: str | None = None) -> dict:
    return {
        "name": name,
        "check": [check or name, "--version"],
        "install": copy.deepcopy(install),
    }


def _problems(install: dict, **kwargs) -> list[str]:
    return validate_extension_manifest(_doc(_program(install, **kwargs)))


class TestAValidSpec:
    def test_an_npm_install_is_read(self):
        manifest = parse_extension_manifest(_doc(_program(NPM)))
        assert manifest.external[0].install == ExternalInstall(
            version="2.1.299", npm="@anthropic-ai/claude-code"
        )

    def test_a_release_install_is_read(self):
        manifest = parse_extension_manifest(
            _doc(_program(RELEASE, name="agy", check="agy"))
        )
        install = manifest.external[0].install
        assert install is not None and install.release is not None
        assert install.version == "1.5.15"
        assert install.release == ReleaseArchive(
            url=RELEASE["release"]["url"],
            sha256={"linux_x64_musl": SHA, "linux_arm64_musl": SHA_ARM},
            binary="antigravity",
            link="agy",
        )

    def test_a_program_without_a_spec_has_none(self):
        program = {"name": "claude", "check": ["claude", "--version"]}
        assert parse_extension_manifest(_doc(program)).external[0].install is None

    def test_a_prerelease_version_is_exact(self):
        assert _problems({**NPM, "version": "2.1.299-beta.1"}) == []

    def test_an_npm_bin_names_the_entry(self):
        install = {"npm": "opencode-ai", "version": "1.19.0", "bin": "opencode"}
        assert _problems(install, name="opencode") == []


class TestAVersionIsExact:
    @pytest.mark.parametrize(
        "version",
        [
            "latest",
            "^2.1.299",
            "~2.1.299",
            ">=2.1.299",
            "2.1",
            "2",
            "2.1.x",
            "*",
            "2.1.299 - 2.2.0",
            "v2.1.299",
            "",
        ],
    )
    def test_a_range_or_a_tag_is_refused(self, version):
        problems = _problems({"npm": NPM["npm"], "version": version})
        assert any("install.version" in p for p in problems), problems

    def test_the_version_is_required(self):
        problems = _problems({"npm": NPM["npm"]})
        assert any("version" in p for p in problems), problems


class TestTheShapeOfASpec:
    def test_it_needs_a_source(self):
        problems = _problems({"version": "1.0.0"})
        assert any("exactly one of 'npm' and 'release'" in p for p in problems)

    def test_it_cannot_have_two_sources(self):
        problems = _problems({**RELEASE, "npm": "x"}, name="agy", check="agy")
        assert any("exactly one of 'npm' and 'release'" in p for p in problems)

    def test_a_bin_belongs_to_an_npm_install(self):
        problems = _problems({**RELEASE, "bin": "agy"}, name="agy", check="agy")
        assert any("install.bin" in p for p in problems), problems

    def test_an_unknown_key_is_refused(self):
        problems = _problems({**NPM, "tag": "latest"})
        assert any("unknown key 'tag'" in p for p in problems), problems

    @pytest.mark.parametrize("name", ["", "Bad Name", "../x", "@scope", "a b"])
    def test_an_npm_package_name_is_a_name(self, name):
        problems = _problems({"npm": name, "version": "1.0.0"})
        assert any("install.npm" in p for p in problems), problems


class TestTheReleaseSpec:
    def _release(self, **change) -> dict:
        spec = copy.deepcopy(RELEASE)
        spec["release"].update(change)
        return spec

    def test_the_url_must_be_https(self):
        problems = _problems(
            self._release(url="http://x.example/a.tgz"), name="agy", check="agy"
        )
        assert any("release.url" in p for p in problems), problems

    @pytest.mark.parametrize(
        "digest", ["", "abc", SHA.upper(), SHA[:-1], SHA + "0", "g" * 64]
    )
    def test_a_checksum_is_64_lowercase_hex(self, digest):
        problems = _problems(
            self._release(sha256={"linux_x64_musl": digest}), name="agy", check="agy"
        )
        assert any("release.sha256" in p for p in problems), problems

    def test_a_platform_is_a_known_one(self):
        problems = _problems(
            self._release(sha256={"plan9_386": SHA}), name="agy", check="agy"
        )
        assert any("release.sha256" in p for p in problems), problems

    def test_a_release_names_a_platform(self):
        problems = _problems(self._release(sha256={}), name="agy", check="agy")
        assert any("lists no platform" in p for p in problems), problems

    @pytest.mark.parametrize("binary", ["../evil", "a/../../b", "/abs", "", "a b"])
    def test_the_binary_stays_inside_the_archive(self, binary):
        problems = _problems(self._release(binary=binary), name="agy", check="agy")
        assert any("release.binary" in p for p in problems), problems

    def test_the_link_is_a_file_name(self):
        problems = _problems(self._release(link="a/b"), name="agy", check="agy")
        assert any("release.link" in p for p in problems), problems


class TestTheCheckRunsTheEntryLink:
    def test_check_runs_the_name_the_link_has(self):
        problems = _problems(RELEASE, name="agy", check="antigravity")
        assert any("entry link is named 'agy'" in p for p in problems), problems

    def test_without_a_link_check_runs_the_program_name(self):
        problems = _problems(NPM, name="claude", check="claude-code")
        assert any("entry link is named 'claude'" in p for p in problems), problems

    def test_a_manifest_that_breaks_a_rule_does_not_parse(self):
        with pytest.raises(ExtensionManifestInvalid):
            parse_extension_manifest(_doc(_program({"npm": "x", "version": "latest"})))
