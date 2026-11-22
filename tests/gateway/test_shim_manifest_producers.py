"""The producers' real manifests against the core schema and the legacy presets.

``tests/fixtures/shim_manifests`` holds the ``manifest.json`` each source
repository's ``scripts/build_manifest.py`` wrote on its ``next`` branch
(contrib#4, argonne#13; commit fields replaced by placeholders).  A producer
change that the gateway cannot read, or a tag rename that would make a legacy
preset select nothing, fails here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_rosetta.contracts.shim_manifest import parse_manifest, validate_manifest
from llm_rosetta_gateway.shims.shim_sources import parse_shims_section
from llm_rosetta_gateway.shims.manifest_sources.selection import _select_vendors

DATA = Path(__file__).resolve().parents[1] / "fixtures" / "shim_manifests"


def _manifest(source: str):
    return json.loads((DATA / f"{source}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("source", ["contrib", "argonne"])
def test_the_published_manifest_follows_the_schema(source):
    document = _manifest(source)
    assert validate_manifest(document) == []
    assert parse_manifest(document).source == source


@pytest.mark.parametrize(
    ("preset", "source", "vendors"),
    [
        ("free", "contrib", ["kilo_free", "llm7", "opencode_zen_free", "ovhcloud"]),
        ("opendesign", "contrib", ["opendesign"]),
        ("alcf", "argonne", ["alcf"]),
        ("argo", "argonne", ["argo"]),
        ("asksage", "argonne", ["asksage"]),
    ],
)
def test_each_legacy_preset_selects_its_vendors(preset, source, vendors):
    plan = parse_shims_section({"presets": [preset]}).manifests
    manifest = parse_manifest(_manifest(source))

    chosen = _select_vendors(manifest, source, plan.selectors, {}, set())

    assert [vendor for vendor, _ in chosen] == vendors


def test_the_catalogue_shims_match_the_published_records():
    """The catalogue lists what each preset installs before anything is fetched."""
    from llm_rosetta_gateway.shims import SHIM_SOURCE_PACKAGES

    for package in SHIM_SOURCE_PACKAGES:
        manifest = parse_manifest(_manifest(str(package.source)))
        published = {
            shim
            for vendor in package.vendors
            for shim in manifest.vendors[vendor][0].shims
        }
        assert published == set(package.shims), package.key
