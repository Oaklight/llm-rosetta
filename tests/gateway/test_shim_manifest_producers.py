"""The producers' real manifests against the core schema and the catalogue.

``tests/fixtures/shim_manifests`` holds the ``manifest.json`` each source
repository's ``scripts/build_manifest.py`` wrote on its ``next`` branch
(contrib#4, argonne#13; commit fields replaced by placeholders).  A producer
change that the gateway cannot read, or a tag rename that would make a
standalone vendor's install selector select nothing, fails here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_rosetta.contracts.shim_manifest import parse_manifest, validate_manifest
from llm_rosetta_gateway.shims.catalog import STANDALONE_VENDORS
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


@pytest.mark.parametrize("vendor", STANDALONE_VENDORS, ids=lambda v: v.key)
def test_each_standalone_install_selects_its_vendors(vendor):
    plan = parse_shims_section(
        {"sources": [vendor.source], "install": [vendor.install]}
    ).manifests
    manifest = parse_manifest(_manifest(vendor.source))

    chosen = _select_vendors(manifest, vendor.source, plan.selectors, {}, set())

    assert [name for name, _ in chosen] == list(vendor.vendors)


def test_the_catalogue_shims_match_the_published_records():
    """The catalogue lists what each install selector installs before any fetch."""
    for vendor in STANDALONE_VENDORS:
        manifest = parse_manifest(_manifest(vendor.source))
        published = {
            shim for name in vendor.vendors for shim in manifest.vendors[name][0].shims
        }
        assert published == set(vendor.shims), vendor.key
