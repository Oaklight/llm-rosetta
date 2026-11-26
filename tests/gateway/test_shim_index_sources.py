"""Installing extensions from a published ``index.json`` (#1704).

An extension repository publishes ``index.json`` and its wheels as release
assets.  Every test serves a fake index and fake wheels through an injected
``http_get``; nothing reaches the network.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from _ext_wheels import build_wheel
from llm_rosetta.api import get_shim
from llm_rosetta.testing import restore_registry, snapshot_registry
from llm_rosetta.contracts import IR_VERSION, SURFACE_VERSION
from llm_rosetta_gateway.shims.shim_sources import apply_shim_sources

RELEASE = "https://repo.example/releases/download/r1/"
INDEX = RELEASE + "index.json"
COMMIT = "a" * 40


@pytest.fixture(autouse=True)
def _restore_registry():
    snapshot = snapshot_registry()
    yield
    restore_registry(snapshot)


class FakeRelease:
    """Serves ``index.json`` and the wheels it names by absolute URL."""

    def __init__(self, generated_at: str = "2026-11-26T08:00:00Z") -> None:
        self.generated_at = generated_at
        self.entries: dict[str, list[dict[str, Any]]] = {}
        self.files: dict[str, bytes] = {}
        self.calls: list[str] = []

    def publish(
        self, ext: str, version: str, *, sha256: str | None = None, **fields: Any
    ) -> None:
        shim = f"{ext}--chat_completions"
        data = build_wheel(ext, version, [shim], marker=version)
        url = f"{RELEASE}llm_rosetta_ext_{ext}-{version}-py3-none-any.whl"
        entry = {
            "version": version,
            "url": url,
            "sha256": sha256 or hashlib.sha256(data).hexdigest(),
            "surface": SURFACE_VERSION,
            "ir": IR_VERSION,
            "requires_python": ">=3.10",
            "name": ext.title(),
            "maturity": "beta",
            "tags": ["tier:free"],
            "shims": [shim],
            "logo": f"https://logos.example/{ext}.svg",
            "commit": COMMIT,
            **fields,
        }
        self.entries.setdefault(ext, []).insert(0, entry)
        self.files[url] = data

    def index(self) -> dict[str, Any]:
        return {
            "index_version": 1,
            "source": "contrib",
            "generated_at": self.generated_at,
            "extensions": self.entries,
        }

    def __call__(self, url: str, *, headers=None, timeout=None, proxy=None) -> bytes:
        self.calls.append(url)
        if url == INDEX:
            return json.dumps(self.index()).encode()
        if url in self.files:
            return self.files[url]
        raise ConnectionError(f"no route: {url}")


def _apply(tmp_path: Path, http: FakeRelease, *install: str, **kw: Any):
    section = {"sources": [INDEX], "install": list(install)}
    return apply_shim_sources(
        {"shims": section}, str(tmp_path / "config.jsonc"), http_get=http, **kw
    )


def _status(report, vendor: str):
    return next(s for s in report.sources if s.vendor == vendor)


def test_the_newest_compatible_wheel_installs_by_its_url(tmp_path):
    http = FakeRelease()
    http.publish("acme", "0.1.0")
    http.publish("acme", "0.2.0", surface=SURFACE_VERSION + 1)

    report = _apply(tmp_path, http, "acme")

    status = _status(report, "acme")
    assert status.ok and status.release["version"] == "0.1.0"
    assert status.archive_url == f"{RELEASE}llm_rosetta_ext_acme-0.1.0-py3-none-any.whl"
    assert get_shim("acme--chat_completions") is not None
    assert not any("0.2.0" in call for call in http.calls)


def test_the_listing_fields_reach_the_release_record(tmp_path):
    http = FakeRelease()
    http.publish("acme", "0.1.0")

    release = _status(_apply(tmp_path, http, "acme"), "acme").release

    assert release["name"] == "Acme"
    assert release["logo"] == "https://logos.example/acme.svg"
    assert release["tags"] == ["tier:free"]
    assert release["maturity"] == "beta"
    assert release["commit"] == COMMIT


def test_a_tag_selector_reads_the_index_tags(tmp_path):
    http = FakeRelease()
    http.publish("acme", "0.1.0")
    http.publish("other", "0.1.0", tags=["tier:paid"])

    report = _apply(tmp_path, http, "tag:tier:free")

    assert {s.vendor for s in report.sources if s.ok} == {"acme"}


def test_a_wheel_with_the_wrong_digest_is_not_installed(tmp_path):
    http = FakeRelease()
    http.publish("acme", "0.1.0", sha256="0" * 64)

    status = _status(_apply(tmp_path, http, "acme"), "acme")

    assert not status.ok
    assert "sha256 mismatch" in (status.error or "")
    assert get_shim("acme--chat_completions") is None


def test_an_older_index_does_not_replace_a_newer_cached_one(tmp_path):
    new = FakeRelease("2026-11-26T08:00:00Z")
    new.publish("acme", "0.2.0")
    _apply(tmp_path, new, "acme")

    old = FakeRelease("2026-11-25T08:00:00Z")
    old.publish("acme", "0.1.0")
    status = _status(_apply(tmp_path, old, "acme", force_refresh=True), "acme")

    assert status.stale
    assert status.release["version"] == "0.2.0"


def test_a_malformed_index_fails_its_source(tmp_path):
    http = FakeRelease()
    http.publish("acme", "0.1.0", sha256="not-hex")

    report = _apply(tmp_path, http, "acme")

    (failed,) = report.failures()
    assert "sha256" in (failed.error or "")
