"""Installing shim vendors from a versioned manifest (#1660 S2/S3).

Every test serves a fake manifest and fake archives through an injected
``http_get``; nothing reaches the network.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import tarfile
from pathlib import Path
from typing import Any

import pytest

from llm_rosetta.api import (
    get_shim,
    list_shims,
    restore_registry,
    snapshot_registry,
)
from llm_rosetta.contracts import IR_VERSION, SURFACE_VERSION
from llm_rosetta_gateway.shim_sources import (
    apply_shim_sources,
    describe_shim_sources,
    parse_shims_section,
)
from llm_rosetta_gateway.shims import get_source_package
from llm_rosetta_gateway.shims.catalog import get_shim_source
from llm_rosetta_gateway.shims.manifest_sources import parse_selector
from llm_rosetta_gateway.shims.sources import check_remote_updates

DIST = "https://shims.example/dist/"
MANIFEST = DIST + "manifest.json"
COMMIT = "c" * 40


@pytest.fixture(autouse=True)
def _restore_registry():
    snapshot = snapshot_registry()
    yield
    restore_registry(snapshot)


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def logs():
    handler = _Capture()
    names = ["llm_rosetta_gateway.shims.sources", "llm-rosetta-gateway"]
    levels = {}
    for name in names:
        logger = logging.getLogger(name)
        levels[name] = logger.level
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
    try:
        yield handler.messages
    finally:
        for name in names:
            logger = logging.getLogger(name)
            logger.removeHandler(handler)
            logger.setLevel(levels[name])


def _archive(vendor: str, shims: list[str], marker: str = "v") -> bytes:
    """A vendor archive: ``providers/<vendor>/<lane>/provider.yaml`` per shim."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for shim in shims:
            lane = shim.split("--", 1)[0]
            body = (
                f"name: {shim}\n"
                "format: chat_completions\nmodel_type: llm\n"
                "connection:\n"
                f"  base_url: https://{lane}.{marker}.example/v1\n"
                "  api_key_env: ACME_API_KEY\n"
            ).encode()
            info = tarfile.TarInfo(f"providers/{vendor}/{lane}/provider.yaml")
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
    return buf.getvalue()


def _release(
    vendor: str,
    version: str,
    data: bytes,
    shims: list[str],
    *,
    surface: int = SURFACE_VERSION,
    ir: int = IR_VERSION,
    tags: list[str] | None = None,
    promoted_to: str | None = None,
    sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "version": version,
        "commit": COMMIT,
        "archive": f"{version}/{vendor}.tar.gz",
        "sha256": sha256 or hashlib.sha256(data).hexdigest(),
        "requires": {"surface": surface, "ir": ir},
        "shims": shims,
        "tags": tags or [],
        "maturity": "beta",
        "promoted_to": promoted_to,
    }


class FakeHttp:
    """Serves a manifest and archives; anything else raises a connection error."""

    def __init__(self, source: str = "contrib", dist: str = DIST) -> None:
        self.source = source
        self.dist = dist
        self.vendors: dict[str, list[dict[str, Any]]] = {}
        self.files: dict[str, bytes] = {}
        self.calls: list[str] = []
        self.down = False

    def publish(self, vendor: str, version: str, shims: list[str], **kw: Any) -> None:
        data = kw.pop("data", None) or _archive(vendor, shims, marker=version)
        record = _release(vendor, version, data, shims, **kw)
        self.vendors.setdefault(vendor, []).insert(0, record)
        self.files[self.dist + record["archive"]] = data

    def manifest(self) -> dict[str, Any]:
        return {
            "manifest_version": 1,
            "source": self.source,
            "repo": f"https://github.com/Oaklight/llm-rosetta-shim-{self.source}",
            "generated_at": "2026-11-21T00:00:00Z",
            "vendors": {v: {"versions": r} for v, r in self.vendors.items()},
        }

    def __call__(self, url: str, *, headers=None, timeout=None, proxy=None) -> bytes:
        self.calls.append(url)
        if self.down:
            raise ConnectionError(f"down: {url}")
        if url == self.dist + "manifest.json":
            return json.dumps(self.manifest()).encode()
        if url in self.files:
            return self.files[url]
        raise ConnectionError(f"no route: {url}")


def _apply(tmp_path: Path, section: dict[str, Any], http: FakeHttp, **kw: Any):
    raw = {"shims": section}
    return apply_shim_sources(raw, str(tmp_path / "config.jsonc"), http_get=http, **kw)


def _url_section(*install: str) -> dict[str, Any]:
    return {"sources": [MANIFEST], "install": list(install)}


def _vendor_status(report, vendor: str):
    return next(s for s in report.sources if s.vendor == vendor)


def _core_name() -> str:
    """A shim name core registers (built-ins load with the package)."""
    return next(s.name for s in list_shims() if "--" in s.name)


class TestSelection:
    def test_the_newest_compatible_version_is_installed(self, tmp_path, logs):
        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"])
        http.publish("acme", "0.2.0", ["acme--chat_completions"], ir=IR_VERSION + 1)
        http.publish(
            "acme", "0.3.0", ["acme--chat_completions"], surface=SURFACE_VERSION + 1
        )

        report = _apply(tmp_path, _url_section("acme"), http)

        status = _vendor_status(report, "acme")
        assert status.ok and status.release["version"] == "0.1.0"
        assert status.shim_names == ("acme--chat_completions",)
        assert get_shim("acme--chat_completions") is not None
        assert DIST + "0.1.0/acme.tar.gz" in http.calls
        assert not any("0.2.0/" in c or "0.3.0/" in c for c in http.calls)

    def test_a_surface_mismatch_is_skipped_with_a_warning(self, tmp_path, logs):
        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"])
        http.publish(
            "acme", "0.2.0", ["acme--chat_completions"], surface=SURFACE_VERSION + 1
        )

        _apply(tmp_path, _url_section("acme"), http)

        assert any(
            "acme 0.2.0 is not installable" in m
            and f"requires surface {SURFACE_VERSION + 1}" in m
            for m in logs
        ), logs

    def test_an_ir_too_new_is_skipped_with_a_warning(self, tmp_path, logs):
        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"])
        http.publish("acme", "0.2.0", ["acme--chat_completions"], ir=IR_VERSION + 1)

        report = _apply(tmp_path, _url_section("acme"), http)

        assert _vendor_status(report, "acme").release["version"] == "0.1.0"
        assert any(f"requires IR {IR_VERSION + 1}" in m for m in logs), logs

    def test_no_installable_version_fails_the_vendor(self, tmp_path):
        http = FakeHttp()
        http.publish(
            "acme", "0.1.0", ["acme--chat_completions"], surface=SURFACE_VERSION + 1
        )

        report = _apply(tmp_path, _url_section("acme"), http)

        status = _vendor_status(report, "acme")
        assert not status.ok and "no version of acme is installable" in status.error
        assert get_shim("acme--chat_completions") is None

    def test_tags_and_source_wildcards_select_vendors(self, tmp_path):
        http = FakeHttp()
        http.publish(
            "free_a", "0.1.0", ["free_a--chat_completions"], tags=["tier:free"]
        )
        http.publish("paid_b", "0.1.0", ["paid_b--chat_completions"])

        tagged = _apply(tmp_path, _url_section("tag:tier:free"), http)
        assert [s.vendor for s in tagged.sources] == ["free_a"]

        everything = _apply(tmp_path, _url_section("contrib:*"), http)
        assert sorted(s.vendor for s in everything.sources) == ["free_a", "paid_b"]

    def test_tags_come_from_the_newest_installable_record(self, tmp_path):
        http = FakeHttp()
        http.publish("acme", "0.2.0", ["acme--chat_completions"], tags=["tier:free"])
        http.publish(
            "acme", "0.3.0", ["acme--chat_completions"], surface=SURFACE_VERSION + 1
        )

        report = _apply(tmp_path, _url_section("tag:tier:free"), http)

        assert _vendor_status(report, "acme").release["version"] == "0.2.0"

    def test_a_newer_prerelease_does_not_replace_a_stable_choice(self, tmp_path):
        http = FakeHttp()
        http.publish("acme", "0.4.0", ["acme--chat_completions"])
        http.publish("acme", "0.5.0-rc.1", ["acme--chat_completions"])

        report = _apply(tmp_path, _url_section("acme"), http)

        assert _vendor_status(report, "acme").release["version"] == "0.4.0"

    def test_an_install_entry_that_matches_nothing_warns(self, tmp_path, logs):
        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"])

        _apply(tmp_path, _url_section("ghost"), http)

        assert any("'ghost'" in m and "matched no vendor" in m for m in logs), logs


class TestIntegrity:
    def test_a_sha_mismatch_is_rejected(self, tmp_path, logs):
        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"], sha256="0" * 64)

        report = _apply(tmp_path, _url_section("acme"), http)

        status = _vendor_status(report, "acme")
        assert not status.ok
        assert status.checksum_ok is False
        assert "sha256 mismatch" in status.error
        assert get_shim("acme--chat_completions") is None
        assert not list((tmp_path / "shims-cache").rglob("provider.yaml"))

    def test_a_sha_mismatch_on_upgrade_keeps_the_last_good_install(self, tmp_path):
        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"])
        _apply(tmp_path, _url_section("acme"), http)
        http.publish("acme", "0.2.0", ["acme--chat_completions"], sha256="0" * 64)

        report = _apply(tmp_path, _url_section("acme"), http, force_refresh=True)

        status = _vendor_status(report, "acme")
        assert status.ok and status.stale and status.checksum_ok is False
        assert status.release["version"] == "0.1.0"
        shim = get_shim("acme--chat_completions")
        assert shim is not None and "0.1.0" in str(shim.connection.base_url)


class TestUpgrades:
    def test_an_unchanged_manifest_downloads_nothing(self, tmp_path):
        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"])
        _apply(tmp_path, _url_section("acme"), http)
        http.calls.clear()

        report = _apply(tmp_path, _url_section("acme"), http, force_refresh=True)

        assert http.calls == [MANIFEST]
        assert _vendor_status(report, "acme").from_cache

    def test_a_fresh_manifest_is_not_refetched_within_the_ttl(self, tmp_path):
        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"])
        _apply(tmp_path, _url_section("acme"), http)
        http.calls.clear()

        _apply(tmp_path, _url_section("acme"), http)

        assert http.calls == []

    def test_a_newer_installable_version_is_installed(self, tmp_path, logs):
        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"])
        _apply(tmp_path, _url_section("acme"), http)
        http.publish("acme", "0.2.0", ["acme--chat_completions"])

        report = _apply(tmp_path, _url_section("acme"), http, force_refresh=True)

        assert _vendor_status(report, "acme").release["version"] == "0.2.0"
        assert any("upgraded 0.1.0 -> 0.2.0" in m for m in logs)

    def test_a_newer_incompatible_version_does_not_replace_the_install(self, tmp_path):
        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"])
        _apply(tmp_path, _url_section("acme"), http)
        http.publish(
            "acme", "0.2.0", ["acme--chat_completions"], surface=SURFACE_VERSION + 1
        )
        http.calls.clear()

        report = _apply(tmp_path, _url_section("acme"), http, force_refresh=True)

        assert _vendor_status(report, "acme").release["version"] == "0.1.0"
        assert http.calls == [MANIFEST]

    def test_an_install_the_manifest_no_longer_lists_is_kept_with_a_warning(
        self, tmp_path, logs
    ):
        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"])
        http.publish("acme", "0.2.0", ["acme--chat_completions"])
        _apply(tmp_path, _url_section("acme"), http)
        http.vendors["acme"] = [
            r for r in http.vendors["acme"] if r["version"] == "0.1.0"
        ]
        http.calls.clear()

        report = _apply(tmp_path, _url_section("acme"), http, force_refresh=True)

        assert _vendor_status(report, "acme").release["version"] == "0.2.0"
        assert http.calls == [MANIFEST]
        assert any(
            "manifest no longer lists acme 0.2.0; keeping the install" in m
            for m in logs
        ), logs

    def test_an_unreachable_source_keeps_the_last_good_install(self, tmp_path):
        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"])
        _apply(tmp_path, _url_section("acme"), http)
        http.down = True

        report = _apply(tmp_path, _url_section("acme"), http, force_refresh=True)

        status = _vendor_status(report, "acme")
        assert status.ok and status.stale and status.from_cache
        assert get_shim("acme--chat_completions") is not None

    def test_an_unreachable_source_with_nothing_cached_fails(self, tmp_path):
        http = FakeHttp()
        http.down = True

        report = _apply(tmp_path, _url_section("acme"), http)

        (status,) = report.sources
        assert not status.ok and status.unreachable and status.location == MANIFEST

    def test_the_update_check_compares_versions(self, tmp_path):
        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"])
        report = _apply(tmp_path, _url_section("acme"), http)
        http.publish("acme", "0.2.0", ["acme--chat_completions"])
        http.publish(
            "acme", "0.3.0", ["acme--chat_completions"], surface=SURFACE_VERSION + 1
        )

        (entry,) = check_remote_updates(report, http_get=http)

        assert entry["current"] == "0.1.0"
        assert entry["head"] == "0.2.0"
        assert entry["update"] is True


class TestPrecedence:
    def test_core_wins_a_shim_name_clash(self, tmp_path, logs):
        core_name = next(s.name for s in list_shims() if "--" in s.name)
        before = get_shim(core_name)
        vendor = core_name.split("--", 1)[0].split("_")[0]
        http = FakeHttp()
        http.publish(vendor, "0.1.0", [core_name])

        report = _apply(tmp_path, _url_section(vendor), http)

        status = _vendor_status(report, vendor)
        assert status.skipped and core_name in status.skipped
        after = get_shim(core_name)
        assert before is not None and after is not None
        assert after.connection.base_url == before.connection.base_url
        assert any(
            "contrib" in m and repr(vendor) in m and "core already provides" in m
            for m in logs
        ), logs
        # The estimate saves the download too.
        assert not any(c.endswith(".tar.gz") for c in http.calls)

    def test_a_promoted_vendor_is_not_installed_from_the_source(self, tmp_path, logs):
        core_name = _core_name()
        http = FakeHttp()
        http.publish("acme", "0.1.0", [core_name], promoted_to="core@0.1.0")

        report = _apply(tmp_path, _url_section("acme"), http)

        status = _vendor_status(report, "acme")
        assert status.skipped == "promoted to core@0.1.0"
        assert describe_shim_sources(report) == [
            "contrib:acme: skipped (promoted to core@0.1.0)"
        ]
        assert any("acme 0.1.0 says promoted_to core@0.1.0" in m for m in logs)
        assert not any(c.endswith(".tar.gz") for c in http.calls)

    def test_a_promotion_core_does_not_bear_out_installs_with_a_warning(
        self, tmp_path, logs
    ):
        http = FakeHttp()
        http.publish(
            "acme", "0.1.0", ["acme--chat_completions"], promoted_to="core@0.1.0"
        )

        report = _apply(tmp_path, _url_section("acme"), http)

        status = _vendor_status(report, "acme")
        assert status.skipped is None
        assert status.shim_names == ("acme--chat_completions",)
        assert any("does not provide acme--chat_completions" in m for m in logs), logs

    def test_a_later_release_without_the_claim_un_promotes(self, tmp_path):
        core_name = _core_name()
        http = FakeHttp()
        http.publish("acme", "0.1.0", [core_name], promoted_to="core@0.1.0")
        http.publish("acme", "0.2.0", ["acme--chat_completions"])

        report = _apply(tmp_path, _url_section("acme"), http)

        status = _vendor_status(report, "acme")
        assert status.skipped is None
        assert status.release["version"] == "0.2.0"

    @pytest.mark.parametrize(
        ("core", "reached"),
        [
            ("0.13.9", False),
            ("0.14.0.dev35", False),
            ("0.14.0rc1", False),
            ("0.14.0", True),
            ("0.14.0.post1", True),
            ("0.14.1.dev1", True),
            ("0.15.0", True),
        ],
    )
    def test_a_dev_or_pre_release_has_not_reached_the_promotion(self, core, reached):
        from llm_rosetta.contracts.shim_manifest import parse_manifest

        from llm_rosetta_gateway.shims.manifest_sources import _promotion

        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--x"], promoted_to="core@0.14.0")
        release = parse_manifest(http.manifest()).vendors["acme"][0]

        result = _promotion(release, core, {"acme--x"}, "contrib")

        assert (result == "core@0.14.0") is reached

    def test_a_promotion_to_a_later_core_still_installs(self, tmp_path):
        http = FakeHttp()
        http.publish(
            "acme", "0.1.0", ["acme--chat_completions"], promoted_to="core@99.0.0"
        )

        report = _apply(tmp_path, _url_section("acme"), http)

        assert _vendor_status(report, "acme").shim_names == ("acme--chat_completions",)

    def test_a_partial_clash_refuses_only_the_core_names(self, tmp_path, logs):
        core_name = _core_name()
        before = get_shim(core_name)
        http = FakeHttp()
        http.publish("acme", "0.1.0", [core_name, "acme--chat_completions"])

        report = _apply(tmp_path, _url_section("acme"), http)

        status = _vendor_status(report, "acme")
        assert status.skipped is None
        assert status.shim_names == ("acme--chat_completions",)
        after = get_shim(core_name)
        assert before is not None and after is not None
        assert after.connection.base_url == before.connection.base_url
        assert any(
            f"core already provides {core_name}; those shims are ignored" in m
            for m in logs
        ), logs

    def test_the_release_record_bounds_what_registers(self, tmp_path, logs):
        """An archive without shims.yaml loads only the record's names."""
        http = FakeHttp()
        data = _archive("acme", ["acme--chat_completions", "sneaky--chat_completions"])
        http.publish("acme", "0.1.0", ["acme--chat_completions"], data=data)

        report = _apply(tmp_path, _url_section("acme"), http)

        assert _vendor_status(report, "acme").shim_names == ("acme--chat_completions",)
        assert get_shim("sneaky--chat_completions") is None
        assert any(
            "'sneaky--chat_completions' is not in the release record" in m for m in logs
        ), logs

    def test_an_explicit_path_overrides_a_manifest_install(self, tmp_path):
        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"])
        local = tmp_path / "local" / "acme"
        local.mkdir(parents=True)
        (local / "provider.yaml").write_text(
            "name: acme--chat_completions\nformat: chat_completions\n"
            "model_type: llm\nconnection:\n  base_url: https://local.example/v1\n"
        )

        _apply(
            tmp_path, {**_url_section("acme"), "paths": [str(tmp_path / "local")]}, http
        )

        shim = get_shim("acme--chat_completions")
        assert shim is not None
        assert shim.connection.base_url == "https://local.example/v1"


class TestLegacyPresets:
    def _contrib(self) -> FakeHttp:
        source = get_shim_source("contrib")
        assert source is not None
        http = FakeHttp("contrib", dist=source.manifest_url.rsplit("/", 1)[0] + "/")
        http.publish(
            "kilo_free", "0.1.0", ["kilo_free--chat_completions"], tags=["tier:free"]
        )
        http.publish("llm7", "0.1.0", ["llm7--chat_completions"], tags=["tier:free"])
        http.publish("opendesign", "0.1.0", ["opendesign--chat_completions"])
        return http

    def test_free_maps_to_the_free_tag_on_contrib(self, tmp_path):
        report = _apply(tmp_path, {"presets": ["free"]}, self._contrib())

        assert sorted(s.vendor for s in report.sources) == ["kilo_free", "llm7"]
        assert get_shim("llm7--chat_completions") is not None

    def test_opendesign_maps_to_its_vendor_on_contrib(self, tmp_path):
        report = _apply(tmp_path, {"presets": ["opendesign"]}, self._contrib())

        assert [s.name for s in report.sources] == ["contrib:opendesign"]

    def test_a_legacy_include_narrows_each_vendor_of_the_tag(self, tmp_path, logs):
        section = {"presets": ["free"], "include": {"free": ["llm7--*"]}}

        report = _apply(tmp_path, section, self._contrib(), strict=True)

        loaded = [s for s in report.sources if s.shim_names]
        assert [s.vendor for s in loaded] == ["llm7"]
        assert not any("matched no shim" in m for m in logs), logs

    def test_a_mapped_preset_the_answering_source_lacks_is_reported(
        self, tmp_path, logs
    ):
        """No legacy fallback when the source answers; a warning and a row."""
        http = self._contrib()
        del http.vendors["opendesign"]

        report = _apply(tmp_path, {"presets": ["opendesign"]}, http)

        (row,) = report.sources
        assert row.skipped == "not in the manifest"
        assert (row.name, row.source, row.vendor) == (
            "contrib:opendesign",
            "contrib",
            "opendesign",
        )
        assert describe_shim_sources(report) == [
            "contrib:opendesign: skipped (not in the manifest)"
        ]
        assert any(
            "'contrib:opendesign' (presets:opendesign) matched no vendor" in m
            for m in logs
        ), logs
        assert not any("github" in c for c in http.calls)

    def test_the_argonne_presets_map_to_their_vendors(self):
        sources = parse_shims_section({"presets": ["alcf", "argo", "asksage"]})
        assert [s.raw for s in sources.manifests.selectors] == [
            "argonne:alcf",
            "argonne:argo",
            "argonne:asksage",
        ]

    def test_a_source_that_never_answered_falls_back_to_the_legacy_remote(
        self, tmp_path, logs
    ):
        http = FakeHttp()
        http.down = True

        report = _apply(tmp_path, {"presets": ["opendesign"]}, http)

        package = get_source_package("opendesign")
        assert package is not None
        assert [s.location for s in report.sources] == [package.remote["url"]]
        assert any("loads its legacy remote" in m for m in logs), logs


class TestConfig:
    @pytest.mark.parametrize(
        ("text", "source", "kind", "value"),
        [
            ("opencode", None, "vendor", "opencode"),
            ("tag:tier:free", None, "tag", "tier:free"),
            ("*", None, "all", ""),
            ("argonne:*", "argonne", "all", ""),
            ("contrib:tag:free", "contrib", "tag", "free"),
            ("argonne:alcf", "argonne", "vendor", "alcf"),
        ],
    )
    def test_selector_spellings(self, text, source, kind, value):
        selector = parse_selector(text)
        assert (selector.source, selector.kind, selector.value) == (source, kind, value)

    @pytest.mark.parametrize("text", ["", "Bad Vendor", "tag:", 3, "x:tag:BAD"])
    def test_bad_selectors_raise(self, text):
        with pytest.raises(ValueError):
            parse_selector(text)

    def test_install_alone_reads_the_default_sources(self):
        sources = parse_shims_section({"install": ["opencode"]})
        assert [s.key for s in sources.manifests.sources] == ["contrib", "argonne"]

    def test_an_unknown_source_id_is_refused_when_strict(self):
        with pytest.raises(ValueError, match="unknown shim source 'ghost'"):
            parse_shims_section(
                {"sources": ["ghost"], "install": ["x"], "strict": True}
            )
        assert not parse_shims_section(
            {"sources": ["ghost"], "install": ["x"]}
        ).manifests

    def test_a_qualifier_naming_an_undeclared_source_is_refused_when_strict(self):
        with pytest.raises(ValueError, match="does not declare"):
            parse_shims_section(
                {"sources": ["contrib"], "install": ["argonne:alcf"], "strict": True}
            )

    def test_a_plain_http_source_url_is_refused(self):
        with pytest.raises(ValueError, match="https"):
            parse_shims_section(
                {"sources": ["http://shims.example/manifest.json"], "install": ["x"]}
            )

    def test_an_object_source_carries_its_fallback_and_token(self):
        sources = parse_shims_section(
            {
                "sources": [
                    {
                        "url": MANIFEST,
                        "fallback": "https://mirror.example/manifest.json",
                        "token_env": "SHIM_TOKEN",
                    }
                ],
                "install": ["acme"],
            }
        )
        (spec,) = sources.manifests.sources
        assert spec.fallback_url == "https://mirror.example/manifest.json"
        assert spec.token_env == "SHIM_TOKEN"

    def test_the_admin_save_keeps_sources_and_install(self):
        from llm_rosetta_gateway.admin.routes.config.shims import (
            _merge_shim_sources_body,
        )

        merged = _merge_shim_sources_body(
            {}, {"sources": ["contrib"], "install": ["opencode"]}
        )
        assert merged == {"sources": ["contrib"], "install": ["opencode"]}
