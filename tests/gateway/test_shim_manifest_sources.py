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
from types import SimpleNamespace
from typing import Any

import pytest

from llm_rosetta.api import (
    get_shim,
    list_shims,
    restore_registry,
    snapshot_registry,
)
from llm_rosetta.contracts import IR_VERSION, SURFACE_VERSION
from llm_rosetta_gateway.shims.shim_sources import (
    apply_shim_sources,
    describe_shim_sources,
    parse_shims_section,
)
from llm_rosetta_gateway.shims.catalog import get_shim_source
from llm_rosetta_gateway.shims.manifest_sources.plan import parse_selector
from llm_rosetta_gateway.shims.sources import check_source_updates

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
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())
        self.records.append(record)

    def at(self, level: int) -> list[str]:
        return [r.getMessage() for r in self.records if r.levelno == level]


@pytest.fixture
def log_handler():
    handler = _Capture()
    names = ["llm_rosetta_gateway.shims.sources", "llm-rosetta-gateway"]
    levels = {}
    for name in names:
        logger = logging.getLogger(name)
        levels[name] = logger.level
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
    try:
        yield handler
    finally:
        for name in names:
            logger = logging.getLogger(name)
            logger.removeHandler(handler)
            logger.setLevel(levels[name])


@pytest.fixture
def logs(log_handler):
    return log_handler.messages


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
        self.generated_at = "2026-11-21T00:00:00Z"

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
            "generated_at": self.generated_at,
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


def _apply(
    tmp_path: Path, section: dict[str, Any], http: FakeHttp | TwoLegs, **kw: Any
):
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

    def test_a_full_disk_keeps_the_last_good_install(self, tmp_path, monkeypatch):
        """ENOSPC says nothing about the bytes: the old install is still the answer."""
        from llm_rosetta_gateway.shims.manifest_sources import install

        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"])
        _apply(tmp_path, _url_section("acme"), http)
        http.publish("acme", "0.2.0", ["acme--chat_completions"])

        def full(src, dest):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(install, "_atomic_replace", full)
        report = _apply(tmp_path, _url_section("acme"), http, force_refresh=True)

        status = _vendor_status(report, "acme")
        assert status.ok and status.stale and status.from_cache
        assert status.release["version"] == "0.1.0"
        assert "No space left on device" in (status.error or "")
        assert get_shim("acme--chat_completions") is not None

    def test_a_cold_full_disk_is_a_plain_source_error(self, tmp_path, monkeypatch):
        """Nothing to keep: the class stays distinguishable from a bad archive."""
        from llm_rosetta_gateway.shims.manifest_sources import install

        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"])

        def full(src, dest):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(install, "_atomic_replace", full)
        report = _apply(tmp_path, _url_section("acme"), http)

        status = _vendor_status(report, "acme")
        assert status.ok is False
        assert (status.error or "").startswith("ShimSourceError:")
        assert "ShimArchiveUnusable" not in (status.error or "")
        assert "No space left on device" in (status.error or "")
        assert status.unreachable is False
        assert get_shim("acme--chat_completions") is None

    def test_a_write_denied_while_extracting_is_the_same_failure(
        self, tmp_path, monkeypatch
    ):
        """EPERM during extraction leaves no half-swapped vendor directory."""
        from llm_rosetta_gateway.shims.manifest_sources import install

        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"])

        def denied(*_a, **_kw):
            raise OSError(1, "Operation not permitted")

        monkeypatch.setattr(install, "_extract_archive", denied)
        report = _apply(tmp_path, _url_section("acme"), http)

        status = _vendor_status(report, "acme")
        assert status.ok is False
        assert (status.error or "").startswith("ShimSourceError:")
        assert "Operation not permitted" in (status.error or "")
        vendor_dirs = [
            p for p in tmp_path.rglob("acme") if p.is_dir() and p.parent.name != ".tmp"
        ]
        assert vendor_dirs == []

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

        (entry,) = check_source_updates(report, http_get=http)

        assert entry["current"] == "0.1.0"
        assert entry["head"] == "0.2.0"
        assert entry["update"] is True


class TestProvidedBy:
    def test_a_plugin_name_names_the_plugin(self, monkeypatch):
        from llm_rosetta_gateway.shims.manifest_sources.selection import _provided_by

        monkeypatch.setattr(
            "llm_rosetta.api.plugin_source", lambda name: "demo-shim 0.1.0"
        )

        assert (
            _provided_by(["demo--chat_completions"])
            == "installed plugin demo-shim 0.1.0"
        )

    def test_a_core_name_stays_core(self, monkeypatch):
        from llm_rosetta_gateway.shims.manifest_sources.selection import _provided_by

        monkeypatch.setattr("llm_rosetta.api.plugin_source", lambda name: None)

        assert _provided_by(["demo--chat_completions"]) == "core"

    def test_a_core_and_plugin_set_names_both(self, monkeypatch):
        from llm_rosetta_gateway.shims.manifest_sources.selection import _provided_by

        monkeypatch.setattr(
            "llm_rosetta.api.plugin_source",
            lambda name: None if name.startswith("core--") else "demo-shim 0.1.0",
        )

        assert (
            _provided_by(["core--chat_completions", "demo--chat_completions"])
            == "core and installed plugin(s) demo-shim 0.1.0"
        )

    def test_two_plugins_without_a_core_name_name_only_the_plugins(self, monkeypatch):
        from llm_rosetta_gateway.shims.manifest_sources.selection import _provided_by

        owners = {
            "alpha--chat_completions": "alpha-shim 1.0",
            "beta--chat_completions": "beta-shim 2.0",
        }
        monkeypatch.setattr("llm_rosetta.api.plugin_source", owners.get)

        assert (
            _provided_by(["alpha--chat_completions", "beta--chat_completions"])
            == "installed plugins alpha-shim 1.0, beta-shim 2.0"
        )


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
            "contrib" in m and repr(vendor) in m and "already provided by core" in m
            for m in logs
        ), logs
        # The estimate saves the download too.
        assert not any(c.endswith(".tar.gz") for c in http.calls)

    def test_a_promoted_vendor_is_not_installed_from_the_source(self, tmp_path, logs):
        core_name = _core_name()
        # A promoted or clashing vendor ships its own names (#1660: a manifest
        # shim name must belong to its vendor), so the vendor is core's.
        vendor = core_name.split("--", 1)[0]
        http = FakeHttp()
        http.publish(vendor, "0.1.0", [core_name], promoted_to="core@0.1.0")

        report = _apply(tmp_path, _url_section(vendor), http)

        status = _vendor_status(report, vendor)
        assert status.skipped == "promoted to core@0.1.0"
        assert describe_shim_sources(report) == [
            f"contrib:{vendor}: skipped (promoted to core@0.1.0)"
        ]
        assert any(f"{vendor} 0.1.0 says promoted_to core@0.1.0" in m for m in logs)
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
        vendor = core_name.split("--", 1)[0]
        http = FakeHttp()
        http.publish(vendor, "0.1.0", [core_name], promoted_to="core@0.1.0")
        http.publish(vendor, "0.2.0", [f"{vendor}--fake_lane"])

        report = _apply(tmp_path, _url_section(vendor), http)

        status = _vendor_status(report, vendor)
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

        from llm_rosetta_gateway.shims.manifest_sources.selection import _promotion

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
        # A promoted or clashing vendor ships its own names (#1660: a manifest
        # shim name must belong to its vendor), so the vendor is core's.
        vendor = core_name.split("--", 1)[0]
        before = get_shim(core_name)
        http = FakeHttp()
        http.publish(vendor, "0.1.0", [core_name, f"{vendor}--fake_lane"])

        report = _apply(tmp_path, _url_section(vendor), http)

        status = _vendor_status(report, vendor)
        assert status.skipped is None
        assert status.shim_names == (f"{vendor}--fake_lane",)
        after = get_shim(core_name)
        assert before is not None and after is not None
        assert after.connection.base_url == before.connection.base_url
        assert any(
            f"already provided by core: {core_name}; those shims are ignored" in m
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


class TestRetiredPresets:
    #: What each refusal tells the operator to do.  ``remotes`` has no migrate
    #: hint on purpose: the tool cannot map an explicit archive URL.
    ADVICE = {
        "presets": "'shims.install'",
        "include": "'shims.install'",
        "exclude": "'!<source>:<vendor>'",
        "remotes": "'shims.paths'",
    }

    @pytest.mark.parametrize("key", ["presets", "include", "exclude", "remotes"])
    def test_a_retired_key_is_refused_with_the_advice(self, key):
        match = f"'shims.{key}' was removed in 0.15.0: .*{self.ADVICE[key]}"
        with pytest.raises(ValueError, match=match):
            parse_shims_section({key: ["free"] if key == "presets" else {"x": ["y"]}})

    def test_remotes_names_both_ways_out_and_no_migrate_hint(self):
        with pytest.raises(ValueError) as caught:
            parse_shims_section({"remotes": [{"url": "https://x.example/a.tar.gz"}]})
        message = str(caught.value)
        assert "shims.sources" in message and "shims.paths" in message
        assert "llm-rosetta-migrate" not in message

    def test_every_retired_key_is_named_in_one_error(self):
        section = {
            "presets": ["free"],
            "include": {"free": ["llm7--*"]},
            "exclude": {"free": ["x--*"]},
            "remotes": [{"url": "https://x.example/a.tar.gz"}],
        }
        with pytest.raises(ValueError) as caught:
            parse_shims_section(section)
        for key in section:
            assert f"'shims.{key}'" in str(caught.value)

    @pytest.mark.parametrize("empty", [None, [], {}])
    @pytest.mark.parametrize("key", ["presets", "include", "exclude", "remotes"])
    def test_a_key_that_holds_nothing_is_not_a_declaration(self, key, empty):
        assert parse_shims_section({key: empty}).install == ()


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


class TestRecordLevelManifest:
    """One bad record must not take the whole source down (#1660)."""

    def test_a_bad_record_is_skipped_and_the_good_ones_install(self, tmp_path, logs):
        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"])
        http.publish("beta", "0.1.0", ["beta--chat_completions"])
        http.publish("acme", "0.2.0", ["acme--chat_completions"])
        # A newer producer writes a record this core cannot read.
        http.vendors["acme"][0]["sha256"] = "not-a-digest"

        report = _apply(tmp_path, _url_section("acme", "beta"), http)

        assert _vendor_status(report, "acme").release["version"] == "0.1.0"
        assert _vendor_status(report, "beta").ok
        assert get_shim("acme--chat_completions") is not None
        assert get_shim("beta--chat_completions") is not None
        skipped = [m for m in logs if "manifest record vendor 'acme'" in m]
        assert len(skipped) == 1
        assert "version '0.2.0' skipped" in skipped[0] and "sha256" in skipped[0]

    def test_a_bad_top_level_still_fails_the_source(self, tmp_path, logs):
        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"])
        manifest = http.manifest

        def broken():
            document = manifest()
            document["manifest_version"] = 99
            return document

        setattr(http, "manifest", broken)

        report = _apply(tmp_path, _url_section("acme"), http)

        assert get_shim("acme--chat_completions") is None
        failed = [s for s in report.sources if not s.ok]
        assert failed and "manifest_version" in (failed[0].error or "")

    def test_a_vendor_with_no_valid_release_says_so(self, tmp_path, logs):
        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"])
        http.publish("beta", "0.1.0", ["beta--chat_completions"])
        http.vendors["acme"][0]["sha256"] = "not-a-digest"

        report = _apply(tmp_path, _url_section("acme", "beta"), http)

        status = _vendor_status(report, "acme")
        assert status.skipped == (
            "no valid release: $.vendors.acme.versions[0].sha256: "
            "'not-a-digest' does not match ^[0-9a-f]{64}$"
        )
        assert get_shim("acme--chat_completions") is None
        assert _vendor_status(report, "beta").ok
        assert any("manifest record vendor 'acme'" in m for m in logs)
        assert any(
            "vendor 'acme' has no valid release" in m and "not installed" in m
            for m in logs
        )

    def test_an_unlisted_vendor_is_still_not_in_the_manifest(self, tmp_path, logs):
        http = FakeHttp()
        http.publish("beta", "0.1.0", ["beta--chat_completions"])

        report = _apply(tmp_path, _url_section("acme", "beta"), http)

        assert _vendor_status(report, "acme").skipped == "not in the manifest"

    def test_a_good_install_is_kept_when_every_record_goes_bad(self, tmp_path, logs):
        """A record that becomes invalid never uninstalls a working vendor."""
        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"])
        _apply(tmp_path, _url_section("acme"), http)
        assert get_shim("acme--chat_completions") is not None

        http.vendors["acme"][0]["sha256"] = "not-a-digest"
        report = _apply(tmp_path, _url_section("acme"), http, force_refresh=True)

        status = _vendor_status(report, "acme")
        assert status.ok and status.stale and status.skipped is None
        assert status.error == (
            "no valid release: $.vendors.acme.versions[0].sha256: "
            "'not-a-digest' does not match ^[0-9a-f]{64}$"
        )
        assert status.release["version"] == "0.1.0"
        assert get_shim("acme--chat_completions") is not None
        assert any("keeping the installed 0.1.0" in m for m in logs)

    def test_a_kept_install_still_answers_a_tag_selector(self, tmp_path, logs):
        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"], tags=["tier:free"])
        _apply(tmp_path, _url_section("tag:tier:free"), http)
        http.vendors["acme"][0]["sha256"] = "not-a-digest"

        report = _apply(
            tmp_path, _url_section("tag:tier:free"), http, force_refresh=True
        )

        assert _vendor_status(report, "acme").stale
        assert get_shim("acme--chat_completions") is not None


MIRROR_DIST = "https://mirror.example/dist/"
MIRROR = MIRROR_DIST + "manifest.json"
OLD = "2026-11-21T11:59:00Z"
NEW = "2026-11-21T16:19:00Z"


class TwoLegs:
    """A primary and a fallback URL, each serving its own manifest copy."""

    def __init__(self, dist: str = DIST, mirror_dist: str = MIRROR_DIST) -> None:
        self.primary = FakeHttp(dist=dist)
        self.mirror = FakeHttp(dist=mirror_dist)
        self.mirror_dist = mirror_dist
        self.calls: list[str] = []

    def publish(self, vendor: str, version: str, shims: list[str], **kw: Any) -> None:
        """Publish to both legs (the same archive bytes)."""
        data = _archive(vendor, shims, marker=version)
        self.primary.publish(vendor, version, shims, data=data, **kw)
        self.mirror.publish(vendor, version, shims, data=data, **kw)

    def manifest_calls(self) -> list[str]:
        return [c for c in self.calls if c.endswith("manifest.json")]

    def __call__(self, url: str, **kw: Any) -> bytes:
        self.calls.append(url)
        leg = self.mirror if url.startswith(self.mirror_dist) else self.primary
        return leg(url, **kw)


def _legs_section(*install: str) -> dict[str, Any]:
    return {
        "sources": [{"url": MANIFEST, "fallback": MIRROR}],
        "install": list(install),
    }


@pytest.fixture
def clock(monkeypatch):
    """The clock the shim loader reads; advance ``clock["t"]`` to move it."""
    from llm_rosetta_gateway.shims import sources as loader

    state = {"t": 1_800_000_000.0}
    monkeypatch.setattr(loader, "time", SimpleNamespace(time=lambda: state["t"]))
    return state


def _cached_generated_at(tmp_path: Path) -> str:
    (doc,) = tmp_path.rglob("manifests/*/manifest.json")
    return json.loads(doc.read_text())["generated_at"]


class TestFreshnessGuard:
    """A manifest URL that answers with an old copy (#1660, the CDN lag)."""

    def test_a_stale_primary_without_the_named_vendor_yields_to_the_fallback(
        self, tmp_path, log_handler
    ):
        legs = TwoLegs()
        legs.publish("other", "0.1.0", ["other--chat_completions"])
        legs.primary.generated_at = OLD
        legs.mirror.generated_at = NEW
        data = _archive("acme", ["acme--chat_completions"])
        legs.mirror.publish("acme", "0.1.0", ["acme--chat_completions"], data=data)

        report = _apply(tmp_path, _legs_section("contrib:acme"), legs)

        status = _vendor_status(report, "acme")
        assert status.ok and not status.stale
        assert status.archive_url == MIRROR_DIST + "0.1.0/acme.tar.gz"
        assert get_shim("acme--chat_completions") is not None
        assert legs.manifest_calls() == [MANIFEST, MIRROR]
        assert _cached_generated_at(tmp_path) == NEW
        (line,) = [m for m in log_handler.at(logging.INFO) if "using" in m]
        assert f"using {MIRROR} (generated_at {NEW})" in line
        assert "does not list acme" in line and OLD in line

    def test_a_tie_keeps_the_primary(self, tmp_path, log_handler):
        legs = TwoLegs()
        legs.publish("other", "0.1.0", ["other--chat_completions"])
        data = _archive("acme", ["acme--chat_completions"])
        legs.mirror.publish("acme", "0.1.0", ["acme--chat_completions"], data=data)

        report = _apply(tmp_path, _legs_section("contrib:acme"), legs)

        assert _vendor_status(report, "acme").skipped == "not in the manifest"
        assert any(f"using {MANIFEST}" in m for m in log_handler.at(logging.INFO))

    def test_a_primary_older_than_the_cache_yields_to_a_fresher_fallback(
        self, tmp_path
    ):
        legs = TwoLegs()
        legs.publish("acme", "0.1.0", ["acme--chat_completions"])
        legs.primary.generated_at = legs.mirror.generated_at = OLD
        _apply(tmp_path, _legs_section("acme"), legs)
        legs.primary.generated_at = "2026-11-21T11:00:00Z"
        legs.mirror.generated_at = NEW
        legs.calls.clear()

        report = _apply(tmp_path, _legs_section("acme"), legs, force_refresh=True)

        assert legs.manifest_calls() == [MANIFEST, MIRROR]
        assert not _vendor_status(report, "acme").stale
        assert _cached_generated_at(tmp_path) == NEW

    def test_a_manifest_older_than_the_cache_never_replaces_it(
        self, tmp_path, log_handler
    ):
        legs = TwoLegs()
        legs.publish("acme", "0.1.0", ["acme--chat_completions"])
        legs.primary.generated_at = legs.mirror.generated_at = NEW
        _apply(tmp_path, _legs_section("acme"), legs)
        legs.primary.generated_at = legs.mirror.generated_at = OLD

        report = _apply(tmp_path, _legs_section("acme"), legs, force_refresh=True)

        status = _vendor_status(report, "acme")
        assert status.ok and status.stale and status.error
        assert f"generated_at {OLD}" in status.error
        assert f"generated_at {NEW}" in status.error
        assert _cached_generated_at(tmp_path) == NEW
        assert get_shim("acme--chat_completions") is not None
        assert any(
            "keeping the cached manifest" in m for m in log_handler.at(logging.WARNING)
        )

    def test_a_single_url_source_never_goes_backwards_either(self, tmp_path):
        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"])
        http.generated_at = NEW
        _apply(tmp_path, _url_section("acme"), http)
        http.generated_at = OLD
        http.calls.clear()

        report = _apply(tmp_path, _url_section("acme"), http, force_refresh=True)

        assert http.calls == [MANIFEST]
        assert _vendor_status(report, "acme").stale
        assert _cached_generated_at(tmp_path) == NEW

    def test_a_kept_cache_is_rechecked_until_a_fresh_one_arrives(self, tmp_path):
        """The never-backwards branch makes the next apply ask again."""
        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"])
        http.generated_at = NEW
        _apply(tmp_path, _url_section("acme"), http)
        http.generated_at = OLD
        _apply(tmp_path, _url_section("acme"), http, force_refresh=True)
        http.calls.clear()

        report = _apply(tmp_path, _url_section("acme"), http)

        assert http.calls == [MANIFEST]
        assert _vendor_status(report, "acme").stale
        http.generated_at = NEW
        http.calls.clear()

        report = _apply(tmp_path, _url_section("acme"), http)

        assert http.calls == [MANIFEST]
        assert not _vendor_status(report, "acme").stale
        http.calls.clear()
        _apply(tmp_path, _url_section("acme"), http)
        assert http.calls == []

    def test_the_normal_case_requests_one_manifest(self, tmp_path, log_handler):
        legs = TwoLegs()
        legs.publish("acme", "0.1.0", ["acme--chat_completions"])

        _apply(tmp_path, _legs_section("contrib:acme"), legs)
        _apply(tmp_path, _legs_section("contrib:acme"), legs, force_refresh=True)

        assert legs.manifest_calls() == [MANIFEST, MANIFEST]
        assert not any("using" in m for m in log_handler.at(logging.INFO))

    def test_an_unqualified_selector_does_not_trigger_the_guard(self, tmp_path):
        legs = TwoLegs()
        legs.publish("other", "0.1.0", ["other--chat_completions"])

        _apply(tmp_path, _legs_section("acme"), legs)

        assert legs.manifest_calls() == [MANIFEST]

    def test_an_unreachable_fallback_during_the_guard_keeps_the_primary(
        self, tmp_path, log_handler
    ):
        legs = TwoLegs()
        legs.publish("other", "0.1.0", ["other--chat_completions"])
        legs.mirror.down = True

        report = _apply(tmp_path, _legs_section("contrib:acme", "contrib:other"), legs)

        assert legs.manifest_calls() == [MANIFEST, MIRROR]
        assert _vendor_status(report, "other").ok
        assert _vendor_status(report, "acme").skipped == "not in the manifest"
        assert any(
            f"{MANIFEST} answered but it does not list acme; {MIRROR} failed" in m
            for m in log_handler.at(logging.WARNING)
        ), log_handler.messages

    def test_a_cached_manifest_without_a_named_vendor_is_refetched(self, tmp_path):
        legs = TwoLegs()
        legs.publish("other", "0.1.0", ["other--chat_completions"])
        _apply(tmp_path, _legs_section("contrib:other"), legs)
        legs.calls.clear()

        _apply(tmp_path, _legs_section("contrib:other"), legs)
        assert legs.manifest_calls() == []

        _apply(tmp_path, _legs_section("contrib:other", "contrib:acme"), legs)
        assert legs.manifest_calls() == [MANIFEST, MIRROR]


@pytest.mark.parametrize(
    ("earlier", "later"),
    [
        ("2026-11-21T11:59:00Z", "2026-11-21T16:19:00Z"),
        ("2026-11-21T16:19:00.123456789Z", "2026-11-21T16:19:00.2Z"),
        ("2026-11-21T20:00:00+08:00", "2026-11-21T12:30:00Z"),
        ("2026-11-21T12:00:00-00:30", "2026-11-21T12:31:00z"),
    ],
)
def test_generated_at_compares_as_time(earlier, later):
    from llm_rosetta.contracts.shim_manifest import SourceManifest
    from llm_rosetta_gateway.shims.manifest_sources.freshness import _older

    def manifest(when: str) -> SourceManifest:
        return SourceManifest(
            manifest_version=1, source="s", repo="r", generated_at=when, vendors={}
        )

    assert _older(manifest(earlier), manifest(later))
    assert not _older(manifest(later), manifest(earlier))
    assert not _older(manifest(earlier), manifest(earlier))
    assert not _older(manifest("not a time"), manifest(later))


class TestFreshnessGuardEdges:
    """Invalid records, the recheck cooldown and the leg order."""

    def test_a_vendor_only_in_dropped_counts_as_missing(self, tmp_path, log_handler):
        """A vendor whose every record is invalid is doubt, not presence."""
        legs = TwoLegs()
        data = _archive("acme", ["acme--chat_completions"])
        legs.primary.publish(
            "acme", "0.1.0", ["acme--chat_completions"], sha256="not-a-digest"
        )
        legs.mirror.publish("acme", "0.1.0", ["acme--chat_completions"], data=data)
        legs.primary.generated_at = OLD
        legs.mirror.generated_at = NEW

        report = _apply(tmp_path, _legs_section("contrib:acme"), legs)

        assert legs.manifest_calls() == [MANIFEST, MIRROR]
        status = _vendor_status(report, "acme")
        assert status.ok and status.archive_url == MIRROR_DIST + "0.1.0/acme.tar.gz"
        assert any("does not list acme" in m for m in log_handler.at(logging.INFO)), (
            log_handler.messages
        )

    def test_a_vendor_no_source_publishes_is_checked_once_per_window(
        self, tmp_path, clock
    ):
        """The missing-vendor recheck has a cooldown."""
        from llm_rosetta_gateway.shims.manifest_sources.freshness import (
            VENDOR_RECHECK_SECONDS,
        )

        legs = TwoLegs()
        legs.publish("acme", "0.1.0", ["acme--chat_completions"])
        section = _legs_section("contrib:acme", "contrib:ghost")

        _apply(tmp_path, section, legs)
        assert legs.manifest_calls() == [MANIFEST, MIRROR]

        legs.calls.clear()
        clock["t"] += VENDOR_RECHECK_SECONDS - 1
        _apply(tmp_path, section, legs)
        assert legs.manifest_calls() == []

        clock["t"] += 2
        _apply(tmp_path, section, legs)
        assert legs.manifest_calls() == [MANIFEST, MIRROR]

        legs.calls.clear()
        clock["t"] += 1
        _apply(tmp_path, section, legs)
        assert legs.manifest_calls() == []

    def test_refetch_ignores_the_cooldown(self, tmp_path, clock):
        legs = TwoLegs()
        legs.publish("acme", "0.1.0", ["acme--chat_completions"])
        section = _legs_section("contrib:acme", "contrib:ghost")
        _apply(tmp_path, section, legs)
        legs.calls.clear()
        clock["t"] += 1

        _apply(tmp_path, section, legs, force_refresh=True)

        assert legs.manifest_calls() == [MANIFEST, MIRROR]

    def test_a_cooldown_covers_only_the_vendors_it_checked(self, tmp_path, clock):
        legs = TwoLegs()
        legs.publish("acme", "0.1.0", ["acme--chat_completions"])
        _apply(tmp_path, _legs_section("contrib:acme", "contrib:ghost"), legs)
        legs.calls.clear()
        clock["t"] += 1

        _apply(
            tmp_path,
            _legs_section("contrib:acme", "contrib:ghost", "contrib:phantom"),
            legs,
        )

        assert legs.manifest_calls() == [MANIFEST, MIRROR]

    def test_after_the_ttl_the_primary_is_tried_first_again(self, tmp_path, clock):
        """A fallback win is not sticky past the TTL."""
        from llm_rosetta_gateway.shims._types import REMOTE_TTL_SECONDS

        legs = TwoLegs()
        legs.publish("other", "0.1.0", ["other--chat_completions"])
        data = _archive("acme", ["acme--chat_completions"])
        legs.mirror.publish("acme", "0.1.0", ["acme--chat_completions"], data=data)
        legs.primary.generated_at = OLD
        legs.mirror.generated_at = NEW
        _apply(tmp_path, _legs_section("contrib:acme"), legs)
        legs.calls.clear()

        clock["t"] += 60
        _apply(tmp_path, _legs_section("contrib:acme"), legs, force_refresh=True)
        assert legs.manifest_calls() == [MIRROR]

        legs.calls.clear()
        clock["t"] += REMOTE_TTL_SECONDS + 1
        _apply(tmp_path, _legs_section("contrib:acme"), legs)
        assert legs.manifest_calls()[0] == MANIFEST


class TestTagSelectorGuard:
    """A tag selector is not checked against a vendor list."""

    FREE = ("kilo_free", "llm7", "opencode_zen_free", "ovhcloud")

    def _legs(self) -> TwoLegs:
        source = get_shim_source("contrib")
        assert source is not None
        legs = TwoLegs(
            dist=source.manifest_url.rsplit("/", 1)[0] + "/",
            mirror_dist=source.fallback_url.rsplit("/", 1)[0] + "/",
        )
        for vendor in self.FREE[:2]:
            legs.publish(
                vendor, "0.1.0", [f"{vendor}--chat_completions"], tags=["tier:free"]
            )
        for vendor in self.FREE[2:]:
            data = _archive(vendor, [f"{vendor}--chat_completions"])
            legs.mirror.publish(
                vendor,
                "0.1.0",
                [f"{vendor}--chat_completions"],
                data=data,
                tags=["tier:free"],
            )
        legs.primary.generated_at = OLD
        legs.mirror.generated_at = NEW
        return legs

    def test_a_plain_tag_selector_is_not_checked(self, tmp_path):
        legs = self._legs()

        report = _apply(
            tmp_path,
            {"sources": ["contrib"], "install": ["contrib:tag:tier:free"]},
            legs,
        )

        assert len(legs.manifest_calls()) == 1
        assert {s.vendor for s in report.sources if s.ok} == set(self.FREE[:2])
