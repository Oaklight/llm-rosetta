"""The per-source vendor view the Shims page reads (#1660 S9).

Every manifest is served by an injected ``http_get``; nothing reaches the
network.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from llm_rosetta.contracts import IR_VERSION, SURFACE_VERSION
from llm_rosetta.contracts.shim_manifest import parse_manifest
from llm_rosetta_gateway.shims.manifest_sources import (
    ManifestSourceSpec,
    parse_selector,
)
from llm_rosetta_gateway.shims.source_view import (
    release_requires_problem,
    source_catalogue,
    vendor_view,
)
from llm_rosetta_gateway.shims.sources import ShimSourceStatus

A = "https://a.example/dist/manifest.json"
B = "https://b.example/dist/manifest.json"


def _record(version: str, *, surface=SURFACE_VERSION, ir=IR_VERSION, tags=(), **kw):
    return {
        "version": version,
        "commit": "c" * 40,
        "archive": f"{version}/acme.tar.gz",
        "sha256": "0" * 64,
        "requires": {"surface": surface, "ir": ir},
        "shims": kw.pop("shims", ["acme--chat_completions"]),
        "tags": list(tags),
        "maturity": kw.pop("maturity", "beta"),
        **kw,
    }


def _manifest(source: str, vendors: dict[str, list[dict[str, Any]]]) -> dict:
    return {
        "manifest_version": 1,
        "source": source,
        "repo": f"https://github.com/example/{source}",
        "generated_at": "2026-11-21T00:00:00Z",
        "vendors": {v: {"versions": r} for v, r in vendors.items()},
    }


def _releases(*records):
    return parse_manifest(_manifest("x", {"acme": list(records)})).vendors["acme"]


def _status(version: str, *, surface=SURFACE_VERSION, **kw) -> ShimSourceStatus:
    release = _record(version, surface=surface)
    return ShimSourceStatus(
        kind="remote",
        location=A + "#acme",
        source="x",
        vendor="acme",
        release=release,
        shim_names=("acme--chat_completions",),
        **kw,
    )


class FakeHttp:
    def __init__(self, docs: dict[str, dict]) -> None:
        self.docs = docs
        self.calls: list[str] = []

    def __call__(self, url, *, headers=None, timeout=None, proxy=None) -> bytes:
        self.calls.append(url)
        if url in self.docs:
            return json.dumps(self.docs[url]).encode()
        raise ConnectionError(f"no route: {url}")


class TestVendorView:
    def test_a_newer_installable_release_is_an_update(self):
        view = vendor_view(
            "acme",
            _releases(_record("0.1.0"), _record("0.2.0")),
            source_id="x",
            selectors=[parse_selector("acme")],
            status=_status("0.1.0"),
        )

        assert view["installed"] == "0.1.0"
        assert view["installable"] == "0.2.0"
        assert view["newest"] == "0.2.0"
        assert view["update_available"] is True
        assert view["requires_ok"] is True
        assert view["state"] == "installed"
        assert view["selected_by"] == "acme"

    def test_a_newest_release_this_core_cannot_run_is_named(self):
        view = vendor_view(
            "acme",
            _releases(_record("0.1.0"), _record("0.2.0", surface=SURFACE_VERSION + 1)),
            source_id="x",
            selectors=[],
            status=None,
        )

        assert view["newest"] == "0.2.0"
        assert view["installable"] == "0.1.0"
        assert "surface" in view["newest_problem"]
        assert view["requires_ok"] is None
        assert view["state"] == "available"
        assert view["selected_by"] is None

    def test_an_installed_release_whose_requires_no_longer_match_says_so(self):
        view = vendor_view(
            "acme",
            _releases(_record("0.1.0")),
            source_id="x",
            selectors=[parse_selector("x:acme")],
            status=_status("0.0.9", surface=SURFACE_VERSION + 1),
        )

        assert view["requires_ok"] is False
        assert "surface" in view["installed_problem"]

    def test_a_tag_selector_reads_the_installable_record(self):
        view = vendor_view(
            "acme",
            _releases(
                _record("0.1.0", tags=["tier:free"]),
                _record("0.2.0", surface=SURFACE_VERSION + 1),
            ),
            source_id="x",
            selectors=[parse_selector("other:*"), parse_selector("tag:tier:free")],
            status=None,
        )

        assert view["selected_by"] == "tag:tier:free"
        assert view["tags"] == ["tier:free"]
        assert view["state"] == "selected"

    def test_a_skipped_vendor_carries_the_reason(self):
        status = ShimSourceStatus(
            kind="remote",
            location=A + "#acme",
            source="x",
            vendor="acme",
            skipped="not in the manifest",
        )

        view = vendor_view(
            "acme",
            _releases(_record("0.1.0")),
            source_id="x",
            selectors=[],
            status=status,
        )

        assert view["state"] == "skipped"
        assert view["skipped"] == "not in the manifest"

    def test_every_preset_that_selects_the_vendor_is_listed(self):
        """A preset behind an explicit entry is still reported (#1663 review)."""
        preset = parse_selector("x:acme", origin="presets:acme_pkg")
        tagged = parse_selector("tag:tier:free", origin="presets:free")
        view = vendor_view(
            "acme",
            _releases(_record("0.1.0", tags=["tier:free"])),
            source_id="x",
            selectors=[parse_selector("x:acme"), preset, tagged],
            status=None,
        )

        assert view["selected_by"] == "x:acme"
        assert view["selected_origin"] == "install"
        assert view["selected_presets"] == ["acme_pkg", "free"]

    def test_a_shim_core_registers_is_reported(self):
        view = vendor_view(
            "acme",
            _releases(_record("0.1.0", shims=["acme--chat_completions", "acme--x"])),
            source_id="x",
            selectors=[],
            status=None,
            core_names={"acme--x", "openai--chat_completions"},
        )

        assert view["core_provides"] == ["acme--x"]

    def test_an_unreadable_requirement_is_a_problem(self):
        assert release_requires_problem({"requires": {}}) is not None
        assert (
            release_requires_problem(
                {"requires": {"surface": SURFACE_VERSION, "ir": IR_VERSION}}
            )
            is None
        )


def _catalogue(tmp_path, specs, http, **kw):
    kw.setdefault("raw_sources", [])
    kw.setdefault("selectors", [])
    kw.setdefault("statuses", [])
    return source_catalogue(specs, cache_root=tmp_path, ttl=3600, http_get=http, **kw)


class TestSourceCatalogue:
    def test_every_vendor_of_every_source_is_listed(self, tmp_path):
        http = FakeHttp(
            {
                A: _manifest(
                    "alpha",
                    {
                        "acme": [_record("0.1.0")],
                        "zed": [_record("1.0.0", shims=["zed--chat_completions"])],
                    },
                ),
                B: _manifest("beta", {"acme": [_record("0.3.0")]}),
            }
        )
        specs = [
            ManifestSourceSpec(key=None, url=A),
            ManifestSourceSpec(key=None, url=B),
        ]

        out = _catalogue(
            tmp_path, specs, http, raw_sources=[A], selectors=[parse_selector("acme")]
        )

        assert [s["id"] for s in out] == ["alpha", "beta"]
        assert [v["vendor"] for v in out[0]["vendors"]] == ["acme", "zed"]
        assert out[0]["configured"] is True and out[0]["entry"] == A
        assert out[1]["configured"] is False and out[1]["entry"] is None
        # The first source in order provides a vendor both ship.
        assert out[0]["vendors"][0]["claimed_by"] is None
        assert out[1]["vendors"][0]["claimed_by"] == "alpha"

    def test_an_unreachable_source_is_reported_without_vendors(self, tmp_path):
        out = _catalogue(tmp_path, [ManifestSourceSpec(key=None, url=A)], FakeHttp({}))

        assert out[0]["ok"] is False
        assert "no route" in out[0]["error"]
        assert out[0]["vendors"] == []

    def test_a_fresh_cache_is_reused_and_refresh_reads_again(self, tmp_path):
        http = FakeHttp({A: _manifest("alpha", {"acme": [_record("0.1.0")]})})
        specs = [ManifestSourceSpec(key=None, url=A)]

        _catalogue(tmp_path, specs, http)
        _catalogue(tmp_path, specs, http)
        assert http.calls == [A]

        _catalogue(tmp_path, specs, http, refresh=True)
        assert http.calls == [A, A]

    def test_a_status_row_is_matched_by_source_and_vendor(self, tmp_path):
        http = FakeHttp({A: _manifest("alpha", {"acme": [_record("0.2.0")]})})
        status = ShimSourceStatus(
            kind="remote",
            location=A + "#acme",
            source="alpha",
            vendor="acme",
            release=_record("0.1.0"),
        )

        out = _catalogue(
            tmp_path, [ManifestSourceSpec(key=None, url=A)], http, statuses=[status]
        )

        vendor = out[0]["vendors"][0]
        assert vendor["installed"] == "0.1.0" and vendor["update_available"] is True


class TestEndpoint:
    def _call(self, request):
        from llm_rosetta_gateway.admin.routes.config.shim_source_catalog import (
            get_shim_source_catalog,
        )

        response = asyncio.run(get_shim_source_catalog(request))
        return response.status_code, json.loads(response.body)

    def test_with_no_source_declared_the_catalogue_sources_are_implied(
        self, tmp_path, monkeypatch
    ):
        from llm_rosetta_gateway.shims import sources as shim_sources
        from llm_rosetta_gateway.shims.catalog import get_shim_source

        contrib = get_shim_source("contrib")
        assert contrib is not None
        http = FakeHttp(
            {contrib.manifest_url: _manifest("contrib", {"acme": [_record("0.1.0")]})}
        )
        monkeypatch.setattr(shim_sources, "_default_http_get", http)
        config = SimpleNamespace(
            shims=SimpleNamespace(
                manifests=None,
                sources=(),
                install=(),
                cache_root=str(tmp_path),
                remote_ttl=60,
            ),
            proxy=None,
        )
        request = SimpleNamespace(
            app=SimpleNamespace(gateway_config=config, shim_source_report=None),
            query_params={},
        )

        status, body = self._call(request)

        assert status == 200
        assert body["core"]["surface"] == SURFACE_VERSION
        ids = [s["id"] for s in body["sources"]]
        assert ids == ["contrib", "argonne"]
        assert all(s["configured"] is False for s in body["sources"])
        assert body["sources"][0]["vendors"][0]["vendor"] == "acme"
        # argonne answered nothing here: reported, not raised.
        assert body["sources"][1]["ok"] is False

    def test_the_route_is_registered_ahead_of_the_shim_name_route(self):
        import inspect

        from llm_rosetta_gateway.admin.routes import config as config_routes

        text = inspect.getsource(config_routes.register_config_routes)
        assert text.index("/admin/api/config/shims/source-catalog") < text.index(
            '"/admin/api/config/shims/<name>", methods=["GET"]'
        )


class TestPackageCard:
    def test_the_card_reports_whether_requires_match(self):
        from llm_rosetta_gateway.admin.routes.config.shim_files import (
            _manifest_package_view,
        )

        good = ShimSourceStatus(
            kind="remote",
            location=A + "#a",
            source="x",
            vendor="a",
            release=_record("0.1.0", tags=["tier:free"]),
        )
        bad = ShimSourceStatus(
            kind="remote",
            location=A + "#b",
            source="x",
            vendor="b",
            release=_record("0.1.0", surface=SURFACE_VERSION + 1),
        )

        view = _manifest_package_view([good, bad], {})

        assert view["requires_problem"]["a"] is None
        assert "surface" in view["requires_problem"]["b"]
        assert view["requires_ok"] is False
        assert view["tags"] == {"a": ["tier:free"], "b": []}

    @pytest.mark.parametrize("statuses", [[]])
    def test_a_package_with_no_manifest_vendor_adds_nothing(self, statuses):
        from llm_rosetta_gateway.admin.routes.config.shim_files import (
            _manifest_package_view,
        )

        assert _manifest_package_view(statuses, {}) == {}


class TestCatalogueMarks:
    """A vendor that is not installed still shows its mark and its name.

    The manifest carries neither, and before #1660 core declared both in the
    vendor's ``provider.yaml``; after the move an uninstalled vendor drew its
    initials (``Gr``, ``Za``) and its key (``Zai``) on the Available view.
    """

    def test_a_catalogue_source_names_the_mark_and_label_of_its_vendors(self, tmp_path):
        from llm_rosetta_gateway.shims.catalog import get_shim_source
        from llm_rosetta_gateway.shims.manifest_sources import (
            source_spec_from_catalogue,
        )

        contrib = get_shim_source("contrib")
        assert contrib is not None
        spec = source_spec_from_catalogue(contrib)
        http = FakeHttp(
            {
                spec.url: _manifest(
                    "contrib",
                    {
                        "zai": [_record("0.5.0", shims=["zai--chat_completions"])],
                        "newcomer": [
                            _record("0.1.0", shims=["newcomer--chat_completions"])
                        ],
                    },
                )
            }
        )

        out = _catalogue(tmp_path, [spec], http)

        by_vendor = {v["vendor"]: v for v in out[0]["vendors"]}
        assert by_vendor["zai"]["label"] == "Z.ai"
        assert by_vendor["zai"]["logo"].endswith("/zhipu.svg")
        # A vendor the catalogue does not know keeps the initials fallback.
        assert by_vendor["newcomer"]["label"] is None
        assert by_vendor["newcomer"]["logo"] is None

    def test_a_url_source_carries_no_catalogue_mark(self, tmp_path):
        zai = [_record("0.5.0", shims=["zai--chat_completions"])]
        http = FakeHttp({A: _manifest("contrib", {"zai": zai})})

        out = _catalogue(tmp_path, [ManifestSourceSpec(key=None, url=A)], http)

        assert out[0]["vendors"][0]["logo"] is None
        assert out[0]["vendors"][0]["label"] is None

    def test_no_mark_points_at_an_archived_package_repository(self):
        from llm_rosetta_gateway.shims.catalog import (
            SHIM_SOURCE_PACKAGES,
            SHIM_SOURCES,
        )

        archived = {
            "llm-rosetta-shim-alcf",
            "llm-rosetta-shim-argo@",
            "llm-rosetta-shim-asksage",
            "llm-rosetta-shim-opendesign",
            "llm-rosetta-shim-free",
        }
        urls = [u for s in SHIM_SOURCES for u in s.logos.values()]
        urls += [u for p in SHIM_SOURCE_PACKAGES for u in p.logos.values()]
        assert urls
        for url in urls:
            assert not any(name in url for name in archived), url

    def test_a_package_reads_its_marks_from_its_source(self):
        from llm_rosetta_gateway.shims.catalog import (
            get_shim_source,
            get_source_package,
        )

        free = get_source_package("free")
        contrib = get_shim_source("contrib")
        assert free is not None and contrib is not None
        assert free.logos == {
            v: contrib.logos[v] for v in free.vendors if v in contrib.logos
        }
        # LLM7 declares no mark of its own.
        assert "llm7" not in free.logos
