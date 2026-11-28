"""The per-source vendor view the Shims page reads (#1660 S9).

Every manifest is served by an injected ``http_get``; nothing reaches the
network.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any


from llm_rosetta.api import parse_index
from llm_rosetta.contracts import IR_VERSION, SURFACE_VERSION
from llm_rosetta_gateway.app.admin_context import GatewayAdminContext
from llm_rosetta_gateway.shims.manifest_sources.plan import (
    ManifestPlan,
    ManifestSourceSpec,
    parse_selector,
    source_spec_from_catalogue,
)
from llm_rosetta_gateway.shims.source_view import (
    catalogue_payload,
    release_requires_problem,
    source_catalogue,
    vendor_view,
)
from llm_rosetta_gateway.shims.sources import ShimSourceStatus

A = "https://a.example/dist/index.json"
B = "https://b.example/dist/index.json"


def _record(version: str, *, surface=SURFACE_VERSION, ir=IR_VERSION, tags=(), **kw):
    """One index entry (also the shape of an install stamp's release record)."""
    return {
        "version": version,
        "url": f"https://a.example/dist/{version}/acme.whl",
        "sha256": "0" * 64,
        "surface": surface,
        "ir": ir,
        "shims": kw.pop("shims", ["acme--chat_completions"]),
        "tags": list(tags),
        "maturity": kw.pop("maturity", "beta"),
        "commit": "c" * 40,
        **kw,
    }


def _manifest(source: str, vendors: dict[str, list[dict[str, Any]]]) -> dict:
    return {
        "index_version": 1,
        "source": source,
        "generated_at": "2026-11-21T00:00:00Z",
        "extensions": vendors,
    }


def _releases(*records):
    return parse_index(_manifest("x", {"acme": list(records)}))["acme"]


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
        assert release_requires_problem({}) is not None
        assert release_requires_problem(_record("0.1.0")) is None
        assert (
            release_requires_problem(_record("0.1.0", surface=SURFACE_VERSION + 1))
            is not None
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
            app=SimpleNamespace(
                admin=GatewayAdminContext(),
                gateway_config=config,
                shim_source_report=None,
            ),
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


class TestManifestMarks:
    """A vendor's name and mark come from its release record (#1671).

    The manifest is the one home: core keeps no per-vendor copy, so a vendor
    that is not installed is drawn from what its source publishes.
    """

    def test_the_card_takes_label_and_logo_from_the_installable_record(self, tmp_path):
        old = _record("0.1.0", name="Kilo", logo="https://cdn.example/old.svg")
        new = _record(
            "0.2.0", name="Kilo (Free)", logo="https://cdn.example/kilocode.svg"
        )
        http = FakeHttp({A: _manifest("contrib", {"acme": [old, new]})})

        out = _catalogue(tmp_path, [ManifestSourceSpec(key=None, url=A)], http)

        card = out[0]["vendors"][0]
        assert card["label"] == "Kilo (Free)"
        assert card["logo"] == "https://cdn.example/kilocode.svg"

    def test_nothing_installable_reads_the_newest_record(self, tmp_path):
        newest = _record(
            "0.3.0",
            surface=SURFACE_VERSION + 1,
            name="Acme",
            logo="https://a.example/a.svg",
        )
        http = FakeHttp({A: _manifest("contrib", {"acme": [newest]})})

        card = _catalogue(tmp_path, [ManifestSourceSpec(key=None, url=A)], http)[0][
            "vendors"
        ][0]

        assert card["installable"] is None
        assert (card["label"], card["logo"]) == ("Acme", "https://a.example/a.svg")

    def test_a_record_without_them_leaves_the_fallbacks_to_the_panel(self, tmp_path):
        http = FakeHttp({A: _manifest("contrib", {"acme": [_record("0.1.0")]})})

        card = _catalogue(tmp_path, [ManifestSourceSpec(key=None, url=A)], http)[0][
            "vendors"
        ][0]

        assert card["label"] is None and card["logo"] is None

    def test_a_bad_logo_costs_the_field_not_the_vendor(self, tmp_path):
        bad = _record("0.1.0", name="Acme", logo="http://insecure.example/a.svg")
        http = FakeHttp({A: _manifest("contrib", {"acme": [bad]})})

        card = _catalogue(tmp_path, [ManifestSourceSpec(key=None, url=A)], http)[0][
            "vendors"
        ][0]

        assert card["installable"] == "0.1.0"
        assert card["logo"] is None and card["label"] == "Acme"

    def test_core_keeps_no_per_vendor_mark(self):
        from llm_rosetta_gateway.shims.catalog import SHIM_SOURCES, ShimSource

        assert not hasattr(ShimSource, "logos") and not hasattr(ShimSource, "labels")
        for source in SHIM_SOURCES:
            assert "vendors" not in source.to_dict()


class TestPinnedMarks:
    def test_a_pinned_vendor_is_drawn_from_its_pinned_record(self, tmp_path):
        old = _record("0.1.0", name="Acme Old", logo="https://a.example/old.svg")
        new = _record("0.2.0", name="Acme New", logo="https://a.example/new.svg")
        http = FakeHttp({A: _manifest("contrib", {"acme": [old, new]})})
        spec = ManifestSourceSpec(key=None, url=A)

        def card(entry):
            return _catalogue(
                tmp_path, [spec], http, selectors=[parse_selector(entry)]
            )[0]["vendors"][0]

        pinned = card("contrib:acme@0.1.0")
        assert (pinned["label"], pinned["logo"]) == (
            "Acme Old",
            "https://a.example/old.svg",
        )
        assert card("contrib:acme")["label"] == "Acme New"

    def test_two_pins_show_as_a_pin_problem(self, tmp_path):
        http = FakeHttp({A: _manifest("contrib", {"acme": [_record("0.1.0")]})})

        card = _catalogue(
            tmp_path,
            [ManifestSourceSpec(key=None, url=A)],
            http,
            selectors=[
                parse_selector("contrib:acme@0.1.0"),
                parse_selector("acme@0.2.0"),
            ],
        )[0]["vendors"][0]

        assert "pinned more than once" in card["pin_problem"]
        assert card["label"] is None

    def test_the_install_row_speaks_for_the_card_not_an_overridden_entry(
        self, tmp_path
    ):
        http = FakeHttp({A: _manifest("alpha", {"acme": [_record("0.1.0")]})})
        install = ShimSourceStatus(
            kind="remote",
            location=A + "#acme",
            source="alpha",
            vendor="acme",
            release=_record("0.1.0"),
            stale=True,
            ok=True,
            error="acme is pinned more than once: alpha:acme@0.1.0, acme@0.2.0",
        )
        overridden = ShimSourceStatus(
            kind="remote",
            location=A + "#acme@0.2.0",
            source="alpha",
            vendor="acme",
            skipped="pinned more than once (alpha:acme@0.1.0, acme@0.2.0)",
        )

        card = _catalogue(
            tmp_path,
            [ManifestSourceSpec(key=None, url=A)],
            http,
            statuses=[install, overridden],
        )[0]["vendors"][0]

        assert card["installed"] == "0.1.0"
        assert card["state"] == "installed"

    def test_implied_defaults_are_all_read_when_a_moved_vendor_adds_one(self, tmp_path):
        """The editor shows both defaults; the catalogue must read both.

        A provider that names a moved shim makes the plan carry that vendor's
        source even when ``shims.sources`` is undeclared.  The other default
        then used to have no catalogue row, so its editor block said "save to
        read" and offered no vendors.
        """
        from llm_rosetta_gateway.shims.catalog import get_shim_source

        contrib = get_shim_source("contrib")
        argonne = get_shim_source("argonne")
        assert contrib is not None and argonne is not None
        http = FakeHttp(
            {
                contrib.manifest_url: _manifest(
                    "contrib", {"kilo": [_record("0.1.0")]}
                ),
                argonne.manifest_url: _manifest(
                    "argonne", {"argo": [_record("0.1.0")]}
                ),
            }
        )
        shims = SimpleNamespace(
            sources=(),
            install=("contrib:kilo",),
            manifests=ManifestPlan(
                sources=(source_spec_from_catalogue(contrib),),
                selectors=(parse_selector("contrib:kilo"),),
            ),
            cache_root=str(tmp_path),
            remote_ttl=3600,
        )
        config = SimpleNamespace(shims=shims, proxy=None)

        payload = catalogue_payload(config, None, http_get=http)

        assert [source["id"] for source in payload["sources"]] == [
            "contrib",
            "argonne",
        ]

    def test_a_mirror_for_a_default_is_not_duplicated(self, tmp_path):
        """A spec that names the source is that source, whatever its URL."""
        from llm_rosetta_gateway.shims.catalog import get_shim_source

        contrib = get_shim_source("contrib")
        argonne = get_shim_source("argonne")
        assert contrib is not None and argonne is not None
        mirror = "https://mirror.invalid/contrib.json"
        http = FakeHttp(
            {
                mirror: _manifest("contrib", {"kilo": [_record("0.1.0")]}),
                argonne.manifest_url: _manifest(
                    "argonne", {"argo": [_record("0.1.0")]}
                ),
            }
        )
        shims = SimpleNamespace(
            sources=(),
            install=("contrib:kilo",),
            manifests=ManifestPlan(
                sources=(ManifestSourceSpec(key="contrib", url=mirror),),
                selectors=(parse_selector("contrib:kilo"),),
            ),
            cache_root=str(tmp_path),
            remote_ttl=3600,
        )
        config = SimpleNamespace(shims=shims, proxy=None)

        payload = catalogue_payload(config, None, http_get=http)

        assert [source["id"] for source in payload["sources"]] == [
            "contrib",
            "argonne",
        ]
        assert [source["url"] for source in payload["sources"]] == [
            mirror,
            argonne.manifest_url,
        ]
