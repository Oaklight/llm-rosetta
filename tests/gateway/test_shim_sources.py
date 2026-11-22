"""Tests for the gateway <-> shim-source bridge (config resolution + wiring)."""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace
from pathlib import Path

import pytest

from llm_rosetta_gateway.admin.routes.config import (
    _shim_status_payload,
    _validate_shim_sources_body,
)
from llm_rosetta_gateway.config import GatewayConfig
from llm_rosetta_gateway.shims.shim_sources import (
    apply_shim_section,
    apply_shim_sources,
    describe_shim_sources,
    resolve_disabled,
    resolve_source_config,
    shims_signature,
)
from llm_rosetta_gateway.shims._types import ShimLoadReport, ShimSourceStatus
from llm_rosetta_gateway.shims.sources import check_source_updates
from llm_rosetta.api import restore_registry, snapshot_registry, staging_registry


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset the shim registry before and after each test.

    Tests here start from *no* shims so ``apply_shim_sources`` is what puts
    shims in the registry.  The built-ins are restored on teardown: leaving the
    registry empty would break every later test in the process that resolves a
    provider through a built-in shim.
    """
    snapshot = snapshot_registry()
    with staging_registry():
        pass
    yield
    restore_registry(snapshot)


def _write_provider(root: Path, slug: str, name: str) -> None:
    directory = root / slug
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "provider.yaml").write_text(
        f"name: {name}\n"
        "format: chat_completions\nmodel_type: llm\n"
        "connection:\n"
        f"  base_url: https://{slug}.example/v1\n"
        "  api_key_env: ACME_API_KEY\n"
    )


class TestShimsSignature:
    def test_empty(self):
        assert shims_signature({}) == ""
        assert shims_signature({"shims": {}}) == ""

    def test_key_order_insensitive(self):
        a = shims_signature({"shims": {"paths": ["x"], "strict": True}})
        b = shims_signature({"shims": {"strict": True, "paths": ["x"]}})
        assert a == b

    def test_changes_with_section(self):
        base = shims_signature({"shims": {"paths": ["x"]}})
        assert base != shims_signature({"shims": {"paths": ["y"]}})
        assert base != shims_signature({})


class TestResolveSourceConfig:
    def test_relative_paths_anchor_to_config_dir(self, tmp_path: Path):
        cfg = str(tmp_path / "config.jsonc")
        dirs, cache = resolve_source_config({"paths": ["./shims", "../shared"]}, cfg)
        assert dirs == [
            os.path.join(str(tmp_path), "./shims"),
            os.path.join(str(tmp_path), "../shared"),
        ]
        assert cache == os.path.join(str(tmp_path), "shims-cache")

    def test_absolute_path_preserved(self, tmp_path: Path):
        cfg = str(tmp_path / "config.jsonc")
        dirs, _ = resolve_source_config({"paths": ["/abs/shims"]}, cfg)
        assert dirs == ["/abs/shims"]

    def test_cache_dir_override_anchored(self, tmp_path: Path):
        cfg = str(tmp_path / "config.jsonc")
        _, cache = resolve_source_config({"cache_dir": "./cc"}, cfg)
        assert cache == os.path.join(str(tmp_path), "./cc")

    def test_no_config_path_uses_home_cache(self):
        _, cache = resolve_source_config({"paths": ["x"]}, None)
        assert cache == str(Path("~/.cache/llm-rosetta/shims").expanduser())

    @pytest.mark.parametrize(
        "section",
        [
            {"paths": "not-a-list"},
            {"paths": [123]},
            {"install": "nope"},
            {"sources": "nope"},
            {"cache_dir": 5},
        ],
    )
    def test_bad_shapes_raise(self, section, tmp_path: Path):
        with pytest.raises(ValueError):
            resolve_source_config(section, str(tmp_path / "c.jsonc"))


class TestApplyShimSources:
    def test_empty_section_only_builtins(self):
        report = apply_shim_sources({"server": {}}, None)
        assert report.sources == []
        assert report.total_shims > 0

    def test_provider_resolves_through_declared_source(self, tmp_path: Path):
        shims_dir = tmp_path / "shims"
        _write_provider(shims_dir, "acme", "acme--chat_completions")
        cfg_path = str(tmp_path / "config.jsonc")
        raw = {
            "providers": {"acme": {"shim": "acme--chat_completions", "api_key": "k"}},
            "models": {"m": "acme"},
            "server": {},
            "shims": {"paths": ["./shims"]},
        }

        report = apply_shim_sources(raw, cfg_path)
        assert report.sources[0].ok is True

        config = GatewayConfig(raw)
        assert config.provider_types["acme"] == "chat_completions"
        assert config.providers["acme"].base_url == "https://acme.example/v1"
        assert config.shims.paths == ("./shims",)
        assert config.shims.declared is True

    def test_dropping_the_section_drops_the_shims(self, tmp_path: Path):
        shims_dir = tmp_path / "shims"
        _write_provider(shims_dir, "acme", "acme--chat_completions")
        cfg_path = str(tmp_path / "config.jsonc")
        apply_shim_sources({"shims": {"paths": ["./shims"]}}, cfg_path)

        report = apply_shim_sources({"shims": {}}, cfg_path)
        assert report.sources == []
        from llm_rosetta.shims import get_shim

        assert get_shim("acme--chat_completions") is None

    def test_force_refresh_reaches_the_rebuild(self, tmp_path: Path, monkeypatch):
        seen: dict = {}

        def fake_rebuild(dirs, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(
                sources=[], total_shims=0, failures=lambda: [], to_dict=lambda: {}
            )

        import llm_rosetta_gateway.shims.shim_sources as mod

        monkeypatch.setattr(mod, "rebuild_shims", fake_rebuild)
        apply_shim_section({}, str(tmp_path / "c.jsonc"), force_refresh=True)

        assert seen["force_refresh"] is True


class TestMergeShimSourcesBody:
    """The PUT handler's merge, in isolation from the request plumbing."""

    def test_install_selectors_are_persisted(self):
        from llm_rosetta_gateway.admin.routes.config import _merge_shim_sources_body

        assert _merge_shim_sources_body({}, {"install": ["argonne:argo"]}) == {
            "install": ["argonne:argo"]
        }

    def test_absent_install_leaves_the_section_alone(self):
        from llm_rosetta_gateway.admin.routes.config import _merge_shim_sources_body

        section = {"install": ["argonne:argo"]}
        assert _merge_shim_sources_body(section, {"paths": []}) == {
            "install": ["argonne:argo"]
        }

    def test_empty_install_is_dropped(self):
        from llm_rosetta_gateway.admin.routes.config import _merge_shim_sources_body

        assert (
            _merge_shim_sources_body({"install": ["argonne:argo"]}, {"install": []})
            == {}
        )

    def test_other_keys_survive_an_install_update(self):
        from llm_rosetta_gateway.admin.routes.config import _merge_shim_sources_body

        merged = _merge_shim_sources_body(
            {"paths": ["/a"], "strict": True}, {"install": ["argonne:alcf"]}
        )
        assert merged == {"paths": ["/a"], "strict": True, "install": ["argonne:alcf"]}


class TestCheckSourceUpdates:
    """``check_source_updates`` reads installed manifest vendors only."""

    def test_local_sources_are_skipped(self):
        report = ShimLoadReport(
            sources=[ShimSourceStatus(kind="path", location="/tmp/shims")]
        )
        assert check_source_updates(report, http_get=lambda *a, **k: b"{}") == []

    def test_a_failed_vendor_is_not_asked_about(self):
        report = ShimLoadReport(
            sources=[
                ShimSourceStatus(
                    kind="remote",
                    location="https://example.test/manifest.json",
                    vendor="argo",
                    ok=False,
                    error="fetch failed: 404",
                )
            ]
        )

        def refuse(*args, **kwargs):
            raise AssertionError("a failed vendor must not be looked up")

        assert check_source_updates(report, http_get=refuse) == []


class TestReportInstallation:
    """A new report must not inherit the previous report's verdicts."""

    def test_installing_a_report_drops_the_cached_verdicts(self):
        from llm_rosetta_gateway.admin.routes._shared import _install_shim_report

        app = SimpleNamespace(
            shim_source_report="old-report",
            shim_source_signature="old-signature",
            shim_update_check={"sources": [{"location": "x"}], "checked_at": 1.0},
        )
        _install_shim_report(SimpleNamespace(app=app), "new-report", "sig")

        assert app.shim_source_report == "new-report"
        assert app.shim_source_signature == "sig"
        assert app.shim_update_check is None


class TestRetiredKeys:
    """A key 0.15.0 removed stops the section; the rest is not applied."""

    def test_a_retired_key_is_refused_with_its_replacement(self, tmp_path: Path):
        with pytest.raises(ValueError, match=r"'shims\.presets'.*0\.15\.0"):
            apply_shim_section(
                {"presets": ["alcf"], "paths": [str(tmp_path / "none")]},
                str(tmp_path / "config.jsonc"),
            )

    def test_the_message_names_the_install_replacement(self, tmp_path: Path):
        with pytest.raises(ValueError, match="shims.install"):
            apply_shim_section({"presets": ["alcf"]}, str(tmp_path / "config.jsonc"))


class TestReloadWiring:
    """The reload paths must install the report through the helper.

    Asserted through the app state, not the call: replacing the helper with
    the two assignments it replaced is exactly the defect the helper exists to
    prevent, and it must red here.
    """

    def test_the_reload_endpoint_drops_the_cached_verdicts(self, monkeypatch):
        from llm_rosetta_gateway.admin.routes.config import shims as route

        app = SimpleNamespace(
            config_path="/tmp/config.jsonc",
            shim_source_report="old-report",
            shim_source_signature="old-signature",
            shim_update_check={"sources": [], "checked_at": 1.0},
            config_io=SimpleNamespace(load_raw=lambda path: {"shims": {}}),
        )
        request = SimpleNamespace(app=app, json=lambda: {"refresh": "always"})
        new_report = SimpleNamespace(to_dict=lambda: {"sources": []})

        monkeypatch.setattr(route, "apply_shim_sources", lambda *a, **k: new_report)
        monkeypatch.setattr(route, "shims_signature", lambda raw: "new-signature")

        async def fake_reload(request, path, *, force_shims=False):
            return SimpleNamespace(providers={})

        monkeypatch.setattr(route, "_reload_gateway_config", fake_reload)
        response = asyncio.run(route.post_shims_reload(request))

        assert response.status_code == 200
        assert app.shim_source_report is new_report
        assert app.shim_source_signature == "new-signature"
        assert app.shim_update_check is None

    def test_the_config_reload_drops_the_cached_verdicts(self, monkeypatch):
        from llm_rosetta_gateway.admin.routes import _shared
        from llm_rosetta_gateway import ops as ops_module
        from llm_rosetta_gateway.shims import shim_sources as shim_sources_module

        app = SimpleNamespace(
            shim_source_report="old-report",
            shim_source_signature="old-signature",
            shim_update_check={"sources": [], "checked_at": 1.0},
            config_io=SimpleNamespace(
                load=lambda path: {"shims": {"install": ["argonne:alcf"]}}
            ),
            ops_ctx=None,
            provider_validation_warnings=[],
        )
        new_report = object()
        monkeypatch.setattr(
            shim_sources_module, "apply_shim_sources", lambda *a, **k: new_report
        )

        class _Validation:
            def __init__(self, *a, **k):
                pass

            async def execute(self):
                return {"disabled": []}

        monkeypatch.setattr(ops_module, "OpsProviderValidation", _Validation)
        monkeypatch.setattr(
            _shared, "apply_admin_theme", lambda *a, **k: None, raising=False
        )
        monkeypatch.setattr(
            _shared, "_sync_auth_middleware", lambda *a, **k: None, raising=False
        )

        response = asyncio.run(
            _shared._reload_gateway_config(
                SimpleNamespace(app=app), "/tmp/config.jsonc"
            )
        )
        assert response is app.gateway_config
        assert app.shim_source_report is new_report
        assert app.shim_update_check is None


class TestCheckUpdatesRoute:
    """The endpoint decides *whether* the check runs; that is what it is tested for."""

    @staticmethod
    def _request(*, report=None, config=None, body=None):
        app = SimpleNamespace(
            shim_source_report=report,
            gateway_config=config,
            shim_update_check=None,
        )
        return SimpleNamespace(app=app, json=lambda: body if body is not None else {})

    @staticmethod
    def _report(*statuses):
        return ShimLoadReport(sources=list(statuses))

    @staticmethod
    def _remote(location: str, resolved_ref: str):
        return ShimSourceStatus(
            kind="remote",
            location=location,
            resolved_ref=resolved_ref,
            vendor="argo",
            ok=True,
        )

    def _call(self, request):
        from llm_rosetta_gateway.admin.routes.config.shims import (
            post_shims_check_updates,
        )

        response = asyncio.run(post_shims_check_updates(request))
        return json.loads(response.body)

    @pytest.fixture()
    def counted(self, monkeypatch):
        calls: list[dict] = []

        def fake(report, *, http_get=None, proxy=None):
            calls.append({"proxy": proxy, "sources": len(report.sources)})
            return [
                {
                    "location": s.location,
                    "current": s.resolved_ref,
                    "head": s.resolved_ref,
                    "update": False,
                    "error": None,
                }
                for s in report.sources
            ]

        from llm_rosetta_gateway.shims import sources as shim_sources

        monkeypatch.setattr(shim_sources, "check_source_updates", fake)
        return calls

    def test_without_a_report_there_is_nothing_to_check(self):
        assert self._call(self._request()) == {"sources": [], "checked_at": 0.0}

    def test_a_second_call_is_served_from_the_cache(self, counted):
        request = self._request(
            report=self._report(self._remote("https://github.com/o/r", "a" * 40))
        )
        first = self._call(request)
        second = self._call(request)
        assert len(counted) == 1
        assert second["checked_at"] == first["checked_at"]

    def test_refresh_asks_again(self, counted):
        request = self._request(
            report=self._report(self._remote("https://github.com/o/r", "a" * 40))
        )
        self._call(request)
        request.json = lambda: {"refresh": True}
        refreshed = self._call(request)
        assert len(counted) == 2
        assert refreshed["checked_at"] >= 0.0

    def test_the_check_uses_the_gateways_proxy(self, counted):
        request = self._request(
            report=self._report(self._remote("https://github.com/o/r", "a" * 40)),
            config=SimpleNamespace(proxy="http://proxy:1"),
        )
        self._call(request)
        assert counted[0]["proxy"] == "http://proxy:1"


class TestAdminHelpers:
    def test_status_payload_without_report(self):
        request = SimpleNamespace(app=SimpleNamespace(shim_source_report=None))
        payload = _shim_status_payload(request)
        assert payload == {
            "sources": [],
            "total_shims": 0,
            "rebuilt_at": 0.0,
            "failure_count": 0,
        }

    def test_status_payload_with_report(self):
        report = SimpleNamespace(to_dict=lambda: {"sources": [{"ok": True}]})
        request = SimpleNamespace(app=SimpleNamespace(shim_source_report=report))
        assert _shim_status_payload(request) == {"sources": [{"ok": True}]}

    def test_validate_accepts_good_body(self, tmp_path: Path):
        cfg = str(tmp_path / "config.jsonc")
        assert _validate_shim_sources_body({"paths": ["a"], "install": []}, cfg) is None

    @pytest.mark.parametrize("key", ["presets", "include", "exclude", "remotes"])
    def test_validate_rejects_a_retired_key(self, key, tmp_path: Path):
        """The admin PUT refuses a key 0.15.0 removed, naming the release."""
        cfg = str(tmp_path / "config.jsonc")
        response = _validate_shim_sources_body({key: {"x": ["y"]}}, cfg)
        assert response is not None
        assert response.status_code == 400
        assert b"0.15.0" in response.body

    @pytest.mark.parametrize("key", ["presets", "include", "exclude", "remotes"])
    def test_validate_accepts_a_retired_key_that_holds_nothing(
        self, key, tmp_path: Path
    ):
        """A stale panel that sends an empty list is not refused for it."""
        cfg = str(tmp_path / "config.jsonc")
        assert _validate_shim_sources_body({key: []}, cfg) is None

    @pytest.mark.parametrize(
        "body",
        [
            {"paths": "x"},
            {"paths": [123]},  # element type — resolver rejects, not the route
            {"install": "x"},
            {"sources": "x"},
            {"cache_dir": 123},  # type — resolver rejects
            {"remote_ttl": "soon"},  # non-numeric TTL — resolver rejects
        ],
    )
    def test_validate_rejects_bad_body(self, body, tmp_path: Path):
        cfg = str(tmp_path / "config.jsonc")
        assert _validate_shim_sources_body(body, cfg) is not None


class TestDisabledShims:
    """``shims.disabled`` drops a name after its source registers it."""

    def test_empty_and_missing(self):
        assert resolve_disabled(None) == []
        assert resolve_disabled({}) == []
        assert resolve_disabled({"disabled": []}) == []

    def test_names_are_kept_in_order(self):
        assert resolve_disabled({"disabled": ["b", "a"]}) == ["b", "a"]

    @pytest.mark.parametrize("section", [{"disabled": "a"}, {"disabled": [1]}])
    def test_bad_shape_raises(self, section):
        with pytest.raises(ValueError, match="shims.disabled"):
            resolve_disabled(section)

    def test_disabled_shim_leaves_the_registry(self, tmp_path: Path):
        from llm_rosetta.shims import get_shim

        _write_provider(tmp_path, "acme", "acme--chat_completions")
        report = apply_shim_section(
            {"paths": [str(tmp_path)], "disabled": ["acme--chat_completions"]}
        )

        assert get_shim("acme--chat_completions") is None
        assert report.sources[0].shim_count == 0
        assert report.sources[0].shim_names == ()

    def test_removing_the_name_brings_the_shim_back(self, tmp_path: Path):
        from llm_rosetta.shims import get_shim

        _write_provider(tmp_path, "acme", "acme--chat_completions")
        apply_shim_section(
            {"paths": [str(tmp_path)], "disabled": ["acme--chat_completions"]}
        )
        apply_shim_section({"paths": [str(tmp_path)]})

        assert get_shim("acme--chat_completions") is not None

    def test_a_disabled_name_the_sources_do_not_have_is_harmless(self):
        report = apply_shim_section({"disabled": ["nope--chat_completions"]})

        assert report.failures() == []
        assert report.sources == []


class TestDescribeShimSources:
    """The startup banner names how each installed vendor was loaded."""

    def test_describe_names_the_leg(self):
        report = ShimLoadReport(
            sources=[
                ShimSourceStatus(
                    kind="remote",
                    location="https://example.test/manifest.json",
                    name="argonne:argo",
                    vendor="argo",
                    release={"version": "1.2.0"},
                    ok=True,
                    leg="primary",
                    from_cache=True,
                ),
                ShimSourceStatus(
                    kind="remote",
                    location="https://example.test/other.json",
                    name="contrib:llm7",
                    vendor="llm7",
                    ok=False,
                    error="ShimSourceUnreachable: nope",
                ),
            ]
        )
        assert describe_shim_sources(report) == [
            "argonne:argo 1.2.0: loaded via primary (cache)",
            "contrib:llm7: failed",
        ]

    def test_describe_leaves_local_paths_out(self):
        report = ShimLoadReport(
            sources=[ShimSourceStatus(kind="path", location="/tmp/shims", ok=True)]
        )
        assert describe_shim_sources(report) == []

    def test_describe_is_empty_without_a_section(self):
        assert describe_shim_sources(None) == []
        assert describe_shim_sources(ShimLoadReport()) == []
