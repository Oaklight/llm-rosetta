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
from llm_rosetta_gateway.shim_sources import (
    apply_shim_section,
    apply_shim_sources,
    resolve_disabled,
    resolve_source_config,
    shims_signature,
)
from llm_rosetta_gateway.shims.sources import check_remote_updates
from llm_rosetta.shims.provider_shim import _reset_registry


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset the shim registry before and after each test.

    Tests here start from *no* shims so ``apply_shim_sources`` is what puts
    shims in the registry.  The built-ins are restored on teardown: leaving the
    registry empty would break every later test in the process that resolves a
    provider through a built-in shim.
    """
    from llm_rosetta.shims.providers import load_providers

    _reset_registry()
    yield
    _reset_registry()
    load_providers()


def _write_provider(root: Path, slug: str, name: str) -> None:
    directory = root / slug
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "provider.yaml").write_text(
        f"name: {name}\n"
        "base: chat_completions\n"
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
        dirs, remotes, cache = resolve_source_config(
            {"paths": ["./shims", "../shared"]}, cfg
        )
        assert dirs == [
            os.path.join(str(tmp_path), "./shims"),
            os.path.join(str(tmp_path), "../shared"),
        ]
        assert remotes == []
        assert cache == os.path.join(str(tmp_path), "shims-cache")

    def test_absolute_path_preserved(self, tmp_path: Path):
        cfg = str(tmp_path / "config.jsonc")
        dirs, _, _ = resolve_source_config({"paths": ["/abs/shims"]}, cfg)
        assert dirs == ["/abs/shims"]

    def test_cache_dir_override_anchored(self, tmp_path: Path):
        cfg = str(tmp_path / "config.jsonc")
        _, _, cache = resolve_source_config({"cache_dir": "./cc"}, cfg)
        assert cache == os.path.join(str(tmp_path), "./cc")

    def test_no_config_path_uses_home_cache(self):
        _, _, cache = resolve_source_config({"paths": ["x"]}, None)
        assert cache == str(Path("~/.cache/llm-rosetta/shims").expanduser())

    def test_remotes_copied(self, tmp_path: Path):
        spec = {"url": "https://github.com/a/b"}
        _, remotes, _ = resolve_source_config(
            {"remotes": [spec]}, str(tmp_path / "c.jsonc")
        )
        assert remotes == [spec]
        assert remotes[0] is not spec  # defensive copy

    @pytest.mark.parametrize(
        "section",
        [
            {"paths": "not-a-list"},
            {"paths": [123]},
            {"remotes": "nope"},
            {"remotes": ["not-a-dict"]},
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
        assert config.shims == {"paths": ["./shims"]}

    def test_dropping_the_section_drops_the_shims(self, tmp_path: Path):
        shims_dir = tmp_path / "shims"
        _write_provider(shims_dir, "acme", "acme--chat_completions")
        cfg_path = str(tmp_path / "config.jsonc")
        apply_shim_sources({"shims": {"paths": ["./shims"]}}, cfg_path)

        report = apply_shim_sources({"shims": {}}, cfg_path)
        assert report.sources == []
        from llm_rosetta.shims import get_shim

        assert get_shim("acme--chat_completions") is None

    def test_force_refresh_marks_remotes(self, tmp_path: Path, monkeypatch):
        seen: list[dict] = []

        def fake_rebuild(dirs, remotes, **kwargs):
            seen.extend(remotes)
            return SimpleNamespace(
                sources=[], total_shims=0, failures=lambda: [], to_dict=lambda: {}
            )

        import llm_rosetta_gateway.shim_sources as mod

        monkeypatch.setattr(mod, "rebuild_shims", fake_rebuild)
        apply_shim_section(
            {"remotes": [{"url": "https://github.com/a/b", "refresh": "never"}]},
            str(tmp_path / "c.jsonc"),
            force_refresh=True,
        )

        assert seen and seen[0]["refresh"] == "always"


class TestMergeShimSourcesBody:
    """The PUT handler's merge, in isolation from the request plumbing."""

    def test_presets_are_persisted(self):
        from llm_rosetta_gateway.admin.routes.config import _merge_shim_sources_body

        assert _merge_shim_sources_body({}, {"presets": ["argo"]}) == {
            "presets": ["argo"]
        }

    def test_absent_presets_leave_the_section_alone(self):
        from llm_rosetta_gateway.admin.routes.config import _merge_shim_sources_body

        section = {"presets": ["argo"]}
        assert _merge_shim_sources_body(section, {"paths": []}) == {"presets": ["argo"]}

    def test_empty_presets_are_dropped(self):
        from llm_rosetta_gateway.admin.routes.config import _merge_shim_sources_body

        assert _merge_shim_sources_body({"presets": ["argo"]}, {"presets": []}) == {}

    def test_other_keys_survive_a_presets_update(self):
        from llm_rosetta_gateway.admin.routes.config import _merge_shim_sources_body

        merged = _merge_shim_sources_body(
            {"paths": ["/a"], "strict": True}, {"presets": ["alcf"]}
        )
        assert merged == {"paths": ["/a"], "strict": True, "presets": ["alcf"]}


class TestCheckRemoteUpdates:
    """``check_remote_updates`` answers "is the pinned commit still HEAD?"."""

    SHA = "a" * 40
    OTHER = "b" * 40

    @staticmethod
    def _report(*statuses):
        from llm_rosetta_gateway.shims.sources import ShimLoadReport

        return ShimLoadReport(sources=list(statuses))

    @staticmethod
    def _remote(location: str, resolved_ref: str | None = None, ref: str | None = None):
        from llm_rosetta_gateway.shims.sources import ShimSourceStatus

        return ShimSourceStatus(
            kind="remote", location=location, resolved_ref=resolved_ref, ref=ref
        )

    @staticmethod
    def _get(sha: str):
        import json as _json

        def get(url, **_kwargs):
            return _json.dumps({"sha": sha}).encode("utf-8")

        return get

    def test_local_sources_are_skipped(self):
        from llm_rosetta_gateway.shims.sources import ShimSourceStatus

        report = self._report(ShimSourceStatus(kind="path", location="/tmp/shims"))
        assert check_remote_updates(report, http_get=self._get(self.SHA)) == []

    def test_head_matches_pin(self):
        report = self._report(self._remote("https://github.com/o/r", self.SHA))
        [entry] = check_remote_updates(report, http_get=self._get(self.SHA))
        assert entry["head"] == self.SHA
        assert entry["update"] is False
        assert entry["error"] is None

    def test_head_moved(self):
        report = self._report(self._remote("https://github.com/o/r", self.OTHER))
        [entry] = check_remote_updates(report, http_get=self._get(self.SHA))
        assert entry["current"] == self.OTHER
        assert entry["head"] == self.SHA
        assert entry["update"] is True

    def test_non_github_source_reports_an_error(self):
        report = self._report(
            self._remote("https://example.com/shims.tar.gz", self.SHA)
        )
        [entry] = check_remote_updates(report, http_get=self._get(self.SHA))
        assert entry["error"] == "not a GitHub repository"
        assert entry["head"] is None
        assert entry["update"] is False

    def test_a_failed_source_reports_its_error_not_a_verdict(self):
        from llm_rosetta_gateway.shims.sources import ShimSourceStatus

        report = self._report(
            ShimSourceStatus(
                kind="remote",
                location="https://github.com/o/r",
                ok=False,
                error="fetch failed: 404",
            )
        )
        [entry] = check_remote_updates(report, http_get=self._get(self.SHA))
        assert entry["error"] == "fetch failed: 404"
        assert entry["update"] is False
        assert entry["head"] is None

    def test_a_branch_pin_is_not_a_commit(self):
        report = self._report(self._remote("https://github.com/o/r", "main"))
        [entry] = check_remote_updates(report, http_get=self._get(self.SHA))
        assert "not a commit" in entry["error"]
        assert entry["update"] is False
        assert entry["head"] is None

    def test_the_check_asks_about_the_declared_ref(self):
        """A branch-pinned source is checked against that branch, not HEAD."""
        seen: list[str] = []

        def get(url, **_kwargs):
            seen.append(url)
            return json.dumps({"sha": self.SHA}).encode()

        report = self._report(
            self._remote("https://github.com/o/r", "b" * 40, ref="main")
        )
        [entry] = check_remote_updates(report, http_get=get)
        assert seen == ["https://api.github.com/repos/o/r/commits/main"]
        assert entry["update"] is True

    def test_the_proxy_reaches_the_head_lookup(self):
        seen = {}

        def get(url, **kwargs):
            seen["proxy"] = kwargs.get("proxy")
            return b'{"sha": "' + self.SHA.encode() + b'"}'

        report = self._report(self._remote("https://github.com/o/r", self.SHA))
        check_remote_updates(report, http_get=get, proxy="http://proxy:1")
        assert seen["proxy"] == "http://proxy:1"

    def test_resolution_failure_is_captured_per_source(self):
        def boom(url, **_kwargs):
            raise OSError("network down")

        report = self._report(
            self._remote("https://github.com/o/r", self.SHA),
            self._remote("https://github.com/o/s", self.SHA),
        )
        entries = check_remote_updates(report, http_get=boom)
        assert [e["error"] for e in entries] == ["network down", "network down"]
        assert all(e["update"] is False for e in entries)


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
        from llm_rosetta_gateway.shims.sources import ShimLoadReport

        return ShimLoadReport(sources=list(statuses))

    @staticmethod
    def _remote(location: str, resolved_ref: str):
        from llm_rosetta_gateway.shims.sources import ShimSourceStatus

        return ShimSourceStatus(
            kind="remote", location=location, resolved_ref=resolved_ref
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

        monkeypatch.setattr(shim_sources, "check_remote_updates", fake)
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
        assert _validate_shim_sources_body({"paths": ["a"], "remotes": []}, cfg) is None

    @pytest.mark.parametrize(
        "body",
        [
            {"paths": "x"},
            {"paths": [123]},  # element type — resolver rejects, not the route
            {"remotes": "x"},
            {"remotes": [{"ref": "v1"}]},
            {"remotes": ["nope"]},
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
