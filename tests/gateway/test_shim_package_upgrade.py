"""Per-package upgrade: the pin-versus-head guard lives on the server.

The panel only decides *when* to offer the action.  A package whose pin equals
its repository head must be a no-op even when the request arrives anyway — a
stale page, or a hand-built call — so every test here drives the route with a
verdict that contradicts what the button would have shown.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from llm_rosetta_gateway.admin.routes.config import shims as shims_module
from llm_rosetta_gateway.shims.sources import ShimLoadReport, ShimSourceStatus

ARGO = "https://github.com/Oaklight/llm-rosetta-shim-argo"
PIN = "a" * 40
HEAD = "b" * 40


def _report(location: str = ARGO, pin: str = PIN) -> ShimLoadReport:
    return ShimLoadReport(
        sources=[ShimSourceStatus(kind="remote", location=location, resolved_ref=pin)]
    )


def _request(report, *, config=None, cache=None) -> SimpleNamespace:
    app = SimpleNamespace(
        shim_source_report=report,
        gateway_config=config or SimpleNamespace(proxy=None),
        shim_update_check=cache,
        provider_validation_warnings=[],
    )
    return SimpleNamespace(app=app, path_params={})


@pytest.fixture()
def verdicts(monkeypatch):
    """Answer the head lookup from a dict the test fills in."""
    answers: dict[str, dict] = {}

    def fake(report, *, http_get=None, proxy=None):
        return [
            answers.get(
                source.location,
                {
                    "location": source.location,
                    "current": source.resolved_ref,
                    "head": source.resolved_ref,
                    "update": False,
                    "error": None,
                },
            )
            for source in report.sources
        ]

    # The handler imports the checker at call time, so the patch belongs on
    # the module it imports from.
    from llm_rosetta_gateway.shims import sources as shim_sources

    monkeypatch.setattr(shim_sources, "check_remote_updates", fake)
    return answers


@pytest.fixture()
def refetches(monkeypatch):
    """Count refetches and reloads; neither may run when there is nothing to do."""
    calls = {"refetch": 0, "reload": 0}

    def fake_apply(*_a, **_k):
        calls["refetch"] += 1
        return _report()

    async def fake_reload(*_a, **_k):
        calls["reload"] += 1
        return SimpleNamespace(providers={})

    monkeypatch.setattr(shims_module, "apply_shim_sources", fake_apply)
    monkeypatch.setattr(shims_module, "_reload_gateway_config", fake_reload)
    monkeypatch.setattr(shims_module, "_install_shim_report", lambda *a, **k: None)
    monkeypatch.setattr(
        shims_module,
        "_get_config_io",
        lambda _r: SimpleNamespace(load_raw=lambda _p: {"shims": {}}),
    )
    monkeypatch.setattr(shims_module, "_get_config_path", lambda _r: "/tmp/c.jsonc")
    return calls


def _call(request, key: str = "argo"):
    from llm_rosetta_gateway.admin.routes.config.shims import (
        post_shim_package_upgrade,
    )

    response = asyncio.run(post_shim_package_upgrade(request, key))
    return response.status_code, json.loads(response.body)


class TestTheGuard:
    def test_an_unchanged_pin_changes_nothing(self, verdicts, refetches):
        """The measured case: pin == head is not an upgrade, whatever was sent."""
        status, body = _call(_request(_report()))

        assert status == 200
        assert body["changed"] is False and body["ok"] is True
        assert body["pin"] == PIN and body["head"] == PIN
        assert refetches == {"refetch": 0, "reload": 0}

    def test_a_stale_cache_does_not_survive_the_recheck(self, verdicts, refetches):
        """A page that cached "update available" cannot force a refetch.

        The verdict is recomputed here, so the guard does not depend on what
        the panel believes; the cached answer is not read at all.
        """
        cache = {
            "sources": [
                {
                    "location": ARGO,
                    "current": PIN,
                    "head": HEAD,
                    "update": True,
                    "error": None,
                }
            ],
            "checked_at": 1.0,
        }
        status, body = _call(_request(_report(), cache=cache))

        assert status == 200 and body["changed"] is False
        assert refetches == {"refetch": 0, "reload": 0}

    def test_a_moved_head_refetches(self, verdicts, refetches):
        verdicts[ARGO] = {
            "location": ARGO,
            "current": PIN,
            "head": HEAD,
            "update": True,
            "error": None,
        }
        request = _request(_report(), cache={"sources": [], "checked_at": 1.0})

        status, body = _call(request)

        assert status == 200 and body["changed"] is True
        assert refetches == {"refetch": 1, "reload": 1}
        # The old verdict is about the previous pin; the next read must ask.
        assert request.app.shim_update_check is None

    def test_an_unreadable_head_is_a_refusal(self, verdicts, refetches):
        verdicts[ARGO] = {
            "location": ARGO,
            "current": PIN,
            "head": None,
            "update": False,
            "error": "API rate limit exceeded",
        }

        status, body = _call(_request(_report()))

        assert status == 409
        assert "rate limit" in body["error"]
        assert refetches == {"refetch": 0, "reload": 0}

    def test_a_package_that_is_not_declared_is_a_refusal(self, verdicts, refetches):
        status, body = _call(_request(_report(location="https://example.test/x")))

        assert status == 409
        assert "not loaded" in body["error"]
        assert refetches == {"refetch": 0, "reload": 0}

    def test_an_unknown_key_is_a_404(self):
        status, _body = _call(_request(None), key="no-such-package")

        assert status == 404


class TestTheCatalogueCarriesThePair:
    def _catalogue(self, request):
        from llm_rosetta_gateway.admin.routes.config.shim_files import get_shim_catalog

        response = asyncio.run(get_shim_catalog(request))
        return {p["key"]: p for p in json.loads(response.body)["packages"]}

    def _request(self, report, cache):
        app = SimpleNamespace(
            shim_source_report=report,
            shim_update_check=cache,
            gateway_config=SimpleNamespace(provider_shim_names={}),
        )
        return SimpleNamespace(app=app, path_params={}, json=lambda: None)

    def test_pin_and_head_travel_with_the_package(self):
        """The card draws both refs from one payload, so they cannot disagree."""
        cache = {
            "sources": [
                {
                    "location": ARGO,
                    "current": PIN,
                    "head": HEAD,
                    "update": True,
                    "error": None,
                }
            ],
            "checked_at": 1.0,
        }

        argo = self._catalogue(self._request(_report(), cache))["argo"]

        assert argo["pin"] == PIN
        assert argo["head"] == HEAD
        assert argo["update_available"] is True
        assert argo["check_error"] is None

    def test_a_package_with_no_verdict_claims_nothing(self):
        argo = self._catalogue(self._request(_report(), None))["argo"]

        assert argo["pin"] == PIN
        assert argo["head"] is None
        assert argo["update_available"] is False

    def test_an_uninstalled_package_has_no_pin(self):
        alcf = self._catalogue(self._request(_report(), None))["alcf"]

        assert alcf["pin"] is None
        assert alcf["head"] is None


class TestRepoKey:
    def test_two_spellings_of_one_repository_compare_equal(self):
        from llm_rosetta_gateway.shims.catalog import repo_key

        assert repo_key("https://github.com/Oaklight/x.git/") == repo_key(
            "https://github.com/Oaklight/x"
        )
        assert repo_key(None) == ""
