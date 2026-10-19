"""Gateway-side tests for the shim-removal advisory (epic #927).

The advisory is the "jiebo" between the built-in vendor shims and the standalone
packages: it tells an operator which provider is about to stop resolving, and
offers the source that keeps it working.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from llm_rosetta.gateway.admin.routes.config import (
    _shim_removal_advisories,
    _validate_shim_sources_body,
)
from llm_rosetta.gateway.config import GatewayConfig
from llm_rosetta.gateway.shim_sources import (
    report_shim_removals,
    shim_removal_advisories,
)
from llm_rosetta.shims import SHIM_SOURCE_PACKAGES
from llm_rosetta.shims.sources import ShimLoadReport, ShimSourceStatus

ARGO_SHIM = "argo--messages"


class _RecordCollector(logging.Handler):
    """Collect the gateway logger's records directly.

    ``caplog`` cannot see them: the gateway owns its handlers and stops the
    records propagating to the root logger.
    """

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def gateway_warnings():
    """The WARNING records the gateway logger emitted during the test."""
    logger = logging.getLogger("llm-rosetta-gateway")
    collector = _RecordCollector()
    logger.addHandler(collector)
    previous_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        yield collector.records
    finally:
        logger.removeHandler(collector)
        logger.setLevel(previous_level)


def _report(*statuses: ShimSourceStatus) -> ShimLoadReport:
    return ShimLoadReport(sources=list(statuses), total_shims=0, rebuilt_at=0.0)


def _loaded(*names: str) -> ShimSourceStatus:
    return ShimSourceStatus(
        kind="remote",
        location="https://github.com/Oaklight/llm-rosetta-shim-argo",
        name="argo",
        ok=True,
        shim_count=len(names),
        shim_names=tuple(names),
    )


class TestAdvisoryWithSources:
    """A declared source of the same name takes the shim off the list."""

    def test_a_loaded_source_silences_its_own_shim(self):
        providers = {
            "argo": {"shim": ARGO_SHIM},
            "argo-chat": {"shim": "argo--chat_completions"},
        }
        advisories = shim_removal_advisories(providers, _report(_loaded(ARGO_SHIM)))
        assert [a.shim for a in advisories] == ["argo--chat_completions"]

    def test_a_failed_source_silences_nothing(self):
        """It overrode nothing, so the provider is still on the built-in."""
        providers = {"argo": {"shim": ARGO_SHIM}}
        failed = ShimSourceStatus(kind="remote", location="x", ok=False, error="boom")
        assert len(shim_removal_advisories(providers, _report(failed))) == 1

    def test_no_report_at_all_still_advises(self):
        assert len(shim_removal_advisories({"argo": {"shim": ARGO_SHIM}}, None)) == 1


class TestReporting:
    """What the operator actually sees at startup."""

    def test_one_warning_per_advisory(self, gateway_warnings):
        providers = {
            "argo": {"shim": ARGO_SHIM},
            "argo-chat": {"shim": "argo--chat_completions"},
        }
        advisories = report_shim_removals(providers, None)
        assert len(advisories) == 2
        assert len(gateway_warnings) == 2
        messages = " ".join(record.getMessage() for record in gateway_warnings)
        assert ARGO_SHIM in messages
        assert "0.16.0" in messages
        # The message has to be actionable: it names the source to declare.
        assert "llm-rosetta-shim-argo" in messages

    def test_a_clean_config_is_silent(self, gateway_warnings):
        assert report_shim_removals({"o": {"shim": "openai--chat_completions"}}) == []
        assert gateway_warnings == []


class TestGatewayConfigView:
    """``raw_providers`` is the unfiltered, corrected view the advisory needs."""

    def test_a_disabled_provider_is_still_visible(self):
        config = GatewayConfig(
            {
                "providers": {
                    "on": {"shim": "argo--messages", "api_key": "k"},
                    "off": {"shim": ARGO_SHIM, "api_key": "k", "enabled": False},
                },
                "models": {"m": "on"},
            }
        )
        assert set(config.raw_providers) == {"on", "off"}
        # ...while the routing view still hides it.
        assert "off" not in config._raw_providers

    def test_legacy_names_arrive_corrected(self):
        config = GatewayConfig(
            {
                "providers": {"p": {"shim": "argo--anthropic", "api_key": "k"}},
                "models": {"m": "p"},
            }
        )
        assert config.raw_providers["p"]["shim"] == ARGO_SHIM
        advisories = shim_removal_advisories(config.raw_providers, None)
        assert [a.shim for a in advisories] == [ARGO_SHIM]


class TestAdminPayload:
    """The panel's two keys, and the preset round-trip it drives."""

    @staticmethod
    def _request(report: ShimLoadReport | None = None):
        return SimpleNamespace(app=SimpleNamespace(shim_source_report=report))

    def test_advisories_are_derived_from_the_raw_config(self):
        raw = {"providers": {"argo": {"shim": ARGO_SHIM}}}
        payload = _shim_removal_advisories(self._request(), raw)
        assert [entry["provider"] for entry in payload] == ["argo"]
        assert payload[0]["removed_in"] == "0.16.0"

    def test_a_declared_source_clears_the_advisory_on_the_next_read(self):
        raw = {"providers": {"argo": {"shim": ARGO_SHIM}}}
        assert _shim_removal_advisories(self._request(), raw) != []
        request = self._request(_report(_loaded(ARGO_SHIM)))
        assert _shim_removal_advisories(request, raw) == []

    def test_every_preset_is_an_acceptable_remotes_entry(self, tmp_path):
        """What the panel appends is exactly what the PUT endpoint validates."""
        config_path = str(tmp_path / "config.jsonc")
        for entry in SHIM_SOURCE_PACKAGES:
            body = {"paths": [], "remotes": [dict(entry.remote)]}
            assert _validate_shim_sources_body(body, config_path) is None, entry.key

    def test_preset_payload_shape(self):
        """The shape the Settings picker consumes is pinned here."""
        from llm_rosetta.gateway.admin.routes import config as routes_config

        payload = routes_config._shim_source_presets_payload()
        assert [entry["key"] for entry in payload] == ["alcf", "argo", "asksage"]
        for entry in payload:
            assert set(entry) == {
                "key",
                "label",
                "source_repo",
                "remote",
                "shims",
            }
            assert set(entry["remote"]) == {"url", "path", "token_env"}
            assert entry["remote"]["token_env"] == "GITHUB_TOKEN"
            assert entry["shims"]

    def test_presets_come_from_the_source_catalog(self):
        """The panel's catalogue is the standalone-package list, not the manifest."""
        from llm_rosetta.gateway.admin.routes import config as routes_config

        assert routes_config.SHIM_SOURCE_PACKAGES is SHIM_SOURCE_PACKAGES


@pytest.mark.parametrize("package", SHIM_SOURCE_PACKAGES, ids=lambda p: p.key)
def test_every_preset_points_at_its_own_source(package):
    assert package.remote["url"] == package.repo


class TestStrandedConfig:
    """A config that still names a shim that left core is told where it went."""

    @pytest.mark.parametrize(
        ("shim", "repo"),
        [
            ("alcf_metis--chat_completions", "llm-rosetta-shim-alcf"),
            ("asksage--messages", "llm-rosetta-shim-asksage"),
        ],
    )
    def test_the_error_names_the_package_and_the_source(self, shim, repo):
        with pytest.raises(ValueError) as excinfo:
            GatewayConfig(
                {
                    "providers": {"p": {"shim": shim, "api_key": "k"}},
                    "models": {"m": "p"},
                }
            )

        message = str(excinfo.value)
        assert repo in message
        assert "shims.remotes" in message
        # The preset is only usable if the private-repo token variable is named.
        assert "token_env" in message
        assert "GITHUB_TOKEN" in message

    def test_an_unrelated_unknown_name_keeps_the_generic_message(self):
        with pytest.raises(ValueError) as excinfo:
            GatewayConfig(
                {
                    "providers": {
                        "p": {"shim": "zzz_unknown--chat_completions", "api_key": "k"}
                    },
                    "models": {"m": "p"},
                }
            )

        assert "llm-rosetta-shim" not in str(excinfo.value)


class TestCreateAppWiring:
    """The advisory reaches an operator's log only through ``create_app``.

    ``report_shim_removals`` is covered directly above; what was not covered is
    that anything calls it.  Without this, deleting the call site leaves the
    whole suite green while the feature goes silent.
    """

    @staticmethod
    def _raw() -> dict:
        return {
            "providers": {"argo": {"shim": ARGO_SHIM, "api_key": "sk-test"}},
            "models": {},
            "server": {},
        }

    @staticmethod
    def _advisories(records: list[logging.LogRecord]) -> list[str]:
        return [
            record.getMessage()
            for record in records
            if ARGO_SHIM in record.getMessage() and "0.16.0" in record.getMessage()
        ]

    @pytest.mark.asyncio
    async def test_starting_the_gateway_warns_about_a_scheduled_removal(
        self, gateway_warnings
    ):
        from llm_rosetta.gateway.app import create_app

        await create_app(GatewayConfig(self._raw()))

        assert self._advisories(gateway_warnings)

    @pytest.mark.asyncio
    async def test_a_declared_source_silences_the_warning_at_the_app_level(
        self, tmp_path, gateway_warnings
    ):
        """The whole point of the advisory: a quiet log means the migration landed."""
        from llm_rosetta.gateway.app import create_app
        from llm_rosetta.shims import rebuild_shims

        leaf = tmp_path / "local-shims" / "messages"
        leaf.mkdir(parents=True)
        (leaf / "provider.yaml").write_text(
            f"name: {ARGO_SHIM}\n"
            "base: messages\n"
            "default_base_url: https://example.invalid/v1\n"
        )
        raw = self._raw()
        raw["shims"] = {"paths": [str(tmp_path / "local-shims")]}

        try:
            await create_app(GatewayConfig(raw))
            assert self._advisories(gateway_warnings) == []
        finally:
            # The declared directory replaced the built-in in the process-wide
            # registry; put the shipped set back for the rest of the session.
            rebuild_shims()

    @pytest.mark.asyncio
    async def test_a_disabled_provider_is_still_reported(self, gateway_warnings):
        """Its config breaks the moment it is switched back on.

        The mutation this catches is ``raw_providers`` -> the enabled-only view:
        the warning for the disabled provider disappears and nothing else moves.
        """
        from llm_rosetta.gateway.app import create_app

        raw = {
            "providers": {
                "argo": {"shim": ARGO_SHIM, "api_key": "sk-test"},
                "argo-chat": {
                    "shim": "argo--chat_completions",
                    "api_key": "sk-test",
                    "enabled": False,
                },
            },
            "models": {},
            "server": {},
        }

        await create_app(GatewayConfig(raw))

        messages = [record.getMessage() for record in gateway_warnings]
        assert any("argo--chat_completions" in message for message in messages)
        assert self._advisories(gateway_warnings)
