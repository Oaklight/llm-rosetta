"""Gateway-side tests for the shim-removal advisory (epic #927).

The advisory is the bridge between the built-in vendor shims and the standalone
packages: it tells an operator which provider is about to stop resolving, and
offers the source that keeps it working.

No removals are scheduled in the shipped manifest — ALCF, AskSage and Argo all
ship standalone from 0.14.0 — so the advisory tests install a synthetic manifest
for a shim that is still built in.  The preset tests below cover the config path
an operator actually uses to keep ``argo--*`` working after the move.
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
    resolve_source_config,
    shim_removal_advisories,
)
from llm_rosetta.shims import SHIM_SOURCE_PACKAGES, ShimRemoval
from llm_rosetta.shims.sources import ShimLoadReport, ShimSourceStatus

ARGO_PACKAGE = next(p for p in SHIM_SOURCE_PACKAGES if p.key == "argo")
ALCF_PACKAGE = next(p for p in SHIM_SOURCE_PACKAGES if p.key == "alcf")

#: A registered built-in shim standing in for a future removal.  Using a real
#: still-shipped shim keeps ``GatewayConfig`` construction valid.
MINIMAX = ShimRemoval(
    shim="minimax--messages",
    removed_in="0.16.0",
    remote={
        "url": "https://github.com/Oaklight/llm-rosetta-shim-minimax",
        "path": "src/llm_rosetta_shim_minimax/providers",
    },
    source_repo="https://github.com/Oaklight/llm-rosetta-shim-minimax",
    legacy_names=("minimax--anthropic",),
)
MINIMAX_CHAT = ShimRemoval(
    shim="minimax--chat_completions",
    removed_in="0.16.0",
    remote={
        "url": "https://github.com/Oaklight/llm-rosetta-shim-minimax",
        "path": "src/llm_rosetta_shim_minimax/providers",
    },
    source_repo="https://github.com/Oaklight/llm-rosetta-shim-minimax",
    legacy_names=("minimax--openai_chat",),
)
MINIMAX_REMOVALS = (MINIMAX, MINIMAX_CHAT)

#: A catalogue-backed entry, so the warning can point at ``shims.presets``.
ARGO_REMOVAL = ShimRemoval(
    shim="argo--messages",
    removed_in="0.16.0",
    remote=ARGO_PACKAGE.remote,
    source_repo=ARGO_PACKAGE.repo,
    legacy_names=("argo--anthropic",),
)


@pytest.fixture
def scheduled_minimax(monkeypatch):
    """Make the detector treat ``minimax--messages`` as a scheduled removal."""
    monkeypatch.setattr("llm_rosetta.shims.removals.SHIM_REMOVALS", MINIMAX_REMOVALS)
    return MINIMAX_REMOVALS


@pytest.fixture
def scheduled_argo(monkeypatch):
    monkeypatch.setattr("llm_rosetta.shims.removals.SHIM_REMOVALS", (ARGO_REMOVAL,))
    return (ARGO_REMOVAL,)


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
        location="https://example.invalid/shim-source",
        name="local",
        ok=True,
        shim_count=len(names),
        shim_names=tuple(names),
    )


class TestAdvisoryWithSources:
    """A declared source of the same name takes the shim off the list."""

    def test_a_loaded_source_silences_its_own_shim(self, scheduled_minimax):
        providers = {
            "mini": {"shim": "minimax--messages"},
            "mini-chat": {"shim": "minimax--chat_completions"},
        }
        advisories = shim_removal_advisories(
            providers, _report(_loaded("minimax--messages"))
        )
        assert [a.shim for a in advisories] == ["minimax--chat_completions"]

    def test_a_failed_source_silences_nothing(self, scheduled_minimax):
        """It overrode nothing, so the provider is still on the built-in."""
        providers = {"mini": {"shim": "minimax--messages"}}
        failed = ShimSourceStatus(kind="remote", location="x", ok=False, error="boom")
        assert len(shim_removal_advisories(providers, _report(failed))) == 1

    def test_no_report_at_all_still_advises(self, scheduled_minimax):
        assert (
            len(shim_removal_advisories({"mini": {"shim": "minimax--messages"}}, None))
            == 1
        )


class TestReporting:
    """What the operator actually sees at startup."""

    def test_one_warning_per_advisory(self, scheduled_minimax, gateway_warnings):
        providers = {
            "mini": {"shim": "minimax--messages"},
            "mini-chat": {"shim": "minimax--chat_completions"},
        }
        advisories = report_shim_removals(providers, None)
        assert len(advisories) == 2
        assert len(gateway_warnings) == 2
        messages = " ".join(record.getMessage() for record in gateway_warnings)
        assert "minimax--messages" in messages
        assert "0.16.0" in messages
        # The message has to be actionable: it names the source to declare.
        assert "llm-rosetta-shim-minimax" in messages

    def test_a_catalogue_removal_names_the_preset(
        self, scheduled_argo, gateway_warnings
    ):
        """The shortest migration is one config line: ``shims.presets``."""
        report_shim_removals({"argo": {"shim": "argo--messages"}}, None)
        messages = " ".join(record.getMessage() for record in gateway_warnings)
        assert "shims.presets" in messages
        assert "'argo'" in messages
        assert "llm-rosetta-shim-argo" in messages

    def test_a_clean_config_is_silent(self, gateway_warnings):
        assert report_shim_removals({"o": {"shim": "openai--chat_completions"}}) == []
        assert gateway_warnings == []

    def test_the_shipped_manifest_advises_nothing(self):
        """Everything moved out already; there is nothing left to migrate."""
        assert shim_removal_advisories({"argo": {"shim": "argo--messages"}}, None) == []


class TestGatewayConfigView:
    """``raw_providers`` is the unfiltered, corrected view the advisory needs."""

    def test_a_disabled_provider_is_still_visible(self, scheduled_minimax):
        config = GatewayConfig(
            {
                "providers": {
                    "on": {"shim": "minimax--messages", "api_key": "k"},
                    "off": {
                        "shim": "minimax--messages",
                        "api_key": "k",
                        "enabled": False,
                    },
                },
                "models": {"m": "on"},
            }
        )
        assert set(config.raw_providers) == {"on", "off"}
        # ...while the routing view still hides it.
        assert "off" not in config._raw_providers

    def test_legacy_names_arrive_corrected(self, scheduled_minimax):
        config = GatewayConfig(
            {
                "providers": {"p": {"shim": "minimax--anthropic", "api_key": "k"}},
                "models": {"m": "p"},
            }
        )
        assert config.raw_providers["p"]["shim"] == "minimax--messages"
        advisories = shim_removal_advisories(config.raw_providers, None)
        assert [a.shim for a in advisories] == ["minimax--messages"]


class TestAdminPayload:
    """The panel's keys, and the preset round-trip it drives."""

    @staticmethod
    def _request(report: ShimLoadReport | None = None):
        return SimpleNamespace(app=SimpleNamespace(shim_source_report=report))

    def test_advisories_are_derived_from_the_raw_config(self, scheduled_minimax):
        raw = {"providers": {"mini": {"shim": "minimax--messages"}}}
        payload = _shim_removal_advisories(self._request(), raw)
        assert [entry["provider"] for entry in payload] == ["mini"]
        assert payload[0]["removed_in"] == "0.16.0"

    def test_a_declared_source_clears_the_advisory_on_the_next_read(
        self, scheduled_minimax
    ):
        raw = {"providers": {"mini": {"shim": "minimax--messages"}}}
        assert _shim_removal_advisories(self._request(), raw) != []
        request = self._request(_report(_loaded("minimax--messages")))
        assert _shim_removal_advisories(request, raw) == []

    def test_every_preset_is_an_acceptable_remotes_entry(self, tmp_path):
        """What the panel appends is exactly what the PUT endpoint validates."""
        config_path = str(tmp_path / "config.jsonc")
        for entry in SHIM_SOURCE_PACKAGES:
            body = {"paths": [], "remotes": [dict(entry.remote)]}
            assert _validate_shim_sources_body(body, config_path) is None, entry.key

    def test_every_preset_is_an_acceptable_presets_entry(self, tmp_path):
        """The preset picker writes keys, not expanded remotes."""
        config_path = str(tmp_path / "config.jsonc")
        for entry in SHIM_SOURCE_PACKAGES:
            body = {"paths": [], "remotes": [], "presets": [entry.key]}
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
            assert set(entry["remote"]) == {"url", "path"}
            assert entry["shims"]

    def test_presets_come_from_the_source_catalog(self):
        """The panel's catalogue is the standalone-package list, not the manifest."""
        from llm_rosetta.gateway.admin.routes import config as routes_config

        assert routes_config.SHIM_SOURCE_PACKAGES is SHIM_SOURCE_PACKAGES


class TestPresets:
    """``shims.presets`` preloads a catalogue package by name."""

    def test_a_preset_expands_to_the_catalogue_remote(self):
        _dirs, remotes, _cache = resolve_source_config({"presets": ["argo"]}, None)
        assert remotes == [ARGO_PACKAGE.remote]

    def test_presets_expand_in_order_and_dedupe(self):
        _dirs, remotes, _cache = resolve_source_config(
            {"presets": ["argo", "alcf", "argo"]}, None
        )
        assert [remote["url"] for remote in remotes] == [
            ARGO_PACKAGE.repo,
            ALCF_PACKAGE.repo,
        ]

    def test_an_explicit_remote_layers_onto_the_same_url(self):
        """A pin adds keys to the preset; it must not drop the path."""
        pin = {"url": ARGO_PACKAGE.repo, "ref": "v1.0.0"}
        _dirs, remotes, _cache = resolve_source_config(
            {"presets": ["argo"], "remotes": [pin]}, None
        )
        assert remotes == [{**ARGO_PACKAGE.remote, "ref": "v1.0.0"}]

    def test_a_spelling_of_the_url_still_matches_the_preset(self):
        """A trailing slash or `.git` addresses the same repository."""
        for spelling in (ARGO_PACKAGE.repo + "/", ARGO_PACKAGE.repo + ".git"):
            _dirs, remotes, _cache = resolve_source_config(
                {"presets": ["argo"], "remotes": [{"url": spelling, "ref": "v1"}]},
                None,
            )
            assert len(remotes) == 1
            assert remotes[0]["path"] == ARGO_PACKAGE.remote["path"]
            assert remotes[0]["ref"] == "v1"

    def test_an_unrelated_explicit_remote_is_kept_alongside(self):
        explicit = {"url": "https://example.invalid/shims"}
        _dirs, remotes, _cache = resolve_source_config(
            {"presets": ["argo"], "remotes": [explicit]}, None
        )
        assert [remote["url"] for remote in remotes] == [
            ARGO_PACKAGE.repo,
            explicit["url"],
        ]

    def test_an_unknown_preset_names_the_known_ones(self):
        with pytest.raises(ValueError, match="unknown shim preset 'nope'"):
            resolve_source_config({"presets": ["nope"]}, None)
        with pytest.raises(ValueError, match="alcf, argo, asksage"):
            resolve_source_config({"presets": ["nope"]}, None)

    @pytest.mark.parametrize("presets", ["argo", [123], [""], [None]])
    def test_bad_preset_shapes_raise(self, presets):
        with pytest.raises(ValueError, match=r"shims\.presets"):
            resolve_source_config({"presets": presets}, None)

    def test_the_preset_remote_needs_no_token_for_a_public_repo(self):
        _dirs, remotes, _cache = resolve_source_config({"presets": ["argo"]}, None)
        assert "token_env" not in remotes[0]

    def test_absent_presets_change_nothing(self):
        assert resolve_source_config({}, None)[1] == []

    def test_explicit_remotes_keep_their_previous_behaviour(self):
        """Only a preset is overridden; two explicit entries both survive."""
        first = {"url": "https://example.invalid/shims", "ref": "a"}
        second = {"url": "https://example.invalid/shims", "ref": "b"}
        _dirs, remotes, _cache = resolve_source_config(
            {"remotes": [first, second]}, None
        )
        assert remotes == [first, second]


@pytest.mark.parametrize("package", SHIM_SOURCE_PACKAGES, ids=lambda p: p.key)
def test_every_preset_points_at_its_own_source(package):
    assert package.remote["url"] == package.repo


class TestStrandedConfig:
    """A config that still names a shim that left core is told where it went."""

    @pytest.mark.parametrize(
        ("shim", "repo"),
        [
            ("alcf_metis--chat_completions", "llm-rosetta-shim-alcf"),
            ("argo--messages", "llm-rosetta-shim-argo"),
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
        # The shortest migration is the preset; the full remote is the fallback.
        assert "shims.presets" in message
        assert "shims.remotes" in message
        # The published repositories are public: no token is required.
        assert "token_env" not in message

    def test_an_unrelated_unknown_name_keeps_the_generic_message(self):
        with pytest.raises(ValueError) as excinfo:
            GatewayConfig(
                {
                    "providers": {
                        "p": {
                            "shim": "zzz_unknown--chat_completions",
                            "api_key": "k",
                        }
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
            "providers": {"mini": {"shim": "minimax--messages", "api_key": "sk-test"}},
            "models": {},
            "server": {},
        }

    @staticmethod
    def _advisories(records: list[logging.LogRecord]) -> list[str]:
        return [
            record.getMessage()
            for record in records
            if "minimax--messages" in record.getMessage()
            and "0.16.0" in record.getMessage()
        ]

    @pytest.mark.asyncio
    async def test_starting_the_gateway_warns_about_a_scheduled_removal(
        self, scheduled_minimax, gateway_warnings
    ):
        from llm_rosetta.gateway.app import create_app

        await create_app(GatewayConfig(self._raw()))

        assert self._advisories(gateway_warnings)

    @pytest.mark.asyncio
    async def test_a_declared_source_silences_the_warning_at_the_app_level(
        self, scheduled_minimax, tmp_path, gateway_warnings
    ):
        """The whole point of the advisory: a quiet log means the migration landed."""
        from llm_rosetta.gateway.app import create_app
        from llm_rosetta.shims import rebuild_shims

        leaf = tmp_path / "local-shims" / "messages"
        leaf.mkdir(parents=True)
        (leaf / "provider.yaml").write_text(
            "name: minimax--messages\n"
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
    async def test_a_disabled_provider_is_still_reported(
        self, scheduled_minimax, gateway_warnings
    ):
        """Its config breaks the moment it is switched back on.

        The mutation this catches is ``raw_providers`` -> the enabled-only view:
        the warning for the disabled provider disappears and nothing else moves.
        """
        from llm_rosetta.gateway.app import create_app

        raw = {
            "providers": {
                "mini": {"shim": "minimax--messages", "api_key": "sk-test"},
                "mini-chat": {
                    "shim": "minimax--chat_completions",
                    "api_key": "sk-test",
                    "enabled": False,
                },
            },
            "models": {},
            "server": {},
        }

        await create_app(GatewayConfig(raw))

        messages = [record.getMessage() for record in gateway_warnings]
        assert any("minimax--chat_completions" in message for message in messages)
        assert self._advisories(gateway_warnings)
