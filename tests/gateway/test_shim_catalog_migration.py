"""Migration advice for shims that left core or were renamed (#1106).

A config written for an older build can name a shim in two ways that no longer
resolve: an older *spelling* of a name that still ships in a standalone package
(``alcf--sophia``, ``asksage--openai_chat``), or a name that was never renamed
but whose package is simply not declared (``alcf_sophia--chat_completions``).
Both cases have to lead to the same two facts — the canonical spelling, and the
preset that makes it loadable — so the catalogue records the legacy spellings
and the advice sentences are built from that record.

The catalogue is hand-maintained and the core alias table is separate, so the
first tests here keep the two in step; the rest pin the sentences and the
warning path that delivers them.
"""

from __future__ import annotations

import logging

import pytest

from llm_rosetta.provider_names import LEGACY_SHIM_ALIASES, canonical_shim_name
from llm_rosetta_gateway.providers import build_provider_info
from llm_rosetta_gateway.shims import (
    SHIM_SOURCE_PACKAGES,
    canonical_shim_for,
    find_legacy_source_package,
    find_source_package,
    legacy_shim_migration_advice,
    source_package_advice,
)

ALL_PACKAGES = pytest.mark.parametrize(
    "package", SHIM_SOURCE_PACKAGES, ids=lambda p: p.key
)

ALL_LEGACY_NAMES = pytest.mark.parametrize(
    "package,legacy",
    [(p, name) for p in SHIM_SOURCE_PACKAGES for name in p.legacy_names],
    ids=lambda value: value if isinstance(value, str) else value.key,
)


class TestCatalogueAndAliasTableAgree:
    """The catalogue's legacy names are only useful if they canonicalise."""

    @ALL_LEGACY_NAMES
    def test_every_legacy_name_resolves_within_its_package(self, package, legacy):
        canonical = canonical_shim_for(package, legacy)
        assert canonical in package.shims, (legacy, canonical)

    @ALL_LEGACY_NAMES
    def test_every_legacy_name_is_in_the_core_alias_table(self, package, legacy):
        """A catalogue entry without an alias silently keeps the old spelling.

        The alias table is what rewrites the name before a lookup, so a legacy
        name recorded only in the catalogue would still be handed to the shim
        registry verbatim — the advice would then fire on a name the operator
        never gets corrected to.
        """
        assert canonical_shim_name(legacy) == canonical_shim_for(package, legacy)

    @ALL_LEGACY_NAMES
    def test_no_legacy_name_is_still_a_registered_shim(self, package, legacy):
        """A legacy spelling that still resolves needs no migration advice."""
        from llm_rosetta.shims import get_shim, list_shims
        from llm_rosetta.shims.providers import load_providers

        load_providers()
        assert legacy not in {s.name for s in list_shims()}
        assert get_shim(legacy) is None

    def test_every_alias_pointing_at_a_package_is_a_catalogue_name(self):
        """The reverse direction: no alias invents a package name.

        ``LEGACY_SHIM_ALIASES`` also carries core renames, so only entries
        whose target a package provides are checked — those must be listed in
        that package's ``legacy_names``, or the advice lookup misses them.
        """
        catalogued = {
            (p.key, name) for p in SHIM_SOURCE_PACKAGES for name in p.legacy_names
        }
        for old, new in LEGACY_SHIM_ALIASES.items():
            package = find_source_package(new)
            if package is None:
                continue
            assert (package.key, old) in catalogued, old

    @ALL_PACKAGES
    def test_lookup_returns_the_owning_package(self, package):
        for legacy in package.legacy_names:
            assert find_legacy_source_package(legacy) is package
        for canonical in package.shims:
            assert find_source_package(canonical) is package


class TestMigrationAdvice:
    """One sentence per situation, naming the canonical name and the preset."""

    @pytest.mark.parametrize(
        "legacy,canonical,preset",
        [
            ("alcf--sophia", "alcf_sophia--chat_completions", "alcf"),
            ("asksage--openai_chat", "asksage--chat_completions", "asksage"),
            ("argo--anthropic", "argo--messages", "argo"),
        ],
    )
    def test_legacy_name_gets_canonical_and_preset(self, legacy, canonical, preset):
        advice = legacy_shim_migration_advice(legacy)
        assert canonical in advice
        assert f"shims.presets: ['{preset}']" in advice

    @pytest.mark.parametrize(
        "canonical,preset",
        [
            ("alcf_sophia--chat_completions", "alcf"),
            ("asksage--messages", "asksage"),
            ("argo--chat_completions", "argo"),
        ],
    )
    def test_canonical_name_gets_only_the_preset(self, canonical, preset):
        assert legacy_shim_migration_advice(canonical) == ""
        advice = source_package_advice(canonical)
        assert f"shims.presets: ['{preset}']" in advice

    @pytest.mark.parametrize(
        "name",
        [None, "", "chat_completions", "deepseek--chat_completions", "not-a-shim"],
    )
    def test_unknown_names_get_no_advice(self, name):
        assert legacy_shim_migration_advice(name) == ""
        assert source_package_advice(name) == ""


class _CollectingHandler(logging.Handler):
    """Collect the gateway logger's records without going through caplog."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture()
def gateway_warnings():
    """Records the gateway logger emits during the test.

    That logger is created with ``propagate = False`` — in the server it owns
    its handler — so caplog, which listens on the root logger, only saw these
    warnings while ``llm_rosetta_gateway.logging`` had not been imported yet in
    the process.  Whether it had been is decided by the rest of the run, which
    is why these three tests passed alone and failed in a full run.

    ``caplog.at_level(logger=...)`` does not settle it either: pytest 9 attaches
    the capturing handler to the named logger and pytest 8 does not, so forcing
    propagation on to reach root counts the record twice on 9.  A handler of our
    own, on the logger that emits, is one path under both.
    """
    logger = logging.getLogger("llm-rosetta-gateway")
    handler = _CollectingHandler()
    logger.addHandler(handler)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)


class TestWarningPath:
    """A stranded shim that still loads must say what to fix."""

    def _build(self, provider_type, shim_name):
        build_provider_info(
            provider_type,
            {"base_url": "https://example.test", "api_key": "k"},
            shim_name=shim_name,
        )

    def test_legacy_spelling_warns_with_the_canonical_name(self, gateway_warnings):
        self._build("alcf--sophia", "alcf--sophia")
        warning = _warning_text(gateway_warnings)
        assert "alcf_sophia--chat_completions" in warning
        assert "shims.presets: ['alcf']" in warning

    def test_unregistered_canonical_name_warns_with_its_preset(self, gateway_warnings):
        self._build("asksage--messages", "asksage--messages")
        warning = _warning_text(gateway_warnings)
        assert "shims.presets: ['asksage']" in warning

    def test_unknown_name_warns_without_advice(self, gateway_warnings):
        self._build("not-a-provider", None)
        warning = _warning_text(gateway_warnings)
        assert "generic URL template" in warning
        assert "shims.presets" not in warning


def _warning_text(records) -> str:
    matching = [r for r in records if "Unknown provider type" in r.getMessage()]
    assert len(matching) == 1, [r.getMessage() for r in records]
    return matching[0].getMessage()


class TestUnknownShimField:
    """``shim:`` is a separate path from ``type:`` and carries the same advice."""

    def _validate(self, cfg: dict):
        import asyncio

        from llm_rosetta_gateway.ops import OpsContext, OpsProviderValidation

        raw = {"providers": {"p": cfg}}
        result = asyncio.run(OpsProviderValidation(OpsContext(), raw).execute())
        return result["disabled"][0]

    def test_legacy_shim_field_names_the_canonical_shim(self):
        item = self._validate({"shim": "alcf--minerva", "api_key": "k"})
        assert item["category"] == "unknown_shim"
        assert "alcf_minerva--chat_completions" in item["reason"]
        assert "shims.presets: ['alcf']" in item["reason"]

    def test_unknown_shim_field_stays_terse(self):
        item = self._validate({"shim": "not-a-shim", "api_key": "k"})
        assert item["category"] == "unknown_shim"
        assert "shims.presets" not in item["reason"]
