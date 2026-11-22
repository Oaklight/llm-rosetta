"""Migration advice for shims that left core or were renamed (#1106).

A config written for an older build can name a shim in two ways that no longer
resolve: an older *spelling* of a name that still ships in a standalone vendor
(``alcf--sophia``, ``asksage--openai_chat``), or a name that was never renamed
but whose vendor is simply not installed (``alcf_sophia--chat_completions``).
Both cases have to lead to the same two facts — the canonical spelling, and the
``shims.install`` entry that makes it loadable — so the advice sentences are
built from the stand-alone vendor table and the core retired-names table.

The two tables are separate, so the first tests here keep them in step; the
rest pin the sentences and the warning path that delivers them.
"""

from __future__ import annotations

import logging

import pytest

from llm_rosetta import retired_names
from llm_rosetta_gateway.config.providers import build_provider_info
from llm_rosetta_gateway.shims import (
    legacy_shim_migration_advice,
    source_package_advice,
)
from llm_rosetta_gateway.shims.catalog import (
    SHIM_SOURCES,
    STANDALONE_VENDORS,
    find_standalone,
    get_shim_source,
    get_standalone,
)
from llm_rosetta.api import load_providers

ALL_VENDORS = pytest.mark.parametrize("vendor", STANDALONE_VENDORS, ids=lambda v: v.key)

#: Rows of the retired-names table that name a stand-alone vendor's shim.
_INSTALL_ROWS = [
    row
    for row in retired_names.RETIRED_NAMES
    if row.kind == retired_names.SHIM and row.install is not None
]


class TestVendorTableAndRetiredNamesAgree:
    """The vendor table and the retired-names table name the same installs."""

    @ALL_VENDORS
    def test_every_install_selector_names_a_known_source(self, vendor):
        source = get_shim_source(vendor.source)
        assert source is not None, vendor.key
        assert vendor.install.split(":", 1)[0] == vendor.source
        assert vendor.source in {s.key for s in SHIM_SOURCES}

    @ALL_VENDORS
    def test_lookup_returns_the_owning_vendor(self, vendor):
        assert get_standalone(vendor.key) is vendor
        for shim in vendor.shims:
            assert find_standalone(shim) is vendor

    @pytest.mark.parametrize("row", _INSTALL_ROWS, ids=lambda r: r.name)
    def test_every_retired_name_resolves_within_its_vendor(self, row):
        vendor = find_standalone(row.replacement)
        assert vendor is not None, row.name
        assert row.install == vendor.install

    @pytest.mark.parametrize("row", _INSTALL_ROWS, ids=lambda r: r.name)
    def test_no_retired_name_is_still_a_registered_shim(self, row):
        """A legacy spelling that still resolves needs no migration advice."""
        from llm_rosetta.shims import get_shim, list_shims

        load_providers()
        assert row.name not in {s.name for s in list_shims()}
        assert get_shim(row.name) is None

    def test_a_retired_name_is_listed_for_every_vendor_rename(self):
        """The table carries the renames the advice lookup depends on."""
        assert {row.install for row in _INSTALL_ROWS} >= {
            "argonne:argo",
            "argonne:alcf",
            "argonne:asksage",
        }


class TestMigrationAdvice:
    """One sentence per situation, naming the canonical name and the install entry."""

    @pytest.mark.parametrize(
        "legacy,canonical,install",
        [
            ("alcf--sophia", "alcf_sophia--chat_completions", "argonne:alcf"),
            ("asksage--openai_chat", "asksage--chat_completions", "argonne:asksage"),
            ("argo--anthropic", "argo--messages", "argonne:argo"),
        ],
    )
    def test_legacy_name_gets_canonical_and_install(self, legacy, canonical, install):
        advice = legacy_shim_migration_advice(legacy)
        assert canonical in advice
        assert f"shims.install: ['{install}']" in advice

    @pytest.mark.parametrize(
        "canonical,install",
        [
            ("alcf_sophia--chat_completions", "argonne:alcf"),
            ("asksage--messages", "argonne:asksage"),
            ("argo--chat_completions", "argonne:argo"),
        ],
    )
    def test_canonical_name_gets_only_the_install(self, canonical, install):
        assert legacy_shim_migration_advice(canonical) == ""
        advice = source_package_advice(canonical)
        assert f"shims.install: ['{install}']" in advice

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
    warnings while ``llm_rosetta_gateway.observability.logging`` had not been imported yet in
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
        assert "shims.install: ['argonne:alcf']" in warning

    def test_unregistered_canonical_name_warns_with_its_install_entry(
        self, gateway_warnings
    ):
        self._build("asksage--messages", "asksage--messages")
        warning = _warning_text(gateway_warnings)
        assert "shims.install: ['argonne:asksage']" in warning

    def test_unknown_name_warns_without_advice(self, gateway_warnings):
        self._build("not-a-provider", None)
        warning = _warning_text(gateway_warnings)
        assert "generic URL template" in warning
        assert "shims.install" not in warning


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
        assert "shims.install: ['argonne:alcf']" in item["reason"]

    def test_unknown_shim_field_stays_terse(self):
        item = self._validate({"shim": "not-a-shim", "api_key": "k"})
        assert item["category"] == "unknown_shim"
        assert "shims.install" not in item["reason"]
