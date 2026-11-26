"""Every vendor of a source is installable, removable and pinnable on its own.

The vendor is the install unit (one archive, one version, one manifest
record); the shim is the enable unit inside it (``shims.disabled``).  These
cases pin the three ``shims.install`` spellings that make a vendor independent
of the tag, ``*`` or legacy preset that selects it:

- an exclusion ``!<source>:<vendor>`` / ``!<vendor>`` always wins;
- an explicit ``<source>:<vendor>`` beats a tag or ``*`` entry, so
- a pin ``<source>:<vendor>@<version>`` holds whatever else selects the vendor;

and the rewrite that spells a legacy ``shims.presets`` key as install entries,
used by ``llm-rosetta migrate --config``.
"""

from __future__ import annotations

import json
import re

import pytest

from llm_rosetta.contracts import SURFACE_VERSION
from llm_rosetta_gateway.maintenance.migrate_0140 import migrate_config_text
from llm_rosetta_gateway.maintenance.migrate_presets import (
    presets_as_install,
    rewrite_presets_section,
)
from llm_rosetta_gateway.shims.manifest_sources.plan import (
    excluded_by,
    parse_selector,
    pick_selector,
)

from .test_shim_manifest_sources import (
    MANIFEST,
    FakeHttp,
    _apply,
    _Capture,
    _url_section,
    _vendor_status,
)


@pytest.fixture(autouse=True)
def _registry():
    from llm_rosetta.testing import restore_registry, snapshot_registry

    snapshot = snapshot_registry()
    yield
    restore_registry(snapshot)


@pytest.fixture
def logs():
    import logging

    handler = _Capture()
    loggers = [
        logging.getLogger(n)
        for n in ("llm_rosetta_gateway.shims.sources", "llm-rosetta-gateway")
    ]
    levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.DEBUG)
        lg.addHandler(handler)
    try:
        yield handler.messages
    finally:
        for lg, level in zip(loggers, levels):
            lg.removeHandler(handler)
            lg.setLevel(level)


def _sel(*entries: str):
    return [parse_selector(e) for e in entries]


class TestGrammar:
    @pytest.mark.parametrize(
        "text,source,value,negate,version",
        [
            ("!llm7", None, "llm7", True, None),
            ("!contrib:llm7", "contrib", "llm7", True, None),
            ("contrib:groq@0.5.0", "contrib", "groq", False, "0.5.0"),
            ("groq@1.2.3-rc.1", None, "groq", False, "1.2.3-rc.1"),
        ],
    )
    def test_exclusion_and_pin_spellings(self, text, source, value, negate, version):
        sel = parse_selector(text)
        assert (sel.source, sel.kind, sel.value, sel.negate, sel.version) == (
            source,
            "vendor",
            value,
            negate,
            version,
        )

    @pytest.mark.parametrize(
        "text",
        [
            "!tag:tier:free",
            "!*",
            "!contrib:*",
            "contrib:tag:x@1.0",
            "!groq@1.0",
            "groq@",
            "groq@x",
        ],
    )
    def test_only_a_vendor_is_excluded_or_pinned(self, text):
        with pytest.raises(ValueError):
            parse_selector(text)


class TestPickRule:
    @pytest.mark.parametrize(
        "entries",
        [
            ("contrib:tag:tier:free", "!contrib:llm7"),
            ("!contrib:llm7", "contrib:tag:tier:free"),
            ("contrib:llm7", "!llm7"),
            ("contrib:*", "!contrib:llm7"),
        ],
    )
    def test_an_exclusion_wins_wherever_it_is_listed(self, entries):
        assert pick_selector(_sel(*entries), "contrib", "llm7", ["tier:free"]) is None
        assert excluded_by(_sel(*entries), "contrib", "llm7") is not None

    def test_an_exclusion_names_one_vendor_of_one_source(self):
        sel = _sel("contrib:tag:tier:free", "*", "!contrib:llm7")
        free = pick_selector(sel, "contrib", "ovhcloud", ["tier:free"])
        assert free is not None and free.raw == "contrib:tag:tier:free"
        other = pick_selector(sel, "mirror", "llm7", [])
        assert other is not None and other.raw == "*"

    def test_an_explicit_entry_beats_a_tag_listed_before_it(self):
        sel = _sel("contrib:tag:tier:free", "contrib:llm7@0.4.0")
        picked = pick_selector(sel, "contrib", "llm7", ["tier:free"])
        assert picked is not None and picked.version == "0.4.0"


class TestInstall:
    def test_an_excluded_vendor_is_not_installed_and_its_siblings_are(self, tmp_path):
        http = FakeHttp()
        http.publish(
            "free_a", "0.1.0", ["free_a--chat_completions"], tags=["tier:free"]
        )
        http.publish(
            "free_b", "0.1.0", ["free_b--chat_completions"], tags=["tier:free"]
        )

        report = _apply(
            tmp_path, _url_section("contrib:tag:tier:free", "!contrib:free_a"), http
        )

        assert [s.vendor for s in report.sources] == ["free_b"]

    def test_an_exclusion_matching_nothing_is_not_a_missing_vendor(
        self,
        tmp_path,
        logs,
    ):
        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"])

        _apply(tmp_path, _url_section("acme", "!contrib:ghost"), http)

        assert not any("matched no vendor" in m for m in logs), logs

    def test_a_pin_installs_that_version_and_holds_it(self, tmp_path):
        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"])
        http.publish("acme", "0.2.0", ["acme--chat_completions"])

        report = _apply(tmp_path, _url_section("contrib:acme@0.1.0"), http)

        assert _vendor_status(report, "acme").release["version"] == "0.1.0"
        assert not any("0.2.0/" in c for c in http.calls)

    def test_a_pin_below_the_installed_version_installs_it(self, tmp_path):
        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"])
        http.publish("acme", "0.2.0", ["acme--chat_completions"])
        assert (
            _vendor_status(
                _apply(tmp_path, _url_section("acme"), http), "acme"
            ).release["version"]
            == "0.2.0"
        )

        report = _apply(
            tmp_path, _url_section("contrib:acme@0.1.0"), http, force_refresh=True
        )

        assert _vendor_status(report, "acme").release["version"] == "0.1.0"

    def test_a_pin_beats_the_tag_that_also_selects_the_vendor(self, tmp_path):
        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"], tags=["tier:free"])
        http.publish("acme", "0.2.0", ["acme--chat_completions"], tags=["tier:free"])

        report = _apply(
            tmp_path, _url_section("contrib:tag:tier:free", "contrib:acme@0.1.0"), http
        )

        assert _vendor_status(report, "acme").release["version"] == "0.1.0"

    @pytest.mark.parametrize("missing", ["absent", "incompatible"])
    def test_a_pin_that_cannot_install_warns_and_keeps_what_runs(
        self,
        tmp_path,
        logs,
        missing,
    ):
        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"])
        _apply(tmp_path, _url_section("acme"), http)
        if missing == "incompatible":
            http.publish(
                "acme", "0.9.0", ["acme--chat_completions"], surface=SURFACE_VERSION + 1
            )

        report = _apply(
            tmp_path, _url_section("contrib:acme@0.9.0"), http, force_refresh=True
        )

        status = _vendor_status(report, "acme")
        assert status.ok and status.stale
        assert status.release["version"] == "0.1.0"
        assert "pinned acme@0.9.0" in status.error
        assert any("is pinned to 0.9.0" in m for m in logs), logs

    def test_a_pin_that_cannot_install_with_nothing_installed_fails(self, tmp_path):
        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"])

        report = _apply(tmp_path, _url_section("contrib:acme@0.3.0"), http)

        status = _vendor_status(report, "acme")
        assert not status.ok and "the index does not list" in status.error


class TestSourceView:
    def test_the_card_reports_the_pin_and_the_exclusion(self, tmp_path):
        from llm_rosetta_gateway.shims.manifest_sources.plan import ManifestSourceSpec
        from llm_rosetta_gateway.shims.source_view import source_catalogue

        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"])
        http.publish("acme", "0.2.0", ["acme--chat_completions"])
        http.publish("zed", "0.1.0", ["zed--chat_completions"])
        spec = ManifestSourceSpec(key=None, url=http.dist + "index.json")

        out = source_catalogue(
            [spec],
            raw_sources=[],
            selectors=_sel("contrib:*", "contrib:acme@0.1.0", "!contrib:zed"),
            statuses=[],
            cache_root=tmp_path,
            ttl=60,
            http_get=http,
        )

        acme, zed = out[0]["vendors"]
        assert acme["pinned"] == "0.1.0" and acme["pin_problem"] is None
        assert acme["installable"] == "0.2.0"
        assert zed["excluded_by"] == "!contrib:zed" and zed["selected_by"] is None
        assert zed["state"] == "available"


class TestPresetRewrite:
    def test_a_preset_becomes_its_mapped_selector(self):
        out = presets_as_install(["free", "alcf"])
        assert out.install == ["contrib:tag:tier:free", "argonne:alcf"]
        assert out.converted == ["free", "alcf"] and out.sources == [
            "contrib",
            "argonne",
        ]

    def test_an_exclude_that_drops_a_vendor_becomes_an_exclusion(self):
        out = presets_as_install(["free"], exclude={"free": ["llm7--*"]})
        assert out.install == ["contrib:tag:tier:free", "!contrib:llm7"]
        assert out.disabled == []

    def test_an_include_becomes_one_explicit_entry_per_kept_vendor(self):
        out = presets_as_install(
            ["free"], include={"free": ["kilo_free--*", "llm7--chat_completions"]}
        )
        assert out.install == ["contrib:kilo_free", "contrib:llm7"]

    def test_a_shim_dropped_from_a_kept_vendor_is_switched_off(self):
        out = presets_as_install(["alcf"], exclude={"alcf": ["alcf_metis--*"]})
        assert out.install == ["argonne:alcf"]
        assert out.disabled == ["alcf_metis--chat_completions"]

    def test_an_unknown_preset_stays(self):
        out = presets_as_install(["argo", "nope"])
        assert out.converted == ["argo"]
        assert out.install == ["argonne:argo"]
        assert set(out.unconvertible) == {"nope"}

    def test_the_section_rewrite_keeps_what_did_not_convert(self):
        section = {
            "presets": ["free", "nope"],
            "exclude": {"free": ["llm7--*"]},
            "install": ["contrib:groq"],
            "sources": ["argonne"],
        }

        new, rewrite = rewrite_presets_section(section)

        assert new["presets"] == ["nope"]
        assert "exclude" not in new
        assert new["install"] == [
            "contrib:groq",
            "contrib:tag:tier:free",
            "!contrib:llm7",
        ]
        assert new["sources"] == ["argonne", "contrib"]
        assert section["presets"] == ["free", "nope"], "the input is not mutated"
        assert rewrite.converted == ["free"]


class TestMigrateRewrite:
    TEXT = (
        "{\n"
        "  // gateway\n"
        '  "shims": {\n'
        '    "presets": ["alcf", "free"],  // legacy\n'
        '    "exclude": {"free": ["llm7--*"]},\n'
        '    "cache_dir": "./c"\n'
        "  },\n"
        '  "providers": {}\n'
        "}\n"
    )

    def test_presets_are_spelled_as_install_entries_and_comments_stay(self):
        result = migrate_config_text(self.TEXT, "c.jsonc")

        assert result.error is None
        assert "// gateway" in result.new_text and "// legacy" in result.new_text
        doc = json.loads(
            "\n".join(line.split("//")[0] for line in result.new_text.splitlines())
        )
        assert doc["shims"] == {
            "install": ["argonne:alcf", "contrib:tag:tier:free", "!contrib:llm7"],
            "cache_dir": "./c",
        }
        rewrites = [(i.old, i.new) for i in result.items if i.where == "shims.presets"]
        assert rewrites == [
            ("alcf", "argonne:alcf"),
            ("free", "contrib:tag:tier:free, !contrib:llm7"),
        ]

    def test_an_existing_install_list_is_extended(self):
        text = '{"shims": {"install": ["contrib:groq"], "presets": ["opendesign"]}}'

        result = migrate_config_text(text)

        assert json.loads(result.new_text.replace(", }", "}")) == {
            "shims": {"install": ["contrib:groq", "contrib:opendesign"]}
        }

    def test_a_renamed_shim_needs_an_install_entry_not_a_preset(self):
        text = '{"providers": {"a": {"shim": "alcf--sophia"}}}'

        result = migrate_config_text(text)

        doc = json.loads(result.new_text)
        assert doc["shims"] == {"install": ["argonne:alcf"]}
        assert doc["providers"]["a"]["shim"] == "alcf_sophia--chat_completions"

    def test_an_unknown_preset_is_left_and_reported(self):
        text = '{"shims": {"presets": ["nope"]}}'

        result = migrate_config_text(text)

        assert result.new_text == text
        assert any(i.action == "unmapped" and i.old == "nope" for i in result.items)

    def test_a_migrated_document_parses_as_the_same_selection(self, tmp_path):
        from llm_rosetta_gateway.shims.shim_sources import parse_shims_section

        result = migrate_config_text(self.TEXT)
        doc = json.loads(
            "\n".join(line.split("//")[0] for line in result.new_text.splitlines())
        )

        parsed = parse_shims_section(doc["shims"], str(tmp_path / "c.jsonc"))

        raws = [s.raw for s in parsed.manifests.selectors]
        assert raws == ["argonne:alcf", "contrib:tag:tier:free", "!contrib:llm7"]


# -- Review round (milo, elena on #1671) -------------------------------------


class TestPinConflict:
    def test_two_pins_of_one_vendor_are_not_order_decided(self):
        for entries in (
            ("contrib:kilo@0.5.0", "contrib:kilo@0.4.0"),
            ("contrib:kilo@0.4.0", "contrib:kilo@0.5.0"),
            ("kilo@0.4.0", "contrib:kilo@0.5.0"),
        ):
            picked = pick_selector(_sel(*entries), "contrib", "kilo", [])
            assert picked is not None
            assert set(picked.pin_conflict) == set(entries), entries

    def test_a_pin_beats_an_unpinned_entry_of_the_same_vendor(self):
        for entries in (
            ("contrib:kilo", "contrib:kilo@0.4.0"),
            ("contrib:kilo@0.4.0", "contrib:kilo"),
        ):
            picked = pick_selector(_sel(*entries), "contrib", "kilo", [])
            assert picked is not None and picked.version == "0.4.0"
            assert not picked.pin_conflict

    def test_the_installer_refuses_to_choose_and_names_both_pins(self, tmp_path, logs):
        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"])
        http.publish("acme", "0.2.0", ["acme--chat_completions"])
        _apply(tmp_path, _url_section("contrib:acme@0.1.0"), http)

        report = _apply(
            tmp_path,
            _url_section("contrib:acme@0.2.0", "contrib:acme@0.1.0"),
            http,
            force_refresh=True,
        )

        rows = [s for s in report.sources if s.vendor == "acme"]
        status = rows[0]
        # What ran keeps running (stale); nothing is silently picked.
        assert status.stale and status.release["version"] == "0.1.0"
        assert "pinned more than once" in status.error
        assert (
            "contrib:acme@0.2.0" in status.error
            and "contrib:acme@0.1.0" in status.error
        )
        assert any("pinned more than once" in m for m in logs), logs
        assert not any("0.2.0/" in c for c in http.calls)


class TestOverriddenRows:
    def test_an_excluded_entry_says_which_exclusion_won(self, tmp_path, logs):
        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"])

        report = _apply(tmp_path, _url_section("contrib:acme", "!contrib:acme"), http)

        row = next(s for s in report.sources if s.vendor == "acme")
        assert row.skipped == (
            "excluded by shims.install ('!contrib:acme'); removing that entry brings it back"
        )
        assert not any("matched no vendor" in m for m in logs), logs

    def test_an_entry_another_entry_beat_says_which(self, tmp_path):
        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"])

        report = _apply(tmp_path, _url_section("contrib:acme@0.1.0", "acme"), http)

        rows = {s.name: s for s in report.sources}
        assert rows["contrib:acme"].skipped == "already selected by contrib:acme@0.1.0"

    def test_a_vendor_the_manifest_lacks_still_says_so(self, tmp_path):
        http = FakeHttp()
        http.publish("acme", "0.1.0", ["acme--chat_completions"])

        report = _apply(tmp_path, _url_section("acme", "ghost"), http)

        assert next(s for s in report.sources if s.vendor == "ghost").skipped == (
            "not in the manifest"
        )


class TestMigrateReview:
    def test_a_non_array_install_refuses_the_rewrite_and_changes_nothing(self):
        for key, value in (
            ("install", '"contrib:qwen"'),
            ("disabled", '"x"'),
            ("sources", '"contrib"'),
        ):
            text = '{"shims": {"presets": ["free"], "' + key + '": ' + value + "}}"

            result = migrate_config_text(text)

            assert result.new_text == text, key
            assert any(
                i.action == "unmapped" and i.where == f"shims.{key}"
                for i in result.items
            ), key
            assert not any(i.action == "rewrite" for i in result.items), key

    def test_a_source_declared_by_its_manifest_url_is_not_declared_twice(self):
        from llm_rosetta_gateway.shims.catalog import get_shim_source

        source = get_shim_source("contrib")
        assert source is not None
        contrib = source.manifest_url
        text = (
            '{\n  "shims": {\n    "sources": [{"url": "'
            + contrib
            + '"}],\n    "presets": ["free"] // the free tier\n  }\n}\n'
        )

        result = migrate_config_text(text)

        doc = json.loads(
            "\n".join(x.split(" //")[0] for x in result.new_text.splitlines())
        )
        assert doc["shims"]["sources"] == [{"url": contrib}]
        # Recognised as contrib, not merely left alone as an unknown URL.
        assert not any(i.where == "shims.sources" for i in result.items)
        # The comment that annotated presets travels with the entries.
        install_line = next(x for x in result.new_text.splitlines() if '"install"' in x)
        assert install_line.rstrip().endswith("// the free tier")
        sources_block = result.new_text.split('"install"')[0]
        assert "the free tier" not in sources_block

    def test_an_unknown_url_source_is_reported_not_doubled(self):
        text = (
            '{"shims": {"sources": [{"url": "https://mirror.example/c/index.json"}], '
            '"presets": ["free"]}}'
        )

        result = migrate_config_text(text)

        doc = json.loads(result.new_text)
        assert doc["shims"]["sources"] == [
            {"url": "https://mirror.example/c/index.json"}
        ]
        assert any(
            i.action == "note" and i.where == "shims.sources" and i.old == "contrib"
            for i in result.items
        )

    def test_a_dropped_last_member_leaves_no_trailing_comma(self):
        text = (
            '{\n  "shims": {\n    "install": ["contrib:groq"],\n'
            '    "presets": ["opendesign"]  // legacy\n  }\n}\n'
        )

        result = migrate_config_text(text)

        assert json.loads(
            "\n".join(x.split(" //")[0] for x in result.new_text.splitlines())
        ) == {"shims": {"install": ["contrib:groq", "contrib:opendesign"]}}
        # No comma between the last member and the closing brace.
        assert not re.search(r",\s*(//[^\n]*)?\s*\}", result.new_text), result.new_text
        assert "// legacy" in result.new_text

    def test_a_preset_that_converts_to_nothing_stays_and_needs_a_person(self):
        text = '{"shims": {"presets": ["free"], "include": {"free": ["opendesign*"]}}}'

        result = migrate_config_text(text)

        assert result.new_text == text
        item = next(i for i in result.items if i.old == "free")
        assert item.action == "unmapped" and "converts to nothing" in item.detail

    def test_the_section_rewrite_keeps_a_preset_that_converts_to_nothing(self):
        section = {"presets": ["free", "alcf"], "include": {"free": ["nothing*"]}}

        new, rewrite = rewrite_presets_section(section)

        assert new["presets"] == ["free"] and new["include"] == {"free": ["nothing*"]}
        assert new["install"] == ["argonne:alcf"]
        assert "converts to nothing" in rewrite.unconvertible["free"]

    def test_the_section_rewrite_detects_a_source_by_url(self):
        from llm_rosetta_gateway.shims.catalog import get_shim_source

        contrib = get_shim_source("contrib")
        assert contrib is not None
        section = {"presets": ["free"], "sources": [{"url": contrib.manifest_url}]}

        new, rewrite = rewrite_presets_section(section)

        assert new["sources"] == [{"url": contrib.manifest_url}]
        assert rewrite.notes == []


class TestExcludedMovedVendor:
    """``!contrib:kilo`` with a provider on ``kilo--chat_completions`` (clementine)."""

    def _doc(self):
        return {
            "providers": {"k": {"shim": "kilo--chat_completions", "api_key": "x"}},
            "shims": {"sources": [MANIFEST], "install": ["!contrib:kilo"]},
        }

    def test_the_row_and_the_log_name_the_exclusion_not_the_publisher(
        self, tmp_path, logs
    ):
        from llm_rosetta_gateway.shims.shim_sources import apply_shim_sources

        http = FakeHttp()
        http.publish("kilo", "0.5.0", ["kilo--chat_completions"])

        report = apply_shim_sources(
            self._doc(), str(tmp_path / "config.jsonc"), http_get=http
        )

        row = next(s for s in report.sources if s.vendor == "kilo")
        assert row.skipped == (
            "excluded by shims.install ('!contrib:kilo'); removing that entry brings it back"
        )
        warned = [m for m in logs if "a configured provider uses vendor 'kilo'" in m]
        assert len(warned) == 1 and "!contrib:kilo" in warned[0], logs
        assert not any("matched no vendor" in m for m in logs)

    def test_the_auto_disable_reason_says_the_exclusion_keeps_it_out(self, tmp_path):
        from types import SimpleNamespace

        from llm_rosetta.api import retired_names
        from llm_rosetta_gateway.ops.provider_validation import _moved_reason

        moved = retired_names.moved_vendor("kilo--chat_completions")
        assert moved is not None
        row = SimpleNamespace(
            vendor="kilo",
            ok=False,
            skipped="excluded by shims.install ('!contrib:kilo'); "
            "removing that entry brings it back",
        )

        reason = _moved_reason(
            "kilo--chat_completions", moved, SimpleNamespace(sources=[row])
        )

        assert "excluded by shims.install ('!contrib:kilo')" in reason
        assert "removing that entry brings it back" in reason
        assert "did not succeed" not in reason and "did not install" not in reason
