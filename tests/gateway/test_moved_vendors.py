"""Vendors that moved from core to contrib keep every config working (#1660 S7).

A config that names a moved vendor's shim (``type: opencode--chat_completions``),
quota probe (``quota.probe: openai_codex``) or typed endpoint shim must keep
working without an edit: the gateway installs the vendor from the ``contrib``
source by itself, with or without a ``shims`` section.  The contrib source is
served from a fake ``http_get``: its archives are built from the frozen vendor
copies in ``tests/fixtures/moved_vendors``, so nothing reaches the network.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import tarfile
from pathlib import Path
from typing import Any

import pytest

from _moved_vendors import MOVED_VENDORS_DIR
from llm_rosetta.api import get_shim, retired_names
from llm_rosetta.contracts import IR_VERSION, SURFACE_VERSION
from llm_rosetta_gateway import shim_sources
from llm_rosetta_gateway.shim_sources import (
    apply_shim_sources,
    parse_shims_section,
    shims_signature,
)
from llm_rosetta_gateway.shims.catalog import get_shim_source
from llm_rosetta_gateway.shims.moved import moved_references, moved_vendor_sources
from llm_rosetta.api import load_providers

_CONTRIB = get_shim_source("contrib")
assert _CONTRIB is not None
CONTRIB_REPO = _CONTRIB.repo
DIST = _CONTRIB.manifest_url.rsplit("/", 1)[0] + "/"
MANIFEST_URL = _CONTRIB.manifest_url
FALLBACK_URL = _CONTRIB.fallback_url
COMMIT = "d" * 40

#: The moved vendors the fake contrib source publishes.  ``opencode``,
#: ``openai_chatgpt`` and ``kilo`` carry production traffic (``kilo`` through a
#: ``shim:`` slot); ``xai`` is a bare shim name; ``siliconflow`` serves a typed
#: (embeddings) endpoint.
PUBLISHED = ("opencode", "openai_chatgpt", "kilo", "xai", "siliconflow")


def _vendor_archive(vendor: str) -> bytes:
    """``providers/<vendor>/...`` from the frozen copy, as the producers pack it."""
    buf = io.BytesIO()
    root = MOVED_VENDORS_DIR / vendor
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in sorted(root.rglob("*")):
            if "__pycache__" in path.parts or not path.is_file():
                continue
            rel = path.relative_to(MOVED_VENDORS_DIR)
            data = path.read_bytes()
            info = tarfile.TarInfo(f"providers/{rel.as_posix()}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class FakeContrib:
    """The contrib source: a manifest and one archive per published vendor."""

    def __init__(self, vendors: tuple[str, ...] = PUBLISHED) -> None:
        self.files: dict[str, bytes] = {}
        self.records: dict[str, Any] = {}
        self.calls: list[str] = []
        self.down = False
        rows = {row.vendor: row for row in retired_names.MOVED_VENDORS}
        for vendor in vendors:
            data = _vendor_archive(vendor)
            archive = f"0.5.0/{vendor}.tar.gz"
            self.files[DIST + archive] = data
            self.records[vendor] = {
                "versions": [
                    {
                        "version": "0.5.0",
                        "commit": COMMIT,
                        "archive": archive,
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "requires": {"surface": SURFACE_VERSION, "ir": IR_VERSION},
                        "shims": list(rows[vendor].shims),
                        "tags": [],
                        "maturity": "stable",
                    }
                ]
            }

    def manifest(self) -> dict[str, Any]:
        return {
            "manifest_version": 1,
            "source": "contrib",
            "repo": CONTRIB_REPO,
            "generated_at": "2026-11-21T00:00:00Z",
            "vendors": self.records,
        }

    def __call__(self, url: str, *, headers=None, timeout=None, proxy=None) -> bytes:
        self.calls.append(url)
        if self.down:
            raise ConnectionError(f"down: {url}")
        if url == DIST + "manifest.json":
            return json.dumps(self.manifest()).encode()
        if url in self.files:
            return self.files[url]
        raise ConnectionError(f"no route: {url}")


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def at(self, level: int) -> list[str]:
        return [r.getMessage() for r in self.records if r.levelno == level]


@pytest.fixture
def logs(monkeypatch):
    # Each test announces its own installs.
    monkeypatch.setattr(shim_sources, "_ANNOUNCED_MOVED", set())
    handler = _Capture()
    names = ["llm-rosetta-gateway", "llm_rosetta_gateway.shims.sources"]
    saved = {}
    for name in names:
        logger = logging.getLogger(name)
        saved[name] = logger.level
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
    yield handler
    for name in names:
        logger = logging.getLogger(name)
        logger.removeHandler(handler)
        logger.setLevel(saved[name])


def _apply(tmp_path: Path, raw: dict[str, Any], http: FakeContrib):
    return apply_shim_sources(raw, str(tmp_path / "config.jsonc"), http_get=http)


OPENCODE = {"providers": {"go": {"type": "opencode--chat_completions", "api_key": "k"}}}


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


def test_every_moved_vendor_has_a_catalogue_source():
    assert set(moved_vendor_sources()) == {
        r.vendor for r in retired_names.MOVED_VENDORS
    }
    assert set(moved_vendor_sources().values()) == {"contrib"}


def test_no_moved_vendor_is_still_built_in():

    assert [
        s.name for s in load_providers() if retired_names.moved_vendor(s.name)
    ] == []


def test_the_references_cover_every_slot():
    raw = {
        "providers": {
            "go": {"type": "opencode--chat_completions"},
            "grok": {"type": "xai"},
            "plan": {
                "type": "openai--open_responses",
                "quota": {"probe": "openai_codex"},
            },
            "emb": {
                "type": "chat_completions",
                "embeddings_shim": "siliconflow--embedding",
            },
            "face": {
                "type": "openai--chat_completions",
                "rerank": {"shim": "zai--rerank"},
            },
            "off": {"shim": "kilo--chat_completions", "enabled": False},
            "openrouter--messages": {"api_key": "k"},
            "plain": {"type": "chat_completions", "base_url": "https://x"},
            "named": {"type": "deepseek--chat_completions"},
        },
        "rerank_providers": {"legacy": {"shim": "openrouter--rerank"}},
    }
    found = {(r.vendor, r.where) for r in moved_references(raw)}
    assert found == {
        ("opencode", "providers.go.type"),
        ("xai", "providers.grok.type"),
        ("openai_chatgpt", "providers.plan.quota.probe"),
        ("siliconflow", "providers.emb.embeddings_shim"),
        ("zai", "providers.face.rerank.shim"),
        ("kilo", "providers.off.shim"),
        ("openrouter", "providers.openrouter--messages"),
        ("openrouter", "rerank_providers.legacy.shim"),
    }


# ---------------------------------------------------------------------------
# Automatic install
# ---------------------------------------------------------------------------


class TestAutoInstall:
    def test_a_config_with_no_shims_section_installs_the_vendor(self, tmp_path, logs):
        assert get_shim("opencode--chat_completions") is None
        parsed = parse_shims_section(None, str(tmp_path / "c.jsonc"), document=OPENCODE)
        assert parsed and not parsed.declared
        assert parsed.auto_install == ("opencode",)
        assert [s.raw for s in parsed.manifests.selectors] == ["contrib:opencode"]

        report = _apply(tmp_path, OPENCODE, FakeContrib())

        status = next(s for s in report.sources if s.vendor == "opencode")
        assert status.ok and status.source == "contrib"
        assert get_shim("opencode--chat_completions") is not None
        assert any(
            "'opencode--chat_completions' (providers.go.type)" in m
            and "installing it from there" in m
            for m in logs.at(logging.INFO)
        )
        assert logs.at(logging.WARNING) == []

    def test_the_announcement_is_made_once_per_vendor(self, tmp_path, logs):
        http = FakeContrib()
        _apply(tmp_path, OPENCODE, http)
        _apply(tmp_path, OPENCODE, http)
        announced = [
            m for m in logs.at(logging.INFO) if "installing it from there" in m
        ]
        assert len(announced) == 1

    def test_the_gateway_config_builds_the_provider(self, tmp_path, logs):
        from llm_rosetta_gateway.config import GatewayConfig

        raw = json.loads(json.dumps(OPENCODE))
        _apply(tmp_path, raw, FakeContrib())
        config = GatewayConfig(raw)
        assert config.providers["go"].base_url == "https://opencode.ai/zen/go/v1"

    def test_a_bare_shim_name_installs_its_vendor(self, tmp_path, logs):
        raw = {"providers": {"grok": {"type": "xai", "api_key": "k"}}}
        _apply(tmp_path, raw, FakeContrib())
        assert get_shim("xai") is not None

    def test_a_shim_slot_installs_its_vendor(self, tmp_path, logs):
        from llm_rosetta_gateway.config import GatewayConfig

        raw = {
            "providers": {
                "kilo": {
                    "type": "chat_completions",
                    "shim": "kilo--chat_completions",
                    "api_key": "k",
                }
            }
        }
        parsed = parse_shims_section(None, None, document=raw)
        assert parsed.auto_install == ("kilo",)

        _apply(tmp_path, raw, FakeContrib())

        assert get_shim("kilo--chat_completions") is not None
        config = GatewayConfig(json.loads(json.dumps(raw)))
        assert config.provider_shim_names["kilo"] == "kilo--chat_completions"

    def test_a_typed_endpoint_shim_installs_its_vendor(self, tmp_path, logs):
        raw = {
            "providers": {
                "sf": {
                    "type": "chat_completions",
                    "base_url": "https://api.siliconflow.cn/v1",
                    "embeddings_shim": "siliconflow--embedding",
                }
            }
        }
        _apply(tmp_path, raw, FakeContrib())
        assert get_shim("siliconflow--embedding") is not None

    def test_no_sources_section_adds_the_contrib_source(self, tmp_path, logs):
        raw = dict(OPENCODE, shims={"install": ["argonne:alcf"]})
        parsed = parse_shims_section(raw["shims"], None, document=raw)
        assert [s.key for s in parsed.manifests.sources] == ["contrib", "argonne"]

    def test_explicit_sources_get_the_selector_but_not_the_source(self, tmp_path, logs):
        """A declared source list is the operator's egress choice."""
        raw = dict(OPENCODE, shims={"sources": ["argonne"], "install": []})
        parsed = parse_shims_section(raw["shims"], None, document=raw)
        assert [s.key for s in parsed.manifests.sources] == ["argonne"]
        assert [s.raw for s in parsed.manifests.selectors] == ["contrib:opencode"]
        assert parsed.auto_install == ("opencode",)

        _apply(tmp_path, raw, FakeContrib())  # serves contrib only

        assert get_shim("opencode--chat_completions") is None
        assert any(
            "The declared shims.sources do not include 'contrib'" in m
            for m in logs.at(logging.WARNING)
        )

    def test_a_declared_mirror_that_publishes_the_vendor_serves_it(
        self, tmp_path, logs
    ):
        # A URL source: its manifest calls itself "contrib", so the
        # ``contrib:<vendor>`` selector applies to it.
        raw = dict(OPENCODE, shims={"sources": [DIST + "manifest.json"]})
        parsed = parse_shims_section(raw["shims"], None, document=raw)
        assert [s.key for s in parsed.manifests.sources] == [None]

        _apply(tmp_path, raw, FakeContrib())

        assert get_shim("opencode--chat_completions") is not None

    def test_an_explicit_selector_is_not_doubled(self, tmp_path, logs):
        raw = dict(OPENCODE, shims={"install": ["opencode"]})
        parsed = parse_shims_section(raw["shims"], None, document=raw)
        assert parsed.auto_install == ()
        assert [s.raw for s in parsed.manifests.selectors] == ["opencode"]

    def test_a_wider_selector_that_covers_it_leaves_no_unmatched_warning(
        self, tmp_path, logs
    ):
        raw = dict(OPENCODE, shims={"install": ["contrib:*"]})
        report = _apply(tmp_path, raw, FakeContrib())
        assert get_shim("opencode--chat_completions") is not None
        assert not any("matched no vendor" in m for m in logs.at(logging.WARNING))
        assert not any(s.skipped for s in report.sources)

    def test_the_signature_follows_a_provider_that_names_a_moved_vendor(self):
        assert shims_signature({"providers": {}}) == ""
        first = shims_signature(OPENCODE)
        assert first != ""
        more = {"providers": dict(OPENCODE["providers"], g={"type": "xai"})}
        assert shims_signature(more) != first
        # A section without moved vendors keeps the hash it always had.
        section = {"shims": {"paths": ["x"]}}
        assert shims_signature(section) == shims_signature(
            {
                "shims": {"paths": ["x"]},
                "providers": {"d": {"type": "deepseek--chat_completions"}},
            }
        )

    def test_the_cli_applies_a_document_with_no_shims_section(
        self, tmp_path, monkeypatch
    ):
        from llm_rosetta_gateway import cli

        seen = []
        monkeypatch.setattr(
            cli, "apply_shim_sources", lambda raw, path: seen.append(raw) or "report"
        )
        assert cli._apply_shim_sources_or_exit(OPENCODE, "c.jsonc") == "report"
        assert cli._apply_shim_sources_or_exit({"providers": {}}, "c.jsonc") is None
        assert len(seen) == 1


#: The vendors contrib's v0.4.0 manifest listed (``dist`` commit d159945824,
#: generated 2026-11-21T11:59:52Z) — the copy jsDelivr kept serving.
STALE_CDN_VENDORS = ("kilo_free", "llm7", "opencode_zen_free", "opendesign", "ovhcloud")


def _stale_record(vendor: str) -> dict[str, Any]:
    return {
        "versions": [
            {
                "version": "0.4.0",
                "commit": COMMIT,
                "archive": f"0.4.0/{vendor}.tar.gz",
                "sha256": "0" * 64,
                "requires": {"surface": SURFACE_VERSION, "ir": IR_VERSION},
                "shims": [f"{vendor}--chat_completions"],
                "tags": ["tier:free"] if vendor != "opendesign" else [],
                "maturity": "beta",
            }
        ]
    }


class LaggingCdn(FakeContrib):
    """jsDelivr still serves the manifest from before the vendors moved.

    The measured failure (#1660): the CDN answered 200 with the previous
    manifest (older ``generated_at``, none of the moved vendors) while the raw
    ``dist`` branch on GitHub already served the new one.
    """

    RAW_DIST = FALLBACK_URL.rsplit("/", 1)[0] + "/"

    def __call__(self, url: str, *, headers=None, timeout=None, proxy=None) -> bytes:
        self.calls.append(url)
        if url == DIST + "manifest.json":
            stale = {**self.manifest(), "generated_at": "2026-11-21T11:59:52Z"}
            stale["vendors"] = {v: _stale_record(v) for v in STALE_CDN_VENDORS}
            return json.dumps(stale).encode()
        if url == self.RAW_DIST + "manifest.json":
            fresh = {**self.manifest(), "generated_at": "2026-11-21T16:19:40Z"}
            return json.dumps(fresh).encode()
        if url.startswith(self.RAW_DIST):
            return self.files[DIST + url[len(self.RAW_DIST) :]]
        raise ConnectionError(f"no route: {url}")


class TestALaggingCdn:
    def test_the_moved_vendor_installs_from_the_fresher_fallback(self, tmp_path, logs):
        http = LaggingCdn()

        report = _apply(tmp_path, OPENCODE, http)

        status = next(s for s in report.sources if s.vendor == "opencode")
        assert status.ok and status.archive_url.startswith(LaggingCdn.RAW_DIST)
        assert get_shim("opencode--chat_completions") is not None
        manifests = [c for c in http.calls if c.endswith("manifest.json")]
        assert manifests == [MANIFEST_URL, FALLBACK_URL]
        assert any(
            f"using {FALLBACK_URL}" in m and "does not list opencode" in m
            for m in logs.at(logging.INFO)
        )
        # The stale copy is the real one: five vendors, none of them moved.
        assert json.loads(http(DIST + "manifest.json"))["vendors"].keys() == set(
            STALE_CDN_VENDORS
        )


class TestTheProbeOnlyCase:
    RAW = {
        "providers": {
            "plan": {
                "type": "openai--open_responses",
                "api_key": "token",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "quota": {"probe": "openai_codex"},
            }
        }
    }

    def test_a_config_that_only_names_the_probe_installs_its_vendor(
        self, tmp_path, logs
    ):
        parsed = parse_shims_section(None, None, document=self.RAW)
        assert parsed.auto_install == ("openai_chatgpt",)
        _apply(tmp_path, self.RAW, FakeContrib())
        assert get_shim("openai_chatgpt--open_responses") is not None

    def test_the_probe_resolves_on_the_provider_that_names_it(self, tmp_path, logs):
        from llm_rosetta_gateway.providers import build_provider_info

        _apply(tmp_path, self.RAW, FakeContrib())
        cfg = self.RAW["providers"]["plan"]
        info = build_provider_info(
            "open_responses", dict(cfg), shim_name="openai--open_responses"
        )
        assert [p.id for p in info.quota_probes] == ["openai_codex"]

    def test_without_the_vendor_the_probe_is_missing_with_a_warning(self, logs):
        from llm_rosetta_gateway.providers import build_provider_info

        cfg = self.RAW["providers"]["plan"]
        info = build_provider_info(
            "open_responses", dict(cfg), shim_name="openai--open_responses"
        )
        assert info.quota_probes == ()
        assert any(
            "Quota probe 'openai_codex' moved from core" in m
            for m in logs.at(logging.WARNING)
        )


class TestAnUnreachableSourceOnAColdCache:
    def test_the_gateway_starts_and_names_what_is_missing(self, tmp_path, logs):
        http = FakeContrib()
        http.down = True

        report = _apply(tmp_path, OPENCODE, http)

        assert get_shim("opencode--chat_completions") is None
        assert report.failures(), "the source failure is reported"
        warnings = logs.at(logging.WARNING)
        assert any(
            "Shim vendor(s) opencode moved from core to the contrib shim source" in m
            and "'opencode--chat_completions' (providers.go.type)" in m
            for m in warnings
        ), warnings

    def test_the_provider_is_disabled_with_a_clear_reason(self, tmp_path, logs):
        from llm_rosetta_gateway.ops import OpsContext, OpsProviderValidation

        http = FakeContrib()
        http.down = True
        raw = json.loads(json.dumps(OPENCODE))
        raw["providers"]["ds"] = {"type": "deepseek--chat_completions", "api_key": "k"}
        report = _apply(tmp_path, raw, http)

        result = asyncio.run(
            OpsProviderValidation(OpsContext(), raw, report=report).execute()
        )

        (item,) = result["disabled"]
        assert item["provider"] == "go"
        assert item["category"] == "missing_external_shim"
        assert "moved from core to llm-rosetta-shim-contrib" in item["reason"]
        assert "source failures" in item["reason"]
        assert raw["providers"]["ds"].get("enabled", True) is True

    def test_a_shim_switched_off_is_blamed_on_the_switch(self, tmp_path, logs):
        """``shims.disabled`` withheld it: the source is not at fault."""
        from llm_rosetta_gateway.ops import OpsContext, OpsProviderValidation

        raw = {
            "providers": {"grok": {"type": "xai", "api_key": "k"}},
            "shims": {"disabled": ["xai"]},
        }
        report = _apply(tmp_path, raw, FakeContrib())
        assert next(s for s in report.sources if s.vendor == "xai").ok

        result = asyncio.run(
            OpsProviderValidation(OpsContext(), raw, report=report).execute()
        )

        (item,) = result["disabled"]
        assert item["reason"] == (
            "shim 'xai' is switched off by shims.disabled; the provider comes "
            "back when the shim is removed from that list"
        )

    def test_a_vendor_the_manifest_lacks_is_named_as_such(self, tmp_path, logs):
        from llm_rosetta_gateway.ops import OpsContext, OpsProviderValidation

        raw = json.loads(json.dumps(OPENCODE))
        report = _apply(tmp_path, raw, FakeContrib(vendors=("xai",)))

        result = asyncio.run(
            OpsProviderValidation(OpsContext(), raw, report=report).execute()
        )

        (item,) = result["disabled"]
        assert item["reason"].endswith(
            "and the gateway did not install it: not in the manifest"
        )
        assert "source answers" not in item["reason"]

    def test_a_vendor_lost_through_a_face_gets_the_move_wording(self, tmp_path, logs):
        """The face's endpoint is disabled with a reason that names the move."""
        from llm_rosetta_gateway.config import GatewayConfig

        http = FakeContrib()
        http.down = True
        raw = {
            "providers": {
                "sf": {
                    "type": "chat_completions",
                    "base_url": "https://api.siliconflow.cn/v1",
                    "embeddings_shim": "siliconflow--embedding",
                }
            }
        }
        _apply(tmp_path, raw, http)

        GatewayConfig(json.loads(json.dumps(raw)))

        (line,) = [m for m in logs.at(logging.WARNING) if "endpoint disabled" in m]
        assert "embeddings_shim names unknown shim 'siliconflow--embedding'" in line
        assert "moved in 0.14 to the shim repository llm-rosetta-shim-contrib" in line

    def test_the_auto_disabled_marker_is_not_read_as_a_face(self, tmp_path, logs):
        raw = json.loads(json.dumps(OPENCODE))
        raw["providers"]["go"]["_auto_disabled"] = {
            "reason": "r",
            "category": "missing_external_shim",
            "shim": "opencode--chat_completions",
        }
        assert [r.where for r in moved_references(raw)] == ["providers.go.type"]

        http = FakeContrib()
        http.down = True
        _apply(tmp_path, raw, http)

        (warning,) = [m for m in logs.at(logging.WARNING) if "Shim vendor(s)" in m]
        assert warning.count("'opencode--chat_completions'") == 1

    def test_a_manifest_without_the_vendor_warns_too(self, tmp_path, logs):
        report = _apply(tmp_path, OPENCODE, FakeContrib(vendors=("xai",)))
        assert get_shim("opencode--chat_completions") is None
        assert any(s.skipped == "not in the manifest" for s in report.sources)
        assert any("Shim vendor(s) opencode" in m for m in logs.at(logging.WARNING))


# ---------------------------------------------------------------------------
# llm-rosetta-migrate
# ---------------------------------------------------------------------------


class TestMigrate:
    def test_a_config_reports_moved_names_as_notes(self, tmp_path, capsys):
        from llm_rosetta_gateway.maintenance import migrate_0140 as mig

        path = tmp_path / "config.jsonc"
        path.write_text(
            json.dumps(
                {
                    "providers": {
                        "go": {"type": "opencode--chat_completions"},
                        "plan": {
                            "type": "openai--open_responses",
                            "quota": {"probe": "openai_codex"},
                        },
                        "grok": {"shim": "xai"},
                    }
                }
            )
        )

        code = mig.main(["--config", str(path), "--json"])

        out = json.loads(capsys.readouterr().out)
        assert code == 0
        assert out["items"] == []
        notes = {(n["where"], n["old"]) for n in out["notes"]}
        assert notes == {
            ("providers.go.type", "opencode--chat_completions"),
            ("providers.plan.quota.probe", "openai_codex"),
            ("providers.grok.shim", "xai"),
        }
        assert all("installs it automatically" in n["detail"] for n in out["notes"])

    def test_a_scan_reports_moved_names_as_notes(self, tmp_path, capsys):
        from llm_rosetta_gateway.maintenance import migrate_0140 as mig

        (tmp_path / "app.py").write_text(
            'TARGET = "openrouter--chat_completions"\nOTHER = "deepseek--chat_completions"\n'
        )

        code = mig.main(["--scan", str(tmp_path), "--json"])

        out = json.loads(capsys.readouterr().out)
        assert code == 0
        assert out["items"] == []
        assert [(n["line"], n["old"]) for n in out["notes"]] == [
            (1, "openrouter--chat_completions")
        ]


class TestScanFindsEveryMovedName:
    """``--scan`` reports bare moved names and probe ids, not only ``a--b``."""

    @staticmethod
    def _notes(tmp_path, name: str, text: str, capsys) -> list[tuple[int, str]]:
        from llm_rosetta_gateway.maintenance import migrate_0140 as mig

        (tmp_path / name).write_text(text)
        code = mig.main(["--scan", str(tmp_path / name), "--json"])
        out = json.loads(capsys.readouterr().out)
        assert code == 0 and out["items"] == []
        return [(n["line"], n["old"]) for n in out["notes"]]

    def test_a_bare_type_value(self, tmp_path, capsys):
        text = '{\n  "providers": {"grok": {"type": "xai"}}\n}\n'
        assert self._notes(tmp_path, "a.jsonc", text, capsys) == [(2, "xai")]

    def test_a_provider_named_after_a_bare_shim(self, tmp_path, capsys):
        text = '{\n  "providers": {\n    "qwen": {"api_key": "k"}\n  }\n}\n'
        assert self._notes(tmp_path, "b.json", text, capsys) == [(3, "qwen")]

    def test_an_install_entry(self, tmp_path, capsys):
        text = '{\n  "shims": {"install": [\n    "zhipu",\n    "contrib:moonshot"\n  ]}\n}\n'
        assert self._notes(tmp_path, "c.json", text, capsys) == [
            (3, "zhipu"),
            (4, "moonshot"),
        ]

    def test_a_quota_probe(self, tmp_path, capsys):
        text = 'PROBE = "openai_codex"\nOTHER = "opencode_go"\n'
        assert self._notes(tmp_path, "d.py", text, capsys) == [
            (1, "openai_codex"),
            (2, "opencode_go"),
        ]

    def test_a_yaml_shim_slot(self, tmp_path, capsys):
        text = "providers:\n  grok:\n    shim: xai\n    embeddings_shim: 'zhipu'\n"
        assert self._notes(tmp_path, "e.yaml", text, capsys) == [
            (3, "xai"),
            (4, "zhipu"),
        ]

    def test_a_bare_word_in_ordinary_code_is_not_reported(self, tmp_path, capsys):
        text = 'VENDOR = "xai"\nprint("qwen")\n'
        assert self._notes(tmp_path, "f.py", text, capsys) == []
