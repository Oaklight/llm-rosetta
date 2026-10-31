"""Reading a config document brings it forward, whichever reader is used (#1046).

The load-time steps used to run in exactly one place — ``JsoncConfigIO``, which
only the admin reload path constructs — so ``load_config``, the documented way
to read a config and the CLI's own start path, handed back whatever had been
written to disk years ago.

What that cost, measured on a ``config_version: 0`` document carrying the legacy
rerank and embedding pools beside ``providers``:

    through the bare parser          through a reader
    providers      ['alcf']          providers      ['alcf', 'emb', 'jina']
    rerank_providers    ['jina']     rerank_providers    ['jina']
    embedding_providers ['emb']      embedding_providers ['emb']
    config_version on disk: 0        config_version on disk: 2

``GatewayConfig`` re-parses both legacy pool sections itself, so rerank and
embedding routing worked either way and nothing was unroutable; this file's
docstring said otherwise once, and the measurement is the correction.  What was
actually missing is those providers in the **unified** ``providers`` map — what
auth, rate limiting and the admin's provider listing read — and the file, which
stayed at version 0 until somebody opened the admin panel.  That read then
rewrote the document and left a ``.pre-migration.bak`` the operator never asked
for.

These tests pin the two readers and the startup path, and the properties the
write-back has to keep while doing it.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pytest

from llm_rosetta_gateway import config as config_module
from llm_rosetta_gateway.maintenance import config as config_maintenance
from llm_rosetta_gateway.maintenance.migrations import CURRENT_VERSION
from llm_rosetta_gateway.cli import _load_startup_config
from llm_rosetta_gateway.config import (
    GatewayConfig,
    JsoncConfigIO,
    load_config,
    load_config_raw,
    load_gateway_config,
)
from llm_rosetta_gateway.maintenance.config import ConfigMigrationError

#: A document as an older release wrote it: version 0, the legacy pools beside
#: ``providers``, and a name that has since been renamed.
_V0 = {
    "providers": {
        "alcf": {"type": "openai_chat", "api_key": "k", "base_url": "https://a"}
    },
    "models": {"m": {"provider": "alcf", "model": "m"}},
    "rerank_providers": {"jina": {"api_key": "k", "base_url": "https://j"}},
    "rerank_models": {"rerank-v1": "jina"},
}


def _write(path: Path, document: dict) -> Path:
    path.write_text(json.dumps(document))
    return path


def _args(**overrides: str | None) -> argparse.Namespace:
    """The CLI options ``_load_startup_config`` reads, all unset by default."""
    options: dict[str, str | None] = {
        "proxy": None,
        "data_dir": None,
        "tls_cert": None,
        "tls_key": None,
    }
    options.update(overrides)
    return argparse.Namespace(**options)


class TestThePublicReader:
    def test_load_config_merges_what_only_the_migration_knows(self, tmp_path):
        """The bug as reported: the rerank pool was dropped on the way in."""
        path = _write(tmp_path / "gateway.jsonc", _V0)

        raw = load_config(str(path))

        assert set(raw["providers"]) == {"alcf", "jina"}
        assert set(raw["models"]) == {"m", "rerank-v1"}
        assert raw["providers"]["alcf"]["type"] == "chat_completions", (
            "the name correction ran too"
        )

    def test_it_leaves_the_file_at_the_current_version(self, tmp_path):
        path = _write(tmp_path / "gateway.jsonc", _V0)

        load_config(str(path))

        written = json.loads(path.read_text())
        assert written["config_version"] == CURRENT_VERSION
        assert "rerank_providers" not in written
        assert (tmp_path / "gateway.jsonc.pre-migration.bak").exists(), (
            "the operator's original is kept, as on the admin path"
        )

    def test_a_current_document_is_not_rewritten(self, tmp_path):
        path = _write(tmp_path / "gateway.jsonc", _V0)
        load_config(str(path))
        before = path.read_bytes()
        (tmp_path / "gateway.jsonc.pre-migration.bak").unlink()

        load_config(str(path))

        assert path.read_bytes() == before
        assert not (tmp_path / "gateway.jsonc.pre-migration.bak").exists()

    def test_load_config_raw_migrates_and_keeps_placeholders(
        self, tmp_path, monkeypatch
    ):
        """It is the edit round-trip reader: migrate, but do not bake a secret in."""
        monkeypatch.setenv("ALCF_KEY", "sk-live-value")
        document = json.loads(json.dumps(_V0))
        document["providers"]["alcf"]["api_key"] = "${ALCF_KEY}"
        path = _write(tmp_path / "gateway.jsonc", document)

        raw = load_config_raw(str(path))

        assert raw["providers"]["alcf"]["api_key"] == "${ALCF_KEY}", (
            "the caller gets the placeholder, so writing it back cannot leak the value"
        )
        assert set(raw["providers"]) == {"alcf", "jina"}, "and it is still migrated"
        assert "${ALCF_KEY}" in path.read_text()

    def test_a_config_that_cannot_be_brought_forward_has_nothing_to_serve(
        self, tmp_path, monkeypatch
    ):
        """LOAD_TIME: the step's own failure is the load's failure."""
        path = _write(tmp_path / "gateway.jsonc", _V0)

        def _explode(raw, **_kwargs):
            raise ValueError("the config is not what this code expects")

        monkeypatch.setattr(config_maintenance, "apply", _explode)

        with pytest.raises(ValueError, match="not what this code expects"):
            load_config(str(path))

    def test_a_document_the_step_cannot_read_names_that_step(self, tmp_path):
        """The refusal is the policy's; the message has to say where it came from.

        The exception out of a migration body is about the shape it found, not
        about which document it found it in, so the load wraps it.
        """
        path = _write(
            tmp_path / "gateway.jsonc",
            {"providers": {"alcf": {"type": "openai_chat"}}, "rerank_providers": []},
        )

        with pytest.raises(ConfigMigrationError) as caught:
            load_config(str(path))

        assert caught.value.step == "config.migrate_structure"
        assert caught.value.path == str(path)
        assert str(path) in str(caught.value)
        assert isinstance(caught.value.__cause__, AttributeError)

    def test_a_read_only_directory_still_serves_the_migrated_document(
        self, tmp_path, caplog
    ):
        """A mounted read-only config is not a refused start.

        The steps have run and their result is the document this load returns;
        the write-back is bookkeeping, so a directory that cannot be written is
        logged and the document is served from memory.
        """
        path = _write(tmp_path / "gateway.jsonc", _V0)
        original = path.read_text()
        tmp_path.chmod(0o555)
        # The gateway logger does not propagate, so it gets caplog's handler.
        gateway_logger = logging.getLogger("llm-rosetta-gateway")
        gateway_logger.addHandler(caplog.handler)
        try:
            raw = load_config(str(path))
            on_disk = path.read_text()
            backup = (tmp_path / "gateway.jsonc.pre-migration.bak").exists()
        finally:
            gateway_logger.removeHandler(caplog.handler)
            tmp_path.chmod(0o755)

        assert set(raw["providers"]) == {"alcf", "jina"}, (
            "the served document is the migrated one either way"
        )
        assert on_disk == original, "and the file is left exactly as it was"
        assert backup is False
        assert "could not be written back" in caplog.text


class TestTheParsePath:
    """Plain JSON takes the stdlib parser; only the JSONC syntax pays for scanning."""

    def test_a_commented_document_still_parses(self, tmp_path):
        """The fallback is what makes a JSONC document with comments readable."""
        text = (
            "{\n"
            "  // the ledger this document carries\n"
            '  "config_version": 3,\n'
            '  "providers": {"alcf": {"type": "chat_completions"}},\n'
            "}\n"
        )
        path = tmp_path / "gateway.jsonc"
        path.write_text(text)

        raw = load_config(str(path))

        assert raw["providers"]["alcf"]["type"] == "chat_completions"
        assert path.read_text() == text, (
            "nothing to migrate, so the comments are still there"
        )

    def test_a_plain_document_never_reaches_the_jsonc_scanner(
        self, tmp_path, monkeypatch
    ):
        """What our own writer produces must not pay for comment stripping."""
        path = _write(tmp_path / "gateway.jsonc", _V0)

        def _explode(text):
            raise AssertionError("a plain document does not need the JSONC parser")

        monkeypatch.setattr(config_module.jsonx, "loads", _explode)

        assert load_config(str(path))["providers"]["alcf"]["type"] == "chat_completions"


class TestTheStartPath:
    """``cli.py`` used the bare parser, which is the whole of #1046."""

    def test_the_document_it_serves_is_the_migrated_one(self, tmp_path):
        path = _write(tmp_path / "gateway.jsonc", _V0)

        config, report = _load_startup_config(str(path), _args())

        assert set(config.providers) == {"alcf", "jina"}, (
            "the legacy pool reaches the provider map, which is where the unmigrated "
            "document lost it"
        )
        assert config.resolve_rerank("rerank-v1").provider_name == "jina"
        assert report is None, "no shims section, so nothing to report"
        assert json.loads(path.read_text())["config_version"] == CURRENT_VERSION

    def test_the_config_carries_the_registry_the_load_ran(self, tmp_path):
        """So the admin surface lists the load-time set for a served config."""
        path = _write(tmp_path / "gateway.jsonc", _V0)

        config, _report = _load_startup_config(str(path), _args())

        assert config.maintenance is not None
        assert config.maintenance.name == "config"
        changed = {r.name: r.changed for r in config.maintenance.results}
        assert changed == {
            "config.migrate_structure": 1,
            "config.correct_legacy_names": 1,
        }

    def test_the_cli_overrides_still_land_on_top(self, tmp_path):
        """They are applied to the migrated document, not instead of migrating."""
        path = _write(tmp_path / "gateway.jsonc", _V0)

        config, _report = _load_startup_config(
            str(path), _args(proxy="http://proxy.test", data_dir="/tmp/data")
        )

        assert config.proxy == "http://proxy.test"
        assert config.data_dir == "/tmp/data"
        assert set(config.providers) == {"alcf", "jina"}


class TestTheOneCallThatKeepsBoth:
    def test_load_gateway_config_returns_a_config_with_its_registry(self, tmp_path):
        path = _write(tmp_path / "gateway.jsonc", _V0)

        config = load_gateway_config(str(path))

        assert isinstance(config, GatewayConfig)
        assert set(config.providers) == {"alcf", "jina"}
        assert config.maintenance is not None
        assert [r.name for r in config.maintenance.results] == [
            "config.migrate_structure",
            "config.correct_legacy_names",
        ]

    def test_a_config_built_by_hand_still_reports_nothing(self, tmp_path):
        """The distinction the slice kept: ran steps vs. has a registry for them."""
        path = _write(tmp_path / "gateway.jsonc", _V0)
        reader = JsoncConfigIO()
        raw = reader.load(str(path))

        assert GatewayConfig(raw).maintenance is None
        assert (
            GatewayConfig(raw, maintenance=reader.last_load_registry).maintenance
            is not None
        )
