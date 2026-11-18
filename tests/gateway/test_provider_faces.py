"""One provider entry, several faces: the account is shared, the host is not (#1595).

A ``providers:`` entry is an account — one key ring, one mint, one refresh —
and each typed endpoint is a face of it.  Before the face block the endpoint
took the entry's ``base_url`` whatever its shim said, so an account signing one
vendor for chat and another for embeddings sent every embedding to the chat
vendor's host, silently.

These cases pin the block, the precedence it reads with, the mismatch the
runtime now reports, the v4→v5 migration that folds the flat keys into it, and
the admin payload that folds them for the panel.
"""

from __future__ import annotations

import json
import logging

import pytest

from llm_rosetta_gateway.config.core import GatewayConfig
from llm_rosetta_gateway.config.typed import face_section, face_value
from llm_rosetta_gateway.maintenance.migrations import CURRENT_VERSION, migrate


class _CollectingHandler(logging.Handler):
    """Keeps the gateway's own log records, whatever the run imported first."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def gateway_warnings():
    """Records the gateway logger emits during the test.

    That logger is created with ``propagate = False``, so caplog — which
    listens on the root logger — only sees its records while
    ``llm_rosetta_gateway.logging`` has not been imported yet in the process,
    and whether it has been is decided by the rest of the run.  A handler on
    the logger that emits is one path under every pytest version.
    """
    logger = logging.getLogger("llm-rosetta-gateway")
    handler = _CollectingHandler()
    logger.addHandler(handler)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)


def _config(entry: dict, *, models: dict | None = None) -> GatewayConfig:
    return GatewayConfig({"providers": {"mix": entry}, "models": models or {}})


class TestTheFaceHost:
    def test_an_endpoint_on_another_vendor_uses_its_own_host(self):
        cfg = _config(
            {
                "type": "google_generate",
                "base_url": "https://generativelanguage.googleapis.com",
                "api_key": "k",
                "embeddings": {
                    "shim": "cohere--embedding",
                    "base_url": "https://api.cohere.com/v2",
                },
            }
        )

        assert cfg.embeddings_provider_infos["mix"].base_url == "https://api.cohere.com"
        assert (
            cfg.providers["mix"].base_url == "https://generativelanguage.googleapis.com"
        )

    def test_an_endpoint_without_a_host_inherits_the_entry(self):
        cfg = _config(
            {
                "type": "google_generate",
                "base_url": "https://generativelanguage.googleapis.com",
                "api_key": "k",
                "embeddings": {"shim": "google--embedding"},
            }
        )

        assert (
            cfg.embeddings_provider_infos["mix"].base_url
            == "https://generativelanguage.googleapis.com"
        )

    def test_a_trailing_slash_is_normalised_away(self):
        cfg = _config(
            {
                "type": "chat_completions",
                "base_url": "https://chat.example/v1",
                "api_key": "k",
                "rerank": {"shim": "jina--rerank", "base_url": "https://api.jina.ai/"},
            }
        )

        assert cfg.rerank_provider_infos["mix"].base_url == "https://api.jina.ai"


class TestTheMismatchIsReported:
    def test_an_inherited_host_that_belongs_to_another_vendor_warns(
        self, gateway_warnings
    ):
        _config(
            {
                "type": "chat_completions",
                "base_url": "https://chat.example/v1",
                "api_key": "k",
                "embeddings_shim": "cohere--embedding",
            }
        )

        messages = [record.getMessage() for record in gateway_warnings]
        assert any("inherits base_url" in message for message in messages), messages
        assert any("embeddings: {base_url" in message for message in messages), messages

    def test_the_face_host_silences_the_warning(self, gateway_warnings):
        _config(
            {
                "type": "chat_completions",
                "base_url": "https://chat.example/v1",
                "api_key": "k",
                "embeddings": {
                    "shim": "cohere--embedding",
                    "base_url": "https://api.cohere.com",
                },
            }
        )

        assert not [
            record
            for record in gateway_warnings
            if "inherits base_url" in record.getMessage()
        ]

    def test_a_shim_on_the_same_host_does_not_warn(self, gateway_warnings):
        """The vendor's own two lanes differ by region, not by vendor."""
        _config(
            {
                "type": "chat_completions",
                "base_url": "https://generativelanguage.googleapis.com",
                "api_key": "k",
                "embeddings_shim": "google--embedding",
            }
        )

        assert not [
            record
            for record in gateway_warnings
            if "inherits base_url" in record.getMessage()
        ]


class TestTheFaceReader:
    def test_the_block_wins_over_the_flat_key(self):
        cfg = {"embeddings_shim": "flat--a", "embeddings": {"shim": "block--b"}}

        assert face_value(cfg, "embeddings_shim", "embeddings") == "block--b"

    def test_an_empty_block_value_falls_through(self):
        cfg = {"embeddings_shim": "flat--a", "embeddings": {"shim": ""}}

        assert face_value(cfg, "embeddings_shim", "embeddings") == "flat--a"

    def test_enabled_false_in_the_block_is_a_value(self):
        cfg = {"embeddings_enabled": True, "embeddings": {"enabled": False}}

        assert face_value(cfg, "embeddings_enabled", "embeddings") is False

    def test_a_flat_key_answers_when_the_block_is_absent(self):
        cfg = {"rerank_path": "/v1/rerank"}

        assert face_value(cfg, "rerank_path", "rerank") == "/v1/rerank"

    def test_a_block_that_is_not_a_mapping_is_ignored(self, gateway_warnings):
        assert face_section({"embeddings": "cohere"}, "embeddings") == {}

        assert any(
            "must be a mapping" in record.getMessage() for record in gateway_warnings
        )

    def test_an_endpoint_block_off_disables_only_that_face(self):
        cfg = _config(
            {
                "type": "chat_completions",
                "base_url": "https://chat.example/v1",
                "api_key": "k",
                "embeddings": {"shim": "cohere--embedding", "enabled": False},
            }
        )

        assert "mix" not in cfg.embeddings_provider_infos
        assert "mix" in cfg.providers


class TestTheMigration:
    def test_flat_keys_become_the_face_block(self):
        raw = {
            "providers": {
                "mix": {
                    "type": "google_generate",
                    "base_url": "https://gl.example",
                    "api_key": "k",
                    "embeddings_shim": "cohere--embedding",
                    "embeddings_format": "cohere",
                    "embeddings_path": "/v2/embed",
                    "embeddings_enabled": False,
                }
            },
            "models": {},
        }

        migrate(raw)

        entry = raw["providers"]["mix"]
        assert entry["embeddings"] == {
            "shim": "cohere--embedding",
            "format": "cohere",
            "path": "/v2/embed",
            "enabled": False,
        }
        assert "embeddings_shim" not in entry
        assert raw["config_version"] == CURRENT_VERSION

    def test_an_empty_flat_value_moves_too(self):
        """The form writes ``""`` to clear a setting; both spellings must agree."""
        raw = {
            "providers": {
                "p": {"api_key": "k", "base_url": "https://x", "rerank_path": ""}
            },
            "models": {},
        }

        migrate(raw)

        assert raw["providers"]["p"]["rerank"] == {"path": ""}
        assert "rerank_path" not in raw["providers"]["p"]

    def test_an_existing_block_is_not_overwritten(self):
        raw = {
            "providers": {
                "p": {
                    "base_url": "https://x",
                    "api_key": "k",
                    "embeddings": {"shim": "kept--embedding"},
                    "embeddings_shim": "flat--embedding",
                }
            },
            "models": {},
        }

        migrate(raw)

        entry = raw["providers"]["p"]
        assert entry["embeddings"]["shim"] == "kept--embedding"
        assert "embeddings_shim" not in entry

    def test_an_entry_without_face_keys_is_untouched(self):
        raw = {
            "providers": {"p": {"base_url": "https://x", "api_key": "k"}},
            "models": {},
        }
        before = json.dumps(raw, sort_keys=True)

        migrate(raw)

        assert "embeddings" not in raw["providers"]["p"]
        assert json.dumps(raw["providers"], sort_keys=True) == json.dumps(
            json.loads(before)["providers"], sort_keys=True
        )

    def test_the_legacy_section_is_left_alone(self):
        """``<type>_providers`` is one face already; its keys are its own."""
        raw = {
            "config_version": 0,
            "providers": {},
            "models": {},
            "embeddings_providers": {
                "jina": {
                    "base_url": "https://api.jina.ai",
                    "api_key": "k",
                    "format": "jina",
                }
            },
        }

        migrate(raw)

        section = raw["embeddings_providers"]["jina"]
        assert section["format"] == "jina"
        assert "embeddings" not in section


class TestTheSave:
    def test_the_face_block_reaches_the_file(self, tmp_path):
        """The editor's block is stored as sent, not dropped by the whitelist."""
        from llm_rosetta_gateway.admin.routes._shared import (
            _apply_face_blocks,
            _build_provider_entry,
        )

        body = {
            "base_url": "https://generativelanguage.googleapis.com",
            "embeddings": {
                "shim": "cohere--embedding",
                "format": "cohere",
                "path": "/v2/embed",
                "base_url": "https://api.cohere.com/v2",
                "enabled": False,
            },
        }

        built = _build_provider_entry(body, "g-key", body["base_url"], {}, "fresh")
        _apply_face_blocks(built, body)

        assert built["embeddings"] == {
            "shim": "cohere--embedding",
            "format": "cohere",
            "path": "/v2/embed",
            "base_url": "https://api.cohere.com/v2",
            "enabled": False,
        }

    def test_an_empty_block_is_removed(self):
        from llm_rosetta_gateway.admin.routes._shared import _apply_face_blocks

        entry = {"embeddings": {"shim": "old--embedding"}}
        _apply_face_blocks(
            entry, {"embeddings": {"shim": "", "format": "", "path": ""}}
        )

        assert "embeddings" not in entry

    def test_a_capability_the_provider_dropped_leaves_no_block(self):
        from llm_rosetta_gateway.admin.routes._shared import _apply_face_blocks

        entry = {"embeddings": {"shim": "x--embedding"}}
        _apply_face_blocks(entry, {"embeddings": {}})

        assert "embeddings" not in entry

    def test_a_block_that_is_not_a_mapping_is_not_stored(self):
        from llm_rosetta_gateway.admin.routes._shared import _apply_face_blocks

        entry: dict = {}
        _apply_face_blocks(entry, {"embeddings": "cohere--embedding"})

        assert "embeddings" not in entry


class TestTheAdminPayload:
    def test_a_flat_entry_is_folded_into_a_block_for_the_panel(self):
        from llm_rosetta_gateway.admin.routes.config.payload import _mask_provider_entry

        masked = _mask_provider_entry(
            "mix",
            {
                "type": "google_generate",
                "base_url": "https://gl.example",
                "api_key": "secret-key",
                "embeddings_shim": "cohere--embedding",
                "embeddings_enabled": False,
            },
        )

        assert masked["embeddings"] == {
            "shim": "cohere--embedding",
            "enabled": False,
        }
        assert "secret-key" not in json.dumps(masked)

    def test_an_entry_that_already_nests_is_left_as_written(self):
        from llm_rosetta_gateway.admin.routes.config.payload import _mask_provider_entry

        entry = {
            "type": "google_generate",
            "base_url": "https://gl.example",
            "embeddings": {
                "shim": "cohere--embedding",
                "base_url": "https://api.cohere.com",
            },
        }

        masked = _mask_provider_entry("mix", entry)

        assert masked["embeddings"] == entry["embeddings"]


@pytest.mark.parametrize("type_name", ["embeddings", "rerank", "decisions"])
def test_every_typed_endpoint_reads_its_block(type_name):
    """The panel and the parser share one field name per endpoint."""
    from llm_rosetta_gateway.config.typed import _typed_descriptors

    descriptors = {desc.name: desc for desc in _typed_descriptors()}
    desc = descriptors[type_name]
    cfg = {type_name: {"shim": "x--y", "path": "/p"}}

    assert face_value(cfg, desc.config_shim_key, type_name) == "x--y"
    assert face_value(cfg, desc.config_path_key, type_name) == "/p"
