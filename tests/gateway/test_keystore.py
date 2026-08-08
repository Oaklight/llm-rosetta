"""Tests for gateway.keystore — SQLite-backed API key storage."""

from __future__ import annotations


import pytest

from llm_rosetta.gateway.keystore import KeyContext, KeyStore


@pytest.fixture()
def keystore(tmp_path):
    ks = KeyStore(tmp_path / "keys.db")
    yield ks
    ks.close()


class TestKeyStoreCreate:
    def test_create_returns_id_and_raw_key(self, keystore):
        key_id, raw_key = keystore.create(label="test")
        assert len(key_id) == 8
        assert raw_key.startswith("rsk-")

    def test_create_with_manual_key(self, keystore):
        key_id, raw_key = keystore.create(label="manual", manual_key="my-secret-key")
        assert raw_key == "my-secret-key"

    def test_create_with_allowed_shims(self, keystore):
        key_id, raw_key = keystore.create(
            label="limited", allowed_shims=["openai", "anthropic"]
        )
        ctx = keystore.validate(raw_key)
        assert ctx is not None
        assert ctx.allowed_shims == frozenset({"openai", "anthropic"})

    def test_default_allowed_shims_is_star(self, keystore):
        _, raw_key = keystore.create(label="default")
        ctx = keystore.validate(raw_key)
        assert ctx is not None
        assert ctx.allowed_shims == frozenset({"*"})


class TestKeyStoreValidate:
    def test_validate_valid_key(self, keystore):
        _, raw_key = keystore.create(label="valid")
        ctx = keystore.validate(raw_key)
        assert ctx is not None
        assert isinstance(ctx, KeyContext)
        assert ctx.label == "valid"

    def test_validate_invalid_key(self, keystore):
        keystore.create(label="exists")
        assert keystore.validate("wrong-key") is None

    def test_validate_empty_store(self, keystore):
        assert keystore.validate("any-key") is None


class TestKeyStoreList:
    def test_list_returns_no_secrets(self, keystore):
        keystore.create(label="a")
        keystore.create(label="b")
        keys = keystore.list_keys()
        assert len(keys) == 2
        for k in keys:
            assert "key" not in k
            assert "key_hash" not in k
            assert "id" in k
            assert "label" in k
            assert "allowed_shims" in k
            assert "created" in k

    def test_list_empty(self, keystore):
        assert keystore.list_keys() == []


class TestKeyStoreUpdate:
    def test_update_label(self, keystore):
        key_id, raw_key = keystore.create(label="old")
        assert keystore.update(key_id, label="new")
        ctx = keystore.validate(raw_key)
        assert ctx is not None
        assert ctx.label == "new"

    def test_update_allowed_shims(self, keystore):
        key_id, raw_key = keystore.create(label="x")
        assert keystore.update(key_id, allowed_shims=["google"])
        ctx = keystore.validate(raw_key)
        assert ctx is not None
        assert ctx.allowed_shims == frozenset({"google"})

    def test_update_nonexistent(self, keystore):
        assert not keystore.update("nonexistent", label="x")

    def test_update_nothing(self, keystore):
        key_id, _ = keystore.create(label="y")
        assert keystore.update(key_id)


class TestKeyStoreDelete:
    def test_delete_existing(self, keystore):
        key_id, raw_key = keystore.create(label="del")
        assert keystore.delete(key_id)
        assert keystore.validate(raw_key) is None

    def test_delete_nonexistent(self, keystore):
        assert not keystore.delete("nonexistent")

    def test_has_keys_after_delete(self, keystore):
        key_id, _ = keystore.create(label="only")
        assert keystore.has_keys()
        keystore.delete(key_id)
        assert not keystore.has_keys()


class TestKeyStoreRotate:
    def test_rotate_returns_new_key(self, keystore):
        key_id, old_key = keystore.create(label="rotate")
        new_key = keystore.rotate(key_id)
        assert new_key is not None
        assert new_key != old_key
        assert new_key.startswith("rsk-")

    def test_rotate_invalidates_old_key(self, keystore):
        key_id, old_key = keystore.create(label="rotate")
        keystore.rotate(key_id)
        assert keystore.validate(old_key) is None

    def test_rotate_new_key_validates(self, keystore):
        key_id, _ = keystore.create(label="rotate")
        new_key = keystore.rotate(key_id)
        ctx = keystore.validate(new_key)
        assert ctx is not None
        assert ctx.label == "rotate"

    def test_rotate_sets_rotated_timestamp(self, keystore):
        key_id, _ = keystore.create(label="ts")
        keystore.rotate(key_id)
        keys = keystore.list_keys()
        entry = next(k for k in keys if k["id"] == key_id)
        assert entry.get("rotated") is not None

    def test_rotate_nonexistent(self, keystore):
        assert keystore.rotate("nonexistent") is None


class TestKeyStoreImport:
    def test_import_from_config(self, keystore):
        config_keys = [
            {"id": "k1", "key": "secret-1", "label": "first", "created": "2024-01-01"},
            {"id": "k2", "key": "secret-2", "label": "second", "created": "2024-01-02"},
        ]
        imported = keystore.import_from_config(config_keys)
        assert imported == 2
        assert keystore.validate("secret-1") is not None
        assert keystore.validate("secret-2") is not None

    def test_import_idempotent(self, keystore):
        config_keys = [
            {"id": "k1", "key": "secret-1", "label": "first", "created": "2024-01-01"},
        ]
        assert keystore.import_from_config(config_keys) == 1
        assert keystore.import_from_config(config_keys) == 0

    def test_import_preserves_label(self, keystore):
        config_keys = [
            {"id": "k1", "key": "secret-1", "label": "mylab", "created": "2024-01-01"},
        ]
        keystore.import_from_config(config_keys)
        ctx = keystore.validate("secret-1")
        assert ctx is not None
        assert ctx.label == "mylab"

    def test_import_skips_empty_keys(self, keystore):
        config_keys = [{"id": "k1", "key": "", "label": "empty", "created": ""}]
        assert keystore.import_from_config(config_keys) == 0


class TestKeyStoreHasKeys:
    def test_has_keys_empty(self, keystore):
        assert not keystore.has_keys()

    def test_has_keys_with_key(self, keystore):
        keystore.create(label="x")
        assert keystore.has_keys()


class TestKeyContext:
    def test_frozen(self):
        ctx = KeyContext(label="test", allowed_shims=frozenset({"*"}))
        with pytest.raises(AttributeError):
            ctx.label = "changed"  # type: ignore

    def test_equality(self):
        a = KeyContext(label="a", allowed_shims=frozenset({"*"}))
        b = KeyContext(label="a", allowed_shims=frozenset({"*"}))
        assert a == b


class TestKeyStoreWAL:
    def test_wal_mode(self, tmp_path):
        ks = KeyStore(tmp_path / "test.db")
        import sqlite3

        conn = sqlite3.connect(str(tmp_path / "test.db"))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        ks.close()
        assert mode == "wal"
