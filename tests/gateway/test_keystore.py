"""Tests for gateway.keystore — async SQLite-backed API key storage."""

from __future__ import annotations

import pytest

from llm_rosetta.gateway.keystore import KeyContext, KeyStore


@pytest.fixture()
async def keystore(tmp_path):
    ks = await KeyStore.create(tmp_path / "keys.db")
    yield ks
    await ks.close()


class TestKeyStoreCreate:
    async def test_create_returns_id_and_raw_key(self, keystore):
        key_id, raw_key = await keystore.create_key(label="test")
        assert len(key_id) == 8
        assert raw_key.startswith("rsk-")

    async def test_create_with_manual_key(self, keystore):
        key_id, raw_key = await keystore.create_key(
            label="manual", manual_key="my-secret-key"
        )
        assert raw_key == "my-secret-key"

    async def test_create_with_allowed_shims(self, keystore):
        key_id, raw_key = await keystore.create_key(
            label="limited", allowed_shims=["openai", "anthropic"]
        )
        result = keystore.validate(raw_key)
        assert result is not None
        _, ctx = result
        assert ctx.allowed_shims == frozenset({"openai", "anthropic"})

    async def test_default_allowed_shims_is_star(self, keystore):
        _, raw_key = await keystore.create_key(label="default")
        result = keystore.validate(raw_key)
        assert result is not None
        _, ctx = result
        assert ctx.allowed_shims == frozenset({"*"})


class TestKeyStoreValidate:
    async def test_validate_valid_key(self, keystore):
        _, raw_key = await keystore.create_key(label="valid")
        result = keystore.validate(raw_key)
        assert result is not None
        _, ctx = result
        assert ctx.label == "valid"

    async def test_validate_invalid_key(self, keystore):
        await keystore.create_key(label="exists")
        assert keystore.validate("wrong-key") is None

    async def test_validate_empty_store(self, keystore):
        assert keystore.validate("any-key") is None


class TestKeyStoreList:
    async def test_list_returns_no_secrets(self, keystore):
        await keystore.create_key(label="a")
        await keystore.create_key(label="b")
        keys = await keystore.list_keys()
        assert len(keys) == 2
        for k in keys:
            assert "key" not in k
            assert "key_hash" not in k
            assert "id" in k
            assert "label" in k
            assert "allowed_shims" in k
            assert "created" in k

    async def test_list_empty(self, keystore):
        assert await keystore.list_keys() == []


class TestKeyStoreUpdate:
    async def test_update_label(self, keystore):
        key_id, raw_key = await keystore.create_key(label="old")
        assert await keystore.update(key_id, label="new")
        result = keystore.validate(raw_key)
        assert result is not None
        _, ctx = result
        assert ctx.label == "new"

    async def test_update_allowed_shims(self, keystore):
        key_id, raw_key = await keystore.create_key(label="x")
        assert await keystore.update(key_id, allowed_shims=["google"])
        result = keystore.validate(raw_key)
        assert result is not None
        _, ctx = result
        assert ctx.allowed_shims == frozenset({"google"})

    async def test_update_nonexistent(self, keystore):
        assert not await keystore.update("nonexistent", label="x")

    async def test_update_nothing(self, keystore):
        key_id, _ = await keystore.create_key(label="y")
        assert await keystore.update(key_id)


class TestKeyStoreDelete:
    async def test_delete_existing(self, keystore):
        key_id, raw_key = await keystore.create_key(label="del")
        assert await keystore.delete(key_id)
        assert keystore.validate(raw_key) is None

    async def test_delete_nonexistent(self, keystore):
        assert not await keystore.delete("nonexistent")

    async def test_has_keys_after_delete(self, keystore):
        key_id, _ = await keystore.create_key(label="only")
        assert keystore.has_keys()
        await keystore.delete(key_id)
        assert not keystore.has_keys()


class TestKeyStoreRotate:
    async def test_rotate_returns_new_key(self, keystore):
        key_id, old_key = await keystore.create_key(label="rotate")
        new_key = await keystore.rotate(key_id)
        assert new_key is not None
        assert new_key != old_key
        assert new_key.startswith("rsk-")

    async def test_rotate_invalidates_old_key(self, keystore):
        key_id, old_key = await keystore.create_key(label="rotate")
        await keystore.rotate(key_id)
        assert keystore.validate(old_key) is None

    async def test_rotate_new_key_validates(self, keystore):
        key_id, _ = await keystore.create_key(label="rotate")
        new_key = await keystore.rotate(key_id)
        result = keystore.validate(new_key)
        assert result is not None
        _, ctx = result
        assert ctx.label == "rotate"

    async def test_rotate_sets_rotated_timestamp(self, keystore):
        key_id, _ = await keystore.create_key(label="ts")
        await keystore.rotate(key_id)
        keys = await keystore.list_keys()
        entry = next(k for k in keys if k["id"] == key_id)
        assert entry.get("rotated") is not None

    async def test_rotate_nonexistent(self, keystore):
        assert await keystore.rotate("nonexistent") is None


class TestKeyStoreHasKeys:
    async def test_has_keys_empty(self, keystore):
        assert not keystore.has_keys()

    async def test_has_keys_with_key(self, keystore):
        await keystore.create_key(label="x")
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
    async def test_wal_mode(self, tmp_path):
        ks = await KeyStore.create(tmp_path / "test.db")
        import sqlite3

        conn = sqlite3.connect(str(tmp_path / "test.db"))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        await ks.close()
        assert mode == "wal"
