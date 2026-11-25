"""Tests for atomic write_config and config_lock."""

from __future__ import annotations

import json
import os
import threading

from llm_rosetta_gateway.config import config_lock, write_config


class TestWriteConfigAtomic:
    """write_config uses tempfile + os.replace for crash-safe writes."""

    def test_creates_file(self, tmp_path):
        path = str(tmp_path / "new.json")
        write_config(path, {"key": "value"})
        with open(path) as f:
            assert json.load(f) == {"key": "value"}

    def test_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "a" / "b" / "config.json")
        write_config(path, {"nested": True})
        assert os.path.isfile(path)

    def test_overwrites_existing(self, tmp_path):
        path = str(tmp_path / "cfg.json")
        write_config(path, {"v": 1})
        write_config(path, {"v": 2})
        with open(path) as f:
            assert json.load(f)["v"] == 2

    def test_no_partial_write_on_error(self, tmp_path):
        path = str(tmp_path / "cfg.json")
        write_config(path, {"original": True})

        class Unserializable:
            pass

        try:
            write_config(path, {"bad": Unserializable()})
        except TypeError:
            pass

        with open(path) as f:
            assert json.load(f) == {"original": True}

    def test_no_leftover_tmp_files(self, tmp_path):
        path = str(tmp_path / "cfg.json")
        write_config(path, {"clean": True})
        files = os.listdir(tmp_path)
        assert files == ["cfg.json"]


class TestRollingBackup:
    """A write keeps the file it replaces as ``<path>.prev`` (#1663 review).

    The writer drops comments, so the previous bytes are the only way back
    to a hand-annotated file after a panel save.  The name is ``.prev``, not
    ``.bak``: ``llm-rosetta migrate --write`` keeps the pre-migration original
    as ``<path>.bak``, and so do operators, and neither may be overwritten.
    """

    def test_the_backup_holds_the_previous_bytes(self, tmp_path):
        path = tmp_path / "config.jsonc"
        original = b'{\n  // an operator note\n  "v": 1\n}\n'
        path.write_bytes(original)

        write_config(str(path), {"v": 2})

        assert (tmp_path / "config.jsonc.prev").read_bytes() == original
        assert json.loads(path.read_text()) == {"v": 2}

    def test_the_backup_rolls(self, tmp_path):
        path = str(tmp_path / "cfg.json")
        write_config(path, {"v": 1})
        write_config(path, {"v": 2})
        write_config(path, {"v": 3})

        with open(path + ".prev") as f:
            assert json.load(f) == {"v": 2}
        assert sorted(os.listdir(tmp_path)) == ["cfg.json", "cfg.json.prev"]

    def test_a_new_file_leaves_no_backup(self, tmp_path):
        path = str(tmp_path / "fresh.json")
        write_config(path, {"v": 1})

        assert not os.path.exists(path + ".prev")

    def test_an_existing_bak_is_left_alone(self, tmp_path):
        """A ``.bak`` another tool wrote (the migrate tool's original) survives."""
        path = tmp_path / "config.jsonc"
        path.write_text('{"v": 2}')
        original = b'{\n  // the pre-migration original\n  "v": 1\n}\n'
        (tmp_path / "config.jsonc.bak").write_bytes(original)

        write_config(str(path), {"v": 3})
        write_config(str(path), {"v": 4})

        assert (tmp_path / "config.jsonc.bak").read_bytes() == original
        assert json.loads((tmp_path / "config.jsonc.prev").read_text()) == {"v": 3}

    def test_a_failed_write_does_not_touch_the_backup(self, tmp_path):
        path = str(tmp_path / "cfg.json")
        write_config(path, {"v": 1})
        write_config(path, {"v": 2})

        class Unserializable:
            pass

        try:
            write_config(path, {"bad": Unserializable()})
        except TypeError:
            pass

        with open(path + ".prev") as f:
            assert json.load(f) == {"v": 1}


class TestConfigLock:
    """config_lock serializes concurrent access to the same config path."""

    def test_same_path_serialized(self, tmp_path):
        path = str(tmp_path / "cfg.json")
        write_config(path, {"counter": 0})
        results = []

        def increment(n):
            for _ in range(n):
                with config_lock(path):
                    with open(path) as f:
                        data = json.load(f)
                    data["counter"] += 1
                    write_config(path, data)
            results.append(True)

        t1 = threading.Thread(target=increment, args=(10,))
        t2 = threading.Thread(target=increment, args=(10,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        with open(path) as f:
            assert json.load(f)["counter"] == 20

    def test_different_paths_independent(self, tmp_path):
        path_a = str(tmp_path / "a.json")
        path_b = str(tmp_path / "b.json")
        acquired = {"a": False, "b": False}
        barrier = threading.Barrier(2, timeout=5)

        def lock_path(path, key):
            with config_lock(path):
                acquired[key] = True
                barrier.wait()

        t1 = threading.Thread(target=lock_path, args=(path_a, "a"))
        t2 = threading.Thread(target=lock_path, args=(path_b, "b"))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)
        assert acquired["a"] and acquired["b"]
