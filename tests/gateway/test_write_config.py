"""Tests for atomic write_config and config_lock."""

from __future__ import annotations

import json
import os
import threading

from llm_rosetta.gateway.config import config_lock, write_config


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
