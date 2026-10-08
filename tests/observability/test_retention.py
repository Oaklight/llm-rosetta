"""Tests for the unified retention module."""

from llm_rosetta.observability.retention import RetentionPolicy, RetentionTracker


class TestRetentionPolicy:
    def test_defaults(self):
        p = RetentionPolicy()
        assert p.success_max == 50_000
        assert p.dump_max == 10_000
        assert p.ops_info_max == 10_000
        assert p.ops_warn_max == 5_000
        assert p.max_age_days == 90

    def test_custom_values(self):
        p = RetentionPolicy(success_max=100, dump_max=50, max_age_days=30)
        assert p.success_max == 100
        assert p.dump_max == 50
        assert p.max_age_days == 30

    def test_mutable(self):
        p = RetentionPolicy()
        p.success_max = 200
        assert p.success_max == 200

    def test_floor_enforcement(self):
        p = RetentionPolicy(success_max=0, dump_max=0, ops_info_max=-1, max_age_days=0)
        assert p.success_max == 1
        assert p.dump_max == 1
        assert p.ops_info_max == 1
        assert p.max_age_days == 1


class TestRetentionTracker:
    def test_note_insert_under_threshold(self):
        t = RetentionTracker()
        assert not t.note_insert("request_log", 50)

    def test_note_insert_at_threshold(self):
        t = RetentionTracker()
        # First 99 inserts: no prune
        for _ in range(99):
            assert not t.note_insert("request_log")
        # 100th: prune
        assert t.note_insert("request_log")
        # Counter resets, next 99 are safe
        assert not t.note_insert("request_log")

    def test_batch_insert_triggers(self):
        t = RetentionTracker()
        assert t.note_insert("request_log", 100)

    def test_independent_categories(self):
        t = RetentionTracker()
        t.note_insert("request_log", 50)
        t.note_insert("ops_log", 50)
        assert not t.note_insert("request_log", 40)
        assert not t.note_insert("ops_log", 40)
        assert t.note_insert("request_log", 10)
        assert t.note_insert("ops_log", 10)

    def test_reset_category(self):
        t = RetentionTracker()
        t.note_insert("request_log", 90)
        t.reset("request_log")
        assert not t.note_insert("request_log", 90)

    def test_reset_all(self):
        t = RetentionTracker()
        t.note_insert("request_log", 90)
        t.note_insert("ops_log", 90)
        t.reset()
        assert not t.note_insert("request_log", 90)
        assert not t.note_insert("ops_log", 90)
