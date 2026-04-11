"""Tests for the admin panel RequestLog."""

from llm_rosetta.gateway.admin.request_log import RequestLog, RequestLogEntry


class TestRequestLogEntry:
    def test_create(self):
        e = RequestLogEntry.create(
            model="gpt-4o",
            source_provider="openai_chat",
            target_provider="anthropic",
            is_stream=False,
            status_code=200,
            duration_ms=123.456,
        )
        assert e.model == "gpt-4o"
        assert e.duration_ms == 123.46  # rounded
        assert e.id  # non-empty uuid
        assert e.timestamp  # non-empty ISO timestamp
        assert e.error_detail is None

    def test_to_dict(self):
        e = RequestLogEntry.create(
            model="gpt-4o",
            source_provider="openai_chat",
            target_provider="anthropic",
            is_stream=True,
            status_code=500,
            duration_ms=50.0,
            error_detail="connection refused",
        )
        d = e.to_dict()
        assert d["model"] == "gpt-4o"
        assert d["is_stream"] is True
        assert d["status_code"] == 500
        assert d["error_detail"] == "connection refused"


class TestRequestLog:
    def _make_entry(self, model="gpt-4o", status=200, provider="openai_chat"):
        return RequestLogEntry.create(
            model=model,
            source_provider="openai_chat",
            target_provider=provider,
            is_stream=False,
            status_code=status,
            duration_ms=10.0,
        )

    def test_add_and_get(self):
        log = RequestLog(max_entries=100)
        log.add(self._make_entry())
        entries, total = log.get_entries()
        assert total == 1
        assert len(entries) == 1

    def test_max_entries_eviction(self):
        log = RequestLog(max_entries=3)
        for i in range(5):
            log.add(self._make_entry(model=f"model-{i}"))
        assert len(log) == 3
        entries, total = log.get_entries(limit=10)
        assert total == 3
        # Should have models 2, 3, 4 (oldest evicted)
        models = [e["model"] for e in entries]
        assert "model-0" not in models
        assert "model-1" not in models
        assert "model-4" in models

    def test_filter_by_model(self):
        log = RequestLog()
        log.add(self._make_entry(model="gpt-4o"))
        log.add(self._make_entry(model="claude"))
        log.add(self._make_entry(model="gpt-4o"))
        entries, total = log.get_entries(model="gpt-4o")
        assert total == 2
        assert all(e["model"] == "gpt-4o" for e in entries)

    def test_filter_by_provider(self):
        log = RequestLog()
        log.add(self._make_entry(provider="openai_chat"))
        log.add(self._make_entry(provider="anthropic"))
        entries, total = log.get_entries(provider="anthropic")
        assert total == 1
        assert entries[0]["target_provider"] == "anthropic"

    def test_filter_by_status(self):
        log = RequestLog()
        log.add(self._make_entry(status=200))
        log.add(self._make_entry(status=500))
        log.add(self._make_entry(status=404))

        ok_entries, ok_total = log.get_entries(status="ok")
        assert ok_total == 1

        err_entries, err_total = log.get_entries(status="error")
        assert err_total == 2

    def test_pagination(self):
        log = RequestLog()
        for i in range(10):
            log.add(self._make_entry(model=f"m-{i}"))

        entries, total = log.get_entries(limit=3, offset=0)
        assert total == 10
        assert len(entries) == 3

        entries2, _ = log.get_entries(limit=3, offset=3)
        assert len(entries2) == 3
        # Different entries
        assert entries[0]["id"] != entries2[0]["id"]

    def test_newest_first(self):
        log = RequestLog()
        log.add(self._make_entry(model="first"))
        log.add(self._make_entry(model="second"))
        entries, _ = log.get_entries()
        assert entries[0]["model"] == "second"
        assert entries[1]["model"] == "first"

    def test_get_entry_by_id(self):
        log = RequestLog()
        e = self._make_entry()
        log.add(e)
        found = log.get_entry(e.id)
        assert found is not None
        assert found["id"] == e.id

    def test_get_entry_not_found(self):
        log = RequestLog()
        assert log.get_entry("nonexistent") is None

    def test_clear(self):
        log = RequestLog()
        log.add(self._make_entry())
        log.add(self._make_entry())
        assert len(log) == 2
        log.clear()
        assert len(log) == 0
