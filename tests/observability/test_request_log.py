"""Tests for the observability RequestLog (standalone, no gateway)."""

from llm_rosetta.observability import RequestLog, RequestLogEntry


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
        assert e.id
        assert e.timestamp
        assert e.error_detail is None

    def test_to_dict_minimal(self):
        e = RequestLogEntry.create(
            model="gpt-4o",
            source_provider="openai_chat",
            target_provider="anthropic",
            is_stream=False,
            status_code=200,
            duration_ms=50.0,
        )
        d = e.to_dict()
        assert d["model"] == "gpt-4o"
        assert "error_detail" not in d  # None fields omitted
        assert "api_key_label" not in d

    def test_to_dict_with_optionals(self):
        e = RequestLogEntry.create(
            model="gpt-4o",
            source_provider="openai_chat",
            target_provider="anthropic",
            is_stream=True,
            status_code=500,
            duration_ms=100.0,
            error_detail="timeout",
            api_key_label="test-key",
            target_provider_name="My Anthropic",
            client_ip="1.2.3.4",
        )
        d = e.to_dict()
        assert d["error_detail"] == "timeout"
        assert d["api_key_label"] == "test-key"
        assert d["target_provider_name"] == "My Anthropic"
        assert d["client_ip"] == "1.2.3.4"


class TestRequestLogInMemory:
    def test_add_and_get(self):
        log = RequestLog(max_entries=10)
        entry = RequestLogEntry.create(
            model="gpt-4o",
            source_provider="openai_chat",
            target_provider="anthropic",
            is_stream=False,
            status_code=200,
            duration_ms=50.0,
        )
        log.add(entry)
        assert len(log) == 1
        entries, total = log.get_entries(limit=10)
        assert total == 1
        assert entries[0]["model"] == "gpt-4o"

    def test_filter_by_status(self):
        log = RequestLog(max_entries=10)
        for sc in [200, 200, 500]:
            log.add(
                RequestLogEntry.create(
                    model="gpt-4o",
                    source_provider="openai_chat",
                    target_provider="anthropic",
                    is_stream=False,
                    status_code=sc,
                    duration_ms=10.0,
                )
            )
        ok_entries, ok_total = log.get_entries(status="ok")
        assert ok_total == 2
        err_entries, err_total = log.get_entries(status="error")
        assert err_total == 1

    def test_clear(self):
        log = RequestLog(max_entries=10)
        log.add(
            RequestLogEntry.create(
                model="gpt-4o",
                source_provider="openai_chat",
                target_provider="anthropic",
                is_stream=False,
                status_code=200,
                duration_ms=10.0,
            )
        )
        assert len(log) == 1
        log.clear()
        assert len(log) == 0
