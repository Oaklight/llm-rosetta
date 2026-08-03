"""Tests for structured JSON logging (issue #126).

Covers:
- Config parsing: log_format values, env-var override, invalid fallback
- JsonFormatter output: schema, extra fields, exception handling
- Auto-detection: JSON when not TTY, text when TTY
- setup_logging integration: formatter selection
- Structured extras in log helpers
"""

from __future__ import annotations

import json
import logging
import os
from io import StringIO
from unittest import mock

from llm_rosetta.gateway.config import GatewayConfig
from llm_rosetta.gateway.logging import (
    JsonFormatter,
    _resolve_log_format,
    _structured_extra,
    setup_logging,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_raw(**debug_overrides) -> dict:
    raw = {
        "providers": {
            "test": {
                "api_key": "***",
                "base_url": "https://api.example.com",
                "type": "openai",
            }
        },
        "models": {"gpt-test": "test"},
        "debug": {},
    }
    raw["debug"].update(debug_overrides)
    return raw


def _capture_json_log(
    message: str,
    level: int = logging.INFO,
    extra: dict | None = None,
) -> dict:
    """Format a log record with JsonFormatter and return parsed JSON."""
    formatter = JsonFormatter()
    logger = logging.getLogger("test.structured")
    record = logger.makeRecord(
        name="test.structured",
        level=level,
        fn="test.py",
        lno=1,
        msg=message,
        args=(),
        exc_info=None,
        extra=extra,
    )
    line = formatter.format(record)
    return json.loads(line)


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


class TestLogFormatConfig:
    def test_default_is_auto(self):
        cfg = GatewayConfig(_minimal_raw())
        assert cfg.log_format == "auto"

    def test_explicit_json(self):
        cfg = GatewayConfig(_minimal_raw(log_format="json"))
        assert cfg.log_format == "json"

    def test_explicit_text(self):
        cfg = GatewayConfig(_minimal_raw(log_format="text"))
        assert cfg.log_format == "text"

    def test_invalid_falls_back_to_auto(self):
        cfg = GatewayConfig(_minimal_raw(log_format="yaml"))
        assert cfg.log_format == "auto"

    @mock.patch.dict(os.environ, {"LLM_ROSETTA_LOG_FORMAT": "json"})
    def test_env_var_overrides_config(self):
        cfg = GatewayConfig(_minimal_raw(log_format="text"))
        assert cfg.log_format == "json"

    @mock.patch.dict(os.environ, {"LLM_ROSETTA_LOG_FORMAT": ""})
    def test_empty_env_var_uses_config(self):
        cfg = GatewayConfig(_minimal_raw(log_format="text"))
        assert cfg.log_format == "text"


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


class TestAutoDetection:
    def test_auto_resolves_to_json_when_not_tty(self):
        with mock.patch("sys.stderr") as mock_stderr:
            mock_stderr.isatty.return_value = False
            assert _resolve_log_format("auto") == "json"

    def test_auto_resolves_to_text_when_tty(self):
        with mock.patch("sys.stderr") as mock_stderr:
            mock_stderr.isatty.return_value = True
            assert _resolve_log_format("auto") == "text"

    def test_explicit_json_not_affected_by_tty(self):
        with mock.patch("sys.stderr") as mock_stderr:
            mock_stderr.isatty.return_value = True
            assert _resolve_log_format("json") == "json"

    def test_explicit_text_not_affected_by_tty(self):
        with mock.patch("sys.stderr") as mock_stderr:
            mock_stderr.isatty.return_value = False
            assert _resolve_log_format("text") == "text"


# ---------------------------------------------------------------------------
# JsonFormatter
# ---------------------------------------------------------------------------


class TestJsonFormatter:
    def test_basic_output_schema(self):
        entry = _capture_json_log("hello world")
        assert entry["level"] == "INFO"
        assert entry["logger"] == "test.structured"
        assert entry["message"] == "hello world"
        assert "timestamp" in entry
        # ISO 8601 shape
        assert entry["timestamp"].endswith("Z")
        assert "T" in entry["timestamp"]

    def test_extra_fields_promoted(self):
        entry = _capture_json_log(
            "request handled",
            extra={"request_id": "abc-123", "model": "gpt-4"},
        )
        assert entry["request_id"] == "abc-123"
        assert entry["model"] == "gpt-4"

    def test_standard_attrs_excluded(self):
        entry = _capture_json_log("test")
        # Standard Python logging attributes should NOT appear
        for attr in ("args", "funcName", "pathname", "lineno", "processName"):
            assert attr not in entry

    def test_levels(self):
        for level, name in [
            (logging.DEBUG, "DEBUG"),
            (logging.WARNING, "WARNING"),
            (logging.ERROR, "ERROR"),
            (logging.CRITICAL, "CRITICAL"),
        ]:
            entry = _capture_json_log("msg", level=level)
            assert entry["level"] == name

    def test_exception_included(self):
        formatter = JsonFormatter()
        logger = logging.getLogger("test.exc")
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = logger.makeRecord(
                name="test.exc",
                level=logging.ERROR,
                fn="test.py",
                lno=1,
                msg="error occurred",
                args=(),
                exc_info=sys.exc_info(),
            )
        line = formatter.format(record)
        entry = json.loads(line)
        assert "exception" in entry
        assert "ValueError: boom" in entry["exception"]

    def test_output_is_single_line(self):
        """Each JSON entry must be one line for log aggregator compat."""
        formatter = JsonFormatter()
        logger = logging.getLogger("test.oneline")
        record = logger.makeRecord(
            name="test.oneline",
            level=logging.INFO,
            fn="test.py",
            lno=1,
            msg="multi\nline\nmessage",
            args=(),
            exc_info=None,
        )
        line = formatter.format(record)
        # json.dumps with default settings doesn't add newlines
        assert "\n" not in line


# ---------------------------------------------------------------------------
# setup_logging integration
# ---------------------------------------------------------------------------


class TestSetupLogging:
    def test_json_format_uses_json_formatter(self):
        logger = setup_logging(log_format="json")
        handler = logger.handlers[-1]
        assert isinstance(handler.formatter, JsonFormatter)

    def test_text_format_uses_colored_formatter(self):
        from llm_rosetta.gateway.logging import ColoredFormatter

        logger = setup_logging(log_format="text")
        handler = logger.handlers[-1]
        assert isinstance(handler.formatter, ColoredFormatter)

    def test_json_output_parseable(self):
        """End-to-end: setup with JSON, emit a log, verify it parses."""
        stream = StringIO()
        logger = setup_logging(log_format="json")
        handler = logger.handlers[-1]
        assert isinstance(handler, logging.StreamHandler)
        handler.stream = stream  # type: ignore[union-attr]  # StreamHandler has .stream

        logger.info("test message", extra={"request_id": "r-1"})

        output = stream.getvalue().strip()
        entry = json.loads(output)
        assert entry["message"] == "test message"
        assert entry["request_id"] == "r-1"


# ---------------------------------------------------------------------------
# _structured_extra helper
# ---------------------------------------------------------------------------


class TestStructuredExtra:
    def test_none_values_excluded(self):
        extra = _structured_extra(request_id="abc", model=None, status=200)
        assert extra == {"request_id": "abc", "status": 200}

    def test_all_none_returns_empty(self):
        extra = _structured_extra(request_id=None, model=None)
        assert extra == {}

    def test_all_values_included(self):
        extra = _structured_extra(
            request_id="r-1",
            model="gpt-4",
            source_provider="openai",
            target_provider="anthropic",
            duration_ms=1234,
            status="success",
        )
        assert extra == {
            "request_id": "r-1",
            "model": "gpt-4",
            "source_provider": "openai",
            "target_provider": "anthropic",
            "duration_ms": 1234,
            "status": "success",
        }


# ---------------------------------------------------------------------------
# Log helpers with structured fields
# ---------------------------------------------------------------------------


class TestLogHelpersStructured:
    """Verify that log helpers accept and pass through structured kwargs."""

    def test_log_request_accepts_structured_kwargs(self):
        """log_request should not raise when given structured kwargs."""
        from llm_rosetta.gateway.logging import log_request

        # Should not raise
        log_request(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
            request_id="req-1",
            model="gpt-4",
            source_provider="openai",
            target_provider="anthropic",
        )

    def test_log_stream_summary_accepts_structured_kwargs(self):
        from llm_rosetta.gateway.logging import log_stream_summary

        log_stream_summary(
            model="gpt-4",
            duration_s=1.5,
            chunk_count=10,
            request_id="req-2",
            source_provider="openai",
            target_provider="anthropic",
            status="success",
        )

    def test_log_upstream_error_accepts_structured_kwargs(self):
        from llm_rosetta.gateway.logging import log_upstream_error

        log_upstream_error(
            500,
            "Internal Server Error",
            endpoint="/v1/chat/completions",
            is_streaming=True,
            request_id="req-3",
            model="gpt-4",
        )

    def test_log_response_accepts_structured_kwargs(self):
        from llm_rosetta.gateway.logging import log_response

        log_response(
            {"choices": [{"message": {"content": "hi"}}]},
            request_id="req-4",
            model="gpt-4",
            duration_ms=500,
            status="success",
        )

    def test_json_format_captures_extras_from_helper(self):
        """End-to-end: log_request with JSON format should include extras."""
        from llm_rosetta.gateway.logging import log_request

        stream = StringIO()
        logger = setup_logging(log_format="json", verbose=True)
        handler = logger.handlers[-1]
        assert isinstance(handler, logging.StreamHandler)
        handler.stream = stream  # type: ignore[union-attr]  # StreamHandler has .stream

        log_request(
            {"model": "gpt-4", "messages": [], "stream": True},
            request_id="req-e2e",
            model="gpt-4",
            source_provider="openai",
            target_provider="anthropic",
        )

        lines = stream.getvalue().strip().split("\n")
        assert len(lines) >= 1
        entry = json.loads(lines[0])
        assert entry["request_id"] == "req-e2e"
        assert entry["model"] == "gpt-4"
        assert entry["source_provider"] == "openai"
        assert entry["target_provider"] == "anthropic"
