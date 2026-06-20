"""Tests for anomalous response content detection."""

import logging


from llm_rosetta.converters.base.anomaly_detector import (
    _detect_anomaly,
    check_ir_response_content,
)


class TestDetectAnomaly:
    def test_clean_text_not_flagged(self):
        assert _detect_anomaly("The weather in New York is sunny today.") is None

    def test_short_text_skipped(self):
        assert _detect_anomaly("TypeError:") is None  # too short

    def test_js_property_error(self):
        text = "Cannot read properties of undefined (reading 'input_tokens')"
        assert _detect_anomaly(text) == "javascript_error"

    def test_js_type_error(self):
        text = "TypeError: Cannot set property 'value' of null at processResponse"
        assert _detect_anomaly(text) == "javascript_error"

    def test_js_reference_error(self):
        text = "ReferenceError: someVariable is not defined in module scope"
        assert _detect_anomaly(text) == "javascript_error"

    def test_js_stack_frame(self):
        # "at Object.foo (file:line:col)" matches the stack_trace group
        text = "  at Object.processResponse (gateway.js:42:10) in handler"
        assert _detect_anomaly(text) == "stack_trace"

    def test_js_bare_stack_frame(self):
        # "at funcName (file:line:col)" — no dot, matches JS stack frame pattern
        text = "  at processTokens (gateway.js:42:10) in error handler"
        assert _detect_anomaly(text) == "javascript_error"

    def test_html_doctype(self):
        text = (
            "<!DOCTYPE html><html><body><h1>503 Service Unavailable</h1></body></html>"
        )
        assert _detect_anomaly(text) == "html_error_page"

    def test_html_tag(self):
        text = "<html><head><title>Bad Gateway</title></head><body></body></html>"
        assert _detect_anomaly(text) == "html_error_page"

    def test_nginx_error(self):
        text = "<html><head><title>nginx/1.18 error page</title></head></html>"
        assert _detect_anomaly(text) is not None  # html or stack trace

    def test_python_traceback(self):
        text = (
            'Traceback (most recent call last):\n  File "app.py", line 42, in handler'
        )
        assert _detect_anomaly(text) == "stack_trace"

    def test_python_file_line(self):
        text = '  File "src/llm_rosetta/gateway/proxy.py", line 123, in handle_non_streaming'
        assert _detect_anomaly(text) == "stack_trace"

    def test_normal_long_text_not_flagged(self):
        text = (
            "Python is a high-level, general-purpose programming language. "
            "Its design philosophy emphasizes code readability."
        )
        assert _detect_anomaly(text) is None


class TestCheckIRResponseContent:
    """Tests for check_ir_response_content using flat content list format."""

    def test_no_warning_for_clean_response(self, caplog):
        ir = {
            "content": [
                {"type": "text", "text": "Hello, world! This is a normal response."}
            ]
        }
        with caplog.at_level(logging.WARNING):
            check_ir_response_content(ir, provider="test", model="test-model")
        assert not caplog.records

    def test_warning_for_js_error(self, caplog):
        ir = {
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Cannot read properties of undefined (reading 'input_tokens') "
                        "at processTokens (gateway.js:42:10)"
                    ),
                }
            ]
        }
        with caplog.at_level(logging.WARNING):
            check_ir_response_content(
                ir, provider="argo", model="gpt-4o", request_id="test-123"
            )
        assert len(caplog.records) == 1
        assert "anomalous" in caplog.records[0].message
        assert "javascript_error" in caplog.records[0].message
        assert "test-123" in caplog.records[0].message

    def test_non_text_parts_skipped(self, caplog):
        ir = {
            "content": [
                {"type": "tool_call", "tool_name": "search", "tool_input": {}},
                {"type": "image", "image_url": "https://example.com/img.png"},
            ]
        }
        with caplog.at_level(logging.WARNING):
            check_ir_response_content(ir, provider="test", model="test")
        assert not caplog.records

    def test_non_dict_response_ignored(self, caplog):
        with caplog.at_level(logging.WARNING):
            check_ir_response_content("not a dict", provider="test")
        assert not caplog.records

    def test_request_id_in_warning(self, caplog):
        ir = {
            "content": [
                {
                    "type": "text",
                    "text": 'Traceback (most recent call last):\n  File "x.py", line 1\nValueError: bad',
                }
            ]
        }
        with caplog.at_level(logging.WARNING):
            check_ir_response_content(ir, request_id="req-abc-123")
        assert any("req-abc-123" in r.message for r in caplog.records)

    def test_empty_content_list_no_warning(self, caplog):
        ir = {"content": []}
        with caplog.at_level(logging.WARNING):
            check_ir_response_content(ir, provider="test")
        assert not caplog.records

    def test_provider_in_warning(self, caplog):
        ir = {
            "content": [
                {
                    "type": "text",
                    "text": "TypeError: something went wrong in middleware",
                }
            ]
        }
        with caplog.at_level(logging.WARNING):
            check_ir_response_content(ir, provider="argo-gateway", model="gpt-4o")
        assert caplog.records
        assert "argo-gateway" in caplog.records[0].message

    def test_model_in_warning(self, caplog):
        ir = {
            "content": [
                {
                    "type": "text",
                    "text": "TypeError: something went wrong in middleware",
                }
            ]
        }
        with caplog.at_level(logging.WARNING):
            check_ir_response_content(ir, model="claude-3-5-sonnet")
        assert caplog.records
        assert "claude-3-5-sonnet" in caplog.records[0].message

    def test_missing_model_shows_unknown(self, caplog):
        ir = {
            "content": [
                {
                    "type": "text",
                    "text": "TypeError: something went wrong in middleware",
                }
            ]
        }
        with caplog.at_level(logging.WARNING):
            check_ir_response_content(ir)
        assert caplog.records
        assert "unknown" in caplog.records[0].message


class TestCheckIRResponseContentChoicesFormat:
    """Tests using IRResponse choices[] format (the real IR shape)."""

    def test_clean_ir_response_no_warning(self, caplog):
        ir = {
            "id": "resp-1",
            "object": "response",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "The capital of France is Paris."}
                        ],
                    },
                    "finish_reason": {"reason": "stop"},
                }
            ],
        }
        with caplog.at_level(logging.WARNING):
            check_ir_response_content(ir, provider="openai", model="gpt-4o")
        assert not caplog.records

    def test_js_error_in_choices_triggers_warning(self, caplog):
        ir = {
            "id": "resp-2",
            "object": "response",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "Cannot read properties of undefined (reading 'tokens')",
                            }
                        ],
                    },
                    "finish_reason": {"reason": "stop"},
                }
            ],
        }
        with caplog.at_level(logging.WARNING):
            check_ir_response_content(
                ir, provider="argo", model="gpt-4o", request_id="rid-42"
            )
        assert caplog.records
        assert "javascript_error" in caplog.records[0].message
        assert "rid-42" in caplog.records[0].message
