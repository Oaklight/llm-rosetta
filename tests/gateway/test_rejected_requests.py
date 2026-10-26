"""Requests refused before a route handler leave a usable trace (#1217).

These drive a real ``create_app`` over a real socket, because the fault they
guard against lives in the seams between the server, the middleware hooks and
the lifecycle: a unit test of any one of them cannot see a row go missing.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest

from llm_rosetta_gateway._vendor import httpserver
from llm_rosetta_gateway.app import create_app
from llm_rosetta_gateway.config import GatewayConfig
from llm_rosetta_gateway.keystore import KeyStore
from llm_rosetta_gateway.observability import PersistenceManager, RequestLog
from llm_rosetta_gateway.observability.error_dump import decompress_body
from llm_rosetta_gateway.observability.sanitize import scrub_body_text
from llm_rosetta_gateway.ops import OpsContext, OpsRejectedRequest
from llm_rosetta_gateway.ops import rejected as rejected_module


class Served:
    """A running gateway, its store, and a raw HTTP client for it."""

    def __init__(
        self, port: int, pm: PersistenceManager, metrics: MagicMock, app: Any
    ) -> None:
        self.port = port
        self.pm = pm
        self.metrics = metrics
        self.app = app

    async def send_raw(
        self, raw: bytes, *, eof: bool = True
    ) -> tuple[int, dict[str, str], bytes]:
        """Send *raw* as written; ``eof`` half-closes so the server sees a short body."""
        reader, writer = await asyncio.open_connection("127.0.0.1", self.port)
        writer.write(raw)
        await writer.drain()
        if eof:
            writer.write_eof()
        data = await reader.read()
        writer.close()
        head, _, payload = data.partition(b"\r\n\r\n")
        status_line, *header_lines = head.decode().split("\r\n")
        parsed = {
            key.strip().lower(): value.strip()
            for key, _, value in (line.partition(":") for line in header_lines)
        }
        return int(status_line.split()[1]), parsed, payload

    async def send(
        self,
        method: str,
        path: str,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        lines = [f"{method} {path} HTTP/1.1", "Host: test", "Connection: close"]
        merged = {"Content-Length": str(len(body)), **(headers or {})}
        lines += [f"{key}: {value}" for key, value in merged.items()]
        reader, writer = await asyncio.open_connection("127.0.0.1", self.port)
        writer.write("\r\n".join(lines).encode() + b"\r\n\r\n" + body)
        await writer.drain()
        raw = await reader.read()
        writer.close()
        head, _, payload = raw.partition(b"\r\n\r\n")
        status_line, *header_lines = head.decode().split("\r\n")
        parsed = {
            key.strip().lower(): value.strip()
            for key, _, value in (line.partition(":") for line in header_lines)
        }
        return int(status_line.split()[1]), parsed, payload

    async def rows(self) -> list[dict[str, Any]]:
        rows, _ = await self.pm.query_log_entries(limit=100)
        return rows


async def _serve(
    tmp_path, server: dict[str, Any], providers: dict[str, Any] | None = None
) -> AsyncIterator[Served]:
    pm = await PersistenceManager.create(str(tmp_path / "gateway.db"))
    server = {"data_dir": str(tmp_path / "data"), **server}
    config = GatewayConfig(
        {"providers": providers or {}, "models": {}, "server": server}
    )
    app = await create_app(config)
    metrics = MagicMock()
    request_log = RequestLog(persistence=pm)
    wired: Any = app
    wired.ops_ctx = OpsContext(request_log=request_log, metrics=metrics, persistence=pm)
    wired.request_log = request_log
    wired.persistence = pm
    listener = await asyncio.start_server(app._handle_connection, "127.0.0.1", 0)
    port = listener.sockets[0].getsockname()[1]
    try:
        yield Served(port, pm, metrics, app)
    finally:
        listener.close()
        await listener.wait_closed()
        await pm.close()


@pytest.fixture()
async def served(tmp_path) -> AsyncIterator[Served]:
    async for gateway in _serve(tmp_path, {"open_on_no_keys": True}):
        yield gateway


@pytest.fixture()
async def holding_keys(tmp_path) -> AsyncIterator[Served]:
    """A gateway that holds two upstream keys of no vendor's shape."""
    providers = {
        "up": {
            "type": "openai",
            "base_url": "https://api.example.com",
            "api_key": "canary-one-7f3a,canary-two-91bc",
        }
    }
    async for gateway in _serve(tmp_path, {"open_on_no_keys": True}, providers):
        yield gateway


@pytest.fixture()
async def keyed(tmp_path) -> AsyncIterator[Served]:
    """A gateway with one API key, so a call without it is a 401."""
    keys_db = tmp_path / "keys.db"
    store = await KeyStore.create(keys_db)
    await store.create_key("test")
    await store.close()
    async for gateway in _serve(tmp_path, {"api_keys_db": str(keys_db)}):
        yield gateway


@pytest.fixture()
async def limited(tmp_path) -> AsyncIterator[Served]:
    server = {
        "open_on_no_keys": True,
        "rate_limit": {"enabled": True, "global": "1/m"},
    }
    async for gateway in _serve(tmp_path, server):
        yield gateway


class TestEarlyRejectionsLeaveARow:
    async def test_an_unmatched_path_is_recorded_with_its_request_id(self, served):
        # The CORS catch-all route matches every path, so an unknown path
        # answers 405 rather than 404.
        status, headers, _ = await served.send("GET", "/nowhere")

        assert status == 405
        (row,) = await served.rows()
        assert row["status_code"] == 405
        assert row["request_id"] == headers["x-request-id"]
        assert row["client_ip"] == "127.0.0.1"
        assert "GET /nowhere" in row["error_detail"]
        assert "Method Not Allowed" in row["error_detail"]

    async def test_a_wrong_method_is_recorded(self, served):
        status, _, _ = await served.send("GET", "/v1/chat/completions")

        assert status == 405
        (row,) = await served.rows()
        assert row["status_code"] == 405
        assert "GET /v1/chat/completions" in row["error_detail"]

    async def test_an_auth_rejection_is_recorded(self, keyed):
        status, headers, _ = await keyed.send("POST", "/v1/chat/completions", b"{}")

        assert status == 401
        (row,) = await keyed.rows()
        assert row["status_code"] == 401
        assert row["request_id"] == headers["x-request-id"]
        assert "POST /v1/chat/completions" in row["error_detail"]

    async def test_a_rate_limit_rejection_is_recorded(self, limited):
        first, _, _ = await limited.send("GET", "/nowhere")
        status, headers, _ = await limited.send("GET", "/nowhere")

        assert (first, status) == (405, 429)
        rows = await limited.rows()
        assert sorted(row["status_code"] for row in rows) == [405, 429]
        limited_row = next(row for row in rows if row["status_code"] == 429)
        assert limited_row["request_id"] == headers["x-request-id"]

    async def test_a_client_supplied_request_id_is_the_one_stored(self, served):
        _, headers, _ = await served.send(
            "GET", "/nowhere", headers={"X-Request-Id": "client-chosen-id"}
        )

        assert headers["x-request-id"] == "client-chosen-id"
        (row,) = await served.rows()
        assert row["request_id"] == "client-chosen-id"

    async def test_rejections_stay_out_of_the_provider_keyed_metrics(self, served):
        await served.send("GET", "/nowhere")

        served.metrics.record_request.assert_not_called()


class TestReadPhaseRejections:
    """Failures while the request is still being read, before a ``Request`` exists."""

    CHAT = b"POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"

    async def test_a_body_shorter_than_its_content_length_is_recorded(self, served):
        raw = (
            self.CHAT + b"Content-Length: 100\r\nContent-Type: application/json\r\n\r\n"
        )
        status, headers, _ = await served.send_raw(raw + b'{"model": "m"')

        assert status == 400
        (row,) = await served.rows()
        assert row["status_code"] == 400
        assert row["request_id"] == headers["x-request-id"]
        assert row["client_ip"] == "127.0.0.1"
        detail = row["error_detail"]
        assert "POST /v1/chat/completions rejected while reading" in detail
        assert "incomplete_body" in detail
        assert "Content-Length=100" in detail
        assert "Transfer-Encoding=None" in detail
        assert "received=13 bytes" in detail
        assert "Content-Type=application/json" in detail

    async def test_an_oversized_body_is_recorded(self, served):
        served.app.max_body_size = 64
        raw = self.CHAT + b"Content-Length: 5000\r\n\r\n"
        status, headers, _ = await served.send_raw(raw, eof=False)

        assert status == 413
        (row,) = await served.rows()
        assert row["status_code"] == 413
        assert row["request_id"] == headers["x-request-id"]
        assert "too_large" in row["error_detail"]
        assert "Content-Length=5000" in row["error_detail"]

    async def test_a_malformed_request_line_is_recorded(self, served):
        status, headers, _ = await served.send_raw(b"garbage\r\n\r\n")

        assert status == 400
        (row,) = await served.rows()
        assert row["request_id"] == headers["x-request-id"]
        assert "bad_request" in row["error_detail"]
        assert row["error_detail"].startswith("? ? rejected while reading")

    async def test_a_stalled_body_times_out_and_is_recorded(self, served):
        served.app.read_timeout = 0.2
        raw = self.CHAT + b"Content-Length: 50\r\n\r\nabc"
        status, headers, _ = await served.send_raw(raw, eof=False)

        assert status == 408
        (row,) = await served.rows()
        assert row["status_code"] == 408
        assert row["request_id"] == headers["x-request-id"]
        assert "timeout" in row["error_detail"]
        assert "received=3 bytes" in row["error_detail"]

    async def test_an_unexpected_read_error_is_recorded_and_logged_with_its_stack(
        self, served, monkeypatch, caplog
    ):
        async def boom(*args: Any, **kwargs: Any) -> Any:
            state = kwargs["state"]
            state.method, state.path = "POST", "/v1/chat/completions"
            state.headers = {"content-type": "application/json"}
            raise RuntimeError("reader exploded")

        monkeypatch.setattr(httpserver, "_read_request", boom)
        with caplog.at_level(logging.WARNING, logger=httpserver.logger.name):
            status, headers, _ = await served.send_raw(self.CHAT + b"\r\n")

        assert status == 500
        (row,) = await served.rows()
        assert row["status_code"] == 500
        assert row["request_id"] == headers["x-request-id"]
        assert "read_error" in row["error_detail"]
        assert "RuntimeError: reader exploded" in row["error_detail"]
        assert "POST /v1/chat/completions" in row["error_detail"]
        stacks = [r for r in caplog.records if r.exc_info and r.exc_info[1]]
        assert any(
            isinstance(r.exc_info[1], RuntimeError)
            and row["request_id"] in r.getMessage()
            for r in stacks
            if r.exc_info
        ), "the traceback reaches the container log, tied to the request id"

    async def test_the_clients_request_id_is_the_one_stored(self, served):
        raw = self.CHAT + b"X-Request-Id: caller-7\r\nContent-Length: 100\r\n\r\n{"
        status, headers, _ = await served.send_raw(raw)

        assert status == 400
        assert headers["x-request-id"] == "caller-7"
        (row,) = await served.rows()
        assert row["request_id"] == "caller-7"

    async def test_admin_paths_are_not_recorded(self, served):
        raw = b"POST /admin/api/x HTTP/1.1\r\nHost: t\r\nContent-Length: 100\r\n\r\n{"
        status, _, _ = await served.send_raw(raw)

        assert status == 400
        assert await served.rows() == []


class TestWhatIsNotRecorded:
    async def test_admin_health_and_successes_leave_no_row(self, served):
        await served.send("GET", "/admin/nope")
        await served.send("GET", "/health")
        await served.send("GET", "/health/nope")
        await served.send("GET", "/favicon.ico")
        status, _, _ = await served.send("GET", "/v1/models")

        assert status == 200
        assert await served.rows() == []

    async def test_a_routed_request_is_recorded_once_by_the_lifecycle(self, served):
        status, headers, _ = await served.send(
            "POST", "/v1/chat/completions", b'{"messages": []}'
        )

        assert status == 400  # no model: rejected by the handler, inside the lifecycle
        (row,) = await served.rows()
        assert row["request_id"] == headers["x-request-id"]
        assert "rejected before routing" not in row.get("error_detail", "")


class TestAnUnreadableBody:
    BODY = b'{"model": "m", "messages": [{"role": "user", "content": "hi'

    async def test_the_row_names_the_cause_and_what_arrived(self, served):
        status, headers, _ = await served.send(
            "POST",
            "/v1/chat/completions",
            self.BODY,
            {"Content-Type": "application/json"},
        )

        assert status == 400
        (row,) = await served.rows()
        assert row["request_id"] == headers["x-request-id"]
        detail = row["error_detail"]
        assert detail.startswith("Invalid JSON body: JSONDecodeError")
        assert f"Content-Length={len(self.BODY)}" in detail
        assert f"received={len(self.BODY)} bytes" in detail
        assert "Content-Type=application/json" in detail
        assert "Content-Encoding=None" in detail
        assert "Transfer-Encoding=None" in detail
        assert f"sha256={hashlib.sha256(self.BODY).hexdigest()[:16]}" in detail

    async def _dumped(self, served) -> dict:
        dumps, total = await served.pm.query_error_dumps()
        assert total == 1
        assert dumps[0]["error_phase"] == "body_parse"
        stored = await served.pm.get_dump_body(dumps[0]["body_hash"])
        assert stored is not None
        return decompress_body(stored)["_unparsed_body"]

    async def test_a_dump_keeps_the_whole_body_and_the_digest(self, served):
        await served.send("POST", "/v1/chat/completions", self.BODY)

        (row,) = await served.rows()
        dumps, _ = await served.pm.query_error_dumps()
        assert dumps[0]["request_log_id"] == row["id"]
        sample = await self._dumped(served)
        assert sample["text"] == self.BODY.decode()
        assert "base64" not in sample
        assert sample["sha256"] == hashlib.sha256(self.BODY).hexdigest()
        assert sample["bytes_received"] == len(self.BODY)

    async def test_a_large_body_is_kept_whole_like_any_other_dump(self, served):
        # A corruption in the middle of a large body shows at neither end.
        body = json.dumps({"model": "m", "filler": "x" * 400_000}).encode()[:-3]

        await served.send("POST", "/v1/chat/completions", body)

        sample = await self._dumped(served)
        assert sample["text"] == body.decode()
        assert sample["sha256"] == hashlib.sha256(body).hexdigest()
        assert sample["bytes_received"] == len(body)

    async def test_a_body_that_is_not_utf8_is_kept_byte_for_byte(self, served):
        body = b'{"model": "m", "x": "\xff\xfe\x00 damaged'

        await served.send("POST", "/v1/chat/completions", body)

        sample = await self._dumped(served)
        assert "text" not in sample
        assert base64.b64decode(sample["base64"]) == body
        assert sample["sha256"] == hashlib.sha256(body).hexdigest()

    async def test_a_chunked_body_names_its_transfer_encoding(self, served):
        raw = (
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n"
            b"6\r\n{bad!!\r\n0\r\n\r\n"
        )
        status, _, _ = await served.send_raw(raw)

        assert status == 400
        (row,) = await served.rows()
        assert "Transfer-Encoding=chunked" in row["error_detail"]
        assert "Content-Length=None" in row["error_detail"]
        assert "received=6 bytes" in row["error_detail"]

    async def test_a_json_array_is_named_for_what_it_is(self, served):
        await served.send("POST", "/v1/chat/completions", b"[1, 2]")

        (row,) = await served.rows()
        assert "JSON value is a list, not an object" in row["error_detail"]


class TestTheRequestIdColumn:
    async def test_the_id_round_trips_through_the_store(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path / "gateway.db"))
        try:
            log = RequestLog(persistence=pm)
            op = OpsRejectedRequest(
                OpsContext(request_log=log, persistence=pm),
                status_code=401,
                duration_ms=1.5,
                error_detail="POST /v1/x rejected before routing: no key",
                request_id="abc-123",
                client_ip="10.0.0.9",
            )
            await op.execute()

            (row,) = (await log.get_entries())[0]
            assert row["request_id"] == "abc-123"
            assert row["client_ip"] == "10.0.0.9"
            assert row["model"] == ""
            assert await log.get_entry(op.entry_id or "") == {**row, "_offset": 0}
            indexes = await pm._conn.execute_fetchall(
                "SELECT name FROM sqlite_master WHERE name = 'idx_rl_request_id'"
            )
            assert indexes
        finally:
            await pm.close()

    async def test_a_row_without_one_omits_the_key(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path / "gateway.db"))
        try:
            log = RequestLog(persistence=pm)
            op = OpsRejectedRequest(
                OpsContext(request_log=log, persistence=pm),
                status_code=404,
                duration_ms=0.0,
                error_detail="x",
                request_id="",
                client_ip=None,
            )
            await op.execute()

            (row,) = (await log.get_entries())[0]
            assert row.get("request_id") in (None, "")
        finally:
            await pm.close()


class TestCredentialsAreNotKept:
    """A client that pastes a key into a request and breaks it must not leave the key."""

    KEY = b"sk-ant-api03-CANARY1234567890abcdef"

    async def _stored_body(self, served) -> dict:
        dumps, total = await served.pm.query_error_dumps()
        assert total == 1
        stored = await served.pm.get_dump_body(dumps[0]["body_hash"])
        assert stored is not None
        return decompress_body(stored)["_unparsed_body"]

    async def test_a_key_in_an_unparseable_body_is_redacted_in_the_dump(self, served):
        body = b'{"model": "m", "api_key": "' + self.KEY + b'", "messages": ['

        await served.send("POST", "/v1/chat/completions", body)

        sample = await self._stored_body(served)
        assert self.KEY.decode() not in sample["text"]
        assert "[REDACTED]" in sample["text"]
        assert sample["sha256"] == hashlib.sha256(body).hexdigest(), (
            "the digest stays that of the original bytes"
        )
        (row,) = await served.rows()
        assert self.KEY.decode() not in row["error_detail"]

    async def test_a_key_in_a_non_utf8_body_is_redacted_and_the_rest_kept(self, served):
        body = b'{"x": "\xff\xfe", "api_key": "' + self.KEY + b'", "m": ['

        await served.send("POST", "/v1/chat/completions", body)

        sample = await self._stored_body(served)
        kept = base64.b64decode(sample["base64"])
        assert self.KEY not in kept
        assert b"[REDACTED]" in kept
        assert b'{"x": "\xff\xfe"' in kept, "the bytes around the key are untouched"
        assert sample["sha256"] == hashlib.sha256(body).hexdigest()

    async def test_a_key_in_a_refused_request_line_is_redacted_in_the_row(self, served):
        status, _, _ = await served.send_raw(self.KEY + b"\r\n\r\n")

        assert status == 400
        (row,) = await served.rows()
        assert self.KEY.decode() not in row["error_detail"]
        assert "[REDACTED]" in row["error_detail"]


# Assembled at import so no scanner mistakes a fixture for a leaked credential.
_OPAQUE = "-".join(["opaque", "fixture", "value", "0042"])
_PAIR = "".join(["dXNl", "cjpw", "YXNz", "d29y", "ZA=="])


class TestCredentialBoundary:
    """What the body scrub recognises, and the line it does not cross."""

    _stored_body = TestCredentialsAreNotKept._stored_body

    @pytest.mark.parametrize(
        "field", ["api_key", "Authorization", "token", "secret", "password"]
    )
    async def test_a_credential_named_field_is_redacted_whatever_the_value(
        self, served, field
    ):
        body = b'{"' + field.encode() + b'": "short-1234", "messages": ['

        await served.send("POST", "/v1/chat/completions", body)

        sample = await self._stored_body(served)
        assert "short-1234" not in sample["text"]
        assert field in sample["text"], "the field name stays, only the value goes"

    @pytest.mark.parametrize(
        ("line", "leaked"),
        [
            (f"Authorization: Bearer {_OPAQUE}", _OPAQUE),
            (f"Authorization: Basic {_PAIR}", _PAIR),
            ("authorization: opaque-internal-token-77", "opaque-internal-token-77"),
            ("Cookie: session=abcdef123456ghijkl; Path=/", "abcdef123456ghijkl"),
            (
                'Authorization: Digest username="u", nonce="n", '
                'response="deadbeefcafe1234"',
                "deadbeefcafe1234",
            ),
            (
                'Authorization: Digest username="u", uri="/a?b=1&c=2", '
                'response="deadbeefcafe1234"',
                "deadbeefcafe1234",
            ),
            (
                "Authorization: Digest username=u, uri=/a?b=1&c=2, "
                "response=deadbeefcafe1234",
                "deadbeefcafe1234",
            ),
            (
                "Proxy-Authorization: AWS4-HMAC-SHA256 "
                "Credential=a/20260101/x, Signature=abc123def456",
                "abc123def456",
            ),
            (
                'authorization="Digest username="u", nonce="n", '
                'response="feedface1234",',
                "feedface1234",
            ),
            ("Set-Cookie: sid=qwerty987654; HttpOnly", "qwerty987654"),
            ('{"cookie": "sid=qwerty987654; a=b"', "qwerty987654"),
        ],
    )
    async def test_a_header_line_pasted_into_a_body_keeps_no_credential(
        self, served, line, leaked
    ):
        body = b"{" + line.encode() + b'\n"messages": ['

        await served.send("POST", "/v1/chat/completions", body)

        sample = await self._stored_body(served)
        assert leaked not in sample["text"]
        assert "[REDACTED]" in sample["text"]

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (f"Authorization: Bearer {_OPAQUE}", "Authorization: [REDACTED]"),
            (
                'Authorization: Digest username="u", response="deadbeef"',
                "Authorization: [REDACTED]",
            ),
            (
                # A Digest ``uri`` carries the query string, so a ``&`` in a
                # pasted header line is part of the credential, not a boundary.
                'Authorization: Digest username="u", uri="/a?b=1&c=2", '
                'response="deadbeef"',
                "Authorization: [REDACTED]",
            ),
            (
                "Authorization: Digest username=u, uri=/a?b=1&c=2, response=deadbeef",
                "Authorization: [REDACTED]",
            ),
            (
                "Proxy-Authorization: AWS4-HMAC-SHA256 Credential=a/x, "
                "Signature=abc123",
                "Proxy-Authorization: [REDACTED]",
            ),
            (
                "authorization=Bearer+tok&model=gpt-4&stream=true",
                "authorization=[REDACTED]&model=gpt-4&stream=true",
            ),
            (
                '{"model":"m","authorization":"Digest username="u", '
                'response="deadbeef","messages":[',
                '{"model":"m","authorization":"[REDACTED]"',
            ),
            (f"authorization=Bearer+{_OPAQUE}", "authorization=[REDACTED]"),
            ("Cookie: session=abc123; Path=/", "Cookie: [REDACTED]"),
            (
                '{"api_key": "x", "max_tokens": 9}',
                '{"api_key": "[REDACTED]", "max_tokens": 9}',
            ),
        ],
    )
    def test_the_scrub_output_is_exact(self, text, expected):
        assert scrub_body_text(text) == expected

    def test_the_scrub_leaves_a_similarly_named_header_alone(self):
        text = '{"X-Authorization-Id": "7", "authorization_note": "draft"}'

        assert scrub_body_text(text) == text

    async def test_an_unquoted_value_is_redacted_too(self, served):
        await served.send("POST", "/v1/chat/completions", b"api_key=short-1234&x=")

        sample = await self._stored_body(served)
        assert "short-1234" not in sample["text"]

    async def test_a_field_that_only_contains_the_word_is_kept(self, served):
        body = b'{"max_tokens": 4096, "model": "m", "messages": ['

        await served.send("POST", "/v1/chat/completions", body)

        sample = await self._stored_body(served)
        assert '"max_tokens": 4096' in sample["text"]

    async def test_a_configured_key_is_redacted_wherever_it_appears(self, holding_keys):
        body = b'{"messages": [{"content": "my key is canary-two-91bc ok"}'

        await holding_keys.send("POST", "/v1/chat/completions", body)

        sample = await TestCredentialsAreNotKept._stored_body(
            holding_keys, holding_keys
        )
        assert "canary-two-91bc" not in sample["text"]
        assert "my key is [REDACTED] ok" in sample["text"]

    async def test_a_configured_key_is_redacted_in_a_non_utf8_body(self, holding_keys):
        body = b'{"x": "\xff", "c": "canary-one-7f3a", "m": ['

        await holding_keys.send("POST", "/v1/chat/completions", body)

        sample = await TestCredentialsAreNotKept._stored_body(
            holding_keys, holding_keys
        )
        kept = base64.b64decode(sample["base64"])
        assert b"canary-one-7f3a" not in kept
        assert b'"x": "\xff"' in kept

    async def test_an_unconfigured_key_under_an_ordinary_name_is_not_recognised(
        self, served
    ):
        body = b'{"note": "canary-three-55de", "messages": ['

        await served.send("POST", "/v1/chat/completions", body)

        sample = await self._stored_body(served)
        assert "canary-three-55de" in sample["text"], (
            "the documented boundary: not configured, not credential-named, "
            "not vendor-shaped"
        )


class TestWithoutAnOpsContext:
    async def test_the_missing_context_is_reported_once(self, served, monkeypatch):
        monkeypatch.setattr(rejected_module, "_warned_without_ops_context", False)
        records: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        handler = _Capture(level=logging.WARNING)
        rejected_module.logger.addHandler(handler)
        try:
            served.app.ops_ctx = None
            await served.send("GET", "/nowhere")
            await served.send_raw(b"garbage\r\n\r\n")
            await served.send("GET", "/nowhere")
        finally:
            rejected_module.logger.removeHandler(handler)

        assert await served.rows() == []
        warnings = [r for r in records if "not being recorded" in r.getMessage()]
        assert len(warnings) == 1


class TestTheSourceIsTheRoutesFormat:
    """A refusal names the same source as a request that reached the route."""

    @pytest.mark.parametrize(
        ("path", "source"),
        [
            ("/v1/chat/completions", "chat_completions"),
            ("/v1/messages", "messages"),
            ("/v1/responses", "open_responses"),
            ("/v1beta/models/m:generateContent", "google_generate"),
            ("/v1beta/interactions", "google_interactions"),
        ],
    )
    async def test_a_refused_request_carries_its_routes_format(
        self, served, path, source
    ):
        status, _, _ = await served.send("GET", path)

        assert status == 405
        (row,) = await served.rows()
        assert row["source_provider"] == source

    async def test_it_matches_what_a_routed_request_records(self, served):
        await served.send("POST", "/v1/chat/completions", b'{"messages": []}')
        await served.send("GET", "/v1/chat/completions")

        routed, refused = sorted(await served.rows(), key=lambda r: r["status_code"])
        assert routed["status_code"] == 400 and refused["status_code"] == 405
        assert refused["source_provider"] == routed["source_provider"]

    async def test_a_read_refusal_on_a_chat_route_carries_it_too(self, served):
        raw = b"POST /v1/messages HTTP/1.1\r\nHost: t\r\nContent-Length: 100\r\n\r\n{"
        await served.send_raw(raw)

        (row,) = await served.rows()
        assert row["source_provider"] == "messages"

    async def test_a_path_that_is_no_chat_route_has_no_source(self, served):
        await served.send("GET", "/nowhere")
        await served.send_raw(b"garbage\r\n\r\n")

        assert {row["source_provider"] for row in await served.rows()} == {""}
