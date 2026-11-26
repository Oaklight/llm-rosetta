"""An extension installed from a served index, end to end (#1704).

A local HTTP server on a loopback port serves an ``index.json`` and two real
wheels (built by ``tests/_ext_wheels.py``) and doubles as the upstream.  A
gateway config names the source, ``apply_shim_sources`` fetches and installs
over real HTTP, and a request goes through the gateway app and the installed
shim to the upstream.  Nothing leaves the machine.

Two paths are covered: a config that declares ``shims.sources`` and
``shims.install``, and a config that only names a vendor that moved out of core
(``opencode--chat_completions``), whose install the gateway adds by itself,
once with a mirror declared and once with no ``shims`` section at all (the
catalogue's index URL pointed at the local server).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from _ext_wheels import build_wheel
from llm_rosetta.api import configure_extensions_dir, get_shim
from llm_rosetta.contracts import IR_VERSION, SURFACE_VERSION
from llm_rosetta.testing import restore_registry, snapshot_registry
from llm_rosetta_gateway.app import create_app
from llm_rosetta_gateway.config import GatewayConfig
from llm_rosetta_gateway.shims import catalog
from llm_rosetta_gateway.shims import sources as sources_mod
from llm_rosetta_gateway.shims.shim_sources import apply_shim_sources

_CHAT = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
_UPSTREAM_REPLY = {
    "id": "r1",
    "object": "chat.completion",
    "model": "m",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "from the served upstream"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
}


class _Served:
    """A loopback server: static files by path, and a chat upstream."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.posts: list[tuple[str, dict[str, Any]]] = []
        served = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return None

            def _send(self, status: int, body: bytes, kind: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802 - the handler's name
                data = served.files.get(self.path)
                if data is None:
                    self._send(404, b"not found", "text/plain")
                else:
                    self._send(200, data, "application/octet-stream")

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                served.posts.append((self.path, body))
                self._send(
                    200, json.dumps(_UPSTREAM_REPLY).encode(), "application/json"
                )

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def publish(self, source: str, extensions: dict[str, str]) -> str:
        """Serve one wheel per ``{ext_id: shim_name}`` and an ``index.json``; return its URL."""
        index: dict[str, Any] = {
            "index_version": 1,
            "source": source,
            "generated_at": "2026-11-26T08:00:00Z",
            "extensions": {},
        }
        for ext_id, shim in extensions.items():
            wheel = build_wheel(ext_id, "0.1.0", [shim])
            name = f"llm_rosetta_ext_{ext_id}-0.1.0-py3-none-any.whl"
            self.files[f"/dist/{name}"] = wheel
            index["extensions"][ext_id] = [
                {
                    "version": "0.1.0",
                    "url": f"{self.base}/dist/{name}",
                    "sha256": hashlib.sha256(wheel).hexdigest(),
                    "surface": SURFACE_VERSION,
                    "ir": IR_VERSION,
                    "requires_python": ">=3.10",
                    "shims": [shim],
                }
            ]
        self.files["/dist/index.json"] = json.dumps(index).encode()
        return f"{self.base}/dist/index.json"


@pytest.fixture
def served() -> Iterator[_Served]:
    server = _Served()
    server.start()
    yield server
    server.stop()


@pytest.fixture(autouse=True)
def _clean_process_state(tmp_path: Path) -> Iterator[None]:
    """Leave the registry, the extensions directory and the namespace as found."""
    snapshot = snapshot_registry()
    sources_mod._source_shims.clear()
    extensions = tmp_path / "extensions"
    extensions.mkdir()
    configure_extensions_dir(extensions)
    yield
    configure_extensions_dir(None)
    restore_registry(snapshot)
    sources_mod._source_shims.clear()
    for name in [n for n in sys.modules if n.split(".")[0] == "llm_rosetta_ext"]:
        del sys.modules[name]


def _raw(tmp_path: Path, served: _Served, shim: str, **extra: Any) -> dict[str, Any]:
    return {
        "providers": {
            "p": {
                "type": shim,
                "api_key": "k",
                # The installed shim names its own upstream; the test points it
                # at the local server.
                "base_url": f"{served.base}/v1",
            }
        },
        "models": {"m": {"providers": [{"name": "p"}]}},
        "server": {"data_dir": str(tmp_path / "data"), "open_on_no_keys": True},
        **extra,
    }


async def _post(port: int, path: str, body: dict[str, Any]) -> tuple[int, Any]:
    raw = json.dumps(body).encode()
    request = (
        f"POST {path} HTTP/1.1\r\nHost: t\r\nConnection: close\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(raw)}\r\n\r\n"
    ).encode() + raw
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(request)
    await writer.drain()
    writer.write_eof()
    data = await reader.read()
    writer.close()
    head, _, payload = data.partition(b"\r\n\r\n")
    return int(head.decode().split("\r\n")[0].split()[1]), json.loads(payload or b"{}")


async def _request_through_the_gateway(
    config: GatewayConfig, report: Any, config_path: str
) -> tuple[int, Any]:
    app: Any = await create_app(
        config, config_path=config_path, shim_source_report=report
    )
    listener = await asyncio.start_server(app._handle_connection, "127.0.0.1", 0)
    try:
        return await _post(
            listener.sockets[0].getsockname()[1], "/v1/chat/completions", _CHAT
        )
    finally:
        listener.close()
        await listener.wait_closed()
        await app.transport.close()


async def test_a_declared_source_installs_two_extensions_and_serves_a_request(
    tmp_path, served
):
    url = served.publish(
        "demo", {"acme": "acme--chat_completions", "beta": "beta--chat_completions"}
    )
    raw = _raw(
        tmp_path,
        served,
        "acme--chat_completions",
        shims={"sources": [url], "install": ["acme", "beta"]},
    )
    config_path = str(tmp_path / "config.jsonc")

    report = await asyncio.to_thread(apply_shim_sources, raw, config_path)

    assert report is not None and not report.failures()
    assert {s.vendor for s in report.sources if s.ok} == {"acme", "beta"}
    assert get_shim("acme--chat_completions") is not None
    assert get_shim("beta--chat_completions") is not None
    config = GatewayConfig(raw, config_path=config_path)
    assert config.provider_shim_names["p"] == "acme--chat_completions"

    status, payload = await _request_through_the_gateway(config, report, config_path)

    assert status == 200
    assert payload["choices"][0]["message"]["content"] == "from the served upstream"
    assert [path for path, _ in served.posts] == ["/v1/chat/completions"]
    assert served.posts[0][1]["messages"][0]["content"] == "hi"


async def test_a_moved_vendor_the_config_names_is_installed_from_the_declared_mirror(
    tmp_path, served
):
    """The mirror calls itself ``contrib``, so it serves the vendor the config names."""
    url = served.publish("contrib", {"opencode": "opencode--chat_completions"})
    raw = _raw(
        tmp_path,
        served,
        "opencode--chat_completions",
        shims={"sources": [url]},
    )
    config_path = str(tmp_path / "config.jsonc")
    assert get_shim("opencode--chat_completions") is None

    report = await asyncio.to_thread(apply_shim_sources, raw, config_path)

    assert report is not None and not report.failures()
    assert get_shim("opencode--chat_completions") is not None
    config = GatewayConfig(raw, config_path=config_path)
    status, payload = await _request_through_the_gateway(config, report, config_path)
    assert status == 200
    assert payload["choices"][0]["message"]["content"] == "from the served upstream"


async def test_a_moved_vendor_is_installed_with_no_shims_section_at_all(
    tmp_path, served, monkeypatch
):
    """The catalogue's own index URL, pointed at the local server."""
    url = served.publish("contrib", {"opencode": "opencode--chat_completions"})
    monkeypatch.setattr(catalog.ShimSource, "manifest_url", property(lambda self: url))
    raw = _raw(tmp_path, served, "opencode--chat_completions")
    assert "shims" not in raw
    config_path = str(tmp_path / "config.jsonc")
    assert get_shim("opencode--chat_completions") is None

    report = await asyncio.to_thread(apply_shim_sources, raw, config_path)

    assert report is not None and not report.failures()
    assert get_shim("opencode--chat_completions") is not None
    config = GatewayConfig(raw, config_path=config_path)
    status, _payload = await _request_through_the_gateway(config, report, config_path)
    assert status == 200
