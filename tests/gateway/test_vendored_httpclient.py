"""Tests for the vendored httpclient's SSL context cache and lock."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from llm_rosetta_gateway._vendor import httpclient

_ROOT = Path(__file__).resolve().parents[2]
_VENDOR = (
    _ROOT
    / "packages"
    / "llm-rosetta-gateway"
    / "src"
    / "llm_rosetta_gateway"
    / "_vendor"
)


@pytest.fixture(autouse=True)
def _clear_cache():
    httpclient._SSL_CONTEXT_CACHE.clear()
    yield
    httpclient._SSL_CONTEXT_CACHE.clear()


def test_ssl_context_cache_reuses_the_same_key(tmp_path, monkeypatch):
    ca = tmp_path / "ca.pem"
    ca.write_text("A" * 10)
    monkeypatch.setenv("SSL_CERT_FILE", str(ca))
    monkeypatch.setenv("SSL_CERT_DIR", "")
    created = []

    def fake_context():
        created.append(1)
        return object()

    monkeypatch.setattr(httpclient.ssl, "create_default_context", fake_context)

    first = httpclient._make_ssl_context(True)
    second = httpclient._make_ssl_context(True)

    assert first is second
    assert len(created) == 1


def test_ssl_context_cache_rebuilds_when_the_bundle_rotates(tmp_path, monkeypatch):
    ca = tmp_path / "ca.pem"
    ca.write_text("A" * 10)
    monkeypatch.setenv("SSL_CERT_FILE", str(ca))
    monkeypatch.setenv("SSL_CERT_DIR", "")
    created = []

    def fake_context():
        created.append(1)
        return object()

    monkeypatch.setattr(httpclient.ssl, "create_default_context", fake_context)

    first = httpclient._make_ssl_context(True)
    ca.write_text("B" * 20)
    second = httpclient._make_ssl_context(True)

    assert second is not first
    assert len(created) == 2
    # Known residual: a same-size edit with the mtime restored is not
    # detected.  A normal rotation replaces the file (mtime and size
    # move); hashing the bundle per connection would reintroduce the
    # cost this cache removes.


def test_verify_false_gets_its_own_context(tmp_path, monkeypatch):
    ca = tmp_path / "ca.pem"
    ca.write_text("A" * 10)
    monkeypatch.setenv("SSL_CERT_FILE", str(ca))
    monkeypatch.setenv("SSL_CERT_DIR", "")
    monkeypatch.setattr(httpclient.ssl, "create_default_context", lambda: object())

    verified = httpclient._make_ssl_context(True)
    unverified = httpclient._make_ssl_context(False)

    assert verified is not unverified
    assert not unverified.check_hostname


def test_vendored_httpclient_matches_the_lock():
    lock = json.loads((_VENDOR / "zerodep.lock.json").read_text("utf-8"))

    assert lock["source"]["ref"] == "v2026.11.4"
    assert lock["modules"]["httpclient"]["version"] == "0.6.4"
    for entry in lock["modules"].values():
        for filename, expected in entry["files"].items():
            actual = hashlib.sha256((_VENDOR / filename).read_bytes()).hexdigest()
            assert actual == expected
