"""Tests for the conformance harness (``tools/conformance``)."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import pytest

from tools.conformance import SCHEMA_VERSION, ProbeFailure, run_suite
from tools.conformance.cli import main
from tools.conformance.model import ProbeResult, ProbeSkip, Status, SuiteReport
from tools.conformance.redact import redact, redact_text
from tools.conformance.suites import SUITES
from tools.conformance.suites.drift import _DriftProbe, compare_provider


class _Pass:
    name = "ok"

    def run(self) -> dict[str, Any]:
        return {"value": 1}


class _Fail:
    name = "fail"

    def run(self) -> None:
        raise ProbeFailure("wrong result")


class _Error:
    name = "error"

    def run(self) -> None:
        raise ValueError("boom")


class _Skip:
    name = "skip"

    def run(self) -> None:
        raise ProbeSkip("not comparable")


class _Slow:
    name = "slow"

    def run(self) -> None:
        time.sleep(0.5)


def test_one_case_never_aborts_the_others():
    report = run_suite("demo", [_Pass(), _Fail(), _Error(), _Skip()], timeout=2)

    assert report.summary == {"pass": 1, "fail": 1, "error": 1, "skip": 1}
    assert not report.ok
    by_case = {result.case: result for result in report.results}
    assert by_case["ok"].data == {"value": 1}
    assert by_case["fail"].status is Status.FAIL
    assert by_case["error"].status is Status.ERROR
    assert "ValueError: boom" in by_case["error"].detail


def test_a_hung_probe_times_out_as_an_error():
    report = run_suite("demo", [_Slow()], timeout=0.05)

    assert report.results[0].status is Status.ERROR
    assert "TimeoutError" in report.results[0].detail


def test_retries_recover_a_transient_error():
    calls = {"n": 0}

    class Flaky:
        name = "flaky"

        def run(self) -> None:
            calls["n"] += 1
            if calls["n"] < 3:
                raise ValueError("transient")

    assert run_suite("demo", [Flaky()], retries=2).results[0].status is Status.PASS
    assert calls["n"] == 3

    calls["n"] = 0
    assert run_suite("demo", [Flaky()], retries=0).results[0].status is Status.ERROR


def test_redaction_covers_the_shapes_that_leak():
    # A header echoed as a mapping: the key names the credential.
    assert redact({"headers": {"x-api-key": "SUPERSECRET12345"}}) == {
        "headers": {"x-api-key": "[REDACTED]"}
    }

    # OAuth-style query parameters.
    assert "SUPERSECRET12345" not in redact_text(
        "https://api.example.com/v1/chat?token=SUPERSECRET12345"
    )

    # A bare occurrence, when the harness knows the secret it resolved.
    assert (
        redact_text("upstream said SUPERSECRET12345", secrets=["SUPERSECRET12345"])
        == "upstream said [REDACTED]"
    )

    # Patterns that need no knowledge of the value.
    assert "secret" not in redact_text("Authorization: Bearer secret-token")
    assert redact_text("https://user:hunter2@example.com/v1") == (
        "https://[REDACTED]@example.com/v1"
    )
    assert "abcdef1234567890abcdef" not in redact_text("sk-ant-abcdef1234567890abcdef")
    assert "0123456789abcdefghijklmnopqrst" not in redact_text(
        "key=AIza0123456789abcdefghijklmnopqrst"
    )


def test_redaction_preserves_nested_shape():
    payload = {
        "url": "https://u:p@h/x",
        "items": [{"key": "sk-abcdef1234567890abcdef"}],
    }
    redacted = redact(payload)

    assert redacted["url"] == "https://[REDACTED]@h/x"
    assert "abcdef1234567890abcdef" not in redacted["items"][0]["key"]
    assert isinstance(redacted["items"], list)


def test_report_is_schema_versioned():
    report = SuiteReport(
        suite="demo",
        generated_at="2026-10-21T00:00:00+00:00",
        environment={"python": "3.10"},
        results=[
            ProbeResult("ok", Status.PASS, 1),
            ProbeResult("bad", Status.FAIL, 2, "detail"),
        ],
    )

    payload = report.to_dict()

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["summary"] == {"pass": 1, "fail": 1, "error": 0, "skip": 0}
    assert payload["results"][1]["status"] == "fail"


def test_compare_reports_only_new_fields():
    current = {"drift": {"T": ["a", "b"], "U": ["x"]}}

    assert compare_provider(current, {"drift": {"T": ["a"], "U": ["x"]}}) == {
        "T": ["b"]
    }
    assert compare_provider(current, {"drift": {"T": ["a", "b"], "U": ["x"]}}) == {}
    assert compare_provider(current, None) == {"T": ["a", "b"], "U": ["x"]}


def test_drift_probe_fails_on_new_drift_and_skips_on_version_change():
    data = {"drift": {"T": ["b"]}, "collisions": []}
    baseline = {
        "sdk_versions": {"pkg": "1.0"},
        "providers": {"p": {"drift": {"T": ["a"]}}},
    }

    with pytest.raises(ProbeFailure, match="new drift"):
        _DriftProbe("p", data, baseline, {"pkg": "1.0"}).run()

    with pytest.raises(ProbeSkip, match="SDK versions changed"):
        _DriftProbe("p", data, baseline, {"pkg": "2.0"}).run()

    with pytest.raises(ProbeSkip, match="no baseline"):
        _DriftProbe("p", data, {}, {"pkg": "1.0"}).run()


def test_cli_writes_the_report(monkeypatch, tmp_path):
    monkeypatch.setitem(SUITES, "demo", lambda: ([_Pass()], {"extra": "yes"}))
    out = tmp_path / "report.json"

    assert main(["run", "--suite", "demo", "--out", str(out)]) == 0

    payload = json.loads(out.read_text())
    assert payload["suite"] == "demo"
    assert payload["environment"]["extra"] == "yes"
    assert payload["results"][0]["status"] == "pass"


def test_cli_redacts_an_environment_credential(monkeypatch, tmp_path):
    class Echo:
        name = "echo"

        def run(self) -> dict[str, str]:
            return {"note": os.environ["TEST_SECRET"]}

    monkeypatch.setitem(SUITES, "demo", lambda: ([Echo()], {"extra": "yes"}))
    monkeypatch.setenv("TEST_SECRET", "SUPERSECRET12345")
    out = tmp_path / "report.json"

    assert (
        main(
            [
                "run",
                "--suite",
                "demo",
                "--out",
                str(out),
                "--secret-env",
                "TEST_SECRET",
            ]
        )
        == 0
    )

    text = out.read_text()
    assert "SUPERSECRET12345" not in text
    assert json.loads(text)["results"][0]["data"]["note"] == "[REDACTED]"
