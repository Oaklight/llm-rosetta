"""Tests for the conformance harness (``tools/conformance``)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
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


def test_drift_probe_fails_on_new_drift_and_compares_across_a_version_change():
    data = {"drift": {"T": ["b"]}, "collisions": []}
    baseline = {
        "sdk_versions": {"pkg": "1.0"},
        "providers": {"p": {"drift": {"T": ["a"]}}},
    }

    with pytest.raises(ProbeFailure, match="new drift"):
        _DriftProbe("p", data, baseline, {"pkg": "1.0"}).run()

    # A release that adds no field passes, and says which pair it compared:
    # a stale baseline must not disable the suite (issue #1389).
    result = _DriftProbe(
        "p", {"drift": {"T": ["a"]}, "collisions": []}, baseline, {"pkg": "2.0"}
    ).run()
    assert result["new"] == {}
    assert result["baseline_sdk_versions"] == {"pkg": "1.0"}
    assert result["sdk_versions"] == {"pkg": "2.0"}
    assert "compared anyway" in result["note"]

    with pytest.raises(ProbeSkip, match="no baseline"):
        _DriftProbe("p", data, {}, {"pkg": "1.0"}).run()


def test_a_version_change_does_not_hide_new_drift():
    """The two halves in one run: a moved version *and* a new field.

    Comparing only when the versions match would satisfy the benign-release
    case above and quietly restore the skip.
    """
    baseline = {
        "sdk_versions": {"pkg": "1.0"},
        "providers": {"p": {"drift": {"T": ["a"]}}},
    }

    with pytest.raises(ProbeFailure, match="new drift"):
        _DriftProbe(
            "p",
            {"drift": {"T": ["a", "b"]}, "collisions": []},
            baseline,
            {"pkg": "2.0"},
        ).run()


def _workflow_texts() -> dict[str, str]:
    """Every workflow, by file name."""
    root = Path(__file__).resolve().parents[2] / ".github/workflows"
    return {path.name: path.read_text() for path in sorted(root.glob("*.yml"))}


def test_every_suite_has_a_workflow_that_runs_it():
    """A suite no job runs is asleep, and every test here stays green.

    The drift step is the one that keeps this suite from silently skipping
    (issue #1389); the same rule holds for the rest.  A suite is run by *some*
    workflow — the credentialled one has its own, so this guard reads all of
    them, not only PR CI.
    """
    texts = _workflow_texts()

    missing = [
        name
        for name in SUITES
        if not any(f"--suite {name}" in text for text in texts.values())
    ]
    assert not missing, f"no workflow runs: {', '.join(missing)}"


def _triggers(text: str) -> dict[str, list[str]]:
    """The event keys of a workflow's top-level ``on:`` block, with their lines.

    A tiny reader rather than a YAML dependency: PyYAML is not installed by
    every test environment (the docs test skips without it), and the guard
    needs the trigger keys only.  Each event maps to the indented lines below
    it, so ``schedule:`` with no cron is distinguishable from one with.

    Args:
        text: A workflow file's text.

    Returns:
        ``{event: [its deeper-indented lines]}``.
    """
    lines = text.splitlines()
    start = next(index for index, line in enumerate(lines) if line.rstrip() == "on:")
    events: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines[start + 1 :]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            break
        stripped = line.strip()
        if indent == 2:
            # ``key:`` and the flow spelling ``key: {}`` both register the
            # event; a reader that only accepted the first would let
            # ``pull_request: {}`` through the exact-set assertion.
            key, _, value = stripped.partition(":")
            current = key.strip()
            events[current] = [value.strip()] if value.strip() else []
        elif current is not None:
            events[current].append(stripped)
    return events


def test_the_trigger_reader_sees_each_event():
    text = (
        "on:\n"
        "  schedule:\n"
        '    - cron: "1 2 * * 3" # weekly\n'
        "  workflow_dispatch:\n"
        "\n"
        "jobs:\n"
        "  x:\n"
        "    steps: []\n"
    )

    assert _triggers(text) == {
        "schedule": ['- cron: "1 2 * * 3" # weekly'],
        "workflow_dispatch": [],
    }


def test_the_trigger_reader_sees_a_flow_style_event():
    """`pull_request: {}` is a trigger GitHub honours and a text scan misses."""
    text = "on:\n  pull_request: {}\n  workflow_dispatch:\n"

    assert _triggers(text) == {"pull_request": ["{}"], "workflow_dispatch": []}


def test_the_live_suite_is_scheduled_and_not_in_pr_ci():
    """A credentialled probe must not run on a pull request.

    The live suite leaves the machine and needs secrets, so exactly one
    scheduled workflow may name it, its triggers must be exactly the weekly
    schedule and manual dispatch — a ``pull_request`` or ``push`` trigger would
    start a credentialled sweep no one asked for, and an empty ``schedule:``
    would never run at all — and PR CI may not name it either.
    """
    texts = _workflow_texts()
    runners = [name for name, text in texts.items() if "--suite live" in text]

    assert "--suite live" not in texts["ci.yml"]
    assert len(runners) == 1, f"the live suite is named by {runners}"
    triggers = _triggers(texts[runners[0]])
    assert set(triggers) == {"schedule", "workflow_dispatch"}, triggers
    assert any("23 4 * * 1" in line for line in triggers["schedule"]), triggers


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
