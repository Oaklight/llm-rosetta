"""The nightly drift reporter talks to the tracker only when the story changes."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = ROOT / "scripts" / "dev" / "report_shim_drift.py"

spec = importlib.util.spec_from_file_location("report_shim_drift", SCRIPT)
assert spec is not None and spec.loader is not None, SCRIPT
report = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = report
spec.loader.exec_module(report)


def _verdict(*problems: str):
    return report.verdict(tuple(sorted(problems)))


def test_only_problem_lines_are_read():
    text = """\
ok      argo: matches the master copy
warning free: has no copy yet
problem alcf: differs from the master copy —
      line 111 differs
problem argo: another one
problem argo: another one
"""
    assert report.read_problems(text) == (
        "problem alcf: differs from the master copy —",
        "problem argo: another one",
    )


def test_the_fingerprint_does_not_depend_on_the_order_of_the_log():
    left = _verdict("problem a", "problem b")
    right = _verdict("problem b", "problem a")

    assert left.fingerprint == right.fingerprint


def test_a_different_problem_set_has_a_different_fingerprint():
    assert (
        _verdict("problem a").fingerprint
        != _verdict("problem a", "problem b").fingerprint
    )


def test_no_problems_and_no_issue_does_nothing():
    assert (
        report.action_for(current=_verdict(), issue_body=None, scheduled=True)
        == "nothing"
    )


def test_no_problems_closes_an_open_issue_on_a_timer_run():
    body = report.render_body(_verdict("problem a"), run_url=None)

    assert (
        report.action_for(current=_verdict(), issue_body=body, scheduled=True)
        == "close"
    )


def test_no_problems_leaves_the_issue_alone_on_a_dispatch():
    """A rehearsal must not close an issue that is still real."""
    body = report.render_body(_verdict("problem a"), run_url=None)

    assert (
        report.action_for(current=_verdict(), issue_body=body, scheduled=False)
        == "nothing"
    )


def test_the_same_problems_again_say_nothing():
    body = report.render_body(_verdict("problem a"), run_url=None)

    assert (
        report.action_for(
            current=_verdict("problem a"), issue_body=body, scheduled=True
        )
        == "nothing"
    )


def test_a_changed_problem_set_updates_the_same_issue():
    body = report.render_body(_verdict("problem a"), run_url=None)

    assert (
        report.action_for(
            current=_verdict("problem a", "problem b"),
            issue_body=body,
            scheduled=True,
        )
        == "update"
    )


def test_the_first_failure_creates_the_issue():
    assert (
        report.action_for(
            current=_verdict("problem a"), issue_body=None, scheduled=True
        )
        == "create"
    )


def test_the_body_carries_its_fingerprint_and_the_problems():
    current = _verdict("problem a", "problem b")

    body = report.render_body(current, run_url="https://example.invalid/run/1")

    assert report.previous_fingerprint(body) == current.fingerprint
    assert "problem a" in body and "problem b" in body
    assert "https://example.invalid/run/1" in body
    # The marker is invisible when rendered, which is what keeps the issue
    # readable while carrying state.
    assert "<!-- shim-drift-fingerprint: " in body


def test_a_body_without_a_marker_reads_as_unknown():
    assert report.previous_fingerprint("no marker here") is None


def test_a_dry_run_never_calls_the_api(tmp_path, capsys, monkeypatch):
    log = tmp_path / "drift.log"
    log.write_text("problem argo: differs\n", encoding="utf-8")

    def _boom(*args, **kwargs):
        raise AssertionError("dry run must not reach the API")

    monkeypatch.setattr(report, "_api", _boom)

    code = report.main(["--problems", str(log), "--repo", "o/r", "--dry-run"])

    assert code == 1
    assert "action" in capsys.readouterr().out
