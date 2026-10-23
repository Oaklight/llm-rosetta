"""Tests for the theme matrix's pure half (``tools/theme_matrix``).

The browser half needs Playwright and a running gateway, so it runs from
``tools/theme_matrix`` and CI's dedicated job.  These tests pin what a refactor
could break silently: which tokens a scheme must resolve and what the matrix
axes expand to.
"""

from __future__ import annotations

import argparse
from typing import Any

import pytest

from tools.theme_matrix import cli
from tools.theme_matrix.cli import _widths
from tools.theme_matrix.runner import (
    Cell,
    CellResult,
    ProbeResult,
    _check_axis_difference,
    _probe_overflow,
    _probe_scheme,
    _probe_scheme_values,
    _probe_summary,
    _probe_token_card_click,
    _probe_tokens,
    _verdict,
    build_cells,
    declared_modes,
    declared_schemes,
    scheme_tokens,
)


def test_scheme_tokens_reads_every_token_the_schemes_declare():
    """The real token sheet is the source, not a hand-kept list."""
    tokens = scheme_tokens()
    # One structural token and one colour token: the two declaration styles
    # in base.css, and the ones a components change would break first.
    assert "--card-texture" in tokens
    assert "--accent" in tokens
    assert "--stats-gap" in tokens


def test_scheme_tokens_is_the_union_across_schemes(tmp_path):
    """Adding a scheme with one extra token extends the check to all cells."""
    declarations = "\n".join(f"  --t{i}: {i};" for i in range(9))
    css = tmp_path / "base.css"
    css.write_text(
        '[data-scheme="one"] {\n'
        f"{declarations}\n"
        "  --shared: 1;\n"
        "}\n"
        '[data-scheme="two"][data-mode="dark"] {\n'
        "  --shared: 2;\n"
        "  --only-two: 3;\n"
        "}\n"
        ":root { --not-a-scheme-token: 0; }\n",
        encoding="utf-8",
    )
    tokens = scheme_tokens(css)
    assert "--shared" in tokens
    assert "--only-two" in tokens
    # A custom property outside a scheme block is not part of the contract.
    assert "--not-a-scheme-token" not in tokens
    assert len(tokens) == 11


def test_scheme_tokens_refuses_a_pattern_that_matches_nothing(tmp_path):
    """A parser that stops matching must fail, not pass vacuously."""
    css = tmp_path / "base.css"
    css.write_text(":root { --one: 1; }\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="scheme tokens found"):
        scheme_tokens(css)


def test_build_cells_crosses_the_axes_in_a_stable_order():
    cells = build_cells(["minimal", "emerald"], ["light", "dark"], [390, 1280])
    assert [cell.name for cell in cells] == [
        "minimal-light-390",
        "minimal-light-1280",
        "minimal-dark-390",
        "minimal-dark-1280",
        "emerald-light-390",
        "emerald-light-1280",
        "emerald-dark-390",
        "emerald-dark-1280",
    ]


def test_build_cells_is_empty_when_any_axis_is_empty():
    assert build_cells([], ["light"], [1280]) == []
    assert build_cells(["minimal"], [], [1280]) == []
    assert build_cells(["minimal"], ["light"], []) == []


def test_widths_rejects_a_non_positive_viewport():
    assert _widths("390, 1280") == [390, 1280]
    with pytest.raises(argparse.ArgumentTypeError):
        _widths("0")
    with pytest.raises(argparse.ArgumentTypeError):
        _widths("")


def test_probe_summary_names_every_failure():
    cell = Cell("minimal", "light", 1280)
    passing = CellResult(
        cell=cell,
        probes=[ProbeResult("scheme-applied", True, "ok")],
    )
    assert _probe_summary(passing) == "all 1 pass"

    failing = CellResult(
        cell=cell,
        probes=[
            ProbeResult("scheme-applied", True, "ok"),
            ProbeResult(
                "tokens-resolve", False, "1/2 tokens resolve; unresolved: --bg"
            ),
        ],
    )
    assert "tokens-resolve" in _probe_summary(failing)
    assert "--bg" in _probe_summary(failing)


def test_declared_axes_come_from_the_token_sheet():
    """The default axes are the vocabulary the stylesheet declares."""
    assert declared_schemes() == ["emerald", "minimal"]
    assert declared_modes() == ["dark", "light"]


def test_declared_schemes_ignores_blocks_outside_the_token_sheet(tmp_path):
    css = tmp_path / "base.css"
    css.write_text(
        '[data-scheme="paper"] { --bg: #fff; }\n'
        '[data-scheme="paper"][data-mode="night"] { --bg: #000; }\n'
        '[data-scheme="paper"][data-mode="day"] { --bg: #fff; }\n'
        ".not-a-scheme { --ignored: 1; }\n",
        encoding="utf-8",
    )
    assert declared_schemes(css) == ["paper"]
    assert declared_modes(css) == ["day", "night"]


def test_declared_schemes_refuses_to_return_an_empty_axis(tmp_path):
    css = tmp_path / "base.css"
    css.write_text(":root { --bg: #fff; }\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no schemes found"):
        declared_schemes(css)
    with pytest.raises(RuntimeError, match="no modes found"):
        declared_modes(css)


class _StubPage:
    """Just enough of a Playwright page for the computed-style probes."""

    def __init__(self, result: Any = None) -> None:
        self._result = result

    def evaluate(self, *_args: Any) -> Any:
        return self._result


def test_probe_scheme_passes_when_the_requested_pair_applied():
    cell = Cell("emerald", "dark", 1280)
    probe = _probe_scheme(_StubPage({"scheme": "emerald", "mode": "dark"}), cell)
    assert probe.ok

    stale = _probe_scheme(_StubPage({"scheme": "minimal", "mode": "dark"}), cell)
    assert not stale.ok
    assert "minimal/dark" in stale.detail


def test_probe_tokens_names_every_unresolved_token():
    passing = _probe_tokens(_StubPage([]), ["--bg", "--accent"])
    assert passing.ok
    assert "2/2" in passing.detail

    failing = _probe_tokens(_StubPage(["--accent"]), ["--bg", "--accent"])
    assert not failing.ok
    assert "--accent" in failing.detail


def test_probe_overflow_names_the_tab_it_measured():
    cell = Cell("minimal", "light", 390)
    dashboard = _probe_overflow(
        _StubPage({"scrollWidth": 375, "innerWidth": 390}), cell, "dashboard"
    )
    assert dashboard.ok
    assert dashboard.name == "no-horizontal-overflow/dashboard"

    providers = _probe_overflow(
        _StubPage({"scrollWidth": 2014, "innerWidth": 390}), cell, "providers"
    )
    assert not providers.ok
    assert providers.name == "no-horizontal-overflow/providers"
    assert "2014" in providers.detail


def test_probe_scheme_values_returns_the_samples_for_comparison():
    page = _StubPage({"--bg": "#fff", "--accent": "#111", "--card-bg": "#fafafa"})
    probe, values = _probe_scheme_values(page)
    assert probe.ok
    assert values == {"--bg": "#fff", "--accent": "#111", "--card-bg": "#fafafa"}

    missing, values = _probe_scheme_values(_StubPage({"--bg": "#fff"}))
    assert not missing.ok
    assert "--accent" in missing.detail


def test_token_card_probe_fails_when_the_centre_hits_the_card():
    blocked = _probe_token_card_click(
        _StubPage({"found": True, "isLink": False, "hit": "DIV.stat-card"})
    )
    assert not blocked.ok
    assert "DIV.stat-card" in blocked.detail

    absent = _probe_token_card_click(_StubPage({"found": False}))
    assert not absent.ok
    assert "no details link" in absent.detail


def _cell_result(cell: Cell, **values: str) -> CellResult:
    return CellResult(cell=cell, values=values)


def test_axis_check_fails_when_two_schemes_render_identically():
    results = [
        _cell_result(
            Cell("minimal", "light", 1280), **{"--bg": "#fff", "--accent": "#111"}
        ),
        _cell_result(
            Cell("emerald", "light", 1280), **{"--bg": "#fff", "--accent": "#111"}
        ),
    ]
    check = _check_axis_difference(results, "scheme")
    assert not check.ok
    assert check.name == "schemes-render-differently"
    assert "agree on" in check.detail


def test_axis_check_passes_when_the_schemes_differ():
    results = [
        _cell_result(
            Cell("minimal", "light", 1280), **{"--bg": "#fff", "--accent": "#111"}
        ),
        _cell_result(
            Cell("emerald", "light", 1280), **{"--bg": "#fff", "--accent": "#34d59a"}
        ),
    ]
    check = _check_axis_difference(results, "scheme")
    assert check.ok
    assert "1 pairs differ" in check.detail


def test_axis_check_fails_when_two_modes_render_identically():
    results = [
        _cell_result(Cell("minimal", "light", 1280), **{"--bg": "#fff"}),
        _cell_result(Cell("minimal", "dark", 1280), **{"--bg": "#fff"}),
    ]
    check = _check_axis_difference(results, "mode")
    assert not check.ok
    assert check.name == "modes-render-differently"


def test_axis_check_with_one_scheme_is_skipped_not_passed():
    """A narrowed run says SKIP, so an absent comparison cannot read as green."""
    results = [_cell_result(Cell("minimal", "light", 1280), **{"--bg": "#fff"})]
    check = _check_axis_difference(results, "scheme")
    assert check.ok
    assert check.skipped
    assert _verdict(check) == "SKIP"
    assert "no pairs" in check.detail


def test_verdict_names_the_three_outcomes():
    assert _verdict(ProbeResult("p", True)) == "PASS"
    assert _verdict(ProbeResult("p", False)) == "FAIL"
    assert _verdict(ProbeResult("p", True, skipped=True)) == "SKIP"


def test_main_prints_a_setup_failure_without_a_traceback(monkeypatch, capsys):
    """A first-time user reads the sentence, not a stack trace."""

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("Playwright is not installed")

    monkeypatch.setattr(cli, "run_matrix", boom)
    code = cli.main(["--base-url", "http://127.0.0.1:1", "--widths", "390"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.err.startswith("error: Playwright is not installed")
    assert "Traceback" not in captured.err
