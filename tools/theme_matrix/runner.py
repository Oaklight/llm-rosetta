"""Drive the admin panel through scheme × mode × width and record what it renders.

The matrix answers the question the contract test cannot: not "may the CSS say
this" but "what does the panel look like, and does it still work".  Each cell
loads the panel with a scheme and mode planted in ``localStorage``, then:

1. asserts the scheme and mode actually reached the document;
2. resolves every token the scheme declares and fails on any that come back
   empty — a token a scheme forgot is invisible in a text test;
3. checks each tab does not scroll sideways at that width;
4. clicks the Token card's "details" link and requires the modal to open —
   the #1149 regression, as a render-level check rather than a CSS-text one;
5. samples the scheme's colour values so the run can assert afterwards that
   schemes and modes actually render *differently*;
6. screenshots the Dashboard and Providers tabs for a human to compare.

Only step 4 is a click; the rest are computed-style reads.  A cell's
screenshots are artifacts, not golden images, so the harness can run on every
platform without a pixel baseline.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Playwright is imported where it is used, not at module level: the token and
# cell helpers are pure and the contract test suite imports them on machines
# that have no browser (and no Playwright installed).

_REPO_ROOT = Path(__file__).resolve().parents[2]
CSS_DIR = (
    _REPO_ROOT
    / "packages"
    / "llm-rosetta-gateway"
    / "src"
    / "llm_rosetta_gateway"
    / "admin"
    / "css"
)

# A scheme block's body holds declarations only and never nests, so a regex is
# exact here; the contract test's parser is for rules with at-rule nesting.
_SCHEME_BLOCK_RE = re.compile(
    r'\[data-scheme="([^"]+)"\](?:\[data-mode="([^"]+)"\])?\s*\{([^}]*)\}'
)
_CUSTOM_PROPERTY_RE = re.compile(r"(--[\w-]+)\s*:")

# The panel keeps these in localStorage; planting them before navigation is
# how a cell picks its look without touching the Settings UI.
_SCHEME_KEY = "llm-rosetta-scheme"
_MODE_KEY = "llm-rosetta-mode"
_LANG_KEY = "llm-rosetta-lang"


@dataclass(frozen=True)
class Cell:
    """One point of the matrix: a scheme, a mode and a viewport width."""

    scheme: str
    mode: str
    width: int

    @property
    def name(self) -> str:
        """Stable key and file-name stem, e.g. ``emerald-dark-1280``."""
        return f"{self.scheme}-{self.mode}-{self.width}"


@dataclass
class ProbeResult:
    """The outcome of one check inside a cell.

    ``skipped`` marks a check that had nothing to compare — a cross-cell check
    in a run narrowed to one scheme, say.  It is not a pass: the report and the
    console say ``SKIP`` so an absent comparison cannot be read as a green one,
    while a legitimate narrowed run still exits zero.
    """

    name: str
    ok: bool
    detail: str = ""
    skipped: bool = False


@dataclass
class CellResult:
    """Every probe and artifact produced for one cell."""

    cell: Cell
    probes: list[ProbeResult] = field(default_factory=list)
    screenshots: list[str] = field(default_factory=list)
    # Sampled colour tokens (``--bg`` etc.) keyed by property name.  The
    # cross-cell checks compare these to prove schemes and modes differ.
    values: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether every probe in this cell passed."""
        return all(probe.ok for probe in self.probes)


def _read_token_sheet(css_path: Path | None) -> str:
    """Read the token sheet."""
    return (css_path or CSS_DIR / "base.css").read_text(encoding="utf-8")


def _scheme_blocks(css: str) -> list[tuple[str, str | None, str]]:
    """Split a token sheet into ``(scheme, mode, declarations)`` blocks.

    ``mode`` is ``None`` for a scheme's structural block, which is shared by
    light and dark.
    """
    return [
        (scheme, mode or None, body)
        for scheme, mode, body in _SCHEME_BLOCK_RE.findall(css)
    ]


def scheme_tokens(css_path: Path | None = None) -> list[str]:
    """Return every custom property any scheme block declares.

    A scheme must declare all of them: the union is the set the components
    consume, and an empty computed value means this scheme left one out.

    Args:
        css_path: Token stylesheet to read; defaults to ``base.css``.

    Returns:
        Sorted token names, e.g. ``['--accent', '--bg', ...]``.

    Raises:
        RuntimeError: If the stylesheet yields implausibly few tokens, which
            means the block pattern stopped matching and the check would have
            passed vacuously.
    """
    css = _read_token_sheet(css_path)
    tokens: set[str] = set()
    for _, _, body in _scheme_blocks(css):
        tokens.update(_CUSTOM_PROPERTY_RE.findall(body))
    if len(tokens) < 10:
        raise RuntimeError(
            f"only {len(tokens)} scheme tokens found in {css_path or CSS_DIR / 'base.css'}; "
            "the scheme-block pattern no longer matches the token sheet"
        )
    return sorted(tokens)


def declared_schemes(css_path: Path | None = None) -> list[str]:
    """Return every scheme the token sheet declares.

    This is the matrix's default scheme axis rather than a hand-kept list: a
    third scheme added to ``base.css`` is covered the next time the matrix
    runs, so "every scheme" stays true.

    Raises:
        RuntimeError: If no scheme block matches, so an empty axis cannot make
            the matrix pass vacuously.
    """
    css = _read_token_sheet(css_path)
    schemes = sorted({scheme for scheme, _, _ in _scheme_blocks(css)})
    if not schemes:
        raise RuntimeError(
            f"no schemes found in {css_path or CSS_DIR / 'base.css'}; "
            "the scheme-block pattern no longer matches the token sheet"
        )
    return schemes


def declared_modes(css_path: Path | None = None) -> list[str]:
    """Return every mode a scheme block declares, e.g. ``['dark', 'light']``.

    Raises:
        RuntimeError: If no mode block matches, for the same reason as
            :func:`declared_schemes`.
    """
    css = _read_token_sheet(css_path)
    modes = sorted({mode for _, mode, _ in _scheme_blocks(css) if mode})
    if not modes:
        raise RuntimeError(
            f"no modes found in {css_path or CSS_DIR / 'base.css'}; "
            "the scheme-block pattern no longer matches the token sheet"
        )
    return modes


def build_cells(schemes: list[str], modes: list[str], widths: list[int]) -> list[Cell]:
    """Cross the three axes into cells, in a stable order."""
    return [
        Cell(scheme, mode, width)
        for scheme in schemes
        for mode in modes
        for width in widths
    ]


def _launch_browser(playwright: Any) -> Any:
    """Launch Chromium, explaining how to install it when it is absent."""
    from playwright.sync_api import Error as PlaywrightError

    try:
        return playwright.chromium.launch(headless=True)
    except PlaywrightError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "could not launch Chromium; install it with "
            "`python -m playwright install chromium` (add `--with-deps` on a "
            f"bare CI image): {exc}"
        ) from exc


def _open_panel(page: Any, base_url: str, password: str) -> None:
    """Load the panel and get past the login overlay.

    A gateway with no admin password renders the panel directly.  When one is
    set, the caller supplies it; failing to open without one is a setup error,
    not a theme failure.
    """
    page.goto(f"{base_url}/admin", wait_until="domcontentloaded")
    page.wait_for_selector(".tab[data-tab='dashboard']", timeout=15_000)
    page.wait_for_timeout(500)

    overlay = page.locator("#loginOverlay")
    if overlay.is_visible():
        if not password:
            raise RuntimeError(
                "the panel asks for a password but none was supplied; pass "
                "--admin-password or point --base-url at a gateway without one"
            )
        page.locator("#loginPassword").fill(password)
        page.locator("#loginPassword").press("Enter")
        overlay.wait_for(state="hidden", timeout=10_000)


def _probe_scheme(page: Any, cell: Cell) -> ProbeResult:
    """The scheme and mode the browser applied are the ones requested."""
    applied = page.evaluate(
        """() => ({
            scheme: document.documentElement.dataset.scheme ?? '',
            mode: document.documentElement.dataset.mode ?? '',
        })"""
    )
    ok = applied["scheme"] == cell.scheme and applied["mode"] == cell.mode
    return ProbeResult(
        "scheme-applied",
        ok,
        f"applied {applied['scheme']}/{applied['mode']}, "
        f"expected {cell.scheme}/{cell.mode}",
    )


def _probe_tokens(page: Any, tokens: list[str]) -> ProbeResult:
    """Every scheme token resolves to a non-empty computed value."""
    missing = page.evaluate(
        """(names) => {
            const style = getComputedStyle(document.documentElement);
            return names.filter(name => !style.getPropertyValue(name).trim());
        }""",
        tokens,
    )
    return ProbeResult(
        "tokens-resolve",
        not missing,
        f"{len(tokens) - len(missing)}/{len(tokens)} tokens resolve"
        + (f"; unresolved: {', '.join(missing)}" if missing else ""),
    )


# Three tokens that carry the scheme's identity: a background, an accent and a
# surface.  They are sampled so the run can compare cells afterwards; a scheme
# pasted from another one renders without a single visible difference and
# would otherwise pass every per-cell probe.
_SAMPLE_TOKENS = ("--bg", "--accent", "--card-bg")


def _probe_scheme_values(page: Any) -> tuple[ProbeResult, dict[str, str]]:
    """Sample the scheme's identity colours and return them for comparison."""
    values = page.evaluate(
        """(names) => {
            const style = getComputedStyle(document.documentElement);
            return Object.fromEntries(
                names.map(name => [name, style.getPropertyValue(name).trim()]));
        }""",
        list(_SAMPLE_TOKENS),
    )
    present = [name for name in _SAMPLE_TOKENS if values.get(name)]
    return (
        ProbeResult(
            "scheme-values",
            len(present) == len(_SAMPLE_TOKENS),
            f"{len(present)}/{len(_SAMPLE_TOKENS)} sampled tokens resolve "
            f"({', '.join(_SAMPLE_TOKENS)})",
        ),
        values,
    )


def _probe_overflow(page: Any, cell: Cell, tab: str) -> ProbeResult:
    """The tab's page does not scroll sideways at this width.

    Both tabs are measured: the Providers tab carries badges, model rows and
    long names that the Dashboard does not, so a Providers-only overflow would
    otherwise survive as a screenshot nobody diffs.
    """
    metrics = page.evaluate(
        """() => ({
            scrollWidth: document.documentElement.scrollWidth,
            innerWidth: window.innerWidth,
        })"""
    )
    ok = metrics["scrollWidth"] <= metrics["innerWidth"]
    return ProbeResult(
        f"no-horizontal-overflow/{tab}",
        ok,
        f"{tab}: scrollWidth {metrics['scrollWidth']}px, viewport {cell.width}px",
    )


def _probe_token_card_click(page: Any) -> ProbeResult:
    """The Token card's "details" link is hit and opens the modal.

    This is the render-level shape of #1149: emerald's card texture sat above
    the card's contents, so the link's centre hit the card and the click went
    nowhere.  A hit test names the element; the click proves the browser agrees
    with the hit test rather than a synthetic event.
    """
    hit = page.evaluate(
        """() => {
            const link = document.querySelector('.stats-grid .stat-card a');
            if (!link) return {found: false};
            link.scrollIntoView({block: 'center'});
            const rect = link.getBoundingClientRect();
            const el = document.elementFromPoint(
                rect.left + rect.width / 2, rect.top + rect.height / 2);
            return {
                found: true,
                hit: el ? el.tagName + (el.className ? '.' + el.className : '') : '(nothing)',
                isLink: !!el && (el === link || link.contains(el)),
            };
        }"""
    )
    if not hit.get("found"):
        return ProbeResult(
            "token-card-clickable", False, "no details link in the Token card"
        )
    if not hit["isLink"]:
        return ProbeResult(
            "token-card-clickable",
            False,
            f"the link's centre hits {hit['hit']} instead of the link",
        )

    link = page.locator(".stats-grid .stat-card a").first
    link.scroll_into_view_if_needed()
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    try:
        link.click(timeout=3_000)
        page.wait_for_selector("#tokenUsageModal.open", timeout=3_000)
        opened = True
    except PlaywrightTimeoutError:
        opened = False
    finally:
        page.evaluate("() => window.closeModal?.('tokenUsageModal')")
    return ProbeResult(
        "token-card-clickable",
        opened,
        "the click opens the token usage modal"
        if opened
        else "the click did not open the token usage modal",
    )


def _shoot(page: Any, out_dir: Path, name: str) -> str:
    """Screenshot the full page and return the artifact's relative name."""
    file_name = f"{name}.png"
    page.screenshot(path=str(out_dir / file_name), full_page=True)
    return file_name


def _run_cell(
    browser: Any,
    base_url: str,
    cell: Cell,
    out_dir: Path,
    tokens: list[str],
    password: str,
) -> CellResult:
    """Run one cell end to end and return its probes and screenshots."""
    context = browser.new_context(
        viewport={"width": cell.width, "height": 900},
        device_scale_factor=1,
    )
    try:
        # Python's Playwright cannot pass an argument to an init script, so the
        # cell's values are rendered into the source.  json.dumps keeps a
        # scheme name with punctuation a literal rather than code.
        context.add_init_script(
            script=(
                "(() => {\n"
                f"  localStorage.setItem({json.dumps(_SCHEME_KEY)}, "
                f"{json.dumps(cell.scheme)});\n"
                f"  localStorage.setItem({json.dumps(_MODE_KEY)}, "
                f"{json.dumps(cell.mode)});\n"
                f'  localStorage.setItem({json.dumps(_LANG_KEY)}, "en");\n'
                "})();"
            )
        )
        page = context.new_page()
        page.set_default_timeout(15_000)
        _open_panel(page, base_url, password)

        result = CellResult(cell=cell)
        result.probes.append(_probe_scheme(page, cell))
        result.probes.append(_probe_tokens(page, tokens))
        values_probe, result.values = _probe_scheme_values(page)
        result.probes.append(values_probe)

        # Dashboard: the KPI grid the scheme textures and the Token card's
        # details link, which is the click the contract must keep working.
        page.locator(".tab[data-tab='dashboard']").click()
        page.wait_for_selector(".stats-grid .stat-card", timeout=15_000)
        page.wait_for_timeout(1_500)
        result.probes.append(_probe_token_card_click(page))
        result.probes.append(_probe_overflow(page, cell, "dashboard"))
        result.screenshots.append(_shoot(page, out_dir, f"{cell.name}-dashboard"))

        # Providers: the card surface — texture, borders, badges and buttons,
        # and the tab where wide content lives, so it gets its own overflow
        # probe rather than only a screenshot.
        page.locator(".tab[data-tab='providers']").click()
        page.wait_for_selector(".provider-card[data-provider]", timeout=15_000)
        page.wait_for_timeout(500)
        result.probes.append(_probe_overflow(page, cell, "providers"))
        result.screenshots.append(_shoot(page, out_dir, f"{cell.name}-providers"))
        return result
    finally:
        context.close()


def _probe_summary(result: CellResult) -> str:
    """One table cell's worth of probe status."""
    failed = [probe for probe in result.probes if not probe.ok]
    if not failed:
        return f"all {len(result.probes)} pass"
    return "; ".join(f"{probe.name}: {probe.detail}" for probe in failed)


def _check_axis_difference(results: list[CellResult], axis: str) -> ProbeResult:
    """Assert that cells differing only on *axis* render differently.

    ``scheme-values`` proves each cell has values; this proves they are the
    *scheme's* values.  A block copy-pasted from another scheme keeps every
    per-cell probe green while changing nothing on screen, which is exactly
    the failure a text-level test cannot see.
    """
    if axis == "scheme":
        name = "schemes-render-differently"
        group_key = lambda result: (result.cell.mode, result.cell.width)  # noqa: E731
    else:
        name = "modes-render-differently"
        group_key = lambda result: (result.cell.scheme, result.cell.width)  # noqa: E731

    groups: dict[tuple[Any, int], list[CellResult]] = defaultdict(list)
    for result in results:
        groups[group_key(result)].append(result)

    problems: list[str] = []
    compared = 0
    for group in groups.values():
        for index, left in enumerate(group):
            for right in group[index + 1 :]:
                compared += 1
                shared = sorted(set(left.values) & set(right.values))
                if shared and all(
                    left.values[token] == right.values[token] for token in shared
                ):
                    problems.append(
                        f"{left.cell.name} and {right.cell.name} agree on "
                        f"{', '.join(shared)}"
                    )

    if compared == 0:
        return ProbeResult(
            name,
            True,
            f"no pairs to compare (one {axis} in the matrix)",
            skipped=True,
        )
    if problems:
        return ProbeResult(name, False, "; ".join(problems))
    return ProbeResult(
        name, True, f"{compared} pairs differ on at least one sampled token"
    )


def _verdict(probe: ProbeResult) -> str:
    """The word for a probe's outcome in the console and the report."""
    if probe.skipped:
        return "SKIP"
    return "PASS" if probe.ok else "FAIL"


def _write_report(
    out_dir: Path,
    base_url: str,
    source: str,
    results: list[CellResult],
    tokens: list[str],
    checks: list[ProbeResult],
) -> dict[str, Any]:
    """Write ``report.json`` and ``report.md``; return the JSON document."""
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report: dict[str, Any] = {
        "generated_at": generated,
        "base_url": base_url,
        "gateway": source,
        "tokens": tokens,
        "ok": all(result.ok for result in results)
        and all(check.ok for check in checks),
        "checks": [
            {
                "name": check.name,
                "ok": check.ok,
                "skipped": check.skipped,
                "detail": check.detail,
            }
            for check in checks
        ],
        "cells": [
            {
                "scheme": result.cell.scheme,
                "mode": result.cell.mode,
                "width": result.cell.width,
                "ok": result.ok,
                "probes": [
                    {
                        "name": probe.name,
                        "ok": probe.ok,
                        "skipped": probe.skipped,
                        "detail": probe.detail,
                    }
                    for probe in result.probes
                ],
                "screenshots": result.screenshots,
            }
            for result in results
        ],
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Admin theme matrix",
        "",
        f"- Generated: {generated}",
        f"- Gateway: `{base_url}` ({source})",
        f"- Result: **{'PASS' if report['ok'] else 'FAIL'}**",
        f"- Cells: {len(results)}",
        "",
        "| check | result | detail |",
        "| --- | --- | --- |",
        *[
            f"| {check.name} | "
            f"{'**FAIL**' if not check.ok and not check.skipped else _verdict(check).lower()} "
            f"| {check.detail} |"
            for check in checks
        ],
        "",
        "| scheme | mode | width | dashboard | providers | probes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        name = result.cell.name
        mark = "pass" if result.ok else "**FAIL**"
        lines.append(
            f"| {result.cell.scheme} | {result.cell.mode} | {result.cell.width} "
            f"| ![{name} dashboard]({name}-dashboard.png) "
            f"| ![{name} providers]({name}-providers.png) "
            f"| {mark} — {_probe_summary(result)} |"
        )
    lines.extend(
        [
            "",
            "Screenshots are full-page captures of the Dashboard and Providers",
            "tabs at each cell. Probes are the assertions: the scheme and mode",
            "reached the document, every scheme token resolved, each tab does",
            "not scroll sideways, and the Token card's details link opens its",
            "modal. The cross-cell checks assert that a scheme change changes",
            "the sampled colours, so a copy-pasted block cannot pass.",
            "",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def run_matrix(
    base_url: str,
    out_dir: Path,
    cells: list[Cell],
    *,
    password: str = "",
    source: str = "spawned",
) -> dict[str, Any]:
    """Run every cell and write the report.

    Args:
        base_url: Root URL of a running gateway (no trailing slash).
        out_dir: Directory for screenshots and the report.
        cells: The matrix to run.
        password: Admin password, when the gateway has one.
        source: How the gateway was obtained, recorded in the report.

    Returns:
        The report document.  ``report["ok"]`` is false when any probe failed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens = scheme_tokens()
    results: list[CellResult] = []

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is not installed; run `pip install 'playwright>=1.40,<2'` "
            "and then `python -m playwright install chromium`."
        ) from exc

    with sync_playwright() as playwright:
        browser = _launch_browser(playwright)
        try:
            for index, cell in enumerate(cells, start=1):
                print(f"[{index}/{len(cells)}] {cell.name}", flush=True)
                result = _run_cell(browser, base_url, cell, out_dir, tokens, password)
                for probe in result.probes:
                    print(
                        f"    {_verdict(probe):4}  {probe.name}: {probe.detail}",
                        flush=True,
                    )
                results.append(result)
        finally:
            browser.close()

    checks = [
        _check_axis_difference(results, "scheme"),
        _check_axis_difference(results, "mode"),
    ]
    for check in checks:
        print(
            f"{_verdict(check):4}  {check.name}: {check.detail}",
            flush=True,
        )

    return _write_report(out_dir, base_url, source, results, tokens, checks)
