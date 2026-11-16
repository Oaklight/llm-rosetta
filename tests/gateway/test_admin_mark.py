"""The console draws the shipped stone mark (issue #1572).

The design files carry the 2026-11-14 draft: 128 points in a ``634.78``
square (``design/logo/out/favicon.svg``).  The console kept an older 48-point
trace in its favicon, its two inline marks and its data-URI page icon.

These checks pin what the fix is about -- the window, the point count, the
geometry itself, and that the inline marks still take their colour from the
theme rather than a hard-coded fill.
"""

from __future__ import annotations

import re
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ADMIN = (
    ROOT / "packages" / "llm-rosetta-gateway" / "src" / "llm_rosetta_gateway" / "admin"
)
DESIGN = ROOT / "design" / "logo" / "out"

ADMIN_HTML = (ADMIN / "admin.html").read_text(encoding="utf-8")


def _design_svg() -> str:
    return (DESIGN / "favicon.svg").read_text(encoding="utf-8")


def _points(svg: str) -> str:
    """The polygon's point list, whatever quoting or classes surround it."""
    match = re.search(r'<polygon[^>]*\bpoints="([^"]+)"', svg)
    assert match, "the svg has no polygon"
    return match.group(1)


def _translate(svg: str) -> str:
    """The ``translate(x,y)`` a mark places its outline with, or fail."""
    match = re.search(r'transform="translate\(([^)]+)\)', svg)
    assert match, "the mark lost its placement transform"
    return match.group(1)


def _outline_bottom() -> float:
    """The lowest y the design's polygon reaches, in its own coordinates."""
    return max(float(pair.split(",")[1]) for pair in _points(_design_svg()).split())


def test_the_cropping_window_is_gone_from_the_shell() -> None:
    assert "30 15 490 500" not in ADMIN_HTML, "a mark is still cut from the old window"
    assert "512.9,287.6" not in ADMIN_HTML, "a point only the old 48-point trace has"


def test_both_inline_marks_are_the_shipped_mark() -> None:
    design = _points(_design_svg())
    assert len(design.split()) == 128
    marks = {
        "header": r'<svg[^>]*class="logo-stone"[^>]*>.*?</svg>',
        "login": r'<div class="login-brand"[^>]*>\s*<svg[^>]*>.*?</svg>',
    }
    for where, pattern in marks.items():
        mark = re.search(pattern, ADMIN_HTML, re.S)
        assert mark, f"the {where} mark is gone"
        assert 'viewBox="0 0 634.78 634.78"' in mark.group(0), (
            f"the {where} mark is the old 600 box again"
        )
        assert 'fill="currentColor"' in mark.group(0), (
            "a mark no longer follows the theme"
        )
        assert _points(mark.group(0)) == design, (
            f"the {where} mark is not the shipped outline"
        )
        # The outline is in the design's own coordinates: the placement is
        # what puts it inside the window, and it is not visible to the point
        # comparison.
        assert _translate(mark.group(0)) == _translate(_design_svg()), (
            f"the {where} mark is placed somewhere else inside its window"
        )


def test_the_page_icon_is_the_served_favicon() -> None:
    match = re.search(r'href="data:image/svg\+xml,([^"]+)"', ADMIN_HTML)
    assert match, "the page lost its inline icon"
    icon = urllib.parse.unquote(match.group(1))
    assert 'viewBox="0 0 634.78 634.78"' in icon
    assert _points(icon) == _points(_design_svg())
    assert _translate(icon) == _translate(_design_svg()), (
        "the page icon is placed somewhere else inside its window"
    )
    assert "#666666" in icon and "#ededed" in icon, "the page icon lost its two inks"


def test_the_favicon_covers_the_outline() -> None:
    from llm_rosetta_gateway.admin.routes.auth import _FAVICON_SVG

    text = _FAVICON_SVG.decode("utf-8")
    assert 'viewBox="0 0 634.78 634.78"' in text
    assert _points(text) == _points(_design_svg()), "the favicon is not the draft"
    # The design's window is padded around the stone; the stone has to fit
    # inside it with room to spare, which is what the old crop violated.
    transform = re.search(r"translate\(([\d.]+),([\d.]+)\)", text)
    assert transform, "the favicon lost its placement transform"
    assert float(transform.group(2)) + _outline_bottom() < 634.78
    assert "#666666" in text and "#ededed" in text, "the favicon lost its two inks"
