"""The gateway draws the whole stone, not 87% of it (issue #1548).

The favicon and the two inline marks were cut from a ``30 15 490 500`` window
while the outline runs to y≈593, so the mark ended on a flat edge where the
stone's base should be.  The design files carry the corrected geometry:
``design/logo/out/favicon.svg`` for the tab icon and the 600-box icon for the
header and login slots.

These checks pin what the fix is about — the window, and that the inline marks
still take their colour from the theme rather than a hard-coded fill.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ADMIN = (
    ROOT / "packages" / "llm-rosetta-gateway" / "src" / "llm_rosetta_gateway" / "admin"
)
DESIGN = ROOT / "design" / "logo" / "out"

ADMIN_HTML = (ADMIN / "admin.html").read_text(encoding="utf-8")


def _outline_bottom() -> float:
    """The lowest y the design's polygon reaches, in its own coordinates."""
    text = (DESIGN / "favicon.svg").read_text(encoding="utf-8")
    match = re.search(r'<polygon class="mark" points="([^"]+)"', text)
    assert match, "the design favicon has no mark polygon"
    points = match.group(1)
    return max(float(pair.split(",")[1]) for pair in points.split())


def test_the_cropping_window_is_gone_from_the_shell() -> None:
    assert "30 15 490 500" not in ADMIN_HTML, "a mark is still cut from the old window"


def test_both_inline_marks_are_the_uncropped_icon() -> None:
    marks = {
        "header": r'<svg[^>]*class="logo-stone"[^>]*>.*?</svg>',
        "login": r'<div class="login-brand"[^>]*>\s*<svg[^>]*>.*?</svg>',
    }
    for where, pattern in marks.items():
        mark = re.search(pattern, ADMIN_HTML)
        assert mark, f"the {where} mark is gone"
        assert 'viewBox="0 0 600 600"' in mark.group(0), (
            f"the {where} mark is cropped again"
        )
        assert 'fill="currentColor"' in mark.group(0), (
            "a mark no longer follows the theme"
        )
        assert "<polygon" in mark.group(0)


def test_the_favicon_covers_the_outline() -> None:
    from llm_rosetta_gateway.admin.routes.auth import _FAVICON_SVG

    text = _FAVICON_SVG.decode("utf-8")
    assert "viewBox='0 0 635.22 635.22'" in text
    # The design's window is padded around the stone; the stone has to fit
    # inside it with room to spare, which is what the crop violated.
    transform = re.search(r"translate\(([\d.]+),([\d.]+)\)", text)
    assert transform, "the favicon lost its placement transform"
    assert float(transform.group(2)) + _outline_bottom() < 635.22
    assert "#666666" in text and "#ededed" in text, "the favicon lost its two inks"
