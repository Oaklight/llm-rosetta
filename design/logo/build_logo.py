#!/usr/bin/env python3
"""LLM-Rosetta logo generator (drawsvg).

Gen 6 — the inscription is now REAL: three API payloads carved as three
scripts, mirroring the real Rosetta Stone (one decree, three scripts →
one request, three API dialects: OpenAI Chat / Anthropic / Google GenAI).

From afar: carved-line texture on dark granodiorite.
Up close: an easter egg — three real, equivalent request bodies.

Design decisions (from review vs Apple's Rosetta 2 icon):
  - Pure monochrome (dropped the blue active-token slab). The Rosetta Stone
    silhouette + real inscription carry the meaning unaided.
  - Real text micro-inscription instead of abstract pills — denser, reads
    unambiguously as writing, and rewards close inspection.
  - Keep the historically-accurate pink granodiorite vein + crystalline grain
    as material depth (they read as texture, not chrome).

Run:
    python design/logo/build_logo.py
"""

from __future__ import annotations

from pathlib import Path

import drawsvg as dw

from scripts_data import SCRIPTS

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent / "fonts"))
from fonts_embed import FONTS  # base64 woff2 subsets  # noqa: E402

# Font THEMES — each maps the three scripts (OpenAI / Anthropic / Google) to
# three embedded typefaces. We render one logo per theme to compare which
# trio reads best as "three distinct scripts".
FONT_THEMES = {
    # current: three geometric/humanist monos
    "monos": {"openai_chat": "jetbrains", "anthropic": "plex", "google": "space"},
    # higher-contrast monos: ligature mono / classic / script-italic mono
    "monos-contrast": {"openai_chat": "fira", "anthropic": "inconsolata",
                       "google": "victor"},
    # mixed families: mono / serif / sans — maximum script differentiation
    "mixed": {"openai_chat": "jetbrains", "anthropic": "spectral",
              "google": "inter"},
    # warm + quirky: clean mono / humanist mono / retro mono
    "warm": {"openai_chat": "redhat", "anthropic": "plex", "google": "space"},
}


def _font_css(theme: dict) -> str:
    """@font-face rules for the three fonts used by *theme*."""
    rules = []
    for key in dict.fromkeys(theme.values()):  # unique, ordered
        rules.append(
            f"@font-face{{font-family:'F-{key}';font-style:normal;"
            f"font-weight:500;src:url(data:font/woff2;base64,{FONTS[key]}) "
            f"format('woff2');}}")
    return "".join(rules)

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

STONE_PATH = (HERE / "references" / "rosetta-stone-path.txt").read_text().strip()
VB = 595.279

# ---------------------------------------------------------------------------
# Inscription layout — an UPRIGHT rectangular text block, deliberately larger
# than the stone, then clipped to the silhouette. Rows are horizontal and
# left-aligned to a single x; the broken-stone outline crops the overhang.
# A real shattered stele truncates lines at its fractured edges — so partial
# rows at the margins are correct and intentional.
# ---------------------------------------------------------------------------
# Overscan the text grid past every edge of the stone (bbox ~x50..520,
# y60..510); the silhouette clip crops the overhang into broken-edge rows.
GRID_LEFT = 48           # left of the stone's leftmost point
GRID_TOP = 60            # above the stone's top
GRID_BOTTOM = 524        # below the stone's base
BAND_GAP_FACTOR = 2.6    # band gap = this * line step (clearer band separation)
DIVIDER_FRAC = 0.5       # divider sits this fraction into the band gap


def layout_inscription(theme: dict):
    """Upright grid: constant left x, uniform horizontal rows top→bottom.

    Each script's lines are tagged with the theme's font family. Step is
    derived from line count so the block always spans the stone; the clip
    mask (silhouette) crops whatever overruns the broken edges.
    """
    n = sum(len(s) for _, s in SCRIPTS)
    n_gaps = len(SCRIPTS) - 1
    span = GRID_BOTTOM - GRID_TOP
    step = span / ((n - 1) + n_gaps * (BAND_GAP_FACTOR - 1))
    band_gap = step * BAND_GAP_FACTOR
    font = round(step * 0.95, 2)

    lines = []  # (x, y, text, font_family)
    dividers = []
    y = GRID_TOP
    for bi, (name, script) in enumerate(SCRIPTS):
        key = theme.get(name)
        fam = f"'F-{key}', monospace" if key else "ui-monospace, monospace"
        for ln in script:
            lines.append((GRID_LEFT, y, ln, fam))  # constant x → upright rows
            y += step
        if bi < len(SCRIPTS) - 1:
            y -= step
            dividers.append(y + band_gap * DIVIDER_FRAC)
            y += band_gap
    return lines, dividers, font


# ---------------------------------------------------------------------------
# Palettes (pure monochrome; no accent token)
# ---------------------------------------------------------------------------
PALETTES = {
    "granodiorite": {
        "stops": [(0, "#3d4757"), (0.5, "#232c3a"), (1, "#10141d")],
        "ink": "#cdd8e6", "ink_op": 0.88,
        "vein": True, "grain": 0.07, "stroke": None,
    },
    "outline": {
        "fill": "#ffffff",
        "ink": "#1e293b", "ink_op": 0.92,
        "vein": False, "grain": 0.0,
        "stroke": "#0f172a", "stroke_w": 16,
    },
    "light": {  # light artifact / museum stone
        "stops": [(0, "#e7e2d8"), (0.5, "#cabfa8"), (1, "#9a8f78")],
        "ink": "#3a352b", "ink_op": 0.72,
        "vein": True, "vein_color": "#b5708a", "grain": 0.06, "stroke": None,
    },
}


def _grain(d: dw.Drawing, fid: str, amount: float) -> None:
    f = dw.Filter(id=fid)
    f.append(dw.FilterItem("feTurbulence", type="fractalNoise",
                           baseFrequency="0.9", numOctaves=2, seed=7, result="n"))
    f.append(dw.FilterItem("feColorMatrix", in_="n", type="matrix",
                           values=f"0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 {amount} 0"))
    d.append(f)


def build(variant: str, theme_name: str = "monos") -> dw.Drawing:
    p = PALETTES[variant]
    theme = FONT_THEMES[theme_name]
    inscription_lines, divider_ys, font_size = layout_inscription(theme)
    d = dw.Drawing(VB, VB, origin=(0, 0))

    # embed only this theme's fonts so the SVG stays self-contained + small
    d.append_css(_font_css(theme))

    clip = dw.ClipPath()
    clip.append(dw.Path(STONE_PATH))
    d.append(clip)

    # stone body
    if "stops" in p:
        grad = dw.LinearGradient(120, 20, 500, 560, gradientUnits="userSpaceOnUse")
        for off, col in p["stops"]:
            grad.add_stop(off, col)
        d.append(grad)
        fill = grad
    else:
        fill = p["fill"]
    stroke_kw = {}
    if p.get("stroke"):
        stroke_kw = dict(stroke=p["stroke"], stroke_width=p["stroke_w"],
                         stroke_linejoin="round")
    d.append(dw.Path(STONE_PATH, fill=fill, **stroke_kw))

    inside = dw.Group(clip_path=clip)

    if p.get("grain"):
        fid = f"grain-{variant}"
        _grain(d, fid, p["grain"])
        inside.append(dw.Rectangle(0, 0, VB, VB, filter=f"url(#{fid})"))

    # pink granodiorite vein, top-left (historically accurate diagonal)
    if p.get("vein"):
        vc = p.get("vein_color", "#d98fb0")
        vfid = f"vein-{variant}"
        vf = dw.Filter(id=vfid, x="-40%", y="-40%", width="180%", height="180%")
        vf.append(dw.FilterItem("feGaussianBlur", stdDeviation=9))
        d.append(vf)
        vg = dw.LinearGradient(70, 30, 230, 240, gradientUnits="userSpaceOnUse")
        vg.add_stop(0, vc, opacity=0.0)
        vg.add_stop(0.45, vc, opacity=0.42)
        vg.add_stop(0.75, vc, opacity=0.16)
        vg.add_stop(1, vc, opacity=0.0)
        d.append(vg)
        vein = dw.Group(filter=f"url(#{vfid})")
        vein.append(dw.Path(
            "M 150 36 C 120 90, 96 130, 84 180 C 78 205, 72 230, 64 250",
            stroke=vg, stroke_width=22, fill="none", stroke_linecap="round"))
        vein.append(dw.Path(
            "M 170 44 C 142 96, 118 134, 108 178",
            stroke=vc, stroke_width=5, stroke_opacity=0.22, fill="none",
            stroke_linecap="round"))
        inside.append(vein)

    # carved dividers between the three script bands — full-width upright rules,
    # clipped to the stone like the text
    for dy in divider_ys:
        inside.append(dw.Rectangle(GRID_LEFT, dy, 470, 2.5, rx=1.25,
                                   fill=p["ink"], fill_opacity=p["ink_op"] * 0.5))

    # the three scripts, each in its own typeface
    for (x, y, text, fam) in inscription_lines:
        inside.append(dw.Text(
            text, font_size, x, y,
            font_family=fam,
            font_weight="500",
            fill=p["ink"], fill_opacity=p["ink_op"],
            letter_spacing="-0.3"))

    d.append(inside)
    return d


THEME_DESC = {
    "monos": "JetBrains · IBM Plex · Space Mono (geometric / humanist / retro)",
    "monos-contrast": "Fira Code · Inconsolata · Victor Mono (ligature / classic / script-italic)",
    "mixed": "JetBrains Mono · Spectral · Inter (mono / serif / sans)",
    "warm": "Red Hat Mono · IBM Plex · Space Mono",
}


def main() -> None:
    # One granodiorite render per font theme, for comparison.
    for tn in FONT_THEMES:
        build("granodiorite", tn).save_svg(str(OUT / f"theme-{tn}.svg"))
    # Standard 3-material set using the default theme.
    for v in PALETTES:
        build(v, "monos").save_svg(str(OUT / f"logo-{v}.svg"))

    theme_cards = "".join(
        f'<section class="card"><div class="box">'
        f'<img src="theme-{tn}.svg" alt="{tn}"></div>'
        f'<h2>{tn}</h2><p class="note">{THEME_DESC.get(tn, "")}</p></section>'
        for tn in FONT_THEMES)
    zoom_cards = "".join(
        f'<section class="card"><div class="box" style="min-height:440px">'
        f'<img src="theme-{tn}.svg" style="width:440px;height:440px"></div>'
        f'<h2>{tn} — zoom</h2></section>'
        for tn in FONT_THEMES)
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM-Rosetta logo — font themes</title><style>
:root{{color-scheme:light dark;--bg:#f8fafc;--fg:#0f172a;--muted:#64748b;--card:#fff;--border:#e2e8f0}}
@media(prefers-color-scheme:dark){{:root{{--bg:#020617;--fg:#e2e8f0;--muted:#94a3b8;--card:#0f172a;--border:#1e293b}}}}
*{{box-sizing:border-box}}body{{margin:0;font-family:Inter,system-ui,sans-serif;background:var(--bg);color:var(--fg);min-height:100vh;padding:40px}}
h1{{font-size:clamp(28px,5vw,48px);letter-spacing:-.05em;margin:0 0 8px}}p{{color:var(--muted);max-width:860px;line-height:1.5;margin:0}}
.grid{{max-width:1180px;margin:28px auto;display:grid;grid-template-columns:repeat(4,1fr);gap:20px}}
.grid2{{max-width:1180px;margin:28px auto;display:grid;grid-template-columns:repeat(2,1fr);gap:24px}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:24px;padding:18px}}
.box{{display:grid;place-items:center;min-height:240px;border-radius:16px;background:linear-gradient(45deg,#0001 25%,transparent 25%),linear-gradient(-45deg,#0001 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#0001 75%),linear-gradient(-45deg,transparent 75%,#0001 75%);background-size:20px 20px;background-position:0 0,0 10px,10px -10px,-10px 0}}
.box img{{width:200px;height:200px;object-fit:contain}}
.note{{font-size:12px;color:var(--muted);margin-top:4px}}
h2{{font-size:16px;margin:12px 0 2px;letter-spacing:-.02em}}
header{{max-width:1180px;margin:0 auto}}
</style></head><body>
<header><h1>LLM-Rosetta logo — font themes</h1>
<p>Same stone, same real convert() inscription — four typeface trios for the
three scripts (OpenAI / Anthropic / Google). Top row = at-size; bottom =
zoomed to read the three distinct "scripts". Which trio reads best?</p></header>
<div class="grid">{theme_cards}</div>
<div class="grid2">{zoom_cards}</div>
</body></html>"""
    (OUT / "preview.html").write_text(html)
    n = sum(len(s) for _, s in SCRIPTS)
    print("wrote", len(FONT_THEMES), "theme SVGs +", len(PALETTES),
          "material SVGs;", n, "inscription lines")


if __name__ == "__main__":
    main()
