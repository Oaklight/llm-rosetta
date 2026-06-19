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

from scripts import SCRIPTS

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

STONE_PATH = (HERE / "references" / "rosetta-stone-path.txt").read_text().strip()
VB = 595.279

# ---------------------------------------------------------------------------
# Inscription layout — real text, three script bands.
# ---------------------------------------------------------------------------
TEXT_TOP = 120
FONT = 11               # tiny carved characters
LINE_STEP = 15          # baseline-to-baseline within a band
BAND_GAP = 22           # extra space between the three script bands
DIVIDER_GAP = 11        # where the carved divider sits within BAND_GAP


def _row_left(y: float) -> float:
    """Left inset following the stone's taper (narrower top, wider base)."""
    t = max(0.0, min(1.0, (y - TEXT_TOP) / (470 - TEXT_TOP)))
    return 150 - 60 * t  # 150 -> 90


def layout_inscription():
    """Return (text_lines, divider_ys).

    text_lines: list of (x, y, string)
    divider_ys: y positions for the carved separators between bands.
    """
    lines = []
    dividers = []
    y = TEXT_TOP
    for bi, (_name, script) in enumerate(SCRIPTS):
        for ln in script:
            lines.append((_row_left(y), y, ln))
            y += LINE_STEP
        if bi < len(SCRIPTS) - 1:
            dividers.append(y - LINE_STEP + DIVIDER_GAP)
            y += BAND_GAP
    return lines, dividers


INSCRIPTION_LINES, DIVIDER_YS = layout_inscription()


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


def build(variant: str) -> dw.Drawing:
    p = PALETTES[variant]
    d = dw.Drawing(VB, VB, origin=(0, 0))

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

    # carved dividers between the three script bands
    for dy in DIVIDER_YS:
        x0 = _row_left(dy)
        inside.append(dw.Rectangle(x0, dy, 470 - x0, 2.5, rx=1.25,
                                   fill=p["ink"], fill_opacity=p["ink_op"] * 0.5))

    # the three scripts, as real tiny monospace text
    for (x, y, text) in INSCRIPTION_LINES:
        inside.append(dw.Text(
            text, FONT, x, y,
            font_family="ui-monospace, 'SF Mono', Menlo, Consolas, monospace",
            font_weight="500",
            fill=p["ink"], fill_opacity=p["ink_op"],
            letter_spacing="-0.3"))

    d.append(inside)
    return d


def main() -> None:
    variants = list(PALETTES.keys())
    for v in variants:
        build(v).save_svg(str(OUT / f"logo-{v}.svg"))

    cards = "".join(
        f'<section class="card"><div class="box"><img src="logo-{v}.svg" alt="{v}">'
        f'</div><h2>{v}</h2></section>' for v in variants)
    fav = "".join(
        f'<img src="logo-outline.svg" width="{s}" height="{s}">' for s in (64, 32, 16))
    zoom = ('<section class="card"><div class="box" style="min-height:420px">'
            '<img src="logo-granodiorite.svg" style="width:420px;height:420px"></div>'
            '<h2>granodiorite — zoom (read the three scripts)</h2></section>')
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM-Rosetta logo — Gen 6</title><style>
:root{{color-scheme:light dark;--bg:#f8fafc;--fg:#0f172a;--muted:#64748b;--card:#fff;--border:#e2e8f0}}
@media(prefers-color-scheme:dark){{:root{{--bg:#020617;--fg:#e2e8f0;--muted:#94a3b8;--card:#0f172a;--border:#1e293b}}}}
*{{box-sizing:border-box}}body{{margin:0;font-family:Inter,system-ui,sans-serif;background:var(--bg);color:var(--fg);min-height:100vh;padding:40px}}
h1{{font-size:clamp(30px,5vw,52px);letter-spacing:-.05em;margin:0 0 8px}}p{{color:var(--muted);max-width:820px;line-height:1.5;margin:0}}
.grid{{max-width:1040px;margin:28px auto;display:grid;grid-template-columns:repeat(3,1fr);gap:24px}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:24px;padding:20px}}
.box{{display:grid;place-items:center;min-height:300px;border-radius:16px;background:linear-gradient(45deg,#0001 25%,transparent 25%),linear-gradient(-45deg,#0001 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#0001 75%),linear-gradient(-45deg,transparent 75%,#0001 75%);background-size:22px 22px;background-position:0 0,0 11px,11px -11px,-11px 0}}
.box img{{width:240px;height:240px;object-fit:contain}}
.fav{{display:flex;gap:24px;align-items:center;justify-content:center;min-height:120px}}
h2{{font-size:17px;margin:14px 0 4px;letter-spacing:-.02em;text-transform:capitalize}}
header{{max-width:1040px;margin:0 auto}}
</style></head><body>
<header><h1>LLM-Rosetta logo — Gen 6</h1>
<p>The inscription is real: the same request body in three API dialects
(OpenAI Chat / Anthropic Messages / Google GenAI), carved as three scripts —
exactly what the real Rosetta Stone does with hieroglyphic / demotic / Greek.
From afar, texture; up close, an easter egg. Pure monochrome (no accent),
real silhouette, pink vein + grain for material depth.</p></header>
<div class="grid">{cards}</div>
<div class="grid">{zoom}<section class="card"><div class="box fav">{fav}</div>
<h2>favicon 64 / 32 / 16</h2></section></div>
</body></html>"""
    (OUT / "preview.html").write_text(html)
    print("wrote", len(variants), "SVGs;",
          len(INSCRIPTION_LINES), "inscription lines across",
          len(SCRIPTS), "scripts")


if __name__ == "__main__":
    main()
