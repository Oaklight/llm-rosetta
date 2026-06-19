#!/usr/bin/env python3
"""LLM-Rosetta logo generator (drawsvg).

Gen 5 — research-driven iteration on the Gen 4 exact-shape base.

Borrowed details (from researching real artifact + Apple Rosetta 2 icon):
  1. Dense, full-width inscription of short broken segments (Apple Rosetta 2
     reads as real carved text, not sparse lines).
  2. The real stone's signature pink granodiorite vein across the top-left.
  3. Crystalline grain (feTurbulence) for stone material, not flat fill.
  4. One blue "active token" slab = the decode-in-progress glint (our
     differentiator vs Apple's pure-monochrome treatment).

Run:
    python design/logo/build_logo.py
Outputs SVG + a preview HTML into design/logo/out/.
"""

from __future__ import annotations

from pathlib import Path

import drawsvg as dw

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

# Real Rosetta Stone silhouette path (viewBox 0 0 595.279 595.279).
# Irreplaceable identity anchor — never hand-redraw it.
STONE_PATH = (HERE / "references" / "rosetta-stone-path.txt").read_text().strip()

VB = 595.279

# ---------------------------------------------------------------------------
# Deterministic dense inscription (Apple-Rosetta-2 style: rows of short,
# uneven segments filling the stone width). No RNG so output is reproducible.
# ---------------------------------------------------------------------------

# Interior text box, tracking the stone's slightly-narrower top.
TEXT_TOP = 108
TEXT_BOTTOM = 470
ROW_STEP = 21          # vertical rhythm
SEG_H = 8              # carved-line thickness
GAP = 11               # gap between segments on a row

# A repeating but irregular cycle of segment widths (px). Cycling through this
# with a per-row offset yields organic, non-repeating-looking rows.
SEG_CYCLE = [34, 52, 20, 44, 28, 60, 24, 40, 30, 48, 22, 56, 36, 26]


def _row_left(y: float) -> float:
    """Left inset of the text box at height y — follows the stone's taper.

    The real stone is a touch narrower near the top, wider at the base.
    """
    t = (y - TEXT_TOP) / (TEXT_BOTTOM - TEXT_TOP)  # 0 at top, 1 at bottom
    return 150 - 64 * t  # 150 -> 86


def _row_right(y: float) -> float:
    t = (y - TEXT_TOP) / (TEXT_BOTTOM - TEXT_TOP)
    return 440 + 70 * t  # 440 -> 510


def dense_inscription():
    """Yield (x, y, w, h) carved segments filling the stone, plus return the
    chosen active-token rect.
    """
    segs = []
    rows = []
    y = TEXT_TOP
    ci = 0
    ri = 0
    while y <= TEXT_BOTTOM:
        left = _row_left(y)
        right = _row_right(y)
        x = left + (ri % 3) * 6  # tiny per-row left jitter
        row_segs = []
        while x < right:
            w = SEG_CYCLE[ci % len(SEG_CYCLE)]
            ci += 1
            if x + w > right:
                w = right - x
                if w < 12:
                    break
            row_segs.append((x, y, w, SEG_H))
            x += w + GAP
        segs.extend(row_segs)
        rows.append(row_segs)
        y += ROW_STEP
        ri += 1
    # active token: pick a mid-stone row, a middle segment in it
    mid_row = rows[len(rows) // 2 - 1]
    token = mid_row[len(mid_row) // 2] if mid_row else None
    return segs, token


INSCRIPTION, ACTIVE_TOKEN = dense_inscription()


# ---------------------------------------------------------------------------
# Material palettes
# ---------------------------------------------------------------------------
PALETTES = {
    "granodiorite": {  # true-to-artifact dark stone + pink vein
        "stops": [(0, "#3d4757"), (0.5, "#232c3a"), (1, "#10141d")],
        "ink": "#cdd8e6",
        "ink_op": 0.85,
        "token": "#38bdf8",
        "vein": True,
        "grain": 0.05,
        "stroke": None,
    },
    "outline": {  # clean favicon-friendly mark
        "fill": "#ffffff",
        "ink": "#0f172a",
        "ink_op": 0.85,
        "token": "#2563eb",
        "vein": False,
        "grain": 0.0,
        "stroke": "#0f172a",
        "stroke_w": 16,
    },
    "ink-amber": {  # near-black slate, amber token, warm vein
        "stops": [(0, "#2c2f37"), (0.5, "#181a20"), (1, "#0a0b0e")],
        "ink": "#ecdcb6",
        "ink_op": 0.82,
        "token": "#f59e0b",
        "vein": True,
        "vein_color": "#c08462",  # warmer copper vein
        "grain": 0.05,
        "stroke": None,
    },
}


def _add_grain_filter(d: dw.Drawing, fid: str, amount: float) -> None:
    """Crystalline speckle via fractal noise, low-alpha overlay."""
    f = dw.Filter(id=fid)
    f.append(dw.FilterItem(
        "feTurbulence", type="fractalNoise", baseFrequency="0.9",
        numOctaves=2, seed=7, result="n"))
    f.append(dw.FilterItem(
        "feColorMatrix", in_="n", type="matrix",
        values=f"0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 {amount} 0"))
    d.append(f)


def build(variant: str) -> dw.Drawing:
    p = PALETTES[variant]
    d = dw.Drawing(VB, VB, origin=(0, 0))

    clip = dw.ClipPath()
    clip.append(dw.Path(STONE_PATH))
    d.append(clip)

    # stone fill
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

    # crystalline grain overlay
    if p.get("grain"):
        fid = f"grain-{variant}"
        _add_grain_filter(d, fid, p["grain"])
        inside.append(dw.Rectangle(0, 0, VB, VB, filter=f"url(#{fid})"))

    # signature pink granodiorite vein across the top-left corner.
    # A thin, soft, slightly-wavy diagonal streak (blurred) — reads as a
    # mineral vein, not a painted band.
    if p.get("vein"):
        vc = p.get("vein_color", "#d98fb0")  # soft rose-pink
        # heavier blur so the streak diffuses into the stone like a real vein
        vfid = f"vein-blur-{variant}"
        vf = dw.Filter(id=vfid, x="-40%", y="-40%", width="180%", height="180%")
        vf.append(dw.FilterItem("feGaussianBlur", stdDeviation=9))
        d.append(vf)
        vgrad = dw.LinearGradient(70, 30, 230, 240, gradientUnits="userSpaceOnUse")
        vgrad.add_stop(0, vc, opacity=0.0)
        vgrad.add_stop(0.45, vc, opacity=0.42)
        vgrad.add_stop(0.75, vc, opacity=0.16)
        vgrad.add_stop(1, vc, opacity=0.0)
        d.append(vgrad)
        vein = dw.Group(filter=f"url(#{vfid})")
        # long diagonal grazing the top-left corner, entering from the top edge
        # and running down-left toward the side — matches the real artifact
        vein.append(dw.Path(
            "M 150 36 C 120 90, 96 130, 84 180 C 78 205, 72 230, 64 250",
            stroke=vgrad, stroke_width=22, fill="none", stroke_linecap="round"))
        # faint parallel hairline for crystalline variation
        vein.append(dw.Path(
            "M 170 44 C 142 96, 118 134, 108 178",
            stroke=vc, stroke_width=5, stroke_opacity=0.22, fill="none",
            stroke_linecap="round"))
        inside.append(vein)

    # dense inscription
    for (x, y, w, h) in INSCRIPTION:
        if ACTIVE_TOKEN and (x, y, w, h) == ACTIVE_TOKEN:
            continue
        inside.append(dw.Rectangle(x, y, w, h, rx=h / 2, fill=p["ink"],
                                   fill_opacity=p["ink_op"]))
    # active token slab (slightly taller, solid)
    if ACTIVE_TOKEN:
        tx, ty, tw, th = ACTIVE_TOKEN
        inside.append(dw.Rectangle(tx, ty - 2, max(tw, 40), th + 4, rx=4,
                                   fill=p["token"], fill_opacity=p.get("token_op", 1.0)))

    d.append(inside)
    return d


def main() -> None:
    variants = list(PALETTES.keys())
    for v in variants:
        build(v).save_svg(str(OUT / f"logo-{v}.svg"))

    cards = "".join(
        f'<section class="card"><div class="box"><img src="logo-{v}.svg" alt="{v}">'
        f'</div><h2>{v}</h2></section>'
        for v in variants
    )
    fav = "".join(
        f'<img src="logo-outline.svg" width="{s}" height="{s}">' for s in (64, 32, 16)
    )
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM-Rosetta logo — Gen 5 (drawsvg)</title><style>
:root{{color-scheme:light dark;--bg:#f8fafc;--fg:#0f172a;--muted:#64748b;--card:#fff;--border:#e2e8f0}}
@media(prefers-color-scheme:dark){{:root{{--bg:#020617;--fg:#e2e8f0;--muted:#94a3b8;--card:#0f172a;--border:#1e293b}}}}
*{{box-sizing:border-box}}body{{margin:0;font-family:Inter,system-ui,sans-serif;background:var(--bg);color:var(--fg);min-height:100vh;padding:40px}}
h1{{font-size:clamp(30px,5vw,52px);letter-spacing:-.05em;margin:0 0 8px}}p{{color:var(--muted);max-width:780px;line-height:1.5;margin:0}}
.grid{{max-width:1040px;margin:28px auto;display:grid;grid-template-columns:repeat(3,1fr);gap:24px}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:24px;padding:20px}}
.box{{display:grid;place-items:center;min-height:300px;border-radius:16px;background:linear-gradient(45deg,#0001 25%,transparent 25%),linear-gradient(-45deg,#0001 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#0001 75%),linear-gradient(-45deg,transparent 75%,#0001 75%);background-size:22px 22px;background-position:0 0,0 11px,11px -11px,-11px 0}}
.box img{{width:240px;height:240px;object-fit:contain}}
.fav{{display:flex;gap:24px;align-items:center;justify-content:center;min-height:120px}}
h2{{font-size:18px;margin:14px 0 4px;letter-spacing:-.02em;text-transform:capitalize}}
header{{max-width:1040px;margin:0 auto}}
</style></head><body>
<header><h1>LLM-Rosetta logo — Gen 5</h1>
<p>Dense full-width inscription (Apple Rosetta 2 style), the real stone's
pink granodiorite vein across the top-left, crystalline grain, and the blue
active-token glint. Real silhouette throughout.</p></header>
<div class="grid">{cards}</div>
<div class="grid"><section class="card"><div class="box fav">{fav}</div>
<h2>favicon 64 / 32 / 16</h2></section></div>
</body></html>"""
    (OUT / "preview.html").write_text(html)
    print("wrote", len(variants), "SVGs + preview.html;",
          len(INSCRIPTION), "inscription segments")


if __name__ == "__main__":
    main()
