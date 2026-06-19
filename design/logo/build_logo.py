#!/usr/bin/env python3
"""LLM-Rosetta logo generator (drawsvg).

Iterates from the Gen 4 "exact-shape pass": the real downloaded Rosetta Stone
SVG silhouette as the base, with a token-block inscription clipped inside.

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

# The real Rosetta Stone silhouette path (viewBox 0 0 595.279 595.279).
# This is the irreplaceable identity anchor — never hand-redraw it.
STONE_PATH = (HERE / "references" / "rosetta-stone-path.txt").read_text().strip()

VB = 595.279  # square viewBox side

# Gen 4 inscription layout: (x, y, width, height) of each carved line.
# Three visual blocks mimicking the stone's three scripts.
INSCRIPTION = [
    # top block (hieroglyphic feel) — sparse
    (178, 112, 70, 10), (262, 112, 92, 10), (368, 112, 44, 10),
    (145, 138, 114, 8), (274, 138, 122, 8),
    (125, 164, 80, 8), (219, 164, 148, 8), (382, 164, 52, 8),
    # middle block (demotic) — dense
    (102, 254, 188, 9), (306, 254, 112, 9),
    (96, 280, 92, 8), (204, 280, 174, 8),
    (96, 306, 152, 8), (264, 306, 116, 8),
    # bottom block (greek) — justified
    (92, 405, 260, 8), (92, 430, 214, 8), (92, 455, 164, 8),
]
# The "active token" — one highlighted line (the decode-in-progress glint).
ACTIVE_TOKEN = (382, 138, 42, 8)


# --- material palettes -----------------------------------------------------
PALETTES = {
    "black": {  # A4 — dark granodiorite
        "stops": [(0, "#475569"), (0.48, "#111827"), (1, "#030712")],
        "ink": "#dbeafe",
        "ink_op": 0.78,
        "token": "#60a5fa",
        "stroke": None,
    },
    "outline": {  # B4 — clean outline mark
        "fill": "#eff6ff",
        "ink": "#0f172a",
        "ink_op": 0.82,
        "token": "#2563eb",
        "stroke": "#0f172a",
        "stroke_w": 16,
    },
    "artifact": {  # C4 — light artifact material
        "stops": [(0, "#f5f5f4"), (0.48, "#a8a29e"), (1, "#57534e")],
        "ink": "#44403c",
        "ink_op": 0.55,
        "token": "#2563eb",
        "token_op": 0.72,
        "stroke": None,
    },
}


def build(variant: str) -> dw.Drawing:
    p = PALETTES[variant]
    d = dw.Drawing(VB, VB, origin=(0, 0))

    # clip path = stone silhouette
    clip = dw.ClipPath()
    clip.append(dw.Path(STONE_PATH))
    d.append(clip)

    # stone body fill (gradient or flat)
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

    # inscription, clipped to the stone
    g = dw.Group(clip_path=clip)
    for (x, y, w, h) in INSCRIPTION:
        g.append(dw.Rectangle(x, y, w, h, rx=h / 2, fill=p["ink"],
                              fill_opacity=p["ink_op"]))
    # active token
    tx, ty, tw, th = ACTIVE_TOKEN
    g.append(dw.Rectangle(tx, ty, tw, th, rx=th / 2, fill=p["token"],
                          fill_opacity=p.get("token_op", 1.0)))
    d.append(g)
    return d


def main() -> None:
    variants = ["black", "outline", "artifact"]
    for v in variants:
        d = build(v)
        d.save_svg(str(OUT / f"logo-{v}.svg"))

    # preview page
    cards = "".join(
        f'<section class="card"><div class="box">'
        f'<img src="logo-{v}.svg" alt="{v}"></div>'
        f'<h2>{v}</h2></section>'
        for v in variants
    )
    fav = "".join(
        f'<img src="logo-outline.svg" width="{s}" height="{s}">' for s in (64, 32, 16)
    )
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM-Rosetta logo (drawsvg)</title><style>
:root{{color-scheme:light dark;--bg:#f8fafc;--fg:#0f172a;--muted:#64748b;--card:#fff;--border:#e2e8f0}}
@media(prefers-color-scheme:dark){{:root{{--bg:#020617;--fg:#e2e8f0;--muted:#94a3b8;--card:#0f172a;--border:#1e293b}}}}
*{{box-sizing:border-box}}body{{margin:0;font-family:Inter,system-ui,sans-serif;background:var(--bg);color:var(--fg);min-height:100vh;padding:40px}}
h1{{font-size:clamp(30px,5vw,52px);letter-spacing:-.05em;margin:0 0 8px}}p{{color:var(--muted);max-width:760px;line-height:1.5;margin:0}}
.grid{{max-width:1000px;margin:28px auto;display:grid;grid-template-columns:repeat(3,1fr);gap:24px}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:24px;padding:20px}}
.box{{display:grid;place-items:center;min-height:300px;border-radius:16px;background:linear-gradient(45deg,#0001 25%,transparent 25%),linear-gradient(-45deg,#0001 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#0001 75%),linear-gradient(-45deg,transparent 75%,#0001 75%);background-size:22px 22px;background-position:0 0,0 11px,11px -11px,-11px 0}}
.box img{{width:240px;height:240px;object-fit:contain}}
.fav{{display:flex;gap:24px;align-items:center;justify-content:center;min-height:120px}}
h2{{font-size:18px;margin:14px 0 4px;letter-spacing:-.02em;text-transform:capitalize}}
header{{max-width:1000px;margin:0 auto}}
</style></head><body>
<header><h1>LLM-Rosetta logo — drawsvg base (from Gen 4)</h1>
<p>Real Rosetta silhouette + token-block inscription, generated with drawsvg.
Three materials: black granodiorite, clean outline, light artifact.</p></header>
<div class="grid">{cards}</div>
<div class="grid"><section class="card"><div class="box fav">{fav}</div>
<h2>favicon 64 / 32 / 16</h2></section></div>
</body></html>"""
    (OUT / "preview.html").write_text(html)
    print("wrote", len(variants), "SVGs + preview.html to", OUT)


if __name__ == "__main__":
    main()
