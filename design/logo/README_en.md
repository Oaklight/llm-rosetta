# The stone mark

Every published copy of the logo in this repository comes from one traced
outline of the Rosetta Stone. The copies differ only in crop, scale and fill.
This folder holds the tooling that regenerates each of them from the source
path.

## Source

`references/rosetta-stone-path.txt` is the trace, and the input for every file
in `out/`: one closed path, 126 cubic segments, 4,726 bytes of `d` data,
bounding box 451.5 x 589.6 user units.

The outline was produced by algorithmic tracing of a Wikimedia Commons file,
[File:Rosetta Stone.svg](https://commons.wikimedia.org/wiki/File:Rosetta_Stone.svg),
authored by kinetoons (2014, from open clip art) and released under CC0 1.0.
`references/SOURCE.md` records the file, the author, the licence and the
caveat.

CC0 requires no attribution, so the mark adds no licence obligation of its
own. The shape is a derived trace, and the chain above records where it came
from. If the mark is ever registered or the project is sold, re-derive it from
a documented source, because a CC0 dedication at the top of a chain does not
clear rights further down that the uploader never held.

## Canonical parameters

| Use | File | points | anchors | framing | fill |
| --- | --- | --- | --- | --- | --- |
| App icon | `out/rosetta-icon.svg` | 48 | 12 | 600 canvas, scale 0.88 | `#1e293b` / `#ededed` |
| README, dark half | `out/rosetta-icon-github.svg` | 48 | 12 | 600 canvas, scale 0.88 | `#1e293b` / `#ededed` |
| README, light half | `out/rosetta-icon-docs.svg` | 48 | 12 | 600 canvas, scale 0.88 | `#1e293b` only |
| Favicon | `out/favicon.svg` | 28 | 10 | tight square, pad 0.04 | `#666666` / `#ededed` |

`points` is the vertex budget. `anchors` is the number of structural vertices
that may never be simplified away, ranked by the triangle area a removal would
cost. Above about 64 vertices the mark stops changing visibly, so what separates
a readable small mark from a poor one is which vertices survive.

## Shipped copies, measured

| Location | File | Framing | Fill |
| --- | --- | --- | --- |
| README (en/zh) | `design/logo/out/rosetta-icon-{github,docs}.svg` | 600 box, scale 0.88 | `#1e293b` / `#ededed`; docs light only |
| Package READMEs | `out/rosetta-icon-github.svg` | 600 box, scale 0.88 | `#1e293b` / `#ededed` |
| Docs header | `docs_{en,zh}/docs/images/rosetta_stone.svg` | 600 box, scale 0.934 | `currentColor` |
| Docs favicon | `docs_{en,zh}/docs/images/favicon.svg` | 600 box, scale 0.934 | `#1a1a1a` / `#ededed` |
| Gateway favicon | `admin/routes/auth.py` `_FAVICON_SVG` | viewBox `30 15 490 500` | `#666` |
| Gateway sidebar/header | `admin/admin.html` inline `<svg>` | viewBox `30 15 490 500` | `currentColor` |

The README pair used to be byte-identical, so the `prefers-color-scheme` switch
rendered the same file twice. The light half is now a light-only file and the
dark half keeps its own media query. The package READMEs reference the dark
file directly, and they still adapt on their own.

The gateway favicon crops to viewBox `30 15 490 500` while the outline spans
y 3.3..593.0, so the bottom ~78 units (13%) of the stone are cut off. That is
why the gateway favicon does not match the README mark. Replacing it with
`out/favicon.svg` (tight square, 4% padding) changes gateway code and is left
for its own PR.

`out/favicon-gateway.svg` is a tracked copy of exactly what the gateway serves
today, so the crop stays visible in the repository instead of living only in a
byte string inside `auth.py`.

## Tooling

`simplify_path.py` (CLI, stdlib only) and `simplify.html` (interactive, no
network) run the same pipeline and produce byte-identical geometry for the same
parameters:

1. parse the path; flatten cubics adaptively (`--tolerance`);
2. pick structural anchors (`--anchor`, `--anchor-mode area|corner|depth|hybrid`);
3. reduce to the budget (`--points`, `--method visvalingam|douglas`) without
   dropping an anchor, trim to a bounded error (`--method fidelity`,
   `--fidelity`), or collapse edges under the same bound (`--method
   decimate`);
4. optional Chaikin rounding (`--smooth`);
5. deform: `--rotate`, `--aspect`, `--inflate`;
6. frame: square canvas (`--canvas`, `--scale`) or tight square (`--tight`,
   `--pad`).

`--method fidelity` ignores the vertex budget. It removes every vertex that can
go while the outline stays within `--fidelity` times the longest side of the
original, measured vertex by vertex against the unsimplified path. Pinned
vertices are kept exactly and split the ring into independent arcs, so a mark
keeps its chosen corners and the rest of the outline follows the fidelity
setting. Use it when the count should follow the shape rather than the other
way round.

`--method decimate` is the mesh-decimation analogue: it collapses the cheapest
edge, re-costs the two neighbours, and repeats, refusing any collapse that
would move the outline beyond `--fidelity`. The vertex budget still applies, so
whichever limit binds first wins. A filled logo has no interior to triangulate,
so the collapse runs on the outline; on the stone it lands within a vertex or
two of `--method fidelity` at the same error, with a slightly more even
distribution because the cost is re-evaluated globally after every collapse.

The anchor modes rank a vertex differently. `area` uses the triangle a removal
would lose (favours the big convex sweeps), `corner` uses the turning angle
(favours sharp jags), `depth` uses the distance inside the convex hull (favours
dents, since a hollow is where the outline leaves the hull), and `hybrid` takes
half from area and fills the rest by depth. On this outline 46% of the
flattened vertices are locally reflex, yet `area` selects 1 of 12, `corner`
7 of 12, and `depth` 12 of 12, all non-hull.

Red anchors reach the CLI as `--lock "u0.3738@214.01,85.47"`. The reference is a
fraction of the outline length, not a vertex index: an index names a different
place as soon as `--tolerance` changes the flattening, which is what used to
move an anchor around the shape. `--pin` takes the same reference, and a bare
integer is still read as the old index form. A locked vertex is forced into the
result even when the budget would drop it; `214.01,85.47` is the canvas point it
is held at. The deformation knobs turn and grow around the centroid of the red
anchors, so dragging one moves the rest of the outline and leaves the anchors
where they are, and the framing maps that centroid back onto the recorded canvas
point; several locks are separated by `;`.

**Local character.** `--sharp u0.10` and `--round u0.60` override the corner
treatment around a place on the outline instead of everywhere. A 锐 place drops
to no rounding at its centre, so the corner stays exactly as traced, and it
keeps the marked vertex plus every second vertex nearby: a 3% stretch of the
outline roughly doubles its vertex count. A 圆 place adds two passes on top of
`--smooth`. Both fade back to the global value by 4.5% of the outline, measured
along it, so the two treatments blend rather than switch. The page marks them
with the 左键模式 control (锚点 / 锐 / 圆) and draws them violet and teal.
Measured at `--points 48 --smooth 2` on the stone: 14 vertices within 60 units
of `u0.10` become 31 with `--sharp u0.10`, and 10 near `u0.60` become 34 with
`--round u0.60`, while the marked vertex itself stays at its traced position to
0.0001 canvas units.

```bash
cd design/logo

# app icon
python simplify_path.py references/rosetta-stone-path.txt \
    -o out/rosetta-icon.svg --points 48 --anchor 12 --scale 0.88

# README pair: light-only half + self-adapting half
python simplify_path.py references/rosetta-stone-path.txt \
    -o out/rosetta-icon-docs.svg --points 48 --anchor 12 --scale 0.88 \
    --fill '#1e293b' --dark-fill ''
python simplify_path.py references/rosetta-stone-path.txt \
    -o out/rosetta-icon-github.svg --points 48 --anchor 12 --scale 0.88

# favicon: tight square, 4% padding, theme-aware
python simplify_path.py references/rosetta-stone-path.txt \
    -o out/favicon.svg --points 28 --anchor 10 --tight --pad 0.04 \
    --fill '#666666'

# cost of several budgets
python simplify_path.py references/rosetta-stone-path.txt --points 16,32,48,64 --stats
```

## Drafts and LAN use

The page keeps a draft in `localStorage` (保存草稿 / 载入草稿) so a session can
be resumed, and 导出草稿 / 导入草稿 move that JSON between machines. The draft
holds every parameter, the source path, manual anchor pins and unpins, and the
view zoom.

To let someone else open the page from another machine on the LAN:

```bash
cd design/logo
python3 -m http.server 8771 --bind 0.0.0.0
# then open http://<lan-ip>:8771/simplify.html
```

There is no server side beyond static files, and drafts never leave the browser
unless they are exported.

Open `simplify.html` through any static server (`python -m http.server`) rather
than as `file://`, so the clipboard and download helpers work as expected.

Interaction:

- click a vertex to add it as an anchor, and click an orange anchor to remove
  it. Hollow dots are candidates the current budget drops; they stay clickable,
  so an anchor can come off and go back on at any budget.
- 左键模式 switches what a left click places: 锚点 (orange, kept through the
  reduction), 锐 (violet, no rounding and more vertices nearby) or 圆 (teal, two
  extra rounding passes). A 锐 and a 圆 place on the same vertex replace each
  other, and either is kept like an anchor.
- right-click near a vertex for a red anchor; the click snaps to the nearest
  one within a few pixels, and the browser menu never opens over the canvas.
  It is stored as a position on the outline, so changing 曲线精度 leaves it on
  the same part of the shape instead of jumping to another vertex. It is kept
  like an orange one and it is also held in place: the deformation knobs turn
  and grow around the anchors' centroid, and that centroid is mapped back to
  the canvas point it had when you marked it. Its corner stays sharp through
  smoothing too. Right-click again, or click the red dot, to release it.
- Ctrl and the wheel zoom the preview around the cursor. Dragging pans once
  zoomed; at 100% the mark already fills the frame, so a drag says so instead
  of moving it. A double click or the 重置视图 button resets the view. The reset
  also re-centres the canvas: red anchors are re-referenced to the centred
  layout, so the mark goes back to the middle instead of drifting off it.
- 重置 in the header restores every parameter, clears manual anchors and resets
  the view.
- the 主题 toggle repaints the preview with that theme's fill: a dark page shows
  the light mark, a light page the dark one, the same pairing the archived SVGs
  use. Exported files keep the `prefers-color-scheme` media query instead, so
  they adapt wherever they land.
- on a wide screen the controls sit in three columns (simplify, preview,
  deform). They collapse to two below 1500 px and to one below 980 px, and the
  source-path panel stays collapsed until opened.
- the canvas-size slider changes the export coordinate space only. On-screen
  size comes from the `scale` knob.

## Copyright

The source file is
[File:Rosetta Stone.svg](https://commons.wikimedia.org/wiki/File:Rosetta_Stone.svg),
released by its author under CC0 1.0, so the trace carries no attribution or
share-alike obligation. The artifact's age does not settle this: the mark is a
trace of a specific image, and a trace is normally a derivative of that image.
CC0 is why that chain ends in no obligations. See `references/SOURCE.md` for
the recorded chain and its one caveat.

The files in this folder are not CC0. They are the project's mark. Their terms
are in [`LICENSE`](LICENSE) in this folder and in
[`TRADEMARKS.md`](../../TRADEMARKS.md). In short: use them to refer to the
project, keep the notice, do not use them as the branding of a product or
service, and do not claim or register them as your own.
