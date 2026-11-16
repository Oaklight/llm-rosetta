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

The design is a draft: `references/rosetta-draft.json` is what the page's
导出草稿 writes, and both front-ends run it. The CLI takes it with `--draft`, so
regenerating a copy does not mean re-typing a dozen flags — and cannot mean
re-typing them wrong.

```bash
cd design/logo
python simplify_path.py --draft references/rosetta-draft.json -o out/rosetta-icon.svg
python simplify_path.py --draft references/rosetta-draft.json -o out/rosetta-icon-github.svg
python simplify_path.py --draft references/rosetta-draft.json --dark-fill '' -o out/rosetta-icon-docs.svg
python simplify_path.py --draft references/rosetta-draft.json --fill '#666666' -o out/favicon.svg
```

A flag given beside `--draft` wins over the draft, which is how the four copies
differ only in fill. The draft asks for 15 points, keeps 15 structural anchors,
flattens at `--tolerance 0.52`, rounds once with `--smooth 1`, and frames tight
with 4% padding; it also carries seven pins, three unpins, four red anchors, two
锐 places and six 圆 places, all named by their position on the outline.

| Use | File | fill |
| --- | --- | --- |
| App icon | `out/rosetta-icon.svg` | `#1e293b` / `#ededed` |
| README, dark half | `out/rosetta-icon-github.svg` | `#1e293b` / `#ededed` |
| README, light half | `out/rosetta-icon-docs.svg` | `#1e293b` only |
| Favicon | `out/favicon.svg` | `#666666` / `#ededed` |

`points` is the vertex budget. `anchors` is the number of structural vertices
that may never be simplified away, ranked by the triangle area a removal would
cost. On this draft the protection binds before the budget does: the 15
structural anchors, 7 pins, 4 red anchors and the 锐 places' detail come to 34
of the 263 flattened vertices, more than the 15 the budget asks for, so the
reduction keeps all 34 and one rounding pass turns them into the 128 points the
file carries.

## Shipped copies, measured

| Location | File | Fill |
| --- | --- | --- |
| README (en/zh) | `design/logo/out/rosetta-icon-{github,docs}.svg` | `#1e293b` / `#ededed`; docs light only |
| Package READMEs | `out/rosetta-icon-github.svg` | `#1e293b` / `#ededed` |
| Docs header | `docs_{en,zh}/docs/images/rosetta_stone.svg` | `currentColor` |
| Docs favicon | `docs_{en,zh}/docs/images/favicon.svg` | `#1a1a1a` / `#ededed` |
| Gateway favicon | `admin/routes/auth.py` `_FAVICON_SVG`; `admin.html` data URI | `#666666` / `#ededed`, tight at viewBox `634.78` |
| Gateway sidebar/header | `admin/admin.html` inline `<svg>` | `currentColor`, `634.78` box |

The README, favicon, gateway and docs copies are generated from it, with the
fill as the only difference, and the docs files land in the docs worktrees:

```bash
python design/logo/simplify_path.py \
    --draft design/logo/references/rosetta-draft.json \
    --fill 'currentColor' --dark-fill '' -o docs_en/docs/images/rosetta_stone.svg
python design/logo/simplify_path.py \
    --draft design/logo/references/rosetta-draft.json \
    --fill '#1a1a1a' -o docs_en/docs/images/favicon.svg
# the same two into docs_zh/docs/images/
```

The README pair used to be byte-identical, so the `prefers-color-scheme` switch
rendered the same file twice. The light half is now a light-only file and the
dark half keeps its own media query. The package READMEs reference the dark
file directly, and they still adapt on their own.

The gateway follows the same draft. Its favicon is the draft at
`--fill '#666666'`, the page carries a data URI of that favicon, and the two
inline marks in `admin.html` use the same 128-point geometry at
`fill="currentColor"`.

`out/favicon-gateway.svg` is a tracked copy of exactly what the gateway serves
today, so the mark it ships stays visible in the repository instead of living
only in a byte string inside `auth.py`.

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
half from area and fills the rest by depth. At the shipped `--tolerance 0.52`,
212 of the 263 flattened vertices are dents, yet `area` spends only 4 of its 15
picks on one, `corner` 8, and `depth` all 15.

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
outline roughly doubles its vertex count. Those extra vertices are protected
through the reduction and are offered as vertices nowhere: inside a 锐 stretch
the page draws only the violet marker, so it cannot fill with dots. A 圆 place
adds two passes on top of
`--smooth`. Both fade back to the global value by 4.5% of the outline, measured
along it, so the two treatments blend rather than switch. The page marks them
with the 左键模式 control (锚点 / 锐 / 圆) and draws them violet and teal.
On the shipped draft the two treatments account for most of the mark's detail:
`--sharp ''` beside `--draft` drops the two 锐 places and leaves 82 of the 128
vertices, and `--round ''` drops the six 圆 places and leaves 42. Those two
flags replace the draft's lists rather than add to them, which is how the
comparison is made.

**Centring.** `--center bbox` (the default) puts the bounding box on the canvas
centre; `--center centroid` puts the filled shape's area centroid there
instead. The two differ on this stone because it is wider at the bottom:
framed the way the draft frames it, the box centre is `(317.4, 317.4)` and the
filled shape's centroid is `(308.2, 331.5)`, 14 units lower and 9 to the left.
The page switches between them with 居中基准 (中心 / 重心) and
bakes the choice with 居中, which also moves any red anchors' recorded points so
they stay on the vertices they were marked on.

**Inscription.** The Commons file is the stone plus the carved text: one black
outline and 22,040 white paths on top of it. `incisions.py` pulls those paths
out into a standalone layer in the same coordinate system:

```bash
python design/logo/incisions.py ~/Downloads/rosettastone.svg \
    -o /tmp/rosetta-incisions.svg
```

The layer is 5.8 MB at two decimals, three orders of magnitude larger than every
shipped copy of the mark together, so it is generated on demand and never
committed. `simplify_path.py --incisions FILE` cuts the mark with it: the data
becomes a `<mask>` the polygon is filled through, so the grooves show whatever
the file sits on instead of a colour that only matches one background, and they
follow the mark because they live in the same user space as the outline. The
page has the same thing behind 碑文: paste a URL (the Commons file serves
`access-control-allow-origin: *`, so the original loads straight from Wikimedia,
and a jsDelivr mirror works if the file is published to GitHub instead) or pick
a local file, tick 显示碑文, and the carving follows the deformation and the
framing, clipped to the outline — on the canvas and on every size preview. The
outline is taken as the *first* path of the source, because the longest one is
a carved glyph; a layer written by `incisions.py` has a single path and is used
as it is. The layer is memory-only: a reload asks for it again, and it is not
written into the draft, which would blow the `localStorage` quota. At icon sizes
the carving reads as texture, not as letters.

Reduction is for the cases where 5.8 MB is too much. Measured on the 8.2 MB
source, 22,149 marks:

| flags | layer | marks |
| --- | --- | --- |
| *(none)* | 5.82 MB | 22,149 |
| `--simplify 5` | 1.72 MB | 22,149, each outline cut to 5 vertices |
| `--step 3` | 1.92 MB | 7,383 |
| `--min-area 0.5` | 4.90 MB | 12,655 |
| `--abstract` | 0.68 MB | 22,149, each mark a square of equal area |
| `--abstract --step 3` | 0.23 MB | 7,383 |

`--abstract` gives up the glyph shapes and keeps their mass and position, which
is what survives at 16–64 px; `--simplify` keeps the shapes and cuts the
vertices, which is what a large rendering needs. `--min-area` and `--step` trim
marks and combine with either.

**Dialects.** The stone carries the same decree three times, and
`json_incisions.py` sets three LLM request dialects into its three registers the
same way — in the only alphabet the stone has. Every distinct character is
mapped to one of the carved marks, so a character always looks the same and the
result is cut from the stone's own vocabulary rather than from a typeface. The
mapping is a hash, so the same inputs always give the same layer.

```bash
python design/logo/json_incisions.py \
    --source ~/Downloads/rosettastone.svg \
    --bands design/logo/references/dialects/chat_completions.json \
            design/logo/references/dialects/messages.json \
            design/logo/references/dialects/google_generate.json \
    -o /tmp/dialects.svg
```

The three payloads in `references/dialects/` are small Chat Completions,
Messages and `generateContent` requests for the same job; any text works, one
to three files, one register each. The layer is written like the tracing's own
carving — one path, same coordinate system — so `--incisions` and the page take
it without a special case. Measured on the three shipped payloads: 26 rows, 661
marks, 0.13 MB.

```bash
cd design/logo

# cost of several budgets
python simplify_path.py references/rosetta-stone-path.txt --points 16,32,48,64 --stats
```

## Drafts and LAN use

The page keeps a draft in `localStorage` (保存草稿 / 载入草稿) so a session can
be resumed, and 导出草稿 / 导入草稿 move that JSON between machines. The draft
holds every parameter, the source path, manual anchor pins and unpins, and the
view zoom.

The same file is the CLI's input. `--draft FILE` loads it, and a flag written
beside `--draft` overrides the draft's value, so the four shipped copies differ
only in fill. The page therefore decides the design, and the CLI replays it.

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
- 居中基准 chooses what 居中 aligns to — 中心 (the bounding box) or 重心 (the
  filled shape's area centroid) — and 居中 applies it before you export,
  re-referencing any red anchors so they stay on their own vertices.
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
