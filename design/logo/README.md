# The stone mark

One silhouette, one path, several framings. Every published copy of the logo —
README, docs, gateway — uses the same traced outline of the Rosetta Stone; only
the crop, the scale and the fill differ. This folder holds the parametric
tooling that regenerates any of those copies from the source path.

## Source

`references/rosetta-stone-path.txt` is the trace and the single source of
truth: one closed path, 126 cubic segments, 4,726 bytes of `d` data, bounding
box 451.5 x 589.6 user units. Every file in `out/` is regenerated from it.

Provenance: the outline was produced by algorithmic tracing of a Wikimedia
Commons file, [File:Rosetta Stone.svg](https://commons.wikimedia.org/wiki/File:Rosetta_Stone.svg),
authored by **kinetoons** (2014, from open clip art) and released under
**CC0 1.0**. `references/SOURCE.md` records the file, the author, the licence
and the caveat.

CC0 requires no attribution, so the mark carries no licence obligation beyond
the repository's own. Two consequences worth keeping in mind:

- the shape is not original artwork — it is a derived trace, and the trace chain
  is recorded rather than claimed;
- if the mark is ever registered or the project is sold, re-derive from a
  documented source anyway; CC0 at the top of a chain does not clear rights
  further down that the uploader never held.

## Canonical parameters

| Use | File | points | anchors | framing | fill |
| --- | --- | --- | --- | --- | --- |
| App icon | `out/rosetta-icon.svg` | 48 | 12 | 600 canvas, scale 0.88 | `#1e293b` / `#ededed` |
| README, dark half | `out/rosetta-icon-github.svg` | 48 | 12 | 600 canvas, scale 0.88 | `#1e293b` / `#ededed` |
| README, light half | `out/rosetta-icon-docs.svg` | 48 | 12 | 600 canvas, scale 0.88 | `#1e293b` only |
| Favicon | `out/favicon.svg` | 28 | 10 | tight square, pad 0.04 | `#666666` / `#ededed` |

`points` is the vertex budget; `anchors` is the number of structural vertices
that may never be simplified away (ranked by the triangle area a removal would
cost). Above ~64 vertices the mark stops changing visibly; the difference
between a good and a bad small mark is which vertices survive, not how many.

## Shipped copies, measured

| Location | File | Framing | Fill |
| --- | --- | --- | --- |
| README (en/zh) | `design/logo/out/rosetta-icon-{github,docs}.svg` | 600 box, scale 0.88 | `#1e293b` / `#ededed`; docs light only |
| Package READMEs | `out/rosetta-icon-github.svg` | 600 box, scale 0.88 | `#1e293b` / `#ededed` |
| Docs header | `docs_{en,zh}/docs/images/rosetta_stone.svg` | 600 box, scale 0.934 | `currentColor` |
| Docs favicon | `docs_{en,zh}/docs/images/favicon.svg` | 600 box, scale 0.934 | `#1a1a1a` / `#ededed` |
| Gateway favicon | `admin/routes/auth.py` `_FAVICON_SVG` | viewBox `30 15 490 500` | `#666` |
| Gateway sidebar/header | `admin/admin.html` inline `<svg>` | viewBox `30 15 490 500` | `currentColor` |

**Fixed here.** The README pair used to be byte-identical, so the
`prefers-color-scheme` switch rendered the same file twice. The light half is
now a light-only file and the dark half keeps its own media query; the package
READMEs, which reference the dark file directly, still adapt on their own.

**Still open.** The gateway favicon crops to viewBox `30 15 490 500` while the
outline spans y 3.3..593.0, so the bottom ~78 units (13%) of the stone are cut
off — that is why the gateway favicon does not match the README mark. Replacing
it with `out/favicon.svg` (tight square, 4% padding) is a gateway change and is
left for its own PR.

## Tooling

`simplify_path.py` (CLI, stdlib only) and `simplify.html` (interactive, no
network) implement the same pipeline and produce byte-identical geometry for
the same parameters:

1. parse the path; flatten cubics adaptively (`--tolerance`);
2. pick structural anchors (`--anchor`, `--anchor-mode area|corner|depth|hybrid`);
3. reduce to the budget (`--points`, `--method visvalingam|douglas`) without
   dropping an anchor;
4. optional Chaikin rounding (`--smooth`);
5. deform: `--rotate`, `--aspect`, `--inflate`, `--symmetrize`, `--snap`;
6. frame: square canvas (`--canvas`, `--scale`) or tight square (`--tight`,
   `--pad`).

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

The page keeps a draft in `localStorage` (**保存草稿** / **载入草稿**) so a
session can be resumed, and **导出草稿 / 导入草稿** move that JSON between
machines. The draft holds every parameter, the source path, manual anchor
pins/unpins and the view zoom.

To let someone else use the page from another machine on the LAN:

```bash
cd design/logo
python3 -m http.server 8771 --bind 0.0.0.0
# then open http://<lan-ip>:8771/simplify.html
```

There is no server side beyond static files; drafts never leave the browser
unless they are exported.

Open `simplify.html` through any static server (`python -m http.server`), not
as `file://`, so the clipboard and download helpers work as expected.

Interaction:

- **click a vertex** to add it as an anchor; **click an orange anchor** to
  remove it. Hollow dots are candidates the current budget drops — they stay
  clickable, so an anchor can be taken off and put back at any budget.
- **Ctrl + wheel** zooms the preview around the cursor; **drag** pans;
  **double click** or the *重置视图* button resets the view. The canvas grid
  stays in step, so pixel detail is inspectable without changing the exported
  geometry.
- **重置** in the header restores every parameter, clears manual anchors and
  resets the view.
- the canvas-size slider only changes the export coordinate space; the visual
  size on screen is the `scale` knob.

## Copyright

Not legal advice. Two separate questions, one now settled.

**Copyright — settled as CC0.** The source file is
[File:Rosetta Stone.svg](https://commons.wikimedia.org/wiki/File:Rosetta_Stone.svg),
released by its author under CC0 1.0, so the trace carries no attribution or
share-alike obligation. This matters more than the age of the artifact: the
mark is not a photograph of the stone but a trace of a specific image, and a
trace is normally a derivative of that image. CC0 is the reason that chain ends
in no obligations. See `references/SOURCE.md` for the recorded chain and its
one caveat.

**Trademark — independent of copyright.** "Rosetta Stone" is a registered mark
for language software, and Apple's Rosetta is an established mark for
translation layers. A stone silhouette is generic enough that neither is a
blocker today, but this is the axis to watch if the project is ever
commercialised or the mark is registered.

**Anchor modes.** `area` ranks by the triangle a removal would lose (favours
the big convex sweeps), `corner` by turning angle (favours sharp jags), `depth`
by distance inside the convex hull (favours dents, since a hollow is exactly
where the outline leaves the hull), `hybrid` takes half from area and fills the
rest by depth. On this outline 46% of the flattened vertices are locally
reflex, yet `area` selects 1 of 12; `corner` 7 of 12; `depth` 12 of 12 (all
non-hull).
