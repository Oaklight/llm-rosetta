#!/usr/bin/env python3
"""Turn a dense traced silhouette into a clean, parametric polygon mark.

The stone in this repo is a single closed path of ~126 cubic segments (748
numbers).  That is sharp at 256 px and mush at 16 px, and its shipped copies
disagree on crop, scale and fill.  This tool flattens the curves, reduces them
to a vertex budget, and re-frames the result from a few explicit numbers:

* ``--points``  vertex budget (8..512) - how much detail survives
* ``--scale``   optical size: the mark's longest side as a fraction of the canvas
* ``--canvas``  square canvas side
* ``--smooth``  optional Chaikin corner-cutting passes
* ``--rotate`` / ``--aspect`` / ``--inflate`` deformation of the prototype shape
* ``--method``  ``visvalingam`` (area) or ``douglas`` (deviation)

The same parameters drive ``simplify.html``, the interactive page next to this
file, so a look found by dragging sliders can be reproduced exactly here.

Examples::

    # app icon: 88% of a 600 canvas, 72 vertices, theme-aware fill
    python design/logo/simplify_path.py \\
        design/logo/out/rosetta-icon-docs.svg -o out/rosetta-icon.svg \\
        --points 72 --scale 0.88

    # favicon: tight crop, 36 vertices, single grey fill
    python design/logo/simplify_path.py \\
        design/logo/out/rosetta-icon-docs.svg -o out/favicon.svg \\
        --points 36 --tight --fill '#666' --dark-fill ''

    # report what several budgets cost
    python design/logo/simplify_path.py design/logo/out/rosetta-icon-docs.svg \\
        --points 16,32,64,128 --stats

No third-party dependencies.
"""

from __future__ import annotations

import argparse
import heapq
import math
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

Point = tuple[float, float]
# A segment is ("L", end) or ("C", c1, c2, end); the start is implied.
LineSeg = tuple[str, Point]
CurveSeg = tuple[str, Point, Point, Point]
Segment = LineSeg | CurveSeg

_TOKEN_RE = re.compile(
    r"[MmLlHhVvCcSsQqTtAaZz]|[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
)


# --------------------------------------------------------------------------
# Path parsing
# --------------------------------------------------------------------------


def _tokenize(d: str) -> list[str]:
    tokens = _TOKEN_RE.findall(d)
    if not tokens:
        raise ValueError("path data is empty")
    return tokens


def parse_path(d: str) -> list[tuple[Point, list[Segment]]]:  # noqa: C901
    """Parse SVG path data into ``(start, segments)`` pairs, one per subpath.

    Supports ``M L H V C S Q T Z``, absolute and relative.  Arcs (``A``) raise
    ``ValueError``: their flattening is easy to botch and nothing here uses them.
    """
    tokens = _tokenize(d)
    subpaths: list[tuple[Point, list[Segment]]] = []
    current: list[Segment] = []
    sub_start: Point = (0.0, 0.0)
    pos: Point = (0.0, 0.0)
    start: Point = (0.0, 0.0)
    prev_ctrl: Point | None = None
    cmd: str | None = None
    i = 0

    def read(n: int) -> list[float]:
        nonlocal i
        values: list[float] = []
        for _ in range(n):
            if i >= len(tokens) or (len(tokens[i]) == 1 and tokens[i].isalpha()):
                raise ValueError(f"path data ends early near token {i}")
            values.append(float(tokens[i]))
            i += 1
        return values

    def take() -> tuple[str, bool]:
        nonlocal cmd, i
        if len(tokens[i]) == 1 and tokens[i].isalpha():
            cmd = tokens[i]
            i += 1
        elif cmd is None:
            raise ValueError(f"path data starts with a number at token {i}")
        assert cmd is not None
        return cmd.upper(), cmd.islower()

    while i < len(tokens):
        upper, rel = take()

        if upper == "Z":
            if current:
                subpaths.append((sub_start, current))
                current = []
            pos = start
            prev_ctrl = None
            cmd = None
            continue

        if upper == "A":
            raise ValueError(
                "arc segments (A/a) are not supported; convert the arc to cubic "
                "curves first, or flatten the shape in your editor"
            )

        if upper == "M":
            x, y = read(2)
            pos = (pos[0] + x, pos[1] + y) if rel else (x, y)
            if current:
                subpaths.append((sub_start, current))
            current = []
            sub_start = start = pos
            prev_ctrl = None
            # Following coordinate pairs are implicit linetos.
            cmd = "l" if rel else "L"
            continue

        if upper == "L":
            x, y = read(2)
            pos = (pos[0] + x, pos[1] + y) if rel else (x, y)
            current.append(("L", pos))
            prev_ctrl = None

        elif upper == "H":
            (x,) = read(1)
            pos = (pos[0] + x, pos[1]) if rel else (x, pos[1])
            current.append(("L", pos))
            prev_ctrl = None

        elif upper == "V":
            (y,) = read(1)
            pos = (pos[0], pos[1] + y) if rel else (pos[0], y)
            current.append(("L", pos))
            prev_ctrl = None

        elif upper == "C":
            x1, y1, x2, y2, x, y = read(6)
            if rel:
                c1 = (pos[0] + x1, pos[1] + y1)
                c2 = (pos[0] + x2, pos[1] + y2)
                end = (pos[0] + x, pos[1] + y)
            else:
                c1, c2, end = (x1, y1), (x2, y2), (x, y)
            current.append(("C", c1, c2, end))
            prev_ctrl, pos = c2, end

        elif upper == "S":
            x2, y2, x, y = read(4)
            c1 = (
                (2 * pos[0] - prev_ctrl[0], 2 * pos[1] - prev_ctrl[1])
                if prev_ctrl
                else pos
            )
            if rel:
                c2 = (pos[0] + x2, pos[1] + y2)
                end = (pos[0] + x, pos[1] + y)
            else:
                c2, end = (x2, y2), (x, y)
            current.append(("C", c1, c2, end))
            prev_ctrl, pos = c2, end

        elif upper == "Q":
            x1, y1, x, y = read(4)
            if rel:
                q = (pos[0] + x1, pos[1] + y1)
                end = (pos[0] + x, pos[1] + y)
            else:
                q, end = (x1, y1), (x, y)
            c1, c2 = _quad_to_cubic(pos, q, end)
            current.append(("C", c1, c2, end))
            prev_ctrl, pos = q, end

        elif upper == "T":
            x, y = read(2)
            q = (
                (2 * pos[0] - prev_ctrl[0], 2 * pos[1] - prev_ctrl[1])
                if prev_ctrl
                else pos
            )
            end = (pos[0] + x, pos[1] + y) if rel else (x, y)
            c1, c2 = _quad_to_cubic(pos, q, end)
            current.append(("C", c1, c2, end))
            prev_ctrl, pos = q, end

        else:  # pragma: no cover - take() only returns known letters
            raise ValueError(f"unsupported path command {upper!r}")

    if current:
        subpaths.append((sub_start, current))
    return subpaths


def _quad_to_cubic(p0: Point, q: Point, p1: Point) -> tuple[Point, Point]:
    c1 = (p0[0] + 2.0 / 3.0 * (q[0] - p0[0]), p0[1] + 2.0 / 3.0 * (q[1] - p0[1]))
    c2 = (p1[0] + 2.0 / 3.0 * (q[0] - p1[0]), p1[1] + 2.0 / 3.0 * (q[1] - p1[1]))
    return c1, c2


# --------------------------------------------------------------------------
# Flattening
# --------------------------------------------------------------------------


def _mid(a: Point, b: Point) -> Point:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _seg_dist(p: Point, a: Point, b: Point) -> float:
    """Distance from *p* to segment *ab*, clamped to the endpoints."""
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return math.hypot(p[0] - ax, p[1] - ay)
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / length_sq))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


def _flatten_cubic(
    p0: Point, c1: Point, c2: Point, p1: Point, tol: float, out: list[Point]
) -> None:
    """Adaptive subdivision: recurse until the curve is within *tol* of a chord."""
    if _seg_dist(c1, p0, p1) <= tol and _seg_dist(c2, p0, p1) <= tol:
        out.append(p1)
        return
    p01, p12, p23 = _mid(p0, c1), _mid(c1, c2), _mid(c2, p1)
    p012, p123 = _mid(p01, p12), _mid(p12, p23)
    mid = _mid(p012, p123)
    _flatten_cubic(p0, p01, p012, mid, tol, out)
    _flatten_cubic(mid, p123, p23, p1, tol, out)


def flatten(
    subpaths: list[tuple[Point, list[Segment]]], tol: float = 0.05
) -> list[list[Point]]:
    """Convert parsed subpaths into dense point rings."""
    rings: list[list[Point]] = []
    for start, segs in subpaths:
        dense: list[Point] = [start]
        pos = start
        for seg in segs:
            if seg[0] == "L":
                pos = seg[1]
                dense.append(pos)
            else:
                _, c1, c2, end = seg
                _flatten_cubic(pos, c1, c2, end, tol, dense)
                pos = end
        rings.append(_dedupe(dense, 1e-6))
    return rings


def _dedupe(pts: list[Point], eps: float) -> list[Point]:
    out: list[Point] = []
    for p in pts:
        if not out or math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > eps:
            out.append(p)
    if (
        len(out) > 1
        and math.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) <= eps
    ):
        out.pop()
    return out


def is_ring(pts: list[Point], eps: float = 1e-6) -> bool:
    return (
        len(pts) > 2
        and math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) <= eps
    )


# --------------------------------------------------------------------------
# Simplification
# --------------------------------------------------------------------------

# Simplifiers work on indices so that "structural" vertices can be pinned: the
# stone's corners are what make it read as the Rosetta Stone, and a pure vertex
# budget will happily delete them.


def _tri_area(a: Point, b: Point, c: Point) -> float:
    return abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2.0


def _keep_visvalingam(
    pts: list[Point], target: int, closed: bool, pinned: frozenset[int] = frozenset()
) -> list[int]:
    """Visvalingam-Whyatt over indices; *pinned* vertices are never removed."""
    n = len(pts)
    if target >= n or target < 3:
        return list(range(n))

    prev = [(i - 1) % n for i in range(n)]
    nxt = [(i + 1) % n for i in range(n)]
    alive = [True] * n
    if not closed:
        prev[0], nxt[-1] = 0, n - 1

    def removable(i: int) -> bool:
        if i in pinned:
            return False
        return closed or i not in (0, n - 1)

    stamps = [0] * n
    heap: list[tuple[float, int, int]] = []

    def push(i: int) -> None:
        if alive[i] and removable(i):
            heapq.heappush(
                heap, (_tri_area(pts[prev[i]], pts[i], pts[nxt[i]]), i, stamps[i])
            )

    for i in range(n):
        push(i)

    remaining = n
    while remaining > target and heap:
        _, i, stamp = heapq.heappop(heap)
        if not alive[i] or stamps[i] != stamp:
            continue
        alive[i] = False
        p, q = prev[i], nxt[i]
        nxt[p], prev[q] = q, p
        remaining -= 1
        for j in (p, q):
            if alive[j]:
                stamps[j] += 1
                push(j)

    return [i for i in range(n) if alive[i]]


def _dp_arc(pts: list[Point], lo: int, hi: int, tol: float) -> set[int]:
    """Douglas-Peucker over the open index range [lo, hi]; endpoints stay."""
    keep = {lo, hi}
    stack = [(lo, hi)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        best, best_i = -1.0, a
        for i in range(a + 1, b):
            d = _seg_dist(pts[i], pts[a], pts[b])
            if d > best:
                best, best_i = d, i
        if best > tol:
            keep.add(best_i)
            stack.extend(((a, best_i), (best_i, b)))
    return keep


def _keep_douglas(
    pts: list[Point], tol: float, closed: bool, pinned: frozenset[int] = frozenset()
) -> list[int]:
    """Douglas-Peucker over indices, split at pinned vertices so they survive."""
    n = len(pts)
    order = sorted(pinned)
    if not order:
        order = [0]
    if not closed:
        keep: set[int] = {0, n - 1}
        bounds = [0] + [i for i in order if 0 < i < n - 1] + [n - 1]
        for a, b in zip(bounds, bounds[1:]):
            keep |= _dp_arc(pts, a, b, tol)
        return sorted(keep)

    # Rotate the ring so it starts at the first pinned vertex, then DP each arc
    # between consecutive pinned vertices.
    rot = order[0]
    idx = [(rot + k) % n for k in range(n + 1)]  # +1 closes the ring
    rot_pts = [pts[i % n] for i in idx]
    bounds = [0] + [idx.index(i) for i in order[1:]] + [n]
    keep = set()
    for a, b in zip(bounds, bounds[1:]):
        for j in _dp_arc(rot_pts, a, b, tol):
            keep.add(idx[j] % n)
    return sorted(keep)


def _apply_keep(pts: list[Point], keep: list[int], closed: bool) -> list[Point]:
    out = [pts[i] for i in keep]
    if closed and len(out) >= 3:
        if out[0] != out[-1]:
            out.append(out[0])
    return out


def _corner_anchors(pts: list[Point], count: int) -> frozenset[int]:
    """Pick the *count* sharpest corners, spaced along the outline."""
    n = len(pts)
    angles = []
    for i in range(n):
        a, b, c = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
        v1 = (b[0] - a[0], b[1] - a[1])
        v2 = (c[0] - b[0], c[1] - b[1])
        l1 = math.hypot(*v1) or 1.0
        l2 = math.hypot(*v2) or 1.0
        cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)))
        angles.append(math.degrees(math.acos(cos)))
    chosen: list[int] = []
    gap = max(1.0, n * 0.03)
    for i in sorted(range(n), key=lambda k: -angles[k]):
        if all(min(abs(i - c), n - abs(i - c)) > gap for c in chosen):
            chosen.append(i)
        if len(chosen) == count:
            break
    return frozenset(chosen)


def _convex_hull(pts: list[Point]) -> list[int]:
    """Monotone chain; returns hull vertex indices in order."""
    order = sorted(range(len(pts)), key=lambda i: (pts[i][0], pts[i][1]))

    def cross(o: int, a: int, b: int) -> float:
        return (pts[a][0] - pts[o][0]) * (pts[b][1] - pts[o][1]) - (
            pts[a][1] - pts[o][1]
        ) * (pts[b][0] - pts[o][0])

    lower: list[int] = []
    for i in order:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], i) <= 0:
            lower.pop()
        lower.append(i)
    upper: list[int] = []
    for i in reversed(order):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], i) <= 0:
            upper.pop()
        upper.append(i)
    return lower[:-1] + upper[:-1]


def _hull_depth(pts: list[Point]) -> list[float]:
    """Distance of each vertex inside the convex hull: 0 on the hull.

    A vertex not on the hull necessarily sits in a concavity, so this ranks the
    support points of the silhouette's dents — the structure an area metric
    never rewards, because the dent's area is small by definition.
    """
    hull = _convex_hull(pts)
    if len(hull) < 3:
        return [0.0] * len(pts)
    edges = [(pts[hull[k]], pts[hull[(k + 1) % len(hull)]]) for k in range(len(hull))]
    return [min(_seg_dist(p, a, b) for a, b in edges) for p in pts]


def _anchors_by(
    order: list[int], count: int, n: int, gap_frac: float = 0.03
) -> list[int]:
    """Greedy pick with arc-length non-maximum suppression."""
    chosen: list[int] = []
    gap = max(1.0, n * gap_frac)
    for i in order:
        if all(min(abs(i - c), n - abs(i - c)) > gap for c in chosen):
            chosen.append(i)
        if len(chosen) == count:
            break
    return chosen


def structural_anchors(
    pts: list[Point], count: int, closed: bool, mode: str = "area"
) -> frozenset[int]:
    """Pick the *count* vertices that carry the silhouette.

    ``area`` ranks by the triangle a removal would lose (favours the big convex
    sweeps); ``corner`` ranks by turning angle (favours sharp jags); ``depth``
    ranks by how far inside the convex hull a vertex sits (favours dents, since
    a hollow is exactly where the outline leaves the hull); ``hybrid`` takes
    half from area and fills the rest by depth.
    """
    if count <= 0 or len(pts) <= count:
        return frozenset()
    if mode == "corner":
        return frozenset(_corner_anchors(pts, count))
    if mode in ("depth", "hybrid"):
        depth = _hull_depth(pts)
        by_depth = sorted(range(len(pts)), key=lambda i: -depth[i])
        if mode == "depth":
            return frozenset(_anchors_by(by_depth, count, len(pts)))
        half = (count + 1) // 2
        area = sorted(_keep_visvalingam(pts, half, closed))
        chosen = _anchors_by(area, half, len(pts))
        for i in by_depth:
            if len(chosen) == count:
                break
            if all(
                min(abs(i - c), len(pts) - abs(i - c)) > max(1.0, len(pts) * 0.03)
                for c in chosen
            ):
                chosen.append(i)
        return frozenset(chosen)
    return frozenset(_keep_visvalingam(pts, count, closed))


def simplify_douglas(pts: list[Point], tol: float, closed: bool = True) -> list[Point]:
    """Douglas-Peucker with a geometric tolerance in user units."""
    return _apply_keep(pts, _keep_douglas(pts, tol, closed), closed)


def _keep_decimate(  # noqa: C901
    pts: list[Point],
    target: int,
    tol: float,
    closed: bool,
    pinned: frozenset[int],
) -> list[int]:
    """Greedy edge collapse with a chain-aware shape error.

    Each surviving vertex owns the arc between it and its predecessor, kept as
    the original points that arc replaced. Removing a vertex re-costs its
    neighbours against the merged arc, and a collapse is allowed only while the
    new edge stays within *tol* of every sample it replaces. Stop when the
    budget is met or the cheapest remaining collapse would exceed *tol*.

    This is the two-dimensional counterpart of mesh decimation: the same
    greedy edge-collapse order and the same shape bound, applied to the outline
    instead of a triangulated interior (a filled logo has no interior).
    """
    n = len(pts)
    if n <= 3:
        return list(range(n))

    prev = [(i - 1) % n for i in range(n)]
    nxt = [(i + 1) % n for i in range(n)]
    alive = [True] * n
    # left_arc[v]: original vertices strictly between prev[v] and v.
    left_arc: list[list[int]] = [[] for _ in range(n)]
    stamps = [0] * n
    if not closed:
        prev[0], nxt[-1] = 0, n - 1

    def removable(i: int) -> bool:
        if i in pinned:
            return False
        return closed or i not in (0, n - 1)

    def arc_between(i: int) -> list[int]:
        return left_arc[i] + [i] + left_arc[nxt[i]]

    def cost_of(i: int) -> float:
        a, b = pts[prev[i]], pts[nxt[i]]
        return max(_seg_dist(pts[j], a, b) for j in arc_between(i))

    heap: list[tuple[float, int, int]] = []

    def push(i: int) -> None:
        if alive[i] and removable(i):
            heapq.heappush(heap, (cost_of(i), i, stamps[i]))

    for i in range(n):
        push(i)

    remaining = n
    while heap and remaining > 3:
        cost, i, stamp = heapq.heappop(heap)
        if not alive[i] or stamps[i] != stamp:
            continue
        if cost > tol:
            break
        alive[i] = False
        p, q = prev[i], nxt[i]
        left_arc[q] = arc_between(i)
        nxt[p], prev[q] = q, p
        remaining -= 1
        if target < n and remaining <= target:
            break
        for j in (p, q):
            if alive[j]:
                stamps[j] += 1
                push(j)

    return [i for i in range(n) if alive[i]]


def reduce_to_budget(
    pts: list[Point],
    target: int,
    method: str,
    closed: bool = True,
    pinned: frozenset[int] = frozenset(),
    fidelity: float = 0.02,
) -> list[Point]:
    """Reduce *pts* to about *target* vertices, never dropping *pinned* ones.

    ``method="fidelity"`` ignores *target*: it trims every vertex that can go
    without moving the outline more than *fidelity* times the longest side, so
    the vertex count follows the shape instead of a budget.  The pinned
    vertices are kept exactly and split the ring into independent arcs.
    """
    n = len(pts)
    if n <= 3:
        return list(pts)
    if method == "fidelity":
        width, height = _bbox_size(pts)
        eps = max(fidelity, 1e-9) * max(width, height)
        return _apply_keep(pts, _keep_douglas(pts, eps, closed, pinned), closed)
    if method == "decimate":
        width, height = _bbox_size(pts)
        eps = max(fidelity, 1e-9) * max(width, height)
        keep = _keep_decimate(pts, target, eps, closed, pinned)
        return _apply_keep(pts, keep, closed)
    if target >= n or target < 3:
        return list(pts)
    if len(pinned) >= target:
        return _apply_keep(pts, sorted(pinned), closed)

    if method == "visvalingam":
        return _apply_keep(pts, _keep_visvalingam(pts, target, closed, pinned), closed)
    if method == "douglas":
        lo, hi = 0.0, max(_bbox_size(pts)) or 1.0
        best = _apply_keep(pts, _keep_douglas(pts, hi, closed, pinned), closed)
        for _ in range(40):
            mid_tol = (lo + hi) / 2.0
            best = _apply_keep(pts, _keep_douglas(pts, mid_tol, closed, pinned), closed)
            if abs(len(best) - target) <= 1:
                break
            if len(best) > target:
                lo = mid_tol
            else:
                hi = mid_tol
        return best
    raise ValueError(f"unknown method {method!r}")


def chaikin_src(
    pts: list[Point],
    src: list[int],
    passes: int,
    closed: bool,
    locked: frozenset[int] = frozenset(),
) -> tuple[list[Point], list[int]]:
    """Corner cutting that carries each vertex's source index along.

    *locked* holds source indices whose corners stay sharp: the vertex itself
    is kept and only its two incident edges are cut.  Without that exemption a
    red anchor stops being a vertex of the result after the first pass, which
    is why a pinned point drifts off the outline once smoothing is on.
    """
    if passes <= 0:
        return list(pts), list(src)
    return chaikin_budgeted(pts, src, [passes] * len(pts), closed, locked)


def chaikin_budgeted(
    pts: list[Point],
    src: list[int],
    budgets: list[int],
    closed: bool,
    locked: frozenset[int] = frozenset(),
) -> tuple[list[Point], list[int]]:
    """Corner cutting with a per-vertex budget of passes.

    A vertex with budget 0 keeps its corner exactly, budget 1 is one round of
    cutting, and so on; the global ``--smooth`` value is the default budget,
    and "锐" / "圆" regions override it locally.  Locked vertices are always
    budget 0.
    """
    ring = list(pts)
    tags = list(src)
    budget = [0 if t in locked else b for b, t in zip(budgets, src)]
    if closed and len(ring) > 1 and ring[0] == ring[-1]:
        ring, tags, budget = ring[:-1], tags[:-1], budget[:-1]
    top = max(budget, default=0)
    for k in range(1, top + 1):
        n = len(ring)
        if n < 3:
            break
        out: list[Point] = []
        out_tags: list[int] = []
        out_budget: list[int] = []
        for i in range(n if closed else n - 1):
            j = (i + 1) % n
            a, b = ring[i], ring[j]
            # A rounded vertex replaces itself with the cut point just after
            # it; a sharp one keeps its exact position.
            if budget[i] >= k:
                out.append((0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]))
                out_tags.append(tags[i])
                out_budget.append(budget[i])
            else:
                out.append(a)
                out_tags.append(tags[i])
                out_budget.append(budget[i])
            if budget[j] >= k:
                out.append((0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]))
                out_tags.append(tags[j])
                out_budget.append(budget[j])
        ring, tags, budget = out, out_tags, out_budget
    if closed and ring:
        ring.append(ring[0])
        tags.append(tags[0])
    return ring, tags


def chaikin(pts: list[Point], passes: int, closed: bool = True) -> list[Point]:
    """Corner cutting; *passes* rounds the polygon, 0 leaves it sharp."""
    if passes <= 0 or len(pts) < 3:
        return list(pts)
    return chaikin_src(pts, list(range(len(pts))), passes, closed)[0]


# --------------------------------------------------------------------------
# Deformation
# --------------------------------------------------------------------------


def _centroid(rings: list[list[Point]]) -> Point:
    pts = [p for r in rings for p in r]
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def rotate(
    rings: list[list[Point]], deg: float, origin: Point | None = None
) -> list[list[Point]]:
    """Rotate every ring about *origin*, or the bounding-box centre."""
    if deg == 0.0:
        return rings
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    if origin is None:
        x0, y0, x1, y1 = _bbox(rings)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    else:
        cx, cy = origin
    return [
        [
            (cx + (x - cx) * ca - (y - cy) * sa, cy + (x - cx) * sa + (y - cy) * ca)
            for x, y in r
        ]
        for r in rings
    ]


def stretch(
    rings: list[list[Point]], aspect: float, origin: Point | None = None
) -> list[list[Point]]:
    """Scale x by *aspect* about *origin*, or the bounding-box centre."""
    if aspect == 1.0:
        return rings
    if origin is None:
        x0, y0, x1, y1 = _bbox(rings)
        cx = (x0 + x1) / 2.0
    else:
        cx = origin[0]
    return [[(cx + (x - cx) * aspect, y) for x, y in r] for r in rings]


def inflate(
    rings: list[list[Point]], amount: float, origin: Point | None = None
) -> list[list[Point]]:
    """Push each vertex along its normal by *amount* times the longest side.

    Positive grows the silhouette, negative shrinks it.  Normals point away
    from *origin*, or from the centroid when no origin is given, so winding
    order does not matter.
    """
    if amount == 0.0:
        return rings
    x0, y0, x1, y1 = _bbox(rings)
    size = max(x1 - x0, y1 - y0) or 1.0
    d = amount * size
    cx, cy = origin if origin is not None else _centroid(rings)
    out: list[list[Point]] = []
    for ring in rings:
        closed = is_ring(ring)
        base = ring[:-1] if closed else ring
        m = len(base)
        if m < 3:
            out.append(ring)
            continue
        result: list[Point] = []
        for i, (x, y) in enumerate(base):
            px, py = base[(i - 1) % m] if closed else base[max(i - 1, 0)]
            nx, ny = base[(i + 1) % m] if closed else base[min(i + 1, m - 1)]
            ax, ay = x - px, y - py
            bx, by = nx - x, ny - y
            la, lb = math.hypot(ax, ay) or 1.0, math.hypot(bx, by) or 1.0
            # Left normals of both incident edges, averaged.
            mx, my = -ay / la - by / lb, ax / la + bx / lb
            length = math.hypot(mx, my)
            if length == 0.0:
                result.append((x, y))
                continue
            mx, my = mx / length, my / length
            if (mx * (x - cx) + my * (y - cy)) < 0.0:  # point away from centre
                mx, my = -mx, -my
            result.append((x + d * mx, y + d * my))
        if closed:
            result.append(result[0])
        out.append(result)
    return out


def deform(
    rings: list[list[Point]], args: argparse.Namespace, pivot: Point | None = None
) -> list[list[Point]]:
    """Apply the deformation knobs in a fixed, reproducible order.

    *pivot* is the point the knobs turn and grow around.  It defaults to the
    bounding-box centre (rotation, stretch) and the vertex centroid (inflate);
    red anchors pass their centroid instead, so a mark with locked points
    rotates and grows around them rather than around its own middle and then
    being shifted back into place.
    """
    rings = rotate(rings, args.rotate, pivot)
    rings = stretch(rings, args.aspect, pivot)
    rings = inflate(rings, args.inflate, pivot)
    return rings


# --------------------------------------------------------------------------
# Framing
# --------------------------------------------------------------------------


def _bbox_size(pts: list[Point]) -> tuple[float, float]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return max(xs) - min(xs), max(ys) - min(ys)


def _bbox(rings: list[list[Point]]) -> tuple[float, float, float, float]:
    xs = [p[0] for r in rings for p in r]
    ys = [p[1] for r in rings for p in r]
    return min(xs), min(ys), max(xs), max(ys)


@dataclass
class Framing:
    """Canvas framing: a square canvas, an optical scale, or a tight crop."""

    canvas: float = 600.0
    scale: float = 0.88
    tight: bool = False
    pad: float = 0.04
    transform: tuple[float, float, float] = field(
        default_factory=lambda: (0.0, 0.0, 1.0)
    )
    viewbox: str = ""

    def apply(
        self,
        rings: list[list[Point]],
        pivot: Point | None = None,
        target: Point | None = None,
    ) -> list[list[Point]]:
        """Place the mark in the canvas.

        Without *pivot* the mark is centred by its bounding box.  With a
        *pivot* and *target* the transform is set so that the pivot lands on
        the target instead, which is how a red anchor keeps its canvas point
        while the scale knob changes.
        """
        x0, y0, x1, y1 = _bbox(rings)
        w, h = x1 - x0, y1 - y0
        if self.tight:
            # Square, centred: icon slots are square, and a non-square viewBox
            # gets letterboxed by the browser at every size.
            side = max(w, h) * (1.0 + 2.0 * self.pad)
            self.transform = (-x0 + (side - w) / 2.0, -y0 + (side - h) / 2.0, 1.0)
            self.viewbox = f"0 0 {fmt(side)} {fmt(side)}"
        else:
            side = max(w, h) or 1.0
            k = (self.canvas * self.scale) / side
            tx = (self.canvas - w * k) / 2.0 - x0 * k
            ty = (self.canvas - h * k) / 2.0 - y0 * k
            self.transform = (tx, ty, k)
            self.viewbox = f"0 0 {fmt(self.canvas)} {fmt(self.canvas)}"
        if pivot is not None and target is not None:
            k = self.transform[2]
            self.transform = (
                target[0] - pivot[0] * k,
                target[1] - pivot[1] * k,
                k,
            )
        return rings


def fmt(value: float, precision: int = 2) -> str:
    text = f"{value:.{precision}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


# --------------------------------------------------------------------------
# Emit
# --------------------------------------------------------------------------


def emit_svg(
    rings: list[list[Point]],
    framing: Framing,
    *,
    fill: str = "#1e293b",
    dark_fill: str | None = "#ededed",
    as_: str = "polygon",
    precision: int = 1,
) -> str:
    """Render the rings as one SVG with a theme-aware fill."""
    tx, ty, k = framing.transform
    vb = framing.viewbox.split()
    out = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{framing.viewbox}" width="{vb[2]}" height="{vb[3]}">'
    ]
    if dark_fill:
        out.append(
            f"<style>.mark{{fill:{fill}}}"
            f"@media(prefers-color-scheme:dark){{.mark{{fill:{dark_fill}}}}}</style>"
        )
    transform = ""
    if (tx, ty) != (0.0, 0.0) or k != 1.0:
        transform = (
            f' transform="translate({fmt(tx, 3)},{fmt(ty, 3)}) scale({fmt(k, 5)})"'
        )
    out.append(f"<g{transform}>")
    fill_attr = "" if dark_fill else f' fill="{fill}"'
    for ring in rings:
        if as_ == "polygon":
            coords = " ".join(
                f"{fmt(p[0], precision)},{fmt(p[1], precision)}" for p in ring
            )
            out.append(f'<polygon class="mark" points="{coords}"{fill_attr}/>')
        else:
            d = f"M{fmt(ring[0][0], precision)},{fmt(ring[0][1], precision)}"
            d += "".join(
                f"L{fmt(p[0], precision)},{fmt(p[1], precision)}" for p in ring[1:]
            )
            out.append(f'<path class="mark" d="{d}Z"{fill_attr}/>')
    out.extend(("</g>", "</svg>"))
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# Input / CLI
# --------------------------------------------------------------------------


def path_data_from_svg(path: Path) -> str:
    """Return the longest ``d`` attribute in an SVG file (the silhouette)."""
    text = path.read_text()
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        # A file that is just path data (e.g. references/rosetta-stone-path.txt).
        stripped = text.strip()
        if stripped[:1].upper() == "M":
            return stripped
        raise
    candidates: list[str] = []
    for el in root.iter():
        if el.get("d"):
            candidates.append(el.get("d"))
        elif el.get("points"):
            # A polygon this tool produced is valid input too.
            pts = " ".join(el.get("points").split())
            candidates.append(
                f"M{pts.split()[0]}" + "".join(f" L{p}" for p in pts.split()[1:]) + " Z"
            )
    if not candidates:
        raise SystemExit(f"no path or polygon data found in {path}")
    return max(candidates, key=len)


def ring_fractions(pts: list[Point]) -> list[float]:
    """Normalised arc-length position of each vertex, 0 at the first one.

    A vertex index means nothing across flattening tolerances: the same index
    lands on a different part of the outline as soon as ``--tolerance``
    changes, which moved anchors around the shape.  A fraction of the outline
    length names the same place at every density.
    """
    ring = pts[:-1] if is_ring(pts) else list(pts)
    n = len(ring)
    if n == 0:
        return []
    steps = [
        math.hypot(ring[(i + 1) % n][0] - ring[i][0], ring[(i + 1) % n][1] - ring[i][1])
        for i in range(n)
    ]
    total = sum(steps) or 1.0
    out: list[float] = []
    acc = 0.0
    for step in steps:
        out.append(acc / total)
        acc += step
    return out


def index_at_fraction(fracs: list[float], u: float) -> int:
    """Index of the vertex closest to arc position *u*."""
    return min(range(len(fracs)), key=lambda i: abs(fracs[i] - u))


# A marked place keeps its effect at full strength within REGION_CORE of the
# outline and fades to the global setting by REGION_REACH.
REGION_CORE = 0.015
REGION_REACH = 0.045


def _arc_distance(fracs: list[float], u: float) -> list[float]:
    """Distance along the outline from every vertex to *u*, 0..0.5."""
    out = []
    for f in fracs:
        d = abs(f - u)
        out.append(min(d, 1.0 - d))
    return out


def _fade(d: float) -> float:
    """1 inside the core, 0 beyond the reach, linear in between."""
    if d <= REGION_CORE:
        return 1.0
    if d >= REGION_REACH:
        return 0.0
    return (REGION_REACH - d) / (REGION_REACH - REGION_CORE)


def region_budgets(
    fracs: list[float],
    base: int,
    sharp: list[float],
    rounded: list[float],
    extra: int = 2,
) -> list[int]:
    """Per-vertex Chaikin budgets with the local overrides applied.

    A 锐 place drops to 0 passes at its centre (the corner stays exactly) and a
    圆 place rises to *base* + *extra*, both fading back to *base* over the
    window, so a mark can be crisp in one place and soft in another.
    """
    budgets = [float(base)] * len(fracs)
    for u in sharp:
        for i, d in enumerate(_arc_distance(fracs, u)):
            t = _fade(d)
            budgets[i] = budgets[i] * (1 - t) + 0.0 * t
    for u in rounded:
        for i, d in enumerate(_arc_distance(fracs, u)):
            t = _fade(d)
            budgets[i] = budgets[i] * (1 - t) + (base + extra) * t
    return [max(0, int(round(b))) for b in budgets]


def region_pins(
    fracs: list[float], sharp: list[float], step: int = 2
) -> frozenset[int]:
    """Extra vertices to keep around a 锐 place, so the line stays detailed.

    Every *step*-th vertex inside the core is pinned, and every second one of
    those in the fade band: a 3% stretch of the outline roughly doubles its
    vertex count there without the whole budget following.
    """
    pinned: set[int] = set()
    for u in sharp:
        pinned.add(index_at_fraction(fracs, u))
        for i, d in enumerate(_arc_distance(fracs, u)):
            if d <= REGION_CORE:
                if i % step == 0:
                    pinned.add(i)
            elif d < REGION_REACH and i % (step * 2) == 0:
                pinned.add(i)
    return frozenset(pinned)


def parse_anchor_token(token: str) -> tuple[str, float]:
    """Read one anchor reference: ``u0.35`` is a fraction, ``120`` an index.

    The index form is what the page emitted before anchors were tied to the
    outline length; it still works but only at the tolerance it was recorded
    with.
    """
    token = token.strip()
    if token[:1] in ("u", "U"):
        return "u", float(token[1:])
    return "i", float(int(token))


def parse_anchors(text: str) -> list[tuple[str, float]]:
    """Parse a comma list of anchor references."""
    return [parse_anchor_token(t) for t in text.split(",") if t.strip()]


def resolve_anchors(
    fracs: list[float], refs: list[tuple[str, float]]
) -> frozenset[int]:
    """Turn anchor references into indices of the current flattened ring."""
    return frozenset(
        index_at_fraction(fracs, v) if kind == "u" else int(v) for kind, v in refs
    )


def parse_locks(text: str) -> dict[tuple[str, float], Point]:
    """Parse ``--lock`` entries: ``REF@X,Y``, several separated by ``;``.

    ``REF`` is ``u0.35`` (a fraction of the outline length) or a bare index,
    the same reference ``--pin`` takes, and ``X,Y`` is the canvas position that
    vertex is held at.
    """
    locks: dict[tuple[str, float], Point] = {}
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        ref, sep, xy = item.partition("@")
        try:
            if not sep:
                raise ValueError
            x, y = xy.split(",")
            locks[parse_anchor_token(ref)] = (float(x), float(y))
        except ValueError as exc:
            raise SystemExit(
                f"--lock entry {item!r} is not REF@X,Y (for example u0.35@312.5,244)"
            ) from exc
    return locks


def lock_pivot(
    rings: list[list[Point]], tags_by_ring: list[list[int]], locks: dict[int, Point]
) -> Point | None:
    """The centroid of the locked vertices, in the coordinates of *rings*.

    This is what the deformation knobs turn and grow around, so a locked point
    stays where it is instead of being pulled back by a translation afterwards.
    """
    if not locks:
        return None
    pts: list[Point] = []
    seen: set[int] = set()
    for ring, tags in zip(rings, tags_by_ring):
        for (x, y), tag in zip(ring, tags):
            if tag in locks and tag not in seen:
                seen.add(tag)
                pts.append((x, y))
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def lock_target(locks: dict[int, Point]) -> Point | None:
    """The canvas point the lock centroid is held at: the mean of the targets."""
    if not locks:
        return None
    pts = list(locks.values())
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Flatten and simplify an SVG silhouette into a clean polygon.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("input", nargs="?", type=Path, help="SVG file to read")
    src.add_argument("--d", metavar="PATH_DATA", help="path data passed inline")
    p.add_argument("-o", "--out", type=Path, help="output SVG (default: stdout)")
    p.add_argument(
        "--points", default="64", help="vertex budget, or a comma list (default: 64)"
    )
    p.add_argument(
        "--method",
        choices=("visvalingam", "douglas", "fidelity", "decimate"),
        default="visvalingam",
        help=(
            "reduction algorithm: visvalingam or douglas cut to --points; "
            "fidelity trims to --fidelity and ignores --points; decimate "
            "collapses edges greedily, bounded by both (default: visvalingam)"
        ),
    )
    p.add_argument(
        "--smooth", type=int, default=0, help="Chaikin passes, 0 = sharp (default: 0)"
    )
    p.add_argument(
        "--fidelity",
        type=float,
        default=0.02,
        help=(
            "max outline error for --method fidelity, as a fraction of the "
            "longest side (default: 0.02)"
        ),
    )
    p.add_argument(
        "--anchor",
        type=int,
        default=12,
        help="structural vertices that the simplifier may never drop (default: 12, 0 = off)",
    )
    p.add_argument(
        "--anchor-mode",
        choices=("area", "corner", "depth", "hybrid"),
        default="area",
        help="how anchors are ranked: area cost or turning angle (default: area)",
    )
    p.add_argument(
        "--pin",
        default="",
        metavar="IDX,...",
        help="extra flattened point indices to keep, per subpath (from simplify.html)",
    )
    p.add_argument(
        "--unpin",
        default="",
        metavar="IDX,...",
        help="auto-anchor indices to suppress, per subpath (from simplify.html)",
    )
    p.add_argument(
        "--lock",
        default="",
        metavar="IDX@X,Y;...",
        help=(
            "red anchors: keep vertex IDX and hold it at canvas point X,Y "
            "(from simplify.html, right-click)"
        ),
    )
    p.add_argument(
        "--sharp",
        default="",
        metavar="REF,...",
        help=(
            "锐 places: no rounding and extra vertices kept around them "
            "(u0.35 or an index)"
        ),
    )
    p.add_argument(
        "--round",
        default="",
        metavar="REF,...",
        help=("圆 places: two extra rounding passes around them, on top of --smooth"),
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="curve flattening tolerance (default: 0.05)",
    )
    p.add_argument(
        "--rotate", type=float, default=0.0, help="rotation in degrees (default: 0)"
    )
    p.add_argument(
        "--aspect",
        type=float,
        default=1.0,
        help="horizontal stretch factor (default: 1)",
    )
    p.add_argument(
        "--inflate",
        type=float,
        default=0.0,
        help="grow/shrink, fraction of the longest side (default: 0)",
    )
    p.add_argument(
        "--canvas", type=float, default=600.0, help="square canvas side (default: 600)"
    )
    p.add_argument(
        "--scale",
        type=float,
        default=0.88,
        help="mark's longest side as a fraction of the canvas (default: 0.88)",
    )
    p.add_argument(
        "--tight", action="store_true", help="crop to the mark's bounding box"
    )
    p.add_argument(
        "--pad",
        type=float,
        default=0.04,
        help="padding with --tight, as a fraction of the longest side (default: 0.04)",
    )
    p.add_argument(
        "--as",
        dest="as_",
        choices=("polygon", "path"),
        default="polygon",
        help="output element (default: polygon)",
    )
    p.add_argument("--fill", default="#1e293b", help="light-mode fill")
    p.add_argument(
        "--dark-fill", default="#ededed", help="dark-mode fill; empty = single fill"
    )
    p.add_argument(
        "--precision", type=int, default=1, help="coordinate decimals (default: 1)"
    )
    p.add_argument(
        "--stats", action="store_true", help="print vertex counts and byte sizes"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    d = args.d if args.d is not None else path_data_from_svg(args.input)

    dense = flatten(parse_path(d), args.tolerance)
    closed = [is_ring(r) for r in dense]
    total_in = sum(len(r) for r in dense)

    budgets = [int(x) for x in str(args.points).replace(" ", "").split(",") if x]
    pin_refs = parse_anchors(args.pin)
    unpin_refs = parse_anchors(args.unpin)
    sharp_refs = parse_anchors(args.sharp)
    round_refs = parse_anchors(args.round)
    lock_refs = parse_locks(args.lock)
    out_path: Path | None = args.out
    if out_path is None and not args.stats:
        out_path = Path("-")

    last_svg = ""
    for budget in budgets:
        rings: list[list[Point]] = []
        tags_by_ring: list[list[int]] = []
        locks: dict[int, Point] = {}
        locked: frozenset[int] = frozenset()
        for ring, is_closed in zip(dense, closed):
            fracs = ring_fractions(ring)
            # Anchors are named on the outline, so resolve them against this
            # ring: the indices they land on change with --tolerance, the
            # places they name do not.
            manual_pin = resolve_anchors(fracs, pin_refs)
            manual_unpin = resolve_anchors(fracs, unpin_refs)
            locks = {
                index_at_fraction(fracs, v) if kind == "u" else int(v): target
                for (kind, v), target in lock_refs.items()
            }
            stray = sorted(i for i in locks if not 0 <= i < len(ring))
            if stray:
                raise SystemExit(
                    f"--lock refers to {stray[0]}, outside the flattened "
                    f"outline (0..{len(ring) - 1})"
                )
            locked = frozenset(locks)
            sharp = [v if kind == "u" else fracs[int(v)] for kind, v in sharp_refs]
            rounded = [v if kind == "u" else fracs[int(v)] for kind, v in round_refs]
            ring_budgets = region_budgets(fracs, args.smooth, sharp, rounded)
            share = max(3, round(budget * len(ring) / max(total_in, 1)))
            pinned = (
                (
                    structural_anchors(ring, args.anchor, is_closed, args.anchor_mode)
                    | manual_pin
                    | region_pins(fracs, sharp)
                )
                - manual_unpin
            ) | locked
            reduced = reduce_to_budget(
                ring, share, args.method, is_closed, pinned, args.fidelity
            )
            # Carry each kept vertex's dense index through smoothing so a red
            # anchor can still be located after the vertex count changes.
            by_point = {p: i for i, p in enumerate(ring)}
            tags = [by_point.get(p, -1) for p in reduced]
            budgets = [
                ring_budgets[t] if 0 <= t < len(ring_budgets) else args.smooth
                for t in tags
            ]
            pts, tags = chaikin_budgeted(reduced, tags, budgets, is_closed, locked)
            rings.append(pts)
            tags_by_ring.append(tags)
        rings = deform(rings, args, lock_pivot(rings, tags_by_ring, locks))
        framing = Framing(
            canvas=args.canvas, scale=args.scale, tight=args.tight, pad=args.pad
        )
        rings = framing.apply(
            rings, lock_pivot(rings, tags_by_ring, locks), lock_target(locks)
        )
        last_svg = emit_svg(
            rings,
            framing,
            fill=args.fill,
            dark_fill=args.dark_fill or None,
            as_=args.as_,
            precision=args.precision,
        )
        if args.stats:
            n_out = sum(len(r) for r in rings)
            label = (
                f"fid {args.fidelity:g}"
                if args.method == "fidelity"
                else f"{budget:>4} pts"
            )
            print(
                f"{label}: {total_in} flattened -> {n_out} vertices, "
                f"{len(last_svg)} bytes, viewBox {framing.viewbox}",
                file=sys.stderr,
            )
        if out_path is not None and out_path != Path("-"):
            target = (
                out_path.with_name(f"{out_path.stem}-{budget}{out_path.suffix}")
                if len(budgets) > 1
                else out_path
            )
            target.write_text(last_svg)
            if not args.stats:
                print(f"wrote {target}", file=sys.stderr)

    if out_path is None or out_path == Path("-"):
        sys.stdout.write(last_svg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
