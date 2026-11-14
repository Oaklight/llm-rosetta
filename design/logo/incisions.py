#!/usr/bin/env python3
"""Pull the carved inscription out of the source tracing into its own layer.

The silhouette in ``references/rosetta-stone-path.txt`` is the *first* path of
the Commons file this mark descends from.  The other 22,000 paths are the
inscription: the stone is one black path, the carving is white shapes on top of
it.  This tool writes those shapes as one standalone layer the mark can knock
out, so ``simplify_path.py --incisions`` and ``simplify.html`` can put the
carving inside the simplified outline.  The layer stays out of the repository:
at two decimals it is about 5.8 MB, three orders of magnitude larger than every
shipped copy of the mark together.

The layer is written in the source's own coordinate system, which is also the
coordinate system of the outline (both come from the same file), so the caller
only has to draw it in the same group as the polygon.  Every element is emitted
as a path: tiny ``<rect>`` dots become four-line squares.

Examples::

    # the layer, ready for the page's 碑文 button
    python design/logo/incisions.py ~/Downloads/rosettastone.svg \\
        -o /tmp/rosetta-incisions.svg

    # keep the silhouette path too (the default drops the first path)
    python design/logo/incisions.py stone.svg --keep-silhouette -o all.svg

No third-party dependencies.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simplify_path as S  # noqa: E402  (sibling tool, same directory)

NUM = re.compile(r"-?(?:\d+\.\d+|\d+\.|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def squeeze(d: str) -> str:
    """Drop every character path data does not need.

    Whitespace is a separator only where removing it would fuse two numbers
    (``M 1 2`` is not ``M12``), so a run between two digits becomes a comma and
    a run anywhere else disappears.
    """
    parts = [p for p in re.split(r"[\s,]+", d.strip()) if p]
    if not parts:
        return ""
    out = parts[0]
    for part in parts[1:]:
        if out[-1] in "0123456789." and part[0] in "0123456789.":
            out += ","
        out += part
    return out


def minify(d: str, precision: int) -> str:
    """Round every number, then drop the separators that are no longer needed.

    Rounding comes first on purpose: ``-0.003`` becomes ``0``, and the sign was
    what separated it from the number before it, so the squeeze has to see the
    rounded text to know a comma is due there.
    """
    if precision >= 0:

        def fix(m: re.Match[str]) -> str:
            raw = m.group(0)
            value = f"{float(raw):.{precision}f}"
            if "." in value:
                value = value.rstrip("0").rstrip(".")
            if value == "-0":
                value = "0"
            # The minus sign separated this number from the previous one; when
            # rounding drops it, put a space back so the squeeze can keep them
            # apart.
            if raw.startswith("-") and not value.startswith("-"):
                value = " " + value
            return value

        d = NUM.sub(fix, d)
    return squeeze(d)


def rect_path(el: ET.Element, precision: int) -> str:
    """A ``<rect>`` as a path, so the layer is one element type."""
    x = float(el.get("x", 0.0))
    y = float(el.get("y", 0.0))
    w = float(el.get("width", 0.0))
    h = float(el.get("height", 0.0))
    f = f"{{:.{precision}f}}"
    return f"M{f.format(x)},{f.format(y)}h{f.format(w)}v{f.format(h)}h-{f.format(w)}z"


def ring_area(ring: list[tuple[float, float]]) -> float:
    """Unsigned area of one flattened ring (shoelace)."""
    total = 0.0
    for i in range(len(ring)):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % len(ring)]
        total += x0 * y1 - x1 * y0
    return abs(total) / 2.0


def mark_area(rings: list[list[tuple[float, float]]]) -> float:
    return sum(ring_area(r) for r in rings if len(r) >= 3)


def mark_centre(rings: list[list[tuple[float, float]]]) -> tuple[float, float]:
    """Bounding-box centre of one mark."""
    xs = [p[0] for r in rings for p in r]
    ys = [p[1] for r in rings for p in r]
    return ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)


def abstract_square(rings: list[list[tuple[float, float]]], precision: int) -> str:
    """One square of the same area for one mark.

    At icon sizes a carved glyph is a few pixels across; what survives is the
    texture of rows of marks, not their shape.  A square of equal area keeps the
    mass and the position and costs about twenty bytes.
    """
    area = mark_area(rings)
    cx, cy = mark_centre(rings)
    half = math.sqrt(area) / 2.0 if area > 0 else 0.0
    f = f"{{:.{precision}f}}"

    def v(x: float) -> str:
        text = f.format(x)
        return text

    return f"M{v(cx - half)},{v(cy - half)}h{v(2 * half)}v{v(2 * half)}h{v(-2 * half)}z"


def simplify_mark(
    rings: list[list[tuple[float, float]]], points: int, precision: int
) -> str:
    """Each mark's outline cut to a few vertices, as path data."""
    out = []
    for ring in rings:
        if len(ring) < 3:
            continue
        red = S.reduce_to_budget(
            ring, min(points, len(ring)), "visvalingam", True, frozenset(), 10.0
        )
        if len(red) < 3:
            continue
        pts = " ".join(
            f"{value:.{precision}f},{value2:.{precision}f}" for value, value2 in red
        )
        out.append("M" + pts.replace(" ", "L") + "Z")
    return "".join(out)


def cook(
    raw: list[str],
    *,
    precision: int,
    tolerance: float,
    simplify_points: int,
    abstract: bool,
    min_area: float,
    step: int,
) -> tuple[list[str], int]:
    """Turn the raw marks into the layer's subpaths.

    ``--min-area`` and ``--step`` trim; ``--simplify`` and ``--abstract`` shrink
    what is left.  The exact mode never flattens, so a layer nobody asked to
    reduce stays the tracing it came from.
    """
    out: list[str] = []
    kept = 0
    for index, d in enumerate(raw):
        if step > 1 and index % step:
            continue
        if min_area > 0 or simplify_points or abstract:
            try:
                rings = S.flatten(S.parse_path(d), tolerance)
            except Exception:
                continue
            if min_area > 0 and mark_area(rings) < min_area:
                continue
            if abstract:
                out.append(abstract_square(rings, precision))
            elif simplify_points:
                out.append(simplify_mark(rings, simplify_points, precision))
            else:
                out.append(d)
        else:
            out.append(d)
        kept += 1
    return out, kept


def extract(src: Path, drop_first: bool, precision: int) -> tuple[list[str], str]:
    """Return the incision paths and the source viewBox."""
    root = ET.fromstring(src.read_text())
    view_box = (
        root.get("viewBox")
        or f"0 0 {root.get('width', '600')} {root.get('height', '600')}"
    )
    paths: list[str] = []
    dropped = False
    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "path" and el.get("d"):
            d = el.get("d", "")
            if drop_first and not dropped:
                dropped = True
                continue
            paths.append(minify(d, precision))
        elif tag == "rect" and el.get("width"):
            if drop_first and not dropped:
                dropped = True
                continue
            paths.append(rect_path(el, precision))
    return paths, view_box


def build(paths: list[str], view_box: str) -> str:
    """One path element: every incision is its own subpath, so one fill rule.

    A separate element per mark costs about 10 bytes of markup for each of the
    22,000 marks and buys nothing — the subpaths do not interact under the
    nonzero rule.
    """
    body = "".join(paths)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{view_box}">\n'
        "<!-- Carved inscription, extracted from the Commons tracing that the\n"
        "     stone outline comes from (design/logo/references/SOURCE.md).\n"
        "     Generated by design/logo/incisions.py; do not edit by hand. -->\n"
        f'<path fill="#000" d="{body}"/>\n'
        "</svg>\n"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract the carved inscription from the source SVG tracing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("src", type=Path, help="the source tracing (Rosetta_Stone.svg)")
    p.add_argument("-o", "--out", type=Path, help="output layer (default: stdout)")
    p.add_argument(
        "--keep-silhouette",
        action="store_true",
        help="keep the first path (the stone outline) in the layer",
    )
    p.add_argument(
        "--precision",
        type=int,
        default=2,
        help="coordinate decimals, -1 to leave the numbers as written (default: 2)",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.1,
        help="flattening tolerance for --min-area / --simplify / --abstract (default: 0.1)",
    )
    p.add_argument(
        "--min-area",
        type=float,
        default=0.0,
        metavar="A",
        help="drop marks whose area is below A (source units squared)",
    )
    p.add_argument(
        "--step",
        type=int,
        default=1,
        metavar="N",
        help="keep every Nth mark (default: 1, keep all)",
    )
    p.add_argument(
        "--simplify",
        type=int,
        default=0,
        metavar="POINTS",
        help="cut each mark's outline to at most POINTS vertices",
    )
    p.add_argument(
        "--abstract",
        action="store_true",
        help="replace each mark with a square of the same area (texture at icon sizes)",
    )
    p.add_argument("--stats", action="store_true", help="print counts and bytes")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths, view_box = extract(args.src, not args.keep_silhouette, args.precision)
    if not paths:
        raise SystemExit(f"no path elements found in {args.src}")
    total = len(paths)
    paths, kept = cook(
        paths,
        precision=args.precision,
        tolerance=args.tolerance,
        simplify_points=args.simplify,
        abstract=args.abstract,
        min_area=args.min_area,
        step=args.step,
    )
    if not paths:
        raise SystemExit("every mark was filtered out; relax --min-area/--step")
    svg = build(paths, view_box)
    if args.stats:
        data = sum(len(p) for p in paths)
        mode = (
            "abstract"
            if args.abstract
            else f"simplify {args.simplify}"
            if args.simplify
            else "exact"
        )
        print(
            f"{kept}/{total} marks, {mode}, {data / 1e6:.2f} MB of path data, "
            f"{len(svg) / 1e6:.2f} MB of SVG, viewBox {view_box}"
        )
    if args.out:
        args.out.write_text(svg)
        if not args.stats:
            print(f"wrote {args.out} ({len(svg) / 1e6:.2f} MB)", file=sys.stderr)
    else:
        sys.stdout.write(svg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
