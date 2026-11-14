#!/usr/bin/env python3
"""Set three API dialects into the stone's own alphabet.

The Rosetta Stone carries the same decree three times — hieroglyphs, demotic,
Greek — and this mark comes from a tracing of it.  The idea here is the same
one rung down: three LLM request dialects, each set into one of the stone's
three registers, in the only alphabet the stone has.

There is no font.  Every distinct character in the three texts is mapped to one
of the 22,000 carved marks of the original tracing, so a character always looks
the same and the result is cut from the stone's own vocabulary of shapes rather
than from a typeface.  The mapping is deterministic (a hash of the character),
so the same inputs always produce the same layer.

Examples::

    python design/logo/json_incisions.py \\
        --source ~/Downloads/rosettastone.svg \\
        --bands design/logo/references/dialects/chat_completions.json \\
                design/logo/references/dialects/messages.json \\
                design/logo/references/dialects/google_generate.json \\
        -o /tmp/dialects.svg --stats

The layer is written in the source's coordinate system, like ``incisions.py``'s,
so ``simplify_path.py --incisions`` and the page's 碑文 panel take it directly.

No third-party dependencies.
"""

from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import incisions  # noqa: E402  (sibling tool, same directory)
import simplify_path as S  # noqa: E402

Point = tuple[float, float]

# The three registers of the stone, as fractions of its height.  The original
# tracing has visible gaps between them; these lines sit inside those gaps.
BANDS = ((0.02, 0.40), (0.43, 0.56), (0.60, 0.97))
# Cell proportions: a carved mark is taller than wide.
CELL_RATIO = 0.62


def load_marks(source: Path, tolerance: float) -> list[list[list[Point]]]:
    """Flatten every carved mark of the tracing into rings."""
    paths, _ = incisions.extract(source, drop_first=True, precision=2)
    marks = []
    for d in paths:
        try:
            rings = S.flatten(S.parse_path(d), tolerance)
        except Exception:  # noqa: BLE001 - a damaged mark is not fatal
            continue
        rings = [r for r in rings if len(r) >= 3]
        if rings:
            marks.append(rings)
    return marks


def bbox(rings: list[list[Point]]) -> tuple[float, float, float, float]:
    xs = [p[0] for r in rings for p in r]
    ys = [p[1] for r in rings for p in r]
    return min(xs), min(ys), max(xs), max(ys)


def alphabet(chars: list[str], marks: list[list[list[Point]]]) -> dict[str, int]:
    """One mark per character, stable across runs and spread over the set.

    A hash picks the starting point and a stride walks to the next unused mark,
    so two characters that hash together still get different shapes as long as
    there are enough marks to go round.
    """
    stride = 7919 % len(marks) or 1
    used: set[int] = set()
    chosen: dict[str, int] = {}
    for ch in chars:
        start = zlib.crc32(ch.encode("utf-8")) % len(marks)
        index = start
        for _ in range(len(marks)):
            if index not in used:
                break
            index = (index + stride) % len(marks)
        used.add(index)
        chosen[ch] = index
    return chosen


def wrap(text: str, columns: int) -> list[str]:
    """Greedy wrap of whitespace-collapsed text."""
    words = " ".join(text.split()).split(" ")
    lines: list[str] = []
    line = ""
    for word in words:
        if not line:
            line = word
        elif len(line) + 1 + len(word) <= columns:
            line += " " + word
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def band_layout(
    text: str, width: float, height: float
) -> tuple[list[str], float, float]:
    """Line breaks and cell size for one register.

    The row height is chosen so the wrapped text fills the register instead of
    leaving it half empty; the joins are monotone, so a few passes settle it.
    """
    cell_h = height / 10.0
    for _ in range(12):
        columns = max(8, int(width / (cell_h * CELL_RATIO)))
        lines = wrap(text, columns)
        if len(lines) * cell_h <= height:
            break
        cell_h *= 0.9
    while len(lines) * cell_h < height * 0.8 and cell_h < height / 3:
        cell_h /= 0.9
        columns = max(8, int(width / (cell_h * CELL_RATIO)))
        lines = wrap(text, columns)
        if len(lines) * cell_h > height:
            break
    cell_w = cell_h * CELL_RATIO
    return lines, cell_w, cell_h


def glyph_path(
    rings: list[list[Point]], cx: float, cy: float, w: float, h: float, precision: int
) -> str:
    """One mark, scaled into a cell and centred, as path data."""
    x0, y0, x1, y1 = bbox(rings)
    bw, bh = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    scale = min(w * 0.92 / bw, h * 0.92 / bh)
    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    out = []
    for ring in rings:
        pts = " ".join(
            f"{(x - mx) * scale + cx:.{precision}f},{(y - my) * scale + cy:.{precision}f}"
            for x, y in ring
        )
        out.append("M" + pts.replace(" ", "L") + "Z")
    return "".join(out)


def build_layer(
    texts: list[str],
    marks: list[list[list[Point]]],
    box: tuple[float, float, float, float],
    precision: int,
) -> tuple[str, dict[str, int]]:
    """All three registers, as one path's worth of subpaths."""
    x0, y0, x1, y1 = box
    pad_x = (x1 - x0) * 0.06
    left, right = x0 + pad_x, x1 - pad_x
    width = right - left
    chars = sorted({ch for text in texts for ch in text if not ch.isspace()})
    chosen = alphabet(chars, marks)
    pieces: list[str] = []
    counts: dict[str, int] = {}
    height = y1 - y0
    for text, (top, bottom) in zip(texts, BANDS):
        band_top = y0 + height * top
        band_h = height * (bottom - top)
        lines, cell_w, cell_h = band_layout(text, width, band_h)
        counts["rows"] = counts.get("rows", 0) + len(lines)
        counts["marks"] = counts.get("marks", 0)
        for row, line in enumerate(lines):
            cy = band_top + (row + 0.5) * cell_h
            if row >= len(lines):
                break
            for col, ch in enumerate(line):
                if ch.isspace() or ch not in chosen:
                    continue
                cx = left + (col + 0.5) * cell_w
                if cx > right:
                    break
                pieces.append(
                    glyph_path(marks[chosen[ch]], cx, cy, cell_w, cell_h, precision)
                )
                counts["marks"] += 1
    return "".join(pieces), counts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Set API dialect payloads into the stone's own carved alphabet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--source", type=Path, required=True, help="the source tracing")
    p.add_argument(
        "--bands",
        type=Path,
        nargs="+",
        required=True,
        metavar="JSON",
        help="one to three payloads; each becomes one register, top to bottom",
    )
    p.add_argument("-o", "--out", type=Path, help="output layer (default: stdout)")
    p.add_argument(
        "--tolerance", type=float, default=0.15, help="flattening (default: 0.15)"
    )
    p.add_argument("--precision", type=int, default=1, help="decimals (default: 1)")
    p.add_argument("--stats", action="store_true", help="print counts and bytes")
    args = p.parse_args(argv)

    texts = [path.read_text() for path in args.bands[: len(BANDS)]]
    marks = load_marks(args.source, args.tolerance)
    if not marks:
        raise SystemExit(f"no carved marks found in {args.source}")

    # The registers span the stone itself, which is what the marking covers.
    _, vb = incisions.extract(args.source, drop_first=True, precision=0)
    vx, vy, vw, vh = (float(v) for v in vb.replace(",", " ").split()[-4:])
    ds, _ = incisions.extract(args.source, drop_first=False, precision=0)
    outline = S.flatten(S.parse_path(ds[0]), args.tolerance)
    box = bbox(outline) if outline else (vx, vy, vx + vw, vy + vh)

    body, counts = build_layer(texts, marks, box, args.precision)
    svg = incisions.build([body], vb)
    if args.stats:
        print(
            f"{len(texts)} registers, {counts.get('rows', 0)} rows, "
            f"{counts.get('marks', 0)} marks, {len(body) / 1e6:.2f} MB of path data"
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
