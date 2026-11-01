#!/usr/bin/env python3
"""Copy the docs changelogs into the gateway package for a local build.

The release workflows do the same against the docs branches, so a wheel from
PyPI carries the changelog the admin panel shows after an update.  A source
checkout has no copy, and the panel's "what changed" dialog stays closed
without one — run this before testing it.

Usage:
    python scripts/dev/sync_changelog.py
    python scripts/dev/sync_changelog.py --docs-en ../docs_en --docs-zh ../docs_zh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE = (
    Path(__file__).resolve().parents[2]
    / "packages"
    / "llm-rosetta-gateway"
    / "src"
    / "llm_rosetta_gateway"
)


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-en", type=Path, default=root / "docs_en")
    parser.add_argument("--docs-zh", type=Path, default=root / "docs_zh")
    args = parser.parse_args(argv)

    sources = {"en": args.docs_en, "zh": args.docs_zh}
    written: list[tuple[Path, int]] = []
    for lang, worktree in sources.items():
        source = worktree / "docs" / "changelog.md"
        if not source.is_file():
            print(
                f"missing {source} — check the {worktree.name} worktree out first",
                file=sys.stderr,
            )
            return 1
        target = PACKAGE / f"CHANGELOG.{lang}.md"
        text = source.read_text(encoding="utf-8")
        target.write_text(text, encoding="utf-8")
        written.append((target, len(text.encode("utf-8"))))

    for target, size in written:
        print(f"{target.relative_to(root)}  {size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
