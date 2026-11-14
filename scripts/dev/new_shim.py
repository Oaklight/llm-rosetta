#!/usr/bin/env python3
"""Repository wrapper around :mod:`llm_rosetta.shims.scaffold`.

The command is installed as ``llm-rosetta-gateway shims new``; this copy exists
so a checkout can scaffold into the built-in tree — which is not writable in an
installed package — without remembering that path.  Keep it a wrapper: the flags
and the prompts live in the module, so the two entry points cannot drift.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILTIN_ROOT = REPO_ROOT / "packages/llm-rosetta/src/llm_rosetta/shims/providers"


def main(argv: list[str] | None = None) -> int:
    """Run the scaffolder against the built-in tree by default."""
    from llm_rosetta.shims.scaffold import main as scaffold_main

    arguments = sys.argv[1:] if argv is None else argv
    return scaffold_main(arguments, default_root=str(BUILTIN_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
