#!/usr/bin/env python3
"""Repository wrapper around :mod:`llm_rosetta.authoring.scaffold`.

The gateway's ``shims new`` is gone (``llm-rosetta ext new`` creates an extension
folder).  This command is how a checkout scaffolds into the built-in tree, which
is not writable in an installed package, without remembering that path.  Keep it
a wrapper: the flags and the prompts live in the module.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILTIN_ROOT = REPO_ROOT / "packages/llm-rosetta/src/llm_rosetta/shims/builtin"


def main(argv: list[str] | None = None) -> int:
    """Run the scaffolder against the built-in tree by default."""
    from llm_rosetta.authoring.scaffold import main as scaffold_main

    arguments = sys.argv[1:] if argv is None else argv
    return scaffold_main(arguments, default_root=str(BUILTIN_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
