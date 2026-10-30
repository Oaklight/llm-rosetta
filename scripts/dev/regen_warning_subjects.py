"""Regenerate ``tests/converters/warning_subjects.json``.

Run from the repository root::

    python scripts/dev/regen_warning_subjects.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages/llm-rosetta/src"))
sys.path.insert(0, str(ROOT / "tests/converters"))


def main() -> int:
    """Write the baseline and say where it went."""
    from test_warning_subjects import build_warning_subjects

    import llm_rosetta

    print("llm_rosetta:", llm_rosetta.__file__)
    target = ROOT / "tests/converters/warning_subjects.json"
    payload = build_warning_subjects()
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("wrote", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
