#!/usr/bin/env python3
"""Regenerate the emitted-field ledger for ``test_leg_agreement.py``.

The ledger is a behaviour baseline: a diff means a response writer started or
stopped emitting a field.  Review the diff before committing the regenerated
file; a dropped field is usually a bug, not a ledger update.

Usage:
    python scripts/dev/regen_leg_agreement_fields.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "tests/converters/test_leg_agreement.py"
TARGET = ROOT / "tests/converters/leg_agreement_fields.json"

# Resolve the package from this checkout, not from whichever editable install
# happens to be first on sys.path.  A stale install would write a ledger from
# the wrong code and the diff would look like a writer change.
sys.path.insert(0, str(ROOT / "packages/llm-rosetta/src"))


def main() -> None:
    spec = importlib.util.spec_from_file_location("test_leg_agreement", TEST)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    package = importlib.import_module("llm_rosetta")
    print(f"llm_rosetta: {package.__file__}")

    ledger = module.build_field_ledger()
    TARGET.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(TARGET)


if __name__ == "__main__":
    main()
