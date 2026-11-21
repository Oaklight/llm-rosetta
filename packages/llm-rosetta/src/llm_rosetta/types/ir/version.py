"""The IR data model's compatibility level (#74)."""

from __future__ import annotations

#: The IR data model's compatibility level.  It rises when a change to the IR
#: types means an IR value written at the old level cannot be read at the new
#: one; an added optional field does not raise it.  Re-exported from
#: :mod:`llm_rosetta.types.ir` and :mod:`llm_rosetta.contracts`.
IR_VERSION = 1

__all__ = ["IR_VERSION"]
