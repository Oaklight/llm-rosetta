"""The model-type vocabulary: categories and their wire formats.

Derived from :mod:`llm_rosetta.formats`, the catalog that owns every
per-format fact.  This module stays a leaf so the shim loader can validate a
``(model_type, format)`` pair while ``llm_rosetta.shims`` is still
initialising, without importing the converter package.
"""

from __future__ import annotations

from llm_rosetta.formats import FORMAT_SPECS as _SPECS
from llm_rosetta.formats import MODEL_TYPES

__all__ = ["FORMATS_BY_TYPE", "MODEL_TYPES"]

#: Formats within each category, in catalog (display) order.
FORMATS_BY_TYPE: dict[str, tuple[str, ...]] = {
    model_type: tuple(spec.format for spec in _SPECS if spec.model_type == model_type)
    for model_type in MODEL_TYPES
}
