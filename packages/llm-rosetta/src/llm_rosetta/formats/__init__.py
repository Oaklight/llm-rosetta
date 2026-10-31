"""The format catalog: record type, records, and body-shape detection."""

from llm_rosetta.formats.catalog import (
    BY_PAIR,
    FORMAT_SPECS,
    LEGACY_MODEL_TYPES,
    MODEL_TYPES,
    canonical_model_type,
    legacy_model_type,
)
from llm_rosetta.formats.detect import DETECTORS, Detector, detect_format
from llm_rosetta.formats.spec import FormatSpec

__all__ = [
    "BY_PAIR",
    "DETECTORS",
    "FORMAT_SPECS",
    "LEGACY_MODEL_TYPES",
    "MODEL_TYPES",
    "Detector",
    "FormatSpec",
    "canonical_model_type",
    "detect_format",
    "legacy_model_type",
]
