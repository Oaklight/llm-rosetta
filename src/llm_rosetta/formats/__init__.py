"""The format catalog: record type, records, and body-shape detection."""

from llm_rosetta.formats.catalog import BY_PAIR, FORMAT_SPECS, MODEL_TYPES
from llm_rosetta.formats.detect import DETECTORS, Detector, detect_format
from llm_rosetta.formats.spec import FormatSpec

__all__ = [
    "BY_PAIR",
    "DETECTORS",
    "FORMAT_SPECS",
    "MODEL_TYPES",
    "Detector",
    "FormatSpec",
    "detect_format",
]
