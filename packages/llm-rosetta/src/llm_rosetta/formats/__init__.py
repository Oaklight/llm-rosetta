"""The format catalog: record type, records, and body-shape detection."""

from llm_rosetta.formats.batch import (
    BATCH_BY_PAIR,
    BATCH_SPECS,
    BY_DIALECT,
    DECLARABLE_DIALECTS,
    BatchLimits,
    BatchSpec,
    batch_spec_for,
    batch_spec_for_dialect,
    batch_specs_for_dialect,
)
from llm_rosetta.formats.catalog import (
    BY_PAIR,
    FORMAT_SPECS,
    LEGACY_MODEL_TYPES,
    MODEL_TYPES,
    canonical_model_type,
    legacy_model_type,
    spec_for,
)
from llm_rosetta.formats.detect import DETECTORS, Detector, detect_format
from llm_rosetta.formats.spec import FormatSpec

__all__ = [
    "BATCH_BY_PAIR",
    "BATCH_SPECS",
    "BY_DIALECT",
    "BY_PAIR",
    "DECLARABLE_DIALECTS",
    "BatchLimits",
    "BatchSpec",
    "DETECTORS",
    "FORMAT_SPECS",
    "LEGACY_MODEL_TYPES",
    "MODEL_TYPES",
    "Detector",
    "FormatSpec",
    "batch_spec_for",
    "batch_spec_for_dialect",
    "batch_specs_for_dialect",
    "canonical_model_type",
    "detect_format",
    "legacy_model_type",
    "spec_for",
]
