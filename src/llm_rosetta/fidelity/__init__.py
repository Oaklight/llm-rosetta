"""Round-trip fidelity checking and persistent baselines.

:mod:`.checker` compares an original body with its IR round-trip;
:mod:`.baseline` stores those differences so converter regressions can be
compared across runs.  Both were top-level modules before; the package keeps
the ``llm_rosetta.fidelity`` import path.
"""

from __future__ import annotations

from llm_rosetta.provider_names import canonical_provider_name  # noqa: F401 (re-exported)

from .baseline import (
    BASELINE_FORMAT_VERSION,
    BaselineComparisonReport,
    BaselineEntry,
    BaselineFormatError,
    FidelityBaseline,
    SeverityChange,
    canonical_json,
    check_fidelity_against_baseline,
    compute_fingerprint,
    make_baseline_key,
    normalize_entries,
)
from .checker import (
    DEFAULT_SEVERITY_BY_KIND,
    FidelityChecker,
    FidelityDiff,
    FidelityMode,
    FidelitySeverity,
    severity_for_kind,
)

__all__ = [
    "BASELINE_FORMAT_VERSION",
    "BaselineComparisonReport",
    "BaselineEntry",
    "BaselineFormatError",
    "DEFAULT_SEVERITY_BY_KIND",
    "FidelityBaseline",
    "FidelityChecker",
    "FidelityDiff",
    "FidelityMode",
    "FidelitySeverity",
    "SeverityChange",
    "canonical_json",
    "check_fidelity_against_baseline",
    "compute_fingerprint",
    "make_baseline_key",
    "normalize_entries",
    "severity_for_kind",
]
