"""SDK drift suite — wraps ``scripts/dev/scan_sdk_types.py``.

One case per provider.  A case fails when a field the baseline did not have
appears in a replica; fields that disappear are reported but are not a
failure.  A different SDK version means the baseline is not comparable, so
the case is skipped rather than reported as a regression.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from ..model import Probe, ProbeFailure, ProbeSkip

_BASELINE = Path(__file__).resolve().parents[1] / "baselines/sdk-types.json"
_SCANNER = Path(__file__).resolve().parents[3] / "scripts/dev/scan_sdk_types.py"


def _load_scanner() -> Any:
    spec = importlib.util.spec_from_file_location("scan_sdk_types", _SCANNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_baseline() -> dict[str, Any]:
    if not _BASELINE.exists():
        return {}
    return json.loads(_BASELINE.read_text())


def compare_provider(
    current: dict[str, Any],
    baseline_provider: dict[str, Any] | None,
) -> dict[str, list[str]]:
    """Return ``{type: [fields]}`` that are new relative to the baseline."""
    known = (baseline_provider or {}).get("drift", {})
    added: dict[str, list[str]] = {}
    for type_name, fields in current.get("drift", {}).items():
        new_fields = sorted(set(fields) - set(known.get(type_name, [])))
        if new_fields:
            added[type_name] = new_fields
    return added


class _DriftProbe:
    def __init__(
        self,
        name: str,
        data: dict[str, Any],
        baseline: dict[str, Any],
        sdk_versions: dict[str, str],
    ) -> None:
        self.name = name
        self._data = data
        self._baseline = baseline
        self._sdk_versions = sdk_versions

    def run(self) -> dict[str, Any]:
        if not self._baseline:
            raise ProbeSkip("no baseline committed yet")
        if self._baseline.get("sdk_versions") != self._sdk_versions:
            raise ProbeSkip(
                "SDK versions changed since the baseline; refresh it with "
                "`python -c 'from tools.conformance.suites.drift import "
                "write_baseline; write_baseline()'`"
            )

        baseline_provider = self._baseline.get("providers", {}).get(self.name)
        added = compare_provider(self._data, baseline_provider)
        if added:
            raise ProbeFailure(f"new drift: {json.dumps(added, sort_keys=True)}")

        return {
            "drift": self._data.get("drift", {}),
            "collisions": self._data.get("collisions", []),
            "new": added,
        }


class _BrokenProbe:
    name = "scan"

    def __init__(self, error: Exception) -> None:
        self._error = error

    def run(self) -> None:
        raise self._error


def build() -> tuple[list[Probe], dict[str, Any]]:
    try:
        scanner = _load_scanner()
        scan = scanner.build_report(scanner.select_providers([]))
    except Exception as exc:  # noqa: BLE001 - reported as an error case
        return [_BrokenProbe(exc)], {"sdk_versions": {}}

    baseline = _read_baseline()
    probes: list[Probe] = [
        _DriftProbe(key, data, baseline, scan["sdk_versions"])
        for key, data in scan["providers"].items()
    ]
    return probes, {"sdk_versions": scan["sdk_versions"]}


def write_baseline() -> Path:
    """Regenerate the committed baseline from the current SDKs."""
    scanner = _load_scanner()
    scan = scanner.build_report(scanner.select_providers([]))
    payload = {
        "schema_version": 1,
        "suite": "drift",
        "generated": scan["generated"],
        "sdk_versions": scan["sdk_versions"],
        "providers": {
            key: {
                "drift": data["drift"],
                "collisions": data["collisions"],
            }
            for key, data in scan["providers"].items()
        },
    }
    _BASELINE.parent.mkdir(parents=True, exist_ok=True)
    _BASELINE.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return _BASELINE
