"""Leaf quota sidecar: imports the probe contract, declares no probe."""

from __future__ import annotations

from llm_rosetta.contracts.quota import QuotaProbe

PROBES: tuple[QuotaProbe, ...] = ()
