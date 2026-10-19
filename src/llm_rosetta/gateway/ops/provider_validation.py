"""Audited validation and recovery for providers backed by external shims."""

from __future__ import annotations

from typing import Any, ClassVar

from llm_rosetta.observability.ops_log import (
    EVENT_PROVIDER_VALIDATION,
    SEVERITY_WARNING,
    SOURCE_CONFIG,
)
from llm_rosetta.shims import find_source_package, get_shim

from .base import OpsBase, OpsContext


class OpsProviderValidation(OpsBase):
    """Disable enabled providers whose standalone shim dependency is missing.

    This operation deliberately runs before ``GatewayConfig`` builds provider
    identities.  A missing external shim must not be inferred as an unrelated
    endpoint type and must not take down healthy providers.
    """

    event_type: ClassVar[str] = EVENT_PROVIDER_VALIDATION
    severity: ClassVar[str] = SEVERITY_WARNING
    source: ClassVar[str] = SOURCE_CONFIG

    __slots__ = ("_raw", "_report", "_persist")

    def __init__(
        self,
        ctx: OpsContext,
        raw: dict[str, Any],
        *,
        report: Any = None,
        persist: bool = False,
    ) -> None:
        super().__init__(ctx)
        self._raw = raw
        self._report = report
        self._persist = persist

    async def _run(self) -> dict[str, Any]:
        providers = self._raw.get("providers", {})
        disabled: list[dict[str, str]] = []
        if not isinstance(providers, dict):
            return {"disabled": disabled}

        for name, cfg in providers.items():
            if not isinstance(cfg, dict) or cfg.get("enabled", True) is False:
                continue
            dependency = _missing_dependency(cfg, self._report)
            if dependency is None:
                continue
            shim_name, reason = dependency
            cfg["enabled"] = False
            cfg["_auto_disabled"] = {
                "reason": reason,
                "shim": shim_name,
            }
            disabled.append(
                {"provider": str(name), "shim": shim_name, "reason": reason}
            )

        if disabled and self._persist and self._ctx.config_io and self._ctx.config_path:
            # Re-read the raw document so `${ENV_VAR}` placeholders are not
            # replaced by values from the runtime-resolved config.
            disk_raw = self._ctx.config_io.load_raw(self._ctx.config_path)
            disk_providers = disk_raw.setdefault("providers", {})
            for item in disabled:
                disk_cfg = disk_providers.get(item["provider"])
                if isinstance(disk_cfg, dict):
                    disk_cfg["enabled"] = False
                    disk_cfg["_auto_disabled"] = {
                        "reason": item["reason"],
                        "shim": item["shim"],
                    }
            self._ctx.config_io.save(self._ctx.config_path, disk_raw)
        return {"disabled": disabled}

    def _message(self, result: Any) -> str:
        disabled = (result or {}).get("disabled", [])
        if not disabled:
            return "External provider shim validation passed"
        return (
            "Automatically disabled providers with unavailable external shims: "
            + ", ".join(item["provider"] for item in disabled)
        )

    def _details(self, result: Any) -> dict[str, Any]:
        return {"disabled": (result or {}).get("disabled", [])}


def _missing_dependency(cfg: dict[str, Any], report: Any) -> tuple[str, str] | None:
    """Return a missing standalone shim dependency, if this config needs one."""
    source = cfg.get("shim") or cfg.get("type")
    if not isinstance(source, str) or not source:
        return None
    package = find_source_package(source)
    if package is None or get_shim(source) is not None:
        return None
    failures = []
    if report is not None:
        failures = [
            f"{status.location}: {status.error}" for status in report.failures()
        ]
    suffix = f"; source failures: {'; '.join(failures)}" if failures else ""
    reason = (
        f"required external shim {source!r} is not registered; declare "
        f"shims.presets: [{package.key!r}] or its remote source{suffix}"
    )
    return source, reason
