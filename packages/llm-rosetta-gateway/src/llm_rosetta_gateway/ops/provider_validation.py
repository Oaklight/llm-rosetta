"""Audited validation and recovery for malformed provider configurations."""

from __future__ import annotations

from typing import Any, ClassVar

from llm_rosetta.observability.ops_log import (
    EVENT_PROVIDER_VALIDATION,
    SEVERITY_WARNING,
    SOURCE_CONFIG,
)
from llm_rosetta.shims import get_shim

from ..shims import find_source_package, legacy_shim_migration_advice

from .base import OpsBase, OpsContext


class OpsProviderValidation(OpsBase):
    """Disable independently invalid providers before building the gateway.

    Provider-local configuration errors are recoverable: the affected provider
    is disabled and recorded, while unrelated providers remain available.
    Global model/routing errors are intentionally left to ``GatewayConfig``.
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

        # Scan first and apply mutations afterwards.  This keeps the decision
        # independent of iteration order and makes one persistence write enough.
        issues: list[tuple[str, ProviderIssue]] = []
        for name, cfg in providers.items():
            provider_name = str(name)
            if not isinstance(cfg, dict):
                issues.append(
                    (
                        provider_name,
                        ProviderIssue(
                            f"Provider {provider_name!r} must be an object, got "
                            f"{type(cfg).__name__}",
                            "invalid_provider_config",
                        ),
                    )
                )
                continue
            if cfg.get("enabled", True) is False:
                continue
            issue = _validate_provider(
                provider_name, cfg, self._report, _global_proxy(self._raw)
            )
            if issue is not None:
                issues.append((provider_name, issue))

        for name, issue in issues:
            cfg = providers[name]
            if not isinstance(cfg, dict):
                cfg = {}
                providers[name] = cfg
            cfg["enabled"] = False
            marker = {
                "reason": issue.reason,
                "category": issue.category,
            }
            if issue.shim:
                marker["shim"] = issue.shim
            cfg["_auto_disabled"] = marker
            item = {
                "provider": name,
                "reason": issue.reason,
                "category": issue.category,
            }
            if issue.shim:
                item["shim"] = issue.shim
            disabled.append(item)

        if disabled and self._persist and self._ctx.config_io and self._ctx.config_path:
            # Re-read the raw document so `${ENV_VAR}` placeholders are not
            # replaced by values from the runtime-resolved config.
            disk_raw = self._ctx.config_io.load_raw(self._ctx.config_path)
            disk_providers = disk_raw.setdefault("providers", {})
            for item in disabled:
                disk_cfg = disk_providers.get(item["provider"])
                if isinstance(disk_cfg, dict):
                    disk_cfg["enabled"] = False
                    marker = {
                        "reason": item["reason"],
                        "category": item["category"],
                    }
                    if item.get("shim"):
                        marker["shim"] = item["shim"]
                    disk_cfg["_auto_disabled"] = marker
            self._ctx.config_io.save(self._ctx.config_path, disk_raw)
        return {"disabled": disabled}

    def _message(self, result: Any) -> str:
        disabled = (result or {}).get("disabled", [])
        if not disabled:
            return "Provider configuration validation passed"
        return "Automatically disabled invalid providers: " + ", ".join(
            item["provider"] for item in disabled
        )

    def _details(self, result: Any) -> dict[str, Any]:
        return {"disabled": (result or {}).get("disabled", [])}


class ProviderIssue:
    """Small internal value object describing one provider-local failure."""

    __slots__ = ("reason", "category", "shim")

    def __init__(self, reason: str, category: str, shim: str | None = None) -> None:
        self.reason = reason
        self.category = category
        self.shim = shim


def _global_proxy(raw: dict[str, Any]) -> str | None:
    server = raw.get("server")
    return server.get("proxy") if isinstance(server, dict) else None


def _validate_provider(
    name: str, cfg: dict[str, Any], report: Any, global_proxy: str | None
) -> ProviderIssue | None:
    """Validate only fields needed to construct one ``ProviderInfo``."""
    source = cfg.get("shim") or cfg.get("type") or name
    shim_name = source if isinstance(source, str) and source else None

    dependency = _missing_dependency(cfg, report)
    if dependency is not None:
        shim, reason = dependency
        return ProviderIssue(reason, "missing_external_shim", shim)
    if "shim" in cfg and isinstance(cfg["shim"], str):
        if get_shim(cfg["shim"]) is None:
            reason = (
                f"Provider {name!r} references unknown shim {cfg['shim']!r}; "
                "choose a registered shim or declare its external source."
            )
            # A spelling the catalogue records gets the canonical name and the
            # preset here as well: the ``shim:`` field does not go through the
            # base_url path that carries the same sentence.
            reason += legacy_shim_migration_advice(cfg["shim"])
            return ProviderIssue(reason, "unknown_shim", cfg["shim"])

    try:
        # Reuse GatewayConfig's identity resolver so validation cannot drift
        # from the runtime resolution rules.  The one-entry mapping is local
        # and the resolver only canonicalizes that mapping in place.
        from ..config.resolve import resolve_provider_identities
        from ..providers import build_provider_info

        identity_map, shim_names = resolve_provider_identities({name: dict(cfg)})
        identity = identity_map[name]
        build_provider_info(
            identity.format,
            cfg,
            global_proxy=global_proxy,
            shim_name=shim_names.get(name),
        )
    except ValueError as exc:
        reason = str(exc)
        category = (
            "missing_base_url" if "no base_url" in reason else "invalid_provider_info"
        )
        reason += legacy_shim_migration_advice(shim_name)
        return ProviderIssue(reason, category, shim_name)
    except (TypeError, KeyError) as exc:
        return ProviderIssue(
            f"Provider {name!r} has malformed configuration: {exc}",
            "invalid_provider_config",
            shim_name,
        )
    return None


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
