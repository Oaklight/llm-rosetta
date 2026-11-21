"""Audited validation and recovery for malformed provider configurations."""

from __future__ import annotations

from typing import Any, ClassVar

from llm_rosetta_gateway.observability.ops_log import (
    EVENT_PROVIDER_VALIDATION,
    SEVERITY_WARNING,
    SOURCE_CONFIG,
    OpsLogEntry,
)
from llm_rosetta.api import get_shim, retired_names

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
        recovered: list[str] = []
        if not isinstance(providers, dict):
            return {"disabled": disabled, "recovered": recovered}

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
            # An entry the gateway disabled itself is revalidated on every
            # load: the marker is our state, not the operator's switch.  An
            # entry without one was turned off by hand and is left alone.
            auto_disabled = bool(cfg.get("_auto_disabled"))
            if cfg.get("enabled", True) is False and not auto_disabled:
                continue
            issue = _validate_provider(
                provider_name,
                cfg,
                self._report,
                _global_proxy(self._raw),
                disabled=_disabled_shims(self._raw),
            )
            if issue is not None:
                issues.append((provider_name, issue))
            elif auto_disabled:
                recovered.append(provider_name)

        for name in recovered:
            cfg = providers[name]
            cfg.pop("_auto_disabled", None)
            cfg["enabled"] = True

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

        self._persist_state(disabled, recovered)
        return {"disabled": disabled, "recovered": recovered}

    def _persist_state(
        self, disabled: list[dict[str, str]], recovered: list[str]
    ) -> None:
        """Write the markers and the recoveries into the config document.

        Only what differs is written.  A lane whose fault has not gone away
        re-enters ``disabled`` on every load, and rewriting an unchanged
        document each time moves its mtime and drops its comments for a
        change that did not happen.
        """
        if not (disabled or recovered):
            return
        if not (self._persist and self._ctx.config_io and self._ctx.config_path):
            return
        # Re-read the raw document so `${ENV_VAR}` placeholders are not
        # replaced by values from the runtime-resolved config.
        disk_raw = self._ctx.config_io.load_raw(self._ctx.config_path)
        disk_providers = disk_raw.setdefault("providers", {})
        changed = False
        for item in disabled:
            disk_cfg = disk_providers.get(item["provider"])
            if not isinstance(disk_cfg, dict):
                continue
            marker = {
                "reason": item["reason"],
                "category": item["category"],
            }
            if item.get("shim"):
                marker["shim"] = item["shim"]
            if disk_cfg.get("enabled") is not False or (
                disk_cfg.get("_auto_disabled") != marker
            ):
                disk_cfg["enabled"] = False
                disk_cfg["_auto_disabled"] = marker
                changed = True
        for name in recovered:
            disk_cfg = disk_providers.get(name)
            if isinstance(disk_cfg, dict) and (
                disk_cfg.get("enabled") is not True or "_auto_disabled" in disk_cfg
            ):
                disk_cfg.pop("_auto_disabled", None)
                disk_cfg["enabled"] = True
                changed = True
        if changed:
            self._ctx.config_io.save(self._ctx.config_path, disk_raw)

    def _message(self, result: Any) -> str:
        result = result or {}
        disabled = result.get("disabled", [])
        recovered = result.get("recovered", [])
        parts = []
        if disabled:
            parts.append(
                "Automatically disabled invalid providers: "
                + ", ".join(item["provider"] for item in disabled)
            )
        if recovered:
            parts.append(
                "Recovered automatically disabled providers: " + ", ".join(recovered)
            )
        return "; ".join(parts) if parts else "Provider configuration validation passed"

    def _details(self, result: Any) -> dict[str, Any]:
        return {
            "disabled": (result or {}).get("disabled", []),
            "recovered": (result or {}).get("recovered", []),
        }


def startup_validation_entry(result: Any) -> OpsLogEntry | None:
    """The audit row for a load-time validation run, or ``None`` if it changed nothing.

    The startup path runs before the ops log exists, so the run itself cannot
    record (issue #1386).  It leaves its result on the config instead, and the
    gateway writes this row once the log is open — the first disable, usually
    the decisive one, is then not the one missing from the audit trail.

    Args:
        result: ``{"disabled": [...], "recovered": [...]}`` as the op returned it.

    Returns:
        An entry to add, or ``None`` when the run disabled and recovered nothing.
    """
    if not result or not (result.get("disabled") or result.get("recovered")):
        return None
    op = OpsProviderValidation(OpsContext(), {})
    return OpsLogEntry.create(
        event_type=OpsProviderValidation.event_type,
        severity=OpsProviderValidation.severity,
        message=op._message(result),
        details=op._details(result),
        source=OpsProviderValidation.source,
    )


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


def _disabled_shims(raw: dict[str, Any]) -> frozenset[str]:
    """The names ``shims.disabled`` keeps out of the registry."""
    section = raw.get("shims")
    names = section.get("disabled") if isinstance(section, dict) else None
    if not isinstance(names, list):
        return frozenset()
    return frozenset(n for n in names if isinstance(n, str))


def _validate_provider(
    name: str,
    cfg: dict[str, Any],
    report: Any,
    global_proxy: str | None,
    *,
    disabled: frozenset[str] = frozenset(),
) -> ProviderIssue | None:
    """Validate only fields needed to construct one ``ProviderInfo``."""
    source = cfg.get("shim") or cfg.get("type") or name
    shim_name = source if isinstance(source, str) and source else None

    dependency = _missing_dependency(cfg, report, disabled)
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


def _missing_dependency(
    cfg: dict[str, Any], report: Any, disabled: frozenset[str] = frozenset()
) -> tuple[str, str] | None:
    """Return a missing standalone shim dependency, if this config needs one.

    The reason names the actual cause, checked in this order: the operator's
    own ``shims.disabled`` switch, then what the load report says about the
    vendor (installed, skipped with a reason, or failed), and only then the
    source failures.
    """
    source = cfg.get("shim") or cfg.get("type")
    if not isinstance(source, str) or not source:
        return None
    package = find_source_package(source)
    moved = retired_names.moved_vendor(source)
    if (package is None and moved is None) or get_shim(source) is not None:
        return None
    if source in disabled:
        return source, (
            f"shim {source!r} is switched off by shims.disabled; the provider "
            "comes back when the shim is removed from that list"
        )
    if moved is not None:
        return source, _moved_reason(source, moved, report)
    assert package is not None
    reason = (
        f"required external shim {source!r} is not registered; declare "
        f"shims.presets: [{package.key!r}] or its remote source"
        f"{_failure_suffix(report)}"
    )
    return source, reason


def _failure_suffix(report: Any) -> str:
    """``; source failures: ...`` from the load report, or ``""``."""
    if report is None:
        return ""
    failures = [f"{status.location}: {status.error}" for status in report.failures()]
    return f"; source failures: {'; '.join(failures)}" if failures else ""


def _moved_reason(source: str, moved: Any, report: Any) -> str:
    """Why a moved vendor's shim is not registered, from its install status."""
    head = (
        f"shim {source!r} is not registered: its vendor {moved.vendor!r} moved "
        f"from core to {moved.package} in {moved.moved_in}"
    )
    statuses = [
        status
        for status in (report.sources if report is not None else ())
        if getattr(status, "vendor", None) == moved.vendor
    ]
    skipped = next((s.skipped for s in statuses if s.skipped), None)
    if skipped and skipped.startswith("excluded by shims.install"):
        # The operator's own entry keeps it out; nothing failed upstream.
        return f"{head}; it is {skipped}"
    if skipped:
        return f"{head}, and the gateway did not install it: {skipped}"
    if any(s.ok for s in statuses):
        return (
            f"{head}; the installed vendor does not register this shim (a "
            "shims include/exclude selection, or a name its release does not list)"
        )
    return (
        f"{head}, and the gateway's automatic install of it did not succeed "
        f"(the provider comes back once the source answers){_failure_suffix(report)}"
    )
