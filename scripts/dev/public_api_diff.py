#!/usr/bin/env python3
"""Compare the public API surface of the latest stable release with this tree.

A tree's surface is the set of names each public module exports plus the
parameter list of every callable or class it exports.  A module that declares
``__all__`` uses it; otherwise the public names the module defines itself are
used, and the report marks that surface as *inferred*.

Scope limits: only module-level names and the constructor signature of an
exported class are compared — methods on a class are not.  The module list is
deliberately explicit, so packages that are not listed (for example
``llm_rosetta.gateway.admin.*``) are out of scope.

The two trees are imported in separate subprocesses, so module state cannot
leak between them.

A change is **breaking** when a public name disappears, an export changes kind,
a parameter disappears, a new parameter is required, or a parameter changes
kind or becomes required.  Additions, optional parameters and relaxed
requirements are reported for information only.

Every breaking finding must be acknowledged in
``scripts/dev/public_api_allowlist.json``; otherwise the check exits 1.  A
breaking change is either a mistake (keep the old name as a deprecated alias
until its removal window) or an intentional removal that has to be written
down.

Usage::

    python scripts/dev/public_api_diff.py                 # latest tag -> HEAD
    python scripts/dev/public_api_diff.py --base-ref v0.13.0
    python scripts/dev/public_api_diff.py --base /tmp/stable
    python scripts/dev/public_api_diff.py --json
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALLOWLIST = Path(__file__).with_name("public_api_allowlist.json")

#: Only a bare ``vMAJOR.MINOR.PATCH`` tag is a stable release.  Anchored with
#: ``fullmatch`` so ``v0.14.0.dev4``, ``v0.14.0rc1`` and ``v1.0.0-beta`` never
#: qualify.
STABLE_TAG_RE = re.compile(r"v\d+\.\d+\.\d+")

#: Modules whose declared or inferred surface is part of the supported public
#: API.  A module without ``__all__`` falls back to the names it defines, and
#: the report labels that surface as inferred.  The list is deliberately
#: explicit: importing every module under the package would execute the gateway
#: CLI's argv parsing.
PUBLIC_MODULES = (
    "llm_rosetta",
    "llm_rosetta.converters",
    "llm_rosetta.converters.base",
    "llm_rosetta.converters.embedding",
    "llm_rosetta.converters.rerank",
    "llm_rosetta.converters.decision",
    "llm_rosetta.converters.registry",
    "llm_rosetta.converters.llm.chat_completions",
    "llm_rosetta.converters.llm.messages",
    "llm_rosetta.converters.llm.open_responses",
    "llm_rosetta.converters.llm.google_generate",
    "llm_rosetta.converters.llm.google_interactions",
    # 0.13.0 package paths, kept as deprecated aliases until 0.16.0.
    "llm_rosetta.converters.openai_chat",
    "llm_rosetta.converters.anthropic",
    "llm_rosetta.converters.openai_responses",
    "llm_rosetta.converters.google_generate",
    "llm_rosetta.converters.google_genai",
    "llm_rosetta.types",
    "llm_rosetta.types.ir",
    "llm_rosetta.shims",
    "llm_rosetta.auto_detect",
    "llm_rosetta.pipeline",
    "llm_rosetta.provider_names",
    "llm_rosetta.tool_ops",
    "llm_rosetta.observability",
    "llm_rosetta.observability.usage",
    "llm_rosetta.observability.metrics",
    "llm_rosetta.observability.persistence",
    "llm_rosetta.observability.request_log",
    "llm_rosetta.observability.ops_log",
    "llm_rosetta.gateway",
)


# ---------------------------------------------------------------------------
# Surface dump (runs in a subprocess against one tree)
# ---------------------------------------------------------------------------


def _describe(obj: object) -> dict:
    if inspect.isclass(obj):
        kind = "class"
    elif callable(obj):
        kind = "function"
    else:
        return {"kind": "other", "params": None}
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return {"kind": kind, "params": None}
    params = []
    for param in signature.parameters.values():
        params.append(
            {
                "name": param.name,
                "kind": str(param.kind),
                "required": (
                    param.default is inspect.Parameter.empty
                    and param.kind
                    not in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    )
                ),
            }
        )
    return {"kind": kind, "params": params}


def dump_surface(root: Path) -> dict:
    src = (root / "src").resolve()
    if not (src / "llm_rosetta").is_dir():
        raise SystemExit(f"{root} does not look like an llm-rosetta source tree")
    sys.path.insert(0, str(src))

    import llm_rosetta  # noqa: PLC0415

    origin = Path(llm_rosetta.__file__).resolve()
    if not origin.is_relative_to(src / "llm_rosetta"):
        raise SystemExit(f"imported llm_rosetta from {origin}, expected under {src}")

    result: dict[str, dict] = {}
    for module_name in PUBLIC_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            result[module_name] = {"status": "absent", "error": str(exc)}
            continue
        except Exception as exc:  # noqa: BLE001 - reported to the user
            result[module_name] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            continue
        declared = getattr(module, "__all__", None)
        if declared is None:
            # No ``__all__`` does not mean no surface: fall back to the public
            # names the module defines itself.  The ``__module__`` check keeps
            # imported callables out (they belong to their defining module).
            names = sorted(
                name
                for name, value in vars(module).items()
                if not name.startswith("_")
                and getattr(value, "__module__", None) == module_name
            )
            source = "inferred"
        else:
            names = declared
            source = "declared"
        result[module_name] = {
            "status": "ok",
            "source": source,
            "exports": {name: _describe(getattr(module, name, None)) for name in names},
        }
    return result


def dump_in_subprocess(root: Path) -> dict:
    """Import *root* in a clean process so two trees never share module state."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str((root / "src").resolve())
    with tempfile.TemporaryDirectory(prefix="llm-rosetta-api-cwd-") as cwd:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--dump", str(root)],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
        )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "unknown error"
        raise SystemExit(f"surface dump failed for {root}:\n{detail}")
    return json.loads(proc.stdout)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _finding(
    severity: str,
    module: str,
    name: str,
    change: str,
    param: str | None = None,
    detail: str = "",
) -> dict:
    parts = [module, name, change]
    if param:
        parts.append(param)
    return {
        "severity": severity,
        "module": module,
        "name": name,
        "change": change,
        "param": param,
        "detail": detail,
        "key": "::".join(parts),
    }


def _compare_export(module: str, name: str, base: dict, head: dict) -> list[dict]:
    if base["kind"] != head["kind"]:
        return [
            _finding(
                "breaking",
                module,
                name,
                "kind-changed",
                detail=f"{base['kind']} -> {head['kind']}",
            )
        ]
    base_params = base.get("params")
    head_params = head.get("params")
    if base_params is None or head_params is None:
        return []

    findings = []
    # A leading underscore marks implementation state, not API: stream-context
    # dataclasses are exported but their private fields may come and go.
    base_by_name = {
        param["name"]: param
        for param in base_params
        if not param["name"].startswith("_")
    }
    head_by_name = {
        param["name"]: param
        for param in head_params
        if not param["name"].startswith("_")
    }

    for param in sorted(set(base_by_name) - set(head_by_name)):
        findings.append(_finding("breaking", module, name, "removed-parameter", param))
    for param in sorted(set(head_by_name) - set(base_by_name)):
        spec = head_by_name[param]
        severity = "breaking" if spec["required"] else "info"
        change = (
            "added-required-parameter"
            if spec["required"]
            else "added-optional-parameter"
        )
        findings.append(_finding(severity, module, name, change, param))
    for param in sorted(set(base_by_name) & set(head_by_name)):
        base_spec = base_by_name[param]
        head_spec = head_by_name[param]
        if base_spec["kind"] != head_spec["kind"]:
            findings.append(
                _finding(
                    "breaking",
                    module,
                    name,
                    "parameter-kind-changed",
                    param,
                    detail=f"{base_spec['kind']} -> {head_spec['kind']}",
                )
            )
        elif not base_spec["required"] and head_spec["required"]:
            findings.append(
                _finding("breaking", module, name, "parameter-became-required", param)
            )
        elif base_spec["required"] and not head_spec["required"]:
            findings.append(
                _finding("info", module, name, "parameter-became-optional", param)
            )
    return findings


def compare(base: dict, head: dict) -> list[dict]:
    findings: list[dict] = []
    for module in PUBLIC_MODULES:
        base_entry = base.get(module, {"status": "absent"})
        head_entry = head.get(module, {"status": "absent"})
        base_status = base_entry["status"]
        head_status = head_entry["status"]

        if "error" in (base_status, head_status):
            findings.append(
                _finding(
                    "error",
                    module,
                    "<module>",
                    "import-failed",
                    detail=base_entry.get("error") or head_entry.get("error", ""),
                )
            )
            continue
        if base_status == "absent" and head_status == "absent":
            continue
        if base_status == "absent":
            findings.append(_finding("info", module, "<module>", "module-added"))
            continue
        if head_status == "absent":
            for name in base_entry.get("exports", {}):
                findings.append(
                    _finding("breaking", module, name, "module-removed", detail=module)
                )
            continue
        if base_status != head_status:
            continue

        if "inferred" in (base_entry.get("source"), head_entry.get("source")):
            findings.append(_finding("info", module, "<module>", "surface-inferred"))

        base_exports = base_entry["exports"]
        head_exports = head_entry["exports"]
        for name in sorted(set(base_exports) - set(head_exports)):
            findings.append(_finding("breaking", module, name, "removed"))
        for name in sorted(set(head_exports) - set(base_exports)):
            findings.append(_finding("info", module, name, "added"))
        for name in sorted(set(base_exports) & set(head_exports)):
            findings.extend(
                _compare_export(module, name, base_exports[name], head_exports[name])
            )
    return findings


# ---------------------------------------------------------------------------
# Allowlist, reporting, entry point
# ---------------------------------------------------------------------------


def load_allowlist(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        entry["key"]: entry.get("reason", "") for entry in data.get("acknowledged", [])
    }


def select_stable_tag(tags: Iterable[str]) -> str | None:
    """Return the first stable ``vX.Y.Z`` tag in *tags*, or ``None``.

    Pre-release spellings (``v0.14.0.dev4``, ``v0.14.0rc1``, ``v1.0.0-beta``)
    are not stable and never match: :data:`STABLE_TAG_RE` is anchored with
    ``fullmatch``, so the suffix has to be absent.
    """
    for value in tags:
        tag = value.strip()
        if STABLE_TAG_RE.fullmatch(tag):
            return tag
    return None


def latest_stable_tag() -> str:
    proc = subprocess.run(
        ["git", "tag", "--sort=-v:refname"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    tag = select_stable_tag(proc.stdout.splitlines())
    if tag is None:
        raise SystemExit("no stable vX.Y.Z tag found")
    return tag


def _describe_finding(finding: dict) -> str:
    where = f"{finding['module']}.{finding['name']}"
    change = finding["change"].replace("-", " ")
    if finding["param"]:
        change += f" {finding['param']!r}"
    if finding["detail"]:
        change += f" ({finding['detail']})"
    return f"{where}: {change}"


def _classify_findings(
    findings: list[dict], allowlist: dict[str, str]
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    """Split findings into breaking, acknowledged, errors, inferred and info."""
    for finding in findings:
        finding["acknowledged"] = finding["key"] in allowlist

    breaking: list[dict] = []
    acknowledged: list[dict] = []
    reasonless: list[dict] = []
    for finding in findings:
        if finding["severity"] != "breaking":
            continue
        if not finding["acknowledged"]:
            breaking.append(finding)
        elif allowlist[finding["key"]].strip():
            acknowledged.append(finding)
        else:
            # An entry without a written reason is not an acknowledgement: the
            # point of the file is the justification, so a blank reason fails.
            finding["acknowledged"] = False
            reasonless.append(finding)

    errors = [finding for finding in findings if finding["severity"] == "error"]
    errors.extend(
        {
            **finding,
            "severity": "error",
            "change": "acknowledged-without-reason",
            "detail": "allowlist entry has no reason",
            "acknowledged": False,
        }
        for finding in reasonless
    )
    inferred = [
        finding for finding in findings if finding["change"] == "surface-inferred"
    ]
    info = [
        finding
        for finding in findings
        if finding["severity"] == "info" and finding["change"] != "surface-inferred"
    ]
    return breaking, acknowledged, errors, inferred, info


def report(
    findings: list[dict], allowlist: dict[str, str], base_label: str, head_label: str
) -> tuple[str, int]:
    breaking, acknowledged, errors, inferred, info = _classify_findings(
        findings, allowlist
    )

    lines = [f"Public API diff: {base_label} -> {head_label}", ""]
    if errors:
        lines.append(f"ERRORS ({len(errors)})")
        lines.extend(f"  - {_describe_finding(finding)}" for finding in errors)
        lines.append("")
    if breaking:
        lines.append(f"BREAKING — {len(breaking)} unacknowledged")
        for finding in breaking:
            lines.append(f"  - {_describe_finding(finding)}")
        lines.append("")
        lines.append(
            "  Keep the old name or parameter as a deprecated alias until its"
            " removal window,"
        )
        lines.append(
            "  or record an intentional break in scripts/dev/public_api_allowlist.json:"
        )
        for finding in breaking:
            lines.append(f'    {{"key": "{finding["key"]}", "reason": "..."}}')
        lines.append("")
    if acknowledged:
        lines.append(f"ACKNOWLEDGED BREAKS ({len(acknowledged)})")
        for finding in acknowledged:
            lines.append(
                f"  - {_describe_finding(finding)} — {allowlist[finding['key']]}"
            )
        lines.append("")
    if inferred:
        lines.append(f"INFERRED SURFACES ({len(inferred)})")
        lines.append(
            "  These modules declare no __all__; the names they define are compared."
        )
        lines.extend(f"  - {finding['module']}" for finding in inferred)
        lines.append("")
    if info:
        lines.append(f"INFORMATIONAL CHANGES ({len(info)})")
        by_module: dict[str, list[str]] = {}
        for finding in info:
            by_module.setdefault(finding["module"], []).append(
                finding["name"]
                if not finding["param"]
                else f"{finding['name']}({finding['param']})"
            )
        for module in sorted(by_module):
            lines.append(f"  {module}: {', '.join(sorted(by_module[module]))}")
        lines.append("")

    stale = set(allowlist) - {finding["key"] for finding in findings}
    if stale:
        lines.append(f"STALE ALLOWLIST ENTRIES ({len(stale)})")
        lines.extend(f"  - {key}" for key in sorted(stale))
        lines.append("  Remove an entry once its break no longer appears.")
        lines.append("")

    lines.append(
        f"{len(breaking)} breaking change(s) need attention;"
        f" {len(acknowledged)} acknowledged, {len(info)} informational."
    )
    return "\n".join(lines), 1 if (breaking or errors or stale) else 0


def _make_base(args: argparse.Namespace) -> tuple[Path, str, str | None]:
    if args.base:
        return Path(args.base), str(Path(args.base)), None
    ref = args.base_ref or latest_stable_tag()
    tmp = tempfile.mkdtemp(prefix="llm-rosetta-api-base-")
    subprocess.run(
        ["git", "worktree", "add", "--detach", tmp, ref],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(tmp), ref, tmp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="directory of the base source tree")
    parser.add_argument(
        "--base-ref",
        help="git ref to check out as the base (default: latest stable vX.Y.Z tag)",
    )
    parser.add_argument("--head", default=".", help="directory of the head tree")
    parser.add_argument(
        "--allowlist", default=str(DEFAULT_ALLOWLIST), help="acknowledged-break list"
    )
    parser.add_argument("--json", action="store_true", help="emit findings as JSON")
    parser.add_argument(
        "--github-summary",
        action="store_true",
        help="also append the report to $GITHUB_STEP_SUMMARY",
    )
    parser.add_argument("--dump", metavar="ROOT", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.dump:
        print(json.dumps(dump_surface(Path(args.dump)), sort_keys=True))
        return 0

    base_root, base_label, cleanup = _make_base(args)
    try:
        base_surface = dump_in_subprocess(base_root)
        head_surface = dump_in_subprocess(Path(args.head).resolve())
    finally:
        if cleanup:
            subprocess.run(
                ["git", "worktree", "remove", "--force", cleanup],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )

    findings = compare(base_surface, head_surface)
    if args.json:
        for finding in findings:
            finding["acknowledged"] = finding["key"] in load_allowlist(
                Path(args.allowlist)
            )
        print(json.dumps(findings, indent=2, sort_keys=True))
        return (
            1
            if any(
                finding["severity"] in ("breaking", "error")
                and not finding["acknowledged"]
                for finding in findings
            )
            else 0
        )

    text, status = report(
        findings,
        load_allowlist(Path(args.allowlist)),
        base_label,
        str(Path(args.head).resolve()),
    )
    print(text)
    if args.github_summary:
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_path:
            with open(summary_path, "a", encoding="utf-8") as handle:
                handle.write(text + "\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
