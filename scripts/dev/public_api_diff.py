#!/usr/bin/env python3
"""Compare the public API surface of the latest stable release with this tree.

A tree's surface is the set of names each public module exports plus the
parameter list of every callable or class it exports.  A module that declares
``__all__`` uses it; otherwise the reachable public names are used and the
report marks that surface as *inferred*.  "Reachable" is read from the source
at module level: classes and functions defined there, UPPER_CASE constants,
and names imported from another ``llm_rosetta`` module (a re-export).  A name
imported from the standard library or a third party is not surface, and
neither is an imported module object.

Scope limits: only module-level names and the constructor signature of an
exported class are compared — methods on a class are not.  The module list is
deliberately explicit, so packages that are not listed (for example
``llm_rosetta_gateway.admin.*``) are out of scope.  Each tree is measured with
its own list, read from its ``llm_rosetta/_tiers.py``; a tree without one (a
release up to v0.13.0) is measured with this script's list.

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
import ast
import importlib
import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALLOWLIST = Path(__file__).with_name("public_api_allowlist.json")

#: Only a bare ``vMAJOR.MINOR.PATCH`` tag is a stable release.  Anchored with
#: ``fullmatch`` so ``v0.14.0.dev4``, ``v0.14.0rc1`` and ``v1.0.0-beta`` never
#: qualify.
STABLE_TAG_RE = re.compile(r"v\d+\.\d+\.\d+")

#: The core tier table (``llm_rosetta/_tiers.py``) of the tree this script
#: belongs to.  It is data only and is loaded by file path, so reading it never
#: imports ``llm_rosetta`` (the base tree is imported in a subprocess).
TIERS_FILE = REPO_ROOT / "packages/llm-rosetta/src/llm_rosetta/_tiers.py"


def _load_tiers(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("_llm_rosetta_tiers", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load the tier table {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TIERS = _load_tiers(TIERS_FILE)

#: Gateway modules in scope.  The gateway distribution has no tier table; its
#: host-facing modules are listed here.
GATEWAY_PUBLIC_MODULES = (
    "llm_rosetta_gateway.observability",
    "llm_rosetta_gateway.observability.metrics",
    "llm_rosetta_gateway.observability.persistence",
    "llm_rosetta_gateway.observability.request_log",
    "llm_rosetta_gateway.observability.ops_log",
    "llm_rosetta_gateway",
)

#: Modules whose declared or inferred surface is part of the supported public
#: API: the core's user, extension and compat tiers, then the gateway's.  A
#: module without ``__all__`` falls back to the names reachable at module scope
#: (definitions, UPPER_CASE constants and intra-package re-exports), and the
#: report labels that surface as inferred.  The list is deliberately explicit:
#: importing every module under the package would execute the gateway CLI's
#: argv parsing.
PUBLIC_MODULES: tuple[str, ...] = (
    *_TIERS.USER,
    *_TIERS.EXTENSION,
    *_TIERS.COMPAT,
    *GATEWAY_PUBLIC_MODULES,
)

#: Modules a final release shipped and this tree removed on purpose (the
#: ``REMOVED`` tier).  They are still compared, so the removal of each name is
#: a breaking finding the allowlist has to acknowledge.
REMOVED_MODULES: tuple[str, ...] = tuple(_TIERS.REMOVED)

#: Everything the surface dump of this script's own tree visits.  It is also
#: the fallback list for a tree that has no tier table (every release up to
#: and including v0.13.0).
COMPARED_MODULES: tuple[str, ...] = PUBLIC_MODULES + REMOVED_MODULES


def _modules_of(tiers: ModuleType) -> tuple[str, ...]:
    return (
        *tiers.USER,
        *tiers.EXTENSION,
        *tiers.COMPAT,
        *GATEWAY_PUBLIC_MODULES,
        *tiers.REMOVED,
    )


def _tree_tiers(root: Path) -> ModuleType | None:
    """The tier table of the tree at *root*, or ``None`` when it has none."""
    path = _core_root(root) / "llm_rosetta" / "_tiers.py"
    return _load_tiers(path) if path.is_file() else None


def tree_modules(root: Path) -> tuple[str, ...]:
    """The modules the surface of the tree at *root* is made of.

    Each tree is read with its **own** tier table, so a base release is
    measured by what it declared public, not by what the head declares.
    Otherwise a PR that drops a module from ``_tiers.py`` and deletes the
    module's allowlist entries in the same change would compare nothing on
    either side and report nothing (#1659 review).  That holds against a base
    that ships ``_tiers.py``, i.e. from 0.14.0 on.  A tree without a tier
    table (v0.13.0 and earlier) falls back to this script's own list, so for
    the 0.13.0 base a dropped module is caught by the frozen module list in
    ``tests/test_public_api_diff.py`` instead.

    Raises:
        SystemExit: The table lists a module twice; the dump is keyed by
            module name, so a duplicate would make the table and the dump
            disagree on the module count.
    """
    tiers = _tree_tiers(root)
    modules = COMPARED_MODULES if tiers is None else _modules_of(tiers)
    duplicates = sorted({name for name in modules if modules.count(name) > 1})
    if duplicates:
        raise SystemExit(f"the tier table of {root} lists {duplicates} twice")
    return modules


def check_dump_complete(surface: dict, modules: Iterable[str], label: str) -> None:
    """Fail unless *surface* holds exactly one entry per module in *modules*.

    The dump runs in a subprocess; this check, in the parent, proves that the
    comparison covers every module the tier table lists and nothing else.
    """
    expected = tuple(modules)
    if len(surface) != len(expected) or set(surface) != set(expected):
        missing = sorted(set(expected) - set(surface))
        extra = sorted(set(surface) - set(expected))
        raise SystemExit(
            f"the surface dump of {label} has {len(surface)} modules, its tier"
            f" table lists {len(expected)} (missing {missing}, extra {extra})"
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


def _src_roots(root: Path) -> list[Path]:
    """The import roots of a checkout, old layout or ``packages/`` layout."""
    candidates = (
        root / "packages/llm-rosetta/src",
        root / "packages/llm-rosetta-gateway/src",
        root / "src",
    )
    # A root is any source directory that holds one of the two import roots.
    # The gateway root holds ``llm_rosetta_gateway`` and no ``llm_rosetta``, so
    # testing for ``llm_rosetta`` alone drops it and the gateway modules are
    # then imported from whatever copy happens to be installed.
    found = [
        path.resolve()
        for path in candidates
        if (path / "llm_rosetta").is_dir() or (path / "llm_rosetta_gateway").is_dir()
    ]
    if not found:
        raise SystemExit(f"{root} does not look like an llm-rosetta source tree")
    return found


def _core_root(root: Path) -> Path:
    """The source directory of *root* that holds ``llm_rosetta``.

    Named by what it holds rather than by its position in
    :func:`_src_roots`, so reordering the candidates cannot make the origin
    check below compare against the gateway root.
    """
    core = next(
        (path for path in _src_roots(root) if (path / "llm_rosetta").is_dir()),
        None,
    )
    if core is None:
        raise SystemExit(f"{root} has no llm_rosetta source directory")
    return core


def dump_surface(root: Path) -> dict:
    src = _core_root(root)
    # The tree's roots go to the front of ``sys.path``; this ordering is what
    # makes the dump measure *root* and not an installed copy (an editable
    # install's finder sits behind the path entries).  The two origin checks
    # below are a backstop, not the protection: while the roots are first they
    # can only fire for a shadow inside the checkout itself, such as a stray
    # ``llm_rosetta_gateway`` directory on an earlier root of the same tree.
    for path in reversed(_src_roots(root)):
        sys.path.insert(0, str(path))

    import llm_rosetta  # noqa: PLC0415

    origin = Path(llm_rosetta.__file__).resolve()
    if not origin.is_relative_to(src / "llm_rosetta"):
        raise SystemExit(f"imported llm_rosetta from {origin}, expected under {src}")

    gateway_src = next(
        (path for path in _src_roots(root) if (path / "llm_rosetta_gateway").is_dir()),
        None,
    )
    if gateway_src is not None:
        import llm_rosetta_gateway  # noqa: PLC0415

        gateway_origin = Path(llm_rosetta_gateway.__file__).resolve()
        if not gateway_origin.is_relative_to(gateway_src / "llm_rosetta_gateway"):
            raise SystemExit(
                f"imported llm_rosetta_gateway from {gateway_origin},"
                f" expected under {gateway_src}"
            )

    modules = tree_modules(root)
    result: dict[str, dict] = {}
    # A tree that predates the split has no ``llm_rosetta_gateway``; importing
    # the name anyway would find this machine's installed copy and measure the
    # wrong tree, so report it absent instead.
    gateway_in_tree = any(
        (path / "llm_rosetta_gateway").is_dir() for path in _src_roots(root)
    )
    for module_name in modules:
        if module_name.startswith("llm_rosetta_gateway") and not gateway_in_tree:
            result[module_name] = {
                "status": "absent",
                "error": "not part of this tree",
            }
            continue
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
            # names reachable at module scope.  ``__module__`` alone would miss
            # two shapes that users can reach and that #1122 removed silently:
            # an imported alias (its ``__module__`` is the defining module, or
            # ``typing`` for a ``Literal`` alias) and a constant (data has no
            # ``__module__``).  The source is the only place both are visible.
            names = _inferred_names(module, module_name)
            source = "inferred"
        else:
            names = declared
            source = "declared"
        result[module_name] = {
            "status": "ok",
            "source": source,
            "exports": {name: _describe(getattr(module, name, None)) for name in names},
        }
    check_dump_complete(result, modules, str(root))
    return result


def _bound_names(target: ast.expr) -> list[str]:
    """Names bound by an assignment target, including tuple unpacking."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: list[str] = []
        for element in target.elts:
            names.extend(_bound_names(element))
        return names
    return []


def _inferred_names(module: object, module_name: str) -> list[str]:
    """Public surface of a module that declares no ``__all__``.

    The surface is what a caller can reach at module scope and that belongs to
    this package:

    * classes and functions defined in the module;
    * module-level UPPER_CASE constants (``DEFAULT_MAX_AGE_DAYS``); and
    * names imported from another ``llm_rosetta`` module — a deliberate
      re-export such as ``routing.ProviderType``.

    Imported stdlib/third-party names (``Any``, ``dataclass``) and imported
    module objects are implementation imports, not API.
    """
    source = inspect.getsourcefile(module)
    if source is None or not Path(source).is_file():
        return []
    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    bound: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            bound.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                # ``logger = get_logger()`` is module state; a constant is API.
                bound.update(name for name in _bound_names(target) if name.isupper())
        elif isinstance(node, ast.ImportFrom):
            intra_package = node.level > 0 or (node.module or "").startswith(
                "llm_rosetta"
            )
            if intra_package:
                for alias in node.names:
                    if alias.name != "*":
                        bound.add(alias.asname or alias.name)
    return sorted(
        name
        for name in bound
        if not name.startswith("_")
        and name in vars(module)
        and not inspect.ismodule(vars(module)[name])
    )


def dump_in_subprocess(root: Path) -> dict:
    """Import *root* in a clean process so two trees never share module state."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in _src_roots(root))
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
    """Compare two surface dumps, each keyed by its own tree's module list.

    A module only the base lists is compared against an absent head, so a
    module the head dropped from its tier table is a breaking finding for
    each of its names, whether or not the module still imports.
    """
    findings: list[dict] = []
    for module in [*base, *(name for name in head if name not in base)]:
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
            detail = module if module in head else f"{module}, not in the head tiers"
            for name in base_entry.get("exports", {}):
                findings.append(
                    _finding("breaking", module, name, "module-removed", detail=detail)
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
    head_root = Path(args.head).resolve()
    try:
        base_surface = dump_in_subprocess(base_root)
        head_surface = dump_in_subprocess(head_root)
        check_dump_complete(base_surface, tree_modules(base_root), base_label)
        check_dump_complete(head_surface, tree_modules(head_root), str(head_root))
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
