"""The admin panel depends on ``admin_api``, not on gateway internals.

``admin/`` is served by the gateway but is meant to be replaceable.  Guards:

* ``admin/**`` may import only the admin package, ``admin_api`` and the vendored
  HTTP server from the gateway distribution.  The imports that predate the rule
  sit on ``_LEGACY_ALLOWLIST``; the list may only shrink, so an entry that no
  longer occurs fails the test, a new offender fails it, and so does an entry
  that is not on ``_LEGACY_ALLOWLIST_FROZEN`` (a copy of the list as it was
  drawn, so adding an entry and the import it covers in one commit, or swapping
  one pair for another, cannot pass).
* no module in the gateway package outside ``admin/`` imports the admin package,
  except the app factory, which mounts it (``tests/``, ``scripts/`` and
  ``tools/`` are not scanned: tests legitimately import ``admin``);
* admin never imports ``app._state`` or the private symbols that were made
  public when the boundary was drawn.  These have no allowlist.

The core distribution (``llm_rosetta``) is a legitimate dependency of admin and
is not scanned here.

After a rebase, print the current offenders with
``python tests/gateway/test_admin_import_boundary.py`` and paste the result over
``_LEGACY_ALLOWLIST``.  Migrating an import deletes its entry from both
``_LEGACY_ALLOWLIST`` and ``_LEGACY_ALLOWLIST_FROZEN``; the frozen copy is never
regenerated, so the printer does not emit it.  The entries carry no line
numbers.
"""

from __future__ import annotations

import ast
from pathlib import Path

_ROOT = (
    Path(__file__).parents[2] / "packages/llm-rosetta-gateway/src/llm_rosetta_gateway"
)
_GATEWAY = "llm_rosetta_gateway"
_ADMIN = _GATEWAY + ".admin"

#: Gateway modules admin may import without an allowlist entry.
_ALLOWED = (_ADMIN, _GATEWAY + ".admin_api", _GATEWAY + "._vendor.httpserver")

#: The one place outside ``admin/`` that may import it, and what it may take.
_FACTORY = "app/factory.py"
_FACTORY_MAY_IMPORT = {"setup_admin", "register_admin_routes"}

#: Names admin once reached through a private path.  They are public (or served
#: through ``admin_api``) now, and there is no way back.
_FORBIDDEN = {
    _GATEWAY + ".app._state",
    _GATEWAY + ".config.providers._account_token_for",
    _GATEWAY + ".transport.provider_info._VERSION_SUFFIXES",
    _GATEWAY + ".shims.sources._selection_predicate",
}

#: Gateway imports in ``admin/**`` that predate the rule, as ``(file, module)``
#: with *file* relative to the gateway package.  Migrate one, delete its entry.
_LEGACY_ALLOWLIST: frozenset[tuple[str, str]] = frozenset(
    {
        ("admin/__init__.py", "llm_rosetta_gateway.config"),
        ("admin/__init__.py", "llm_rosetta_gateway.jobs"),
        ("admin/__init__.py", "llm_rosetta_gateway.jobs.handlers.data"),
        ("admin/__init__.py", "llm_rosetta_gateway.observability"),
        ("admin/__init__.py", "llm_rosetta_gateway.ops"),
        ("admin/__init__.py", "llm_rosetta_gateway.ops.data"),
        ("admin/routes/_shared.py", "llm_rosetta_gateway.config"),
        ("admin/routes/_shared.py", "llm_rosetta_gateway.config.resolve"),
        ("admin/routes/_shared.py", "llm_rosetta_gateway.observability.logging"),
        ("admin/routes/_shared.py", "llm_rosetta_gateway.model_types"),
        ("admin/routes/_shared.py", "llm_rosetta_gateway.observability.ops_log"),
        ("admin/routes/_shared.py", "llm_rosetta_gateway.ops"),
        ("admin/routes/_shared.py", "llm_rosetta_gateway.config.providers"),
        ("admin/routes/_shared.py", "llm_rosetta_gateway.shim_sources"),
        ("admin/routes/auth.py", "llm_rosetta_gateway.config"),
        ("admin/routes/auth.py", "llm_rosetta_gateway.middleware.auth"),
        ("admin/routes/auth.py", "llm_rosetta_gateway.ops.security"),
        ("admin/routes/capture.py", "llm_rosetta_gateway.observability.capture"),
        ("admin/routes/config/model_providers.py", "llm_rosetta_gateway.config.parse"),
        ("admin/routes/config/models.py", "llm_rosetta_gateway.model_types"),
        ("admin/routes/config/payload.py", "llm_rosetta_gateway.__version__"),
        ("admin/routes/config/payload.py", "llm_rosetta_gateway.config"),
        ("admin/routes/config/payload.py", "llm_rosetta_gateway.config.parse"),
        ("admin/routes/config/payload.py", "llm_rosetta_gateway.model_types"),
        ("admin/routes/config/payload.py", "llm_rosetta_gateway.config.providers"),
        ("admin/routes/config/payload.py", "llm_rosetta_gateway.shims"),
        (
            "admin/routes/config/payload.py",
            "llm_rosetta_gateway.transport.token_refresh",
        ),
        ("admin/routes/config/providers.py", "llm_rosetta_gateway.config.typed"),
        ("admin/routes/config/providers.py", "llm_rosetta_gateway.model_types"),
        ("admin/routes/config/settings.py", "llm_rosetta_gateway.config.settings"),
        ("admin/routes/config/settings.py", "llm_rosetta_gateway.middleware.ratelimit"),
        ("admin/routes/config/shim_files.py", "llm_rosetta_gateway._vendor.yaml"),
        ("admin/routes/config/shim_files.py", "llm_rosetta_gateway.shims.catalog"),
        ("admin/routes/config/shim_files.py", "llm_rosetta_gateway.shims.sources"),
        ("admin/routes/config/shim_manifests.py", "llm_rosetta_gateway._vendor.yaml"),
        ("admin/routes/config/shims.py", "llm_rosetta_gateway.shim_sources"),
        ("admin/routes/config/shims.py", "llm_rosetta_gateway.shims.catalog"),
        ("admin/routes/config/shims.py", "llm_rosetta_gateway.shims.sources"),
        ("admin/routes/config/upstream.py", "llm_rosetta_gateway._vendor.httpclient"),
        ("admin/routes/config/upstream.py", "llm_rosetta_gateway.config"),
        ("admin/routes/config/upstream.py", "llm_rosetta_gateway.config.providers"),
        ("admin/routes/connectivity.py", "llm_rosetta_gateway._vendor.httpclient"),
        ("admin/routes/connectivity.py", "llm_rosetta_gateway.config.typed"),
        ("admin/routes/connectivity.py", "llm_rosetta_gateway.config.providers"),
        ("admin/routes/connectivity.py", "llm_rosetta_gateway.transport.provider_info"),
        ("admin/routes/credential.py", "llm_rosetta_gateway.config.providers"),
        ("admin/routes/credential.py", "llm_rosetta_gateway.transport.provider_info"),
        ("admin/routes/credential.py", "llm_rosetta_gateway.transport.token_refresh"),
        ("admin/routes/keys.py", "llm_rosetta_gateway._vendor.ratelimit"),
        ("admin/routes/keys.py", "llm_rosetta_gateway.keystore"),
        ("admin/routes/keys.py", "llm_rosetta_gateway.ops.keys"),
        ("admin/routes/logos.py", "llm_rosetta_gateway._vendor.httpclient"),
        ("admin/routes/maintenance.py", "llm_rosetta_gateway.maintenance.backfills"),
        ("admin/routes/maintenance.py", "llm_rosetta_gateway.maintenance.keystore"),
        (
            "admin/routes/maintenance.py",
            "llm_rosetta_gateway.observability.maintenance",
        ),
        ("admin/routes/observability.py", "llm_rosetta_gateway._vendor.httpclient"),
        ("admin/routes/observability.py", "llm_rosetta_gateway.config"),
        ("admin/routes/observability.py", "llm_rosetta_gateway.jobs.enqueue"),
        (
            "admin/routes/observability.py",
            "llm_rosetta_gateway.observability.error_dump",
        ),
        ("admin/routes/observability.py", "llm_rosetta_gateway.observability.ops_log"),
        ("admin/routes/observability.py", "llm_rosetta_gateway.ops.data"),
        ("admin/routes/profiling.py", "llm_rosetta_gateway.observability.profiling"),
        ("admin/routes/quota.py", "llm_rosetta_gateway.observability.sanitize"),
        ("admin/routes/quota.py", "llm_rosetta_gateway.quota.service"),
        ("admin/routes/routing.py", "llm_rosetta_gateway.quota.service"),
        ("admin/routes/routing.py", "llm_rosetta_gateway.routing"),
        ("admin/routes/routing.py", "llm_rosetta_gateway.routing.eligibility"),
        ("admin/routes/testing.py", "llm_rosetta_gateway._vendor.httpclient"),
        ("admin/routes/testing.py", "llm_rosetta_gateway.config"),
        ("admin/routes/updates.py", "llm_rosetta_gateway.ops.security"),
        ("admin/routes/updates.py", "llm_rosetta_gateway.updates"),
    }
)

#: The pairs on the list when the boundary was drawn, never added to.  Delete an
#: entry here together with the same entry in ``_LEGACY_ALLOWLIST`` when its
#: import migrates.  A pair on the live list but not here is a new offender that
#: was allow-listed instead of routed through ``admin_api``.
_LEGACY_ALLOWLIST_FROZEN: frozenset[tuple[str, str]] = frozenset(
    {
        ("admin/__init__.py", "llm_rosetta_gateway.config"),
        ("admin/__init__.py", "llm_rosetta_gateway.jobs"),
        ("admin/__init__.py", "llm_rosetta_gateway.jobs.handlers.data"),
        ("admin/__init__.py", "llm_rosetta_gateway.observability"),
        ("admin/__init__.py", "llm_rosetta_gateway.ops"),
        ("admin/__init__.py", "llm_rosetta_gateway.ops.data"),
        ("admin/routes/_shared.py", "llm_rosetta_gateway.config"),
        ("admin/routes/_shared.py", "llm_rosetta_gateway.config.resolve"),
        ("admin/routes/_shared.py", "llm_rosetta_gateway.observability.logging"),
        ("admin/routes/_shared.py", "llm_rosetta_gateway.model_types"),
        ("admin/routes/_shared.py", "llm_rosetta_gateway.observability.ops_log"),
        ("admin/routes/_shared.py", "llm_rosetta_gateway.ops"),
        ("admin/routes/_shared.py", "llm_rosetta_gateway.config.providers"),
        ("admin/routes/_shared.py", "llm_rosetta_gateway.shim_sources"),
        ("admin/routes/auth.py", "llm_rosetta_gateway.config"),
        ("admin/routes/auth.py", "llm_rosetta_gateway.middleware.auth"),
        ("admin/routes/auth.py", "llm_rosetta_gateway.ops.security"),
        ("admin/routes/capture.py", "llm_rosetta_gateway.observability.capture"),
        ("admin/routes/config/model_providers.py", "llm_rosetta_gateway.config.parse"),
        ("admin/routes/config/models.py", "llm_rosetta_gateway.model_types"),
        ("admin/routes/config/payload.py", "llm_rosetta_gateway.__version__"),
        ("admin/routes/config/payload.py", "llm_rosetta_gateway.config"),
        ("admin/routes/config/payload.py", "llm_rosetta_gateway.config.parse"),
        ("admin/routes/config/payload.py", "llm_rosetta_gateway.model_types"),
        ("admin/routes/config/payload.py", "llm_rosetta_gateway.config.providers"),
        ("admin/routes/config/payload.py", "llm_rosetta_gateway.shims"),
        (
            "admin/routes/config/payload.py",
            "llm_rosetta_gateway.transport.token_refresh",
        ),
        ("admin/routes/config/providers.py", "llm_rosetta_gateway.config.typed"),
        ("admin/routes/config/providers.py", "llm_rosetta_gateway.model_types"),
        ("admin/routes/config/settings.py", "llm_rosetta_gateway.config.settings"),
        ("admin/routes/config/settings.py", "llm_rosetta_gateway.middleware.ratelimit"),
        ("admin/routes/config/shim_files.py", "llm_rosetta_gateway._vendor.yaml"),
        ("admin/routes/config/shim_files.py", "llm_rosetta_gateway.shims.catalog"),
        ("admin/routes/config/shim_files.py", "llm_rosetta_gateway.shims.sources"),
        ("admin/routes/config/shim_manifests.py", "llm_rosetta_gateway._vendor.yaml"),
        ("admin/routes/config/shims.py", "llm_rosetta_gateway.shim_sources"),
        ("admin/routes/config/shims.py", "llm_rosetta_gateway.shims.catalog"),
        ("admin/routes/config/shims.py", "llm_rosetta_gateway.shims.sources"),
        ("admin/routes/config/upstream.py", "llm_rosetta_gateway._vendor.httpclient"),
        ("admin/routes/config/upstream.py", "llm_rosetta_gateway.config"),
        ("admin/routes/config/upstream.py", "llm_rosetta_gateway.config.providers"),
        ("admin/routes/connectivity.py", "llm_rosetta_gateway._vendor.httpclient"),
        ("admin/routes/connectivity.py", "llm_rosetta_gateway.config.typed"),
        ("admin/routes/connectivity.py", "llm_rosetta_gateway.config.providers"),
        ("admin/routes/connectivity.py", "llm_rosetta_gateway.transport.provider_info"),
        ("admin/routes/credential.py", "llm_rosetta_gateway.config.providers"),
        ("admin/routes/credential.py", "llm_rosetta_gateway.transport.provider_info"),
        ("admin/routes/credential.py", "llm_rosetta_gateway.transport.token_refresh"),
        ("admin/routes/keys.py", "llm_rosetta_gateway._vendor.ratelimit"),
        ("admin/routes/keys.py", "llm_rosetta_gateway.keystore"),
        ("admin/routes/keys.py", "llm_rosetta_gateway.ops.keys"),
        ("admin/routes/logos.py", "llm_rosetta_gateway._vendor.httpclient"),
        ("admin/routes/maintenance.py", "llm_rosetta_gateway.maintenance.backfills"),
        ("admin/routes/maintenance.py", "llm_rosetta_gateway.maintenance.keystore"),
        (
            "admin/routes/maintenance.py",
            "llm_rosetta_gateway.observability.maintenance",
        ),
        ("admin/routes/observability.py", "llm_rosetta_gateway._vendor.httpclient"),
        ("admin/routes/observability.py", "llm_rosetta_gateway.config"),
        ("admin/routes/observability.py", "llm_rosetta_gateway.jobs.enqueue"),
        (
            "admin/routes/observability.py",
            "llm_rosetta_gateway.observability.error_dump",
        ),
        ("admin/routes/observability.py", "llm_rosetta_gateway.observability.ops_log"),
        ("admin/routes/observability.py", "llm_rosetta_gateway.ops.data"),
        ("admin/routes/profiling.py", "llm_rosetta_gateway.observability.profiling"),
        ("admin/routes/quota.py", "llm_rosetta_gateway.observability.sanitize"),
        ("admin/routes/quota.py", "llm_rosetta_gateway.quota.service"),
        ("admin/routes/routing.py", "llm_rosetta_gateway.quota.service"),
        ("admin/routes/routing.py", "llm_rosetta_gateway.routing"),
        ("admin/routes/routing.py", "llm_rosetta_gateway.routing.eligibility"),
        ("admin/routes/testing.py", "llm_rosetta_gateway._vendor.httpclient"),
        ("admin/routes/testing.py", "llm_rosetta_gateway.config"),
        ("admin/routes/updates.py", "llm_rosetta_gateway.ops.security"),
        ("admin/routes/updates.py", "llm_rosetta_gateway.updates"),
    }
)


def _absolute_module(path: Path, node: ast.ImportFrom) -> str:
    if not node.level:
        return node.module or ""
    package = [_GATEWAY, *path.relative_to(_ROOT).parent.parts]
    base = package[: len(package) - (node.level - 1)]
    if node.module:
        base += node.module.split(".")
    return ".".join(base)


def _is_gateway(name: str) -> bool:
    return name == _GATEWAY or name.startswith(_GATEWAY + ".")


def _dynamic_target(node: ast.Call) -> str | None:
    is_import = (isinstance(node.func, ast.Name) and node.func.id == "__import__") or (
        isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"
    )
    if not is_import or not node.args:
        return None
    arg = node.args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    return None


def _node_modules(path: Path, node: ast.AST) -> list[str]:
    """Return the gateway modules one AST node imports, fully qualified.

    A ``from pkg import name`` names *pkg*, and also ``pkg.name`` because
    *name* may be a submodule.  ``from llm_rosetta_gateway import name`` names
    only the qualified form: the bare package is not an import of interest.
    """
    if isinstance(node, ast.ImportFrom):
        module = _absolute_module(path, node)
        if not _is_gateway(module):
            return []
        qualified = [f"{module}.{a.name}" for a in node.names if a.name != "*"]
        return qualified if module == _GATEWAY else [module, *qualified]
    if isinstance(node, ast.Import):
        return [a.name for a in node.names if _is_gateway(a.name)]
    if isinstance(node, ast.Call):
        target = _dynamic_target(node)
        if target is not None and _is_gateway(target):
            return [target]
    return []


def _imported_modules(path: Path, source: str) -> list[tuple[int, str]]:
    """Return ``(line, qualified module)`` for every gateway import at any depth."""
    return [
        (getattr(node, "lineno", 0), module)
        for node in ast.walk(ast.parse(source))
        for module in _node_modules(path, node)
    ]


def _is_allowed(module: str) -> bool:
    return any(module == ok or module.startswith(ok + ".") for ok in _ALLOWED)


def _offending_pairs(path: Path, source: str) -> set[tuple[str, str]]:
    """Return the ``(file, module)`` pairs the import rule rejects in *path*.

    The qualified ``pkg.name`` of a ``from pkg import name`` is judged only
    when *pkg* is the package root, so a name imported from an allowed module
    is not reported as a separate module.
    """
    rel = path.relative_to(_ROOT).as_posix()
    pairs: set[tuple[str, str]] = set()
    for node in ast.walk(ast.parse(source)):
        modules = _node_modules(path, node)
        if (
            isinstance(node, ast.ImportFrom)
            and _absolute_module(path, node) != _GATEWAY
        ):
            modules = modules[:1]
        pairs.update((rel, m) for m in modules if not _is_allowed(m))
    return pairs


def _admin_files() -> list[Path]:
    return sorted((_ROOT / "admin").rglob("*.py"))


def _admin_offenders() -> set[tuple[str, str]]:
    offenders: set[tuple[str, str]] = set()
    for path in _admin_files():
        offenders |= _offending_pairs(path, path.read_text())
    return offenders


def test_admin_imports_only_its_boundary():
    new = sorted(_admin_offenders() - _LEGACY_ALLOWLIST)
    assert new == [], (
        "admin/ must reach the gateway through llm_rosetta_gateway.admin_api; "
        f"new gateway imports: {new}"
    )


def test_legacy_allowlist_only_shrinks():
    stale = sorted(_LEGACY_ALLOWLIST - _admin_offenders())
    assert stale == [], f"remove these entries from _LEGACY_ALLOWLIST: {stale}"


def test_legacy_allowlist_never_grows():
    # The two checks above compare the list with the tree, so a commit that adds
    # an entry together with the import it covers passes both.  The frozen copy
    # pins membership, which also bounds the size, so a swap of one pair for
    # another fails too (this replaces a size ceiling, which a swap passed).
    added = sorted(_LEGACY_ALLOWLIST - _LEGACY_ALLOWLIST_FROZEN)
    assert added == [], (
        f"new entries on _LEGACY_ALLOWLIST: {added}. Route the import through "
        "llm_rosetta_gateway.admin_api instead of allow-listing it."
    )


def test_nothing_outside_admin_imports_admin_but_the_factory_mount():
    offenders: list[str] = []
    for path in sorted(_ROOT.rglob("*.py")):
        rel = path.relative_to(_ROOT).as_posix()
        if rel.startswith(("admin/", "_vendor/")):
            continue
        for line, module in _imported_modules(path, path.read_text()):
            if module != _ADMIN and not module.startswith(_ADMIN + "."):
                continue
            if rel == _FACTORY and (
                module in (_ADMIN, _ADMIN + ".routes")
                or module.rsplit(".", 1)[-1] in _FACTORY_MAY_IMPORT
            ):
                continue
            offenders.append(f"{rel}:{line} {module}")
    assert offenders == []


def test_admin_never_reaches_the_private_symbols_it_once_used():
    offenders = [
        f"{path.relative_to(_ROOT).as_posix()}:{line} {module}"
        for path in _admin_files()
        for line, module in _imported_modules(path, path.read_text())
        if module in _FORBIDDEN
    ]
    assert offenders == []


def test_guard_catches_absolute_relative_and_dynamic_imports():
    probe = _ROOT / "admin" / "routes" / "keys.py"

    def hits(source: str) -> set[tuple[str, str]]:
        return _offending_pairs(probe, source)

    assert hits("from llm_rosetta_gateway.keystore import KeyStore")
    assert hits("from ...keystore import KeyStore")
    assert hits("from ... import keystore")
    assert hits("import llm_rosetta_gateway.config")
    assert hits('import importlib\nimportlib.import_module("llm_rosetta_gateway.x")')
    assert hits("from llm_rosetta_gateway.app import _state")
    assert hits("from llm_rosetta_gateway.admin_api import admin_context") == set()
    assert hits("from llm_rosetta_gateway._vendor.httpserver import Response") == set()
    assert hits("from ..routes import _shared") == set()
    assert hits("from llm_rosetta.shims import get_shim") == set()
    forbidden = {m for _, m in _imported_modules(probe, "from ...app import _state")}
    assert forbidden & _FORBIDDEN


if __name__ == "__main__":
    print("_LEGACY_ALLOWLIST: frozenset[tuple[str, str]] = frozenset(\n    {")
    for _file, _module in sorted(_admin_offenders()):
        print(f'        ("{_file}", "{_module}"),')
    print("    }\n)")
