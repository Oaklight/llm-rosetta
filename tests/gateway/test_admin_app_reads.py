"""``admin/**`` reads the app only through ``AdminContext``, and the reads that remain only shrink.

The import test (``test_admin_import_boundary.py``) sees what admin imports; it
cannot see ``request.app.<attr>``, which reaches the same state without an
import.  This test finds those reads in the syntax tree:

* ``<name>.app.<attr>`` and ``app.<attr>`` (a read, or a write);
* ``getattr`` / ``setattr`` / ``hasattr`` / ``delattr`` on ``<name>.app`` or on a
  variable called ``app``, with a literal attribute name;
* ``<name>.app`` handed on whole (``f(request.app)``), which hides every read the
  callee makes; it is counted under ``<app passed on>``.

An app held under another name is not seen; the convention in ``admin/`` is
``app`` and ``request.app``.

``_APP_READS`` lists, per file and attribute, how many there are today.  The
list may only shrink: a read that is not on it fails the test, a count that
dropped must be lowered, and ``_APP_READS_FROZEN`` (the list as it was drawn,
never raised) refuses an entry that was added or raised in the same commit as
the read it covers.  Replace a read with a member of ``AdminContext``
(``admin_api.py``) and lower both lists.

After a rebase, print the current counts with
``python tests/gateway/test_admin_app_reads.py`` and paste the result over
``_APP_READS``; lower the same entries in ``_APP_READS_FROZEN`` by hand.
"""

from __future__ import annotations

import ast
import threading
from collections import Counter
from pathlib import Path

_ROOT = (
    Path(__file__).parents[2]
    / "packages/llm-rosetta-gateway/src/llm_rosetta_gateway/admin"
)
_PASSED_ON = "<app passed on>"
_ATTR_FUNCS = {"getattr", "setattr", "hasattr", "delattr"}
#: ``app.route(...)`` registers a handler while the panel is mounted; it reads no
#: state, so it is not a read.
_MOUNTING = {"route"}

#: ``(file relative to admin/, attribute)`` to how many times admin reads it.
_APP_READS: dict[tuple[str, str], int] = {
    ("__init__.py", "_admin_head_prefix"): 2,
    ("__init__.py", "admin_custom_head"): 1,
    ("__init__.py", "disabled_tabs"): 1,
    ("routes/_shared.py", "<app passed on>"): 1,
    ("routes/_shared.py", "config_io"): 1,
    ("routes/_shared.py", "config_path"): 1,
    ("routes/_shared.py", "disabled_tabs"): 1,
    ("routes/_shared.py", "provider_validation_warnings"): 1,
    ("routes/_shared.py", "shim_source_report"): 1,
    ("routes/_shared.py", "shim_source_signature"): 1,
    ("routes/_shared.py", "shim_update_check"): 1,
    ("routes/auth.py", "admin_custom_head"): 1,
    ("routes/auth.py", "auth_state"): 6,
    ("routes/auth.py", "internal_token"): 1,
    ("routes/auth.py", "ops_ctx"): 3,
    ("routes/capture.py", "capture_state"): 6,
    ("routes/config/extensions.py", "gateway_config"): 2,
    ("routes/config/extensions.py", "shim_source_report"): 2,
    ("routes/config/payload.py", "auth_state"): 1,
    ("routes/config/payload.py", "gateway_config"): 3,
    ("routes/config/providers.py", "transports"): 1,
    ("routes/config/settings.py", "auth_state"): 1,
    ("routes/config/shim_files.py", "_shim_file_index"): 2,
    ("routes/config/shim_files.py", "gateway_config"): 1,
    ("routes/config/shim_files.py", "shim_source_report"): 2,
    ("routes/config/shim_manifests.py", "_shim_manifest_index"): 2,
    ("routes/config/shim_manifests.py", "shim_source_report"): 1,
    ("routes/config/shim_source_catalog.py", "gateway_config"): 1,
    ("routes/config/shim_source_catalog.py", "shim_source_report"): 1,
    ("routes/config/shims.py", "gateway_config"): 1,
    ("routes/config/shims.py", "provider_validation_warnings"): 2,
    ("routes/config/shims.py", "shim_source_report"): 2,
    ("routes/config/shims.py", "shim_update_check"): 2,
    ("routes/config/upstream.py", "config"): 1,
    ("routes/connectivity.py", "config"): 1,
    ("routes/connectivity.py", "gateway_config"): 1,
    ("routes/connectivity.py", "transports"): 1,
    ("routes/credential.py", "deferred_startup"): 2,
    ("routes/credential.py", "transports"): 1,
    ("routes/jobs.py", "jobs"): 1,
    ("routes/keys.py", "_usage_payload"): 2,
    ("routes/keys.py", "internal_token"): 1,
    ("routes/keys.py", "keystore"): 1,
    ("routes/keys.py", "ops_ctx"): 4,
    ("routes/keys.py", "persistence"): 1,
    ("routes/logos.py", "<app passed on>"): 2,
    ("routes/logos.py", "config"): 1,
    ("routes/maintenance.py", "<app passed on>"): 4,
    ("routes/maintenance.py", "_maintenance_registry"): 2,
    ("routes/maintenance.py", "deferred_startup"): 1,
    ("routes/maintenance.py", "gateway_config"): 2,
    ("routes/maintenance.py", "keystore"): 2,
    ("routes/maintenance.py", "persistence"): 4,
    ("routes/observability.py", "auth_state"): 1,
    ("routes/observability.py", "gateway_config"): 5,
    ("routes/observability.py", "jobs"): 1,
    ("routes/observability.py", "metrics"): 1,
    ("routes/observability.py", "ops_ctx"): 1,
    ("routes/observability.py", "ops_log"): 1,
    ("routes/observability.py", "persistence"): 9,
    ("routes/observability.py", "rate_limit_state"): 1,
    ("routes/observability.py", "request_log"): 3,
    ("routes/profiling.py", "profiler_state"): 7,
    ("routes/routing.py", "request_log"): 2,
    ("routes/testing.py", "_bind_host"): 1,
    ("routes/testing.py", "_bind_port"): 1,
    ("routes/testing.py", "gateway_config"): 1,
    ("routes/testing.py", "internal_token"): 1,
    ("routes/updates.py", "config"): 1,
    ("routes/updates.py", "ops_ctx"): 1,
}

#: The counts when the rule was drawn.  Lower an entry when its reads migrate;
#: never add one or raise a number.
_APP_READS_FROZEN: dict[tuple[str, str], int] = {
    ("__init__.py", "_admin_head_prefix"): 2,
    ("__init__.py", "admin_custom_head"): 1,
    ("__init__.py", "disabled_tabs"): 1,
    ("routes/_shared.py", "<app passed on>"): 1,
    ("routes/_shared.py", "config_io"): 1,
    ("routes/_shared.py", "config_path"): 1,
    ("routes/_shared.py", "disabled_tabs"): 1,
    ("routes/_shared.py", "provider_validation_warnings"): 1,
    ("routes/_shared.py", "shim_source_report"): 1,
    ("routes/_shared.py", "shim_source_signature"): 1,
    ("routes/_shared.py", "shim_update_check"): 1,
    ("routes/auth.py", "admin_custom_head"): 1,
    ("routes/auth.py", "auth_state"): 6,
    ("routes/auth.py", "internal_token"): 1,
    ("routes/auth.py", "ops_ctx"): 3,
    ("routes/capture.py", "capture_state"): 6,
    ("routes/config/extensions.py", "gateway_config"): 2,
    ("routes/config/extensions.py", "shim_source_report"): 2,
    ("routes/config/payload.py", "auth_state"): 1,
    ("routes/config/payload.py", "gateway_config"): 3,
    ("routes/config/providers.py", "transports"): 1,
    ("routes/config/settings.py", "auth_state"): 1,
    ("routes/config/shim_files.py", "_shim_file_index"): 2,
    ("routes/config/shim_files.py", "gateway_config"): 1,
    ("routes/config/shim_files.py", "shim_source_report"): 2,
    ("routes/config/shim_manifests.py", "_shim_manifest_index"): 2,
    ("routes/config/shim_manifests.py", "shim_source_report"): 1,
    ("routes/config/shim_source_catalog.py", "gateway_config"): 1,
    ("routes/config/shim_source_catalog.py", "shim_source_report"): 1,
    ("routes/config/shims.py", "gateway_config"): 1,
    ("routes/config/shims.py", "provider_validation_warnings"): 2,
    ("routes/config/shims.py", "shim_source_report"): 2,
    ("routes/config/shims.py", "shim_update_check"): 2,
    ("routes/config/upstream.py", "config"): 1,
    ("routes/connectivity.py", "config"): 1,
    ("routes/connectivity.py", "gateway_config"): 1,
    ("routes/connectivity.py", "transports"): 1,
    ("routes/credential.py", "deferred_startup"): 2,
    ("routes/credential.py", "transports"): 1,
    ("routes/jobs.py", "jobs"): 1,
    ("routes/keys.py", "_usage_payload"): 2,
    ("routes/keys.py", "internal_token"): 1,
    ("routes/keys.py", "keystore"): 1,
    ("routes/keys.py", "ops_ctx"): 4,
    ("routes/keys.py", "persistence"): 1,
    ("routes/logos.py", "<app passed on>"): 2,
    ("routes/logos.py", "config"): 1,
    ("routes/maintenance.py", "<app passed on>"): 4,
    ("routes/maintenance.py", "_maintenance_registry"): 2,
    ("routes/maintenance.py", "deferred_startup"): 1,
    ("routes/maintenance.py", "gateway_config"): 2,
    ("routes/maintenance.py", "keystore"): 2,
    ("routes/maintenance.py", "persistence"): 4,
    ("routes/observability.py", "auth_state"): 1,
    ("routes/observability.py", "gateway_config"): 5,
    ("routes/observability.py", "jobs"): 1,
    ("routes/observability.py", "metrics"): 1,
    ("routes/observability.py", "ops_ctx"): 1,
    ("routes/observability.py", "ops_log"): 1,
    ("routes/observability.py", "persistence"): 9,
    ("routes/observability.py", "rate_limit_state"): 1,
    ("routes/observability.py", "request_log"): 3,
    ("routes/profiling.py", "profiler_state"): 7,
    ("routes/routing.py", "request_log"): 2,
    ("routes/testing.py", "_bind_host"): 1,
    ("routes/testing.py", "_bind_port"): 1,
    ("routes/testing.py", "gateway_config"): 1,
    ("routes/testing.py", "internal_token"): 1,
    ("routes/updates.py", "config"): 1,
    ("routes/updates.py", "ops_ctx"): 1,
}


def _is_request_app(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "app"
        and isinstance(node.value, ast.Name)
    )


def _is_app(node: ast.AST) -> bool:
    return _is_request_app(node) or (isinstance(node, ast.Name) and node.id == "app")


def app_reads() -> Counter[tuple[str, str]]:
    """Every read of the app in ``admin/**``, counted per file and attribute.

    The syntax trees are built on a thread of their own: CPython 3.11 can fail
    ``ast.parse`` with "AST constructor recursion depth mismatch" on a thread
    where an earlier compile raised, which an earlier test in the same process
    may have done.  A fresh thread has no such history.
    """
    result: list[Counter[tuple[str, str]]] = []
    worker = threading.Thread(target=lambda: result.append(_app_reads()))
    worker.start()
    worker.join()
    if not result:
        raise RuntimeError("the scan thread died; its traceback is above")
    return result[0]


def _app_reads() -> Counter[tuple[str, str]]:
    found: Counter[tuple[str, str]] = Counter()
    for path in sorted(_ROOT.rglob("*.py")):
        rel = path.relative_to(_ROOT).as_posix()
        tree = ast.parse(path.read_text(), filename=str(path))
        named: set[int] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and _is_app(node.value)
                and node.attr not in _MOUNTING
            ):
                found[(rel, node.attr)] += 1
                named.add(id(node.value))
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in _ATTR_FUNCS
                and len(node.args) >= 2
                and _is_app(node.args[0])
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                found[(rel, node.args[1].value)] += 1
                named.add(id(node.args[0]))
        for node in ast.walk(tree):
            if _is_request_app(node) and id(node) not in named:
                found[(rel, _PASSED_ON)] += 1
    return found


def test_a_read_of_the_app_is_listed_and_the_list_only_shrinks() -> None:
    actual = app_reads()
    new = {k: v for k, v in actual.items() if v > _APP_READS.get(k, 0)}
    assert not new, (
        "admin reads the app outside AdminContext; add a member to "
        f"admin_api.AdminContext instead of listing it: {new}"
    )
    stale = {k: v for k, v in _APP_READS.items() if actual.get(k, 0) < v}
    assert not stale, f"lower these counts in _APP_READS: {stale}"


def test_the_list_is_never_raised() -> None:
    raised = {k: v for k, v in _APP_READS.items() if v > _APP_READS_FROZEN.get(k, 0)}
    assert not raised, f"_APP_READS gained or raised an entry: {raised}"


def _printed(counts: Counter[tuple[str, str]]) -> str:
    lines = ["{"]
    for (rel, attr), count in sorted(counts.items()):
        lines.append(f"    ({rel!r}, {attr!r}): {count},")
    lines.append("}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(_printed(app_reads()))
