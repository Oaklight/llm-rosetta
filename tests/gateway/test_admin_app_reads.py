"""``admin/**`` never reads or writes the app; it asks ``AdminContext``.

The import test (``test_admin_import_boundary.py``) sees what admin imports; it
cannot see ``request.app.<attr>``, which reaches the same state without an
import.  This test finds those reads in the syntax tree:

* ``<name>.app.<attr>`` and ``app.<attr>`` (a read, or a write);
* ``getattr`` / ``setattr`` / ``hasattr`` / ``delattr`` on ``<name>.app`` or on a
  variable called ``app``, with a literal attribute name;
* ``<name>.app`` handed on whole (``f(request.app)``), which hides every read the
  callee makes; it is reported as ``<app passed on>``.

An app held under another name is not seen; the convention in ``admin/`` is
``app`` and ``request.app``.  A need for gateway state becomes a member of
``AdminContext`` (``admin_api.py``) and its implementation in
``app/admin_context.py``.

Print what the scan finds with ``python tests/gateway/test_admin_app_reads.py``.
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


def test_admin_never_reads_the_app() -> None:
    found = {key: count for key, count in sorted(app_reads().items())}
    assert not found, (
        "admin reads the app outside AdminContext; add a member to "
        f"admin_api.AdminContext instead: {found}"
    )


def _printed(counts: Counter[tuple[str, str]]) -> str:
    lines = ["{"]
    for (rel, attr), count in sorted(counts.items()):
        lines.append(f"    ({rel!r}, {attr!r}): {count},")
    lines.append("}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(_printed(app_reads()))
