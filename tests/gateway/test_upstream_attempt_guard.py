"""One home for the bookkeeping of an upstream attempt (#1830 A4).

An attempt on an upstream is counted in three ways: the lane it was made on
(``attempted_lane``), a served answer clearing that lane's record
(``note_upstream_success``), and a host that did not answer sinking it
(``note_transport_refusal``).  Five call sites used to spell all three out —
the JSON and streaming chat paths, the typed pipelines and the relayed
``compact`` and ``count_tokens`` routes — and each copy could drift from the
others.  ``proxy/attempt.py::attempt_upstream`` is now the one place; this test
fails when a site calls the three names itself again, and when one of the five
stops going through the helper.

The scan is on the source: a call by bare name or by attribute, and an import
of the name, outside the modules that own the names.
"""

from __future__ import annotations

import ast
from pathlib import Path

_GATEWAY = "llm_rosetta_gateway"
_ROOT = Path(__file__).parents[2] / "packages/llm-rosetta-gateway/src" / _GATEWAY

#: The three bookkeeping names a site must not use itself.
_GUARDED = frozenset(
    {"attempted_lane", "note_upstream_success", "note_transport_refusal"}
)

#: The helper's module, the modules that define or compose the names, and the
#: quota package that records them.  Everything else goes through the helper.
_OWNERS = (
    "proxy/attempt.py",
    "proxy/errors.py",
    "proxy/failover.py",
    "quota/",
)

#: The five sites that used to carry their own copy; each must call the helper.
_SITES = (
    "proxy/engine.py",
    "proxy/streaming.py",
    "pipelines/typed.py",
    "app/routes_compact.py",
    "app/routes_count_tokens.py",
)


def _owned(relative: str) -> bool:
    return any(relative.startswith(owner) for owner in _OWNERS)


def _name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _uses(source: str) -> list[tuple[int, str]]:
    """The ``(line, name)`` of each guarded call or import in *source*."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            name = _name(node.func)
            if name in _GUARDED:
                found.append((node.lineno, name))
        elif isinstance(node, ast.ImportFrom):
            found.extend(
                (node.lineno, alias.name)
                for alias in node.names
                if alias.name in _GUARDED
            )
    return found


def _calls_helper(source: str) -> bool:
    return any(
        isinstance(node, ast.Call) and _name(node.func) == "attempt_upstream"
        for node in ast.walk(ast.parse(source))
    )


def _modules() -> list[tuple[str, str]]:
    return [
        (path.relative_to(_ROOT).as_posix(), path.read_text(encoding="utf-8"))
        for path in sorted(_ROOT.rglob("*.py"))
        if "_vendor" not in path.parts
    ]


def test_no_site_keeps_its_own_copy_of_the_bookkeeping() -> None:
    offenders = [
        f"{relative}:{line} {name}"
        for relative, source in _modules()
        if not _owned(relative)
        for line, name in _uses(source)
    ]
    assert not offenders, (
        "an upstream attempt is counted through proxy/attempt.py::attempt_upstream; "
        f"these call the bookkeeping directly: {offenders}"
    )


def test_each_of_the_five_sites_goes_through_the_helper() -> None:
    by_path = dict(_modules())
    missing = [site for site in _SITES if not _calls_helper(by_path[site])]
    assert not missing, f"these sites no longer call attempt_upstream: {missing}"


def test_the_scan_sees_a_copy() -> None:
    """The guard reaches each spelling it claims to forbid."""
    assert _uses("x = attempted_lane(selection, route, model)") == [
        (1, "attempted_lane")
    ]
    assert _uses("errors.note_transport_refusal(p, rate_limit=None, lane=l)") == [
        (1, "note_transport_refusal")
    ]
    assert _uses("from ..quota.service import note_upstream_success") == [
        (1, "note_upstream_success")
    ]
    assert _uses("attempt.served()") == []
    assert _calls_helper("a = attempt_upstream(config, p, route=r, model=m)")
    assert not _calls_helper("a = other(config)")
