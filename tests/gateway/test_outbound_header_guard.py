"""The address an upstream sees is the gateway host's own (#1830 A12).

A client's address, and the forwarding headers that carry it
(``X-Forwarded-For``, ``X-Real-IP``, ``Forwarded`` and the rest), must never
reach an upstream provider.  Today that holds because
``middleware/headers.py::build_upstream_extra_headers`` is an allowlist and
every upstream call takes its ``extra_headers`` from it.  Nothing checked it, so
a new route that built its own dict, or a new line in the allowlist, would
leak the client with no test red.

Guards:

* only ``middleware/request_context.py`` names a forwarding or client-address
  header (a string constant equal to one of them);
* every ``extra_headers=`` argument in the gateway source is the result of
  ``build_upstream_extra_headers``, a local assigned from it, or a parameter
  of the enclosing function (a value passed through a layer).  The calls that
  are not upstream request headers sit on ``EXTRA_HEADERS_EXCEPTIONS``, keyed
  by file, enclosing function and callee; the list may only shrink, so a new
  call fails and so does an entry whose call has gone;
* ``build_upstream_extra_headers`` drops all of them for an inbound request
  that carries them.

The scan reads keyword arguments only.  A value passed positionally is not
seen; the helpers in ``proxy/`` take ``extra_headers`` as a keyword.  The rule
is on the source, not on the address a socket reports: how the gateway reads
the client address is ``request_context.py``'s concern.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import llm_rosetta_gateway.middleware.headers as headers_module
from llm_rosetta_gateway.middleware.headers import build_upstream_extra_headers

_GATEWAY = "llm_rosetta_gateway"
_ROOT = Path(__file__).parents[2] / "packages/llm-rosetta-gateway/src" / _GATEWAY

_BUILDER = "build_upstream_extra_headers"
_BUILDER_FILE = "middleware/headers.py"
_BUILDER_MODULE = f"{_GATEWAY}.middleware.headers"

#: The one module that may read a forwarding header: it decides, from the
#: socket peer and the operator's trusted list, whether to believe it.
_READER = "middleware/request_context.py"

#: Header names that carry a client's address or its connection.  A string
#: constant equal to one of these, in any case, is a read or a write of it.
FORWARDING_HEADERS = frozenset(
    {
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-port",
        "x-forwarded-proto",
        "x-forwarded-server",
        "x-real-ip",
        "forwarded",
        "x-client-ip",
        "true-client-ip",
        "cf-connecting-ip",
    }
)

#: ``extra_headers=`` arguments that are not upstream request headers, as
#: ``(file, enclosing function, callee)``.  Never add to it: route the value
#: through ``build_upstream_extra_headers`` instead.
EXTRA_HEADERS_EXCEPTIONS: frozenset[tuple[str, str, str]] = frozenset(
    {
        # The headers of the gateway's own 429 response to its client.
        ("middleware/ratelimit.py", "_rate_limit_response", "format_error_response"),
        # A shim's declared connection headers, fixed config and no request.
        ("config/providers.py", "build_provider_info", "ProviderInfo"),
    }
)


def _files() -> list[Path]:
    return sorted(p for p in _ROOT.rglob("*.py") if "_vendor" not in p.parts)


def _rel(path: Path) -> str:
    return path.relative_to(_ROOT).as_posix()


def _name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _is_builder_call(node: ast.expr) -> bool:
    return isinstance(node, ast.Call) and _name(node.func) == _BUILDER


def _forwarding_constants(tree: ast.AST) -> list[tuple[int, str]]:
    return [
        (n.lineno, n.value)
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and n.value.lower() in FORWARDING_HEADERS
    ]


def _builder_imports(tree: ast.AST) -> bool:
    """Whether the file imports the builder from its own module."""
    return any(
        isinstance(n, ast.ImportFrom)
        and (n.module or "").endswith("middleware.headers")
        and any(a.name == _BUILDER for a in n.names)
        for n in ast.walk(tree)
    )


class _Scan(ast.NodeVisitor):
    """Collect every ``extra_headers=`` argument that is not allowed."""

    def __init__(self) -> None:
        self.stack: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        self.found: list[tuple[str, str, int]] = []  # (function, callee, line)
        self.sites: list[tuple[str, str]] = []  # every (function, callee)
        self.builder_calls = 0

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.stack.append(node)
        self.generic_visit(node)
        self.stack.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def _params(self) -> set[str]:
        names: set[str] = set()
        for fn in self.stack:
            a = fn.args
            for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs):
                names.add(arg.arg)
        return names

    def _from_builder(self, name: str) -> bool:
        """Whether a function in scope assigns ``name`` from the builder."""
        for fn in self.stack:
            for n in ast.walk(fn):
                if isinstance(n, ast.Assign) and _is_builder_call(n.value):
                    if any(_name(t) == name for t in n.targets):
                        return True
                if isinstance(n, ast.AnnAssign) and n.value is not None:
                    if _is_builder_call(n.value) and _name(n.target) == name:
                        return True
        return False

    def visit_Call(self, node: ast.Call) -> None:
        if _is_builder_call(node):
            self.builder_calls += 1
        function = self.stack[-1].name if self.stack else "<module>"
        for kw in node.keywords:
            if kw.arg != "extra_headers":
                continue
            callee = _name(node.func)
            self.sites.append((function, callee))
            value = kw.value
            ok = _is_builder_call(value) or (
                isinstance(value, ast.Name)
                and (value.id in self._params() or self._from_builder(value.id))
            )
            if not ok:
                self.found.append((function, callee, kw.value.lineno))
        self.generic_visit(node)


def _scan(source: str) -> _Scan:
    scan = _Scan()
    scan.visit(ast.parse(source))
    return scan


def _offences() -> tuple[list[tuple[str, str, str, int]], set[tuple[str, str, str]]]:
    """Return the unallowed arguments and every exception key that occurs."""
    offences: list[tuple[str, str, str, int]] = []
    seen: set[tuple[str, str, str]] = set()
    for path in _files():
        scan = _scan(path.read_text(encoding="utf-8"))
        rel = _rel(path)
        for function, callee, line in scan.found:
            key = (rel, function, callee)
            if key in EXTRA_HEADERS_EXCEPTIONS:
                seen.add(key)
            else:
                offences.append((rel, function, callee, line))
    return offences, seen


# --- the product's own function --------------------------------------------


def test_the_guarded_builder_is_the_products_own_function():
    assert build_upstream_extra_headers is headers_module.build_upstream_extra_headers
    assert build_upstream_extra_headers.__module__ == _BUILDER_MODULE
    assert (_ROOT / _BUILDER_FILE).is_file()
    tree = ast.parse((_ROOT / _BUILDER_FILE).read_text(encoding="utf-8"))
    assert any(isinstance(n, ast.FunctionDef) and n.name == _BUILDER for n in tree.body)


def test_every_file_that_calls_the_builder_imports_it_from_that_module():
    callers = []
    for path in _files():
        source = path.read_text(encoding="utf-8")
        if _scan(source).builder_calls:
            callers.append(_rel(path))
            assert _builder_imports(ast.parse(source)), _rel(path)
    # The four upstream-facing entry points the invariant was drawn over.
    assert {
        "app/routes_llm.py",
        "app/routes_compact.py",
        "app/routes_count_tokens.py",
        "pipelines/typed.py",
    } <= set(callers)


# --- rule 1: only request_context names a forwarding header -----------------


def test_only_request_context_names_a_forwarding_header():
    bad = []
    for path in _files():
        if _rel(path) == _READER:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        bad += [(_rel(path), line, text) for line, text in _forwarding_constants(tree)]
    assert not bad, (
        f"a forwarding / client-address header is named outside {_READER}: {bad}"
    )


def test_request_context_does_read_the_forwarding_headers():
    # The scan can reach the one reader: a guard that sees nothing proves nothing.
    tree = ast.parse((_ROOT / _READER).read_text(encoding="utf-8"))
    names = {text.lower() for _line, text in _forwarding_constants(tree)}
    assert {"x-forwarded-for", "x-real-ip", "x-forwarded-proto"} <= names


# --- rule 2: extra_headers come from the builder ---------------------------


def test_every_extra_headers_argument_comes_from_the_builder():
    offences, _seen = _offences()
    assert not offences, (
        "extra_headers= must be build_upstream_extra_headers(...), a local "
        f"assigned from it, or a parameter: {offences}"
    )


def test_the_extra_headers_exception_list_names_only_calls_that_exist():
    _offences_found, seen = _offences()
    stale = EXTRA_HEADERS_EXCEPTIONS - seen
    assert not stale, f"exceptions whose call has gone: {sorted(stale)}"


def test_the_scan_sees_the_upstream_call_sites():
    files = {
        _rel(path) for path in _files() if _scan(path.read_text(encoding="utf-8")).sites
    }
    # Pass-through parameters in the engine and a builder result in a route.
    assert {"proxy/engine.py", "app/routes_llm.py", "pipelines/typed.py"} <= files


def test_the_scan_flags_a_bypassing_literal_and_accepts_the_allowed_forms():
    bypass = _scan(
        "def route(request):\n"
        "    return send(extra_headers={'X-Forwarded-For': request.client})\n"
    )
    assert [f[:2] for f in bypass.found] == [("route", "send")]
    local = _scan(
        "def route(request):\n    h = {'a': 'b'}\n    return send(extra_headers=h)\n"
    )
    assert [f[:2] for f in local.found] == [("route", "send")]
    allowed = _scan(
        "def route(request, extra_headers):\n"
        "    hh = build_upstream_extra_headers(request, 'id')\n"
        "    send(extra_headers=hh)\n"
        "    send(extra_headers=build_upstream_extra_headers(request, 'id'))\n"
        "    return send(extra_headers=extra_headers)\n"
    )
    assert allowed.found == []
    assert len(allowed.sites) == 3


# --- the builder itself -----------------------------------------------------


class _Request:
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers


_CLIENT_HEADERS = {
    "X-Forwarded-For": "203.0.113.7, 10.0.0.1",
    "x-forwarded-for": "203.0.113.7, 10.0.0.1",
    "X-Real-IP": "203.0.113.7",
    "x-real-ip": "203.0.113.7",
    "Forwarded": "for=203.0.113.7;proto=https",
    "forwarded": "for=203.0.113.7;proto=https",
    "X-Forwarded-Proto": "https",
    "x-forwarded-proto": "https",
}


def test_forwarding_headers_never_reach_the_upstream_headers():
    request: Any = _Request(
        {**_CLIENT_HEADERS, "user-agent": "agent/1", "x-request-id": "r1"}
    )
    result = build_upstream_extra_headers(request, "r1", session_header="x-session")
    assert not {k.lower() for k in result} & FORWARDING_HEADERS
    assert not any("203.0.113.7" in v for v in result.values())
    # The allowlist is exactly what the docstring says, nothing else rides.
    assert result == {
        "x-request-id": "r1",
        "x-rosetta-hop-count": "1",
        "User-Agent": "agent/1",
    }
