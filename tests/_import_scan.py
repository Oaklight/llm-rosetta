"""Shared AST scan for the import-boundary tests.

Every boundary test answers the same question ("which imports of package X
does this tree contain?") and used to carry its own copy of the scan.  This
module is the one copy.  It sees an import at any depth: module scope, inside
a function, behind ``TYPE_CHECKING``, absolute, relative, and dynamic
(``importlib.import_module`` / ``__import__``).

A dynamic import whose argument is not a string literal cannot be attributed
to a package, so it is reported as its own kind
(:data:`DYNAMIC_NONLITERAL`) instead of being dropped; a caller decides
whether that is an offence.

Not a test module: pytest does not collect it (no ``test_`` prefix).
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

IMPORT = "import"
FROM = "from"
DYNAMIC = "dynamic"
DYNAMIC_NONLITERAL = "dynamic-nonliteral"

#: Contract packages whose submodules are internal (#1657): a host or a shim
#: imports the package, never one of its submodules.  Shared by the gateway
#: and the shim-sidecar boundary tests so the rule is stated once.
CONTRACT_PACKAGE_ONLY = frozenset(("llm_rosetta.contracts.usage",))


@dataclass(frozen=True)
class ImportSite:
    """One import statement or dynamic import call found in a source file.

    Attributes:
        lineno: Line of the statement or call.
        kind: One of ``import``, ``from``, ``dynamic``, ``dynamic-nonliteral``.
        module: Absolute module name.  For ``from`` it is the module after
            ``from``; for ``dynamic`` the literal module string; empty for
            ``import`` (the modules are in ``names``) and for a non-literal
            dynamic import.
        names: For ``from`` the imported names (``*`` included); for
            ``import`` the imported modules.
        function: The qualified name of the innermost function the import
            sits in -- ``ClassName.method`` for a method, ``"<lambda>"`` for a
            lambda body, empty at module scope -- so two imports of one module
            in two functions, or in a ``ready`` method of two classes, differ
            here.  A caller can then check that no *new* site appeared in a
            file it already knows.
        type_checking: The import sits inside an ``if TYPE_CHECKING:`` block.
    """

    lineno: int
    kind: str
    module: str
    names: tuple[str, ...] = ()
    function: str = ""
    type_checking: bool = False

    @property
    def function_level(self) -> bool:
        """The import sits inside a function body."""
        return bool(self.function)

    def targets(self) -> Iterator[str]:
        """Yield every module path the statement may refer to, in match order.

        A ``from pkg import name`` may import the submodule ``pkg.name``, so
        both ``pkg`` and ``pkg.name`` are yielded (``*`` excluded).
        """
        if self.kind == FROM:
            yield self.module
            for name in self.names:
                if name != "*":
                    yield f"{self.module}.{name}" if self.module else name
        elif self.kind == IMPORT:
            yield from self.names
        elif self.module:
            yield self.module


def absolute_module(package: tuple[str, ...], node: ast.ImportFrom) -> str:
    """Resolve an ``ImportFrom`` to its absolute module name.

    Args:
        package: The importing file's package as dotted parts, for example
            ``("llm_rosetta", "ir", "primitives")``.  An ``__init__.py`` and a
            plain module in the same directory share one ``package``.
        node: The ``from ... import`` node.
    """
    if not node.level:
        return node.module or ""
    base = list(package[: len(package) - (node.level - 1)])
    if node.module:
        base += node.module.split(".")
    return ".".join(base)


def is_dynamic_import(node: ast.Call) -> bool:
    """Return True for ``__import__(...)`` and ``<x>.import_module(...)``."""
    return (isinstance(node.func, ast.Name) and node.func.id == "__import__") or (
        isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"
    )


def dynamic_target(node: ast.Call) -> str | None:
    """Return the literal module string of a dynamic import, else None."""
    if not is_dynamic_import(node) or not node.args:
        return None
    arg = node.args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    return None


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _flag_nodes(tree: ast.AST) -> tuple[dict[int, str], set[int]]:
    """Return the innermost enclosing function of every node, and the TYPE_CHECKING ids.

    The function map holds the qualified name of the ``def`` (or
    ``"<lambda>"``) a node sits in -- ``ClassName.method`` for a method, so two
    classes that each define ``ready`` are two sites.  A node at module scope
    is absent.
    """
    in_function: dict[int, str] = {}
    in_type_checking: set[int] = set()

    def mark(node: ast.AST, function: str, scope: str, checking: bool) -> None:
        for child in ast.iter_child_nodes(node):
            child_scope = scope
            if isinstance(child, ast.ClassDef):
                child_scope = f"{scope}.{child.name}" if scope else child.name
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                child_function = (
                    f"{child_scope}.{child.name}" if child_scope else child.name
                )
            elif isinstance(child, ast.Lambda):
                child_function = (
                    f"{child_scope}.<lambda>" if child_scope else "<lambda>"
                )
            else:
                child_function = function
            child_checking = checking or (
                isinstance(node, ast.If)
                and _is_type_checking(node.test)
                and child in node.body
            )
            if child_function:
                in_function[id(child)] = child_function
            if child_checking:
                in_type_checking.add(id(child))
            mark(child, child_function, child_scope, child_checking)

    mark(tree, "", "", False)
    return in_function, in_type_checking


def import_sites(source: str, package: tuple[str, ...]) -> list[ImportSite]:
    """Return every import in *source*, in ``ast.walk`` order.

    Args:
        source: Python source text.
        package: The file's package as dotted parts (see
            :func:`absolute_module`).
    """
    tree = ast.parse(source)
    in_function, in_type_checking = _flag_nodes(tree)
    sites: list[ImportSite] = []
    for node in ast.walk(tree):
        function = in_function.get(id(node), "")
        type_checking = id(node) in in_type_checking
        if isinstance(node, ast.ImportFrom):
            names = tuple(alias.name for alias in node.names)
            sites.append(
                ImportSite(
                    node.lineno,
                    FROM,
                    absolute_module(package, node),
                    names,
                    function=function,
                    type_checking=type_checking,
                )
            )
        elif isinstance(node, ast.Import):
            names = tuple(alias.name for alias in node.names)
            sites.append(
                ImportSite(
                    node.lineno,
                    IMPORT,
                    "",
                    names,
                    function=function,
                    type_checking=type_checking,
                )
            )
        elif isinstance(node, ast.Call) and is_dynamic_import(node) and node.args:
            target = dynamic_target(node)
            if target is None:
                sites.append(
                    ImportSite(
                        node.lineno,
                        DYNAMIC_NONLITERAL,
                        "",
                        function=function,
                        type_checking=type_checking,
                    )
                )
            else:
                sites.append(
                    ImportSite(
                        node.lineno,
                        DYNAMIC,
                        target,
                        function=function,
                        type_checking=type_checking,
                    )
                )
    return sites


def package_parts(path: Path, root: Path, root_package: str) -> tuple[str, ...]:
    """Return the dotted package of *path*, a file under the tree at *root*."""
    return (root_package, *path.relative_to(root).parent.parts)


def import_hits(
    path: Path,
    source: str,
    package: tuple[str, ...],
    matches: Callable[[str], bool],
) -> list[str]:
    """Return ``path:line module`` for each import that reaches a match.

    A statement is reported once: for the module itself when it matches,
    otherwise for the first imported name whose qualified path matches.
    A dynamic import is reported as ``path:line <dynamic> module``.  A
    non-literal dynamic import is not attributable and is not reported.

    Args:
        path: File the source came from (only used in the report).
        source: Python source text.
        package: The file's package as dotted parts.
        matches: Predicate over an absolute module path.
    """
    hits: list[str] = []
    for site in import_sites(source, package):
        if site.kind == DYNAMIC_NONLITERAL:
            continue
        for target in site.targets():
            if matches(target):
                label = "<dynamic> " if site.kind == DYNAMIC else ""
                hits.append(f"{path}:{site.lineno} {label}{target}")
                break
    return hits
