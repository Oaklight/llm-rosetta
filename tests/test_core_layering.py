"""The conversion layers do not know the shim layer (#1706).

Core is a plain conversion library when no shim is present.  A shim is data a
provider hands in (:class:`~llm_rosetta.contracts.shim.ProviderShim`); reading
a shim *name* goes through the resolver in :mod:`llm_rosetta.contracts.resolver`,
which the shim layer registers.  Two checks hold that:

* a static scan: no conversion-layer module imports ``llm_rosetta.shims`` or
  ``llm_rosetta.extensions``, at module level or inside a function, and no
  ``contracts`` module imports them either;
* a run: with both packages blocked from import, ``import llm_rosetta`` still
  works and every chat format converts to every other.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import llm_rosetta

_CORE = Path(llm_rosetta.__file__).parent

#: The layers that convert.  ``contracts`` is checked on its own below.
CONVERSION_LAYERS = (
    "types",
    "ir",
    "converters",
    "pipeline",
    "formats",
    "fidelity",
    "profiling",
    "auto_detect.py",
    "tool_ops.py",
    "model_vocabulary.py",
    "conversion_warnings.py",
)

FORBIDDEN = ("llm_rosetta.shims", "llm_rosetta.extensions")


def _module_name(path: Path) -> list[str]:
    parts = ["llm_rosetta", *path.relative_to(_CORE).with_suffix("").parts]
    return parts[:-1] if parts[-1] == "__init__" else parts


def _imports(path: Path) -> list[tuple[int, str]]:
    """Every absolute module name *path* imports, with its line."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = _module_name(path)
    if path.name != "__init__.py":
        package = package[:-1]
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - (node.level - 1)]
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            found.append((node.lineno, module))
            found += [(node.lineno, f"{module}.{alias.name}") for alias in node.names]
    return found


def _files(entry: str) -> list[Path]:
    base = _CORE / entry
    return [base] if base.is_file() else sorted(base.rglob("*.py"))


def test_no_conversion_layer_imports_the_shim_or_extension_layer():
    offenders = [
        f"{path.relative_to(_CORE)}:{line} imports {name}"
        for entry in CONVERSION_LAYERS
        for path in _files(entry)
        for line, name in _imports(path)
        if name.startswith(FORBIDDEN)
    ]
    assert offenders == []


def test_no_contract_module_imports_the_shim_or_extension_layer():
    offenders = [
        f"{path.relative_to(_CORE)}:{line} imports {name}"
        for path in _files("contracts")
        for line, name in _imports(path)
        if name.startswith(FORBIDDEN)
    ]
    assert offenders == []


def test_the_extension_loader_imports_neither_the_shim_layer_nor_the_converters():
    """The extension loader manages packages; the shim layer registers a handler.

    The shim layer imports the extension loader to register its ``shims`` kind
    handler, never the other way round.  The ``ext new`` and ``ext check`` actions
    come from ``devtools`` as :class:`ExtVerb` records handed in by the CLI.
    """
    forbidden = (
        "llm_rosetta.shims",
        "llm_rosetta.converters",
        "llm_rosetta.pipeline",
        "llm_rosetta.api",
        "llm_rosetta.devtools",
    )
    offenders = [
        f"{path.relative_to(_CORE)}:{line} imports {name}"
        for path in _files("extensions")
        for line, name in _imports(path)
        if name.startswith(forbidden)
    ]
    assert offenders == []


#: What a contract module may import from the rest of core, besides other
#: contracts and the vendored helpers.  Each entry is data, never behaviour:
#: ``ir.model`` (the IR dataclasses and enums a declaration names),
#: ``formats`` (the catalog records), ``retired_names`` (the table of retired
#: spellings a declaration is refused with).  None of them imports a converter,
#: the pipeline, the shim layer or the extension layer.
CONTRACT_DATA_DEPENDENCIES = (
    "llm_rosetta.ir.model",
    "llm_rosetta.formats",
    "llm_rosetta.retired_names",
)

#: The one contract module that depends on IR behaviour: the transform contract
#: re-exports the primitives shims use.
CONTRACT_BEHAVIOUR_EDGES = {"contracts/transform.py": ("llm_rosetta.ir.primitives",)}


def test_a_contract_imports_only_data_beyond_the_transform_contract():
    """The declaration layer is a leaf: ``routing`` no longer reads ``auto_detect``.

    ``ProviderType`` is defined in ``contracts.routing`` and ``auto_detect``
    imports it, so the dependency runs from the conversion layer to the contract.
    """
    offenders = []
    for path in _files("contracts"):
        relative = path.relative_to(_CORE).as_posix()
        allowed = (
            "llm_rosetta.contracts",
            "llm_rosetta._vendor",
            *CONTRACT_DATA_DEPENDENCIES,
            *CONTRACT_BEHAVIOUR_EDGES.get(relative, ()),
        )
        for line, name in _imports(path):
            if name == "llm_rosetta" or name.startswith(allowed):
                continue
            if name.startswith("llm_rosetta."):
                offenders.append(f"{relative}:{line} imports {name}")
    assert offenders == []


def test_no_converter_imports_the_pipeline():
    """The converters sit below the pipeline: nothing in them imports it."""
    offenders = [
        f"{path.relative_to(_CORE)}:{line} imports {name}"
        for path in _files("converters")
        for line, name in _imports(path)
        if name == "llm_rosetta.pipeline" or name.startswith("llm_rosetta.pipeline.")
    ]
    assert offenders == []


def test_the_pipeline_takes_conversion_context_from_the_ir():
    """``ConversionContext`` lives in ``ir.context``; the old converters path only forwards it."""
    offenders = [
        f"{path.relative_to(_CORE)}:{line} imports {name}"
        for path in _files("pipeline")
        for line, name in _imports(path)
        if name.startswith("llm_rosetta.converters.base.context")
    ]
    assert offenders == []


def test_conversion_error_is_one_class_under_every_name():
    """``except`` and ``isinstance`` must agree whichever name a caller imported."""
    import llm_rosetta
    import llm_rosetta.api
    import llm_rosetta.pipeline
    import llm_rosetta.pipeline.chat
    from llm_rosetta.converters.base.errors import ConversionError

    for owner in (
        llm_rosetta,
        llm_rosetta.api,
        llm_rosetta.pipeline,
        llm_rosetta.pipeline.chat,
    ):
        assert owner.ConversionError is ConversionError, owner.__name__


_PURE_MODE = textwrap.dedent(
    """
    import sys

    # ``-I`` ignores PYTHONPATH, so name the source tree that is under test.
    sys.path.insert(0, {src!r})

    # A ``None`` entry makes the import of that name raise ImportError.
    for name in ("llm_rosetta.shims", "llm_rosetta.extensions"):
        sys.modules[name] = None

    import llm_rosetta
    from llm_rosetta import convert
    from llm_rosetta.contracts.resolver import ShimResolverUnavailable, lookup_shim

    assert not hasattr(llm_rosetta, "get_shim"), "the shim layer was not blocked"

    FORMATS = (
        "chat_completions",
        "open_responses",
        "messages",
        "google_generate",
        "google_interactions",
    )
    seed = {{
        "model": "m",
        "messages": [{{"role": "user", "content": "hello"}}],
        "max_tokens": 16,
    }}
    bodies = {{"chat_completions": seed}}
    for fmt in FORMATS:
        if fmt not in bodies:
            bodies[fmt] = convert(seed, fmt, "chat_completions", baseline=False)
    done = 0
    for source in FORMATS:
        for target in FORMATS:
            out = convert(bodies[source], target, source, baseline=False)
            assert isinstance(out, dict) and out, (source, target)
            done += 1
    assert done == len(FORMATS) ** 2

    try:
        lookup_shim("some_vendor--chat_completions")
    except ShimResolverUnavailable:
        pass
    else:
        raise SystemExit("a shim name resolved with no shim layer")
    print("pure mode ok", done)
    """
)


def test_every_converter_pair_converts_without_the_shim_layer():
    # Run from the core's parent so the stdlib ``types`` is not shadowed by
    # the package's own ``types`` directory.
    result = subprocess.run(
        [sys.executable, "-I", "-c", _PURE_MODE.format(src=str(_CORE.parent))],
        capture_output=True,
        text=True,
        cwd=str(_CORE.parents[2]),
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "pure mode ok 25" in result.stdout


# ---------------------------------------------------------------------------
# The edge table
# ---------------------------------------------------------------------------

#: Every import from one top-level unit of ``llm_rosetta`` (a package or a root
#: module; ``<root>`` is ``llm_rosetta/__init__.py``) into another, module level
#: and function-local alike, ``_vendor`` left out.  The layers, bottom to top:
#:
#: * foundation: ``types``, ``ir``, ``formats``, ``retired_names``,
#:   ``conversion_warnings``, ``model_vocabulary``;
#: * ``contracts``: what an outside party implements or receives;
#: * conversion: ``converters``, ``tool_ops``, ``auto_detect``, ``pipeline``,
#:   ``fidelity``, ``profiling``;
#: * optional layers: ``shims``, ``extensions``, ``routing``, ``devtools``,
#:   ``testing``;
#: * entry: ``cli``, ``api``, ``capabilities`` (the 0.13.0 forwarder, imported by
#:   nothing), ``<root>``, ``__main__``.
#:
#: The table only records the edges that exist.  A new edge fails the first test
#: below until its author adds it here, where a reviewer sees it; an edge that
#: has gone fails the second until it is deleted, so a removed edge stays
#: removed.  The pairs that point both ways (``ir`` and ``contracts``;
#: ``formats`` and ``contracts``, the latter only lazily) are the ones left to
#: untie.
CORE_EDGES: dict[str, tuple[str, ...]] = {
    "<root>": (
        "auto_detect",
        "contracts",
        "converters",
        "ir",
        "pipeline",
        "shims",
        "tool_ops",
    ),
    "__main__": ("cli",),
    "api": (
        "cli",
        "contracts",
        "converters",
        "extensions",
        "fidelity",
        "formats",
        "ir",
        "model_vocabulary",
        "pipeline",
        "profiling",
        "retired_names",
        "shims",
    ),
    "auto_detect": (
        "contracts",
        "converters",
        "formats",
        "pipeline",
        "retired_names",
    ),
    "capabilities": (
        "contracts",
        "converters",
        "ir",
        "pipeline",
    ),
    "cli": (
        "cli_areas",
        "contracts",
        "devtools",
        "extensions",
    ),
    "cli_areas": (
        "contracts",
        "extensions",
    ),
    "contracts": (
        "formats",
        "ir",
        "retired_names",
    ),
    "converters": (
        "contracts",
        "conversion_warnings",
        "ir",
        "model_vocabulary",
        "retired_names",
    ),
    "devtools": (
        "contracts",
        "converters",
        "extensions",
        "model_vocabulary",
        "shims",
    ),
    "extensions": ("contracts",),
    "fidelity": ("retired_names",),
    "formats": ("ir",),
    "ir": ("conversion_warnings",),
    "model_vocabulary": ("formats",),
    "pipeline": (
        "contracts",
        "conversion_warnings",
        "converters",
        "fidelity",
        "ir",
        "model_vocabulary",
        "retired_names",
    ),
    "routing": ("contracts",),
    "shims": (
        "auto_detect",
        "contracts",
        "extensions",
        "formats",
        "ir",
        "model_vocabulary",
        "retired_names",
    ),
    "testing": ("shims",),
    "tool_ops": (
        "converters",
        "ir",
        "retired_names",
    ),
    "types": ("ir",),
}


_SKIPPED_PARTS = ("_vendor", "__pycache__")


def _unit_of(path: Path) -> str:
    first = path.relative_to(_CORE).parts[0]
    return "<root>" if first == "__init__.py" else first.removesuffix(".py")


def _core_units() -> set[str]:
    return {
        entry.name.removesuffix(".py")
        for entry in _CORE.iterdir()
        if (entry.is_dir() or entry.suffix == ".py")
        and entry.name not in _SKIPPED_PARTS
        and entry.name != "__init__.py"
    }


def _actual_edges() -> dict[tuple[str, str], str]:
    """Every unit-to-unit import, with one ``file:line`` that shows it."""
    units = _core_units()
    found: dict[tuple[str, str], str] = {}
    for path in sorted(_CORE.rglob("*.py")):
        if any(part in _SKIPPED_PARTS for part in path.parts):
            continue
        source = _unit_of(path)
        for line, name in _imports(path):
            parts = name.split(".")
            if parts[0] != "llm_rosetta" or len(parts) < 2 or parts[1] not in units:
                continue
            if parts[1] != source:
                found.setdefault(
                    (source, parts[1]), f"{path.relative_to(_CORE)}:{line}"
                )
    return found


#: The edges that exist only as function-local imports, each with why it is not
#: at the top of its module.  ``CORE_EDGES`` lists every edge, eager or lazy; this
#: table names the ones a reader of ``CORE_EDGES`` cannot see are lazy.  A new
#: lazy-only edge must be listed with its reason; an edge that becomes eager (or
#: goes) must leave it.
LAZY_ONLY_EDGES: dict[tuple[str, str], str] = {
    ("auto_detect", "converters"): "a converter is built on the first call",
    ("auto_detect", "pipeline"): "convert() builds its pipeline on the first call",
    ("cli", "cli_areas"): "a command loads only the code it runs",
    ("cli", "devtools"): "a command loads only the code it runs",
    ("cli", "extensions"): "a command loads only the code it runs",
    (
        "devtools",
        "converters",
    ): "read when a command runs; a partial install still starts",
    ("devtools", "shims"): "the scaffold commands load the shim loader when they run",
    ("pipeline", "fidelity"): "the checker is built only when a fidelity mode is set",
    ("shims", "auto_detect"): "the registry clears the converter cache when it is "
    "rebuilt or restored",
}


def _imported_names(node: ast.AST, package: list[str]) -> list[str]:
    """The absolute module names one import statement names (empty otherwise)."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if not isinstance(node, ast.ImportFrom):
        return []
    if node.level:
        base = package[: len(package) - (node.level - 1)]
        module = ".".join(base + ([node.module] if node.module else []))
    else:
        module = node.module or ""
    return [module] + [f"{module}.{alias.name}" for alias in node.names]


def _top_level_imports(tree: ast.AST, package: list[str]) -> list[str]:
    """Names imported outside any function, however deeply nested in ``if``/``try``."""
    names: list[str] = []
    pending = list(ast.iter_child_nodes(tree))
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        names += _imported_names(node, package)
        pending += list(ast.iter_child_nodes(node))
    return names


def _eager_edges() -> set[tuple[str, str]]:
    """Unit-to-unit imports that run when their module is imported.

    An import inside a function does not count.
    """
    units = _core_units()
    found: set[tuple[str, str]] = set()
    for path in sorted(_CORE.rglob("*.py")):
        if any(part in _SKIPPED_PARTS for part in path.parts):
            continue
        package = _module_name(path)
        if path.name != "__init__.py":
            package = package[:-1]
        tree = ast.parse(path.read_text(encoding="utf-8"))
        source = _unit_of(path)
        for name in _top_level_imports(tree, package):
            parts = name.split(".")
            if (
                parts[0] == "llm_rosetta"
                and len(parts) > 1
                and parts[1] in units
                and parts[1] != source
            ):
                found.add((source, parts[1]))
    return found


def _lazy_only_edges() -> set[tuple[str, str]]:
    return set(_actual_edges()) - _eager_edges()


def test_every_edge_that_exists_only_inside_functions_is_listed_with_its_reason():
    new = sorted(_lazy_only_edges() - set(LAZY_ONLY_EDGES))
    assert new == [], (
        "these edges are imported only inside functions; import them at the top of "
        "the module, or add them to LAZY_ONLY_EDGES in tests/test_core_layering.py "
        f"with the reason: {new}"
    )


def test_the_lazy_only_table_lists_no_edge_that_is_eager_or_gone():
    stale = sorted(set(LAZY_ONLY_EDGES) - _lazy_only_edges())
    assert stale == [], f"delete these rows from LAZY_ONLY_EDGES: {stale}"


def test_no_core_package_gains_an_import_the_table_does_not_list():
    allowed = {(src, dst) for src, dsts in CORE_EDGES.items() for dst in dsts}
    new = [
        f"{src} -> {dst} ({where})"
        for (src, dst), where in sorted(_actual_edges().items())
        if (src, dst) not in allowed
    ]
    assert new == [], (
        "a new package-level import; if it is meant, add it to CORE_EDGES in "
        "tests/test_core_layering.py so a reviewer sees it"
    )


def test_the_table_lists_no_import_that_has_gone():
    actual = set(_actual_edges())
    gone = [
        f"{src} -> {dst}"
        for src, dsts in CORE_EDGES.items()
        for dst in dsts
        if (src, dst) not in actual
    ]
    assert gone == [], "delete these rows from CORE_EDGES: the import is gone"
