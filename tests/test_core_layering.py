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
