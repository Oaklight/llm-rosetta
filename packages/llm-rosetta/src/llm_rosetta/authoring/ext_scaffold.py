"""Create and check an extension folder.

An extension is the installable unit of a shim or a transport: one folder
``packages/<id>/`` that is itself a Python package, imported as
``llm_rosetta_ext.<id>``.  It holds ``plugin.yaml``, ``pyproject.toml``,
``README.md``, ``__init__.py``, the vendor manifest ``provider.yaml``, the
``llm/<format>/provider.yaml`` leaves, optional ``auth.py`` / ``quota.py`` /
``transforms.py`` sidecars, an optional ``transport/`` package and ``tests/``.

:func:`run_ext_new` writes that shape from the templates in ``_ext_template/``.
:func:`run_ext_check` validates one folder, or every ``packages/*`` under a
root, and reports what it could not check as *skipped*, never as passed.

The folder is the package, so a build writes ``build/`` and ``*.egg-info`` inside
it.  The wheel check therefore builds from a clean copy.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_rosetta.contracts import IR_VERSION, SURFACE_VERSION
from llm_rosetta.contracts.extension import (
    COMMANDS_ATTRIBUTE,
    COMMANDS_MODULE,
    MANIFEST_FILENAME,
    TRANSPORT_MODULE,
    TRANSPORTS_ATTRIBUTE,
    ExtensionManifest,
    ExtensionManifestInvalid,
    parse_extension_manifest,
)
from llm_rosetta.authoring.scaffold import _render
from llm_rosetta.extensions._cli import ExtVerb
from llm_rosetta.model_vocabulary import FORMATS_BY_TYPE

#: The template this module renders.
EXT_TEMPLATE_DIR = Path(__file__).with_name("_ext_template")

#: The namespace every extension is imported under.
NAMESPACE = "llm_rosetta_ext"

#: The shape of an extension id: the vendor key and the import name.
ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

#: A command verb: what follows ``llm-rosetta <id>``.
COMMAND_VERB = re.compile(r"^[a-z][a-z0-9-]*$")

#: Names a folder under ``packages/`` cannot take: they are the repository's own.
RESERVED_IDS = frozenset(
    {"scripts", "skills", "docs", "tests", "packages", "core", "contrib"}
)

#: The package-data globs a shim tree needs.  ``.py`` files in a package folder
#: ship by themselves; the leaves are not packages, so their files are listed.
PACKAGE_DATA = ("plugin.yaml", "provider.yaml", "llm/*/provider.yaml", "llm/*/*.py")

#: Files and folders that are in the folder but not in the wheel.
_NOT_SHIPPED_FILES = frozenset({"pyproject.toml", "README.md"})
_NOT_SHIPPED_DIRS = frozenset({"tests", "build", "dist", "__pycache__"})

#: Template files and the folders they land in, relative to the template.
_BASE_TEMPLATES = (
    ("plugin.yaml.tmpl", "plugin.yaml"),
    ("provider.yaml.tmpl", "provider.yaml"),
    ("__init__.py.tmpl", "__init__.py"),
    ("README.md.tmpl", "README.md"),
    ("tests/test_extension.py.tmpl", "tests/test_extension.py"),
)
_COMMAND_TEMPLATES = (("commands.py.tmpl", f"{COMMANDS_MODULE}.py"),)
_TRANSPORT_TEMPLATES = (
    ("transport/__init__.py.tmpl", f"{TRANSPORT_MODULE}/__init__.py"),
    ("transport/transport.py.tmpl", f"{TRANSPORT_MODULE}/transport.py"),
)


class ExtensionScaffoldError(ValueError):
    """The inputs of ``ext new`` cannot make an extension."""


def chat_formats() -> tuple[str, ...]:
    """Return the chat formats the catalog defines."""
    return tuple(FORMATS_BY_TYPE["llm"])


def id_problem(ext_id: str) -> str | None:
    """Say why *ext_id* cannot be an extension id, or return ``None``."""
    if not ID_PATTERN.match(ext_id):
        return (
            f"id {ext_id!r}: an id is lowercase letters, digits and underscores, "
            "and starts with a letter"
        )
    if ext_id in RESERVED_IDS:
        return f"id {ext_id!r} is reserved ({', '.join(sorted(RESERVED_IDS))})"
    return None


def distribution_name(ext_id: str) -> str:
    """Return the PyPI distribution name of extension *ext_id*."""
    return "llm-rosetta-ext-" + ext_id.replace("_", "-")


def _class_name(ext_id: str) -> str:
    return "".join(part.capitalize() for part in ext_id.split("_")) + "Transport"


def _core_floor() -> str:
    from llm_rosetta import __version__

    match = re.match(r"(\d+)\.(\d+)", __version__)
    return f"{match.group(1)}.{match.group(2)}.0" if match else __version__


def _quote(items: list[str]) -> str:
    return ", ".join(f'"{item}"' for item in items)


def render_extension(
    ext_id: str,
    *,
    format_: str,
    name: str | None = None,
    transport: bool = False,
    command: str | None = None,
) -> dict[str, str]:
    """Render the files of a new extension.

    Args:
        ext_id: The extension id.
        format_: A chat format from the catalog.
        name: The display name; the id by default.
        transport: Add a ``transport/`` package and declare it.
        command: Add a ``commands.py`` with this verb and declare it, so
            ``llm-rosetta <id> <verb>`` runs it.

    Returns:
        The file contents, keyed by path relative to the extension folder.

    Raises:
        ExtensionScaffoldError: The id or the format is not acceptable.
    """
    problem = id_problem(ext_id)
    if problem is not None:
        raise ExtensionScaffoldError(problem)
    if format_ not in chat_formats():
        raise ExtensionScaffoldError(
            f"format {format_!r} is not a chat format; available: "
            + ", ".join(chat_formats())
        )
    display = name or ext_id
    if re.search(r'["\\\x00-\x1f]', display):
        raise ExtensionScaffoldError(
            "name may not contain a quote, a backslash or a control character"
        )
    if command is not None and not COMMAND_VERB.match(command):
        raise ExtensionScaffoldError(
            f"command {command!r}: a verb is lowercase letters, digits and "
            "hyphens, and starts with a letter"
        )
    transport_id = ext_id if transport else ""
    shim = f"{ext_id}--{format_}"
    templates = list(_BASE_TEMPLATES) + [
        ("llm/format/provider.yaml.tmpl", f"llm/{format_}/provider.yaml")
    ]
    if transport:
        templates.extend(_TRANSPORT_TEMPLATES)
    if command is not None:
        templates.extend(_COMMAND_TEMPLATES)
    packages = [f"{NAMESPACE}.{ext_id}"]
    for _, target in templates:
        if target.endswith("/__init__.py") and not target.startswith("tests/"):
            packages.append(f"{NAMESPACE}.{ext_id}." + target[: -len("/__init__.py")])
    values = {
        "id": ext_id,
        "name": display,
        "name_yaml": f'"{display}"',
        "format": format_,
        "shim": shim,
        "surface": str(SURFACE_VERSION),
        "ir": str(IR_VERSION),
        "transport_id": transport_id,
        "command": command or "",
        "command_function": (command or "").replace("-", "_"),
        "command_summary": f"Run {ext_id} {command}." if command else "",
        "transport_class": _class_name(ext_id) if transport else "",
        "api_key_env": ext_id.upper() + "_API_KEY",
        "distribution": distribution_name(ext_id),
        "core_floor": _core_floor(),
        "packages": _quote(packages),
        "package_data": _quote(list(PACKAGE_DATA)),
    }
    files = {
        target: _render((EXT_TEMPLATE_DIR / source).read_text(encoding="utf-8"), values)
        for source, target in templates + [("pyproject.toml.tmpl", "pyproject.toml")]
    }
    return files


def run_ext_new(args: argparse.Namespace) -> int:
    """Create ``<dest>/<id>/`` from the extension template.

    Args:
        args: The parsed flags (see :func:`add_ext_arguments`).

    Returns:
        A process exit code: 0 when the folder was written.
    """
    ext_id = getattr(args, "ext_id", "") or ""
    try:
        files = render_extension(
            ext_id,
            format_=getattr(args, "format", "") or "chat_completions",
            name=getattr(args, "name", None),
            transport=bool(getattr(args, "transport", False)),
            command=getattr(args, "command", None),
        )
    except ExtensionScaffoldError as exc:
        print(f"error   {exc}", file=sys.stderr)
        return 2
    folder = Path(getattr(args, "dest", None) or "packages") / ext_id
    if folder.exists():
        print(f"error   {folder} already exists", file=sys.stderr)
        return 2
    for relative, text in files.items():
        path = folder / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote   {path}")
    print(f"\nnext: llm-rosetta ext check {folder}")
    return 0


def add_ext_arguments(parser: argparse.ArgumentParser, *, command: str = "new") -> None:
    """Register the flags of ``ext new`` or ``ext check`` on *parser*.

    Args:
        parser: The parser for the subcommand.
        command: ``"new"`` or ``"check"``.

    Raises:
        ValueError: *command* is neither.
    """
    if command == "new":
        parser.add_argument("ext_id", help="Extension id, e.g. together")
        parser.add_argument("--name", help="Display name; default the id")
        parser.add_argument(
            "--format",
            default="chat_completions",
            help="Chat format of the first lane. Available: "
            + ", ".join(chat_formats()),
        )
        parser.add_argument(
            "--dest",
            default="packages",
            help="Directory the extension folder is created in",
        )
        parser.add_argument(
            "--transport",
            action="store_true",
            help="Add a transport/ package and declare it in plugin.yaml",
        )
        parser.add_argument(
            "--command",
            metavar="VERB",
            help="Add commands.py with this verb and declare it, so "
            "'llm-rosetta <id> VERB' runs it",
        )
    elif command == "check":
        parser.add_argument(
            "path",
            nargs="?",
            default=".",
            help="An extension folder, or a root whose packages/* are checked",
        )
        parser.add_argument(
            "--no-build",
            dest="build",
            action="store_false",
            help="Skip the wheel build (reported as skipped)",
        )
    else:
        raise ValueError(f"unknown ext command {command!r}")


# ---------------------------------------------------------------------------
# ext check
# ---------------------------------------------------------------------------


@dataclass
class ExtCheckResult:
    """What checking one extension folder found.

    Attributes:
        problems: One sentence per defect; empty when the folder is sound.
        skipped: One sentence per check that did not run.
    """

    problems: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def _toml_loader() -> Any | None:
    for module in ("tomllib", "tomli"):
        if importlib.util.find_spec(module) is not None:
            return importlib.import_module(module)
    return None


def _read_manifest(folder: Path) -> ExtensionManifest:
    from llm_rosetta._vendor.yaml import load as yaml_load

    document = yaml_load((folder / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    return parse_extension_manifest(document)


def _tree_shims(folder: Path) -> set[str]:
    """Load the folder's shim tree off to the side and name what registered."""
    from llm_rosetta.contracts.shim import load_providers_from_dir
    from llm_rosetta.shims.provider_shim import staging_registry

    class _Discard(Exception):
        """Leaves the staging block so the live registry is not replaced."""

    names: set[str] = set()
    try:
        with staging_registry():
            loaded = load_providers_from_dir(
                folder.parent, only_paths=(folder.name,), on_error="raise"
            )
            names = {shim.name for shim in loaded}
            raise _Discard
    except _Discard:
        pass
    return names


def expected_wheel_files(folder: Path) -> set[str]:
    """List the wheel members the folder should produce.

    Everything except the packaging files, ``tests/`` and build leftovers.
    """
    files: set[str] = set()
    for path in folder.rglob("*"):
        relative = path.relative_to(folder)
        if not path.is_file() or relative.parts[0] in _NOT_SHIPPED_DIRS:
            continue
        if len(relative.parts) == 1 and relative.name in _NOT_SHIPPED_FILES:
            continue
        if "__pycache__" in relative.parts or relative.suffix == ".pyc":
            continue
        if any(part.endswith(".egg-info") for part in relative.parts):
            continue
        files.add(f"{NAMESPACE}/{folder.name}/{relative.as_posix()}")
    return files


def pip_available() -> bool:
    """Whether pip can be imported in this interpreter."""
    return importlib.util.find_spec("pip") is not None


def local_backend_available() -> bool:
    """Whether this interpreter can build a wheel without fetching a backend.

    Setuptools needs ``wheel`` before 70.1; from 70.1 it carries its own
    ``bdist_wheel`` command.
    """
    if importlib.util.find_spec("setuptools") is None:
        return False
    if importlib.util.find_spec("wheel") is not None:
        return True
    try:
        return importlib.util.find_spec("setuptools.command.bdist_wheel") is not None
    except Exception:  # noqa: BLE001 - a broken setuptools is not a backend
        return False


def _wheel_members(folder: Path) -> tuple[set[str] | None, str]:
    """Build a wheel from a clean copy of *folder* and list its members.

    Returns:
        The member names and an empty string, or ``None`` and the build error.
    """
    with tempfile.TemporaryDirectory(prefix="ext-check-") as work:
        source = Path(work) / "src" / folder.name
        shutil.copytree(
            folder,
            source,
            ignore=shutil.ignore_patterns(
                "build", "dist", "*.egg-info", "__pycache__", "*.pyc"
            ),
        )
        command = [sys.executable, "-m", "pip", "wheel", "--no-deps", "-q"]
        # An offline machine cannot fetch the build backend into an isolated
        # environment; use the one already installed when there is one.
        if local_backend_available():
            command.append("--no-build-isolation")
        out = Path(work) / "out"
        command += ["-w", str(out), str(source)]
        built = subprocess.run(command, capture_output=True, text=True)
        wheels = list(out.glob("*.whl")) if out.is_dir() else []
        if built.returncode != 0 or len(wheels) != 1:
            return None, (built.stderr.strip() or built.stdout.strip())[-400:]
        with zipfile.ZipFile(wheels[0]) as archive:
            return {n for n in archive.namelist() if n.startswith(NAMESPACE + "/")}, ""


def _check_pyproject(
    folder: Path, manifest: ExtensionManifest, result: ExtCheckResult
) -> None:
    path = folder / "pyproject.toml"
    if not path.is_file():
        result.problems.append("missing pyproject.toml")
        return
    loader = _toml_loader()
    if loader is None:
        result.skipped.append(
            "pyproject.toml content: no TOML parser (Python 3.10 without tomli)"
        )
        return
    try:
        document = loader.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        result.problems.append(f"pyproject.toml does not parse: {exc}")
        return
    project = document.get("project", {})
    want = distribution_name(manifest.id)
    if project.get("name") != want:
        result.problems.append(
            f"pyproject.toml project.name is {project.get('name')!r}, expected {want!r}"
        )
    if project.get("version") != manifest.version:
        result.problems.append(
            f"pyproject.toml project.version is {project.get('version')!r}, "
            f"plugin.yaml says {manifest.version!r}"
        )
    package = f"{NAMESPACE}.{manifest.id}"
    setuptools = document.get("tool", {}).get("setuptools", {})
    if setuptools.get("package-dir", {}).get(package) != ".":
        result.problems.append(
            f"pyproject.toml tool.setuptools.package-dir must map {package!r} to '.'"
        )
    if package not in setuptools.get("packages", []):
        result.problems.append(
            f"pyproject.toml tool.setuptools.packages must list {package!r}"
        )
    entry = project.get("entry-points", {}).get("llm_rosetta.extensions", {})
    if entry.get(manifest.id) != package:
        result.problems.append(
            f"pyproject.toml must declare the entry point "
            f"llm_rosetta.extensions: {manifest.id} = {package!r}"
        )


def _check_wheel(folder: Path, build: bool, result: ExtCheckResult) -> None:
    if not build:
        result.skipped.append("wheel contents: build not requested")
        return
    if not pip_available():
        result.skipped.append("wheel contents: pip is not available")
        return
    members, error = _wheel_members(folder)
    if members is None:
        result.problems.append(f"wheel build failed: {error}")
        return
    expected = expected_wheel_files(folder)
    for missing in sorted(expected - members):
        result.problems.append(f"wheel lacks {missing}")
    for extra in sorted(members - expected):
        result.problems.append(f"wheel ships {extra}, which is not in the folder")


def _literal_keys(tree: ast.Module, name: str) -> set[str] | None:
    """Return the string keys of the module-level dict literal *name*, if any."""
    keys: set[str] | None = None
    for node in tree.body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else []
        )
        value = getattr(node, "value", None)
        if any(isinstance(t, ast.Name) and t.id == name for t in targets) and (
            isinstance(value, ast.Dict)
        ):
            keys = {
                key.value
                for key in value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
    return keys


def _command_problems(folder: Path, declared: tuple[str, ...]) -> list[str]:
    """Check ``commands.py`` against ``provides.commands``.

    The dispatcher reads a ``COMMANDS`` mapping (verb to callable) from
    ``llm_rosetta_ext.<id>.commands`` when a verb runs.  The check is static,
    so it never imports the extension.
    """
    module = next(
        (
            path
            for path in (
                folder / f"{COMMANDS_MODULE}.py",
                folder / COMMANDS_MODULE / "__init__.py",
            )
            if path.is_file()
        ),
        None,
    )
    if module is None:
        return [
            f"provides.commands lists {', '.join(declared)} but the folder has "
            "no commands module"
        ]
    try:
        tree = ast.parse(module.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return [f"{module.name} does not parse: {exc}"]
    keys = _literal_keys(tree, COMMANDS_ATTRIBUTE)
    if keys is None:
        return [
            "the commands module has no COMMANDS mapping (a dict literal of verb "
            f"to callable); the dispatcher needs it to run {', '.join(declared)}"
        ]
    if keys != set(declared):
        return [
            f"COMMANDS has {sorted(keys)} but provides.commands lists "
            f"{sorted(declared)}"
        ]
    return []


def _transport_problems(folder: Path, declared: tuple[str, ...]) -> list[str]:
    """Check the transport module against ``provides.transports``.

    The gateway reads a ``TRANSPORTS`` mapping (transport id to factory) from
    ``llm_rosetta_ext.<id>.transport`` when it builds the lane.  The check is
    static, so it never imports the extension.
    """
    module = next(
        (
            path
            for path in (
                folder / TRANSPORT_MODULE / "__init__.py",
                folder / f"{TRANSPORT_MODULE}.py",
            )
            if path.is_file()
        ),
        None,
    )
    if module is None:
        return [
            f"provides.transports lists {', '.join(declared)} but the folder has "
            "no transport module or package"
        ]
    try:
        tree = ast.parse(module.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return [f"{module.name} does not parse: {exc}"]
    keys = _literal_keys(tree, TRANSPORTS_ATTRIBUTE)
    if keys is None:
        return [
            f"the transport module has no TRANSPORTS mapping (a dict literal of "
            f"transport id to factory); the gateway needs it to build "
            f"{', '.join(declared)}"
        ]
    if keys != set(declared):
        return [
            f"TRANSPORTS has {sorted(keys)} but provides.transports lists "
            f"{sorted(declared)}"
        ]
    return []


def check_extension(folder: Path, *, build: bool = True) -> ExtCheckResult:
    """Validate one extension folder.

    Args:
        folder: The extension folder, ``<root>/packages/<id>``.
        build: Build a wheel from a clean copy and compare its members with
            the folder.  The check is reported as skipped when pip is absent.

    Returns:
        The problems and the skipped checks.
    """
    result = ExtCheckResult()
    folder = Path(folder).resolve()
    problem = id_problem(folder.name)
    if problem is not None:
        result.problems.append(problem.replace("id ", "folder ", 1))
    if not (folder / MANIFEST_FILENAME).is_file():
        result.problems.append(f"no {MANIFEST_FILENAME}")
        return result
    try:
        manifest = _read_manifest(folder)
    except ExtensionManifestInvalid as exc:
        result.problems.extend(f"{MANIFEST_FILENAME}: {p}" for p in exc.problems)
        return result
    except Exception as exc:  # noqa: BLE001 - a malformed file is a finding
        result.problems.append(f"{MANIFEST_FILENAME} does not parse: {exc}")
        return result
    if manifest.id != folder.name:
        result.problems.append(
            f"folder is {folder.name!r} but {MANIFEST_FILENAME} declares id "
            f"{manifest.id!r}; they must be the same string"
        )
    for relative in ("README.md", "__init__.py"):
        if not (folder / relative).is_file():
            result.problems.append(f"missing {relative}")
    if not (folder / "tests").is_dir():
        result.problems.append("missing tests/")
    if manifest.transports:
        result.problems.extend(_transport_problems(folder, manifest.transports))
    if manifest.commands:
        result.problems.extend(_command_problems(folder, tuple(manifest.commands)))
    try:
        registered = _tree_shims(folder)
    except Exception as exc:  # noqa: BLE001 - name the loader's complaint
        result.problems.append(f"shim tree does not load: {type(exc).__name__}: {exc}")
    else:
        declared = set(manifest.shims)
        for name in sorted(declared - registered):
            result.problems.append(
                f"provides.shims lists {name!r} but the tree does not register it"
            )
        for name in sorted(registered - declared):
            result.problems.append(
                f"the tree registers {name!r} but provides.shims does not list it"
            )
    _check_pyproject(folder, manifest, result)
    _check_wheel(folder, build, result)
    return result


def _targets(path: Path) -> list[Path]:
    if (path / MANIFEST_FILENAME).is_file() or not (path / "packages").is_dir():
        return [path]
    return sorted(
        p
        for p in (path / "packages").iterdir()
        if p.is_dir() and not p.name.startswith((".", "__"))
    )


def run_ext_check(
    target: str | Path | argparse.Namespace, *, build: bool | None = None
) -> int:
    """Check an extension folder, or every ``packages/*`` under a root.

    Prints one line per problem and per skipped check.

    Args:
        target: A path, or the parsed flags of ``ext check``.
        build: Build a wheel to compare its contents; the flag's value, or
            true, by default.

    Returns:
        A process exit code: 0 when no problem was found.  A skipped check does
        not change the code and is printed apart from the problems.
    """
    if isinstance(target, argparse.Namespace):
        path = Path(getattr(target, "path", ".") or ".")
        if build is None:
            build = bool(getattr(target, "build", True))
    else:
        path = Path(target)
        build = True if build is None else build
    path = path.expanduser().resolve()
    failed = 0
    for folder in _targets(path):
        result = check_extension(folder, build=build)
        for line in result.problems:
            print(f"problem {folder.name}: {line}")
        for line in result.skipped:
            print(f"skipped {folder.name}: {line}")
        if result.problems:
            failed += 1
        else:
            print(f"ok      {folder.name}")
    return 1 if failed else 0


#: The ``ext`` actions this module supplies; ``llm_rosetta.cli`` hands them to
#: ``extensions._cli.ext_main``, so the extension package imports no dev tool.
EXT_VERBS: tuple[ExtVerb, ...] = (
    ExtVerb(
        "new",
        "Create the folder of a new extension",
        lambda parser: add_ext_arguments(parser, command="new"),
        run_ext_new,
    ),
    ExtVerb(
        "check",
        "Check an extension folder",
        lambda parser: add_ext_arguments(parser, command="check"),
        run_ext_check,
    ),
)
