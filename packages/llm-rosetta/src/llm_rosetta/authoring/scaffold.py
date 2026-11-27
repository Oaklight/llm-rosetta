"""The shim tree generator: the files of a shim, written from the template.

A shim is a small, boring set of files — a ``provider.yaml``, sometimes a
``transforms.py``, and often a vendor manifest above them — and getting the
shape wrong is not caught until a gateway loads it.  This module holds what both
authoring commands share: the templates (``_shim_template/tree``), the layout
rules (a lane's directory, the move a second lane forces), the renderers of a
vendor manifest and a leaf (:func:`vendor_manifest_text`, :func:`leaf_text`) and
:func:`scaffold`, which writes a leaf under a root and loads it with the real
loader.  ``llm-rosetta ext new`` renders the shims kind of an extension folder
from the same functions; ``scripts/dev/new_shim.py`` adds the prompts, the base
URL probe and the icon lookup for adding a vendor to the built-in tree, which
need no place in an installed package.

Layouts::

    typed (default)   <root>/<vendor>/llm/<format>/provider.yaml
                      <root>/<vendor>/<type>/provider.yaml      (non-chat)
    flat              <root>/<vendor>/<format>/provider.yaml      (argo, alcf)

``<root>`` is the directory a gateway scans: the built-in ``providers`` tree.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from llm_rosetta.contracts import AUTH_MODULE, QUOTA_MODULE, TRANSFORMS_MODULE

from .tree_contract import TYPE_DIR_NAMES, yaml_document

MODEL_TYPES = ("llm", "embeddings", "rerank", "decisions")


#: The template this module renders.  A shim's shape is documentation first:
#: the files below are what the guide shows and what the scaffold writes, so
#: neither can describe a tree the other does not mean.
TEMPLATE_DIR = Path(__file__).with_name("_shim_template")
TREE_DIR = TEMPLATE_DIR / "tree"

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")

#: The leaf modules a template may ship beside ``provider.yaml``.
LEAF_MODULES = (TRANSFORMS_MODULE, AUTH_MODULE, QUOTA_MODULE)


def _render(text: str, values: dict[str, str]) -> str:
    """Fill a template's placeholders.

    A line that names a placeholder with no value is dropped, so an optional
    field can stay on one line in the template.  A placeholder the caller does
    not know at all is an error rather than an empty substitution: a typo in a
    template must fail loudly, and a test renders every template for that.

    Args:
        text: The template's contents.
        values: The value of every placeholder the template may name.

    Returns:
        The rendered text.

    Raises:
        KeyError: The template names a placeholder *values* does not carry.
    """
    rendered: list[str] = []
    for line in text.splitlines(keepends=True):
        names = _PLACEHOLDER.findall(line)
        unknown = [name for name in names if name not in values]
        if unknown:
            raise KeyError(f"template names unknown placeholder(s) {unknown}")
        if any(not values[name] for name in names):
            continue
        rendered.append(_PLACEHOLDER.sub(lambda m: values[m.group(1)], line))
    return "".join(rendered)


def _template(relative: str) -> str:
    """Return the text of one file under the template directory."""
    return (TEMPLATE_DIR / relative).read_text(encoding="utf-8")


def _write_from_template(
    template: str, destination: Path, values: dict[str, str]
) -> None:
    """Render *template* into *destination*, creating the directory."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_render(_template(template), values), encoding="utf-8")


def _yaml_scalar(value: str) -> str:
    """Quote a scalar only when YAML would otherwise misread it."""
    if not value or value != value.strip():
        return f'"{value}"'
    if value[0] in "[{&*!|>'\"%@`#":
        return f'"{value}"'
    if " #" in value or value.endswith(":"):
        return f'"{value}"'
    return value


def _type_dir(root: Path, vendor: str, model_type: str, lane: str | None) -> Path:
    """Return the type directory a leaf belongs under.

    The type directory is always present; whether the leaf sits directly in it
    or one format level deeper is decided by :func:`_format_level`.
    """
    base = root / vendor / lane if lane else root / vendor
    return base / model_type


def _format_level(type_dir: Path) -> tuple[bool, bool]:
    """Whether *type_dir* already uses a format level, and whether a move is due.

    A type directory holding a leaf directly has exactly one lane.  The second
    lane moves that leaf into ``<its format>/`` so both live at one level; a
    directory that already holds format directories needs no move.

    Returns:
        ``(uses_formats, move_due)``.
    """
    if not type_dir.is_dir():
        return False, False
    if (type_dir / "provider.yaml").is_file():
        return True, True
    for child in sorted(type_dir.iterdir()):
        if child.is_dir() and (child / "provider.yaml").is_file():
            return True, False
    return False, False


def _declared_format(directory: Path) -> str | None:
    """The ``format:`` a leaf declares, for the move a second lane forces."""
    document = yaml_document(directory / "provider.yaml") or {}
    value = document.get("format") or document.get("base")
    return value if isinstance(value, str) and value else None


def _move_to_format_level(type_dir: Path) -> tuple[str | None, str]:
    """Move the lone leaf of *type_dir* into ``<type>/<format>/``.

    Returns the format it moved under (``None`` when the leaf does not say) and
    a note for the operator.  Every recognised leaf module moves with it,
    because they belong to that leaf.
    """
    fmt = _declared_format(type_dir)
    if not fmt:
        return None, (
            f"{type_dir}/provider.yaml declares no 'format:', so a second lane "
            f"cannot be placed beside it; move it into a format directory first"
        )
    destination = type_dir / fmt
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("provider.yaml", *LEAF_MODULES):
        source = type_dir / name
        if source.is_file():
            source.rename(destination / name)
    return fmt, f"moved the existing lane into {destination}"


def _declared_name(provider_yaml: Path) -> str | None:
    """The shim name a ``provider.yaml`` declares, or ``None`` for a manifest."""
    document = yaml_document(provider_yaml)
    if document is None:
        return None
    name = document.get("name")
    return name if isinstance(name, str) and name else None


def _vendor_layout(directory: Path) -> str | None:
    """``typed`` / ``flat`` for a vendor group, ``None`` when it holds no lane."""
    typed = flat = False
    for child in sorted(directory.iterdir()):
        if not child.is_dir() or child.name.startswith(("_", ".")):
            continue
        if child.name in TYPE_DIR_NAMES:
            typed = True
        elif (child / "provider.yaml").is_file():
            flat = True
    if typed:
        return "typed"
    return "flat" if flat else None


def vendor_name_error(vendor: str) -> str | None:
    """Why *vendor* cannot be a directory name here, or ``None`` when it can.

    The loader skips ``_``/``.`` directories and reads a directory named after a
    model type as that model type, so both would write a shim nothing loads.
    """
    if not vendor:
        return "vendor must not be empty"
    if "/" in vendor or "\\" in vendor or vendor in {".", ".."}:
        return f"vendor {vendor!r} must be a single directory name"
    if vendor.startswith(("_", ".")):
        return (
            f"vendor {vendor!r} starts with {vendor[0]!r}; the loader skips such "
            f"directories, so nothing under it would ever load"
        )
    if vendor in TYPE_DIR_NAMES:
        return (
            f"vendor {vendor!r} is a model-type directory name; the loader would "
            f"read it as one. Pick another name, e.g. {vendor}-api"
        )
    return None


def _existing_names(root: Path) -> dict[str, Path]:
    """Every shim name already declared under *root*, with the file declaring it."""
    names: dict[str, Path] = {}
    for provider_yaml in sorted(root.rglob("provider.yaml")):
        relative = provider_yaml.relative_to(root)
        if any(part.startswith(("_", ".")) for part in relative.parts):
            continue
        name = _declared_name(provider_yaml)
        if name:
            names.setdefault(name, provider_yaml)
    return names


def inspect_target(root: Path, vendor: str, name: str) -> tuple[str | None, list[str]]:
    """Check the target against what *root* already holds.

    Three things the loader cannot honour are refused: writing below a vendor
    whose ``provider.yaml`` is a shim (nothing below such a directory is read),
    a name another file already declares (the registry keeps one), and a name
    that cannot be a directory.  A vendor still holding lanes in the old
    shapes is legal and merely noted, with the command that reports the rest.

    Args:
        root: The directory a gateway scans.
        vendor: Vendor directory name.
        name: The shim name the new leaf will declare.

    Returns:
        ``(error, notes)`` — an error refuses the write; a note is printed and
        the write goes ahead.
    """
    notes: list[str] = []
    vendor_dir = root / vendor
    provider_yaml = vendor_dir / "provider.yaml"
    if provider_yaml.is_file() and _declared_name(provider_yaml) is not None:
        return (
            f"{provider_yaml} is a shim, not a vendor manifest: the loader reads "
            f"{vendor}/ as that one shim and never looks below it. Add a new "
            f"top-level vendor, or turn {vendor}/ into a group — move its "
            f"provider.yaml into {vendor}/<type>/ and write a manifest at "
            f"{vendor}/provider.yaml",
            notes,
        )
    taken = _existing_names(root)
    if name in taken:
        return f"{name!r} is already declared by {taken[name]}", notes
    if vendor_dir.is_dir():
        legacy = [
            child.name
            for child in sorted(vendor_dir.iterdir())
            if child.is_dir()
            and child.name not in TYPE_DIR_NAMES
            and (child / "provider.yaml").is_file()
        ]
        if legacy:
            notes.append(
                f"{vendor} still holds format directories directly under it "
                f"({', '.join(legacy)}); the tree check (`llm-rosetta ext check` on an extension folder) reports the "
                f"canonical path"
            )
    return None, notes


def scaffold(
    root: Path,
    vendor: str,
    *,
    format_: str,
    model_type: str = "llm",
    lane: str | None = None,
    base_url: str | None = None,
    models_path: str | None = None,
    api_key_env: str | None = None,
    display_name: str | None = None,
    logo: str | None = None,
    group: str | None = None,
    group_label: str | None = None,
    transform: bool = False,
    auth: bool = False,
    quota: bool = False,
    check: bool = False,
    name: str | None = None,
) -> int:
    """Write the files for one shim from the template.

    Args:
        root: The directory a gateway scans: the built-in ``providers`` tree.
        vendor: Vendor directory name, e.g. ``together``.
        format_: Wire format, e.g. ``chat_completions``.
        model_type: One of :data:`MODEL_TYPES`.
        lane: A named instance of the vendor (ALCF's clusters), written between
            the vendor and its type directories.
        base_url: Upstream base URL.  A placeholder is written when omitted, so
            the file always loads and the missing value is obvious.
        models_path: Model-list URL or path, when it is not the default.
        api_key_env: Credential environment variable.  Defaults to
            ``<VENDOR>_API_KEY``.
        display_name: The shim's row label inside the vendor group.
        logo: Vendor mark URL, written to the vendor manifest.
        group: Vendor group key.  Defaults to the vendor name in the manifest.
        group_label: Vendor group title.
        transform: Write a ``transforms.py`` stub beside the leaf.
        auth: Write a vendor-level ``auth.py`` stub for a credential exchange.
        quota: Write a vendor-level ``quota.py`` stub.
        check: Load the new leaf through the real loader and report.
        name: Shim name.  Defaults to ``<vendor>--<format>``.

    Returns:
        A process exit code: ``0`` on success, ``1`` when the leaf already
        exists or does not load, ``2`` on a bad argument.
    """
    vendor = vendor.strip()
    error = vendor_name_error(vendor)
    if error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if lane is not None:
        lane = lane.strip()
        error = vendor_name_error(lane)
        if error:
            print(f"error: lane {error}", file=sys.stderr)
            return 2
    name = name or f"{vendor}--{format_}"
    api_key_env = api_key_env or vendor.upper().replace("-", "_") + "_API_KEY"
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2

    error, notes = inspect_target(root, vendor, name)
    if error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for note in notes:
        print(f"note: {note}")

    leaf = _resolve_leaf_dir(root, vendor, model_type, lane, format_)
    if leaf is None:
        return 1

    written: list[Path] = []
    written.extend(
        _write_leaf_files(
            root,
            leaf,
            vendor=vendor,
            name=name,
            model_type=model_type,
            format_=format_,
            base_url=base_url,
            models_path=models_path,
            display_name=display_name,
            api_key_env=api_key_env,
            logo=logo,
            group=group,
            group_label=group_label,
            transform=transform,
            auth=auth,
            quota=quota,
        )
    )

    for path in written:
        print(f"wrote   {path}")

    if check:
        return _check(root, leaf, name)
    print(
        "\nnext: fill in the measured facts, add a test under tests/, and update "
        "docs_en + docs_zh."
    )
    return 0


def _resolve_leaf_dir(
    root: Path,
    vendor: str,
    model_type: str,
    lane: str | None,
    format_: str,
) -> Path | None:
    """Return the directory the new leaf goes in, or ``None`` when it cannot.

    The format level appears when a type directory already holds a lane: the
    first lane lives at ``<type>/provider.yaml`` and the second moves it into
    ``<type>/<its format>/`` so both sit at the same level.
    """
    type_dir = _type_dir(root, vendor, model_type, lane or None)
    uses_formats, move_due = _format_level(type_dir)
    if move_due:
        moved, note = _move_to_format_level(type_dir)
        print(f"note: {note}")
        if moved is None:
            return None
        uses_formats = True
    leaf = type_dir / format_ if uses_formats else type_dir
    if (leaf / "provider.yaml").exists():
        print(f"error: {leaf / 'provider.yaml'} already exists", file=sys.stderr)
        return None
    return leaf


def vendor_manifest_text(
    vendor: str,
    *,
    api_key_env: str,
    logo: str | None = None,
    group: str | None = None,
    group_label: str | None = None,
) -> str:
    """Render the vendor manifest: the defaults every leaf below it inherits."""
    return _render(
        _template("tree/vendor/provider.yaml.tmpl"),
        {
            "vendor": vendor,
            "logo": _yaml_scalar(logo) if logo else "",
            "group": _yaml_scalar(group or vendor),
            "group_label": _yaml_scalar(group_label) if group_label else "",
            "api_key_env": _yaml_scalar(api_key_env),
        },
    )


def leaf_text(
    vendor: str,
    *,
    name: str,
    model_type: str,
    format_: str,
    base_url: str | None = None,
    display_name: str | None = None,
    models_path: str | None = None,
    transport: str | None = None,
) -> str:
    """Render the ``provider.yaml`` of one leaf (one lane of a vendor)."""
    return _render(
        _template("tree/vendor/type/format/provider.yaml.tmpl"),
        {
            "vendor": vendor,
            "name": _yaml_scalar(name),
            "model_type": model_type,
            "format": _yaml_scalar(format_),
            "base_url": _yaml_scalar(base_url or "https://example.invalid/v1"),
            "display_name": _yaml_scalar(display_name) if display_name else "",
            "models_path": _yaml_scalar(models_path) if models_path else "",
            "transport": _yaml_scalar(transport) if transport else "",
        },
    )


def _write_leaf_files(
    root: Path,
    leaf: Path,
    *,
    vendor: str,
    name: str,
    model_type: str,
    format_: str,
    base_url: str | None,
    models_path: str | None,
    display_name: str | None,
    api_key_env: str,
    logo: str | None,
    group: str | None,
    group_label: str | None,
    transform: bool,
    auth: bool,
    quota: bool,
) -> list[Path]:
    """Write the leaf, the vendor manifest and any optional module.

    Returns:
        Every file written, in the order it was written.
    """
    written: list[Path] = []
    manifest = root / vendor / "provider.yaml"
    if not manifest.is_file():
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            vendor_manifest_text(
                vendor,
                api_key_env=api_key_env,
                logo=logo,
                group=group,
                group_label=group_label,
            ),
            encoding="utf-8",
        )
        written.append(manifest)

    leaf.mkdir(parents=True, exist_ok=True)
    (leaf / "provider.yaml").write_text(
        leaf_text(
            vendor,
            name=name,
            model_type=model_type,
            format_=format_,
            base_url=base_url,
            display_name=display_name,
            models_path=models_path,
        ),
        encoding="utf-8",
    )
    written.append(leaf / "provider.yaml")

    if transform:
        _write_from_template(
            "tree/vendor/type/format/transforms.py.tmpl",
            leaf / "transforms.py",
            {"name": name, "vendor": vendor, "format": format_},
        )
        written.append(leaf / "transforms.py")

    for module, wanted in (("auth.py", auth), ("quota.py", quota)):
        if not wanted:
            continue
        destination = root / vendor / module
        _write_from_template(
            f"tree/vendor/{module}.tmpl",
            destination,
            {"name": name, "vendor": vendor, "format": format_},
        )
        written.append(destination)
    return written


def _check(root: Path, leaf: Path, name: str) -> int:
    """Load the new leaf through the real loader.

    A leaf that refuses to load is the failure this step exists to catch, so the
    loader's own complaint is printed and turned into an exit code rather than
    raised: the operator typed flags, not a stack trace.
    """
    from llm_rosetta.shims._loader import load_providers_from_dir

    relative = leaf.relative_to(root).as_posix()
    try:
        shims = load_providers_from_dir(root, only_paths={relative})
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        print(f"error: the new shim does not load: {exc}", file=sys.stderr)
        return 1
    names = [s.name for s in shims]
    if names != [name]:
        print(
            f"error: the loader produced {names}, expected [{name!r}]",
            file=sys.stderr,
        )
        return 1
    shim = shims[0]
    print(
        f"check   {name} loads: group={shim.group!r} label={shim.group_label!r} "
        f"logo={'yes' if shim.logo else 'no'}"
    )
    return 0
