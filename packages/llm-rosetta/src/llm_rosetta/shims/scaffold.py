"""Scaffold the files for a new provider shim.

A shim is a small, boring set of files — a ``provider.yaml``, sometimes a
``transforms.py``, and often a vendor manifest above them — and getting the
shape wrong is not caught until a gateway loads it.  This module writes that
shape, either from flags or by asking for each field, and then *loads* it: the
last step runs the real loader over the new leaf, so a scaffold no gateway
could use fails here rather than at startup.

The same code backs two entry points: ``llm-rosetta-gateway shims new`` (the
installed command) and ``scripts/dev/new_shim.py`` (a repository checkout, where
the built-in tree is the default target).

Layouts::

    typed (default)   <root>/<vendor>/llm/<format>/provider.yaml
                      <root>/<vendor>/<type>/provider.yaml      (non-chat)
    flat              <root>/<vendor>/<format>/provider.yaml      (argo, alcf)

``<root>`` is the directory a gateway scans: the built-in ``providers`` tree, or
a package's ``providers`` directory (``src/<package>/providers`` in the
``llm-rosetta-shim-*`` repositories).
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import urlopen

MODEL_TYPES = ("llm", "embeddings", "rerank", "decisions")

#: Largest mark this scaffolder will save.
MAX_LOGO_BYTES = 1024 * 1024

#: The icon set the admin panel already shows beside a vendor, served by jsDelivr.
#: One SVG per provider slug; a vendor name that matches one gets it offered.
LOBEHUB_ICON_URL = (
    "https://cdn.jsdelivr.net/npm/@lobehub/icons-static-svg@latest/icons/{slug}.svg"
)

#: Suffixes a vendor directory adds around the provider it wraps.
_VENDOR_SUFFIXES = ("_free", "-free", "_zen", "-zen", "_pro", "-pro")

#: Content types a vendor mark may have, and the suffix each one gets.
LOGO_SUFFIXES = {
    "image/svg+xml": ".svg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
}

#: The format a model type is usually scaffolded with.
DEFAULT_FORMATS = {
    "llm": "chat_completions",
    "embeddings": "openai",
    "rerank": "jina",
    "decisions": "systemone",
}

LEAF_TEMPLATE = """\
{comment}name: {name}
model_type: {model_type}
format: {format}
{identity}
connection:
{connection}
"""

TRANSFORMS_TEMPLATE = '''\
"""Field-level transforms for {name}.

The loader imports this module when it sits beside ``provider.yaml`` and reads
the tuples it exports; there is nothing to declare in the YAML.  A transform
takes and returns a plain body dict, and returning the argument unchanged keeps
the field.
"""

from __future__ import annotations

from typing import Any


def _keep(body: dict[str, Any]) -> dict[str, Any]:
    """Return the body unchanged."""

    return body


#: A tuple, not a function: the loader reads the exports and the pipeline
#: iterates them, so a callable export would fail at request time, not here.
response_body_transforms = (_keep,)
'''


def _leaf_comment(name: str, format_: str) -> str:
    """A first line that says why this file exists, in the house style."""
    return (
        f"# {name}: the {format_} lane.  Delete this comment once the shim has\n"
        f"# something worth explaining — a measured quirk, a field the upstream\n"
        f"# renames, a range it accepts.\n"
    )


def _yaml_scalar(value: str) -> str:
    """Quote a scalar only when YAML would otherwise misread it."""
    if not value or value != value.strip():
        return f'"{value}"'
    if value[0] in "[{&*!|>'\"%@`#":
        return f'"{value}"'
    if " #" in value or value.endswith(":"):
        return f'"{value}"'
    return value


def _leaf_path(root: Path, vendor: str, model_type: str, format_: str, layout: str):
    """Return the leaf directory for the requested layout."""
    if layout == "flat":
        return root / vendor / format_
    if model_type == "llm":
        return root / vendor / "llm" / format_
    # A non-chat type names its own directory, and the leaf inside names the
    # dialect: ``<vendor>/embeddings/provider.yaml`` with ``format: openai``.
    return root / vendor / model_type


#: Directory names the loader reads as model types rather than vendors.
_TYPE_DIR_NAMES = frozenset(MODEL_TYPES)


@dataclass(frozen=True)
class PackageShape:
    """What a scan root already holds, and the conventions it follows.

    Attributes:
        layout: ``"typed"`` or ``"flat"`` when every vendor agrees on one
            shape, ``None`` when they do not.
        leaf_vendors: Vendors whose ``provider.yaml`` is a shim, not a manifest
            — the ``llm-rosetta-shim-free`` bundle shape.
        typed_vendors: Vendor groups with ``<type>/`` directories.
        flat_vendors: Vendor groups with ``<format>/`` children.
    """

    layout: str | None
    leaf_vendors: tuple[str, ...]
    typed_vendors: tuple[str, ...]
    flat_vendors: tuple[str, ...]

    @property
    def leaf_only(self) -> bool:
        """True when every vendor is a single leaf directory."""
        return bool(self.leaf_vendors) and not (self.typed_vendors or self.flat_vendors)

    def describe(self) -> str:
        """One sentence naming what was found; empty for a root with no shims."""
        if self.leaf_only:
            return (
                f"{len(self.leaf_vendors)} single-lane vendor(s), each a leaf "
                f"directory — flat layout, no vendor manifest"
            )
        parts = []
        if self.typed_vendors:
            parts.append(f"{len(self.typed_vendors)} with model-type directories")
        if self.flat_vendors:
            parts.append(f"{len(self.flat_vendors)} with format directories")
        if self.leaf_vendors:
            parts.append(f"{len(self.leaf_vendors)} single-lane leaf vendor(s)")
        if not parts:
            return ""
        tail = f" — {self.layout} layout" if self.layout else " — mixed shapes"
        return "; ".join(parts) + tail


def _yaml_document(path: Path) -> dict[str, Any] | None:
    """Read a ``provider.yaml`` as a mapping, or ``None`` when it is not one."""
    from llm_rosetta._vendor.yaml import load as yaml_load

    try:
        document = yaml_load(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a broken file is not this tool's business
        return None
    return document if isinstance(document, dict) else None


def _declared_name(provider_yaml: Path) -> str | None:
    """The shim name a ``provider.yaml`` declares, or ``None`` for a manifest."""
    document = _yaml_document(provider_yaml)
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
        if child.name in _TYPE_DIR_NAMES:
            typed = True
        elif (child / "provider.yaml").is_file():
            flat = True
    if typed:
        return "typed"
    return "flat" if flat else None


def detect_package(root: Path) -> PackageShape:
    """Read the scan root's existing conventions.

    The two shipped shapes are a bundle of single-lane vendors
    (``llm-rosetta-shim-free``: one directory per provider, the directory name
    *is* the vendor) and a vendor manifest over model-type directories
    (``google``: ``llm/<format>`` plus ``embeddings/``).  A root that mixes them
    has no convention worth copying, so ``layout`` stays ``None``.

    Args:
        root: The directory a gateway scans.

    Returns:
        What the root holds, in the terms a new shim has to follow.
    """
    leaf: list[str] = []
    typed: list[str] = []
    flat: list[str] = []
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name.startswith(("_", ".")):
                continue
            provider_yaml = child / "provider.yaml"
            if provider_yaml.is_file() and _declared_name(provider_yaml) is not None:
                leaf.append(child.name)
                continue
            layout = _vendor_layout(child)
            if layout == "typed":
                typed.append(child.name)
            elif layout == "flat":
                flat.append(child.name)
    if typed and not flat and not leaf:
        layout = "typed"
    elif (flat or leaf) and not typed:
        layout = "flat"
    else:
        layout = None
    return PackageShape(layout, tuple(leaf), tuple(typed), tuple(flat))


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
    if vendor in _TYPE_DIR_NAMES:
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


def inspect_target(
    root: Path, vendor: str, name: str, *, layout: str
) -> tuple[str | None, list[str]]:
    """Check the target against what *root* already holds.

    Three things the loader cannot honour are refused: writing below a vendor
    whose ``provider.yaml`` is a shim (nothing below such a directory is read),
    a name another file already declares (the registry keeps one), and a name
    that cannot be a directory.  A vendor that already holds lanes in the other
    layout is legal and merely noted.

    Args:
        root: The directory a gateway scans.
        vendor: Vendor directory name.
        name: The shim name the new leaf will declare.
        layout: The layout the leaf will be written in.

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
            f"provider.yaml into {vendor}/<format>/ and write a manifest at "
            f"{vendor}/provider.yaml",
            notes,
        )
    taken = _existing_names(root)
    if name in taken:
        return f"{name!r} is already declared by {taken[name]}", notes
    current = _vendor_layout(vendor_dir) if vendor_dir.is_dir() else None
    if current is not None and current != layout:
        notes.append(
            f"{vendor} already holds leaves in the {current} layout; this one is "
            f"{layout} — both load, but the vendor now carries two shapes"
        )
    return None, notes


def _existing_vendor_fields(manifest: Path) -> set[str]:
    """The identity fields a vendor manifest already declares."""
    if not manifest.is_file():
        return set()
    from llm_rosetta._vendor.yaml import load as yaml_load

    document = yaml_load(manifest.read_text(encoding="utf-8")) or {}
    if not isinstance(document, dict):
        return set()
    return {key for key in ("logo", "group", "group_label") if key in document}


def _append_registry_entry(root: Path, name: str, leaf: Path) -> str | None:
    """Add the leaf to the package's ``shims.yaml``, when it ships one.

    Returns a note for the operator, or ``None`` when there was nothing to do.
    The manifest is authoritative for a gateway, so a new directory that is not
    listed there simply never loads — which makes this the one edit that must
    not be forgotten.
    """
    manifest = root / "shims.yaml"
    if not manifest.is_file():
        return None
    from llm_rosetta._vendor.yaml import load as yaml_load

    document = yaml_load(manifest.read_text(encoding="utf-8")) or {}
    entries = document.get("shims") if isinstance(document, dict) else None
    if isinstance(entries, list):
        declared = {e.get("name") for e in entries if isinstance(e, dict)}
        if name in declared:
            return None
    relative = leaf.relative_to(root).as_posix()
    if not isinstance(document, dict) or list(document)[-1:] != ["shims"]:
        return (
            f"{manifest}: add this entry by hand — the section is not the last "
            f"key, so appending would move it:\n"
            f"  - name: {name}\n    path: {relative}"
        )
    text = manifest.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        text += "\n"
    text += f"  - name: {name}\n    path: {relative}\n"
    manifest.write_text(text, encoding="utf-8")
    return f"updated {manifest}"


def _identity_lines(
    vendor: str,
    *,
    logo: str | None,
    group: str | None,
    group_label: str | None,
    inherited: set[str],
    default_group: str | None,
) -> list[str]:
    """The vendor-level lines a file declares, skipping what it inherits."""
    lines = []
    for key, value in (
        ("logo", logo),
        ("group", group or default_group),
        ("group_label", group_label),
    ):
        if value and key not in inherited:
            lines.append(f"{key}: {_yaml_scalar(value)}")
    return lines


def _git(*args: str, cwd: Path) -> str | None:
    """Run one git query, or return ``None`` when git or the query is absent."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _assets_dir(root: Path) -> Path | None:
    """The ``assets/`` directory of the repository the scan root lives in.

    A published shim package keeps its marks at the repository root, beside
    ``src/``, because they are not part of the archive: a gateway reads the mark
    from the repository over the CDN, and the archive only ever holds
    ``providers/``.
    """
    top = _git("rev-parse", "--show-toplevel", cwd=root)
    return Path(top) / "assets" if top else None


def _jsdelivr_slug(repo_dir: Path) -> str | None:
    """``<owner>/<repo>@<ref>`` for the repository *repo_dir* is in.

    Read from the git remote rather than asked for: it is the same repository
    the package will be published from, so a typo here would produce a mark URL
    that 404s in the panel and nowhere else.
    """
    remote = _git("remote", "get-url", "origin", cwd=repo_dir)
    if not remote or "github.com" not in remote:
        return None
    tail = remote.split("github.com", 1)[1].lstrip(":/")
    if tail.endswith(".git"):
        tail = tail[: -len(".git")]
    parts = [part for part in tail.split("/") if part]
    if len(parts) < 2:
        return None
    branch = _git("symbolic-ref", "--short", "refs/remotes/origin/HEAD", cwd=repo_dir)
    ref = branch.split("/", 1)[1] if branch and "/" in branch else "main"
    return f"{parts[0]}/{parts[1]}@{ref}"


def _download_logo(url: str) -> tuple[bytes, str]:
    """Fetch a mark, size-capped and content-typed.

    Args:
        url: An ``https://`` URL the operator typed.

    Returns:
        The bytes and the suffix their content type implies.

    Raises:
        ValueError: On a non-https URL, a non-image answer, or an oversized one.
    """
    if urlsplit(url).scheme != "https":
        raise ValueError("the mark URL must use https")
    with urlopen(url, timeout=20) as response:  # noqa: S310 - operator-supplied
        content_type = (response.headers.get_content_type() or "").lower()
        data = response.read(MAX_LOGO_BYTES + 1)
    if len(data) > MAX_LOGO_BYTES:
        raise ValueError(f"the mark is larger than {MAX_LOGO_BYTES} bytes")
    suffix = LOGO_SUFFIXES.get(content_type)
    if suffix is None:
        raise ValueError(
            f"the URL answered {content_type or 'no type'!r}, not an image"
        )
    return data, suffix


def is_jsdelivr(url: str) -> bool:
    """True when *url* is already served by jsDelivr."""
    host = urlsplit(url).netloc.lower()
    return host == "jsdelivr.net" or host.endswith(".jsdelivr.net")


def lobehub_candidates(vendor: str) -> tuple[str, ...]:
    """Icon slugs to try for *vendor*, most specific first.

    A vendor directory often wraps a provider with a pool marker
    (``opencode_zen_free``, ``openai_chatgpt``), so the leading segment is
    tried as well as the whole name.  Anything that is not a plain slug is
    dropped: the slug goes into a URL, and a name the loader would accept is
    not necessarily a name a CDN path can hold.

    Args:
        vendor: The vendor directory name.

    Returns:
        Up to four distinct slugs.
    """
    slug = vendor.strip().lower()
    candidates: list[str] = [slug]
    for separator in ("_", "-"):
        if separator in slug:
            candidates.append(slug.split(separator, 1)[0])
    for marker in _VENDOR_SUFFIXES:
        if slug.endswith(marker):
            candidates.append(slug[: -len(marker)])
    kept: list[str] = []
    for candidate in candidates:
        if not candidate or candidate in kept:
            continue
        if not all(char.isalnum() or char in "._-" for char in candidate):
            continue
        kept.append(candidate)
    return tuple(kept[:4])


def find_lobehub_logo(vendor: str) -> str | None:
    """The lobehub icon URL for *vendor*, or ``None`` when there is none.

    One ``HEAD`` per candidate, and the search stops at the first answer that
    is not a plain status: a blocked or unreachable network should cost one
    timeout, not four, and it is not this tool's business to tell them apart.

    Args:
        vendor: The vendor directory name.

    Returns:
        An ``https://cdn.jsdelivr.net/...`` URL, or ``None``.
    """
    for slug in lobehub_candidates(vendor):
        url = LOBEHUB_ICON_URL.format(slug=slug)
        status, _error = _http_status(url, timeout=5.0, method="HEAD")
        if status == 200:
            return url
        if status is None:
            return None
    return None


def localize_logo(url: str, root: Path, vendor: str) -> tuple[str, str]:
    """Save a remote mark into the package and return its jsDelivr URL.

    The panel loads a mark from wherever the shim points it, so a third-party
    URL is a dependency on someone else's host and their uptime.  A published
    package instead keeps its marks in ``assets/`` and links the CDN copy.

    Nothing is changed when the package cannot be identified (no git remote, no
    repository around the scan root): the URL is returned unchanged with the
    reason, and the operator decides.  A URL that already points at jsDelivr is
    kept as typed — the mark is served from the CDN either way.

    Args:
        url: The mark URL the operator gave.
        root: The scan root being scaffolded into.
        vendor: Vendor directory name; also the asset's file name.

    Returns:
        ``(url_to_write, note)`` — the jsDelivr URL when the mark was saved,
        else the original URL, plus a line to print.
    """
    if is_jsdelivr(url):
        return url, f"kept {url}: already served from jsDelivr, nothing to copy"
    assets = _assets_dir(root)
    if assets is None:
        return url, f"kept {url}: the scan root is not inside a git repository"
    slug = _jsdelivr_slug(assets.parent)
    if slug is None:
        return url, f"kept {url}: no github.com 'origin' remote to build a CDN URL"
    try:
        data, suffix = _download_logo(url)
    except Exception as exc:  # noqa: BLE001 - reported, the URL is kept
        return url, f"kept {url}: {exc}"

    target = assets / f"{vendor}{suffix}"
    if target.is_file() and target.read_bytes() == data:
        return (
            f"https://cdn.jsdelivr.net/gh/{slug}/assets/{target.name}",
            f"kept existing {target}",
        )
    assets.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return (
        f"https://cdn.jsdelivr.net/gh/{slug}/assets/{target.name}",
        f"saved   {target}",
    )


def scaffold(
    root: Path,
    vendor: str,
    *,
    format_: str,
    model_type: str = "llm",
    layout: str = "typed",
    base_url: str | None = None,
    models_path: str | None = None,
    api_key_env: str | None = None,
    display_name: str | None = None,
    logo: str | None = None,
    group: str | None = None,
    group_label: str | None = None,
    vendor_manifest: bool = True,
    registry_manifest: bool = True,
    transform: bool = False,
    check: bool = False,
    name: str | None = None,
) -> int:
    """Write the files for one shim.

    Args:
        root: The directory a gateway scans — a built-in ``providers`` tree, or
            a package's ``providers`` directory.
        vendor: Vendor directory name, e.g. ``acme``.
        format_: Wire format, e.g. ``chat_completions``.
        model_type: One of :data:`MODEL_TYPES`.
        layout: ``"typed"`` (``<vendor>/llm/<format>``) or ``"flat"``
            (``<vendor>/<format>``).
        base_url: Upstream base URL.  A placeholder is written when omitted, so
            the file always loads and the missing value is obvious.
        models_path: Model-list URL or path, when it is not the default.
        api_key_env: Credential environment variable.  Defaults to
            ``<VENDOR>_API_KEY``.
        display_name: The shim's row label inside the vendor group.
        logo: Vendor mark URL, written to the vendor manifest.
        group: Vendor group key.  Defaults to the vendor name in the manifest.
        group_label: Vendor group title.
        vendor_manifest: Write ``<vendor>/provider.yaml`` when the vendor has
            none, and keep shared values there.
        registry_manifest: Add the leaf to ``<root>/shims.yaml`` when that file
            exists.
        transform: Write a ``transforms.py`` stub beside the leaf.
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
    name = name or f"{vendor}--{format_}"
    api_key_env = api_key_env or vendor.upper().replace("-", "_") + "_API_KEY"
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2
    error, notes = inspect_target(root, vendor, name, layout=layout)
    if error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for note in notes:
        print(f"note: {note}")

    leaf = _leaf_path(root, vendor, model_type, format_, layout)
    provider_yaml = leaf / "provider.yaml"
    if provider_yaml.exists():
        print(f"error: {provider_yaml} already exists", file=sys.stderr)
        return 1

    manifest = root / vendor / "provider.yaml"
    if vendor_manifest and not manifest.is_file():
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest_lines = [
            f"# Vendor manifest: the defaults every {vendor} shim shares.",
            "# A leaf inherits every field here and may override any of them.",
        ]
        manifest_lines.extend(
            _identity_lines(
                vendor,
                logo=logo,
                group=group,
                group_label=group_label,
                inherited=set(),
                default_group=vendor,
            )
        )
        manifest_lines.append("connection:")
        manifest_lines.append(f"  api_key_env: {_yaml_scalar(api_key_env)}")
        manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    # Read back what the vendor already declares, including a manifest just
    # written: the leaf states only what it does not inherit.
    inherited = set() if not vendor_manifest else _existing_vendor_fields(manifest)

    identity_lines = _identity_lines(
        vendor,
        logo=logo,
        group=group,
        group_label=group_label,
        inherited=inherited,
        default_group=None,
    )
    if display_name:
        identity_lines.append(f"display_name: {_yaml_scalar(display_name)}")

    connection_lines: list[str] = []
    connection_lines.append(
        f"  base_url: {_yaml_scalar(base_url)}"
        if base_url
        else "  base_url: https://example.invalid/v1"
    )
    if not vendor_manifest:
        connection_lines.append(f"  api_key_env: {_yaml_scalar(api_key_env)}")
    if models_path:
        connection_lines.append(f"  models_path: {_yaml_scalar(models_path)}")

    leaf.mkdir(parents=True, exist_ok=True)
    provider_yaml.write_text(
        LEAF_TEMPLATE.format(
            comment=_leaf_comment(name, format_),
            name=name,
            model_type=model_type,
            format=format_,
            identity=("\n".join(identity_lines) + "\n") if identity_lines else "",
            connection="\n".join(connection_lines),
        ),
        encoding="utf-8",
    )
    written = [provider_yaml]
    if vendor_manifest:
        written.append(manifest)

    if transform:
        transforms = leaf / "transforms.py"
        transforms.write_text(TRANSFORMS_TEMPLATE.format(name=name), encoding="utf-8")
        written.append(transforms)

    note = _append_registry_entry(root, name, leaf) if registry_manifest else None

    for path in written:
        print(f"wrote   {path}")
    if note:
        print(note)

    if check:
        return _check(root, leaf, name)
    print(
        "\nnext: fill in the measured facts, add a test under tests/, and update "
        "docs_en + docs_zh."
    )
    return 0


def _check(root: Path, leaf: Path, name: str) -> int:
    """Load the new leaf through the real loader.

    A leaf that refuses to load is the failure this step exists to catch, so the
    loader's own complaint is printed and turned into an exit code rather than
    raised: the operator typed flags, not a stack trace.
    """
    from llm_rosetta.shims.providers import load_providers_from_dir

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


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------


def available_formats(model_type: str) -> tuple[str, ...]:
    """The formats *model_type* can be scaffolded with, in display order.

    A shim's format is not free text: the converters are registered per
    ``(model_type, format)`` pair, and a pair that is not registered cannot
    load.  This is that registry's own list, so the prompt and the refusal
    below cannot disagree with the loader.

    Args:
        model_type: One of :data:`MODEL_TYPES`.

    Returns:
        The formats, in the catalog's display order; empty when the type is
        unknown (which the CLI already excludes with ``choices=``).
    """
    try:
        from llm_rosetta.converters.registry import formats_for
    except Exception:  # noqa: BLE001 - a partial install still gets a default
        return ()
    try:
        return formats_for(model_type)
    except ValueError:
        return ()


def _format_help() -> str:
    """One line per model type, for ``--format``'s help and the refusal."""
    parts = []
    for model_type in MODEL_TYPES:
        formats = available_formats(model_type)
        if formats:
            parts.append(f"{model_type}: {', '.join(formats)}")
    return "; ".join(parts)


def _http_status(
    url: str, *, timeout: float = 6.0, method: str = "GET"
) -> tuple[int | None, str]:
    """Request *url* and return ``(status, error)``; ``status`` is None on failure."""
    from urllib.error import HTTPError
    from urllib.request import Request

    request = Request(
        url, headers={"User-Agent": "llm-rosetta-scaffold"}, method=method
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.status, ""
    except HTTPError as exc:
        return exc.code, ""
    except Exception as exc:  # noqa: BLE001 - the reason is the answer
        return None, str(exc)


def normalize_base_url(url: str) -> tuple[str, str]:
    """Strip a pasted model-list suffix, and say so.

    People paste whatever the vendor's console shows, and ``…/v1/models`` is a
    common one.  A base URL with it would be joined again by every lane, so the
    suffix is removed here rather than discovered as a ``/models/models`` 404.

    Args:
        url: The base URL as typed.

    Returns:
        ``(base_url, note)`` — the note is empty when nothing changed.
    """
    trimmed = url.strip().rstrip("/")
    if trimmed.endswith("/models"):
        return trimmed[
            : -len("/models")
        ], f"dropped the pasted /models suffix: {trimmed}"
    return trimmed, ""


def probe_base_url(url: str) -> tuple[bool, str]:
    """Check that *url* answers, and say what it answered.

    No credential is sent, so an authentication refusal is a *pass*: it proves
    the host answers on that path.  A ``404`` on ``/models`` is the interesting
    failure — the base URL usually has the wrong suffix — so the base itself is
    tried before the verdict is negative.

    Args:
        url: The base URL as typed.

    Returns:
        ``(ok, detail)`` — the detail is a sentence fit to print.
    """
    base = url.rstrip("/")
    status, error = _http_status(f"{base}/models")
    if status is None:
        return False, f"cannot reach {base}: {error}"
    if status in (401, 403):
        return True, f"{base} answers {status} without a key — reachable"
    if status == 200:
        return True, f"{base}/models answers 200"
    base_status, _ = _http_status(base)
    if base_status is not None and base_status != 404:
        return False, (
            f"{base}/models answers {status}, {base} answers {base_status} — "
            f"check whether the base URL carries the right suffix"
        )
    return False, f"{base}/models answers {status}"


class _CancelGuard:
    """Turn Ctrl+C into one clean cancellation, and a double press into an exit.

    ``input()`` raises ``KeyboardInterrupt`` on SIGINT.  Catching that naively
    loses the second press: the operator repeats it while the first one is
    still unwinding, and the traceback they were trying to avoid appears.  The
    guard counts the presses instead.  The first one leaves ``presses`` at 1
    for the caller to see; the second restores the default handler, so a third
    press ends the process at once.

    Attributes:
        presses: How many SIGINTs arrived inside the ``with`` block.
        writing: Set once the files are being written, so the cancellation
            message can say whether anything reached the disk.
    """

    def __init__(self) -> None:
        self.presses = 0
        self.writing = False
        self._previous: Any = None

    def __enter__(self) -> _CancelGuard:
        if threading.current_thread() is threading.main_thread():
            self._previous = signal.signal(signal.SIGINT, self._handle)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._previous is not None:
            signal.signal(signal.SIGINT, self._previous)

    def _handle(self, _signum: int, _frame: object) -> None:
        self.presses += 1
        if self.presses > 1:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
        raise KeyboardInterrupt


class _InputClosed(Exception):
    """The operator ended the input stream (Ctrl+D) at a prompt."""


#: The guard in force for the current ``run``, so ``_ask`` can tell a first
#: Ctrl+C (re-ask) from a second one (abort).
_guard: _CancelGuard | None = None


def _ask(question: str, default: str | None = None) -> str:
    """Ask for a string, with a default shown in brackets.

    Ctrl+D ends the questionnaire: there is no value to take, and every later
    prompt would read end-of-file too, so the run is cancelled instead.

    Raises:
        _InputClosed: The input stream ended.
    """
    suffix = f" [{default}]" if default else ""
    while True:
        try:
            answer = input(f"{question}{suffix}: ").strip()
        except EOFError as exc:
            raise _InputClosed from exc
        except KeyboardInterrupt:
            if _guard is None or _guard.presses > 1:
                raise
            print("\n  Ctrl+C again to quit; no files have been written.")
            continue
        return answer or (default or "")


def _ask_choice(question: str, choices: tuple[str, ...], default: str) -> str:
    """Ask for one of *choices*, showing them."""
    answer = _ask(f"{question} ({'/'.join(choices)})", default)
    while answer not in choices:
        print(f"  {answer!r} is not one of {', '.join(choices)}")
        answer = _ask(question, default)
    return answer


def _ask_bool(question: str, default: bool = False) -> bool:
    """Ask a yes/no question."""
    default_text = "Y/n" if default else "y/N"
    answer = _ask(f"{question} [{default_text}]") or ("y" if default else "n")
    return answer.lower().startswith("y")


def _fill_interactively(args: argparse.Namespace) -> None:
    """Ask for every field the command line left out."""
    print("Creating a shim.  Enter accepts the value in brackets.\n")
    args.root = _ask("Scan root (the providers directory)", args.root)
    shape = detect_package(Path(args.root).expanduser())
    if shape.describe():
        print(f"  found {shape.describe()}")
    args.vendor = args.vendor or _ask("Vendor directory name (e.g. acme)")
    while (error := vendor_name_error(args.vendor)) is not None:
        print(f"  {error}")
        args.vendor = _ask("Vendor directory name (e.g. acme)")
    args.model_type = _ask_choice("Model type", MODEL_TYPES, args.model_type)
    formats = available_formats(args.model_type) or (
        DEFAULT_FORMATS.get(args.model_type, "chat_completions"),
    )
    default_format = args.format or DEFAULT_FORMATS.get(args.model_type, formats[0])
    args.format = _ask_choice("Wire format", formats, default_format)
    print("Directory layout:")
    print("  typed  <vendor>/llm/<format>/ for chat, <vendor>/<type>/ otherwise")
    print("  flat   <vendor>/<format>/ — the format directory sits under the vendor")
    args.layout = _ask_choice(
        "Layout", ("typed", "flat"), args.layout or shape.layout or "typed"
    )
    args.base_url = _ask("Upstream base URL", args.base_url or "")
    if args.base_url:
        args.base_url, note = normalize_base_url(args.base_url)
        if note:
            print(f"  {note}")
        ok, detail = probe_base_url(args.base_url)
        print(f"  {'ok:  ' if ok else 'WARN:'} {detail}")
    args.api_key_env = _ask(
        "Credential env var",
        args.api_key_env or args.vendor.upper().replace("-", "_") + "_API_KEY",
    )
    args.models_path = _ask("Model-list URL or path (optional)", args.models_path or "")
    args.display_name = _ask(
        "Row label inside the vendor group (optional)", args.display_name or ""
    )
    icon = find_lobehub_logo(args.vendor)
    if icon:
        print(f"  lobehub has an icon for {args.vendor}: {icon}")
    args.logo = _ask("Vendor logo URL (optional)", args.logo or icon or "")
    if args.logo:
        if is_jsdelivr(args.logo):
            print("  already served from jsDelivr; nothing to copy into the package")
            args.localize_logo = False
        else:
            args.localize_logo = _ask_bool(
                "Save the mark into the package and reference its jsDelivr URL?",
                True,
            )
    args.group_label = _ask(
        "Group title (optional)", args.group_label or args.vendor.title()
    )
    args.transform = _ask_bool("Add a transforms.py stub?", args.transform)
    args.check = True


def add_arguments(parser: argparse.ArgumentParser, *, default_root: str) -> None:
    """Register the ``shims new`` flags on *parser*.

    Shared by the gateway command and the repository script so the two cannot
    drift apart; only the default root differs.

    Args:
        parser: The parser for this subcommand.
        default_root: The scan root used when ``--root`` is omitted.
    """
    parser.add_argument(
        "vendor", nargs="?", default="", help="Vendor directory name, e.g. acme"
    )
    parser.add_argument("--name", help="Shim name; default '<vendor>--<format>'")
    parser.add_argument(
        "--format",
        default="",
        help="Wire format; default depends on the model type. Available: "
        + (_format_help() or ", ".join(sorted(DEFAULT_FORMATS.values()))),
    )
    parser.add_argument(
        "--model-type", default="llm", choices=MODEL_TYPES, help="Model type"
    )
    parser.add_argument(
        "--root",
        default=default_root,
        help="Directory a gateway scans (a providers tree)",
    )
    parser.add_argument(
        "--layout",
        choices=("typed", "flat"),
        default=None,
        help="typed: <vendor>/llm/<format>; flat: <vendor>/<format>.  Default: "
        "follow the scan root's existing shape, else typed",
    )
    parser.add_argument("--base-url", help="Upstream base URL")
    parser.add_argument(
        "--models-path", help="Model-list URL or path, if not the default"
    )
    parser.add_argument(
        "--api-key-env", help="Credential env var; default <VENDOR>_API_KEY"
    )
    parser.add_argument("--display-name", help="Row label inside the vendor group")
    parser.add_argument("--logo", help="Vendor mark URL")
    parser.add_argument(
        "--auto-logo",
        action="store_true",
        help="Take the mark from the lobehub icon set on jsDelivr, looked up "
        "by vendor name; fails when there is no icon under that name",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Check that --base-url answers, and report what it answered",
    )
    parser.add_argument(
        "--localize-logo",
        action="store_true",
        help="Save the mark into the package's assets/ and reference its "
        "jsDelivr URL (the interactive mode asks)",
    )
    parser.add_argument("--group", help="Group key; default the vendor name")
    parser.add_argument("--group-label", help="Group title")
    parser.add_argument(
        "--vendor-manifest",
        dest="vendor_manifest",
        action="store_const",
        const=True,
        default=None,
        help="Write <vendor>/provider.yaml, when it has none, for the identity "
        "and credential every lane shares.  Default: yes, unless the scan root "
        "is a bundle of single-lane vendors",
    )
    parser.add_argument(
        "--no-vendor-manifest",
        dest="vendor_manifest",
        action="store_const",
        const=False,
        help="Write every field on the leaf instead of a vendor manifest",
    )
    parser.add_argument(
        "--no-registry-manifest",
        action="store_true",
        help="Do not add the leaf to the package's shims.yaml",
    )
    parser.add_argument(
        "--transform", action="store_true", help="Add a transforms.py stub"
    )
    parser.add_argument(
        "--check", action="store_true", help="Load the new leaf through the real loader"
    )
    parser.add_argument(
        "--no-input",
        action="store_true",
        help="Never prompt; every needed value must be given as a flag "
        "(implied when stdin is not a terminal)",
    )


def run(args: argparse.Namespace) -> int:
    """Scaffold from parsed arguments, cancelling cleanly on Ctrl+C.

    Args:
        args: The parsed ``shims new`` arguments.

    Returns:
        0 when the shim was written and loaded, 1 when the loader refused it,
        2 for a usage error, and 130 when the operator interrupted the run.
    """
    global _guard
    guard = _CancelGuard()
    _guard = guard
    with guard:
        try:
            return _run(args)
        except KeyboardInterrupt:
            print(f"\ncancelled — {_written_state(guard)}.", file=sys.stderr)
            return 130
        except _InputClosed:
            print(
                f"\ncancelled — input ended (Ctrl+D); {_written_state(guard)}.",
                file=sys.stderr,
            )
            return 130
        finally:
            _guard = None


def _written_state(guard: _CancelGuard) -> str:
    """How far the run got, for the cancellation message."""
    return (
        "files written so far were kept" if guard.writing else "no files were written"
    )


def _run(args: argparse.Namespace) -> int:
    """Scaffold from parsed arguments, prompting first when that is allowed."""
    interactive = not args.no_input and sys.stdin.isatty()
    if interactive:
        _fill_interactively(args)
    elif not args.vendor:
        print(
            "error: a vendor name is required when not prompting "
            "(pass --no-input to say so explicitly)",
            file=sys.stderr,
        )
        return 2
    if not args.format:
        args.format = DEFAULT_FORMATS.get(args.model_type, "chat_completions")
    formats = available_formats(args.model_type)
    if formats and args.format not in formats:
        print(
            f"error: unknown format {args.format!r} for {args.model_type}; "
            f"available: {', '.join(formats)}",
            file=sys.stderr,
        )
        return 2
    root = Path(args.root).expanduser().resolve()
    shape = detect_package(root)
    layout = args.layout or shape.layout or "typed"
    if args.layout is None and shape.layout:
        print(f"note: {root} holds {shape.describe()}; using the {layout} layout")
    vendor_manifest = args.vendor_manifest
    if vendor_manifest is None:
        vendor_manifest = not shape.leaf_only
        if shape.leaf_only:
            print(
                "note: this package models each provider as its own vendor; "
                "the credential and the identity go on the leaf"
            )
    base_url = args.base_url or None
    if base_url and args.probe:
        base_url, note = normalize_base_url(base_url)
        if note:
            print(note)
        ok, detail = probe_base_url(base_url)
        print(f"{'ok:  ' if ok else 'WARN:'} {detail}")
    logo = args.logo or None
    if logo is None and args.auto_logo:
        logo = find_lobehub_logo(args.vendor)
        if logo is None:
            print(
                f"error: no lobehub icon found for {args.vendor!r}; pass --logo "
                f"with the URL you want",
                file=sys.stderr,
            )
            return 2
        print(f"logo    {logo} (lobehub, {args.vendor})")
    if logo and args.localize_logo:
        logo, note = localize_logo(logo, root, args.vendor)
        print(note)
    if _guard is not None:
        _guard.writing = True
    return scaffold(
        root,
        args.vendor,
        format_=args.format,
        model_type=args.model_type,
        layout=layout,
        base_url=base_url,
        models_path=args.models_path or None,
        api_key_env=args.api_key_env,
        display_name=args.display_name or None,
        logo=logo,
        group=args.group,
        group_label=args.group_label or None,
        vendor_manifest=vendor_manifest,
        registry_manifest=not args.no_registry_manifest,
        transform=args.transform,
        check=args.check,
        name=args.name,
    )


def main(argv: list[str] | None = None, *, default_root: str = "providers") -> int:
    """Standalone entry point, for the repository script and for tests."""
    parser = argparse.ArgumentParser(
        prog="new_shim", description="Create the files for a new provider shim."
    )
    add_arguments(parser, default_root=default_root)
    return run(parser.parse_args(argv))


def _namespace(**overrides: Any) -> argparse.Namespace:
    """An all-defaults namespace, for callers that need to build one."""
    parser = argparse.ArgumentParser()
    add_arguments(parser, default_root="providers")
    args = parser.parse_args(["--no-input"])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args
