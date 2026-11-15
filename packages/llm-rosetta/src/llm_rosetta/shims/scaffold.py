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
import os
import re
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Collection
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

#: The template this module renders.  A shim's shape is documentation first:
#: the files below are what the guide shows and what the scaffold writes, so
#: neither can describe a tree the other does not mean.
TEMPLATE_DIR = Path(__file__).with_name("_shim_template")
TREE_DIR = TEMPLATE_DIR / "tree"
PACKAGE_DIR = TEMPLATE_DIR / "package"

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")

#: The leaf modules a template may ship beside ``provider.yaml``.
LEAF_MODULES = ("transforms.py", "auth.py", "quota.py")


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
    document = _yaml_document(directory / "provider.yaml") or {}
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


#: Directory names the loader reads as model types rather than vendors.
_TYPE_DIR_NAMES = frozenset(MODEL_TYPES)

#: Keys that make a ``provider.yaml`` a shim rather than a vendor manifest.
_IDENTITY_KEYS = ("name", "format", "base")


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
            and child.name not in _TYPE_DIR_NAMES
            and (child / "provider.yaml").is_file()
        ]
        if legacy:
            notes.append(
                f"{vendor} still holds format directories directly under it "
                f"({', '.join(legacy)}); `llm-rosetta-gateway shims check --root "
                f"{root}` reports the canonical path"
            )
    return None, notes


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
    lane: str | None = None,
    base_url: str | None = None,
    models_path: str | None = None,
    api_key_env: str | None = None,
    display_name: str | None = None,
    logo: str | None = None,
    group: str | None = None,
    group_label: str | None = None,
    registry_manifest: bool = True,
    transform: bool = False,
    auth: bool = False,
    quota: bool = False,
    package: str | None = None,
    check: bool = False,
    name: str | None = None,
) -> int:
    """Write the files for one shim from the template.

    Args:
        root: The directory a gateway scans — a built-in ``providers`` tree, or
            a package's ``providers`` directory.  With *package* it is the
            repository root instead, and the tree is written under
            ``src/<package>/providers``.
        vendor: Vendor directory name, e.g. ``acme``.
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
        registry_manifest: Add the leaf to ``<root>/shims.yaml`` when that file
            exists.
        transform: Write a ``transforms.py`` stub beside the leaf.
        auth: Write a vendor-level ``auth.py`` stub for a credential exchange.
        quota: Write a vendor-level ``quota.py`` stub.
        package: Distribution name; write the external package shell too.
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

    scan_root = root
    if package:
        package_module = package.replace("-", "_")
        scan_root = root / "src" / package_module / "providers"
        scan_root.mkdir(parents=True, exist_ok=True)
        _write_package_shell(root, package, package_module, vendor)

    error, notes = inspect_target(scan_root, vendor, name)
    if error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for note in notes:
        print(f"note: {note}")

    leaf = _resolve_leaf_dir(scan_root, vendor, model_type, lane, format_)
    if leaf is None:
        return 1

    written: list[Path] = []
    written.extend(
        _write_leaf_files(
            scan_root,
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

    note = _append_registry_entry(scan_root, name, leaf) if registry_manifest else None

    for path in written:
        print(f"wrote   {path}")
    if note:
        print(note)

    if check:
        return _check(scan_root, leaf, name)
    print(
        "\nnext: fill in the measured facts, add a test under tests/, and update "
        "docs_en + docs_zh."
    )
    return 0


def _resolve_leaf_dir(
    scan_root: Path,
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
    type_dir = _type_dir(scan_root, vendor, model_type, lane or None)
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


def _write_leaf_files(
    scan_root: Path,
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
    manifest = scan_root / vendor / "provider.yaml"
    if not manifest.is_file():
        _write_from_template(
            "tree/vendor/provider.yaml.tmpl",
            manifest,
            {
                "vendor": vendor,
                "logo": _yaml_scalar(logo) if logo else "",
                "group": _yaml_scalar(group or vendor),
                "group_label": _yaml_scalar(group_label) if group_label else "",
                "api_key_env": _yaml_scalar(api_key_env),
            },
        )
        written.append(manifest)

    _write_from_template(
        "tree/vendor/type/format/provider.yaml.tmpl",
        leaf / "provider.yaml",
        {
            "vendor": vendor,
            "name": _yaml_scalar(name),
            "model_type": model_type,
            "format": _yaml_scalar(format_),
            "base_url": _yaml_scalar(base_url or "https://example.invalid/v1"),
            "display_name": _yaml_scalar(display_name) if display_name else "",
            "models_path": _yaml_scalar(models_path) if models_path else "",
        },
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
        destination = scan_root / vendor / module
        _write_from_template(
            f"tree/vendor/{module}.tmpl",
            destination,
            {"name": name, "vendor": vendor, "format": format_},
        )
        written.append(destination)
    return written


def _write_package_shell(
    repo: Path, distribution: str, package_module: str, vendor: str
) -> list[Path]:
    """Write the files that make *repo* an installable shim package.

    A file that already exists is left alone: the shell carries the release
    workflow and the version, which a repository owns once it exists.

    Returns:
        The files that were written.
    """
    from llm_rosetta import __version__ as core_version

    values = {
        "distribution": distribution,
        "package": package_module,
        "vendor": vendor,
        "core_floor": core_version,
    }
    written: list[Path] = []
    for template in sorted(PACKAGE_DIR.rglob("*.tmpl")):
        relative = template.relative_to(PACKAGE_DIR).as_posix()
        destination = repo / relative.removesuffix(".tmpl")
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            _render(template.read_text(encoding="utf-8"), values), encoding="utf-8"
        )
        written.append(destination)
    for path in written:
        print(f"wrote   {path}")
    return written


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
    args.lane = _ask("Lane directory (optional; ALCF's clusters)", args.lane or "")
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
        "--lane",
        default=None,
        help="A named instance of the vendor (ALCF's clusters), written "
        "between the vendor and its type directories",
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
        "--auth",
        action="store_true",
        help="Write a vendor-level auth.py stub for a credential exchange",
    )
    parser.add_argument(
        "--quota",
        action="store_true",
        help="Write a vendor-level quota.py stub",
    )
    parser.add_argument(
        "--package",
        metavar="DISTRIBUTION",
        help="Write the external package shell into --root as well, e.g. "
        "llm-rosetta-shim-acme; the tree then goes to src/<package>/providers",
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
    if shape.leaf_only:
        print(
            "note: this package models each provider as its own vendor leaf; "
            "the scaffold now writes <vendor>/<type>/ — run `shims check` for "
            "the canonical path of what is already there"
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
        lane=args.lane or None,
        base_url=base_url,
        models_path=args.models_path or None,
        api_key_env=args.api_key_env,
        display_name=args.display_name or None,
        logo=logo,
        group=args.group,
        group_label=args.group_label or None,
        registry_manifest=not args.no_registry_manifest,
        transform=args.transform,
        auth=args.auth,
        quota=args.quota,
        package=args.package,
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


# ---------------------------------------------------------------------------
# shims check — report a tree that does not follow the contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """One deviation from the shim tree contract.

    Attributes:
        path: The file the deviation is about.
        message: What is wrong, in the terms of the contract.
    """

    path: Path
    message: str

    def render(self, root: Path) -> str:
        """Return the line the CLI prints."""
        try:
            shown = self.path.relative_to(root).as_posix()
        except ValueError:
            shown = str(self.path)
        return f"{shown}: {self.message}"


def _is_leaf_document(document: dict[str, Any]) -> bool:
    """Whether a ``provider.yaml`` declares a shim rather than a manifest."""
    return any(document.get(key) for key in _IDENTITY_KEYS)


def _lane_or_format_problem(root: Path, tail: list[str]) -> str:
    """Why a leaf's directory is not canonical, as one sentence."""
    positions = [i for i, part in enumerate(tail) if part in _TYPE_DIR_NAMES]
    if not positions:
        return (
            "no <type>/ directory above the leaf; the leaf belongs at "
            "<vendor>/<type>/[<format>/]"
        )
    first = positions[0]
    if len(positions) > 1:
        return "more than one <type>/ directory above the leaf"
    if first > 2:
        return "more than one lane directory above the <type>/ directory"
    if len(tail) - first > 2:
        return "more than one directory below the <type>/ directory"
    if len(tail) - first == 2:
        type_dir = root.joinpath(*tail[: first + 1])
        formats = [
            child
            for child in sorted(type_dir.iterdir())
            if child.is_dir() and not child.name.startswith(("_", "."))
        ]
        if len(formats) == 1 and not (type_dir / "provider.yaml").is_file():
            return (
                f"the only lane of {tail[first]}; it belongs directly under "
                f"{type_dir.relative_to(root).as_posix()}/, and the format "
                f"directory appears when a second lane arrives"
            )
    return ""


def _mixed_level_problem(root: Path, tail: list[str]) -> str:
    """Why a leaf that sits in a type directory shares it with lanes.

    A type directory holding a leaf *and* format directories is the one shape
    the loader reads halfway: ``_load_type_folder`` takes the leaf and never
    scans below it, so the format lanes load nowhere and nothing says so.  The
    advice has to differ from the only-lane case, where the lone lane moves up —
    here the flat leaf is the one that moves down.
    """
    positions = [i for i, part in enumerate(tail) if part in _TYPE_DIR_NAMES]
    if len(positions) != 1 or len(tail) - positions[0] != 1:
        return ""
    type_dir = root.joinpath(*tail)
    formats = sorted(
        child.name
        for child in type_dir.iterdir()
        if child.is_dir()
        and not child.name.startswith(("_", "."))
        and (child / "provider.yaml").is_file()
    )
    if not formats:
        return ""
    return (
        f"{type_dir.relative_to(root).as_posix()}/ holds a leaf and format "
        f"directories at once ({', '.join(formats)}); the loader reads only "
        f"provider.yaml there and never scans them. Move this leaf down into "
        f"{type_dir.relative_to(root).as_posix()}/<its format>/"
    )


def _token_command_problems(root: Path) -> list[Finding]:
    """Report a relative ``token_command`` entry whose file is not there.

    A credential-exchange command runs with the leaf directory as its working
    directory, so ``../token.py`` is resolved against it.  A leaf that moved —
    which is what ``shims migrate`` does — leaves such a path pointing at one
    more level of parent than the file it names, and nothing fails until the
    first exchange.  Only entries that look like a path are checked: an
    interpreter name or a flag is not a file to look for.
    """
    findings: list[Finding] = []
    for provider_yaml in sorted(root.rglob("provider.yaml")):
        if any(
            part.startswith(("_", "."))
            for part in provider_yaml.relative_to(root).parts
        ):
            continue
        document = _yaml_document(provider_yaml)
        if not isinstance(document, dict):
            continue
        connection = document.get("connection")
        if not isinstance(connection, dict):
            continue
        command = connection.get("token_command")
        entries = command if isinstance(command, list) else [command]
        for entry in entries:
            if not isinstance(entry, str) or entry.startswith("-"):
                continue
            if "/" not in entry and not entry.endswith(".py"):
                continue
            target = Path(entry)
            if target.is_absolute():
                continue
            if not (provider_yaml.parent / target).exists():
                findings.append(
                    Finding(
                        provider_yaml,
                        f"connection.token_command runs relative to this "
                        f"directory, and {entry} is not here",
                    )
                )
    return findings


def _module_problems(root: Path) -> list[Finding]:
    """Report the places a module sits where the loader will not read it."""
    findings: list[Finding] = []
    for vendor_dir in sorted(root.iterdir()):
        if not vendor_dir.is_dir() or vendor_dir.name.startswith(("_", ".")):
            continue
        transforms = vendor_dir / "transforms.py"
        if transforms.is_file():
            findings.append(
                Finding(
                    transforms,
                    "the loader imports transforms.py beside a leaf only; "
                    "a vendor-level one is never read",
                )
            )
        vendor_auth = vendor_dir / "auth.py"
        if vendor_auth.is_file():
            for leaf_auth in sorted(vendor_dir.rglob("auth.py")):
                if leaf_auth == vendor_auth:
                    continue
                # ``_``/``.`` directories are where a package parks what the
                # loader must not read, so a module there is not a second
                # exchange — no scan ever reaches it.
                if any(
                    part.startswith(("_", "."))
                    for part in leaf_auth.relative_to(vendor_dir).parts
                ):
                    continue
                findings.append(
                    Finding(
                        leaf_auth,
                        f"the vendor already ships its exchange in {vendor_auth}; "
                        "a second one is refused at load",
                    )
                )
    return findings


def _field_problems(root: Path) -> list[Finding]:
    """Report a ``fields:`` block the loader would refuse.

    The panel renders those entries verbatim, so the loader validates them
    strictly and one bad entry stops the whole root from loading.  Running the
    loader's own validator here turns that into a finding that names the file,
    before the tree is installed rather than on the gateway's first read.
    """
    from llm_rosetta.shims.providers import _declared_fields

    findings: list[Finding] = []
    for provider_yaml in sorted(root.rglob("provider.yaml")):
        if any(
            part.startswith(("_", "."))
            for part in provider_yaml.relative_to(root).parts
        ):
            continue
        document = _yaml_document(provider_yaml)
        if not isinstance(document, dict) or "fields" not in document:
            continue
        try:
            _declared_fields(document.get("fields"), provider_yaml)
        except ValueError as exc:
            # The validator names the file itself; the finding already carries
            # the path, so only the sentence after it is kept.
            prefix = f"{provider_yaml}: "
            message = str(exc)
            findings.append(
                Finding(
                    provider_yaml,
                    message[len(prefix) :] if message.startswith(prefix) else message,
                )
            )
    return findings


def _manifest_problems(root: Path) -> list[Finding]:
    """Report a ``shims.yaml`` entry whose directory holds no leaf."""
    manifest = root / "shims.yaml"
    if not manifest.is_file():
        return []
    from llm_rosetta._vendor.yaml import load as yaml_load

    try:
        document = yaml_load(manifest.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001 - the loader reports the same thing
        return [Finding(manifest, f"cannot be read: {exc}")]
    entries = document.get("shims") if isinstance(document, dict) else None
    if not isinstance(entries, list):
        return [Finding(manifest, "declares no 'shims' list")]
    findings: list[Finding] = []
    for entry in entries:
        if not isinstance(entry, dict):
            findings.append(Finding(manifest, "an entry is not a mapping"))
            continue
        relative = entry.get("path")
        name = entry.get("name") or "?"
        if not isinstance(relative, str) or not relative:
            findings.append(Finding(manifest, f"{name}: declares no 'path'"))
            continue
        if not (root / relative / "provider.yaml").is_file():
            findings.append(
                Finding(manifest, f"{name}: {relative}/ holds no provider.yaml")
            )
    return findings


def check_tree(root: Path) -> list[Finding]:
    """Report every place *root* deviates from the shim tree contract.

    The contract is the one ``_shim_template/README_en.md`` states: a vendor
    directory carries a manifest rather than a leaf, every leaf sits under a
    model-type directory, and the format level appears only when a type holds
    more than one lane.  Private helper modules are not reported: the loader
    opens none of them directly, and a module one of the five allowed files
    imports — ``transforms.py`` may — is the package's own business.

    Args:
        root: The directory a gateway scans.

    Returns:
        The findings, sorted by path.  An empty list means the tree conforms.
    """
    if not root.is_dir():
        return [Finding(root, "is not a directory")]
    findings: list[Finding] = []
    for provider_yaml in sorted(root.rglob("provider.yaml")):
        parts = provider_yaml.relative_to(root).parts
        if any(part.startswith(("_", ".")) for part in parts):
            continue
        document = _yaml_document(provider_yaml)
        if document is None:
            findings.append(Finding(provider_yaml, "is not a YAML mapping"))
            continue
        leaf = _is_leaf_document(document)
        tail = list(parts[:-1])
        if len(tail) == 1:
            if leaf:
                findings.append(
                    Finding(
                        provider_yaml,
                        "a vendor directory is a leaf; the vendor carries a "
                        "manifest and the leaf goes under <type>/",
                    )
                )
            continue
        if not leaf:
            findings.append(
                Finding(
                    provider_yaml,
                    "a manifest below the vendor root; only a vendor directory "
                    "may carry one",
                )
            )
            continue
        problem = _mixed_level_problem(root, tail) or _lane_or_format_problem(
            root, tail
        )
        if problem:
            findings.append(Finding(provider_yaml, problem))
    findings.extend(_module_problems(root))
    findings.extend(_manifest_problems(root))
    findings.extend(_token_command_problems(root))
    findings.extend(_field_problems(root))
    return sorted(findings, key=lambda f: str(f.path))


def run_check(root: Path) -> int:
    """Print the findings for *root* and return a process exit code."""
    findings = check_tree(root)
    if not findings:
        print(f"ok      {root}: no deviation from the shim contract")
        return 0
    for finding in findings:
        print(f"problem {finding.render(root)}")
    print(f"\n{len(findings)} finding(s); see shims/_shim_template/README_en.md")
    return 1


# ---------------------------------------------------------------------------
# shims migrate — move a tree onto the contract, mechanically
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Move:
    """One directory the migration relocates.

    Attributes:
        source: The directory holding the leaf today.
        destination: Where the contract puts it.
        vendor_leaf: True when *source* is the vendor root itself, in which
            case the migration leaves a manifest marker behind.
        keep: Names inside *source* that belong to another leaf and must not
            travel.  A leaf that sits in a directory another leaf also lives in
            -- the flat lane of a type that already holds a sibling lane -- is
            moved entry by entry, not as a directory, or the sibling would be
            dragged one level down with it.
    """

    source: Path
    destination: Path
    vendor_leaf: bool = False
    keep: tuple[str, ...] = ()


def _leaf_entry(root: Path, provider_yaml: Path) -> tuple[dict[str, Any] | None, str]:
    """Describe one leaf for the migration, or say why it is left alone.

    Returns:
        ``(entry, note)`` — exactly one of them is set.
    """
    document = _yaml_document(provider_yaml)
    if document is None or not _is_leaf_document(document):
        return None, ""
    model_type = document.get("model_type") or "llm"
    format_ = document.get("format") or document.get("base")
    if not isinstance(model_type, str) or model_type not in _TYPE_DIR_NAMES:
        return None, f"{provider_yaml}: unknown model_type {model_type!r}; left alone"
    tail = list(provider_yaml.relative_to(root).parts[:-1])
    below = tail[1:]
    type_at = next((i for i, part in enumerate(below) if part in _TYPE_DIR_NAMES), None)
    if type_at is None:
        lanes = tuple(below[:-1])
        name_dir = below[-1] if below else ""
        if name_dir and name_dir != format_:
            lanes = (*lanes, name_dir)
    else:
        lanes = tuple(below[:type_at])
    if len(lanes) > 1:
        return None, f"{provider_yaml}: more than one lane directory; left alone"
    return (
        {
            "key": (tail[0], lanes, model_type),
            "path": provider_yaml,
            "directory": provider_yaml.parent,
            "format": format_,
            "vendor_leaf": len(tail) == 1,
        },
        "",
    )


def _group_moves(
    root: Path,
    key: tuple[str, tuple[str, ...], str],
    leaves: list[dict[str, Any]],
    notes: list[str],
    leaf_dirs: Collection[Path],
) -> list[Move]:
    """Turn one ``(vendor, lanes, type)`` group into its moves.

    A group of one lane belongs directly under the type directory; a group of
    several needs a format level, because a type directory holds one leaf.
    The leaves of a directory another leaf also occupies are moved by name, so
    a sibling lane stays where it is.

    Args:
        root: The directory a gateway scans.
        key: The ``(vendor, lanes, model_type)`` this group shares.
        leaves: The leaves of this group.
        notes: Collected operator notes; appended to.
        leaf_dirs: Every leaf directory in the tree, across all groups, so a
            sibling of a different model type is kept too.
    """
    vendor, lanes, model_type = key
    base = root.joinpath(vendor, *lanes, model_type)
    moves: list[Move] = []
    for leaf in leaves:
        if len(leaves) == 1:
            destination = base
        elif not leaf["format"]:
            notes.append(
                f"{leaf['path']}: declares no 'format:' but shares its type "
                f"with another lane; left alone"
            )
            continue
        else:
            destination = base / leaf["format"]
        if destination == leaf["directory"]:
            continue
        shared = tuple(
            sorted(
                other.name
                for other in leaf_dirs
                if other != leaf["directory"] and other.parent == leaf["directory"]
            )
        )
        moves.append(
            Move(
                leaf["directory"],
                destination,
                vendor_leaf=leaf["vendor_leaf"],
                keep=shared,
            )
        )
    return moves


def plan_migration(root: Path) -> tuple[list[Move], list[str]]:
    """Plan the moves that put *root* on the contract.

    Only the mechanical part is planned: a leaf that sits at the wrong depth,
    and a vendor directory used as a leaf.  A leaf whose type cannot be read,
    or that sits deeper than a lane and a format, is reported and left alone —
    a guess there would move code the author has to reconcile by hand.

    Args:
        root: The directory a gateway scans.

    Returns:
        ``(moves, notes)``; apply the moves with :func:`apply_migration`.
    """
    notes: list[str] = []
    groups: dict[tuple[str, tuple[str, ...], str], list[dict[str, Any]]] = {}
    leaf_dirs: set[Path] = set()
    for provider_yaml in sorted(root.rglob("provider.yaml")):
        parts = provider_yaml.relative_to(root).parts
        if any(part.startswith(("_", ".")) for part in parts):
            continue
        entry, note = _leaf_entry(root, provider_yaml)
        if note:
            notes.append(note)
        if entry is None:
            continue
        leaf_dirs.add(entry["directory"])
        groups.setdefault(entry.pop("key"), []).append(entry)

    moves: list[Move] = []
    for key, leaves in sorted(groups.items()):
        moves.extend(_group_moves(root, key, leaves, notes, leaf_dirs))
    return moves, notes


def _move_entries(root: Path, move: Move) -> list[str]:
    """Move one leaf directory's contents, and report what was done."""
    actions: list[str] = []
    # Snapshot before the destination exists: when it sits inside the source
    # (``alcf/metis`` -> ``alcf/metis/llm``) the destination would otherwise
    # appear in its own move list.
    entries = sorted(move.source.iterdir())
    move.destination.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        # An entry that is the destination, or holds it, stays: moving it would
        # carry the tree being written into itself.
        if entry == move.destination or move.destination.is_relative_to(entry):
            continue
        if entry.name in move.keep:
            actions.append(f"kept    {entry} (another leaf)")
            continue
        os.rename(entry, move.destination / entry.name)
    if move.vendor_leaf:
        marker = move.source / "provider.yaml"
        marker.write_text(
            "# Vendor manifest: the defaults every "
            f"{move.source.name} shim shares.\n"
            "# A leaf inherits every field here and may override any of them.\n"
            "# This lane declares its own connection values, so the block is\n"
            "# empty on purpose.\n"
            "connection: {}\n",
            encoding="utf-8",
        )
        actions.append(f"wrote   {marker} (manifest marker)")
    else:
        try:
            move.source.rmdir()
        except OSError:
            pass
    actions.append(
        f"moved   {move.source.relative_to(root)} -> "
        f"{move.destination.relative_to(root)}"
    )
    return actions


_MANIFEST_KEY_ORDER = ("version", "core", "shims")


def _manifest_value(value: Any) -> str | None:
    """Render *value* as a one-line YAML scalar, or ``None`` when it is not one."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "null"
    if isinstance(value, str):
        return _yaml_scalar(value)
    return None


def _render_manifest_header(document: dict[str, Any]) -> list[str] | None:
    """The top-level keys, in a stable order, or ``None`` on a non-scalar.

    ``version`` and ``core`` lead; anything else the operator wrote keeps its
    order behind them, and a missing ``version`` is written rather than lost.
    """
    ordered = [key for key in ("version", "core") if key in document]
    ordered += [key for key in document if key not in _MANIFEST_KEY_ORDER]
    lines: list[str] = []
    for key in ordered:
        rendered = _manifest_value(document[key])
        if rendered is None:
            return None
        lines.append(f"{key}: {rendered}")
    if "version" not in document:
        lines.insert(0, "version: 1")
    return lines


def _render_manifest_entry(entry: Any) -> list[str] | None:
    """One ``shims:`` item, or ``None`` when a value is not a scalar."""
    if not isinstance(entry, dict):
        return None
    ordered = [key for key in ("name", "path") if key in entry]
    ordered += [key for key in entry if key not in ("name", "path")]
    lines: list[str] = []
    for index, key in enumerate(ordered):
        rendered = _manifest_value(entry[key])
        if rendered is None:
            return None
        lines.append(("  - " if index == 0 else "    ") + f"{key}: {rendered}")
    return lines


def _render_manifest(document: dict[str, Any], entries: list[Any]) -> list[str] | None:
    """Rebuild ``shims.yaml`` from its parsed document, or refuse.

    Returns:
        The file's lines, or ``None`` when a value is not a scalar this writer
        can render — a list or a nested mapping is the operator's to keep, and
        rewriting it lossily would be worse than leaving the file alone.
    """
    header = _render_manifest_header(document)
    if header is None:
        return None
    lines = [*header, "shims:"]
    for entry in entries:
        rendered = _render_manifest_entry(entry)
        if rendered is None:
            return None
        lines.extend(rendered)
    return lines


def _rewrite_manifest(root: Path, moves: list[Move]) -> list[str]:
    """Point every ``shims.yaml`` entry at where its leaf moved to.

    Only ``path`` is rewritten.  Every other key stays as it was — ``core``,
    and any the writer does not know — and the file is rebuilt from its parsed
    document, so comments do not survive.  A document holding a value that is
    not a scalar (a list, a nested mapping) is left untouched with a note
    instead: a lossy rewrite of the operator's file is worse than a path they
    fix by hand.
    """
    manifest = root / "shims.yaml"
    if not moves or not manifest.is_file():
        return []
    from llm_rosetta._vendor.yaml import load as yaml_load

    document = yaml_load(manifest.read_text(encoding="utf-8")) or {}
    if not isinstance(document, dict):
        return [f"left {manifest.relative_to(root)} alone: it is not a mapping"]
    entries = document.get("shims")
    if entries is None:
        entries = []
    if not isinstance(entries, list):
        return [f"left {manifest.relative_to(root)} alone: 'shims' is not a list"]
    relocated = [(Path(str(m.source)), Path(str(m.destination))) for m in moves]
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            continue
        old = root / entry["path"]
        for source, destination in relocated:
            if old == source or source in old.parents:
                suffix = old.relative_to(source)
                entry["path"] = (destination / suffix).relative_to(root).as_posix()
                break
    lines = _render_manifest(document, entries)
    if lines is None:
        return [
            f"left {manifest.relative_to(root)} alone: it holds a value this "
            "rewrite cannot render; update its paths by hand"
        ]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [f"rewrote {manifest.relative_to(root)}"]


def apply_migration(root: Path, moves: list[Move]) -> list[str]:
    """Perform the planned moves and rewrite ``shims.yaml`` paths.

    The moves are directory-level: every file beside the leaf — transforms,
    a leaf exchange, a quota module — travels with it.  Two things move with it
    and are deliberately not rewritten, because a tool that guesses there can
    break a module the author has to reconcile by hand:

    - **Import depth.**  ``from ..model_utils`` in a leaf one level higher
      needs one more dot after the move, and a skipped ``transforms.py`` fails
      quietly: the source still reports ``ok``, with the leaf's transforms
      missing.
    - **Relative paths in ``provider.yaml``.**  A ``connection.token_command``
      such as ``[python3, ../token.py]`` resolves against the leaf directory,
      so a file that stays behind stops resolving — and nothing fails until the
      first credential exchange.  ``shims check`` reports a relative entry whose
      target does not exist.

    The command prints the same warning after ``--write``; run the package's
    tests before trusting the result.

    Returns:
        One line per action, for the operator.
    """
    actions: list[str] = []
    for move in moves:
        actions.extend(_move_entries(root, move))
    actions.extend(_rewrite_manifest(root, moves))
    return actions


def run_migrate(root: Path, *, write: bool = False) -> int:
    """Plan (and optionally apply) the migration of *root*.

    Returns:
        ``0`` when there is nothing to do or the migration was applied;
        ``1`` when a dry run found work, so a script can tell the two apart.
    """
    moves, notes = plan_migration(root)
    for note in notes:
        print(f"note    {note}")
    if not moves:
        print(f"ok      {root}: nothing to migrate")
        return 0
    for move in moves:
        print(
            f"move    {move.source.relative_to(root)} -> "
            f"{move.destination.relative_to(root)}"
        )
    if not write:
        print(f"\n{len(moves)} move(s); rerun with --write to apply")
        return 1
    for action in apply_migration(root, moves):
        print(action)
    print(
        "\nnext: imports that moved one level deeper need a hand — check "
        "`from ..` against the new depth, then run the package's tests."
    )
    return 0
