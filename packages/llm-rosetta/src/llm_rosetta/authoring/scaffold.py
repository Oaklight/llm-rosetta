"""Scaffold the files for a new provider shim.

A shim is a small, boring set of files — a ``provider.yaml``, sometimes a
``transforms.py``, and often a vendor manifest above them — and getting the
shape wrong is not caught until a gateway loads it.  This module writes that
shape, either from flags or by asking for each field, and then *loads* it: the
last step runs the real loader over the new leaf, so a scaffold no gateway
could use fails here rather than at startup.

The command is ``scripts/dev/new_shim.py`` (a repository checkout, where the
built-in tree is the default target); ``llm-rosetta ext new`` renders an
extension folder from its own templates.

Layouts::

    typed (default)   <root>/<vendor>/llm/<format>/provider.yaml
                      <root>/<vendor>/<type>/provider.yaml      (non-chat)
    flat              <root>/<vendor>/<format>/provider.yaml      (argo, alcf)

``<root>`` is the directory a gateway scans: the built-in ``providers`` tree.
"""

from __future__ import annotations

import argparse
import re
import signal
import sys
import threading
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from llm_rosetta.contracts import AUTH_MODULE, QUOTA_MODULE, TRANSFORMS_MODULE

from .tree_contract import TYPE_DIR_NAMES, yaml_document

MODEL_TYPES = ("llm", "embeddings", "rerank", "decisions")


#: The icon set the admin panel already shows beside a vendor, served by jsDelivr.
#: One SVG per provider slug; a vendor name that matches one gets it offered.
LOBEHUB_ICON_URL = (
    "https://cdn.jsdelivr.net/npm/@lobehub/icons-static-svg@latest/icons/{slug}.svg"
)

#: Suffixes a vendor directory adds around the provider it wraps.
_VENDOR_SUFFIXES = ("_free", "-free", "_zen", "-zen", "_pro", "-pro")


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
                f"({', '.join(legacy)}); the tree check (`run_scaffold_check`, or "
                f"`llm-rosetta ext check` on an extension folder) reports the "
                f"canonical path"
            )
    return None, notes


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
    args.vendor = args.vendor or _ask("Vendor directory name (e.g. together)")
    while (error := vendor_name_error(args.vendor)) is not None:
        print(f"  {error}")
        args.vendor = _ask("Vendor directory name (e.g. together)")
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
        "vendor", nargs="?", default="", help="Vendor directory name, e.g. together"
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
        transform=args.transform,
        auth=args.auth,
        quota=args.quota,
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
