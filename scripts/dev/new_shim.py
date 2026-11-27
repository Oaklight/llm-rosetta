#!/usr/bin/env python3
"""Scaffold the files of a new provider shim into the built-in tree.

The wheel holds the generator (:mod:`llm_rosetta.authoring.scaffold`: the
templates, the layout rules, :func:`~llm_rosetta.authoring.scaffold.scaffold`)
and ``llm-rosetta ext new`` renders an extension folder.  This command is how a
checkout adds a vendor to the built-in tree, which is not writable in an
installed package: it asks for each field (or reads flags), probes the base URL,
looks up the vendor's icon, writes the files through the generator and *loads*
them with the real loader.  Only the people who add built-in vendors need the
prompts, the probe and the icon lookup, so they live here and not in the wheel.

Layouts::

    typed (default)   <root>/<vendor>/llm/<format>/provider.yaml
                      <root>/<vendor>/<type>/provider.yaml      (non-chat)
    flat              <root>/<vendor>/<format>/provider.yaml      (argo, alcf)

``<root>`` is the directory a gateway scans: the built-in ``shims/builtin`` tree.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from llm_rosetta.authoring.scaffold import scaffold, vendor_name_error
from llm_rosetta.model_vocabulary import MODEL_TYPES

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILTIN_ROOT = REPO_ROOT / "packages/llm-rosetta/src/llm_rosetta/shims/builtin"

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


def main(
    argv: list[str] | None = None, *, default_root: str = str(BUILTIN_ROOT)
) -> int:
    """Run the scaffolder; the built-in tree is the default target."""
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


if __name__ == "__main__":
    raise SystemExit(main())
