"""The shim tree contract: where a shim's files may sit, and how to get there.

A vendor directory carries a manifest rather than a leaf, every leaf sits under
a model-type directory, and the format level appears only when a type holds
more than one lane (``_shim_template/README_en.md`` states it).  The functions
here read a tree against that rule (:func:`check_tree`), plan the moves that put
a tree on it (:func:`plan_migration`) and apply them (:func:`apply_migration`),
so the scaffolder that writes a tree, the check and the migration cannot
disagree about the shape.
"""

from __future__ import annotations

import os
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_rosetta.model_vocabulary import MODEL_TYPES

#: Directory names the loader reads as model types rather than vendors.
TYPE_DIR_NAMES = frozenset(MODEL_TYPES)

#: Keys that make a ``provider.yaml`` a shim rather than a vendor manifest.
_IDENTITY_KEYS = ("name", "format", "base")


def yaml_document(path: Path) -> dict[str, Any] | None:
    """Read a ``provider.yaml`` as a mapping, or ``None`` when it is not one."""
    from llm_rosetta._vendor.yaml import load as yaml_load

    try:
        document = yaml_load(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a broken file is not this tool's business
        return None
    return document if isinstance(document, dict) else None


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
    positions = [i for i, part in enumerate(tail) if part in TYPE_DIR_NAMES]
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
    positions = [i for i, part in enumerate(tail) if part in TYPE_DIR_NAMES]
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


#: Folders of an extension that hold no shim tree.
_NOT_TREE = frozenset({"tests", "build", "dist", "__pycache__"})


def _provider_yamls(root: Path, only: str | None) -> list[Path]:
    """The ``provider.yaml`` files below *root*, or below its vendor *only*."""
    if only is None:
        return sorted(root.rglob("provider.yaml"))
    base = root / only
    return [
        path
        for path in sorted(base.rglob("provider.yaml"))
        if not set(path.relative_to(base).parts[:-1]) & _NOT_TREE
        and not any(part.endswith(".egg-info") for part in path.relative_to(base).parts)
    ]


def _vendor_dirs(root: Path, only: str | None) -> list[Path]:
    """The vendor directories of *root*, or just *only*."""
    if only is None:
        return sorted(root.iterdir())
    return [root / only] if (root / only).is_dir() else []


def _token_command_problems(root: Path, only: str | None = None) -> list[Finding]:
    """Report a relative ``token_command`` entry whose file is not there.

    A credential-exchange command runs with the leaf directory as its working
    directory, so ``../token.py`` is resolved against it.  A leaf that moved —
    which is what ``shims migrate`` does — leaves such a path pointing at one
    more level of parent than the file it names, and nothing fails until the
    first exchange.  Only entries that look like a path are checked: an
    interpreter name or a flag is not a file to look for.
    """
    findings: list[Finding] = []
    for provider_yaml in _provider_yamls(root, only):
        if any(
            part.startswith(("_", "."))
            for part in provider_yaml.relative_to(root).parts
        ):
            continue
        document = yaml_document(provider_yaml)
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


def _module_problems(root: Path, only: str | None = None) -> list[Finding]:
    """Report the places a module sits where the loader will not read it."""
    findings: list[Finding] = []
    for vendor_dir in _vendor_dirs(root, only):
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


def _field_problems(root: Path, only: str | None = None) -> list[Finding]:
    """Report a ``fields:`` block the loader would refuse.

    The panel renders those entries verbatim, so the loader validates them
    strictly and one bad entry stops the whole root from loading.  Running the
    loader's own validator here turns that into a finding that names the file,
    before the tree is installed rather than on the gateway's first read.
    """
    from llm_rosetta.shims._loader import _declared_fields

    findings: list[Finding] = []
    for provider_yaml in _provider_yamls(root, only):
        if any(
            part.startswith(("_", "."))
            for part in provider_yaml.relative_to(root).parts
        ):
            continue
        document = yaml_document(provider_yaml)
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


def check_tree(root: Path, *, only: str | None = None) -> list[Finding]:
    """Report every place *root* deviates from the shim tree contract.

    The contract is the one ``_shim_template/README_en.md`` states: a vendor
    directory carries a manifest rather than a leaf, every leaf sits under a
    model-type directory, and the format level appears only when a type holds
    more than one lane.  Private helper modules are not reported: the loader
    opens none of them directly, and a module one of the five allowed files
    imports — ``transforms.py`` may — is the package's own business.

    Args:
        root: The directory a gateway scans.
        only: The name of one vendor directory of *root* to check alone: an
            extension folder is one vendor, and its siblings are not its tree.

    Returns:
        The findings, sorted by path.  An empty list means the tree conforms.
    """
    if not root.is_dir():
        return [Finding(root, "is not a directory")]
    findings: list[Finding] = []
    for provider_yaml in _provider_yamls(root, only):
        parts = provider_yaml.relative_to(root).parts
        if any(part.startswith(("_", ".")) for part in parts):
            continue
        document = yaml_document(provider_yaml)
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
    findings.extend(_module_problems(root, only))
    findings.extend(_token_command_problems(root, only))
    findings.extend(_field_problems(root, only))
    return sorted(findings, key=lambda f: str(f.path))


def run_check(root: Path) -> int:
    """Print the findings for *root* and return a process exit code."""
    findings = check_tree(root)
    if not findings:
        print(f"ok      {root}: no deviation from the shim contract")
        return 0
    for finding in findings:
        print(f"problem {finding.render(root)}")
    print(f"\n{len(findings)} finding(s); see authoring/_shim_template/README_en.md")
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
    document = yaml_document(provider_yaml)
    if document is None or not _is_leaf_document(document):
        return None, ""
    model_type = document.get("model_type") or "llm"
    format_ = document.get("format") or document.get("base")
    if not isinstance(model_type, str) or model_type not in TYPE_DIR_NAMES:
        return None, f"{provider_yaml}: unknown model_type {model_type!r}; left alone"
    tail = list(provider_yaml.relative_to(root).parts[:-1])
    below = tail[1:]
    type_at = next((i for i, part in enumerate(below) if part in TYPE_DIR_NAMES), None)
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


def plan_migration(
    root: Path, *, only: str | None = None
) -> tuple[list[Move], list[str]]:
    """Plan the moves that put *root* on the contract.

    Only the mechanical part is planned: a leaf that sits at the wrong depth,
    and a vendor directory used as a leaf.  A leaf whose type cannot be read,
    or that sits deeper than a lane and a format, is reported and left alone —
    a guess there would move code the author has to reconcile by hand.

    Args:
        root: The directory a gateway scans.
        only: The name of one vendor directory of *root* to plan alone.

    Returns:
        ``(moves, notes)``; apply the moves with :func:`apply_migration`.
    """
    notes: list[str] = []
    groups: dict[tuple[str, tuple[str, ...], str], list[dict[str, Any]]] = {}
    leaf_dirs: set[Path] = set()
    for provider_yaml in _provider_yamls(root, only):
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


def apply_migration(root: Path, moves: list[Move]) -> list[str]:
    """Perform the planned moves and report what moved.

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
