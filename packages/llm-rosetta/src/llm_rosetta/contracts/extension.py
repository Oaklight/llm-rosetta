"""The extension manifest: what one extension folder declares in ``plugin.yaml``.

An extension is the installable unit: one folder, one distribution, one
``plugin.yaml`` beside its code.  It provides shims (a ``provider.yaml`` tree
with optional sidecars), transports (code that reaches a lane the host does not
ship), or both.  The host reads this manifest before it imports a line of the
extension, so a core that cannot run the extension skips it with a sentence
instead of failing on import.

The format is stated once, in ``extension_manifest.schema.json`` beside this
module.  :func:`validate_extension_manifest` interprets it with the schema
interpreter of :mod:`llm_rosetta.contracts._schema`, so no second copy of
the rules and no third-party validator exists.  This module takes the parsed
mapping; reading the YAML file is the loader's job, which keeps this module
standard-library only.

Compatibility (:func:`compatibility_problems`) is: ``requires.surface`` equals the core's
:data:`~llm_rosetta.contracts.SURFACE_VERSION`, ``requires.ir`` does not exceed
its :data:`~llm_rosetta.contracts.IR_VERSION` — plus ``platforms``.

Transport scope (:func:`resolve_transport`): a shim's ``connection.transport``
names its own extension's transport first, then a host built-in, then a
transport of an extension that this one lists under ``requires.extensions``.
An extension never reaches another extension's transport without saying so.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_rosetta.contracts import IR_VERSION, SURFACE_VERSION
from llm_rosetta.contracts.command import CORE_COMMAND_NAMES
from llm_rosetta.contracts._schema import _check

#: The manifest file an extension carries beside its code.
MANIFEST_FILENAME = "plugin.yaml"

#: The schema file, shipped beside this module.
SCHEMA_FILENAME = "extension_manifest.schema.json"

#: The maturity ladder below core, lowest first.
MATURITY_LEVELS: tuple[str, ...] = ("experimental", "beta", "stable")

#: The platform names ``platforms`` may list.
PLATFORMS: tuple[str, ...] = ("linux", "darwin", "windows")


def _shim_belongs_to(name: str, vendor: str) -> bool:
    """Whether shim *name* may be provided by extension *vendor*.

    A shim name is the extension id itself (the bare chat shims ``xai``,
    ``qwen``, kept unchanged so a ``type:`` value keeps resolving),
    ``<id>--<name>``, or ``<id>_<lane>--<name>`` for an extension that
    qualifies its lanes (``alcf_sophia--chat_completions`` under ``alcf``).

    Args:
        name: A shim name from a manifest.
        vendor: The extension id.
    """
    if name == vendor:
        return True
    head, sep, _ = name.partition("--")
    return bool(sep) and (head == vendor or head.startswith(f"{vendor}_"))


class ExtensionManifestInvalid(ValueError):
    """A manifest document does not follow the schema.

    Attributes:
        problems: One sentence per violation, each naming its path.
    """

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = tuple(problems)
        shown = "; ".join(self.problems[:5])
        more = len(self.problems) - 5
        if more > 0:
            shown += f"; and {more} more"
        super().__init__(f"invalid extension manifest: {shown}")


class UnresolvedTransport(LookupError):
    """A shim names a transport that its extension cannot reach."""


@dataclass(frozen=True)
class ExtensionRequirement:
    """What an extension needs from its host.

    Attributes:
        surface: The extension surface level it was built for.
        ir: The lowest IR level it reads.
        extensions: Ids of the extensions whose transports it names.
    """

    surface: int
    ir: int
    extensions: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExternalProgram:
    """A program an extension drives and does not ship.

    Attributes:
        name: The program's name.
        check: The command that reports its version.
        minimum: The oldest version the extension supports, if it states one.
        install_hint: What to tell the operator when it is missing.
    """

    name: str
    check: tuple[str, ...]
    minimum: str | None = None
    install_hint: str | None = None


@dataclass(frozen=True)
class ExtensionManifest:
    """One extension's declaration.

    Attributes:
        id: The vendor key, the import name and the identity.
        name: Human-readable name.
        version: The extension's version.
        maturity: One of :data:`MATURITY_LEVELS`.
        requires: What it needs from the host.
        shims: The shim names it registers.
        transports: The transport ids it implements.
        commands: The CLI verbs it adds under ``llm-rosetta <id>``, each mapped
            to its summary per language.
        tags: Free-form tags (``tier:free``, ``auth:cli``).
        homepage: Where to read about it.
        external: Programs it drives.
        platforms: The platforms it runs on; empty means any.
        summary: One line per language.
    """

    id: str
    name: str
    version: str
    maturity: str
    requires: ExtensionRequirement
    shims: tuple[str, ...] = ()
    transports: tuple[str, ...] = ()
    commands: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    homepage: str | None = None
    external: tuple[ExternalProgram, ...] = ()
    platforms: tuple[str, ...] = ()
    summary: Mapping[str, str] | None = None


@dataclass(frozen=True)
class TransportOwner:
    """Where a shim's transport comes from.

    Attributes:
        kind: ``"own"`` (the shim's own extension), ``"builtin"`` (the host
            ships it) or ``"dependency"`` (a required extension).
        extension: The providing extension id; ``None`` for a built-in.
    """

    kind: str
    extension: str | None


def load_extension_schema() -> dict[str, Any]:
    """Return the extension manifest JSON Schema shipped with this core."""
    path = Path(__file__).with_name(SCHEMA_FILENAME)
    return json.loads(path.read_text(encoding="utf-8"))


_SCHEMA_CACHE: dict[str, Any] = {}


def _schema() -> dict[str, Any]:
    if not _SCHEMA_CACHE:
        _SCHEMA_CACHE.update(load_extension_schema())
    return _SCHEMA_CACHE


def validate_extension_manifest(document: Any) -> list[str]:
    """Check *document* against the schema and the rules the schema cannot say.

    Beyond the schema: ``provides`` is not empty, every shim name belongs to the
    extension id, an extension that adds commands has an id the dispatcher can
    spell and does not take a core command name, and an extension does not
    require itself.

    Args:
        document: The parsed ``plugin.yaml`` value.

    Returns:
        One sentence per violation; empty when the document is valid.
    """
    root = _schema()
    problems: list[str] = []
    _check(document, root, "$", root, problems)
    if problems or not isinstance(document, Mapping):
        return problems
    ext_id = document["id"]
    provides = document["provides"]
    shims = provides.get("shims", [])
    transports = provides.get("transports", [])
    commands = provides.get("commands", {})
    if not shims and not transports and not commands:
        problems.append("$.provides: lists no shim, no transport and no command")
    if commands:
        area = command_area(ext_id)
        if area in CORE_COMMAND_NAMES:
            problems.append(
                f"$.provides.commands: the area {area!r} is a core command name; "
                "rename the extension"
            )
    for index, name in enumerate(shims):
        if not _shim_belongs_to(name, ext_id):
            problems.append(
                f"$.provides.shims[{index}]: {name!r} is not {ext_id!r}, "
                f"'{ext_id}--<name>' or '{ext_id}_<lane>--<name>'"
            )
    if ext_id in document["requires"].get("extensions", []):
        problems.append("$.requires.extensions: an extension cannot require itself")
    return problems


def parse_extension_manifest(document: Any) -> ExtensionManifest:
    """Validate *document* and build the manifest.

    Args:
        document: The parsed ``plugin.yaml`` value.

    Returns:
        The manifest.

    Raises:
        ExtensionManifestInvalid: The document breaks the schema or a rule.
    """
    problems = validate_extension_manifest(document)
    if problems:
        raise ExtensionManifestInvalid(problems)
    requires = document["requires"]
    provides = document["provides"]
    summary = document.get("summary")
    return ExtensionManifest(
        id=document["id"],
        name=document["name"],
        version=document["version"],
        maturity=document["maturity"],
        requires=ExtensionRequirement(
            surface=requires["surface"],
            ir=requires["ir"],
            extensions=tuple(requires.get("extensions", ())),
        ),
        shims=tuple(provides.get("shims", ())),
        transports=tuple(provides.get("transports", ())),
        commands={
            verb: dict(summary)
            for verb, summary in provides.get("commands", {}).items()
        },
        tags=tuple(document.get("tags", ())),
        homepage=document.get("homepage"),
        external=tuple(
            ExternalProgram(
                name=item["name"],
                check=tuple(item["check"]),
                minimum=item.get("minimum"),
                install_hint=item.get("install_hint"),
            )
            for item in document.get("external", ())
        ),
        platforms=tuple(document.get("platforms", ())),
        summary=dict(summary) if summary is not None else None,
    )


def command_area(ext_id: str) -> str:
    """Spell an extension id as the CLI area name: underscores become hyphens.

    Args:
        ext_id: The extension id (``[a-z][a-z0-9_]*``).

    Returns:
        The word after ``llm-rosetta`` that selects the extension's commands.
    """
    return ext_id.replace("_", "-")


def current_platform() -> str:
    """Name this process's platform in the vocabulary of ``platforms``."""
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "darwin"
    if sys.platform in ("win32", "cygwin"):
        return "windows"
    return sys.platform


def compatibility_problems(
    manifest: ExtensionManifest,
    *,
    surface: int | None = None,
    ir: int | None = None,
    platform: str | None = None,
) -> list[str]:
    """Why *manifest* cannot run on this core and platform.

    Args:
        manifest: The extension's manifest.
        surface: The surface level to check against; this core's by default.
        ir: The IR level to check against; this core's by default.
        platform: The platform to check against; this process's by default.

    Returns:
        One sentence per reason; empty when the extension can run.
    """
    problems: list[str] = []
    surface = SURFACE_VERSION if surface is None else surface
    ir = IR_VERSION if ir is None else ir
    if manifest.requires.surface != surface:
        problems.append(
            f"requires surface {manifest.requires.surface}, "
            f"this core provides surface {surface}"
        )
    if manifest.requires.ir > ir:
        problems.append(
            f"requires IR {manifest.requires.ir}, this core reads up to IR {ir}"
        )
    platform = current_platform() if platform is None else platform
    if manifest.platforms and platform not in manifest.platforms:
        problems.append(
            f"supports {', '.join(manifest.platforms)}, this platform is {platform}"
        )
    return problems


def resolve_transport(
    name: str,
    manifest: ExtensionManifest,
    installed: Mapping[str, ExtensionManifest],
    builtin: Collection[str],
) -> TransportOwner:
    """Find where the transport a shim of *manifest* names comes from.

    The order is the extension's own transports, the host's built-ins, then the
    transports of the extensions *manifest* requires, in the order it lists
    them.

    Args:
        name: The ``connection.transport`` value.
        manifest: The extension whose shim names the transport.
        installed: Every installed extension's manifest, by id.
        builtin: The transport ids the host ships.

    Returns:
        The owner of the transport.

    Raises:
        UnresolvedTransport: Nothing the extension may reach provides *name*.
            The message names an extension that provides it but is not
            required, when there is one.
    """
    if name in manifest.transports:
        return TransportOwner("own", manifest.id)
    if name in builtin:
        return TransportOwner("builtin", None)
    for dependency in manifest.requires.extensions:
        provider = installed.get(dependency)
        if provider is not None and name in provider.transports:
            return TransportOwner("dependency", dependency)
    hint = ""
    for other_id, other in installed.items():
        if other_id != manifest.id and name in other.transports:
            hint = (
                f"; extension {other_id!r} provides it, but {manifest.id!r} does "
                "not list it under requires.extensions"
            )
            break
    raise UnresolvedTransport(
        f"extension {manifest.id!r} names transport {name!r}, which it does not "
        f"provide, the host does not ship, and no required extension provides"
        f"{hint}"
    )


__all__ = [
    "MANIFEST_FILENAME",
    "MATURITY_LEVELS",
    "PLATFORMS",
    "SCHEMA_FILENAME",
    "ExtensionManifest",
    "ExtensionManifestInvalid",
    "ExtensionRequirement",
    "ExternalProgram",
    "TransportOwner",
    "UnresolvedTransport",
    "command_area",
    "compatibility_problems",
    "current_platform",
    "load_extension_schema",
    "parse_extension_manifest",
    "resolve_transport",
    "validate_extension_manifest",
]
