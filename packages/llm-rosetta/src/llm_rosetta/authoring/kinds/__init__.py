"""What each kind of an extension means to its author: the rules ``ext check`` applies.

A kind is one of the ``provides`` keys of a manifest (``shims``,
``transports``, ``commands``; :data:`llm_rosetta.extensions.KINDS`).  Each module
here holds the static rules of one kind, as ``problems(folder, manifest)``: it
returns one sentence per defect, and never imports the extension.  The module
and mapping names it reads are the contract's
(:mod:`llm_rosetta.contracts.extension`).  The template folder of a kind is
``_ext_template/<kind>/``.

A kind registered with the extension loader lives with the layer that loads it
(the gateway for transports), which an author's core-only install does not
have; these rules therefore sit in core, beside the contract they check.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from llm_rosetta.contracts.extension import ExtensionManifest

from . import commands, shims, transports

#: Kind name to its rules.  Every kind is asked for every extension: a kind the
#: manifest does not list can still be defective (a tree that registers a shim
#: ``provides.shims`` does not list).
PROBLEMS: dict[str, Callable[[Path, ExtensionManifest], list[str]]] = {
    "shims": shims.problems,
    "transports": transports.problems,
    "commands": commands.problems,
}

__all__ = ["PROBLEMS"]
