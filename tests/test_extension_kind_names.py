"""The module and mapping names of the commands and transports kinds have one home.

``llm_rosetta.contracts.extension`` names where an extension keeps what
``provides.commands`` and ``provides.transports`` name (``commands`` /
``COMMANDS``, ``transport`` / ``TRANSPORTS``).  The dispatcher, the import probe,
``ext check`` and the gateway read those names from there: four copies of the
string used to drift apart.
"""

from __future__ import annotations

import ast
from pathlib import Path

import llm_rosetta
from llm_rosetta.contracts import extension as contract

_PACKAGES = Path(llm_rosetta.__file__).resolve().parents[3]
_HOME = "llm_rosetta/contracts/extension.py"
_NAMES = {"COMMANDS", "TRANSPORTS"}


def _string_literals():
    for path in sorted(_PACKAGES.glob("*/src/**/*.py")):
        relative = path.relative_to(_PACKAGES).as_posix()
        if "_vendor" in relative or relative.endswith(_HOME):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and node.value in _NAMES:
                yield f"{relative}:{node.lineno} {node.value!r}"


def test_the_mapping_names_are_written_once():
    assert list(_string_literals()) == [], (
        "the names of the COMMANDS and TRANSPORTS mappings come from "
        "llm_rosetta.contracts.extension"
    )


def test_the_names_are_what_the_loaders_expect():
    assert (contract.COMMANDS_MODULE, contract.COMMANDS_ATTRIBUTE) == (
        "commands",
        "COMMANDS",
    )
    assert (contract.TRANSPORT_MODULE, contract.TRANSPORTS_ATTRIBUTE) == (
        "transport",
        "TRANSPORTS",
    )


def test_every_reader_uses_the_contract_names():
    from llm_rosetta.cli import areas
    from llm_rosetta_gateway.transport import registry

    assert areas.COMMANDS_MODULE is contract.COMMANDS_MODULE
    assert areas.COMMANDS_ATTRIBUTE is contract.COMMANDS_ATTRIBUTE
    assert registry.TRANSPORTS_ATTRIBUTE is contract.TRANSPORTS_ATTRIBUTE
    assert registry.TRANSPORT_MODULE is contract.TRANSPORT_MODULE
