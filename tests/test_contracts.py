"""``llm_rosetta.contracts``: the version constants and the sidecar hook names (#1657)."""

from __future__ import annotations

import llm_rosetta.contracts as contracts
from llm_rosetta import api
from llm_rosetta.types import ir


def test_the_versions_have_one_definition_each():
    assert contracts.IR_VERSION is ir.IR_VERSION
    assert api.SURFACE_VERSION is contracts.SURFACE_VERSION
    assert isinstance(contracts.SURFACE_VERSION, int)
    assert isinstance(contracts.IR_VERSION, int)


def test_the_sidecar_hook_names_are_pinned():
    # Every shim in core and in the external repos exports these names; a
    # rename here silently drops their transforms, probes and exchanges.
    assert {
        name: getattr(contracts, name)
        for name in contracts.__all__
        if name not in {"IR_VERSION", "SURFACE_VERSION"}
    } == {
        "TRANSFORMS_MODULE": "transforms.py",
        "AUTH_MODULE": "auth.py",
        "QUOTA_MODULE": "quota.py",
        "PRE_IR_TRANSFORMS": "pre_ir_transforms",
        "POST_IR_TRANSFORMS": "post_ir_transforms",
        "IR_TRANSFORMS": "ir_transforms",
        "RESPONSE_BODY_TRANSFORMS": "response_body_transforms",
        "MODEL_LIST_TRANSFORM": "model_list_transform",
        "QUOTA_PROBES": "PROBES",
        "AUTH_EXCHANGE": "EXCHANGE",
    }


def test_the_loader_reads_the_names_from_contracts():
    # The loader modules import the constants instead of restating the strings.
    from pathlib import Path

    providers = Path(contracts.__file__).parents[1] / "shims" / "providers"
    for loader in ("_sidecars.py", "_leaf.py", "_manifest.py", "__init__.py"):
        source = (providers / loader).read_text(encoding="utf-8")
        for literal in (
            '"pre_ir_transforms"',
            '"post_ir_transforms"',
            '"ir_transforms"',
            '"response_body_transforms"',
            '"model_list_transform"',
            '"PROBES"',
            '"EXCHANGE"',
            '"transforms.py"',
            '"auth.py"',
            '"quota.py"',
        ):
            assert literal not in source, f"{loader} restates {literal}"
