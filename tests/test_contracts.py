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


def test_provider_shim_reexports_the_contract_data_classes():
    # 0.13.0 shipped the data classes in shims.provider_shim; the module keeps
    # them, as the same objects, until 0.15.0.
    import llm_rosetta.contracts.shim as shim
    import llm_rosetta.shims.provider_shim as provider_shim

    for name in set(shim.__all__) - {"load_providers_from_dir"}:
        assert getattr(provider_shim, name) is getattr(shim, name), name
    assert "0.15.0" in (provider_shim.__doc__ or "")
    assert "llm_rosetta.contracts.shim" in (provider_shim.__doc__ or "")
    # The registry stays internal: the contract holds values, not state.
    for name in ("register_shim", "get_shim", "list_shims", "staging_registry"):
        assert not hasattr(shim, name)


def test_core_imports_the_data_classes_from_contracts():
    import ast
    from pathlib import Path

    data = set(__import__("llm_rosetta.contracts.shim", fromlist=["x"]).__all__)
    core = Path(contracts.__file__).parents[1]
    offenders = []
    for path in sorted(core.rglob("*.py")):
        if "_vendor" in path.parts or path.name == "provider_shim.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.ImportFrom)
                and (node.module or "").endswith("provider_shim")
                and data & {alias.name for alias in node.names}
            ):
                offenders.append(f"{path.relative_to(core)}:{node.lineno}")
    assert offenders == []


def test_the_transform_contract_names_the_implementations():
    import llm_rosetta.contracts.transform as transform
    import llm_rosetta.ir.primitives as primitives

    for name in transform.__all__:
        assert getattr(transform, name) is getattr(primitives, name), name

    def keep(body):
        return body

    def keep_ir(body, context):
        return body

    named = transform.NamedTransform(keep, "keep")
    assert repr(named) == "keep"
    assert named(dict(a=1)) == dict(a=1)
    assert repr(transform.NamedIRTransform(keep_ir, "keep_ir")) == "keep_ir"
