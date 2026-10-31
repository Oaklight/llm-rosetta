"""Loaded-shim inventory: origins, editing, deleting and the enable switch.

The panel may only rewrite shims that live in a directory the config declares;
a built-in ships in the package and a remote one is a fetched artifact, so both
are read-only.  The enable switch is a config declaration rather than a
deletion, which is what makes it reversible.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_rosetta_gateway.admin.routes.config import shim_files
from llm_rosetta_gateway.admin.routes.config.shim_files import (
    _yaml_problem,
    delete_loaded_shim,
    get_loaded_shim,
    get_loaded_shims,
    post_shim_enabled,
    put_loaded_shim,
)
from llm_rosetta_gateway.config import JsoncConfigIO
from llm_rosetta_gateway.shims.sources import ShimLoadReport, ShimSourceStatus


def _yaml(name: str, display: str = "Demo") -> str:
    return (
        f"name: {name}\n"
        "model_type: llm\n"
        "format: chat_completions\n"
        f"display_name: {display}\n"
        "connection:\n"
        "  base_url: https://demo.example/v1\n"
        "  api_key_env: DEMO_KEY\n"
    )


@pytest.fixture
def local_source(tmp_path: Path) -> Path:
    """A declared local shim directory holding one shim."""
    root = tmp_path / "shims"
    shim = root / "demo--chat_completions"
    shim.mkdir(parents=True)
    (shim / "provider.yaml").write_text(
        _yaml("demo--chat_completions"), encoding="utf-8"
    )
    return root


def _request(
    tmp_path: Path,
    raw: dict,
    source: Path | None = None,
    body: dict | None = None,
) -> SimpleNamespace:
    path = tmp_path / "config.json"
    if source is not None:
        raw = {**raw, "shims": {"paths": [str(source)]}}
    path.write_text(json.dumps(raw), encoding="utf-8")
    report = ShimLoadReport(
        sources=[
            ShimSourceStatus(
                kind="path",
                location=str(source),
                ok=True,
                shim_count=1,
                shim_names=("demo--chat_completions",),
                resolved_dir=str(source),
            )
        ]
        if source is not None
        else [],
        total_shims=1 if source is not None else 0,
        rebuilt_at=123.0,
    )
    app = SimpleNamespace(
        config_path=str(path),
        config_io=JsoncConfigIO(),
        gateway_config=SimpleNamespace(provider_shim_names={}),
        provider_validation_warnings=[],
        shim_source_report=report,
        shim_source_signature=None,
        rate_limit_state=None,
    )
    return SimpleNamespace(
        app=app,
        path_params={},
        method="GET",
        headers={},
        json=lambda: body,
    )


def _body(response) -> dict:
    return json.loads(response.body)


class TestYamlProblem:
    def test_accepts_the_same_name(self):
        assert (
            _yaml_problem(_yaml("demo--chat_completions"), "demo--chat_completions")
            is None
        )

    def test_rejects_a_rename(self):
        problem = _yaml_problem(
            _yaml("other--chat_completions"), "demo--chat_completions"
        )
        assert problem and "other--chat_completions" in problem

    def test_rejects_broken_yaml(self):
        assert _yaml_problem("name: [unclosed", "demo") is not None

    def test_rejects_a_missing_format(self):
        assert _yaml_problem("name: demo\n", "demo") is not None


@pytest.mark.asyncio
async def test_inventory_marks_a_declared_shim_editable(tmp_path, local_source):
    request = _request(tmp_path, {"providers": {}, "models": {}}, local_source)

    rows = _body(await get_loaded_shims(request))["shims"]
    demo = next((r for r in rows if r["name"] == "demo--chat_completions"), None)

    # The shim is only in the file index (the report is synthetic), so this
    # row comes from the directory walk — which is what marks it editable.
    assert demo is not None
    assert demo["origin"] == "path"
    assert demo["editable"] is True and demo["deletable"] is True


@pytest.fixture()
def root_definition(tmp_path: Path) -> Path:
    """A source whose provider.yaml sits at the root, beside a nested shim.

    The loader only looks one level down, so this file is not a shim
    directory — and deleting "its directory" would take the whole source.
    """
    root = tmp_path / "flat"
    nested = root / "sibling--chat_completions"
    nested.mkdir(parents=True)
    (nested / "provider.yaml").write_text(
        _yaml("sibling--chat_completions"), encoding="utf-8"
    )
    (root / "provider.yaml").write_text(
        _yaml("rooted--chat_completions"), encoding="utf-8"
    )
    (root / "README.md").write_text("keep me\n", encoding="utf-8")
    return root


@pytest.mark.asyncio
async def test_a_definition_at_the_source_root_is_read_only(tmp_path, root_definition):
    request = _request(tmp_path, {"providers": {}, "models": {}}, root_definition)

    rows = _body(await get_loaded_shims(request))["shims"]
    rooted = next(r for r in rows if r["name"] == "rooted--chat_completions")
    detail = _body(await get_loaded_shim(request, "rooted--chat_completions"))

    assert rooted["editable"] is False and rooted["deletable"] is False
    assert "root of a declared source" in (detail["read_only_reason"] or "")


@pytest.mark.asyncio
async def test_deleting_at_the_source_root_keeps_the_source(
    tmp_path, root_definition, monkeypatch
):
    request = _request(tmp_path, {"providers": {}, "models": {}}, root_definition)

    async def _no_rebuild(_request):
        return None

    monkeypatch.setattr(shim_files, "_rebuild_shims", _no_rebuild)
    response = await delete_loaded_shim(request, "rooted--chat_completions")

    assert response.status_code == 403
    # The one thing this guard exists for: the source and its siblings survive.
    assert (root_definition / "provider.yaml").exists()
    assert (root_definition / "README.md").exists()
    assert (root_definition / "sibling--chat_completions" / "provider.yaml").exists()


@pytest.mark.asyncio
async def test_a_builtin_is_read_only(tmp_path):
    request = _request(tmp_path, {"providers": {}, "models": {}})

    body = _body(await get_loaded_shim(request, "deepseek--chat_completions"))

    assert body["origin"] == "builtin"
    assert body["editable"] is False
    assert "built-in" in (body["read_only_reason"] or "")


@pytest.mark.asyncio
async def test_editing_a_builtin_is_refused(tmp_path):
    request = _request(
        tmp_path,
        {"providers": {}, "models": {}},
        body={"yaml": _yaml("deepseek--chat_completions")},
    )

    response = await put_loaded_shim(request, "deepseek--chat_completions")

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_disabling_a_shim_a_provider_uses_is_refused(tmp_path):
    request = _request(
        tmp_path,
        {"providers": {"p": {"type": "deepseek--chat_completions"}}, "models": {}},
        body={"enabled": False},
    )
    request.app.gateway_config = SimpleNamespace(
        provider_shim_names={"p": "deepseek--chat_completions"}
    )

    response = await post_shim_enabled(request, "deepseek--chat_completions")

    assert response.status_code == 409
    assert "p" in _body(response)["providers"]
