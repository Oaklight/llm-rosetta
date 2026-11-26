"""Loaded-shim inventory: origins, read-only definitions and the enable switch.

A built-in ships in the package and an installed extension is a fetched
artifact, so every definition is read-only in the panel.  The enable switch is
a config declaration rather than a deletion, which is what makes it reversible.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_rosetta_gateway.admin.routes.config import shim_files
from llm_rosetta_gateway.admin.routes.config.shim_files import (
    get_loaded_shim,
    get_loaded_shims,
    post_shim_enabled,
)
from llm_rosetta_gateway.config import JsoncConfigIO
from llm_rosetta.contracts.shim import ProviderShim
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


def _request(
    tmp_path: Path,
    raw: dict,
    source: Path | None = None,
    body: dict | None = None,
) -> SimpleNamespace:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    report = ShimLoadReport(
        sources=[
            ShimSourceStatus(
                kind="remote",
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


@pytest.fixture()
def typed_source(tmp_path: Path) -> Path:
    """A declared source that groups by model type, like the built-ins."""
    root = tmp_path / "typed"
    chat = root / "demo" / "llm" / "chat_completions"
    chat.mkdir(parents=True)
    (chat / "provider.yaml").write_text(
        _yaml("demo--chat_completions"), encoding="utf-8"
    )
    embedding = root / "demo" / "embeddings"
    embedding.mkdir(parents=True)
    (embedding / "provider.yaml").write_text(
        "name: demo--embedding\n"
        "model_type: embeddings\n"
        "format: openai\n"
        "connection:\n"
        "  base_url: https://demo.example/v1\n",
        encoding="utf-8",
    )
    return root


@pytest.mark.asyncio
async def test_the_inventory_reads_a_type_level_source(tmp_path, typed_source):
    """A shim one model-type level down is still found."""
    request = _request(tmp_path, {"providers": {}, "models": {}}, typed_source)

    rows = _body(await get_loaded_shims(request))["shims"]
    embedding = next((r for r in rows if r["name"] == "demo--embedding"), None)

    assert embedding is not None
    assert embedding["origin"] == "remote"
    assert embedding["model_type"] == "embeddings"
    assert "editable" not in embedding and "deletable" not in embedding


@pytest.mark.asyncio
async def test_a_declined_shim_keeps_the_vendors_mark(tmp_path):
    """A shim the loader did not register still carries its mark.

    It has no registered shim to read identity from, so without the file's
    own ``logo`` and ``group_label`` its card would paint initials beside a
    loaded sibling that shows the vendor's mark.
    """
    root = tmp_path / "free"
    leaf = root / "kilo_free" / "provider.yaml"
    leaf.parent.mkdir(parents=True)
    leaf.write_text(
        _yaml("kilo_free--chat_completions").replace(
            "connection:",
            "group_label: Kilo\nlogo: https://cdn.example/kilocode.svg\nconnection:",
        ),
        encoding="utf-8",
    )
    request = _request(tmp_path, {"providers": {}, "models": {}}, root)

    rows = _body(await get_loaded_shims(request))["shims"]
    row = next(r for r in rows if r["name"] == "kilo_free--chat_completions")

    assert row["enabled"] is False
    assert row["logo"] == "https://cdn.example/kilocode.svg"
    assert row["group_label"] == "Kilo"


@pytest.mark.asyncio
async def test_a_shim_from_an_installed_package_is_not_filed_as_builtin(
    tmp_path, monkeypatch
):
    """A plugin shim has no file to walk to, so the walk cannot place it.

    Without the distribution the inventory knows nothing about it and files it
    beside the built-ins — where the card draws no origin badge, and the source
    line is empty, so an installed package looks like something we ship.
    """
    from llm_rosetta.shims.provider_shim import register_shim, unregister_shim

    name = "plugin_demo--chat_completions"
    monkeypatch.setattr(
        shim_files,
        "_plugin_source",
        lambda n: "llm_rosetta_shim_plugin_demo 1.2.3" if n == name else None,
    )
    register_shim(ProviderShim(name=name, format="chat_completions"))
    try:
        request = _request(tmp_path, {"providers": {}, "models": {}})

        rows = _body(await get_loaded_shims(request))["shims"]
        row = next(r for r in rows if r["name"] == name)
        detail = _body(await get_loaded_shim(request, name))

        assert row["origin"] == "plugin"
        assert row["source"] == "llm_rosetta_shim_plugin_demo 1.2.3"
        assert "installed package" in (detail["read_only_reason"] or "")
    finally:
        unregister_shim(name)


@pytest.mark.asyncio
async def test_a_builtin_is_read_only(tmp_path):
    request = _request(tmp_path, {"providers": {}, "models": {}})

    body = _body(await get_loaded_shim(request, "deepseek--chat_completions"))

    assert body["origin"] == "builtin"
    assert "editable" not in body
    assert "built-in" in (body["read_only_reason"] or "")


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
