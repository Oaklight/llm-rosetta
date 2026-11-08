"""A rename-merge that cannot add the provider must not drop a model (#1429).

Renaming a model onto a name already served by the *same* provider used to
report success while deleting the source: the merge helper saw the provider in
the target's list, added nothing, and the handler deleted ``rename_from``
anyway.  The target kept its first upstream (``deepseek-v4.1-flash``) and the
requested one (``deepseek-v4.1-flash-eco``) vanished with its model.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from llm_rosetta_gateway.admin.routes.config.models import put_model
from llm_rosetta_gateway.admin.routes.config.providers import _handle_rename_merge
from llm_rosetta_gateway.config import JsoncConfigIO

PROVIDER = {"type": "openai", "base_url": "https://a.example/v1"}

TARGET = "deepseek-v4.1-flash"
SOURCE = "deepseek-v4.1-flash-eco"


def _raw(*providers: str) -> dict[str, Any]:
    """A config whose two models route the first provider to different upstreams."""
    return {
        "providers": {name: dict(PROVIDER) for name in providers},
        "models": {
            TARGET: {"provider": providers[0], "upstream_model": TARGET},
            SOURCE: {"provider": providers[0], "upstream_model": SOURCE},
        },
    }


def _request(
    tmp_path: Path, raw: dict[str, Any], *, name: str, body: dict[str, Any]
) -> SimpleNamespace:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    app = SimpleNamespace(
        config_path=str(path),
        config_io=JsoncConfigIO(),
        gateway_config=None,
        provider_validation_warnings=[],
        shim_source_signature=None,
        rate_limit_state=None,
    )
    return SimpleNamespace(
        app=app,
        path_params={"name": name},
        method="PUT",
        headers={},
        json=lambda: dict(body),
    )


def _on_disk(request: SimpleNamespace) -> dict[str, Any]:
    """The saved document (the loader adds config_version and canonical types)."""
    return json.loads(Path(request.app.config_path).read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_same_provider_rename_merge_is_refused_and_drops_nothing(
    tmp_path: Path,
):
    raw = _raw("opendesign-go")
    request = _request(
        tmp_path,
        raw,
        name=TARGET,
        body={"provider": "opendesign-go", "rename_from": SOURCE, "merge": True},
    )

    response = await put_model(request)
    payload = json.loads(response.body)

    assert response.status_code == 409
    assert "already routes through provider 'opendesign-go'" in payload["error"]
    # The refusal names the upstream the target already uses.
    assert f"{TARGET}" in payload["error"]
    # Nothing to confirm: the merge can never succeed.
    assert "merge_possible" not in payload
    on_disk = _on_disk(request)["models"]
    assert on_disk[SOURCE] == raw["models"][SOURCE]
    assert on_disk[TARGET] == raw["models"][TARGET]


@pytest.mark.asyncio
async def test_the_offered_merge_only_appears_when_it_can_happen(tmp_path: Path):
    """The first 409 asks for confirmation only for a merge that adds a leg."""
    raw = _raw("opendesign-go")
    request = _request(
        tmp_path,
        raw,
        name=TARGET,
        body={"provider": "opendesign-go", "rename_from": SOURCE},
    )

    payload = json.loads((await put_model(request)).body)

    assert "merge_possible" not in payload
    assert "already routes through provider" in payload["error"]
    on_disk = _on_disk(request)["models"]
    assert on_disk[SOURCE]["upstream_model"] == SOURCE
    assert on_disk[TARGET] == raw["models"][TARGET]


@pytest.mark.asyncio
async def test_a_different_provider_still_merges(tmp_path: Path):
    raw = _raw("opendesign-go", "backup-go")
    request = _request(
        tmp_path,
        raw,
        name=TARGET,
        body={"provider": "backup-go", "rename_from": SOURCE},
    )

    # Without the flag the first answer offers the merge.
    offer = json.loads((await put_model(request)).body)
    assert offer["merge_possible"] is True
    assert offer["existing_provider"] == "opendesign-go"
    assert _on_disk(request)["models"] == raw["models"]

    request = _request(
        tmp_path,
        raw,
        name=TARGET,
        body={"provider": "backup-go", "rename_from": SOURCE, "merge": True},
    )
    payload = json.loads((await put_model(request)).body)

    assert payload["ok"] is True
    assert payload["merged"] is True
    on_disk = _on_disk(request)
    assert SOURCE not in on_disk["models"]
    providers = on_disk["models"][TARGET]["providers"]
    assert [(p["name"], p.get("upstream_model")) for p in providers] == [
        ("opendesign-go", TARGET),
        ("backup-go", SOURCE),
    ]


@pytest.mark.asyncio
async def test_the_merged_lane_keeps_the_source_upstream(tmp_path: Path):
    """Omitting `upstream_model` must not substitute the source model's name.

    The source entry asks its provider for ``source-real-upstream``; the model
    is *named* ``deepseek-v4.1-flash-eco``.  The new leg must carry the former.
    """
    raw = {
        "providers": {"opendesign-go": dict(PROVIDER), "backup-go": dict(PROVIDER)},
        "models": {
            TARGET: {"provider": "opendesign-go", "upstream_model": TARGET},
            SOURCE: {"provider": "backup-go", "upstream_model": "source-real-upstream"},
        },
    }
    request = _request(
        tmp_path,
        raw,
        name=TARGET,
        body={"provider": "backup-go", "rename_from": SOURCE, "merge": True},
    )

    payload = json.loads((await put_model(request)).body)

    assert payload["ok"] is True
    providers = _on_disk(request)["models"][TARGET]["providers"]
    assert [(p["name"], p.get("upstream_model")) for p in providers] == [
        ("opendesign-go", TARGET),
        ("backup-go", "source-real-upstream"),
    ]


@pytest.mark.asyncio
async def test_a_plain_rename_moves_the_entry_and_nothing_else(tmp_path: Path):
    """Target absent, source present: the entry moves, the rest is untouched."""
    raw = _raw("opendesign-go")
    request = _request(
        tmp_path,
        raw,
        name="deepseek-v4.1-flash-next",
        body={
            "provider": "opendesign-go",
            "upstream_model": "deepseek-v4.1-flash",
            "rename_from": TARGET,
        },
    )

    payload = json.loads((await put_model(request)).body)

    assert payload["ok"] is True
    # A plain rename is not a merge: nothing was combined.
    assert "merged" not in payload
    models = _on_disk(request)["models"]
    assert TARGET not in models
    assert models["deepseek-v4.1-flash-next"]["provider"] == "opendesign-go"
    assert models["deepseek-v4.1-flash-next"]["upstream_model"] == TARGET
    assert models[SOURCE] == raw["models"][SOURCE]


@pytest.mark.parametrize("source_lanes", ["declared-first", "declared-second"])
@pytest.mark.asyncio
async def test_the_source_upstream_is_found_on_any_lane(
    tmp_path: Path, source_lanes: str
):
    """The source may list one provider twice; the declared upstream wins.

    Both orders are legal in a config (the keys differ), so the lookup must
    not stop at a lane that merely names the provider.
    """
    lanes = (
        [{"name": "backup-go"}, {"name": "backup-go", "upstream_model": "real"}]
        if source_lanes == "declared-second"
        else [{"name": "backup-go", "upstream_model": "real"}, {"name": "backup-go"}]
    )
    raw = {
        "providers": {"opendesign-go": dict(PROVIDER), "backup-go": dict(PROVIDER)},
        "models": {
            TARGET: {"provider": "opendesign-go", "upstream_model": TARGET},
            SOURCE: {"providers": lanes, "capabilities": ["text"]},
        },
    }
    request = _request(
        tmp_path,
        raw,
        name=TARGET,
        body={"provider": "backup-go", "rename_from": SOURCE, "merge": True},
    )

    payload = json.loads((await put_model(request)).body)

    assert payload["ok"] is True
    providers = _on_disk(request)["models"][TARGET]["providers"]
    assert [(p["name"], p.get("upstream_model")) for p in providers] == [
        ("opendesign-go", TARGET),
        ("backup-go", "real"),
    ]


@pytest.mark.parametrize(
    ("existing", "outcome_hint"),
    [
        ({"provider": "p", "upstream_model": "u-1"}, "same provider"),
        ({"providers": [{"name": "p", "upstream_model": "u-1"}]}, "provider list"),
        (42, "unrecognised entry"),
    ],
)
def test_the_handler_refuses_without_touching_the_models(
    existing: Any, outcome_hint: str
):
    """Unit-level: every unmergeable shape leaves both entries as they were."""
    models = {TARGET: existing, SOURCE: {"provider": "p", "upstream_model": SOURCE}}
    before = json.loads(json.dumps(models, default=str))

    response = _handle_rename_merge(
        models, TARGET, SOURCE, {"provider": "p", "merge": True}, "p"
    )

    assert response is not None
    assert response.status_code == 409
    assert outcome_hint
    left = json.loads(json.dumps(models, default=str))
    assert left[SOURCE] == before[SOURCE]
    assert left[TARGET] == before[TARGET]
