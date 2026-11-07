"""Vendor manifests in the shim-files panel: rows, inheritance and editing.

A manifest is ``<vendor>/provider.yaml`` with a ``connection:`` block and no
identity.  It is not a shim, so it never enters the inventory's ``shims`` list;
it is listed beside the leaves that inherit it, and an edit is shown with its
blast radius before it is written.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest

from llm_rosetta_gateway.admin.routes.config import shim_manifests
from llm_rosetta_gateway.admin.routes.config.shim_manifests import (
    _fields,
    _leaf_entries,
    _merged_connection,
    get_shim_manifest,
    get_shim_manifests,
    manifest_problem,
    manifest_rows,
    post_preview_shim_manifest,
    put_shim_manifest,
)
from llm_rosetta_gateway.admin.routes.config.shim_files import get_loaded_shims
from llm_rosetta_gateway.config import JsoncConfigIO
from llm_rosetta_gateway.shims.sources import ShimLoadReport, ShimSourceStatus


def _manifest(connection: str) -> str:
    return "connection:\n" + connection


def _leaf(name: str, connection: str) -> str:
    return (
        f"name: {name}\n"
        "model_type: llm\n"
        "format: chat_completions\n"
        "connection:\n" + connection
    )


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """A declared source with one vendor manifest and three leaves.

    ``alpha`` overrides ``base_url``; ``beta`` overrides one header;
    ``gamma`` overrides nothing.
    """
    root = tmp_path / "shims"
    vendor = root / "demo"
    vendor.mkdir(parents=True)
    (vendor / "provider.yaml").write_text(
        _manifest("  base_url: https://demo.example/v1\n  api_key_env: DEMO_KEY\n")
        + "  headers:\n    x-api-version: '2024-01'\n",
        encoding="utf-8",
    )
    for name, extra in (
        ("demo--alpha", "  base_url: https://alpha.example/v1\n"),
        ("demo--beta", "  headers:\n    x-api-version: '2023-06'\n"),
        ("demo--gamma", "  display_name: ignored\n"),
    ):
        leaf = vendor / name
        leaf.mkdir()
        (leaf / "provider.yaml").write_text(_leaf(name, extra), encoding="utf-8")
    return root


def _request(
    tmp_path: Path,
    source: Path | None = None,
    kind: Literal["path", "remote"] = "path",
    body: dict | None = None,
    query: dict | None = None,
) -> SimpleNamespace:
    path = tmp_path / "config.json"
    raw = {"providers": {}, "models": {}}
    if source is not None:
        raw = {**raw, "shims": {"paths": [str(source)]}}
    path.write_text(json.dumps(raw), encoding="utf-8")
    report = ShimLoadReport(
        sources=[
            ShimSourceStatus(
                kind=kind,
                location=str(source),
                ok=True,
                shim_count=3,
                shim_names=("demo--alpha", "demo--beta", "demo--gamma"),
                resolved_dir=str(source),
            )
        ]
        if source is not None
        else [],
        total_shims=3 if source is not None else 0,
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
        # The server parses a query string into {key: [values]}.
        query_params={k: [v] for k, v in (query or {}).items()},
        json=lambda: body,
    )


def _body(response) -> dict:
    return json.loads(response.body)


def _manifest_path(source: Path) -> str:
    return str(source / "demo" / "provider.yaml")


class TestManifestProblem:
    def test_accepts_a_manifest(self):
        assert (
            manifest_problem(
                "connection:\n  base_url: https://demo.example/v1\n"
                "  api_key_env: DEMO_KEY\n  headers:\n    x-a: '1'\n"
            )
            is None
        )

    def test_accepts_an_empty_file(self):
        assert manifest_problem("") is None
        assert manifest_problem("# the shared facts are in auth.py\n") is None

    def test_refuses_identity(self):
        problem = manifest_problem("name: demo\nconnection:\n  api_key_env: X\n")
        assert problem and "'name'" in problem

    def test_refuses_an_unknown_top_level_key(self):
        problem = manifest_problem("display_name: Demo\n")
        assert problem and "display_name" in problem

    def test_refuses_an_unknown_connection_key(self):
        problem = manifest_problem("connection:\n  base_uri: https://x\n")
        assert problem and "base_uri" in problem

    @pytest.mark.parametrize(
        "connection, expected",
        [
            ("  base_url: 5\n", "base_url must be a string"),
            ("  keyless: 'yes'\n", "keyless must be true or false"),
            ("  token_command: []\n", "at least 1 item"),
            ("  headers:\n    x-a: 7\n", "x-a is not a string"),
            ("  session_id_format: nope\n", "must be one of: opencode"),
            ("  token_refresh_interval: -1\n", "at least 0"),
        ],
    )
    def test_refuses_a_bad_value(self, connection, expected):
        problem = manifest_problem("connection:\n" + connection)
        assert problem and expected in problem

    def test_refuses_broken_yaml(self):
        assert manifest_problem("connection: [unclosed") is not None

    def test_the_schema_and_the_loader_agree_on_the_connection_keys(self):
        """One list of keys: the schema's properties are the config's fields."""
        from llm_rosetta.shims.providers import _CONNECTION_KEYS

        schema = shim_manifests._manifest_schema()
        assert set(schema["connection"]["properties"]) == set(_CONNECTION_KEYS)


@pytest.mark.asyncio
async def test_the_inventory_carries_manifests_beside_shims(tmp_path, source):
    request = _request(tmp_path, source)

    payload = _body(await get_loaded_shims(request))

    demo = [row for row in payload["manifests"] if row["vendor"] == "demo"]
    assert demo and demo[0]["path"] == _manifest_path(source)
    # A manifest is not a shim: it has no name, so it cannot be resolved by
    # one, and the list a picker reads must not contain it.
    assert "demo" not in {row["name"] for row in payload["shims"]}


@pytest.mark.asyncio
async def test_a_manifest_from_a_writable_source_is_editable(tmp_path, source):
    request = _request(tmp_path, source)

    rows = _body(await get_shim_manifests(request))["manifests"]
    (row,) = [r for r in rows if r["vendor"] == "demo"]

    assert row["editable"] is True and row["read_only_reason"] is None
    assert row["vendor"] == "demo"
    assert row["leaves"] == ["demo--alpha", "demo--beta", "demo--gamma"]


@pytest.mark.asyncio
async def test_a_manifest_from_a_remote_source_is_read_only(tmp_path, source):
    request = _request(tmp_path, source, kind="remote")

    rows = _body(await get_shim_manifests(request))["manifests"]
    (row,) = [r for r in rows if r["vendor"] == "demo"]

    assert row["editable"] is False
    assert "fetched artifact" in row["read_only_reason"]


@pytest.mark.asyncio
async def test_the_detail_names_who_inherits_each_field(tmp_path, source):
    request = _request(tmp_path, source, query={"path": _manifest_path(source)})

    detail = _body(await get_shim_manifest(request))

    assert detail["connection"]["base_url"] == "https://demo.example/v1"
    assert detail["fields"]["base_url"] == {
        "inherited_by": ["demo--beta", "demo--gamma"],
        "overridden_by": ["demo--alpha"],
    }
    # headers merge per key, so the override is per header, not per leaf.
    assert detail["fields"]["headers"]["x-api-version"] == {
        "inherited_by": ["demo--alpha", "demo--gamma"],
        "overridden_by": ["demo--beta"],
    }
    assert detail["yaml"].startswith("connection:")


@pytest.mark.asyncio
async def test_the_detail_refuses_a_path_no_source_defines(tmp_path, source):
    request = _request(tmp_path, source, query={"path": str(tmp_path / "secret.yaml")})

    response = await get_shim_manifest(request)

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_a_preview_names_only_the_leaves_that_inherit_the_touched_field(
    tmp_path, source
):
    request = _request(
        tmp_path,
        source,
        body={
            "path": _manifest_path(source),
            "yaml": _manifest(
                "  base_url: https://new.example/v1\n"
                "  api_key_env: DEMO_KEY\n"
                "  headers:\n    x-api-version: '2024-01'\n"
            ),
        },
    )

    preview = _body(await post_preview_shim_manifest(request))

    assert preview["changed_leaves"] == ["demo--beta", "demo--gamma"]
    assert preview["unchanged_leaves"] == ["demo--alpha"]
    (change,) = preview["changes"]
    assert change["field"] == "base_url" and change["leaves"] == [
        "demo--beta",
        "demo--gamma",
    ]


@pytest.mark.asyncio
async def test_a_preview_of_a_removed_field_names_the_leaves_left_empty(
    tmp_path, source
):
    request = _request(
        tmp_path,
        source,
        body={
            "path": _manifest_path(source),
            "yaml": _manifest("  headers:\n    x-api-version: '2024-01'\n"),
        },
    )

    preview = _body(await post_preview_shim_manifest(request))

    removed = {c["field"] for c in preview["changes"] if c["removed"]}
    assert removed == {"base_url", "api_key_env"}
    # alpha overrides base_url but still inherits api_key_env, so it changes.
    assert preview["changed_leaves"] == ["demo--alpha", "demo--beta", "demo--gamma"]


@pytest.mark.asyncio
async def test_a_preview_that_does_not_validate_writes_nothing(tmp_path, source):
    before = (source / "demo" / "provider.yaml").read_text(encoding="utf-8")
    request = _request(
        tmp_path,
        source,
        body={"path": _manifest_path(source), "yaml": "name: demo\n"},
    )

    response = await post_preview_shim_manifest(request)

    assert response.status_code == 400
    assert (source / "demo" / "provider.yaml").read_text(encoding="utf-8") == before


@pytest.mark.asyncio
async def test_saving_a_manifest_writes_it_and_names_the_changed_leaves(
    tmp_path, source, monkeypatch
):
    async def _rebuild(_request):
        return None

    monkeypatch.setattr(shim_manifests, "_rebuild_shims", _rebuild)
    edited = _manifest(
        "  base_url: https://new.example/v1\n"
        "  api_key_env: DEMO_KEY\n"
        "  headers:\n    x-api-version: '2024-01'\n"
    )
    request = _request(
        tmp_path, source, body={"path": _manifest_path(source), "yaml": edited}
    )

    response = await put_shim_manifest(request)

    assert response.status_code == 200
    body = _body(response)
    assert body["changed_leaves"] == ["demo--beta", "demo--gamma"]
    assert (source / "demo" / "provider.yaml").read_text(encoding="utf-8") == edited


@pytest.mark.asyncio
async def test_a_save_the_loader_rejects_restores_the_previous_file(
    tmp_path, source, monkeypatch
):
    before = (source / "demo" / "provider.yaml").read_text(encoding="utf-8")

    async def _rebuild(_request):
        return "demo--gamma: bad shim"

    monkeypatch.setattr(shim_manifests, "_rebuild_shims", _rebuild)
    request = _request(
        tmp_path,
        source,
        body={
            "path": _manifest_path(source),
            "yaml": _manifest("  base_url: https://new.example/v1\n"),
        },
    )

    response = await put_shim_manifest(request)

    assert response.status_code == 400
    assert "demo--gamma" in _body(response)["error"]
    assert (source / "demo" / "provider.yaml").read_text(encoding="utf-8") == before


@pytest.mark.asyncio
async def test_a_read_only_manifest_cannot_be_saved(tmp_path, source):
    request = _request(
        tmp_path,
        source,
        kind="remote",
        body={"path": _manifest_path(source), "yaml": _manifest("  keyless: true\n")},
    )

    response = await put_shim_manifest(request)

    assert response.status_code == 403
    assert "fetched artifact" in _body(response)["error"]


@pytest.mark.asyncio
async def test_a_manifest_path_outside_the_sources_cannot_be_written(tmp_path, source):
    outside = tmp_path / "outside.yaml"
    outside.write_text("connection:\n  keyless: true\n", encoding="utf-8")
    request = _request(
        tmp_path, source, body={"path": str(outside), "yaml": _manifest("  x: 1\n")}
    )

    response = await put_shim_manifest(request)

    assert response.status_code == 404
    assert outside.read_text(encoding="utf-8").startswith("connection:\n  keyless")


def test_a_manifest_above_no_leaf_is_not_listed(tmp_path):
    """A ``provider.yaml`` nothing inherits from is not a manifest row.

    Without the guard every vendor directory a source root happens to hold
    becomes an editable manifest, at a path the loader does not read as one.
    Drive the walk rather than the field formatter: this test used to assert
    ``_fields({}, []) == {}`` and could not fail.
    """
    root = tmp_path / "shims"
    (root / "demo").mkdir(parents=True)
    (root / "demo" / "provider.yaml").write_text(
        _manifest("  api_key_env: DEMO_KEY\n"), encoding="utf-8"
    )

    rows = manifest_rows(_request(tmp_path, root))

    assert not [row for row in rows if Path(row["path"]).is_relative_to(root)]


def test_a_nested_manifest_does_not_borrow_the_outer_one_s_leaves(tmp_path):
    """The loader does not nest: a leaf belongs to its own vendor's manifest.

    A plain ``rglob`` let the outer manifest claim ``nested/beta`` as well, so
    the inheritance view said one leaf inherited two manifests' ``api_key_env``
    at once — an inheritance the loader never performs.
    """
    root = tmp_path / "shims"
    outer = root / "demo"
    nested = outer / "nested"
    (outer / "alpha").mkdir(parents=True)
    (nested / "beta").mkdir(parents=True)
    (outer / "provider.yaml").write_text(
        _manifest("  api_key_env: DEMO_KEY\n"), encoding="utf-8"
    )
    (nested / "provider.yaml").write_text(
        _manifest("  api_key_env: NESTED_KEY\n"), encoding="utf-8"
    )
    (outer / "alpha" / "provider.yaml").write_text(
        _leaf("demo--alpha", "  display_name: alpha\n"), encoding="utf-8"
    )
    (nested / "beta" / "provider.yaml").write_text(
        _leaf("nested--beta", "  display_name: beta\n"), encoding="utf-8"
    )

    rows = {row["vendor"]: row for row in manifest_rows(_request(tmp_path, root))}

    assert rows["demo"]["leaves"] == ["demo--alpha"]
    assert rows["nested"]["leaves"] == ["nested--beta"]
    # The same rule one level down: beta inherits from `nested`, not from
    # `demo`, so the outer field view must not name it.
    fields = _fields(
        {"api_key_env": "DEMO_KEY"}, _leaf_entries(outer / "provider.yaml")
    )
    assert fields["api_key_env"]["inherited_by"] == ["demo--alpha"]


def _within(seconds: float, call: Any) -> Any:
    """Run *call* in a thread and fail the test if it is still running.

    A termination assertion cannot sit after the call: with the loop guard
    removed the call never returns, so the suite would hang instead of a test
    going red.
    """
    box: dict[str, Any] = {}

    def runner() -> None:
        box["value"] = call()

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(seconds)
    assert not thread.is_alive(), f"call did not return within {seconds}s"
    return box["value"]


def _symlink(target: Path, link: Path) -> None:
    """Create a directory symlink, or skip where the platform refuses one."""
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):  # pragma: no cover - platform
        pytest.skip("directory symlinks are not available here")


def test_a_leaf_reached_through_a_symlink_still_inherits(tmp_path):
    """The loader follows a directory link, so the view must too.

    ``demo/beta`` points at a vendor directory elsewhere.  The loader loads
    ``demo--beta`` and merges the manifest into it, so a row that listed only
    ``demo--alpha`` would answer "nobody inherits this" for a leaf that does.
    """
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "beta").mkdir(parents=True)
    (elsewhere / "beta" / "provider.yaml").write_text(
        _leaf("demo--beta", "  display_name: beta\n"), encoding="utf-8"
    )
    root = tmp_path / "shims"
    demo = root / "demo"
    (demo / "alpha").mkdir(parents=True)
    (demo / "provider.yaml").write_text(
        _manifest("  api_key_env: DEMO_KEY\n"), encoding="utf-8"
    )
    (demo / "alpha" / "provider.yaml").write_text(
        _leaf("demo--alpha", "  display_name: alpha\n"), encoding="utf-8"
    )
    _symlink(elsewhere / "beta", demo / "beta")

    rows = manifest_rows(_request(tmp_path, root))

    row = next(r for r in rows if r["vendor"] == "demo")
    assert row["leaves"] == ["demo--alpha", "demo--beta"]

    # And the view agrees with the loader, field by field, for the linked leaf.
    from llm_rosetta.shims import load_providers_from_dir

    loaded = {shim.name: shim for shim in load_providers_from_dir(root)}
    beta = _merged_connection(
        {"api_key_env": "DEMO_KEY"},
        next(
            leaf["own"]
            for leaf in _leaf_entries(demo / "provider.yaml")
            if leaf["name"] == "demo--beta"
        ),
        demo / "provider.yaml",
    )
    assert beta["api_key_env"] == loaded["demo--beta"].connection.api_key_env


def test_a_manifest_behind_a_symlinked_vendor_is_still_listed(tmp_path):
    """A whole vendor directory behind a link is still a manifest row.

    The inventory's own scan used ``rglob``, which does not follow the link:
    no row appeared, while the loader still loaded ``linked--alpha`` and
    merged the manifest into it.  Listing the leaves was half the fix; this is
    the other half.
    """
    vendor = tmp_path / "elsewhere" / "demo"
    (vendor / "alpha").mkdir(parents=True)
    (vendor / "provider.yaml").write_text(
        _manifest("  api_key_env: LINKED_KEY\n"), encoding="utf-8"
    )
    (vendor / "alpha" / "provider.yaml").write_text(
        _leaf("linked--alpha", "  display_name: alpha\n"), encoding="utf-8"
    )
    root = tmp_path / "shims"
    root.mkdir()
    _symlink(vendor, root / "linked")

    rows = manifest_rows(_request(tmp_path, root))

    row = next(r for r in rows if r["vendor"] == "linked")
    assert row["leaves"] == ["linked--alpha"]

    # The row exists because the loader loads this shim; the merge it shows is
    # the merge the loader applies, field by field.
    from llm_rosetta.shims import load_providers_from_dir

    loaded = {shim.name: shim for shim in load_providers_from_dir(root)}
    manifest_path = root / "linked" / "provider.yaml"
    leaf = next(
        leaf for leaf in _leaf_entries(manifest_path) if leaf["name"] == "linked--alpha"
    )
    view = _merged_connection({"api_key_env": "LINKED_KEY"}, leaf["own"], manifest_path)
    assert view["api_key_env"] == loaded["linked--alpha"].connection.api_key_env
    assert view["api_key_env"] == "LINKED_KEY"


def test_a_symlink_loop_terminates_without_duplicates(tmp_path):
    """A link back to an ancestor must not expand forever.

    ``demo/loop`` points at ``demo``: two names, one real directory.  The
    visited set is keyed on the resolved path, so the second visit is dropped;
    the timeout is what makes removing that guard a failing test rather than a
    hung suite.
    """
    root = tmp_path / "shims"
    demo = root / "demo"
    (demo / "alpha").mkdir(parents=True)
    (demo / "provider.yaml").write_text(
        _manifest("  api_key_env: DEMO_KEY\n"), encoding="utf-8"
    )
    (demo / "alpha" / "provider.yaml").write_text(
        _leaf("demo--alpha", "  display_name: alpha\n"), encoding="utf-8"
    )
    _symlink(demo, demo / "loop")

    rows = _within(5.0, lambda: manifest_rows(_request(tmp_path, root)))
    row = next(r for r in rows if r["vendor"] == "demo")
    assert row["leaves"] == ["demo--alpha"]

    # No duplicates in the walk itself: a repeated real directory would show
    # up here even where a set comparison would hide it.
    paths = _within(5.0, lambda: shim_manifests._walk_definitions(demo, prune=False))
    assert len(paths) == len(set(paths))


def test_the_effective_connection_is_the_loader_s_merge(tmp_path, source):
    """The preview believes this merge, so pin it to what the loader produces.

    ``_merged_connection`` delegates to the loader today; a second rule that
    drifted from it would name the wrong leaves after a save, and nothing else
    in this file loads the fixture through the core.
    """
    from llm_rosetta.shims import load_providers_from_dir

    manifest_path = source / "demo" / "provider.yaml"
    connection = {
        "base_url": "https://demo.example/v1",
        "api_key_env": "DEMO_KEY",
        "headers": {"x-api-version": "2024-01"},
    }
    loaded = {shim.name: shim for shim in load_providers_from_dir(source)}

    for leaf in _leaf_entries(manifest_path):
        view = _merged_connection(connection, leaf["own"], manifest_path)
        shim = loaded[leaf["name"]]
        assert view["base_url"] == shim.connection.base_url
        assert view["api_key_env"] == shim.connection.api_key_env
        assert view["headers"] == shim.connection.headers


def test_fields_reports_a_header_override_that_is_not_a_field_override():
    fields = _fields(
        {"headers": {"x-a": "1", "x-b": "2"}},
        [{"name": "leaf", "own": {"headers": {"x-a": "own"}}}],
    )

    assert fields["headers"]["x-a"]["overridden_by"] == ["leaf"]
    assert fields["headers"]["x-b"]["inherited_by"] == ["leaf"]


@pytest.mark.asyncio
async def test_shim_files_still_lists_only_shims(tmp_path, source):
    """The split is a contract: `shims` resolves a shim, `manifests` does not."""
    payload = _body(await get_loaded_shims(_request(tmp_path, source)))

    names = {row["name"] for row in payload["shims"]}
    assert {"demo--alpha", "demo--beta", "demo--gamma"} <= names
    # A manifest has no identity, so folding it into `shims` would put a row
    # there that no picker can resolve.
    assert "demo" not in names
    assert "demo" in {row["vendor"] for row in payload["manifests"]}
    # The manifest's own file never appears among the shim rows.
    manifest_paths = {row["path"] for row in payload["manifests"]}
    assert not manifest_paths & {row["path"] for row in payload["shims"]}
