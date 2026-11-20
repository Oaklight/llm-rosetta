"""Retired names through the real gateway config loader."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pytest

from llm_rosetta import retired_names
from llm_rosetta.retired_names import (
    MODEL_TYPE,
    PROVIDER_TYPE,
    RETIRED_NAMES,
    SHIM,
)
from llm_rosetta_gateway.config.loader import load_gateway_config
from llm_rosetta_gateway.maintenance.names import (
    correct_legacy_names,
    legacy_name_error,
)
from llm_rosetta_gateway.providers import BASE_FORMATS
from llm_rosetta_gateway.shims.catalog import (
    SHIM_SOURCE_PACKAGES,
    canonical_shim_for,
    find_legacy_source_package,
)

GOLDEN = Path(__file__).parent.parent / "data" / "retired_names_golden.json"
URL = "http://upstream.invalid/v1"


@pytest.fixture
def aliases_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retired_names, "ALIASES_ENABLED", False)


def _load(tmp_path: Path, doc: dict[str, Any]):
    path = tmp_path / "config.jsonc"
    path.write_text(json.dumps(doc))
    return load_gateway_config(str(path))


def _provider_docs(row: retired_names.RetiredName) -> list[dict[str, Any]]:
    """Every config shape through which *row*'s name can enter a provider."""
    docs: list[dict[str, Any]] = []
    if row.kind == PROVIDER_TYPE:
        docs.append({"p": {"type": row.name, "base_url": URL}})
        docs.append({row.name: {"base_url": URL}})
    elif row.kind == SHIM:
        docs.append({"p": {"shim": row.name, "base_url": URL}})
        if row.name not in BASE_FORMATS:
            docs.append({"p": {"type": row.name, "base_url": URL}})
            docs.append({row.name: {"base_url": URL}})
    return docs


def _cases():
    for row in RETIRED_NAMES:
        if row.kind == MODEL_TYPE:
            yield pytest.param(
                row,
                {
                    "providers": {"p": {"type": "chat_completions", "base_url": URL}},
                    "models": {"m": {"provider": "p", "type": row.name}},
                    # A pre-v5 document has its model type rewritten by the
                    # versioned migration (hard-coded, not an alias); only a
                    # current document reaches the alias path.
                    "config_version": 5,
                },
                id=f"{row.kind}:{row.name}:model-type",
            )
            continue
        for i, providers in enumerate(_provider_docs(row)):
            yield pytest.param(
                row,
                {"providers": providers, "models": {}},
                id=f"{row.kind}:{row.name}:{i}",
            )


@pytest.mark.parametrize(("row", "doc"), list(_cases()))
def test_loader_rejects_every_retired_name_with_aliases_off(
    row: retired_names.RetiredName,
    doc: dict[str, Any],
    aliases_off: None,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError) as info:
        _load(tmp_path, doc)
    message = str(info.value)
    replacement = row.replacement
    slot_is_shim = "shim" in next(iter(doc["providers"].values()), {})
    if row.kind == SHIM and not slot_is_shim:
        # ``type:`` and a bare provider key are read as a provider type first,
        # so a name retired under both kinds (``google``, ``anthropic``)
        # answers with the provider-type replacement.
        other = retired_names.lookup(row.name, PROVIDER_TYPE)
        if other is not None:
            replacement = other.replacement
    assert f"{row.name!r} was renamed to {replacement!r}" in message
    if row.preset:
        assert f"shims.presets: ['{row.preset}']" in message


def test_every_retired_name_has_a_loader_case() -> None:
    covered = {c.values[0].name + c.values[0].kind for c in _cases()}
    assert covered == {r.name + r.kind for r in RETIRED_NAMES}


def test_type_openai_chat_with_base_url_no_longer_loads_silently(
    aliases_off: None, tmp_path: Path
) -> None:
    """Measured before this change: it loaded as an unknown type."""
    doc = {"providers": {"p": {"type": "openai_chat", "base_url": URL}}, "models": {}}
    with pytest.raises(ValueError, match="'chat_completions'"):
        _load(tmp_path, doc)


@pytest.mark.parametrize(
    "typ", ["openai_chatt", "chat_completionz", "antropic", "embeddings"]
)
def test_typos_are_not_turned_into_rename_hints(
    typ: str, aliases_off: None, tmp_path: Path
) -> None:
    doc = {"providers": {"p": {"type": typ, "base_url": URL}}, "models": {}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            _load(tmp_path, doc)
        except ValueError as exc:
            assert "was renamed" not in str(exc)
    assert legacy_name_error({"p": {"type": typ}}) is None


def test_aliases_on_still_corrects_and_writes_back(tmp_path: Path) -> None:
    path = tmp_path / "config.jsonc"
    path.write_text(
        json.dumps(
            {"providers": {"p": {"type": "openai_chat", "base_url": URL}}, "models": {}}
        )
    )
    config = load_gateway_config(str(path))
    assert config.provider_types == {"p": "chat_completions"}
    assert json.loads(path.read_text())["providers"]["p"]["type"] == "chat_completions"


def test_aliases_off_leaves_the_config_file_untouched(
    aliases_off: None, tmp_path: Path
) -> None:
    path = tmp_path / "config.jsonc"
    path.write_text(
        json.dumps(
            {"providers": {"p": {"type": "openai_chat", "base_url": URL}}, "models": {}}
        )
    )
    with pytest.raises(ValueError):
        load_gateway_config(str(path))
    # The versioned migration may still rewrite the file's structure, but the
    # retired name itself is left for the operator to fix.
    assert json.loads(path.read_text())["providers"]["p"]["type"] == "openai_chat"


# ---------------------------------------------------------------------------
# The catalogue and the table agree
# ---------------------------------------------------------------------------


def test_presets_match_the_shim_catalogue() -> None:
    for row in RETIRED_NAMES:
        if row.kind != SHIM:
            continue
        package = find_legacy_source_package(row.name)
        assert (package.key if package else None) == row.preset, row.name
        if package is not None:
            assert canonical_shim_for(package, row.name) == row.replacement


def test_every_catalogue_legacy_name_is_retired() -> None:
    for package in SHIM_SOURCE_PACKAGES:
        for name in package.legacy_names:
            assert retired_names.lookup(name, SHIM) is not None, name


# ---------------------------------------------------------------------------
# Default behaviour is unchanged: gateway half of the differential test
# ---------------------------------------------------------------------------


def _call(fn, *args):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out: dict[str, Any] = {"ok": repr(fn(*args))}
        except Exception as exc:  # noqa: BLE001 - recorded, compared verbatim
            out = {"err": f"{type(exc).__name__}: {exc}"}
        out["warnings"] = [f"{w.category.__name__}: {w.message}" for w in caught]
    return out


def _identities(cfgs: dict[str, Any]) -> dict[str, tuple[str, str]]:
    from llm_rosetta_gateway.config.resolve import resolve_provider_identities

    return {
        k: (v.model_type, v.format)
        for k, v in resolve_provider_identities(cfgs)[0].items()
    }


@pytest.mark.parametrize("name", sorted({row.name for row in RETIRED_NAMES}))
def test_default_gateway_behaviour_matches_the_recording(name: str) -> None:
    recorded = json.loads(GOLDEN.read_text())[name]
    live: dict[str, Any] = {}
    for slot in ("shim", "type"):
        raw = {"providers": {"p": {slot: name}}}
        live[f"correct_{slot}"] = _call(
            lambda raw=raw, slot=slot: (
                correct_legacy_names(raw),
                raw["providers"]["p"][slot],
            )
        )
        live[f"legacy_error_{slot}"] = _call(legacy_name_error, {"p": {slot: name}})
        live[f"identities_{slot}"] = _call(
            _identities, {"p": {slot: name, "base_url": "http://x.invalid"}}
        )
    live["identities_key"] = _call(
        _identities, {name: {"base_url": "http://x.invalid"}}
    )
    for key, value in live.items():
        assert value == recorded[key], key
