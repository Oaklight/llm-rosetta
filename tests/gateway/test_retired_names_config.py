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
from llm_rosetta_gateway.maintenance.names import legacy_name_error
from llm_rosetta_gateway.providers import BASE_FORMATS
from llm_rosetta_gateway.shims.catalog import (
    SHIM_SOURCE_PACKAGES,
    canonical_shim_for,
    find_legacy_source_package,
)

URL = "http://upstream.invalid/v1"


def _load(tmp_path: Path, doc: dict[str, Any]):
    path = tmp_path / "config.jsonc"
    path.write_text(json.dumps(doc))
    return load_gateway_config(str(path))


def _provider_docs(row: retired_names.RetiredName) -> list[dict[str, Any]]:
    """Every config shape through which *row*'s name can enter a provider."""
    docs: list[dict[str, Any]] = []
    if row.kind == PROVIDER_TYPE:
        docs.append(dict(p=dict(type=row.name, base_url=URL)))
        docs.append({row.name: dict(base_url=URL)})
    elif row.kind == SHIM:
        docs.append(dict(p=dict(shim=row.name, base_url=URL)))
        if row.name not in BASE_FORMATS:
            docs.append(dict(p=dict(type=row.name, base_url=URL)))
            docs.append({row.name: dict(base_url=URL)})
    return docs


def _cases():
    for row in RETIRED_NAMES:
        if row.kind == MODEL_TYPE:
            yield pytest.param(
                row,
                dict(
                    providers=dict(p=dict(type="chat_completions", base_url=URL)),
                    models=dict(m=dict(provider="p", type=row.name)),
                    # A pre-v5 document has its model type rewritten by the
                    # versioned migration (hard-coded, not an alias); only a
                    # current document reaches the alias path.
                    config_version=5,
                ),
                id=f"{row.kind}:{row.name}:model-type",
            )
            continue
        for i, providers in enumerate(_provider_docs(row)):
            yield pytest.param(
                row,
                dict(providers=providers, models={}),
                id=f"{row.kind}:{row.name}:{i}",
            )


@pytest.mark.parametrize(("row", "doc"), list(_cases()))
def test_loader_rejects_every_retired_name(
    row: retired_names.RetiredName,
    doc: dict[str, Any],
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
    assert message.endswith(retired_names.MIGRATE_HINT)
    if row.preset:
        assert f"shims.presets: ['{row.preset}']" in message


def test_every_retired_name_has_a_loader_case() -> None:
    covered = {c.values[0].name + c.values[0].kind for c in _cases()}
    assert covered == {r.name + r.kind for r in RETIRED_NAMES}


def test_type_openai_chat_with_base_url_does_not_load(tmp_path: Path) -> None:
    """Measured before #919: it loaded as an unknown type."""
    doc = dict(providers=dict(p=dict(type="openai_chat", base_url=URL)), models={})
    with pytest.raises(ValueError, match="'chat_completions'"):
        _load(tmp_path, doc)


@pytest.mark.parametrize(
    "typ", ["openai_chatt", "chat_completionz", "antropic", "embeddings"]
)
def test_typos_are_not_turned_into_rename_hints(typ: str, tmp_path: Path) -> None:
    doc = dict(providers=dict(p=dict(type=typ, base_url=URL)), models={})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            _load(tmp_path, doc)
        except ValueError as exc:
            assert "was renamed" not in str(exc)
    assert legacy_name_error(dict(p=dict(type=typ))) is None


def test_the_config_file_is_left_for_the_operator_to_fix(tmp_path: Path) -> None:
    """A retired name is never rewritten on disk: the load fails instead."""
    path = tmp_path / "config.jsonc"
    path.write_text(
        json.dumps(
            dict(providers=dict(p=dict(type="openai_chat", base_url=URL)), models={})
        )
    )
    with pytest.raises(ValueError):
        load_gateway_config(str(path))
    assert json.loads(path.read_text())["providers"]["p"]["type"] == "openai_chat"


def test_canonical_names_load_without_warnings(tmp_path: Path) -> None:
    doc = dict(
        providers=dict(p=dict(type="chat_completions", base_url=URL)),
        models={},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config = _load(tmp_path, doc)
    assert config.provider_types == dict(p="chat_completions")


def test_a_base_format_type_is_not_read_as_a_retired_shim(tmp_path: Path) -> None:
    """``type: google_generate`` names the format; the shim row must not fire."""
    doc = dict(providers=dict(p=dict(type="google_generate", base_url=URL)), models={})
    config = _load(tmp_path, doc)
    assert config.provider_types == dict(p="google_generate")


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


def test_every_shim_replacement_is_registered_or_packaged() -> None:
    """A retired shim name has to lead somewhere that exists.

    The replacement is a shim core ships, one a standalone package provides,
    or one a vendor that moved to contrib ships (#1660) — the gateway installs
    that vendor by itself.  A typo on any side shows up here rather than as a
    silent ``get_shim(...) is None`` at runtime.
    """
    from llm_rosetta.retired_names import moved_vendor
    from llm_rosetta.shims import list_shims
    from llm_rosetta.shims.providers import load_providers
    from llm_rosetta_gateway.shims import find_source_package

    load_providers()
    registered = {s.name for s in list_shims()}

    def moved(name: str) -> bool:
        row = moved_vendor(name)
        return row is not None and name in row.shims

    missing = [
        row.name
        for row in RETIRED_NAMES
        if row.kind == SHIM
        and row.replacement not in registered
        and find_source_package(row.replacement) is None
        and not moved(row.replacement)
    ]
    assert missing == []


# ---------------------------------------------------------------------------
# Retired model-type spellings in config keys
# ---------------------------------------------------------------------------

_KEY_CASES = [
    ("embedding_providers", "embeddings_providers", "root"),
    ("embedding_models", "embeddings_models", "root"),
    ("default_embedding_format", "default_embeddings_format", "root"),
    ("decision_providers", "decisions_providers", "root"),
    ("embedding_format", "embeddings_format", "provider"),
    ("embedding_path", "embeddings_path", "provider"),
    ("embedding_shim", "embeddings_shim", "provider"),
    ("decision_enabled", "decisions_enabled", "provider"),
    ("embedding", "embeddings", "provider"),
]


@pytest.mark.parametrize(
    ("old", "new", "where"), _KEY_CASES, ids=[case[0] for case in _KEY_CASES]
)
def test_a_retired_config_key_names_its_replacement(
    old: str, new: str, where: str, tmp_path: Path
) -> None:
    provider: dict[str, Any] = dict(type="chat_completions", base_url=URL, api_key="k")
    doc: dict[str, Any] = dict(config_version=5, providers=dict(p=provider), models={})
    if where == "root":
        doc[old] = {}
    else:
        provider[old] = {} if old == "embedding" else "x"
    with pytest.raises(ValueError) as info:
        _load(tmp_path, doc)
    message = str(info.value)
    assert f"{old!r} was renamed to {new!r}" in message
    assert message.endswith(retired_names.MIGRATE_HINT)


@pytest.mark.parametrize("capability", ["embedding", "decision"])
def test_a_retired_capability_spelling_names_its_replacement(
    capability: str, tmp_path: Path
) -> None:
    doc = dict(
        config_version=5,
        providers=dict(p=dict(type="chat_completions", base_url=URL)),
        models=dict(m=dict(provider="p", capabilities=[capability])),
    )
    with pytest.raises(ValueError, match="was renamed"):
        _load(tmp_path, doc)


def test_the_current_spellings_still_load(tmp_path: Path) -> None:
    doc = dict(
        config_version=5,
        providers=dict(
            p=dict(
                type="chat_completions",
                base_url=URL,
                api_key="k",
                embeddings_format="openai",
            )
        ),
        models={},
    )
    config = _load(tmp_path, doc)
    assert "p" in config.embeddings_providers
