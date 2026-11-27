"""``llm-rosetta gateway migrate`` and the old ``manifest.json`` shim sources (#1818).

A config written before the extension index lists ``shims.sources`` entries that
point at a ``dist/manifest.json``.  The gateway reads only ``index.json`` now, so
the entry has to be rewritten to the published index of the same repository, and
an address the tool cannot map has to be reported, never skipped in silence.
"""

from __future__ import annotations

import pytest

from llm_rosetta_gateway._vendor import jsonx
from llm_rosetta_gateway.config.migrate import migrate_0140 as mig
from llm_rosetta_gateway.shims import shim_sources
from llm_rosetta_gateway.shims.catalog import get_shim_source
from llm_rosetta_gateway.shims.manifest_sources.plan import parse_source_entry

CONTRIB = "https://github.com/Oaklight/llm-rosetta-ext-contrib/releases/latest/download/index.json"
ARGONNE = "https://github.com/Oaklight/llm-rosetta-ext-argonne/releases/latest/download/index.json"

#: The repositories that published a manifest, with the index that replaces each.
REPOS = {
    "llm-rosetta-shim-contrib": CONTRIB,
    "llm-rosetta-shim-free": CONTRIB,
    "llm-rosetta-ext-contrib": CONTRIB,
    "llm-rosetta-shim-argonne": ARGONNE,
    "llm-rosetta-shim-argo": ARGONNE,
    "llm-rosetta-ext-argonne": ARGONNE,
}

#: Each spelling of "the manifest of repository *name*" a config ever held.
SPELLINGS = {
    "jsdelivr": "https://cdn.jsdelivr.net/gh/Oaklight/{name}@dist/manifest.json",
    "raw": "https://raw.githubusercontent.com/Oaklight/{name}/dist/manifest.json",
    "jsdelivr-commit": "https://cdn.jsdelivr.net/gh/Oaklight/{name}@dist/0a1b2c3d4e5f/manifest.json",
    "raw-commit": "https://raw.githubusercontent.com/Oaklight/{name}/dist/0a1b2c3d4e5f/manifest.json",
}

OLD_CASES = [
    pytest.param(name, SPELLINGS[kind].format(name=name), index, id=f"{kind}-{name}")
    for name, index in REPOS.items()
    for kind in SPELLINGS
]


def _doc(sources: str, extra: str = "") -> str:
    """A config shaped like a production one: comments, providers, a ``shims`` section."""
    return (
        "{\n"
        "  // gateway\n"
        '  "server": {"port": 8000},\n'
        '  "providers": {\n'
        '    "argo": {"shim": "argonne--argo", "api_key": "PLACEHOLDER-KEY"},\n'
        '    "free": {"type": "chat_completions", "api_key": "PLACEHOLDER-KEY"}\n'
        "  },\n"
        '  "shims": {\n'
        f'    "sources": {sources},\n'
        '    "install": ["argonne:argo"]' + extra + "\n"
        "  }\n"
        "}\n"
    )


def _unmapped(result: mig.ConfigResult) -> list[mig.Item]:
    return [i for i in result.items if i.action == "unmapped"]


@pytest.mark.parametrize(("name", "old", "index"), OLD_CASES)
def test_an_old_manifest_source_is_rewritten_to_the_index_of_its_repository(
    name: str, old: str, index: str
) -> None:
    result = mig.migrate_config_text(_doc(f'["{old}"]'))
    assert jsonx.loads(result.new_text)["shims"]["sources"] == [index]
    rewrites = [i for i in result.items if i.action == "rewrite"]
    assert [(i.where, i.old, i.new) for i in rewrites] == [
        ("shims.sources[0]", old, index)
    ]
    assert not _unmapped(result)


def test_only_the_source_entry_changes_in_the_text() -> None:
    old = SPELLINGS["jsdelivr"].format(name="llm-rosetta-shim-argonne")
    text = _doc(f'["{old}"]')
    result = mig.migrate_config_text(text)
    assert result.new_text == text.replace(old, ARGONNE)
    assert "// gateway" in result.new_text


def test_a_production_shaped_source_list_is_rewritten_whole() -> None:
    contrib = SPELLINGS["raw"].format(name="llm-rosetta-shim-contrib")
    argonne = SPELLINGS["jsdelivr"].format(name="llm-rosetta-shim-argonne")
    result = mig.migrate_config_text(_doc(f'["{argonne}", "{contrib}"]'))
    assert jsonx.loads(result.new_text)["shims"]["sources"] == [ARGONNE, CONTRIB]
    assert len([i for i in result.items if i.action == "rewrite"]) == 2
    # Nothing is left for a second run, and the gateway parses what was written.
    assert not mig.migrate_config_text(result.new_text).changed
    catalogue = [get_shim_source(key) for key in ("argonne", "contrib")]
    urls = {source.manifest_url for source in catalogue if source is not None}
    assert len(urls) == 2
    for entry in jsonx.loads(result.new_text)["shims"]["sources"]:
        assert parse_source_entry(entry).url in urls


def test_an_object_entry_keeps_its_token_and_drops_a_fallback_that_is_a_manifest() -> (
    None
):
    primary = SPELLINGS["jsdelivr"].format(name="llm-rosetta-shim-contrib")
    fallback = SPELLINGS["raw"].format(name="llm-rosetta-shim-contrib")
    entry = (
        f'[{{"url": "{primary}", "fallback": "{fallback}", "token_env": "SHIM_TOKEN"}}]'
    )
    result = mig.migrate_config_text(_doc(entry))
    assert jsonx.loads(result.new_text)["shims"]["sources"] == [
        {"url": CONTRIB, "token_env": "SHIM_TOKEN"}
    ]
    assert not _unmapped(result)


def test_a_manifest_this_tool_cannot_map_is_reported_and_left_alone() -> None:
    third = "https://shims.example.org/acme/dist/manifest.json"
    text = _doc(f'["{third}"]')
    result = mig.migrate_config_text(text)
    assert result.new_text == text
    (item,) = _unmapped(result)
    assert (item.where, item.old) == ("shims.sources[0]", third)
    # The operator learns what the gateway reads now and what to do by hand.
    assert "index.json" in item.detail
    assert "shims.sources" in item.detail


def test_a_fork_with_the_name_of_a_known_repository_is_not_mapped() -> None:
    """A repository is known by owner and name; the name alone proves nothing."""
    fork = (
        "https://cdn.jsdelivr.net/gh/acme/llm-rosetta-shim-contrib@dist/manifest.json"
    )
    result = mig.migrate_config_text(_doc(f'["{fork}"]'))
    assert not result.changed
    assert [i.where for i in _unmapped(result)] == ["shims.sources[0]"]


def test_an_unmapped_manifest_in_an_object_entry_is_reported_with_its_key() -> None:
    third = "https://shims.example.org/acme/dist/manifest.json"
    result = mig.migrate_config_text(_doc(f'[{{"url": "{third}", "token_env": "T"}}]'))
    assert [i.where for i in _unmapped(result)] == ["shims.sources[0].url"]
    assert not result.changed


def test_the_mapped_entries_are_rewritten_beside_an_unmapped_one() -> None:
    old = SPELLINGS["jsdelivr"].format(name="llm-rosetta-shim-argonne")
    third = "https://shims.example.org/acme/dist/manifest.json"
    result = mig.migrate_config_text(_doc(f'["{old}", "{third}"]'))
    assert jsonx.loads(result.new_text)["shims"]["sources"] == [ARGONNE, third]
    assert [i.where for i in _unmapped(result)] == ["shims.sources[1]"]


@pytest.mark.parametrize(
    "entry",
    [
        "contrib",
        CONTRIB,
        ARGONNE,
        "https://shims.example.org/acme/index.json",
        "https://cdn.jsdelivr.net/gh/Oaklight/some-other-repo@dist/data.json",
    ],
)
def test_a_current_or_unrelated_source_is_not_touched(entry: str) -> None:
    text = _doc(f'["{entry}"]')
    result = mig.migrate_config_text(text)
    assert not result.changed
    assert not _unmapped(result)


def test_a_config_with_no_sources_is_not_touched() -> None:
    text = '{"shims": {"install": ["argonne:argo"]}}\n'
    assert not mig.migrate_config_text(text).changed


def test_the_retired_paths_advice_names_the_file_the_loader_reads() -> None:
    advice = shim_sources.RETIRED_KEYS["paths"]
    assert "plugin.yaml" in advice
    assert "extension.yaml" not in advice
