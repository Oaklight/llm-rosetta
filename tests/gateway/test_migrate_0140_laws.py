"""Laws of ``llm-rosetta-migrate``: running it twice equals running it once.

``test_migrate_0140.py`` checks each retired name alone, and a second run for
that one config.  The checks here state the rules over combinations:

* the table has no chains: no replacement is itself a retired name, so a
  rewrite cannot be rewritten again (the structural reason a second run finds
  nothing);
* a config that spells any subset of the 31 retired names at once migrates to
  one with none of them left, and migrating the result changes nothing and
  reports no work (a note is not work);
* a name is migrated the same way alone or beside others: each provider entry
  of the combined config reads as it does in a single-row config;
* a config with no retired name comes back byte for byte, comments included,
  and comments survive a rewrite;
* a database with retired tags in its columns and counters is rewritten once:
  no retired value remains, ids and untouched values are unchanged, the
  source-provider counters keep their total, and a second run finds no work
  and writes no second backup; a dry run changes nothing and reports the
  counts the rewrite then makes.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from llm_rosetta.api import retired_names  # noqa: E402

from llm_rosetta_gateway._vendor import jsonx  # noqa: E402
from llm_rosetta_gateway.config.migrate import migrate_0140 as mig  # noqa: E402
from tests.gateway.test_migrate_0140 import _config_for  # noqa: E402

ROWS = retired_names.RETIRED_NAMES
_SETTINGS = settings(max_examples=120, deadline=None, derandomize=True)


def test_the_target_is_the_product_s_own() -> None:
    assert mig.retired_names is retired_names


def _names_of(kind: str) -> set[str]:
    return {r.name for r in ROWS if r.kind == kind}


@pytest.mark.parametrize("kind", retired_names.KINDS)
def test_no_replacement_is_a_retired_name_of_its_own_kind(kind: str) -> None:
    """Names are retired per slot: ``google_generate`` is a retired *shim* name
    and a current provider type, so the check is per kind, not over the table."""
    for row in (r for r in ROWS if r.kind == kind):
        assert row.replacement not in _names_of(kind), row.name
        assert mig.format_of(row.replacement) not in _names_of(
            retired_names.PROVIDER_TYPE
        ) | _names_of(retired_names.MODEL_TYPE), row.name


def test_the_database_rewrite_map_has_no_chains() -> None:
    mapping = mig._db_tag_map()
    assert set(mapping) & set(mapping.values()) == set()


# --- configs -----------------------------------------------------------------


def _combined(rows: list[retired_names.RetiredName]) -> dict:
    """One config that spells every row in *rows*, each under its own names."""
    config: dict = {"providers": {}, "models": {}}
    for index, row in enumerate(rows):
        single = _config_for(row)
        for name, entry in single.get("providers", {}).items():
            config["providers"][f"{name}{index}"] = entry
        for name, entry in single.get("models", {}).items():
            entry = {**entry, "provider": f"p{index}"}
            config["models"][f"{name}{index}"] = entry
        for key, value in single.items():
            if key not in ("providers", "models"):
                config[key] = value
    return config


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [s for item in value for s in _strings(item)]
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in [str(k), *_strings(v)]]
    return []


def _work(result: mig.ConfigResult) -> list[mig.Item]:
    return [i for i in result.items if i.action != "note"]


_SUBSETS = st.lists(
    st.sampled_from(range(len(ROWS))), unique=True, min_size=1, max_size=8
)


@_SETTINGS
@given(picked=_SUBSETS)
def test_a_config_with_many_retired_names_migrates_once_and_stays_migrated(
    picked,
) -> None:
    rows = [ROWS[i] for i in picked]
    text = json.dumps(_combined(rows), indent=2)
    first = mig.migrate_config_text(text)
    assert first.error is None
    assert not [i for i in first.items if i.action == "unmapped"]
    assert first.changed

    again = mig.migrate_config_text(first.new_text)
    assert not again.changed and again.new_text == first.new_text
    assert _work(again) == []

    new = jsonx.loads(first.new_text)
    for entry in new["providers"].values():
        if "type" in entry:
            assert entry["type"] not in _names_of(retired_names.PROVIDER_TYPE), entry
        if "shim" in entry:
            assert entry["shim"] not in _names_of(retired_names.SHIM), entry
    for entry in new["models"].values():
        assert entry["type"] not in {
            r.name for r in ROWS if r.kind == retired_names.MODEL_TYPE
        }
    # No top-level key still carries a retired model type's prefix.
    for r in rows:
        if r.kind == retired_names.MODEL_TYPE:
            assert not [k for k in new if k.startswith(f"{r.name}_")]


@_SETTINGS
@given(picked=_SUBSETS)
def test_a_provider_entry_migrates_the_same_beside_others_as_alone(picked) -> None:
    rows = [
        ROWS[i]
        for i in picked
        if ROWS[i].kind in (retired_names.PROVIDER_TYPE, retired_names.SHIM)
    ]
    if not rows:
        return
    together = jsonx.loads(
        mig.migrate_config_text(json.dumps(_combined(rows), indent=2)).new_text
    )
    for index, row in enumerate(rows):
        alone = jsonx.loads(
            mig.migrate_config_text(json.dumps(_config_for(row), indent=2)).new_text
        )
        assert together["providers"][f"p{index}"] == alone["providers"]["p"], row.name


_CLEAN_ENTRY = st.fixed_dictionaries(
    {"type": st.sampled_from(["chat_completions", "messages", "open_responses"])},
    optional={"api_key": st.text(alphabet="abc123", min_size=1, max_size=6)},
)


@_SETTINGS
@given(
    providers=st.dictionaries(
        st.text(alphabet="abcxyz", min_size=1, max_size=5), _CLEAN_ENTRY, max_size=4
    ),
    comment=st.text(alphabet="abc xyz", min_size=1, max_size=12),
)
def test_a_config_with_no_retired_name_comes_back_byte_for_byte(
    providers, comment
) -> None:
    body = json.dumps({"providers": providers}, indent=2)
    text = f"// {comment}\n" + body.replace("{", "{ /* keep */", 1) + "\n"
    result = mig.migrate_config_text(text)
    assert result.error is None
    assert not result.changed and result.new_text == text
    assert _work(result) == []


@_SETTINGS
@given(picked=_SUBSETS, comment=st.text(alphabet="abc xyz", min_size=1, max_size=12))
def test_comments_survive_a_rewrite(picked, comment) -> None:
    rows = [ROWS[i] for i in picked]
    body = json.dumps(_combined(rows), indent=2)
    text = f"// head {comment}\n" + body.replace("{", "{ /* keep */", 1) + "\n"
    out = mig.migrate_config_text(text).new_text
    assert f"// head {comment}" in out and "/* keep */" in out
    assert mig.migrate_config_text(out).new_text == out


# --- the database ------------------------------------------------------------

_TAGS = sorted(mig._db_tag_map()) + ["chat_completions", "messages", "llm", "main", ""]
_TAG = st.sampled_from(_TAGS)


def _build_db(path: Path, rows: list[tuple], counters: dict) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE request_log (id TEXT, source_provider TEXT, "
        "target_provider TEXT, target_provider_name TEXT, model_type TEXT)"
    )
    conn.execute(
        "CREATE TABLE error_dumps (id TEXT, source_provider TEXT, target_provider TEXT)"
    )
    conn.execute("CREATE TABLE metrics (key TEXT PRIMARY KEY, value TEXT)")
    conn.executemany("INSERT INTO request_log VALUES (?, ?, ?, ?, ?)", rows)
    conn.executemany(
        "INSERT INTO error_dumps VALUES (?, ?, ?)",
        [(r[0], r[1], r[2]) for r in rows[:2]],
    )
    conn.execute("INSERT INTO metrics VALUES ('counters', ?)", (json.dumps(counters),))
    conn.commit()
    conn.close()


def _snapshot(path: Path) -> list:
    conn = sqlite3.connect(path)
    try:
        return [
            conn.execute(f"SELECT * FROM {t} ORDER BY 1").fetchall()
            for t in ("request_log", "error_dumps", "metrics")
        ]
    finally:
        conn.close()


@st.composite
def _databases(draw):
    rows = [
        (str(i), draw(_TAG), draw(_TAG), draw(st.one_of(st.none(), _TAG)), draw(_TAG))
        for i in range(draw(st.integers(1, 6)))
    ]
    counters = {
        "by_source_provider": draw(
            st.dictionaries(_TAG, st.integers(1, 50), max_size=5)
        ),
        "by_target_provider": draw(
            st.dictionaries(_TAG, st.integers(1, 50), max_size=5)
        ),
    }
    return rows, counters


@_SETTINGS
@given(db=_databases())
def test_a_database_is_rewritten_once_and_a_second_run_finds_nothing(db) -> None:
    rows, counters = db
    mapping = mig._db_tag_map()
    types = mig._model_type_map()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gateway.db"
        _build_db(path, rows, counters)
        before = _snapshot(path)

        dry = mig.migrate_db(path)
        assert dry.error is None and dry.backup is None
        assert _snapshot(path) == before  # a dry run changes nothing

        result = mig.migrate_db(path, write=True)
        assert result.error is None
        after = _snapshot(path)

        logged = after[0]
        assert [r[0] for r in logged] == [r[0] for r in sorted(rows)]
        for old, new in zip(sorted(rows), logged):
            assert new[1] == mapping.get(old[1], old[1])
            assert new[2] == mapping.get(old[2], old[2])
            assert new[3] == old[3]  # the provider name column is never a tag
            assert new[4] == types.get(old[4], old[4])
        for _, src, dst in after[1]:
            assert src not in mapping and dst not in mapping

        folded = json.loads(after[2][0][1])
        assert sum(folded["by_source_provider"].values()) == sum(
            counters["by_source_provider"].values()
        )
        assert not set(folded["by_source_provider"]) & set(mapping)
        # by_target_provider mixes names and tags, and is left as it was.
        assert folded["by_target_provider"] == counters["by_target_provider"]

        # The dry run reported exactly the rows the rewrite changed.
        reported = {
            (i.where, i.old): i.count for i in dry.items if i.action == "rewrite"
        }
        for column_index, column in ((1, "source_provider"), (2, "target_provider")):
            for old in {r[column_index] for r in rows} & set(mapping):
                count = sum(1 for r in rows if r[column_index] == old)
                assert reported[(f"request_log.{column}", old)] == count

        files = sorted(p.name for p in Path(tmp).iterdir())
        again = mig.migrate_db(path, write=True)
        assert again.error is None and again.pending == 0 and again.backup is None
        assert sorted(p.name for p in Path(tmp).iterdir()) == files
        assert _snapshot(path) == after
        if before == after:
            assert result.backup is None and result.pending == 0
        else:
            assert result.backup is not None


def test_the_commented_sample_of_the_old_tests_is_idempotent_too() -> None:
    from tests.gateway.test_migrate_0140 import COMMENTED, COMMENTED_EXPECTED

    first = mig.migrate_config_text(COMMENTED)
    assert first.new_text == COMMENTED_EXPECTED
    again = mig.migrate_config_text(first.new_text)
    assert not again.changed and _work(again) == []
