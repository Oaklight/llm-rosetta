"""``llm-rosetta-migrate``: the one-shot migration for the 0.14.0 removals (#919).

Config rewrites are checked against the retired-names table itself, so a row
added to the table without a rewrite fails here.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from llm_rosetta.api import FORMATS_BY_TYPE, retired_names

from llm_rosetta_gateway._vendor import jsonx
from llm_rosetta_gateway.maintenance import migrate_0140 as mig

ROWS = retired_names.RETIRED_NAMES


def _load(text: str) -> dict:
    return jsonx.loads(text)


def _config_for(row: retired_names.RetiredName) -> dict:
    """The smallest config that spells *row* where the loader reads it."""
    if row.kind == retired_names.PROVIDER_TYPE:
        return dict(providers=dict(p=dict(type=row.name, api_key="k")))
    if row.kind == retired_names.SHIM:
        return dict(providers=dict(p=dict(shim=row.name, api_key="k")))
    config: dict = dict(
        providers=dict(p=dict(api_key="k", type="chat_completions")),
        models=dict(m=dict(provider="p", type=row.name, capabilities=[row.name])),
    )
    config[f"{row.name}_providers"] = dict(q=dict(format="openai"))
    config[f"default_{row.name}_format"] = "openai"
    config["providers"]["p"][f"{row.name}_format"] = "openai"
    return config


def _strings(value: object) -> list[str]:
    """Every string, key or value, in a decoded config."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [s for item in value for s in _strings(item)]
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in [str(k), *_strings(v)]]
    return []


def _brace(text: str) -> str:
    """Samples spell braces, slashes and the dollar sign as markers."""
    slash = chr(47)
    marks = dict(
        ENV=chr(36) + chr(123) + "OPENAI_API_KEY" + chr(125),
        C=slash * 2,
        BO=slash + "*",
        BC="*" + slash,
        P=slash + "v1" + slash + "embeddings",
    )
    text = text.replace("\u00ab", chr(123)).replace("\u00bb", chr(125))
    for key, value in marks.items():
        text = text.replace(f"@{key}@", value)
    return text


def test_the_table_has_the_31_rows_the_guide_counts() -> None:
    assert len(ROWS) == 31


@pytest.mark.parametrize("row", ROWS, ids=lambda r: f"{r.kind}:{r.name}")
def test_every_retired_name_migrates(row: retired_names.RetiredName) -> None:
    text = json.dumps(_config_for(row), indent=2)
    result = mig.migrate_config_text(text)
    assert result.error is None
    assert not [i for i in result.items if i.action == "unmapped"]
    new = _load(result.new_text)
    if row.kind == retired_names.MODEL_TYPE:
        assert new["models"]["m"]["type"] == row.replacement
        assert new["models"]["m"]["capabilities"] == [row.replacement]
        assert f"{row.replacement}_providers" in new
        assert f"default_{row.replacement}_format" in new
        assert f"{row.replacement}_format" in new["providers"]["p"]
        keys = [k for k in _strings(new) if k.startswith(f"{row.name}_")]
        assert keys == []
    elif row.kind == retired_names.SHIM:
        assert new["providers"]["p"]["shim"] == row.replacement
        presets = new.get("shims", dict()).get("presets", [])
        assert (row.preset in presets) == bool(row.preset)
    else:
        assert new["providers"]["p"]["type"] == row.replacement
    # A second run finds nothing to do.
    again = mig.migrate_config_text(result.new_text)
    assert not again.changed
    assert again.items == []


def _doc(*lines: str) -> str:
    return _brace("\n".join(lines) + "\n")


COMMENTED = _doc(
    "«",
    "  @C@ gateway config",
    '  "providers": «',
    "    @BO@ the main one @BC@",
    '    "main": «',
    '      "type": "openai_chat",  @C@ the chat lane',
    '      "api_key": "@ENV@",',
    "    »,",
    '    "claude": «',
    '      "shim": "anthropic",',
    '      "embedding_format": "openai",',
    '      "embedding": «',
    '        "format": "google",',
    "      »,",
    "    »,",
    "  »,",
    '  "models": «',
    '    "emb": «',
    '      "provider": "main",',
    '      "type": "embedding",   @C@ keep this comment',
    '      "capabilities": ["embedding"],',
    "    »,",
    "  »,",
    '  "default_embedding_format": "openai",',
    '  "embedding_providers": «',
    '    "e1": «',
    '      "format": "google",',
    '      "embedding_path": "@P@",',
    "    »,",
    "  »,",
    "»",
)

COMMENTED_EXPECTED = _doc(
    "«",
    "  @C@ gateway config",
    '  "providers": «',
    "    @BO@ the main one @BC@",
    '    "main": «',
    '      "type": "chat_completions",  @C@ the chat lane',
    '      "api_key": "@ENV@",',
    "    »,",
    '    "claude": «',
    '      "shim": "anthropic--messages",',
    '      "embeddings_format": "openai",',
    '      "embeddings": «',
    '        "format": "google_generate",',
    "      »,",
    "    »,",
    "  »,",
    '  "models": «',
    '    "emb": «',
    '      "provider": "main",',
    '      "type": "embeddings",   @C@ keep this comment',
    '      "capabilities": ["embeddings"],',
    "    »,",
    "  »,",
    '  "default_embeddings_format": "openai",',
    '  "embeddings_providers": «',
    '    "e1": «',
    '      "format": "google_generate",',
    '      "embeddings_path": "@P@",',
    "    »,",
    "  »,",
    "»",
)


def test_comments_order_and_formatting_survive() -> None:
    result = mig.migrate_config_text(COMMENTED)
    assert result.error is None
    # Byte-for-byte: only the retired names changed, nothing else moved.
    assert result.new_text == COMMENTED_EXPECTED
    assert [i for i in result.items if i.action == "unmapped"] == []
    for comment in ("@C@ gateway config", "@BO@ the main one @BC@", "@C@ keep this"):
        assert _brace(comment) in result.new_text


def test_preset_is_appended_to_an_existing_list_and_comments_stay() -> None:
    text = _doc(
        "«",
        '  "providers": «',
        '    "a": « "shim": "argo--openai_chat" »,',
        '    "b": « "shim": "alcf--sophia" »,',
        "  »,",
        '  "shims": «',
        '    "presets": ["asksage"],  @C@ keep',
        "  »,",
        "»",
    )
    result = mig.migrate_config_text(text)
    assert _brace('"presets": ["asksage", "alcf", "argo"],  @C@ keep') in (
        result.new_text
    )
    new = _load(result.new_text)
    assert new["providers"]["a"]["shim"] == "argo--chat_completions"
    assert new["providers"]["b"]["shim"] == "alcf_sophia--chat_completions"
    assert [i.new for i in result.items if i.action == "add"] == ["alcf", "argo"]
    assert not mig.migrate_config_text(result.new_text).changed


def test_preset_creates_the_shims_section_when_absent() -> None:
    text = _doc(
        "«",
        "  @C@ only a shim",
        '  "providers": «',
        '    "a": « "shim": "asksage--anthropic" »,',
        "  »,",
        "»",
    )
    result = mig.migrate_config_text(text)
    assert _brace("@C@ only a shim") in result.new_text
    assert _load(result.new_text)["shims"]["presets"] == ["asksage"]
    assert not mig.migrate_config_text(result.new_text).changed


def test_preset_already_declared_is_not_added_twice() -> None:
    config = dict(
        providers=dict(a=dict(shim="argo--anthropic")),
        shims=dict(presets=["argo"]),
    )
    result = mig.migrate_config_text(json.dumps(config, indent=2))
    assert _load(result.new_text)["shims"]["presets"] == ["argo"]
    assert [i for i in result.items if i.action == "add"] == []


def test_shims_lists_and_endpoint_shims_are_rewritten() -> None:
    config = dict(
        providers=dict(a=dict(type="chat_completions", embeddings_shim="openai")),
        shims=dict(disabled=["minimax--openai_chat"]),
    )
    result = mig.migrate_config_text(json.dumps(config, indent=2))
    new = _load(result.new_text)
    assert new["providers"]["a"]["embeddings_shim"] == "openai--chat_completions"
    assert new["shims"]["disabled"] == ["minimax--chat_completions"]


def test_a_current_format_that_is_also_a_retired_shim_is_kept() -> None:
    config = dict(providers=dict(a=dict(type="google_generate")))
    assert not mig.migrate_config_text(json.dumps(config)).changed


def test_replacement_formats_are_real_formats() -> None:
    llm = FORMATS_BY_TYPE["llm"]
    assert all(fmt in llm for fmt in mig._format_tag_map().values())
    assert mig._format_tag_map()["openai_responses"] == "open_responses"


def _unmapped(text: str) -> list[mig.Item]:
    result = mig.migrate_config_text(text)
    return [i for i in result.items if i.action == "unmapped"]


def test_a_provider_key_that_is_a_retired_name_is_reported_not_renamed() -> None:
    config = dict(providers=dict(anthropic=dict(api_key="k")))
    text = json.dumps(config, indent=2)
    result = mig.migrate_config_text(text)
    assert result.new_text == text
    [item] = [i for i in result.items if i.action == "unmapped"]
    assert item.old == "anthropic"
    assert "anthropic--messages" in item.detail


def test_a_key_collision_is_reported_and_nothing_is_overwritten() -> None:
    config = dict(embedding_providers=dict(), embeddings_providers=dict())
    text = json.dumps(config)
    result = mig.migrate_config_text(text)
    assert result.new_text == text
    assert [i.old for i in _unmapped(text)] == ["embedding_providers"]


def test_a_compound_retired_name_in_an_unknown_place_is_reported() -> None:
    config = dict(models=dict(m=dict(provider="argo--openai_chat", note="google")))
    items = _unmapped(json.dumps(config))
    assert [i.old for i in items] == ["argo--openai_chat"]
    assert items[0].where == "models.m.provider"


def test_unparseable_config_is_an_error_and_unchanged() -> None:
    result = mig.migrate_config_text("not json")
    assert result.error and not result.changed


# --------------------------------------------------------------------------
# The command: dry run, --write, report, exit codes
# --------------------------------------------------------------------------


def _write(tmp_path: Path, text: str = COMMENTED) -> Path:
    path = tmp_path / "config.jsonc"
    path.write_bytes(text.encode())
    return path


def test_dry_run_prints_a_diff_and_touches_nothing(tmp_path, capsys) -> None:
    path = _write(tmp_path)
    before = path.read_bytes()
    assert mig.main(["--config", str(path)]) == 0
    out = capsys.readouterr().out
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["config.jsonc"]
    assert '-      "type": "openai_chat",' in out
    assert '+      "type": "chat_completions",' in out
    assert "dry run" in out


def test_write_backs_up_then_rewrites_and_is_idempotent(tmp_path) -> None:
    path = _write(tmp_path)
    original = path.read_bytes()
    assert mig.main(["--config", str(path), "--write"]) == 0
    assert path.read_text() == COMMENTED_EXPECTED
    assert (tmp_path / "config.jsonc.bak").read_bytes() == original
    after = path.read_bytes()
    files = sorted(p.name for p in tmp_path.iterdir())
    # The second run has nothing to do: no change, no new backup.
    assert mig.main(["--config", str(path), "--write"]) == 0
    assert path.read_bytes() == after
    assert sorted(p.name for p in tmp_path.iterdir()) == files


def test_write_keeps_a_bom_and_crlf(tmp_path) -> None:
    text = COMMENTED.replace("\n", "\r\n")
    path = tmp_path / "config.jsonc"
    path.write_bytes(b"\xef\xbb\xbf" + text.encode())
    assert mig.main(["--config", str(path), "--write"]) == 0
    expected = COMMENTED_EXPECTED.replace("\n", "\r\n").encode()
    assert path.read_bytes() == b"\xef\xbb\xbf" + expected


def test_check_exits_3_when_rewrites_are_pending(tmp_path) -> None:
    path = _write(tmp_path)
    assert mig.main(["--config", str(path), "--check"]) == 3
    assert mig.main(["--config", str(path), "--write"]) == 0
    assert mig.main(["--config", str(path), "--check"]) == 0


def test_unmapped_items_exit_1_and_the_rest_is_still_written(tmp_path) -> None:
    config = dict(
        providers=dict(
            anthropic=dict(api_key="k"),
            other=dict(type="openai_chat"),
        )
    )
    path = _write(tmp_path, json.dumps(config, indent=2))
    assert mig.main(["--config", str(path), "--write"]) == 1
    new = _load(path.read_text())
    assert new["providers"]["other"]["type"] == "chat_completions"
    assert "anthropic" in new["providers"]


def test_unparseable_and_missing_files_exit_2(tmp_path) -> None:
    bad = _write(tmp_path, "not json")
    assert mig.main(["--config", str(bad)]) == 2
    assert mig.main(["--config", str(tmp_path / "missing.jsonc")]) == 2


def test_json_report(tmp_path, capsys) -> None:
    path = _write(tmp_path)
    assert mig.main(["--config", str(path), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["exit_code"] == 0 and report["write"] is False
    assert report["configs"][0]["changed"] is True
    olds = [i["old"] for i in report["items"]]
    assert "openai_chat" in olds and "embedding_providers" in olds
    assert all(i["scope"] == "config" for i in report["items"])


# --------------------------------------------------------------------------
# The database
# --------------------------------------------------------------------------


def _make_db(path: Path) -> None:
    counters = dict(
        by_source_provider=dict(openai_chat=3, chat_completions=2, anthropic=1),
        by_target_provider=dict(anthropic=4, main=1),
    )
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE request_log (id TEXT, source_provider TEXT, "
        "target_provider TEXT, target_provider_name TEXT, model_type TEXT)"
    )
    conn.execute(
        "CREATE TABLE error_dumps (id TEXT, source_provider TEXT, target_provider TEXT)"
    )
    conn.execute("CREATE TABLE metrics (key TEXT PRIMARY KEY, value TEXT)")
    rows = [
        ("1", "openai_chat", "anthropic", "anthropic", "llm"),
        ("2", "openai_responses", "google", "google", "embedding"),
        ("3", "chat_completions", "messages", "claude", "decision"),
        ("4", "anthropic", "openai_chat", None, "llm"),
    ]
    conn.executemany("INSERT INTO request_log VALUES (?, ?, ?, ?, ?)", rows)
    conn.execute("INSERT INTO error_dumps VALUES ('e1', 'google', 'openai_responses')")
    conn.execute("INSERT INTO metrics VALUES ('counters', ?)", (json.dumps(counters),))
    conn.commit()
    conn.close()


def _dump(path: Path) -> list:
    conn = sqlite3.connect(path)
    try:
        return [
            conn.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
            for table in ("request_log", "error_dumps", "metrics")
        ]
    finally:
        conn.close()


def test_db_dry_run_counts_and_changes_nothing(tmp_path) -> None:
    db = tmp_path / "gateway.db"
    _make_db(db)
    before = db.read_bytes()
    result = mig.migrate_db(db)
    assert result.error is None and result.backup is None
    assert db.read_bytes() == before
    rewrites = [i for i in result.items if i.action == "rewrite"]
    counts = dict(((i.where, i.old), i.count) for i in rewrites)
    assert counts[("request_log.source_provider", "openai_chat")] == 1
    assert counts[("request_log.source_provider", "openai_responses")] == 1
    assert counts[("request_log.source_provider", "anthropic")] == 1
    assert counts[("request_log.target_provider", "anthropic")] == 1
    assert counts[("request_log.target_provider", "google")] == 1
    assert counts[("request_log.target_provider", "openai_chat")] == 1
    assert counts[("request_log.model_type", "embedding")] == 1
    assert counts[("request_log.model_type", "decision")] == 1
    assert counts[("error_dumps.source_provider", "google")] == 1
    assert counts[("error_dumps.target_provider", "openai_responses")] == 1
    assert counts[(mig._COUNTER_SOURCE, "openai_chat")] == 3


def test_db_write_is_one_backed_up_idempotent_rewrite(tmp_path) -> None:
    db = tmp_path / "gateway.db"
    _make_db(db)
    legacy = _dump(db)
    result = mig.migrate_db(db, write=True)
    assert result.error is None and result.backup
    assert _dump(Path(result.backup)) == legacy
    requests, dumps, metrics = _dump(db)
    assert requests == [
        ("1", "chat_completions", "messages", "anthropic", "llm"),
        ("2", "open_responses", "google_generate", "google", "embeddings"),
        ("3", "chat_completions", "messages", "claude", "decisions"),
        ("4", "messages", "chat_completions", None, "llm"),
    ]
    assert dumps == [("e1", "google_generate", "open_responses")]
    counters = json.loads(metrics[0][1])
    assert counters["by_source_provider"] == dict(chat_completions=5, messages=1)
    # by_target_provider mixes provider names and tags: left as it was.
    assert counters["by_target_provider"] == dict(anthropic=4, main=1)
    assert [i.old for i in result.items if i.action == "unmapped"] == ["anthropic"]
    # Second run: nothing pending, no second backup.
    files = sorted(p.name for p in tmp_path.iterdir())
    again = mig.migrate_db(db, write=True)
    assert again.pending == 0 and again.backup is None
    assert sorted(p.name for p in tmp_path.iterdir()) == files


def test_db_write_rolls_back_as_a_whole(tmp_path, monkeypatch) -> None:
    db = tmp_path / "gateway.db"
    _make_db(db)
    legacy = _dump(db)

    def partial(conn, counters):
        conn.execute("UPDATE request_log SET source_provider = 'x'")
        raise sqlite3.OperationalError("boom")

    monkeypatch.setattr(mig, "_apply_db", partial)
    result = mig.migrate_db(db, write=True)
    assert result.error and "boom" in result.error
    assert _dump(db) == legacy


def test_db_write_refuses_while_a_process_holds_the_file(tmp_path, monkeypatch) -> None:
    db = tmp_path / "gateway.db"
    _make_db(db)
    legacy = _dump(db)
    monkeypatch.setattr(mig, "db_holders", lambda path: [4242])
    result = mig.migrate_db(db, write=True)
    assert result.error and "4242" in result.error
    assert _dump(db) == legacy
    assert [p.name for p in tmp_path.iterdir()] == ["gateway.db"]


def test_db_write_refuses_while_another_connection_writes(tmp_path) -> None:
    db = tmp_path / "gateway.db"
    _make_db(db)
    holder = sqlite3.connect(db, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        result = mig.migrate_db(db, write=True)
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    assert result.error and "locked" in result.error


def test_db_without_the_optional_tables_or_columns_is_fine(tmp_path) -> None:
    db = tmp_path / "gateway.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE request_log (id TEXT, source_provider TEXT)")
    conn.execute("INSERT INTO request_log VALUES ('1', 'google')")
    conn.commit()
    conn.close()
    result = mig.migrate_db(db, write=True)
    assert result.error is None
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT source_provider FROM request_log").fetchall()
    conn.close()
    assert rows == [("google_generate",)]


def test_db_command_accepts_a_data_directory_and_reports_exit_codes(
    tmp_path, capsys
) -> None:
    _make_db(tmp_path / "gateway.db")
    # The unmapped by_target_provider key makes the exit code 1.
    assert mig.main(["--db", str(tmp_path), "--write", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["databases"][0]["backup"]
    assert mig.main(["--db", str(tmp_path / "nope.db")]) == 2


# --------------------------------------------------------------------------
# The source scan
# --------------------------------------------------------------------------


def _snapshot(root: Path) -> list:
    files = [p for p in sorted(root.rglob("")) if p.is_file()]
    return [(str(p), p.read_bytes()) for p in files]


def test_scan_reports_retired_names_and_edits_nothing(tmp_path, capsys) -> None:
    (tmp_path / "app.py").write_text(
        "from x import convert\n"
        'convert(body, "openai_chat", "anthropic")\n'
        "label = 'embedding'\n"
        "cfg = dict(embedding_providers=1)\n"
        'key = "embedding_providers"\n'
    )
    (tmp_path / "conf.yaml").write_text(
        "providers:\n  a:\n    type: openai_chat\n    shim: argo--anthropic  # old\n"
    )
    (tmp_path / "conf.json").write_text(json.dumps(dict(type="google-genai")))
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "skip.py").write_text('x = "openai_chat"\n')
    (tmp_path / "notes.txt").write_text('"openai_chat"\n')
    before = _snapshot(tmp_path)

    items = mig.scan_path(tmp_path)

    found = sorted((Path(i.file).name, i.line, i.old) for i in items)
    assert found == [
        ("app.py", 2, "anthropic"),
        ("app.py", 2, "openai_chat"),
        ("app.py", 5, "embedding_providers"),
        ("conf.json", 1, "google-genai"),
        ("conf.yaml", 3, "openai_chat"),
        ("conf.yaml", 4, "argo--anthropic"),
    ]
    assert _snapshot(tmp_path) == before
    assert mig.main(["--scan", str(tmp_path)]) == 1
    assert "chat_completions" in capsys.readouterr().out


def test_scan_ignores_names_that_are_already_current(tmp_path, capsys) -> None:
    """No finding for a current format, a current shim or a vendor word."""
    (tmp_path / "ok.py").write_text(
        'a = "google_generate"\n'
        'b = "google_interactions"\n'
        'c = "google--google_generate"\n'
        'd = "openai"\n'
        'e = "openai--chat_completions"\n'
        'f = convert(body, "messages", source_provider="chat_completions")\n'
        'g = dict(embeddings_providers=1, default_embeddings_format="openai")\n'
    )
    (tmp_path / "ok.yaml").write_text(
        "providers:\n  a:\n    type: google_generate\n    shim: openai--open_responses\n"
    )

    assert mig.scan_path(tmp_path) == []
    assert mig.main(["--scan", str(tmp_path)]) == 0


def test_scan_reports_old_names_beside_current_ones(tmp_path) -> None:
    (tmp_path / "mixed.py").write_text(
        'ok = "google_generate"\n'
        'old = "openai_chat"\n'
        'older = "volcengine--openai_chat"\n'
        'vendor = "openai"\n'
        'kind = "google"\n'
    )

    found = sorted((i.line, i.old, i.new) for i in mig.scan_path(tmp_path))

    assert found == [
        (2, "openai_chat", "chat_completions (provider_type)"),
        (3, "volcengine--openai_chat", "volcengine--chat_completions (shim)"),
        (5, "google", "google_generate (provider_type)"),
    ]


def test_scan_of_a_clean_tree_exits_0(tmp_path) -> None:
    (tmp_path / "ok.py").write_text('fmt = "chat_completions"\n')
    assert mig.main(["--scan", str(tmp_path)]) == 0


def test_the_scan_vocabulary_is_the_table_plus_derived_keys() -> None:
    vocab = mig._scan_vocabulary()
    bare_shims = set(["openai", "google", "anthropic"])
    current = set(["google_generate", "google_interactions"])
    for row in ROWS:
        if row.kind == retired_names.MODEL_TYPE:
            continue
        left_out = row.name in current or (
            row.kind == retired_names.SHIM and row.name in bare_shims
        )
        notes = vocab.get(row.name, [])
        listed = any(note.endswith(f"({row.kind})") for note in notes)
        assert listed is not left_out, row
    assert "embedding" not in vocab and "decision" not in vocab
    assert vocab["default_decision_format"] == ["default_decisions_format"]
    assert vocab["google"] == ["google_generate (provider_type)"]
    assert "google_generate" not in vocab and "google_interactions" not in vocab
    assert "openai" not in vocab


# --------------------------------------------------------------------------
# The provider tag ``embedding`` (a typed request was logged under its type)
# --------------------------------------------------------------------------


def _make_embedding_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE request_log (id TEXT, source_provider TEXT, "
        "target_provider TEXT, model_type TEXT, total_tokens INTEGER)"
    )
    conn.execute(
        "CREATE TABLE error_dumps (id TEXT, source_provider TEXT, target_provider TEXT)"
    )
    conn.execute("CREATE TABLE metrics (key TEXT PRIMARY KEY, value TEXT)")
    conn.executemany(
        "INSERT INTO request_log VALUES (?, ?, ?, ?, ?)",
        [
            ("1", "embedding", "embedding", "embeddings", 7),
            ("2", "embedding", "openai", "embeddings", 11),
            ("3", "chat_completions", "embedding", "embeddings", 13),
            ("4", "chat_completions", "messages", "llm", 17),
        ],
    )
    conn.execute("INSERT INTO error_dumps VALUES ('e1', 'embedding', 'embedding')")
    counters = dict(
        by_source_provider=dict(embedding=5, embeddings=2, chat_completions=1),
        by_target_provider=dict(embedding=4, main=1),
    )
    conn.execute("INSERT INTO metrics VALUES ('counters', ?)", (json.dumps(counters),))
    conn.commit()
    conn.close()


def _provider_columns(path: Path) -> list:
    conn = sqlite3.connect(path)
    try:
        return conn.execute(
            "SELECT id, source_provider, target_provider, model_type, total_tokens "
            "FROM request_log ORDER BY id"
        ).fetchall()
    finally:
        conn.close()


def test_db_dry_run_reports_the_embedding_provider_tag(tmp_path) -> None:
    db = tmp_path / "gateway.db"
    _make_embedding_db(db)
    before = db.read_bytes()

    result = mig.migrate_db(db)

    assert db.read_bytes() == before
    counts = dict(
        ((i.where, i.old), (i.new, i.count))
        for i in result.items
        if i.action == "rewrite"
    )
    assert counts[("request_log.source_provider", "embedding")] == ("embeddings", 2)
    assert counts[("request_log.target_provider", "embedding")] == ("embeddings", 2)
    assert counts[("error_dumps.source_provider", "embedding")] == ("embeddings", 1)
    assert counts[("error_dumps.target_provider", "embedding")] == ("embeddings", 1)
    assert counts[(mig._COUNTER_SOURCE, "embedding")] == ("embeddings", 5)
    unmapped = [(i.where, i.old) for i in result.items if i.action == "unmapped"]
    assert unmapped == [(mig._COUNTER_TARGET, "embedding")]


def test_db_write_rewrites_the_embedding_provider_tag_and_nothing_else(
    tmp_path,
) -> None:
    db = tmp_path / "gateway.db"
    _make_embedding_db(db)

    result = mig.migrate_db(db, write=True)

    assert result.error is None and result.backup
    assert _provider_columns(db) == [
        ("1", "embeddings", "embeddings", "embeddings", 7),
        ("2", "embeddings", "openai", "embeddings", 11),
        ("3", "chat_completions", "embeddings", "embeddings", 13),
        ("4", "chat_completions", "messages", "llm", 17),
    ]
    conn = sqlite3.connect(db)
    try:
        dumps = conn.execute("SELECT * FROM error_dumps").fetchall()
        counters = json.loads(
            conn.execute("SELECT value FROM metrics WHERE key = 'counters'").fetchone()[
                0
            ]
        )
    finally:
        conn.close()
    assert dumps == [("e1", "embeddings", "embeddings")]
    assert counters["by_source_provider"] == dict(embeddings=7, chat_completions=1)
    # The target counter mixes provider names and tags, so it is only reported.
    assert counters["by_target_provider"] == dict(embedding=4, main=1)


def test_db_embedding_rewrite_is_idempotent(tmp_path) -> None:
    db = tmp_path / "gateway.db"
    _make_embedding_db(db)
    mig.migrate_db(db, write=True)
    after_first = _dump(db)
    files = sorted(p.name for p in tmp_path.iterdir())

    again = mig.migrate_db(db, write=True)

    assert again.error is None and again.backup is None
    assert [i for i in again.items if i.action == "rewrite"] == []
    assert _dump(db) == after_first
    assert sorted(p.name for p in tmp_path.iterdir()) == files
