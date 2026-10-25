"""Tests for ``llm-rosetta-usage``.

The CLI's claim is narrow and worth holding to: it reads one store through the
same code the gateway does and prints the same report, so what it prints can be
compared with what the library computes in-process.  These tests run it as a
real subprocess — a console script is a promise about an installed program, and
calling ``main()`` in-process would test a function instead of the promise —
and compare its output with a report built here from the same store.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from llm_rosetta_gateway.observability import PersistenceManager
from llm_rosetta.observability.usage import (
    INTERNAL_LABEL,
    INTERNAL_SUBJECT,
    UsageReport,
)

_K1 = "credential-one"
_K2 = "credential-two"


def _ago(**kwargs) -> str:
    return (datetime.now(timezone.utc) - timedelta(**kwargs)).isoformat()


def _row(
    bucket: str | None,
    label: str | None,
    total: int,
    *,
    model: str = "gpt-5",
    model_type: str = "llm",
    when: str | None = None,
    inp: int | None = None,
    cache_read: int = 0,
) -> dict[str, object]:
    """One request log row, minimal but shaped like the real thing."""
    timestamp = when or _ago(minutes=5)
    return {
        "id": f"{bucket}-{label}-{model}-{timestamp}",
        "timestamp": timestamp,
        "model": model,
        "model_type": model_type,
        "source_provider": "openai",
        "target_provider": "anthropic",
        "is_stream": False,
        "status_code": 200,
        "duration_ms": 1.0,
        "api_key_bucket": bucket,
        "api_key_label": label,
        "input_tokens": total if inp is None else inp,
        "output_tokens": 0 if inp is None else total - inp,
        "total_tokens": total,
        "cache_read_tokens": cache_read,
    }


@pytest.fixture()
async def store(tmp_path: Path) -> Path:
    """A data directory with a small, mixed store and its keys database.

    The traffic covers the classes the report separates — two live
    credentials, the gateway's own token, a retired credential, an anonymous
    request — and a row old enough to fall out of the 24h window, so the
    window a caller asks for is observable in the output.
    """
    data_dir = tmp_path / "data"
    manager = await PersistenceManager.create(str(data_dir))
    await manager.insert_log_entries(
        [
            _row(_K1, "team-a", 1000, inp=600, cache_read=200),
            _row(_K1, "team-a", 1000, inp=600, when=_ago(hours=30)),
            _row(_K2, "team-b", 300, model="text-embedding-3", model_type="embedding"),
            _row("retired-bucket", "retired", 70, inp=50, when=_ago(hours=3)),
            _row(None, None, 20, when=_ago(hours=4)),
            _row(INTERNAL_SUBJECT, INTERNAL_LABEL, 11, model="gpt-5-nano"),
        ]
    )
    await manager.close()

    conn = sqlite3.connect(data_dir / "keys.db")
    conn.execute(
        "CREATE TABLE api_keys (id TEXT PRIMARY KEY, label TEXT, usage_bucket TEXT)"
    )
    conn.executemany(
        "INSERT INTO api_keys VALUES (?,?,?)",
        [
            ("key-1", "team-a", _K1),
            ("key-2", "team-b", _K2),
            # A key whose bucket was never backfilled cannot be attributed;
            # it must not appear as a subject that spent nothing.
            ("key-3", "unbackfilled", None),
        ],
    )
    conn.commit()
    conn.close()
    return data_dir


def _run(store: Path | str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the CLI the way a shell would, from the source tree."""
    root = Path(__file__).resolve().parents[3]
    src = os.pathsep.join(
        str(root / "packages" / pkg / "src")
        for pkg in ("llm-rosetta", "llm-rosetta-gateway")
    )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "llm_rosetta_gateway.observability.usage.cli",
            "--db",
            str(store),
            *args,
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": src},
    )


def _json(store: Path, *args: str) -> dict:
    result = _run(store, *args, "--json")
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _totals(text: str) -> list[str]:
    """The last number on every TOTAL row of the printed tables."""
    return [line.split()[-1] for line in text.splitlines() if line.startswith("TOTAL")]


def _body_rows(text: str) -> list[str]:
    """The data rows of one table: no headings, rules, notes or totals."""
    return [
        line
        for line in text.splitlines()
        if line
        and not line.startswith(
            ("store:", "window:", "subject", "model", "---", "TOTAL", "…")
        )
    ]


def _schema_version(store: Path) -> int:
    """The store's recorded schema version, read as data."""
    conn = sqlite3.connect(store / "gateway.db")
    try:
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    finally:
        conn.close()
    return row[0] if row and row[0] is not None else 0


def _settle(store: Path) -> None:
    """Fold the write-ahead log back into the store and close it.

    A read-only SQLite connection cannot attach a pending ``-wal`` — attaching
    needs the shared-memory file, and creating that needs a writable
    directory — so a test that compares read-only against read-write is
    comparing the same state.  The connection is closed explicitly: an open one
    keeps the sidecars alive, whichever block it was opened in.
    """
    conn = sqlite3.connect(store / "gateway.db")
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.commit()
    finally:
        conn.close()


def _make_stale(store: Path) -> None:
    """Leave the store one migration behind, as an older release would."""
    conn = sqlite3.connect(store / "gateway.db")
    try:
        conn.execute(
            "DELETE FROM schema_version WHERE version = ?",
            (PersistenceManager.expected_schema_version(),),
        )
        conn.commit()
    finally:
        conn.close()
    _settle(store)


class TestItPrintsTheReportTheLibraryBuilds:
    async def test_the_json_subject_cut_is_the_admin_payload(self, store):
        """The point of the command: one report, two readers.

        Compared key by key against the payload the admin API returns, so a
        script reading the CLI and one reading the panel see one shape.
        """
        document = _json(store, "--by", "subject")
        manager = await PersistenceManager.create(str(store), backup=False)
        try:
            groups = (await manager.query_usage_groups())["groups"]
        finally:
            await manager.close()
        payload = UsageReport.build(
            groups=groups,
            subjects=[(_K1, "team-a"), (_K2, "team-b")],
            internal_subject=INTERNAL_SUBJECT,
            internal_label=INTERNAL_LABEL,
        ).to_payload({_K1: "key-1", _K2: "key-2"})
        assert {key: document[key] for key in payload} == payload

    async def test_every_class_adds_up_to_the_stores_total(self, store):
        """The reconciliation the report exists for, over the CLI's own text."""
        result = _run(store, "--by", "subject,type,model")
        assert result.returncode == 0, result.stderr
        totals = _totals(result.stdout)
        assert len(totals) == 3, result.stdout
        assert len(set(totals)) == 1, result.stdout
        # 1000 + 300 + 70 + 20 + 11 inside 24h.
        assert totals[0] == "1,401"

        week = _run(store, "--by", "subject,type,model", "--window", "7d")
        assert set(_totals(week.stdout)) == {"2,401"}, (
            "the 7d window carries the 30h row"
        )

    async def test_the_model_cut_keeps_its_own_type(self, store):
        """An embedding call is not chat traffic, and says so."""
        document = _json(store, "--by", "model")
        models = {group["model"]: group for group in document["models"]}
        assert models["text-embedding-3"]["model_type"] == "embedding"
        assert models["gpt-5"]["model_type"] == "llm"
        # Ordered by the widest window's total, so the busiest model leads.
        assert document["models"][0]["model"] == "gpt-5"

    async def test_a_row_with_no_traffic_in_the_window_is_not_shown_first(
        self, tmp_path
    ):
        """Every cut is ordered by the window it shows (review on #1031).

        `beta-model` has all its traffic ten days back, `alpha-model` all of it
        inside the day.  Ordering by the widest window put beta at the top of
        the 24h table — a row of zeros above the row with the traffic — and
        `--top 1` then hid the row that mattered.  The widest window is the
        tiebreaker now, so it still decides among rows with nothing to show.
        """
        data_dir = tmp_path / "ranked"
        manager = await PersistenceManager.create(str(data_dir))
        await manager.insert_log_entries(
            [
                _row("old-bucket", "old", 4000, model="beta-model", when=_ago(days=10)),
                _row("new-bucket", "new", 1000, model="alpha-model"),
            ]
        )
        await manager.close()

        rows = _body_rows(_run(data_dir, "--by", "model").stdout)
        assert rows[0].startswith("alpha-model"), rows

        # And with the one-row cut, the row that has the traffic is the row
        # that survives it.
        cut = _run(data_dir, "--by", "model", "--top", "1").stdout
        assert _body_rows(cut)[0].startswith("alpha-model"), cut
        assert "and 1 more" in cut

        # In the month, beta leads — the tiebreaker is a ranking of its own.
        month = _run(data_dir, "--by", "model", "--window", "30d").stdout
        assert _body_rows(month)[0].startswith("beta-model"), month

    async def test_the_type_cut_covers_every_row_the_subject_cut_covers(self, store):
        document = _json(store, "--by", "subject,type")
        per_type = sum(
            windows["24h"]["total_tokens"]
            for windows in document["model_types"].values()
        )
        subjects = document["keys"]
        per_class = (
            sum(entry["windows"]["24h"]["total_tokens"] for entry in subjects.values())
            + document["internal"]["windows"]["24h"]["total_tokens"]
            + document["unattributed"]["windows"]["24h"]["total_tokens"]
        )
        assert per_type == per_class


class TestItsEdges:
    async def test_a_missing_store_is_refused_not_created(self, tmp_path):
        target = tmp_path / "nowhere"
        result = _run(target)
        assert result.returncode == 1
        assert "no store at" in result.stderr
        assert not target.exists(), "a lookup must not create the store it missed"

    async def test_a_store_that_saw_nothing_prints_zeros(self, tmp_path):
        """A gateway that has just started is a store with no rows.

        Every cut has to render an empty report — the model table has no rows
        at all to show — and the totals have to be zeros rather than a
        traceback.
        """
        data_dir = tmp_path / "fresh"
        manager = await PersistenceManager.create(str(data_dir))
        await manager.close()

        result = _run(data_dir, "--by", "subject,type,model")
        assert result.returncode == 0, result.stderr
        assert set(_totals(result.stdout)) == {"0"}, result.stdout

    async def test_a_data_directory_and_its_store_file_read_the_same(self, store):
        """--db takes either, because operators have both in their history."""
        assert _json(store, "--by", "subject") == _json(
            store / "gateway.db", "--by", "subject"
        )

    async def test_a_store_without_a_keys_database_says_so(self, store):
        """Attribution is impossible without the credentials, and that is said.

        The totals stay the store's — a report is not wrong for being unable to
        name who spent the tokens, but a reader must not take the silence for
        "nobody did".
        """
        (store / "keys.db").unlink()
        result = _run(store, "--by", "subject,type", "--json")
        assert result.returncode == 0, result.stderr
        assert "no keys database" in result.stderr
        document = json.loads(result.stdout)
        assert document["keys"] == {}
        # Nothing is attributed, but nothing is lost either: the gateway's own
        # traffic is still its own class, and the two together are the store.
        day = document["unattributed"]["windows"]["24h"]["total_tokens"]
        gateway = document["internal"]["windows"]["24h"]["total_tokens"]
        assert (day, gateway) == (1_390, 11)
        assert day + gateway == 1_401

    async def test_reading_leaves_no_backup_behind(self, store):
        """A lookup should not leave a copy of the database in the store."""
        _run(store)
        _run(store)
        assert not (store / "backups").exists()

    async def test_a_store_behind_this_build_is_refused_not_migrated(self, store):
        """The default path is read-only, so a stale store is not written to.

        Migrating it here would take a write lock on a store another process is
        serving, and would apply someone else's release notes to it; the report
        refuses and names the two ways out instead (review on #1031).
        """
        _make_stale(store)
        result = _run(store, "--by", "subject")
        assert result.returncode == 1
        behind = PersistenceManager.expected_schema_version() - 1
        assert f"schema version {behind}" in result.stderr, result.stderr
        assert "--migrate" in result.stderr
        # Nothing was written: the store is still behind and has no backup.
        assert _schema_version(store) == behind
        assert not (store / "backups").exists()

    async def test_a_store_whose_wal_cannot_be_attached_is_not_called_stale(
        self, store
    ):
        """The diagnosis is the read, not the schema (review on #1031).

        A crashed gateway leaves a `-wal` with frames in it.  A read-only
        reader cannot create the missing `-shm` (here the directory is
        read-only, as a mounted snapshot's would be), so SQLite shows it an
        empty database and the schema reads as nothing.  Reporting that as
        "schema version 0, this build expects 6" sends the reader to migrate a
        store that is already current — to a migration that fails on the same
        permission.

        Built by killing a writer mid-flight, which is how a pending `-wal`
        actually happens, rather than by fabricating one.
        """
        if os.geteuid() == 0:
            pytest.skip("root writes through the mode bits this test needs")
        _settle(store)
        writer = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import sqlite3;"
                f"c = sqlite3.connect({str(store / 'gateway.db')!r});"
                "c.execute('PRAGMA wal_autocheckpoint=0');"
                "c.execute('UPDATE request_log SET status_code = status_code');"
                "c.commit();"
                "print('written', flush=True);"
                "input()",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        assert writer.stdout is not None
        assert writer.stdout.readline().strip() == "written"
        writer.kill()  # no clean shutdown, so the `-wal` stays pending
        writer.wait()
        (store / "gateway.db-shm").unlink(missing_ok=True)
        # The state the test is about has to be the state it runs against.
        assert (store / "gateway.db-wal").stat().st_size > 0
        assert not (store / "gateway.db-shm").exists()

        store.chmod(0o555)
        try:
            result = _run(store, "--by", "subject")
        finally:
            store.chmod(0o755)

        assert result.returncode == 1, result.stdout
        assert "could not read the schema" in result.stderr, result.stderr
        assert "version 0" not in result.stderr, result.stderr
        assert "--migrate" not in result.stderr, result.stderr

    async def test_migrate_opens_the_store_the_way_the_gateway_would(self, store):
        """The one write is opt-in, and it comes with the gateway's backup."""
        _make_stale(store)
        result = _run(store, "--by", "subject", "--migrate")
        assert result.returncode == 0, result.stderr
        assert _schema_version(store) == PersistenceManager.expected_schema_version()
        assert (store / "backups").is_dir(), "an explicit migration takes a backup"

    async def test_a_store_we_cannot_write_reads_while_it_is_being_served(self, store):
        """A mounted volume or a snapshot copy, measured (review on #1031).

        The state that reads is the deployment's: another process (the
        gateway) is already attached, so the `-shm` exists and a read-only
        connection can attach to it — the directory does not have to be
        writable for that.  Behind this build, the same store is refused for
        the reason that is true (the schema), not for the reason the write
        would have given ("attempt to write a readonly database", which blames
        the read), and `--migrate`'s own failure is readable rather than a
        traceback.
        """
        if os.geteuid() == 0:
            pytest.skip("root writes through the mode bits this test needs")
        _settle(store)

        def mode(readonly: bool) -> None:
            """Set the tree's modes, re-globbing each time.

            SQLite's ``-wal``/``-shm`` sidecars appear and vanish with the
            connections that touch the store, so a list captured once goes
            stale between calls; a path that has gone is one there is nothing
            left to lock.
            """
            for path in [store, *store.rglob("*")]:
                try:
                    if path.is_dir():
                        path.chmod(0o555 if readonly else 0o755)
                    else:
                        path.chmod(0o444 if readonly else 0o644)
                except FileNotFoundError:
                    continue

        # A process is serving the store: it holds the sidecars open, which is
        # why the modes below do not stop the reader.
        serving = sqlite3.connect(store / "gateway.db")
        serving.execute("PRAGMA wal_autocheckpoint=0")
        serving.execute("UPDATE request_log SET status_code = status_code")
        serving.commit()
        try:
            mode(readonly=True)
            try:
                current = _run(store, "--by", "subject")
            finally:
                mode(readonly=False)
            assert current.returncode == 0, current.stderr

            _make_stale(store)
            mode(readonly=True)
            try:
                stale = _run(store, "--by", "subject")
                migrating = _run(store, "--by", "subject", "--migrate")
            finally:
                mode(readonly=False)
        finally:
            serving.close()

        assert stale.returncode == 1
        behind = PersistenceManager.expected_schema_version() - 1
        assert f"schema version {behind}" in stale.stderr, stale.stderr
        assert "readonly database" not in stale.stderr, stale.stderr
        assert migrating.returncode == 1
        assert "could not migrate the store" in migrating.stderr, migrating.stderr
        assert "Traceback" not in migrating.stderr

    async def test_json_carries_every_model_but_the_table_is_what_is_cut(self, store):
        """`--top 1 --json` is the whole report, as `--help` promises."""
        document = _json(store, "--by", "model", "--top", "1")
        assert len(document["models"]) == 3

    async def test_an_unknown_cut_is_a_usage_error(self, store):
        result = _run(store, "--by", "subject,weather")
        assert result.returncode == 2
        assert "--by takes" in result.stderr

    async def test_top_truncates_the_table_but_not_the_total(self, store):
        """Truncation is a property of the screen, never of the arithmetic."""
        result = _run(store, "--by", "model", "--top", "1")
        assert result.returncode == 0, result.stderr
        # Three models in the store, the busiest one shown, the note where the
        # two it hides would have been.
        assert "and 2 more" in result.stdout, result.stdout
        assert _totals(result.stdout) == ["1,401"]

    async def test_top_reaches_every_table_not_only_the_model_cut(self, store):
        """The cut that grows with the deployment is the one that needs it.

        `--top` used to bound the model cut only — the one bounded by the
        provider list — while the subject table, which grows with every
        credential the gateway has ever seen, printed every row and ignored the
        flag (review on #1031, from elena and milo independently).
        """
        # Four classes in the fixture (two keys, internal, unattributed): one
        # row shown, three disclosed, and a total that still spans all four.
        subject = _run(store, "--by", "subject", "--top", "1")
        assert subject.returncode == 0, subject.stderr
        assert "and 3 more" in subject.stdout, subject.stdout
        assert len(_body_rows(subject.stdout)) == 1, subject.stdout
        assert _totals(subject.stdout) == ["1,401"], subject.stdout

        # Two model types, and the same rule.
        by_type = _run(store, "--by", "type", "--top", "1")
        assert "and 1 more" in by_type.stdout, by_type.stdout
        assert _totals(by_type.stdout) == ["1,401"], by_type.stdout

    async def test_a_subject_cut_does_not_read_the_model_aggregate(
        self, store, monkeypatch
    ):
        """A cut nobody asked for is not read (elena's nit on #1031).

        Asserted on ``_collect`` rather than through ``main``: ``main`` owns an
        ``asyncio.run``, which cannot be called from inside this test's own
        event loop.  The gate itself is what the finding was about.
        """
        from llm_rosetta_gateway.observability import persistence as persistence_module
        from llm_rosetta_gateway.observability.usage import cli

        async def _unwanted(self):
            raise AssertionError("the model aggregate was read unasked")

        monkeypatch.setattr(
            persistence_module.PersistenceManager, "query_usage_by_model", _unwanted
        )
        _report, _names, models, _note = await cli._collect(
            store, store / "keys.db", migrate=False, by_model=False
        )
        assert models == []
        # The gate is a gate, not a removal: the cut that asked for it reads it.
        with pytest.raises(AssertionError):
            await cli._collect(store, store / "keys.db", migrate=False, by_model=True)

    async def test_the_window_selects_what_is_shown(self, store):
        """A 30-hour-old row is in the week and not in the day."""
        day = _json(store, "--window", "24h")
        week = _json(store, "--window", "7d")
        assert day["window"] == "24h"
        assert week["window"] == "7d"
        assert day["unattributed"]["windows"]["24h"]["request_count"] == 2
        # The window flag moves the tables, not the report: both documents
        # carry every rolling window.
        assert day["windows"] == week["windows"]


class TestItIsWiredUp:
    def test_the_console_script_points_at_main(self):
        """The entry point is what a shell runs; a rename here breaks it."""
        pyproject = (
            Path(__file__).resolve().parents[3]
            / "packages"
            / "llm-rosetta-gateway"
            / "pyproject.toml"
        )
        text = pyproject.read_text()
        assert (
            'llm-rosetta-usage = "llm_rosetta_gateway.observability.usage.cli:main"'
            in text
        ), "the console script must keep pointing at the module's main()"
