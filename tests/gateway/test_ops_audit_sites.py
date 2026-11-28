"""The lifecycle audit rows are written through the ops layer (#1830 A7).

The first group pins the rows the gateway wrote before the change, through the
two functions that can be driven with a stand-in app; it passes on either side
of the change.  The last test is the guard: no gateway module outside ``ops/``
and ``observability/`` calls ``ops_log.add`` itself.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from llm_rosetta_gateway.app.factory import _record_startup_entries
from llm_rosetta_gateway.app.tasks import _flush_now


class _Log:
    """An ops log that remembers each row and how it was added."""

    def __init__(self) -> None:
        self.rows: list[tuple[Any, bool]] = []

    async def add(self, entry: Any, *, _skip_prune: bool = False) -> None:
        self.rows.append((entry, _skip_prune))


class _Persistence:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _startup_app(config: Any) -> Any:
    return SimpleNamespace(ops_log=_Log(), gateway_config=config)


def test_the_startup_row_carries_the_address_and_the_counts() -> None:
    config = SimpleNamespace(
        providers={"a": 1, "b": 2},
        models={"m": 1},
        provider_validation_result=None,
    )
    app = _startup_app(config)
    asyncio.run(_record_startup_entries(app, "127.0.0.1", 8080))
    [(entry, skip)] = app.ops_log.rows
    assert skip is False
    assert entry.event_type == "startup"
    assert entry.severity == "info"
    assert entry.source == "gateway"
    assert entry.message == "Gateway starting on 127.0.0.1:8080"
    assert entry.details == {
        "host": "127.0.0.1",
        "port": 8080,
        "provider_count": 2,
        "model_count": 1,
    }


def test_the_startup_row_counts_zero_without_a_config() -> None:
    app = _startup_app(None)
    asyncio.run(_record_startup_entries(app, "h", 1))
    [(entry, _)] = app.ops_log.rows
    assert entry.details["provider_count"] == 0
    assert entry.details["model_count"] == 0


def test_the_validation_row_comes_first_and_is_a_warning() -> None:
    config = SimpleNamespace(
        providers={},
        models={},
        provider_validation_result={
            "disabled": [{"provider": "argo", "reason": "r", "category": "c"}],
            "recovered": [],
        },
    )
    app = _startup_app(config)
    asyncio.run(_record_startup_entries(app, "h", 1))
    assert [row.event_type for row, _ in app.ops_log.rows] == [
        "provider_validation",
        "startup",
    ]
    assert app.ops_log.rows[0][0].severity == "warning"
    assert app.ops_log.rows[0][0].source == "config"


def test_the_shutdown_row_skips_pruning_and_precedes_the_close() -> None:
    persistence = _Persistence()

    class OrderedLog(_Log):
        def __init__(self) -> None:
            super().__init__()
            self.closed_when_added: list[bool] = []

        async def add(self, entry: Any, *, _skip_prune: bool = False) -> None:
            self.closed_when_added.append(persistence.closed)
            await super().add(entry, _skip_prune=_skip_prune)

    log = OrderedLog()
    app: Any = SimpleNamespace(persistence=persistence, ops_log=log)
    asyncio.run(_flush_now(app))
    [(entry, skip)] = log.rows
    assert skip is True
    assert log.closed_when_added == [False]
    assert persistence.closed
    assert entry.event_type == "shutdown"
    assert entry.severity == "info"
    assert entry.source == "gateway"
    assert entry.message == "Gateway shutting down"
    assert not entry.details


def test_no_gateway_module_outside_ops_writes_the_ops_log_directly() -> None:
    root = Path(__file__).parents[2] / "packages/llm-rosetta-gateway/src"
    allowed = {"ops", "observability"}
    hits: list[str] = []
    for path in sorted((root / "llm_rosetta_gateway").rglob("*.py")):
        parts = path.relative_to(root / "llm_rosetta_gateway").parts
        if parts[0] in allowed or "_vendor" in parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "ops_log"
            ):
                hits.append(f"{'/'.join(parts)}:{node.lineno}")
    assert hits == []
