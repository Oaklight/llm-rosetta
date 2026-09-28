"""Tests for the observability OpsLog (standalone, no gateway)."""

import pytest
import pytest_asyncio

from llm_rosetta.observability import (
    OpsLog,
    OpsLogEntry,
    PersistenceManager,
)
from llm_rosetta.observability.ops_log import (
    ALL_EVENT_TYPES,
    ALL_SEVERITIES,
    ALL_SOURCES,
    EVENT_CONFIG_RELOAD,
    EVENT_KEY_CREATE,
    EVENT_OPS_LOG_CLEARED,
    EVENT_STARTUP,
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    SOURCE_ADMIN,
    SOURCE_GATEWAY,
)


class TestOpsLogEntry:
    def test_create(self):
        e = OpsLogEntry.create(
            event_type=EVENT_STARTUP,
            severity=SEVERITY_INFO,
            message="Gateway started",
        )
        assert e.event_type == EVENT_STARTUP
        assert e.severity == SEVERITY_INFO
        assert e.message == "Gateway started"
        assert e.id
        assert e.timestamp
        assert e.details is None
        assert e.source is None

    def test_to_dict_minimal(self):
        e = OpsLogEntry.create(
            event_type=EVENT_STARTUP,
            severity=SEVERITY_INFO,
            message="Started",
        )
        d = e.to_dict()
        assert d["event_type"] == EVENT_STARTUP
        assert "details" not in d
        assert "source" not in d

    def test_to_dict_with_optionals(self):
        e = OpsLogEntry.create(
            event_type=EVENT_CONFIG_RELOAD,
            severity=SEVERITY_INFO,
            message="Config reloaded",
            details={"provider_count": 3, "model_count": 10},
            source=SOURCE_ADMIN,
        )
        d = e.to_dict()
        assert d["details"]["provider_count"] == 3
        assert d["source"] == SOURCE_ADMIN

    def test_frozen(self):
        e = OpsLogEntry.create(
            event_type=EVENT_STARTUP,
            severity=SEVERITY_INFO,
            message="Started",
        )
        with pytest.raises(AttributeError):
            e.message = "changed"  # type: ignore[misc]  # ty: ignore[invalid-assignment]


class TestOpsLogInMemory:
    @pytest.mark.asyncio
    async def test_add_and_get(self):
        log = OpsLog(max_entries=10)
        entry = OpsLogEntry.create(
            event_type=EVENT_STARTUP,
            severity=SEVERITY_INFO,
            message="Started",
        )
        await log.add(entry)
        entries, total = await log.get_entries(limit=10)
        assert total == 1
        assert entries[0]["event_type"] == EVENT_STARTUP

    @pytest.mark.asyncio
    async def test_newest_first(self):
        log = OpsLog(max_entries=10)
        for i in range(3):
            await log.add(
                OpsLogEntry.create(
                    event_type=EVENT_STARTUP,
                    severity=SEVERITY_INFO,
                    message=f"Event {i}",
                )
            )
        entries, _ = await log.get_entries(limit=10)
        assert entries[0]["message"] == "Event 2"
        assert entries[2]["message"] == "Event 0"

    @pytest.mark.asyncio
    async def test_filter_by_event_type(self):
        log = OpsLog(max_entries=10)
        await log.add(
            OpsLogEntry.create(
                event_type=EVENT_STARTUP,
                severity=SEVERITY_INFO,
                message="Started",
            )
        )
        await log.add(
            OpsLogEntry.create(
                event_type=EVENT_CONFIG_RELOAD,
                severity=SEVERITY_INFO,
                message="Reloaded",
            )
        )
        entries, total = await log.get_entries(event_type=EVENT_STARTUP)
        assert total == 1
        assert entries[0]["event_type"] == EVENT_STARTUP

    @pytest.mark.asyncio
    async def test_filter_by_severity(self):
        log = OpsLog(max_entries=10)
        await log.add(
            OpsLogEntry.create(
                event_type=EVENT_STARTUP,
                severity=SEVERITY_INFO,
                message="OK",
            )
        )
        await log.add(
            OpsLogEntry.create(
                event_type=EVENT_STARTUP,
                severity=SEVERITY_ERROR,
                message="Bad",
            )
        )
        entries, total = await log.get_entries(severity=SEVERITY_ERROR)
        assert total == 1
        assert entries[0]["message"] == "Bad"

    @pytest.mark.asyncio
    async def test_filter_by_source(self):
        log = OpsLog(max_entries=10)
        await log.add(
            OpsLogEntry.create(
                event_type=EVENT_STARTUP,
                severity=SEVERITY_INFO,
                message="From gateway",
                source=SOURCE_GATEWAY,
            )
        )
        await log.add(
            OpsLogEntry.create(
                event_type=EVENT_KEY_CREATE,
                severity=SEVERITY_INFO,
                message="From admin",
                source=SOURCE_ADMIN,
            )
        )
        entries, total = await log.get_entries(source=SOURCE_GATEWAY)
        assert total == 1
        assert entries[0]["source"] == SOURCE_GATEWAY

    @pytest.mark.asyncio
    async def test_pagination(self):
        log = OpsLog(max_entries=20)
        for i in range(10):
            await log.add(
                OpsLogEntry.create(
                    event_type=EVENT_STARTUP,
                    severity=SEVERITY_INFO,
                    message=f"Event {i}",
                )
            )
        entries, total = await log.get_entries(limit=3, offset=0)
        assert total == 10
        assert len(entries) == 3

        entries2, _ = await log.get_entries(limit=3, offset=3)
        assert len(entries2) == 3
        assert entries[0]["id"] != entries2[0]["id"]

    @pytest.mark.asyncio
    async def test_clear_returns_count(self):
        log = OpsLog(max_entries=10)
        for _ in range(5):
            await log.add(
                OpsLogEntry.create(
                    event_type=EVENT_STARTUP,
                    severity=SEVERITY_INFO,
                    message="Event",
                )
            )
        count = await log.clear()
        assert count == 5

    @pytest.mark.asyncio
    async def test_deque_max_entries(self):
        log = OpsLog(max_entries=3)
        for i in range(5):
            await log.add(
                OpsLogEntry.create(
                    event_type=EVENT_STARTUP,
                    severity=SEVERITY_INFO,
                    message=f"Event {i}",
                )
            )
        # In-memory deque limits to 3
        assert len(log) == 3


class TestOpsLogPersistence:
    @pytest_asyncio.fixture()
    async def pm(self, tmp_path):
        return await PersistenceManager.create(str(tmp_path), success_max=100)

    @pytest.mark.asyncio
    async def test_add_and_query(self, pm):
        log = OpsLog(persistence=pm)
        await log.add(
            OpsLogEntry.create(
                event_type=EVENT_STARTUP,
                severity=SEVERITY_INFO,
                message="Started",
                details={"host": "0.0.0.0", "port": 8080},
                source=SOURCE_GATEWAY,
            )
        )
        entries, total = await log.get_entries(limit=10)
        assert total == 1
        assert entries[0]["event_type"] == EVENT_STARTUP
        assert entries[0]["details"]["host"] == "0.0.0.0"
        assert entries[0]["source"] == SOURCE_GATEWAY

    @pytest.mark.asyncio
    async def test_filter_event_type(self, pm):
        log = OpsLog(persistence=pm)
        await log.add(
            OpsLogEntry.create(
                event_type=EVENT_STARTUP,
                severity=SEVERITY_INFO,
                message="Started",
            )
        )
        await log.add(
            OpsLogEntry.create(
                event_type=EVENT_CONFIG_RELOAD,
                severity=SEVERITY_INFO,
                message="Reloaded",
            )
        )
        entries, total = await log.get_entries(event_type=EVENT_CONFIG_RELOAD)
        assert total == 1
        assert entries[0]["message"] == "Reloaded"

    @pytest.mark.asyncio
    async def test_filter_severity(self, pm):
        log = OpsLog(persistence=pm)
        await log.add(
            OpsLogEntry.create(
                event_type=EVENT_STARTUP,
                severity=SEVERITY_INFO,
                message="OK",
            )
        )
        await log.add(
            OpsLogEntry.create(
                event_type=EVENT_STARTUP,
                severity=SEVERITY_WARNING,
                message="Warn",
            )
        )
        entries, total = await log.get_entries(severity=SEVERITY_WARNING)
        assert total == 1

    @pytest.mark.asyncio
    async def test_filter_source(self, pm):
        log = OpsLog(persistence=pm)
        await log.add(
            OpsLogEntry.create(
                event_type=EVENT_STARTUP,
                severity=SEVERITY_INFO,
                message="From gateway",
                source=SOURCE_GATEWAY,
            )
        )
        await log.add(
            OpsLogEntry.create(
                event_type=EVENT_KEY_CREATE,
                severity=SEVERITY_INFO,
                message="From admin",
                source=SOURCE_ADMIN,
            )
        )
        entries, total = await log.get_entries(source=SOURCE_ADMIN)
        assert total == 1
        assert entries[0]["source"] == SOURCE_ADMIN

    @pytest.mark.asyncio
    async def test_clear(self, pm):
        log = OpsLog(persistence=pm)
        await log.add(
            OpsLogEntry.create(
                event_type=EVENT_STARTUP,
                severity=SEVERITY_INFO,
                message="Started",
            )
        )
        count = await log.clear()
        assert count == 1

    @pytest.mark.asyncio
    async def test_skip_prune(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path), ops_log_max=5)
        log = OpsLog(persistence=pm)
        for i in range(6):
            await log.add(
                OpsLogEntry.create(
                    event_type=EVENT_STARTUP,
                    severity=SEVERITY_INFO,
                    message=f"Event {i}",
                ),
                _skip_prune=True,
            )
        # skip_prune means we exceed the cap
        assert await pm.count_ops_log_entries() == 6

    @pytest.mark.asyncio
    async def test_retention_dual_threshold(self, tmp_path):
        pm = await PersistenceManager.create(
            str(tmp_path), ops_info_max=5, ops_warn_max=3
        )
        log = OpsLog(persistence=pm)
        # Insert 200 mixed entries — pruning fires at 100 and 200
        for i in range(200):
            sev = SEVERITY_INFO if i % 2 == 0 else SEVERITY_WARNING
            await log.add(
                OpsLogEntry.create(
                    event_type=EVENT_STARTUP,
                    severity=sev,
                    message=f"Event {i}",
                )
            )
        assert await pm.count_ops_info_entries() <= 5
        assert await pm.count_ops_warn_entries() <= 3

    @pytest.mark.asyncio
    async def test_cleanup_by_age(self, tmp_path):
        from datetime import datetime, timedelta, timezone

        pm = await PersistenceManager.create(str(tmp_path))
        # Insert an entry with an old timestamp
        old_ts = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        await pm.insert_ops_log_entries(
            [
                {
                    "id": "old1",
                    "timestamp": old_ts,
                    "event_type": EVENT_STARTUP,
                    "severity": SEVERITY_INFO,
                    "message": "Old event",
                }
            ]
        )
        await pm.insert_ops_log_entries(
            [
                OpsLogEntry.create(
                    event_type=EVENT_STARTUP,
                    severity=SEVERITY_INFO,
                    message="Recent event",
                ).to_dict()
            ]
        )
        assert await pm.count_ops_log_entries() == 2
        result = await pm.cleanup_ops_log_by_age(90)
        assert result["deleted"] == 1
        assert await pm.count_ops_log_entries() == 1

    @pytest.mark.asyncio
    async def test_details_none_omitted(self, pm):
        log = OpsLog(persistence=pm)
        await log.add(
            OpsLogEntry.create(
                event_type=EVENT_STARTUP,
                severity=SEVERITY_INFO,
                message="No details",
            )
        )
        entries, _ = await log.get_entries()
        assert "details" not in entries[0]
        assert "source" not in entries[0]


class TestConstants:
    def test_all_event_types(self):
        assert EVENT_STARTUP in ALL_EVENT_TYPES
        assert EVENT_OPS_LOG_CLEARED in ALL_EVENT_TYPES
        assert len(ALL_EVENT_TYPES) == 10

    def test_all_severities(self):
        assert SEVERITY_INFO in ALL_SEVERITIES
        assert SEVERITY_WARNING in ALL_SEVERITIES
        assert SEVERITY_ERROR in ALL_SEVERITIES
        assert len(ALL_SEVERITIES) == 3

    def test_all_sources(self):
        assert SOURCE_GATEWAY in ALL_SOURCES
        assert SOURCE_ADMIN in ALL_SOURCES
        assert len(ALL_SOURCES) == 5
