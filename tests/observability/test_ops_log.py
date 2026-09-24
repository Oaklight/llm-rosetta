"""Tests for the observability OpsLog (standalone, no gateway)."""

import pytest

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
    def test_add_and_get(self):
        log = OpsLog(max_entries=10)
        entry = OpsLogEntry.create(
            event_type=EVENT_STARTUP,
            severity=SEVERITY_INFO,
            message="Started",
        )
        log.add(entry)
        assert len(log) == 1
        entries, total = log.get_entries(limit=10)
        assert total == 1
        assert entries[0]["event_type"] == EVENT_STARTUP

    def test_newest_first(self):
        log = OpsLog(max_entries=10)
        for i in range(3):
            log.add(
                OpsLogEntry.create(
                    event_type=EVENT_STARTUP,
                    severity=SEVERITY_INFO,
                    message=f"Event {i}",
                )
            )
        entries, _ = log.get_entries(limit=10)
        assert entries[0]["message"] == "Event 2"
        assert entries[2]["message"] == "Event 0"

    def test_filter_by_event_type(self):
        log = OpsLog(max_entries=10)
        log.add(
            OpsLogEntry.create(
                event_type=EVENT_STARTUP,
                severity=SEVERITY_INFO,
                message="Started",
            )
        )
        log.add(
            OpsLogEntry.create(
                event_type=EVENT_CONFIG_RELOAD,
                severity=SEVERITY_INFO,
                message="Reloaded",
            )
        )
        entries, total = log.get_entries(event_type=EVENT_STARTUP)
        assert total == 1
        assert entries[0]["event_type"] == EVENT_STARTUP

    def test_filter_by_severity(self):
        log = OpsLog(max_entries=10)
        log.add(
            OpsLogEntry.create(
                event_type=EVENT_STARTUP,
                severity=SEVERITY_INFO,
                message="OK",
            )
        )
        log.add(
            OpsLogEntry.create(
                event_type=EVENT_STARTUP,
                severity=SEVERITY_ERROR,
                message="Bad",
            )
        )
        entries, total = log.get_entries(severity=SEVERITY_ERROR)
        assert total == 1
        assert entries[0]["message"] == "Bad"

    def test_filter_by_source(self):
        log = OpsLog(max_entries=10)
        log.add(
            OpsLogEntry.create(
                event_type=EVENT_STARTUP,
                severity=SEVERITY_INFO,
                message="From gateway",
                source=SOURCE_GATEWAY,
            )
        )
        log.add(
            OpsLogEntry.create(
                event_type=EVENT_KEY_CREATE,
                severity=SEVERITY_INFO,
                message="From admin",
                source=SOURCE_ADMIN,
            )
        )
        entries, total = log.get_entries(source=SOURCE_GATEWAY)
        assert total == 1
        assert entries[0]["source"] == SOURCE_GATEWAY

    def test_pagination(self):
        log = OpsLog(max_entries=20)
        for i in range(10):
            log.add(
                OpsLogEntry.create(
                    event_type=EVENT_STARTUP,
                    severity=SEVERITY_INFO,
                    message=f"Event {i}",
                )
            )
        entries, total = log.get_entries(limit=3, offset=0)
        assert total == 10
        assert len(entries) == 3

        entries2, _ = log.get_entries(limit=3, offset=3)
        assert len(entries2) == 3
        assert entries[0]["id"] != entries2[0]["id"]

    def test_clear_returns_count(self):
        log = OpsLog(max_entries=10)
        for _ in range(5):
            log.add(
                OpsLogEntry.create(
                    event_type=EVENT_STARTUP,
                    severity=SEVERITY_INFO,
                    message="Event",
                )
            )
        assert len(log) == 5
        count = log.clear()
        assert count == 5
        assert len(log) == 0

    def test_deque_max_entries(self):
        log = OpsLog(max_entries=3)
        for i in range(5):
            log.add(
                OpsLogEntry.create(
                    event_type=EVENT_STARTUP,
                    severity=SEVERITY_INFO,
                    message=f"Event {i}",
                )
            )
        assert len(log) == 3


class TestOpsLogPersistence:
    @pytest.fixture()
    def pm(self, tmp_path):
        return PersistenceManager(str(tmp_path), success_max=100)

    def test_add_and_query(self, pm):
        log = OpsLog(persistence=pm)
        log.add(
            OpsLogEntry.create(
                event_type=EVENT_STARTUP,
                severity=SEVERITY_INFO,
                message="Started",
                details={"host": "0.0.0.0", "port": 8080},
                source=SOURCE_GATEWAY,
            )
        )
        assert len(log) == 1
        entries, total = log.get_entries(limit=10)
        assert total == 1
        assert entries[0]["event_type"] == EVENT_STARTUP
        assert entries[0]["details"]["host"] == "0.0.0.0"
        assert entries[0]["source"] == SOURCE_GATEWAY

    def test_filter_event_type(self, pm):
        log = OpsLog(persistence=pm)
        log.add(
            OpsLogEntry.create(
                event_type=EVENT_STARTUP,
                severity=SEVERITY_INFO,
                message="Started",
            )
        )
        log.add(
            OpsLogEntry.create(
                event_type=EVENT_CONFIG_RELOAD,
                severity=SEVERITY_INFO,
                message="Reloaded",
            )
        )
        entries, total = log.get_entries(event_type=EVENT_CONFIG_RELOAD)
        assert total == 1
        assert entries[0]["message"] == "Reloaded"

    def test_filter_severity(self, pm):
        log = OpsLog(persistence=pm)
        log.add(
            OpsLogEntry.create(
                event_type=EVENT_STARTUP,
                severity=SEVERITY_INFO,
                message="OK",
            )
        )
        log.add(
            OpsLogEntry.create(
                event_type=EVENT_STARTUP,
                severity=SEVERITY_WARNING,
                message="Warn",
            )
        )
        entries, total = log.get_entries(severity=SEVERITY_WARNING)
        assert total == 1

    def test_filter_source(self, pm):
        log = OpsLog(persistence=pm)
        log.add(
            OpsLogEntry.create(
                event_type=EVENT_STARTUP,
                severity=SEVERITY_INFO,
                message="From gateway",
                source=SOURCE_GATEWAY,
            )
        )
        log.add(
            OpsLogEntry.create(
                event_type=EVENT_KEY_CREATE,
                severity=SEVERITY_INFO,
                message="From admin",
                source=SOURCE_ADMIN,
            )
        )
        entries, total = log.get_entries(source=SOURCE_ADMIN)
        assert total == 1
        assert entries[0]["source"] == SOURCE_ADMIN

    def test_clear(self, pm):
        log = OpsLog(persistence=pm)
        log.add(
            OpsLogEntry.create(
                event_type=EVENT_STARTUP,
                severity=SEVERITY_INFO,
                message="Started",
            )
        )
        count = log.clear()
        assert count == 1
        assert len(log) == 0

    def test_skip_prune(self, tmp_path):
        pm = PersistenceManager(str(tmp_path), ops_log_max=5)
        log = OpsLog(persistence=pm)
        for i in range(6):
            log.add(
                OpsLogEntry.create(
                    event_type=EVENT_STARTUP,
                    severity=SEVERITY_INFO,
                    message=f"Event {i}",
                ),
                _skip_prune=True,
            )
        # skip_prune means we exceed the cap
        assert len(log) == 6

    def test_retention_dual_threshold(self, tmp_path):
        pm = PersistenceManager(str(tmp_path), ops_info_max=5, ops_warn_max=3)
        log = OpsLog(persistence=pm)
        # Insert 200 mixed entries — pruning fires at 100 and 200
        for i in range(200):
            sev = SEVERITY_INFO if i % 2 == 0 else SEVERITY_WARNING
            log.add(
                OpsLogEntry.create(
                    event_type=EVENT_STARTUP,
                    severity=sev,
                    message=f"Event {i}",
                )
            )
        assert pm.count_ops_info_entries() <= 5
        assert pm.count_ops_warn_entries() <= 3

    def test_cleanup_by_age(self, tmp_path):
        from datetime import datetime, timedelta, timezone

        pm = PersistenceManager(str(tmp_path))
        # Insert an entry with an old timestamp
        old_ts = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        pm.insert_ops_log_entries(
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
        pm.insert_ops_log_entries(
            [
                OpsLogEntry.create(
                    event_type=EVENT_STARTUP,
                    severity=SEVERITY_INFO,
                    message="Recent event",
                ).to_dict()
            ]
        )
        assert pm.count_ops_log_entries() == 2
        result = pm.cleanup_ops_log_by_age(90)
        assert result["deleted"] == 1
        assert pm.count_ops_log_entries() == 1

    def test_details_none_omitted(self, pm):
        log = OpsLog(persistence=pm)
        log.add(
            OpsLogEntry.create(
                event_type=EVENT_STARTUP,
                severity=SEVERITY_INFO,
                message="No details",
            )
        )
        entries, _ = log.get_entries()
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
