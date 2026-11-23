"""The allowance an ACP agent reports during a turn.

Claude's adapter sends the account's rolling windows as ``usage_update``
metadata; there is no key to probe.  These tests pin the translation, the
place the reading is stored, and the payload the Quota card reads.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from llm_rosetta.contracts.quota import QuotaProbeError, QuotaWindow
from llm_rosetta_gateway.quota.models import credential_fingerprint
from llm_rosetta_gateway.quota.service import read_provider_quota
from llm_rosetta_gateway.quota.state import QUOTA_STATE
from llm_rosetta_gateway.transport.acp_chunks import _chunk_of
from llm_rosetta_gateway.transport.acp_quota import (
    ACP_QUOTA_PROBE,
    LaneReading,
    clear_lane_readings,
    lane_credential,
    lanes_reading,
    observation_of,
    reading_of,
    record_reading,
    refresh_local,
    store_reading,
    windows_of,
)

RATE_LIMIT = {
    "status": "allowed",
    "rateLimitType": "five_hour",
    "resetsAt": 1_800_000_000,
    "unifiedWindows": {
        "five_hour": {"utilization": 0.02},
        "seven_day": {"utilization": 0.31, "resetsAt": 1_800_600_000},
    },
}

UPDATE = {
    "sessionUpdate": "usage_update",
    "used": 42,
    "size": 200000,
    "cost": {"amount": 0.01, "currency": "USD"},
    "_meta": {"_claude/rateLimit": RATE_LIMIT},
}


def _provider(tmp_path):
    return SimpleNamespace(
        config_name="claude",
        keyless=True,
        transport="acp",
        acp_state_dir=str(tmp_path / "data" / "acp" / "claude"),
        quota_identity=None,
        quota_probe=None,
        key_ring=None,
    )


@pytest.fixture(autouse=True)
def _clean_quota_state():
    QUOTA_STATE.clear()
    clear_lane_readings()
    yield
    QUOTA_STATE.clear()
    clear_lane_readings()


def test_the_usage_chunk_keeps_the_metadata_and_cost():
    chunk = _chunk_of(UPDATE)

    assert chunk is not None
    assert chunk["type"] == "usage"
    assert chunk["used"] == 42
    assert chunk["meta"]["_claude/rateLimit"]["unifiedWindows"]


def test_a_usage_update_without_metadata_is_only_a_token_count():
    chunk = _chunk_of({"sessionUpdate": "usage_update", "used": 7})

    assert chunk is not None
    assert "meta" not in chunk


def test_the_rate_limit_is_found_under_the_adapter_meta_key():
    assert observation_of({"update": UPDATE}) is RATE_LIMIT


def test_utilization_is_a_fraction_and_resets_are_unix_seconds():
    windows = {window.name: window for window in windows_of(RATE_LIMIT)}

    five_hour = windows["5 hours"]
    assert five_hour.used_percent == pytest.approx(2.0)
    assert five_hour.resets_at == datetime.fromtimestamp(1_800_000_000, tz=timezone.utc)
    assert five_hour.span_seconds == 5 * 3600
    assert five_hour.status == "allowed"
    assert windows["7 days"].used_percent == pytest.approx(31.0)


def test_a_reading_is_keyed_by_the_lane_profile(tmp_path):
    provider = _provider(tmp_path)

    reading = reading_of(UPDATE, provider)

    assert reading is not None
    assert reading.probe == ACP_QUOTA_PROBE
    assert reading.credential == lane_credential(provider)
    assert reading.credential != lane_credential(
        SimpleNamespace(
            config_name="claude-2",
            acp_state_dir=str(tmp_path / "data" / "acp" / "claude-2"),
        )
    )


def test_the_quota_payload_says_how_to_get_a_first_reading(tmp_path):
    provider = _provider(tmp_path)
    config = SimpleNamespace(providers={"claude": provider})

    payload = asyncio.run(read_provider_quota(config, "claude"))

    assert payload["supported"] is True
    assert payload["credentials"] == []
    assert "press Test" in payload["reason"]


def test_the_quota_payload_returns_the_recorded_windows(tmp_path):
    provider = _provider(tmp_path)
    config = SimpleNamespace(providers={"claude": provider})
    assert record_reading(UPDATE, provider) is True

    payload = asyncio.run(read_provider_quota(config, "claude"))

    assert payload["supported"] is True
    assert payload["status"] == "ok"
    entry = payload["credentials"][0]
    assert entry["credential"] == lane_credential(provider)
    assert [window["name"] for window in entry["reading"]["windows"]] == [
        "5 hours",
        "7 days",
    ]


def test_the_lane_fingerprint_matches_the_quota_rule(tmp_path):
    """The transport may not import quota, so the digest rule is duplicated."""
    provider = _provider(tmp_path)

    assert lane_credential(provider) == credential_fingerprint(
        str(provider.acp_state_dir)
    )


def test_two_entries_with_different_profiles_do_not_share_a_reading(tmp_path):
    first = _provider(tmp_path)
    second = SimpleNamespace(
        config_name="claude-work",
        keyless=True,
        transport="acp",
        acp_state_dir=str(tmp_path / "acp" / "claude-work"),
        quota_identity=None,
        quota_probe=None,
        key_ring=None,
    )
    assert record_reading(UPDATE, first) is True

    first_reading = lanes_reading(first)
    assert first_reading is not None
    assert first_reading.to_dict()["windows"]
    assert lanes_reading(second) is None


class _LocalProbe:
    """A probe whose allowance comes from a command's JSON output."""

    id = "local_test"
    last: dict | None = None

    def parse_local(self, body):
        type(self).last = body
        if body.get("error"):
            raise QuotaProbeError(str(body["error"]))
        bucket = body["bucket"]
        return None, [
            QuotaWindow(
                name="5 hours",
                used_percent=(1 - float(bucket["remaining_fraction"])) * 100,
                span_seconds=5 * 3600,
            )
        ]


def _local_provider(tmp_path, script, **env):
    return SimpleNamespace(
        config_name="local",
        keyless=True,
        transport="acp",
        acp_state_dir=str(tmp_path / "acp" / "local"),
        acp_argv=("fake-agent", "acp"),
        acp_env={"FROM_LANE": "lane"},
        acp_env_unset=(),
        quota_identity=None,
        quota_probe=None,
        key_ring=None,
        quota_probes=(_LocalProbe(),),
        quota_probe_id="local_test",
        quota_probe_impl=_LocalProbe(),
        quota_local_argv=(sys.executable, str(script)),
        quota_local_env=env,
        quota_local_timeout=5.0,
    )


def _script(tmp_path, body, *, exit_code=0, marker=None, suffix=""):
    """A stand-in vendor CLI that prints *body* as JSON."""
    payload = {"bucket": {"remaining_fraction": 0.75}, **body}
    path = tmp_path / "quota.py"
    source = [
        "import json, os, sys",
        f"payload = {payload!r}",
        "payload['argv1'] = sys.argv[1] if len(sys.argv) > 1 else None",
        "payload['lane'] = os.environ.get('FROM_LANE')",
        "payload['extra'] = os.environ.get('EXTRA')",
        "print(json.dumps(payload))",
    ]
    if suffix:
        source.append(f"print({suffix!r})")
    source.append(f"sys.exit({exit_code})")
    if marker is not None:
        source.insert(1, f"open({str(marker)!r}, 'a').write('run\\n')")
    path.write_text("\n".join(source), encoding="utf-8")
    return path


def _child_script(tmp_path, pid_file):
    """A CLI that leaves a helper holding stdout and then sleeps."""
    path = tmp_path / "child.py"
    path.write_text(
        "import subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(60)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(p.pid))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    return path


def _wait_for_exit(pid, timeout=10.0):
    """Wait until *pid* is gone, failing the test if it outlives *timeout*."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    raise AssertionError(f"process {pid} was still running after {timeout:g}s")


def test_a_local_command_is_run_with_the_lane_profile(tmp_path):
    script = _script(tmp_path, {})
    provider = _local_provider(tmp_path, script, EXTRA="{acp_home}")
    config = SimpleNamespace(providers={"local": provider})

    payload = asyncio.run(read_provider_quota(config, "local"))

    assert payload["status"] == "ok"
    assert payload["probe"] == "local_test"
    assert payload["refreshed"] is True
    windows = payload["credentials"][0]["reading"]["windows"]
    assert windows[0]["name"] == "5 hours"
    assert windows[0]["used_percent"] == pytest.approx(25.0)


def test_a_fresh_reading_is_not_reread_without_a_refresh(tmp_path):
    marker = tmp_path / "runs"
    script = _script(tmp_path, {}, marker=marker)
    provider = _local_provider(tmp_path, script)
    config = SimpleNamespace(providers={"local": provider})

    first = asyncio.run(read_provider_quota(config, "local"))
    second = asyncio.run(read_provider_quota(config, "local"))
    forced = asyncio.run(read_provider_quota(config, "local", force=True))

    assert first["refreshed"] is True
    assert second["refreshed"] is False
    assert forced["refreshed"] is True
    assert marker.read_text().count("run") == 2


def test_a_placeholder_is_expanded_in_argv_and_env(tmp_path):
    script = _script(tmp_path, {})
    provider = _local_provider(tmp_path, script, EXTRA="{acp_home}")
    provider.quota_local_argv = (
        sys.executable,
        str(script),
        "{acp_argv0}",
    )
    provider.acp_argv = ("fake-agent", "acp")

    reading = asyncio.run(
        refresh_local(
            provider,
            provider.quota_probe_impl,
            provider.quota_local_argv,
            provider.quota_local_env,
            timeout=5,
        )
    )

    assert reading.status == "ok"
    stored = lanes_reading(provider)
    assert stored is not None
    assert stored.to_dict()["windows"]
    assert _LocalProbe.last is not None
    assert _LocalProbe.last["argv1"] == "fake-agent"
    assert _LocalProbe.last["lane"] == "lane"
    assert _LocalProbe.last["extra"] == provider.acp_state_dir


def test_a_failed_command_is_stored_as_an_error_reading(tmp_path):
    script = _script(tmp_path, {}, exit_code=2)
    provider = _local_provider(tmp_path, script)
    config = SimpleNamespace(providers={"local": provider})

    payload = asyncio.run(read_provider_quota(config, "local"))

    assert payload["status"] == "error"
    assert "the quota command failed" in payload["reason"]
    reading = lanes_reading(provider)
    assert reading is not None
    assert reading.status == "error"


def test_a_parser_refusal_is_an_error_reading(tmp_path):
    script = _script(tmp_path, {"error": "no allowance in this account"})
    provider = _local_provider(tmp_path, script)

    reading = asyncio.run(
        refresh_local(
            provider,
            provider.quota_probe_impl,
            provider.quota_local_argv,
            provider.quota_local_env,
            timeout=5,
        )
    )

    assert reading.status == "error"
    assert reading.error == "no allowance in this account"


def test_a_reading_from_another_source_can_be_stored(tmp_path):
    provider = _local_provider(tmp_path, _script(tmp_path, {}))
    window = QuotaWindow(name="7 days", used_percent=10.0)
    store_reading(
        provider,
        LaneReading(
            probe="other",
            credential=lane_credential(provider),
            windows=[window],
        ),
    )

    stored = lanes_reading(provider)
    assert stored is not None
    assert stored.windows[0].name == "7 days"


def test_a_local_command_that_leaves_a_helper_is_bounded(tmp_path):
    """A helper holding the pipes cannot stretch the declared timeout."""
    pid_file = tmp_path / "helper.pid"
    provider = _local_provider(tmp_path, _child_script(tmp_path, pid_file))
    started = time.monotonic()

    reading = asyncio.run(
        refresh_local(
            provider,
            provider.quota_probe_impl,
            provider.quota_local_argv,
            provider.quota_local_env,
            timeout=0.5,
        )
    )

    elapsed = time.monotonic() - started
    assert reading.status == "error"
    assert reading.error is not None
    assert "did not finish" in reading.error
    assert elapsed < 8, f"the read took {elapsed:.1f}s"
    assert pid_file.exists()
    _wait_for_exit(int(pid_file.read_text()))


def test_cancelling_a_local_read_kills_the_command(tmp_path):
    pid_file = tmp_path / "helper.pid"
    provider = _local_provider(tmp_path, _child_script(tmp_path, pid_file))

    async def run():
        task = asyncio.create_task(
            refresh_local(
                provider,
                provider.quota_probe_impl,
                provider.quota_local_argv,
                provider.quota_local_env,
                timeout=30.0,
            )
        )
        deadline = time.monotonic() + 10
        while not pid_file.exists() and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        assert pid_file.exists(), "the helper never started"
        pid = int(pid_file.read_text())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return pid

    _wait_for_exit(asyncio.run(run()))


def test_concurrent_reads_run_the_local_command_once(tmp_path):
    marker = tmp_path / "runs"
    provider = _local_provider(tmp_path, _script(tmp_path, {}, marker=marker))
    config = SimpleNamespace(providers={"local": provider})

    async def both():
        return await asyncio.gather(
            read_provider_quota(config, "local"),
            read_provider_quota(config, "local"),
        )

    first, second = asyncio.run(both())

    assert marker.read_text().count("run") == 1
    assert first["status"] == "ok"
    assert second["status"] == "ok"


def test_a_failed_local_read_keeps_the_agents_reading(tmp_path):
    script = _script(tmp_path, {}, exit_code=2)
    provider = _local_provider(tmp_path, script)
    store_reading(
        provider,
        LaneReading(
            probe="agent",
            credential=lane_credential(provider),
            windows=[QuotaWindow(name="7 days", used_percent=12.0)],
        ),
    )
    config = SimpleNamespace(providers={"local": provider})

    payload = asyncio.run(read_provider_quota(config, "local", force=True))

    assert payload["status"] == "ok"
    assert payload["credentials"][0]["reading"]["windows"][0]["name"] == "7 days"
    assert "the quota command failed" in payload["warning"]
    stored = lanes_reading(provider)
    assert stored is not None
    assert stored.windows[0].name == "7 days"


def test_a_command_that_floods_stdout_is_cut_off(tmp_path):
    script = tmp_path / "flood.py"
    script.write_text(
        "import sys\nsys.stdout.write('x' * (2 << 20))\n", encoding="utf-8"
    )
    provider = _local_provider(tmp_path, script)

    reading = asyncio.run(
        refresh_local(
            provider,
            provider.quota_probe_impl,
            provider.quota_local_argv,
            provider.quota_local_env,
            timeout=10.0,
        )
    )

    assert reading.status == "error"
    assert "printed more than" in (reading.error or "")


def test_a_footer_after_the_json_is_tolerated(tmp_path):
    script = _script(tmp_path, {}, suffix="read complete")
    provider = _local_provider(tmp_path, script)

    reading = asyncio.run(
        refresh_local(
            provider,
            provider.quota_probe_impl,
            provider.quota_local_argv,
            provider.quota_local_env,
            timeout=5.0,
        )
    )

    assert reading.status == "ok"
    assert [window.name for window in reading.windows] == ["5 hours"]


class TestTheLocalCommandDeclaration:
    """The shim declares the command; a provider entry may override it."""

    name = "local-quota-test--acp"

    @pytest.fixture(autouse=True)
    def _shim(self):
        from llm_rosetta.contracts.shim import (
            ConnectionConfig,
            ProviderShim,
            QuotaDeclaration,
        )
        from llm_rosetta.shims import register_shim, unregister_shim

        register_shim(
            ProviderShim(
                name=self.name,
                format="acp",
                connection=ConnectionConfig(
                    base_url="http://agent.local", transport="acp"
                ),
                quota=QuotaDeclaration(
                    probe="local_test",
                    local_argv=("agy", "-p", "/usage"),
                    local_env={"FROM_SHIM": "1"},
                    local_timeout=15.0,
                ),
                quota_probes=(_LocalProbe(),),
            )
        )
        yield
        unregister_shim(self.name)

    def _info(self, cfg=None):
        from llm_rosetta_gateway.config.providers import build_provider_info

        return build_provider_info("acp", cfg or {}, shim_name=self.name)

    def test_the_shim_command_reaches_the_provider(self):
        info = self._info()

        assert info.quota_local_argv == ("agy", "-p", "/usage")
        assert info.quota_local_env == {"FROM_SHIM": "1"}
        assert info.quota_local_timeout == 15.0

    def test_a_provider_entry_overrides_the_command(self):
        info = self._info(
            {
                "quota": {
                    "local": {
                        "argv": ["other", "-p", "/usage"],
                        "env": {"FROM_ENTRY": "2"},
                        "timeout": 7,
                    }
                }
            }
        )

        assert info.quota_local_argv == ("other", "-p", "/usage")
        assert info.quota_local_env == {"FROM_SHIM": "1", "FROM_ENTRY": "2"}
        assert info.quota_local_timeout == 7.0

    def test_an_entry_override_is_merged_field_by_field(self):
        """Env layers, and the shim's command and timeout stay when unnamed."""
        info = self._info({"quota": {"local": {"env": {"FROM_ENTRY": "2"}}}})

        assert info.quota_local_argv == ("agy", "-p", "/usage")
        assert info.quota_local_env == {"FROM_SHIM": "1", "FROM_ENTRY": "2"}
        assert info.quota_local_timeout == 15.0

    def test_a_local_command_without_a_probe_is_refused(self):
        from llm_rosetta.contracts.shim import ConnectionConfig, ProviderShim
        from llm_rosetta.shims import register_shim, unregister_shim
        from llm_rosetta_gateway.config.providers import build_provider_info

        register_shim(
            ProviderShim(
                name="local-no-probe--acp",
                format="acp",
                connection=ConnectionConfig(
                    base_url="http://agent.local", transport="acp"
                ),
            )
        )
        try:
            with pytest.raises(ValueError, match="needs a probe"):
                build_provider_info(
                    "acp",
                    {"quota": {"local": {"argv": ["agy", "-p", "/usage"]}}},
                    shim_name="local-no-probe--acp",
                )
        finally:
            unregister_shim("local-no-probe--acp")

    @pytest.mark.parametrize(
        ("local", "message"),
        [
            ({"argv": []}, "quota.local.argv"),
            ({"argv": ["agy"], "env": {"A": 1}}, "quota.local.env"),
            ({"argv": ["agy"], "timeout": 0}, "quota.local.timeout"),
        ],
    )
    def test_a_malformed_override_is_refused(self, local, message):
        with pytest.raises(ValueError, match=message):
            self._info({"quota": {"local": local}})
