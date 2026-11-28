"""What ``build_provider_info`` makes of an ACP lane, frozen (#1830 A8).

The lane facts of a provider entry (its transport, agent command, hardening,
environment, profile directories and limits) are resolved in ``config/providers.py``
today and move to ``transport/acp/lane.py``.  This snapshot was recorded from
``master`` before that move, so the expectation is the old code's behaviour and the
tree under test has to reproduce it, an error message included.  Regenerate it
deliberately, never to make a red test green:

    env PYTHONPATH=packages/llm-rosetta/src:packages/llm-rosetta-gateway/src \\
        python tests/gateway/test_acp_lane_snapshot.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

SNAPSHOT = Path(__file__).parent / "fixtures" / "acp_lane_snapshot.json"

#: The fields of ``ProviderInfo`` the lane resolution sets.
FIELDS = (
    "transport",
    "transport_owner",
    "acp_argv",
    "acp_argv_custom",
    "acp_harden",
    "acp_env",
    "acp_env_unset",
    "acp_session_meta",
    "transport_options",
    "acp_state_dir",
    "managed_acp_state_dir",
    "managed_auth_state_dir",
    "acp_processes",
    "acp_max_sessions",
    "acp_session_ttl",
)

#: ``(case id, provider type, shim name, provider entry, with a data dir)``.
CASES: tuple[tuple[str, str, str | None, dict[str, Any], bool], ...] = (
    ("bare", "acp", None, {"base_url": "http://lane.local", "transport": "acp"}, True),
    (
        "bare_without_a_data_dir",
        "acp",
        None,
        {"base_url": "http://lane.local", "transport": "acp"},
        False,
    ),
    (
        "relative_home",
        "acp",
        None,
        {
            "base_url": "http://lane.local",
            "transport": "acp",
            "acp": {"home": "profiles/one"},
        },
        True,
    ),
    (
        "absolute_home",
        "acp",
        None,
        {
            "base_url": "http://lane.local",
            "transport": "acp",
            "acp": {"home": "<data>/elsewhere/one"},
        },
        True,
    ),
    (
        "home_climbs_out",
        "acp",
        None,
        {
            "base_url": "http://lane.local",
            "transport": "acp",
            "acp": {"home": "../out"},
        },
        True,
    ),
    (
        "acp_block_not_a_mapping",
        "acp",
        None,
        {"base_url": "http://lane.local", "transport": "acp", "acp": ["x"]},
        True,
    ),
    (
        "custom_command_in_the_block",
        "acp",
        None,
        {
            "base_url": "http://lane.local",
            "transport": "acp",
            "acp": {"argv": ["my-agent", "serve"]},
        },
        True,
    ),
    (
        "top_level_command",
        "acp",
        None,
        {
            "base_url": "http://lane.local",
            "transport": "acp",
            "acp_argv": ["old-agent", "acp"],
        },
        True,
    ),
    (
        "block_command_beats_top_level",
        "acp",
        None,
        {
            "base_url": "http://lane.local",
            "transport": "acp",
            "acp_argv": ["old-agent"],
            "acp": {"argv": ["new-agent"]},
        },
        True,
    ),
    (
        "bad_command",
        "acp",
        None,
        {"base_url": "http://lane.local", "transport": "acp", "acp_argv": ["a", ""]},
        True,
    ),
    (
        "agy_is_hardened_by_name",
        "acp",
        None,
        {
            "base_url": "http://lane.local",
            "transport": "acp",
            "acp": {"argv": ["agy", "acp"]},
        },
        True,
    ),
    (
        "wrapper_forced_to_harden",
        "acp",
        None,
        {
            "base_url": "http://lane.local",
            "transport": "acp",
            "acp": {"argv": ["run-agy.sh"], "harden": True},
        },
        True,
    ),
    (
        "harden_not_a_boolean",
        "acp",
        None,
        {
            "base_url": "http://lane.local",
            "transport": "acp",
            "acp": {"harden": "yes"},
        },
        True,
    ),
    (
        "agy_unhardened_without_a_profile_flag",
        "acp",
        None,
        {
            "base_url": "http://lane.local",
            "transport": "acp",
            "acp": {"argv": ["agy", "acp"], "harden": False},
        },
        True,
    ),
    (
        "agy_unhardened_with_its_own_profile",
        "acp",
        None,
        {
            "base_url": "http://lane.local",
            "transport": "acp",
            "acp": {
                "argv": ["agy", "acp", "--gemini-dir", "<data>/mine"],
                "harden": False,
            },
        },
        True,
    ),
    (
        "gemini_dir_twice",
        "acp",
        None,
        {
            "base_url": "http://lane.local",
            "transport": "acp",
            "acp": {
                "argv": ["agy", "--gemini-dir", "/a", "--gemini-dir", "/b"],
                "harden": False,
            },
        },
        True,
    ),
    (
        "gemini_dir_relative",
        "acp",
        None,
        {
            "base_url": "http://lane.local",
            "transport": "acp",
            "acp": {"argv": ["agy", "--gemini-dir", "rel"], "harden": False},
        },
        True,
    ),
    (
        "gemini_dir_and_home_disagree",
        "acp",
        None,
        {
            "base_url": "http://lane.local",
            "transport": "acp",
            "acp": {
                "argv": ["agy", "--gemini-dir", "/a"],
                "harden": False,
                "home": "/b",
            },
        },
        True,
    ),
    (
        "environment_with_the_home_placeholder",
        "acp",
        None,
        {
            "base_url": "http://lane.local",
            "transport": "acp",
            "acp_env": {"A": "{acp_home}/x", "B": "{acp_state_dir}"},
        },
        True,
    ),
    (
        "environment_not_strings",
        "acp",
        None,
        {"base_url": "http://lane.local", "transport": "acp", "acp_env": {"A": 1}},
        True,
    ),
    (
        "transport_options_with_the_home_placeholder",
        "acp",
        None,
        {
            "base_url": "http://lane.local",
            "transport": "acp",
            "transport_options": {"env": {"DIR": "{acp_home}"}, "n": [1, "{acp_home}"]},
        },
        True,
    ),
    (
        "transport_options_not_an_object",
        "acp",
        None,
        {"base_url": "http://lane.local", "transport": "acp", "transport_options": 3},
        True,
    ),
    (
        "limits",
        "acp",
        None,
        {
            "base_url": "http://lane.local",
            "transport": "acp",
            "acp": {"processes": 3, "max_sessions": 7, "session_ttl": 60},
        },
        True,
    ),
    (
        "limit_processes_zero",
        "acp",
        None,
        {
            "base_url": "http://lane.local",
            "transport": "acp",
            "acp": {"processes": 0},
        },
        True,
    ),
    (
        "limit_ttl_not_a_number",
        "acp",
        None,
        {
            "base_url": "http://lane.local",
            "transport": "acp",
            "acp": {"session_ttl": "soon"},
        },
        True,
    ),
    (
        "an_http_lane_ignores_the_acp_settings",
        "chat_completions",
        None,
        {
            "base_url": "http://lane.local",
            "api_key": "k",
            "acp": {"home": "profiles/x", "processes": 2},
            "acp_env": {"A": "{acp_home}"},
        },
        True,
    ),
    (
        "unknown_transport",
        "acp",
        None,
        {"base_url": "http://lane.local", "transport": "carrier-pigeon"},
        True,
    ),
    (
        "antigravity_shim",
        "acp",
        "antigravity--acp",
        {},
        True,
    ),
    (
        "a_plain_http_lane",
        "chat_completions",
        None,
        {"base_url": "http://lane.local", "api_key": "k"},
        True,
    ),
    (
        "a_shim_that_declares_the_lane_facts",
        "acp",
        "snapshot-lane--acp",
        {
            "transport_options": {"extra": 2},
            "acp_env": {"C": "entry"},
        },
        True,
    ),
    (
        "antigravity_shim_with_its_own_command",
        "acp",
        "antigravity--acp",
        {"acp": {"argv": ["agy-1.7", "acp"]}},
        True,
    ),
)


def _fill(value: Any, data: str) -> Any:
    if isinstance(value, str):
        return value.replace("<data>", data)
    if isinstance(value, list):
        return [_fill(item, data) for item in value]
    if isinstance(value, dict):
        return {key: _fill(item, data) for key, item in value.items()}
    return value


def _mask(value: Any, data: str) -> Any:
    if isinstance(value, str):
        # The per-user fallback root, and the transports an environment has
        # installed, differ between machines (and the test suite moves the root).
        from llm_rosetta_gateway.transport.acp.profile import fallback_root

        masked = value.replace(data, "<data>").replace(
            str(fallback_root()), "<fallback>"
        )
        return masked.split(" Reachable from this shim")[0]
    if isinstance(value, (list, tuple)):
        return [_mask(item, data) for item in value]
    if isinstance(value, dict):
        return {key: _mask(item, data) for key, item in value.items()}
    return value


def _register_lane_shim() -> None:
    """A copy of the antigravity shim that declares what that one leaves empty."""
    import dataclasses

    from llm_rosetta.api import get_shim, register_shim

    base = get_shim("antigravity--acp")
    assert base is not None, "load the moved vendors first"
    connection = dataclasses.replace(
        base.connection,
        acp_env={"A": "{acp_home}/a", "C": "shim"},
        acp_env_unset=("B", "D"),
        acp_session_meta={"mode": "text"},
        transport_options={"dir": "{acp_home}", "keep": 1},
    )
    register_shim(
        dataclasses.replace(base, name="snapshot-lane--acp", connection=connection)
    )


def observe(case_id: str, data: str) -> Any:
    """What ``build_provider_info`` makes of one case: the fields, or the error."""
    from llm_rosetta_gateway.config.providers import build_provider_info

    _, provider_type, shim_name, entry, with_data = next(
        case for case in CASES if case[0] == case_id
    )
    if shim_name == "snapshot-lane--acp":
        _register_lane_shim()
    try:
        info = build_provider_info(
            provider_type,
            _fill(entry, data),
            shim_name=shim_name,
            data_dir=data if with_data else None,
            config_name="lane",
            quiet=True,
        )
    except ValueError as exc:
        return {"error": _mask(str(exc), data)}
    return {name: _mask(getattr(info, name), data) for name in FIELDS}


def _record() -> None:
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from _moved_vendors import load_moved_vendors

    load_moved_vendors()
    with tempfile.TemporaryDirectory() as raw:
        data = os.path.realpath(raw)
        snapshot = {case[0]: observe(case[0], data) for case in CASES}
    SNAPSHOT.write_text(json.dumps(snapshot, indent=1, sort_keys=True) + "\n")
    print(f"recorded {len(snapshot)} cases in {SNAPSHOT}")


if __name__ == "__main__":
    _record()
else:
    import pytest

    EXPECTED = json.loads(SNAPSHOT.read_text()) if SNAPSHOT.exists() else {}

    def test_every_case_is_recorded():
        assert set(EXPECTED) == {case[0] for case in CASES}

    @pytest.mark.parametrize("case_id", [case[0] for case in CASES])
    def test_an_acp_lane_resolves_as_it_did_on_master(case_id, tmp_path, moved_vendors):
        data = os.path.realpath(str(tmp_path))
        observed = json.loads(json.dumps(observe(case_id, data)))
        assert observed == EXPECTED[case_id]
