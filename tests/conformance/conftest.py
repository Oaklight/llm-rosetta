"""Make the repo-root ``tools`` package importable from this test directory."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest  # noqa: E402

#: Suites that read a committed baseline and therefore need the pinned SDKs.
_SDK_BOUND = frozenset(
    {
        "test_fields_suite.py",
        "test_request_fields_suite.py",
        "test_stream_events_suite.py",
        "test_wire_suite.py",
    }
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip the baseline suites when the environment is not the pinned one.

    The ledgers are generated against ``tools/conformance/sdk-pins.txt``; a
    venv holding other SDK versions fails on fields those versions declare,
    which reads like a conversion regression.  CI installs with the pins, so
    this only fires for a local run.
    """
    from tools.conformance.pins import describe, mismatch

    differing = mismatch()
    if not differing:
        return
    reason = describe(differing)
    for item in items:
        if Path(str(item.path)).name in _SDK_BOUND:
            item.add_marker(pytest.mark.skip(reason=reason))
