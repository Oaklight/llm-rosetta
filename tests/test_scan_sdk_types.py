"""Tests for the SDK type scanner (``scripts/dev/scan_sdk_types.py``)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts/dev/scan_sdk_types.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("scan_sdk_types", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scan():
    return _load_script()


def test_a_missing_replica_module_is_an_error(scan):
    with pytest.raises(scan.ScanError, match="replica module"):
        scan.collect_our_types(["llm_rosetta.types.does_not_exist"])


def test_a_missing_sdk_module_is_an_error(scan):
    with pytest.raises(scan.ScanError, match="SDK module"):
        scan.collect_sdk_types(["openai.types.does_not_exist"])


def test_main_returns_non_zero_for_a_broken_provider(scan, monkeypatch, capsys):
    monkeypatch.setitem(
        scan.PROVIDERS,
        "broken",
        {
            "our_modules": ["llm_rosetta.types.does_not_exist"],
            "sdk_modules": ["openai.types.chat"],
            "relevance_patterns": [],
            "exclude_patterns": [],
            "name_collisions": [],
            "sdk_package": "openai",
        },
    )

    assert scan.main(["broken"]) == 1
    assert "cannot be imported" in capsys.readouterr().err


def test_collisions_are_excluded_and_named(scan):
    kept, excluded = scan.split_collisions(
        {"Function": ["parameters"], "Other": ["phase"]},
        ["Function", "NotPresent"],
    )

    assert kept == {"Other": ["phase"]}
    assert excluded == ["Function"]


def test_reviewed_collisions_are_declared(scan):
    assert "Function" in scan.PROVIDERS["openai_chat"]["name_collisions"]
    assert "FunctionToolParam" in scan.PROVIDERS["openai_responses"]["name_collisions"]
    assert "FunctionResponsePart" in scan.PROVIDERS["google_genai"]["name_collisions"]


def test_anthropic_alias_selects_the_messages_provider(scan):
    assert set(scan.select_providers(["anthropic"])) == {"messages"}


def test_unknown_provider_is_an_error(scan):
    with pytest.raises(scan.ScanError, match="unknown provider"):
        scan.select_providers(["does-not-exist"])


def test_markdown_names_the_excluded_collisions(scan):
    report: dict[str, Any] = {
        "generated": "2026-10-21 00:00 UTC",
        "sdk_versions": {"pkg": "1.0"},
        "providers": {
            "demo": {
                "drift": {"Real": ["phase"]},
                "relevant_new": ["NewType"],
                "collisions": ["Function"],
                "coverage": {"matched": 1, "replicas": 2, "sdk_types": 3},
            }
        },
    }

    text = scan.render_markdown(report)

    assert "Excluded as name collisions: `Function`" in text
    assert "`Real`: missing fields `phase`" in text
    assert "`NewType`" in text
