"""The gateway asserts the core surface level at startup (#1630)."""

from __future__ import annotations

import pytest

from llm_rosetta import contracts
from llm_rosetta_gateway.surface import REQUIRED_SURFACE_VERSION, check_core_surface


def test_the_installed_core_satisfies_the_gateway():
    assert isinstance(contracts.SURFACE_VERSION, int)
    assert contracts.SURFACE_VERSION >= REQUIRED_SURFACE_VERSION
    check_core_surface()


def test_a_newer_core_is_accepted():
    check_core_surface(REQUIRED_SURFACE_VERSION + 1)


@pytest.mark.parametrize("installed", [REQUIRED_SURFACE_VERSION - 1, "1", True])
def test_an_older_or_malformed_level_fails_with_a_clear_message(installed):
    with pytest.raises(RuntimeError, match="core surface level"):
        check_core_surface(installed)


def test_a_core_without_the_constant_counts_as_level_zero(monkeypatch):
    monkeypatch.delattr(contracts, "SURFACE_VERSION")
    with pytest.raises(RuntimeError, match="reports 0"):
        check_core_surface()


async def test_create_app_checks_the_surface(monkeypatch):
    import llm_rosetta_gateway.surface as surface
    from llm_rosetta_gateway.app import create_app

    monkeypatch.setattr(
        surface, "REQUIRED_SURFACE_VERSION", contracts.SURFACE_VERSION + 1
    )
    with pytest.raises(RuntimeError, match="core surface level"):
        await create_app(object())  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
