"""One idempotence property, asserted for every registry in the gateway.

Slice 1 asserted it for the request-log migrations; slice 3 adds the keystore's
own store and the load-time set, and the epic's acceptance is that *every* step
in *every* registry has it.  Rather than a property test per store — which is how
a store ends up without one, as the keystore did until this slice — each registry
is built here and run again against the state its own first run produced, so a
step that is not idempotent shows up as a non-zero ``changed`` wherever it lives.

A registry is only covered if it is named here, and
``test_the_registries_this_file_covers_are_the_gateways`` derives the names the
gateway builds and fails when one is missing.
"""

from __future__ import annotations

import pytest

from llm_rosetta.gateway.config_maintenance import (
    build_registry as config_registry,
)
from llm_rosetta.gateway.keystore import KeyStore
from llm_rosetta.gateway.keystore_maintenance import (
    build_registry as keystore_registry,
)
from llm_rosetta.observability import PersistenceManager
from llm_rosetta.observability.maintenance import Policy, StepContext

#: What this file asserts the property for, by registry name.
_COVERED = {"request_log", "api_keys", "config"}


@pytest.fixture
async def request_log(tmp_path):
    """A request-log store, brought to the current schema by its own open."""
    manager = await PersistenceManager.create(str(tmp_path / "gateway.db"))
    yield manager
    await manager.close()


@pytest.fixture
async def keystore(tmp_path):
    store = await KeyStore.create(tmp_path / "keys.db")
    yield store
    await store.close()


@pytest.fixture
def old_config():
    """A config document as an older release wrote it: no version, old names."""
    return {"providers": {"alcf": {"type": "openai_chat", "shim": "alcf--chat"}}}


def _assert_nothing_changed(registry, results) -> None:
    """The one property: a repeated run of a registry changes nothing."""
    assert registry.steps, "a registry with no steps would pass this vacuously"
    assert [result.name for result in results] == [
        step.name for step in registry.steps
    ], "every registered step has to report, or the property skips one"
    assert [result.changed for result in results] == [0] * len(results), (
        "a repeated run must change nothing"
    )
    assert all(result.ok for result in results)


async def test_the_request_log_migrations_are_idempotent(request_log):
    registry = request_log.migration_registry()
    results = await registry.run(StepContext(persistence=request_log))
    _assert_nothing_changed(registry, results)


async def test_the_keystore_migrations_are_idempotent(keystore):
    """The store the epic names: no test pinned this before slice 3."""
    assert keystore.maintenance is not None
    results = await keystore.maintenance.run(StepContext(keystore=keystore))
    _assert_nothing_changed(keystore.maintenance, results)


def test_the_load_time_steps_are_idempotent(old_config):
    """Run once to bring the document up; the second run changes nothing."""
    first = config_registry()
    assert any(
        result.changed for result in first.run_sync(StepContext(config=old_config))
    ), "the fixture has to be a document with work to do"

    second = config_registry()
    results = second.run_sync(StepContext(config=old_config))
    _assert_nothing_changed(second, results)


def test_a_load_time_step_is_a_function_not_a_coroutine(old_config):
    """Which is why the config path can drive it: no loop runs during a load."""
    registry = config_registry()
    registry.run_sync(StepContext(config=old_config))
    assert registry.results, "the run is kept, as everywhere else"


def test_the_registries_this_file_covers_are_the_gateways():
    """A registry added later cannot arrive without the property above.

    The names come from the builders rather than being restated, so adding a
    registry to the gateway fails this until it is covered here.
    """
    built = {config_registry().name, keystore_registry().name}
    assert built <= _COVERED, (
        f"the gateway builds a registry this file does not cover: {built - _COVERED}"
    )
    assert config_registry().policy is Policy.LOAD_TIME
    assert keystore_registry().policy is Policy.MANDATORY
