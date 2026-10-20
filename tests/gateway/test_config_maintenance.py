"""The load-time steps: structure, then names, once each, reporting what moved.

Slice 3 of #1027.  These two mechanisms already had tests of their own
(``test_migrations.py``, ``test_name_migration.py``), and those still pass
unchanged — what is new here is the order they now run in, that each reports
whether it changed anything, and that the loader keeps the registry so the admin
surface can list it beside the repairs the databases needed.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_rosetta.gateway.config import GatewayConfig, JsoncConfigIO
from llm_rosetta.gateway.maintenance.config import STEPS, apply, build_registry
from llm_rosetta.observability.maintenance import Policy, StepContext

#: A document as an older release wrote it: version 0, and both spellings that
#: have since been renamed.
_OLD = {
    "providers": {"alcf": {"type": "openai_chat", "shim": "volcengine--openai_chat"}},
    "rerank_providers": {"jina": {"api_key": "k", "base_url": "https://x"}},
    "rerank_models": {"rerank-v1": "jina"},
}

#: The same document, current in structure but keeping the old names.
_NAMES_ONLY = {"config_version": 1, "providers": {"x": {"type": "openai_chat"}}}


def _by_name(registry):
    return {result.name: result.changed for result in registry.results}


def test_the_registry_is_load_time_and_ordered():
    registry = build_registry()
    assert registry.policy is Policy.LOAD_TIME, (
        "a failure here fails the load: there is no config to serve without it"
    )
    assert [step.name for step in registry.steps] == STEPS
    assert [step.store for step in registry.steps] == ["config", "config"]


def test_a_document_with_work_to_do_reports_both_steps():
    raw = json.loads(json.dumps(_OLD))
    document, changed, registry = apply(raw)

    assert changed is True
    assert document is raw, "the document is corrected in place, as it always was"
    assert raw["config_version"] == 1, "the structure migration keeps the ledger"
    assert raw["providers"]["alcf"]["type"] == "chat_completions"
    assert raw["providers"]["alcf"]["shim"] == "volcengine--chat_completions"
    assert _by_name(registry) == {
        "config.migrate_structure": 1,
        "config.correct_legacy_names": 1,
    }


def test_a_current_document_with_old_names_reports_only_the_names():
    raw = json.loads(json.dumps(_NAMES_ONLY))
    _document, changed, registry = apply(raw)

    assert changed is True
    assert _by_name(registry) == {
        "config.migrate_structure": 0,
        "config.correct_legacy_names": 1,
    }


def test_a_current_document_reports_nothing():
    raw = json.loads(json.dumps(_OLD))
    apply(raw)
    _document, changed, registry = apply(raw)

    assert changed is False
    assert set(_by_name(registry).values()) == {0}


def test_the_name_correction_runs_on_a_document_the_structure_step_migrated():
    """The order is load-bearing: v0's legacy spellings only reach v1's shape first."""
    raw = json.loads(json.dumps(_OLD))
    apply(raw)
    # v1 is what gives the rerank provider its format; the name correction then
    # sees a document that already has one, rather than correcting a shape that
    # is about to be rewritten.
    assert "rerank_providers" not in raw
    assert raw["models"]["rerank-v1"]["type"] == "rerank"


def test_the_reader_records_the_registry_it_ran(tmp_path):
    path = tmp_path / "gateway.jsonc"
    path.write_text(json.dumps(_OLD))

    reader = JsoncConfigIO()
    assert reader.last_load_registry is None, (
        "a reader that has read nothing ran nothing"
    )
    reader.load(str(path))

    registry = reader.last_load_registry
    assert registry is not None, "the reader is what knows what its read ran"
    assert registry.name == "config"
    assert [result.name for result in registry.results] == STEPS
    assert (tmp_path / "gateway.jsonc.pre-migration.bak").exists(), (
        "the write-back is unchanged: this slice only changed who runs the steps"
    )


def test_the_config_built_from_the_loaded_document_reports_it(tmp_path):
    """The handoff, both halves: the reader that read it, the config built from it."""
    path = tmp_path / "gateway.jsonc"
    path.write_text(json.dumps(_OLD))

    reader = JsoncConfigIO()
    raw = reader.load(str(path))
    config = GatewayConfig(raw, maintenance=reader.last_load_registry)

    assert config.maintenance is not None
    assert config.maintenance.name == "config"
    assert _by_name(config.maintenance) == {
        "config.migrate_structure": 1,
        "config.correct_legacy_names": 1,
    }, "the config carries that load's per-step results, not a fresh registry"


def test_a_config_built_from_a_literal_reports_nothing(tmp_path):
    """Even immediately after a load: the registry belongs to the document read.

    This is the case a process-wide slot got wrong — the next construction took
    it, so a config built from a literal reported a load it never ran, and the
    config built from the document reported nothing.
    """
    path = tmp_path / "gateway.jsonc"
    path.write_text(json.dumps(_OLD))

    reader = JsoncConfigIO()
    raw = reader.load(str(path))
    assert reader.last_load_registry is not None, "the load did run the steps"

    literal = GatewayConfig({"config_version": 1, "providers": {}})
    assert literal.maintenance is None
    assert (
        GatewayConfig(raw, maintenance=reader.last_load_registry).maintenance
        is not None
    )


def test_one_reader_does_not_inherit_another_readers_registry(tmp_path):
    """The answer is the reader's, so a second reader starts with nothing."""
    path = tmp_path / "gateway.jsonc"
    path.write_text(json.dumps(_OLD))

    JsoncConfigIO().load(str(path))

    assert JsoncConfigIO().last_load_registry is None


def test_load_raw_records_its_own_run(tmp_path):
    """The admin edit round-trip reads through the same steps, and says so."""
    path = tmp_path / "gateway.jsonc"
    path.write_text(json.dumps(_OLD))

    reader = JsoncConfigIO()
    reader.load_raw(str(path))

    assert reader.last_load_registry is not None
    assert [result.name for result in reader.last_load_registry.results] == STEPS


def test_a_step_that_fails_fails_the_load():
    """LOAD_TIME: a config that cannot be brought forward is not served."""
    registry = build_registry()

    def _explode(context):
        raise ValueError("the config is not what this code expects")

    broken = type(registry)(
        name=registry.name,
        steps=[
            registry.steps[0].__class__(
                name="config.migrate_structure",
                description="a step that cannot run",
                store="config",
                run=_explode,
            )
        ],
        policy=Policy.LOAD_TIME,
    )
    try:
        broken.run_sync(StepContext(config={}))
    except ValueError as exc:
        assert "not what this code expects" in str(exc)
    else:  # pragma: no cover - the assertion above is the point
        raise AssertionError("a failing load-time step has to fail the load")


def test_the_loader_writes_back_once(tmp_path):
    """Both steps' changes travel in one write, as they did before."""
    path = tmp_path / "gateway.jsonc"
    path.write_text(json.dumps(_OLD))

    JsoncConfigIO().load(str(path))

    written = json.loads(Path(path).read_text())
    assert written["config_version"] == 1
    assert written["providers"]["alcf"]["type"] == "chat_completions"
