"""The maintenance surface: what the deferred steps did, and re-running one.

The handlers are called the way the routes call them — with a request whose
``app`` carries the stores — so these pin the payloads an operator and the
admin panel's buttons read, including the two single-purpose routes, which are
now handlers over the same steps rather than a second implementation.
"""

from __future__ import annotations

import json
from typing import Any

from unittest.mock import AsyncMock, MagicMock

from llm_rosetta.gateway.admin.routes import maintenance as routes
from llm_rosetta.gateway.deferred_backfills import build_registry


class _App:
    def __init__(self, *, persistence: Any = None, keystore: Any = None) -> None:
        self.persistence = persistence
        self.keystore = keystore
        self.gateway_config = None
        # Attached by the test that checks a started run's registry wins.
        self.deferred_startup: Any = None


class _Request:
    def __init__(self, app: _App, **path_params: str) -> None:
        self.app = app
        self.path_params = path_params


def _stores(*, keystore: bool = True) -> tuple[Any, Any]:
    persistence = MagicMock()
    persistence.db_path = "/tmp/gateway.db"
    persistence.backfill_provider_names = AsyncMock(return_value=3)
    persistence.backfill_total_tokens = AsyncMock(return_value=7)
    persistence.backfill_usage_buckets = AsyncMock(return_value=4)
    persistence.backfill_error_dump_log_ids = AsyncMock(return_value=1)
    key_store = None
    if keystore:
        key_store = MagicMock()
        key_store.backfill_last_used = AsyncMock(return_value=2)
        key_store.label_bucket_map = AsyncMock(return_value={"wired": "bucket-1"})
    return persistence, key_store


def _body(response) -> dict:
    return json.loads(response.body)


class TestTheSurface:
    async def test_every_step_is_listed_before_anything_has_run(self):
        persistence, key_store = _stores()
        request = _Request(_App(persistence=persistence, keystore=key_store))

        body = _body(await routes.get_maintenance(request))

        # Every registry this app has, the deferred set first: one surface for
        # the schema of each store, the startup repairs, and the load-time steps.
        assert [entry["name"] for entry in body["registries"]] == [
            "deferred",
            "api_keys",
        ]
        assert body["registries"][0]["policy"] == "deferred"
        assert body["registries"][1]["policy"] == "mandatory"
        steps = body["registries"][0]["steps"]
        assert all(step["ran"] is False for step in steps)
        assert all(step["description"] for step in steps)
        # The whole entry, so the store an operator reads is pinned here too and
        # not only on the POST payload.
        assert steps[0] == {
            "name": "request_log.backfill_provider_names",
            "description": (
                "name the provider on rows written before the column existed"
            ),
            "store": "gateway.db",
            "ran": False,
        }

    async def test_a_step_that_has_run_reports_what_it_changed(self):
        persistence, key_store = _stores()
        app = _App(persistence=persistence, keystore=key_store)
        await routes.run_step(app, "request_log.backfill_total_tokens")

        body = _body(await routes.get_maintenance(_Request(app)))

        steps = {step["name"]: step for step in body["registries"][0]["steps"]}
        assert steps["request_log.backfill_total_tokens"]["ran"] is True
        assert steps["request_log.backfill_total_tokens"]["changed"] == 7
        assert steps["request_log.backfill_total_tokens"]["ok"] is True
        assert steps["error_dumps.backfill_log_ids"]["ran"] is False

    async def test_the_surface_shows_the_steps_the_deployment_has(self):
        persistence, _ = _stores(keystore=False)
        request = _Request(_App(persistence=persistence))

        body = _body(await routes.get_maintenance(request))

        names = [step["name"] for step in body["registries"][0]["steps"]]
        assert "api_keys.backfill_last_used" not in names


class TestReRunningOneStep:
    async def test_re_running_a_step_returns_its_result(self):
        persistence, key_store = _stores()
        app = _App(persistence=persistence, keystore=key_store)

        response = await routes.post_maintenance_run(
            _Request(app, name="request_log.backfill_usage_buckets")
        )

        assert response.status_code == 200
        assert _body(response) == {
            "name": "request_log.backfill_usage_buckets",
            "store": "gateway.db",
            "ok": True,
            "changed": 4,
            "seconds": 0.0,
            "error": None,
        }

    async def test_re_running_runs_only_that_step(self):
        persistence, key_store = _stores()
        app = _App(persistence=persistence, keystore=key_store)

        await routes.post_maintenance_run(
            _Request(app, name="request_log.backfill_total_tokens")
        )

        persistence.backfill_total_tokens.assert_awaited_once()
        persistence.backfill_provider_names.assert_not_awaited()
        persistence.backfill_error_dump_log_ids.assert_not_awaited()

    async def test_a_step_that_fails_answers_with_its_failure_not_a_500(self):
        """The operator asked a question; the answer is that the step failed."""
        persistence, key_store = _stores()
        persistence.backfill_total_tokens = AsyncMock(
            side_effect=RuntimeError("database is locked")
        )
        app = _App(persistence=persistence, keystore=key_store)

        response = await routes.post_maintenance_run(
            _Request(app, name="request_log.backfill_total_tokens")
        )

        assert response.status_code == 200
        body = _body(response)
        assert body["ok"] is False
        assert body["changed"] == 0
        assert body["error"] == "RuntimeError: database is locked"

    async def test_the_failure_is_there_for_the_next_reader(self):
        persistence, key_store = _stores()
        persistence.backfill_total_tokens = AsyncMock(
            side_effect=RuntimeError("database is locked")
        )
        app = _App(persistence=persistence, keystore=key_store)

        await routes.post_maintenance_run(
            _Request(app, name="request_log.backfill_total_tokens")
        )
        body = _body(await routes.get_maintenance(_Request(app)))

        steps = {step["name"]: step for step in body["registries"][0]["steps"]}
        assert steps["request_log.backfill_total_tokens"]["ok"] is False
        assert (
            "database is locked" in steps["request_log.backfill_total_tokens"]["error"]
        )

    async def test_an_unknown_step_is_not_found(self):
        persistence, key_store = _stores()
        app = _App(persistence=persistence, keystore=key_store)

        response = await routes.post_maintenance_run(
            _Request(app, name="request_log.no_such_step")
        )

        assert response.status_code == 404
        assert "no_such_step" in _body(response)["error"]


class TestTheTwoOlderRoutes:
    """The admin panel's buttons read ``updated``; they now read the step's."""

    async def test_the_keys_button_runs_the_key_step(self):
        persistence, key_store = _stores()
        request = _Request(_App(persistence=persistence, keystore=key_store))

        response = await routes.backfill_keys_last_used(request)

        assert _body(response) == {"updated": 2}
        key_store.backfill_last_used.assert_awaited_once_with("/tmp/gateway.db")

    async def test_the_dumps_button_runs_the_dump_step(self):
        persistence, key_store = _stores()
        request = _Request(_App(persistence=persistence, keystore=key_store))

        response = await routes.backfill_dump_log_ids(request)

        assert _body(response) == {"updated": 1}

    async def test_a_button_without_a_store_says_so(self):
        request = _Request(_App())

        response = await routes.backfill_keys_last_used(request)
        assert response.status_code == 400
        assert _body(response)["error"] == "No persistence configured"

        response = await routes.backfill_dump_log_ids(request)
        assert response.status_code == 400

    async def test_the_keys_button_without_a_key_store_answers_rather_than_raises(
        self,
    ):
        """A gateway with a request log and no key store: the shape this covers.

        Before these routes were handlers over the maintenance steps, this
        request raised out of the handler (``KeyStore not configured on this
        application``) and reached the client as a 500.  The step is simply not
        registered without a key store, so the answer is a 400 with a reason.
        """
        persistence, _ = _stores(keystore=False)
        request = _Request(_App(persistence=persistence))

        response = await routes.backfill_keys_last_used(request)

        assert response.status_code == 400
        assert _body(response)["error"] == "No key store configured"

    async def test_the_dumps_button_still_works_without_a_key_store(self):
        """Its step belongs to the request log, so a key store is no business of it."""
        persistence, _ = _stores(keystore=False)
        request = _Request(_App(persistence=persistence))

        response = await routes.backfill_dump_log_ids(request)

        assert response.status_code == 200
        assert _body(response) == {"updated": 1}


class TestTheLoadTimeSteps:
    """They are listed, and they are not re-runnable: their document is stale."""

    #: A document with both kinds of work in it, old names and all.
    _DOCUMENT = {
        "config_version": 1,
        "providers": {"alcf": {"type": "openai_chat", "shim": "deepseek--openai_chat"}},
    }

    def _app_with_a_loaded_config(self) -> _App:
        from llm_rosetta.gateway.config_maintenance import build_registry

        persistence, key_store = _stores()
        app = _App(persistence=persistence, keystore=key_store)
        app.gateway_config = MagicMock(maintenance=build_registry())
        return app

    def _app_with_a_real_load(self, tmp_path) -> _App:
        """The same surface, over the load a real reader ran rather than a stand-in."""
        from llm_rosetta.gateway.config import GatewayConfig, JsoncConfigIO

        path = tmp_path / "gateway.jsonc"
        path.write_text(json.dumps(self._DOCUMENT))
        reader = JsoncConfigIO()
        raw = reader.load(str(path))

        persistence, key_store = _stores()
        app = _App(persistence=persistence, keystore=key_store)
        app.gateway_config = GatewayConfig(raw, maintenance=reader.last_load_registry)
        return app

    async def test_the_config_registry_is_listed_beside_the_stores(self):
        app = self._app_with_a_loaded_config()

        body = _body(await routes.get_maintenance(_Request(app)))

        names = [entry["name"] for entry in body["registries"]]
        assert names == ["deferred", "api_keys", "config"]
        config = body["registries"][-1]
        assert config["policy"] == "load_time"
        assert [step["store"] for step in config["steps"]] == ["config", "config"]

    async def test_the_config_entry_reports_the_load_a_reader_ran(self, tmp_path):
        """End to end, without a substitute: the results are that load's own."""
        app = self._app_with_a_real_load(tmp_path)

        body = _body(await routes.get_maintenance(_Request(app)))

        config = body["registries"][-1]
        assert config["name"] == "config"
        assert config["policy"] == "load_time"
        steps = {step["name"]: step for step in config["steps"]}
        names_step = steps["config.correct_legacy_names"]
        assert names_step.pop("seconds") >= 0
        assert names_step == {
            "name": "config.correct_legacy_names",
            "description": "rewrite provider and shim names that have been renamed",
            "store": "config",
            "ran": True,
            "ok": True,
            "changed": 1,
            "error": None,
        }, "the surface reports what that load did to the document it read"
        assert steps["config.migrate_structure"]["ran"] is True

    async def test_all_four_registries_are_listed(self, tmp_path):
        """One surface for every store and the load, each with its own policy."""
        from llm_rosetta.gateway.config_maintenance import (
            build_registry as config_steps,
        )
        from llm_rosetta.gateway.keystore_maintenance import (
            build_registry as keystore_steps,
        )
        from llm_rosetta.observability.persistence import PersistenceManager

        app = self._app_with_a_real_load(tmp_path)
        app.persistence.maintenance = PersistenceManager.migration_registry()
        app.keystore.maintenance = keystore_steps()

        body = _body(await routes.get_maintenance(_Request(app)))

        assert [entry["name"] for entry in body["registries"]] == [
            "deferred",
            "request_log",
            "api_keys",
            "config",
        ]
        assert [entry["policy"] for entry in body["registries"]] == [
            "deferred",
            "mandatory",
            "mandatory",
            "load_time",
        ]
        # Each entry lists its own registry's steps, in the registry's order —
        # derived from the builders, so a step added later cannot go unlisted.
        expected = [
            [step.name for step in build_registry(app).steps],
            [step.name for step in PersistenceManager.migration_registry().steps],
            [step.name for step in keystore_steps().steps],
            [step.name for step in config_steps().steps],
        ]
        assert [
            [step["name"] for step in entry["steps"]] for entry in body["registries"]
        ] == expected

    async def test_re_running_a_load_time_step_is_refused(self):
        """Its context is the document a previous load read, not the one on disk."""
        app = self._app_with_a_loaded_config()

        response = await routes.post_maintenance_run(
            _Request(app, name="config.correct_legacy_names")
        )

        assert response.status_code == 409
        assert "loaded" in _body(response)["error"]

    async def test_a_stores_step_is_still_re_runnable_with_the_config_listed(self):
        app = self._app_with_a_loaded_config()

        response = await routes.post_maintenance_run(
            _Request(app, name="api_keys.backfill_usage_bucket")
        )

        assert response.status_code == 200
        assert _body(response)["store"] == "keys.db"


class TestWhichRegistryIsUsed:
    async def test_the_registry_a_startup_run_already_used_wins(self):
        """Its results are the ones the operator asked about."""
        persistence, key_store = _stores()
        app = _App(persistence=persistence, keystore=key_store)
        registry = build_registry(app)
        await registry.run(_context(app))
        app.deferred_startup = MagicMock(maintenance=registry)

        body = _body(await routes.get_maintenance(_Request(app)))

        steps = {step["name"]: step for step in body["registries"][0]["steps"]}
        assert steps["request_log.backfill_provider_names"]["ran"] is True

    async def test_a_fresh_registry_is_built_once_and_kept(self):
        """Otherwise a re-run's result would go with the object that made it."""
        persistence, key_store = _stores()
        app = _App(persistence=persistence, keystore=key_store)

        first = routes.registry_for(app)
        second = routes.registry_for(app)

        assert first is second
        await routes.post_maintenance_run(
            _Request(app, name="request_log.backfill_total_tokens")
        )
        listed = _body(await routes.get_maintenance(_Request(app)))
        steps = {step["name"]: step for step in listed["registries"][0]["steps"]}
        assert steps["request_log.backfill_total_tokens"]["ran"] is True


def _context(app: _App):
    return routes.context_for(app)
