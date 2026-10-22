"""The repairs that run after the socket is open, as maintenance steps.

Five of them, and none is worth failing a boot for: the provider name on rows
written before that column existed, the ``last_used`` timestamp of a key that
predates the request log, the token totals that excluded cache tokens, the usage
bucket on rows written before the bucket column, and the request-log id of an
error dump.  They used to run under a single ``try``, so the first failure
skipped the other four — and the two an operator would look for first are the
last two.

Each is a :class:`~llm_rosetta_gateway.observability.maintenance.Step` in one
:attr:`~llm_rosetta_gateway.observability.maintenance.Policy.DEFERRED` registry: a
failure is recorded against the step that failed and the rest still run, and
the results are readable afterwards — the log line, the operator's
``/health/ready`` and ``/admin/api/maintenance``.

A step resolves its own inputs from the :class:`StepContext` rather than the
runner assembling them per step, which is what lets the same registry be run at
startup and re-run one step at a time from the admin surface.  A step whose
input is not configured (no key store, no config) returns 0: there is nothing
for it to repair, which is the same answer as "already repaired".
"""

from __future__ import annotations

import logging
from typing import Any

from llm_rosetta_gateway.observability.maintenance import (
    Policy,
    Registry,
    Step,
    StepContext,
)
from llm_rosetta_gateway.observability.persistence import _DB_FILENAME as GATEWAY_DB

logger = logging.getLogger(__name__)

#: What the admin surface and the log lines call this registry.
NAME = "deferred"


async def _backfill_provider_names(context: StepContext) -> int:
    """Name the provider on rows written before the column existed."""
    config = context.config
    if config is None:
        return 0
    model_to_provider = {
        model: route.provider_names[0] for model, route in config.models.items()
    }
    return await context.persistence.backfill_provider_names(model_to_provider) or 0


async def _backfill_key_last_used(context: StepContext) -> int:
    """Give a key with no ``last_used`` the timestamp its usage implies."""
    keystore = context.keystore
    if keystore is None:
        return 0
    return await keystore.backfill_last_used(context.persistence.db_path) or 0


async def _backfill_total_tokens(context: StepContext) -> int:
    """Repair totals written while cache tokens were excluded from them."""
    return await context.persistence.backfill_total_tokens() or 0


async def _backfill_usage_buckets(context: StepContext) -> int:
    """Attribute rows written before the bucket column existed."""
    keystore = context.keystore
    if keystore is None:
        return 0
    return (
        await context.persistence.backfill_usage_buckets(
            await keystore.label_bucket_map()
        )
        or 0
    )


async def _backfill_error_dump_log_ids(context: StepContext) -> int:
    """Match an error dump to its request log row by timestamp."""
    config = context.config
    aliases = config.model_upstream_names if config is not None else {}
    return (
        await context.persistence.backfill_error_dump_log_ids(model_aliases=aliases)
        or 0
    )


def build_registry(app: Any) -> Registry:
    """The deferred steps this app can run, given the stores it has.

    A step that needs the key store is registered only when there is one, so
    what the admin surface lists is what this deployment actually runs, and the
    ``DEFERRED`` policy is a property of the registry rather than a flag the
    startup task remembers to pass.
    """
    persistence = getattr(app, "persistence", None)
    keystore = getattr(app, "keystore", None)

    # Imported here, not at module level: the key store imports this package
    # for its own registry, and the filename is only needed when one exists.
    from ..keystore import _DB_FILENAME as KEYS_DB

    steps: list[Step] = []
    if persistence is not None:
        steps.append(
            Step(
                name="request_log.backfill_provider_names",
                description=(
                    "name the provider on rows written before the column existed"
                ),
                store=GATEWAY_DB,
                run=_backfill_provider_names,
            )
        )
        if keystore is not None:
            steps.append(
                Step(
                    name="api_keys.backfill_last_used",
                    description=(
                        "give a key with no last_used the timestamp its usage implies"
                    ),
                    store=KEYS_DB,
                    run=_backfill_key_last_used,
                )
            )
        steps.append(
            Step(
                name="request_log.backfill_total_tokens",
                description=("repair totals written while cache tokens were excluded"),
                store=GATEWAY_DB,
                run=_backfill_total_tokens,
            )
        )
        if keystore is not None:
            steps.append(
                Step(
                    name="request_log.backfill_usage_buckets",
                    description=(
                        "attribute rows written before the bucket column existed"
                    ),
                    store=GATEWAY_DB,
                    run=_backfill_usage_buckets,
                )
            )
        steps.append(
            Step(
                name="error_dumps.backfill_log_ids",
                description="match an error dump to its request log row",
                store=GATEWAY_DB,
                run=_backfill_error_dump_log_ids,
            )
        )
    return Registry(name=NAME, steps=steps, policy=Policy.DEFERRED)
