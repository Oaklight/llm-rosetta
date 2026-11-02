"""Provider and model parsing.

Everything that turns the ``providers`` / ``models`` sections of a document
into the pools :class:`~config.core.GatewayConfig` routes through.  Each
function takes the config object it fills, so the parsing reads as it did as a
method while living in its own module.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from ..routing import ModelRoute, ProviderEntry, strategy_for
from ..routing.model_routing import parse_model_routing, routing_from_legacy

if TYPE_CHECKING:
    from .core import GatewayConfig


logger = logging.getLogger("llm-rosetta-gateway")

#: Capabilities a model entry gets when it declares none.
DEFAULT_CAPABILITIES: list[str] = ["text"]

#: The capabilities a provider entry may declare.  These are the only names a
#: request can require, so an unknown one reserves nothing: a typo would
#: silently narrow the entry to no traffic at all, which is why it is refused.
ENTRY_CAPABILITIES = frozenset({"text", "vision", "tools", "reasoning"})


def parse_entry_capabilities(raw: Any, *, label: str) -> tuple[str, ...] | None:
    """Read one provider entry's ``capabilities`` field.

    ``None`` and ``[]`` both mean "declares nothing": the entry inherits the
    provider's own capabilities.  The file and the admin therefore agree on
    what an empty list means, instead of one deleting the key and the other
    keeping a declaration that covers nothing.

    Args:
        raw: The field's value, or ``None`` when it is absent.
        label: The config path, for the error message.

    Returns:
        The capability names, lowercased, or ``None`` for "inherit".

    Raises:
        ValueError: If the value is not a list of known names.
    """
    if raw is None:
        return None
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError(
            f"config: {label} has invalid capabilities {raw!r} (must be a list of names)"
        )
    names = [item.strip().lower() for item in raw]
    unknown = sorted(set(names) - ENTRY_CAPABILITIES)
    if unknown:
        raise ValueError(
            f"config: {label} declares unknown "
            f"{'capability' if len(unknown) == 1 else 'capabilities'} "
            f"{', '.join(unknown)}; known: {', '.join(sorted(ENTRY_CAPABILITIES))}"
        )
    return tuple(names) or None


def parse_models(
    raw_models: dict[str, Any],
    raw_providers: dict[str, dict[str, str]],
) -> tuple[dict[str, ModelRoute], dict[str, list[str]], dict[str, str]]:
    """Parse model routing entries from config.

    Supports string, single-provider dict, and multi-provider formats:
      - ``"model": "provider"`` (legacy)
      - ``"model": {"provider": "p", "capabilities": [...]}``
      - ``"model": {"providers": ["p1", "p2"]}`` (equal weight)
      - ``"model": {"providers": [{"name": "p1", "weight": 3}, ...]}``

    Models referencing disabled/missing providers are silently skipped.

    Returns:
        Tuple of (models, model_capabilities, model_upstream_names).
    """
    models: dict[str, ModelRoute] = {}
    model_capabilities: dict[str, list[str]] = {}
    model_upstream_names: dict[str, str] = {}
    for name, value in raw_models.items():
        route = parse_model_route(name, value, raw_providers)
        if route is None:
            continue
        models[name] = route
        caps, upstream = extract_model_metadata(value)
        model_capabilities[name] = caps
        if upstream:
            model_upstream_names[name] = upstream
    return models, model_capabilities, model_upstream_names


def parse_model_route(
    name: str,
    value: Any,
    raw_providers: dict[str, dict[str, str]],
) -> ModelRoute | None:
    """Resolve a single model config entry to a :class:`ModelRoute`.

    Returns ``None`` if the model should be skipped.
    """
    if isinstance(value, str):
        if value not in raw_providers:
            return None
        return ModelRoute([ProviderEntry(value)], name)
    if not isinstance(value, dict):
        raise ValueError(f"config: invalid model entry for '{name}'")
    if value.get("enabled") is False:
        return None
    if "provider" in value and "providers" in value:
        raise ValueError(
            f"config: model '{name}' has both 'provider' and "
            f"'providers' — use one or the other"
        )
    if "providers" in value:
        return parse_multi_provider(name, value, raw_providers)
    provider_name = value["provider"]
    if provider_name not in raw_providers:
        return None
    # A single-provider entry carries a routing block like any other: without
    # this the panel would write a block the runtime never reads.
    routing = _model_routing(name, value, {provider_name})
    return ModelRoute(
        [ProviderEntry(provider_name)], name, strategy_for(routing), routing=routing
    )


def declared_provider_names(value: Any) -> list[str]:
    """The providers a model entry names, in declared order.

    The unified schema accepts ``provider`` (singular) or ``providers``
    (a failover list of names or ``{"name": ...}`` entries); a legacy
    entry may be the bare provider name.  Typed-endpoint resolution uses
    this so an embedding/rerank/decision model declared with the list
    form is routed instead of being read as ``provider: ''``.
    """
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, dict):
        return []
    single = value.get("provider")
    if isinstance(single, str) and single:
        return [single]
    raw_list = value.get("providers")
    if not isinstance(raw_list, list):
        return []
    names: list[str] = []
    for item in raw_list:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            names.append(item["name"])
    return [name for name in names if name]


def extract_model_metadata(value: Any) -> tuple[list[str], str | None]:
    """Extract capabilities and upstream_model from a model entry."""
    if isinstance(value, str):
        return list(DEFAULT_CAPABILITIES), None
    caps = value.get("capabilities", list(DEFAULT_CAPABILITIES))
    return caps, value.get("upstream_model")


def parse_multi_provider(
    model_name: str,
    value: dict[str, Any],
    raw_providers: dict[str, dict[str, str]],
) -> ModelRoute | None:
    """Parse a multi-provider model entry.

    Returns ``None`` if no valid providers remain after filtering.
    """
    raw_list = value["providers"]
    if not isinstance(raw_list, list) or not raw_list:
        raise ValueError(
            f"config: model '{model_name}' has empty or invalid 'providers' list"
        )

    entries: list[ProviderEntry] = []
    for item in raw_list:
        if isinstance(item, str):
            pname, weight = item, 1
        elif isinstance(item, dict):
            pname = item.get("name", "")
            weight = int(item.get("weight", 1))
            if not pname:
                raise ValueError(
                    f"config: model '{model_name}' has a provider entry "
                    f"without a 'name' field"
                )
        else:
            raise ValueError(
                f"config: model '{model_name}' has invalid provider entry: {item}"
            )
        if weight < 1:
            raise ValueError(
                f"config: model '{model_name}' provider '{pname}' "
                f"has invalid weight {weight} (must be >= 1)"
            )
        # A provider the operator turned off keeps its place in the list but
        # takes no traffic, so the model can be narrowed without editing the
        # list by hand.
        if isinstance(item, dict) and item.get("enabled") is False:
            continue
        if pname not in raw_providers:
            logger.warning(
                "Model %r: provider %r not found, skipping",
                model_name,
                pname,
            )
            continue
        upstream = item.get("upstream_model") if isinstance(item, dict) else None
        caps = item.get("capabilities") if isinstance(item, dict) else None
        if caps is not None:
            caps = parse_entry_capabilities(
                caps, label=f"model '{model_name}' provider '{pname}'"
            )
        entries.append(
            ProviderEntry(pname, weight, upstream_model=upstream, capabilities=caps)
        )

    if not entries:
        return None

    routing = _model_routing(model_name, value, {entry.name for entry in entries})
    return ModelRoute(entries, model_name, strategy_for(routing), routing=routing)


def _model_routing(model_name: str, value: Any, members: set[str]):
    """Read a model entry's routing block, absorbing the old ``strategy`` key.

    The block is the only place routing is configured.  The pre-block
    ``strategy`` key is still read so an existing config keeps its behaviour,
    but naming both is a mistake the load refuses rather than a silent
    precedence rule.

    A rule may only promote or drop a member of this model's route; naming
    anyone else is a config error here, not a rule that quietly never matches.
    """
    raw_routing = value.get("routing") if isinstance(value, dict) else None
    legacy = value.get("strategy") if isinstance(value, dict) else None
    if raw_routing is not None and legacy is not None:
        raise ValueError(
            f"config: model '{model_name}' sets both 'strategy' and 'routing'; "
            "move the strategy into routing.strategy"
        )
    if legacy is not None:
        return routing_from_legacy(legacy, label=f"models.{model_name}.strategy")
    routing = parse_model_routing(raw_routing, label=f"models.{model_name}.routing")
    for index, rule in enumerate(routing.rules):
        target = rule.use or rule.pause
        if target not in members:
            raise ValueError(
                f"config: models.{model_name}.routing.rules[{index}] names "
                f"{target!r}, which is not a provider of this model; a rule "
                "may promote or drop a member, never introduce one"
            )
    return routing


def parse_model_overrides(
    raw_models: dict[str, Any],
) -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, dict[str, Any]],
    dict[str, bool],
    dict[str, float],
    dict[str, int],
]:
    """Extract per-model URL templates, reasoning overrides, flatten_system, timeouts, and max_tool_description_length."""
    url_templates: dict[str, str] = {}
    stream_url_templates: dict[str, str] = {}
    reasoning_overrides: dict[str, dict[str, Any]] = {}
    flatten_system: dict[str, bool] = {}
    timeouts: dict[str, float] = {}
    max_tool_desc_lengths: dict[str, int] = {}
    for model_name, value in raw_models.items():
        if isinstance(value, dict):
            if "url_template" in value:
                url_templates[model_name] = value["url_template"]
            if "stream_url_template" in value:
                stream_url_templates[model_name] = value["stream_url_template"]
            if value.get("reasoning_override"):
                reasoning_overrides[model_name] = value["reasoning_override"]
            if "flatten_system" in value:
                flatten_system[model_name] = bool(value["flatten_system"])
            if "timeout" in value:
                timeouts[model_name] = float(value["timeout"])
            if (
                "max_tool_description_length" in value
                and value["max_tool_description_length"] is not None
            ):
                max_tool_desc_lengths[model_name] = int(
                    value["max_tool_description_length"]
                )
        if model_name not in flatten_system and re.search(
            r"gemini", model_name, re.IGNORECASE
        ):
            flatten_system[model_name] = True
    return (
        url_templates,
        stream_url_templates,
        reasoning_overrides,
        flatten_system,
        timeouts,
        max_tool_desc_lengths,
    )


def parse_model_failover(raw_models: dict[str, Any]) -> dict[str, bool]:
    """Read the per-model ``failover_on_quota`` flags.

    Only models that turn it on appear in the result.  The default is off
    because a failover trades a warm upstream prompt cache for a served
    request: it is a choice an operator makes per model, not a global one.

    Args:
        raw_models: The ``models`` mapping as it appears in the config.

    Returns:
        ``{model: True}`` for models that opt in.
    """
    return {
        name: True
        for name, value in raw_models.items()
        if isinstance(value, dict) and value.get("failover_on_quota")
    }


def declared_provider_entries(value: Any) -> list[Any]:
    """The providers a model entry names, with the weight each one carries.

    The same shapes :func:`declared_provider_names` reads, kept as entries so
    a typed model's candidate pool carries its weights into the selector.  A
    typed model has no ``upstream_model`` per entry — the model's rename map
    covers that — so only the name and the weight survive.

    Args:
        value: A model entry: a provider name, or an object with ``provider``
            or ``providers``.

    Returns:
        ``ProviderEntry`` values in declared order.
    """
    from ..routing import ProviderEntry

    if isinstance(value, str):
        return [ProviderEntry(name=value)] if value else []
    if not isinstance(value, dict):
        return []
    if isinstance(value.get("provider"), str) and value["provider"]:
        return [ProviderEntry(name=value["provider"])]
    raw_list = value.get("providers")
    if not isinstance(raw_list, list):
        return []
    entries: list[Any] = []
    for item in raw_list:
        if isinstance(item, str):
            entries.append(ProviderEntry(name=item))
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            weight = item.get("weight", 1)
            try:
                weight = float(weight)
            except (TypeError, ValueError):
                weight = 1.0
            caps = item.get("capabilities")
            entries.append(
                ProviderEntry(
                    name=item["name"],
                    weight=weight,
                    capabilities=parse_entry_capabilities(
                        caps, label=f"provider entry '{item['name']}'"
                    ),
                )
            )
    return [entry for entry in entries if entry.name]


def distribute_typed_models(config: GatewayConfig, raw_models: dict[str, Any]) -> None:
    """Move models with non-LLM ``type`` from the main models pool
    into the corresponding specialized models dicts.

    Uses the model type registry to determine which types are valid
    non-LLM types.  Models whose provider lacks the corresponding
    endpoint config are silently skipped.
    """
    from llm_rosetta.formats import canonical_model_type

    from ..model_types import registered_type_names

    # Build a mapping from type name -> (providers dict, models dict)
    # so the distribution loop is data-driven.
    type_pools: dict[str, tuple[dict[str, Any], dict[str, str]]] = {}
    for tname in registered_type_names():
        if tname == "llm":
            continue
        providers_attr = f"{tname}_providers"
        models_attr = f"{tname}_models"
        if hasattr(config, providers_attr) and hasattr(config, models_attr):
            type_pools[tname] = (
                getattr(config, providers_attr),
                getattr(config, models_attr),
            )

    to_remove: list[str] = []
    for name, value in raw_models.items():
        if not isinstance(value, dict):
            continue
        # A model written before the rename says ``type: embedding``; the
        # pools are keyed by the canonical name.
        model_type = canonical_model_type(value.get("type", "llm"))
        if model_type == "llm" or value.get("enabled") is False:
            continue

        pool = type_pools.get(model_type)
        if pool is None:
            warn_unpooled_type(name, model_type)
            continue

        type_providers, type_models = pool
        provider_name = first_pool_provider(value, type_providers)
        if provider_name:
            type_models.setdefault(name, provider_name)
            # The whole candidate list, so the selector can score and fail
            # over across a typed model's providers rather than reading the
            # first one only.
            candidates = [
                entry
                for entry in declared_provider_entries(value)
                if entry.name in type_providers
            ]
            config.typed_model_candidates.setdefault(model_type, {})[name] = candidates
            to_remove.append(name)
        else:
            warn_skipped_typed_model(name, model_type, value)

    # Remove from main LLM pool (they belong in the specialized pool)
    for name in to_remove:
        config.models.pop(name, None)
        config.model_capabilities.pop(name, None)


def first_pool_provider(value: Any, type_providers: dict[str, Any]) -> str:
    """The first provider a model names that serves the typed endpoint."""
    return next(
        (
            provider
            for provider in declared_provider_names(value)
            if provider in type_providers
        ),
        "",
    )


def warn_unpooled_type(name: str, model_type: str) -> None:
    """Log a typed model whose type has no providers/models pool."""
    from ..model_types import get_model_type

    if get_model_type(model_type) is not None:
        # Type is registered but no providers/models attributes exist yet
        # (e.g. a new type without config integration).
        logger.debug(
            "Model %r has type=%s which is registered but has no config pool — skipped",
            name,
            model_type,
        )


def warn_skipped_typed_model(name: str, model_type: str, value: dict[str, Any]) -> None:
    """Log a typed model none of whose providers serve the endpoint."""
    from ..model_types import get_model_type

    desc = get_model_type(model_type)
    fmt_key = desc.config_format_key if desc else f"{model_type}_format"
    provider_names = declared_provider_names(value)
    declared = provider_names[0] if len(provider_names) == 1 else provider_names
    logger.warning(
        "Model %r has type=%s but provider %r lacks %s — skipped",
        name,
        model_type,
        declared,
        fmt_key,
    )
