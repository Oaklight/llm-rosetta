"""Gateway configuration: JSONC loading, env-var substitution, validation."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager, suppress
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol, runtime_checkable

from llm_rosetta._vendor import jsonx
from llm_rosetta.auto_detect import ProviderType
from llm_rosetta.observability.maintenance import Registry
from llm_rosetta.routing import ResolvedRoute

from .model_types import ModelTypeDescriptor
from .providers import BASE_FORMATS, build_provider_info
from .routing import (
    DEFAULT_STRATEGY,
    AnchorStage,
    ModelRoute,
    ProviderEntry,
    RoutingChain,
    RoutingStage,
    SameFormatStage,
    SelectionContext,
    create_strategy,
)
from .transport import ProviderInfo

logger = logging.getLogger("llm-rosetta-gateway")


@dataclass(frozen=True)
class _TypedKeys:
    """The config keys and defaults one typed endpoint is parsed with."""

    type_name: str
    format_key: str
    path_key: str
    shim_key: str
    default_path: str
    default_format: str

    @classmethod
    def for_descriptor(
        cls, descriptor: ModelTypeDescriptor, format_key: str | None = None
    ) -> _TypedKeys:
        """The keys *descriptor* declares, with *format_key* overriding one."""
        name = descriptor.name
        return cls(
            type_name=name,
            format_key=format_key or descriptor.config_format_key or f"{name}_format",
            path_key=descriptor.config_path_key or f"{name}_path",
            shim_key=descriptor.config_shim_key or f"{name}_shim",
            default_path=descriptor.default_path or "",
            default_format=descriptor.default_format or "",
        )


def _typed_descriptors() -> list[ModelTypeDescriptor]:
    """The registered model types that carry a typed (non-LLM) endpoint.

    A descriptor joins the loop by declaring ``config_format_key``; ``llm``
    routes through the chat pipeline instead and has none.
    """
    from .model_types import all_model_types

    return [
        desc
        for desc in all_model_types()
        if not desc.is_llm and desc.config_format_key is not None
    ]


class TypedRoute(NamedTuple):
    """Resolved provider routing info for a typed (non-LLM) model type.

    Embedding, rerank and decision resolve to the same four facts — which
    provider, in which format, at which upstream path, with which connection
    — so they share one value.  ``path`` is the configured endpoint path; the
    upstream URL itself lives in ``provider_info``.
    """

    provider_name: str
    format: str
    path: str
    provider_info: ProviderInfo


@dataclass(frozen=True)
class ProviderIdentity:
    """What a configured provider is: its capability category and wire format.

    The pair is the gateway-side counterpart of a shim's
    ``(model_type, format)``.  ``model_type`` is one of the registry's
    categories (``llm``, ``embedding``, ``rerank``, ``decision``); ``format``
    is the dialect within that category.  A format name alone is ambiguous —
    ``openai`` is a vendor, a chat shim and an embedding format — so consumers
    that route or convert read the pair together.
    """

    model_type: str
    format: str


# ---------------------------------------------------------------------------
# Config file search paths (checked in order)
# ---------------------------------------------------------------------------

PATHS_TO_TRY = [
    "./config.jsonc",
    os.path.expanduser("~/.config/llm-rosetta-gateway/config.jsonc"),
    os.path.expanduser("~/.llm-rosetta-gateway/config.jsonc"),
]

# ---------------------------------------------------------------------------
# JSONC loader
# ---------------------------------------------------------------------------

_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")


def _substitute_env_vars(text: str) -> str:
    """Replace ${ENV_VAR} placeholders with environment variable values."""

    def _replace(m: re.Match) -> str:
        var_name = m.group(1)
        value = os.environ.get(var_name)
        if value is None:
            logger.warning("Environment variable %s is not set", var_name)
            return m.group(0)  # leave placeholder intact
        return value

    return _ENV_VAR_RE.sub(_replace, text)


def _loads(text: str) -> Any:
    """Plain JSON first; the JSONC preprocessor only when the text needs it.

    ``jsonx.loads`` strips comments and trailing commas with two
    character-by-character passes *before* handing the text to ``json.loads`` —
    3.5 ms against 0.3 ms for a 35 KiB document, and it charges that to
    documents that have neither, which is every file :func:`write_config` has
    ever written.  JSON is a subset of JSONC, so a ``json.loads`` that succeeds
    is the same answer; only its failure needs the scanning parser.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return jsonx.loads(text)


def _parse_config(path: str) -> dict[str, Any]:
    """Parse a JSONC config file with env-var substitution, and nothing else."""
    with open(path) as f:
        raw = f.read()
    substituted = _substitute_env_vars(raw)
    return _loads(substituted)


def _parse_config_raw(path: str) -> dict[str, Any]:
    """Parse a JSONC config file without env-var substitution, and nothing else."""
    with open(path) as f:
        return _loads(f.read())


def load_config(path: str) -> dict[str, Any]:
    """Load a JSONC config file for use, with env-var substitution.

    The document comes back through the load-time maintenance steps
    (:class:`JsoncConfigIO`), so a caller gets the shape the current code
    expects rather than whatever was written to disk, however long ago.  A step
    that changed something writes the migrated document back and leaves a
    ``.pre-migration.bak`` beside it; a step that cannot run raises, and there
    is no config to serve.  Env-var placeholders are substituted *and* kept in
    the written-back copy, so a round-trip does not bake a secret into the file.

    The registry the steps ran belongs to the reader that ran them, which this
    one-shot call drops.  A caller that serves an admin surface wants it —
    :func:`load_gateway_config` is that caller's entry point.
    """
    return JsoncConfigIO().load(path)


def load_config_raw(path: str) -> dict[str, Any]:
    """Load a JSONC config file for editing, without env-var substitution.

    The same steps as :func:`load_config` run, for the same reason: a document
    that is about to be written back should be brought forward first.  Env-var
    placeholders are left as they are, so editing a config does not replace them
    with their values.
    """
    return JsoncConfigIO().load_raw(path)


def write_config(path: str, data: dict[str, Any]) -> None:
    """Write a config dict as formatted JSON to *path* atomically.

    Creates parent directories if needed.  Writes to a temporary file in
    the same directory, flushes to disk, then atomically replaces the
    target via ``os.replace``.  Readers never see a partially-written file.

    Comments in the original JSONC file (if any) are **not** preserved.
    """
    import tempfile

    dir_ = os.path.dirname(path) or "."
    os.makedirs(dir_, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp)
        raise


# ---------------------------------------------------------------------------
# Config-file read-modify-write serialization
# ---------------------------------------------------------------------------


def _lock_exclusive(f: Any) -> None:
    """Acquire an exclusive lock on an open file (cross-platform)."""
    if sys.platform == "win32":
        import msvcrt  # noqa: F811  # Windows-only

        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined]
    else:
        import fcntl

        fcntl.flock(f, fcntl.LOCK_EX)


@contextmanager
def config_lock(path: str) -> Generator[None, None, None]:
    """Serialize read-modify-write cycles on the same config file.

    Admin routes must wrap their ``load_raw → modify → save`` sequences
    with this lock.  Without it, concurrent requests that each load the
    same version can silently overwrite each other's changes.

    Uses a ``.lock`` sidecar file with ``fcntl.flock`` (Unix) or
    ``msvcrt.locking`` (Windows) for cross-process mutual exclusion.
    This protects against multiple gateway instances sharing the same
    config file (e.g. dev + prod on the same host).
    """
    lock_path = os.path.realpath(path) + ".lock"
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    with open(lock_path, "a+") as lf:
        _lock_exclusive(lf)
        yield


# Per-path asyncio locks for in-process serialization.
# Keyed by (realpath, loop_id) to handle event loop replacement (e.g. tests).
_async_locks: dict[tuple[str, int], asyncio.Lock] = {}


def _get_async_lock(path: str) -> asyncio.Lock:
    real = os.path.realpath(path)
    loop = asyncio.get_running_loop()
    loop_id = id(loop)
    key = (real, loop_id)
    if key not in _async_locks:
        # Evict stale entries from previous loops
        stale = [k for k in _async_locks if k[1] != loop_id]
        for k in stale:
            del _async_locks[k]
        _async_locks[key] = asyncio.Lock()
    return _async_locks[key]


@asynccontextmanager
async def async_config_lock(path: str) -> AsyncGenerator[None, None]:
    """Async version of :func:`config_lock` for use in event-loop contexts.

    Acquires an in-process ``asyncio.Lock`` first (non-blocking to the
    event loop), then delegates the cross-process file lock to a thread
    executor so ``fcntl.flock`` never blocks the loop.
    """
    lock = _get_async_lock(path)
    async with lock:
        lock_path = os.path.realpath(path) + ".lock"
        os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
        lf = open(lock_path, "a+")  # noqa: SIM115
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _lock_exclusive, lf)
            yield
        finally:
            lf.close()


@runtime_checkable
class ConfigIO(Protocol):
    """Protocol for reading/writing gateway config files.

    The default :class:`JsoncConfigIO` handles JSONC files.  Downstream
    projects (e.g. argo-proxy) can supply an alternative implementation
    for other formats (YAML, TOML, etc.) via ``setup_admin(..., config_io=...)``.

    A reader that runs load-time maintenance steps records the registry it ran
    as ``last_load_registry``, and the caller that builds a
    :class:`GatewayConfig` from the document it returned passes that on.  A
    reader without that attribute runs no such steps, which reads the same as a
    config built programmatically: nothing ran.
    """

    def load(self, path: str) -> dict[str, Any]:
        """Read config with env-var substitution (for runtime use)."""
        ...

    def load_raw(self, path: str) -> dict[str, Any]:
        """Read config without env-var substitution (for edit round-trips)."""
        ...

    def save(self, path: str, data: dict[str, Any]) -> None:
        """Write config back to disk."""
        ...


class JsoncConfigIO:
    """Default :class:`ConfigIO` — reads/writes JSONC with env-var substitution.

    On load, applies any pending config migrations and writes back the
    migrated config if changes were made.
    """

    def __init__(self) -> None:
        #: The registry the most recent read ran, for the config object built
        #: from the document that read returned.  A load-time step runs while a
        #: document is read, so the reader is what knows the answer — it is not
        #: left in a process-wide slot, where the *next* :class:`GatewayConfig`
        #: construction would take it whether or not it came from a document that
        #: ran anything.  The caller that reads and then builds the config hands
        #: it over explicitly, so a config built from a literal reports nothing —
        #: the same answer as "no load ran".
        self.last_load_registry: Registry | None = None

    def load(self, path: str) -> dict[str, Any]:
        raw = _parse_config(path)
        return self._apply_migrations(raw, path, is_raw=False)

    def load_raw(self, path: str) -> dict[str, Any]:
        raw = _parse_config_raw(path)
        return self._apply_migrations(raw, path, is_raw=True)

    def save(self, path: str, data: dict[str, Any]) -> None:
        write_config(path, data)

    def _apply_migrations(
        self, raw: dict[str, Any], path: str, *, is_raw: bool = False
    ) -> dict[str, Any]:
        """Bring one config document up to date, through the load-time registry.

        The steps are the versioned structure migrations and the name
        correction, in that order, under
        :attr:`Policy.LOAD_TIME <llm_rosetta.observability.maintenance.Policy>`;
        the registry that ran is kept on this reader, which is what read the
        document, and is handed to the config built from it.
        """
        from .maintenance.config import apply

        raw, changed, registry = apply(raw, path=path)
        self.last_load_registry = registry
        if changed:
            self._write_back(raw, path, is_raw=is_raw)
        return raw

    def _write_back(self, raw: dict[str, Any], path: str, *, is_raw: bool) -> None:
        """Write the migrated document back, and keep the original beside it.

        Best-effort on purpose.  The steps have already run and their result is
        the document this load returns, so nothing here is needed to serve it —
        and a config that is mounted read-only (a k8s ConfigMap, a container
        layer) would otherwise turn a working deployment into a refused start
        the first time a migration had something to do.  A directory that cannot
        be written is logged, and the document is served from memory instead:
        the process holds the migrated one either way, and the version on disk
        simply does not advance.
        """
        from .maintenance.config import apply

        if is_raw:
            disk_data = raw
        else:
            disk_data = _parse_config_raw(path)
            # The same steps over the on-disk copy, so a substituted env var
            # is not written back in place of its placeholder.
            disk_data, _changed, _registry = apply(disk_data, path=path)
        try:
            import shutil

            bak = path + ".pre-migration.bak"
            if not os.path.exists(bak):
                shutil.copy2(path, bak)
                logger.info("config: backed up original to %s", bak)
            with config_lock(path):
                write_config(path, disk_data)
        except OSError as exc:
            logger.warning(
                "config: %s was brought forward in memory but could not be "
                "written back (%s); serving the migrated document and leaving "
                "the file as it is",
                path,
                exc,
            )
            return
        logger.info(
            "config: migrated %s to version %s",
            path,
            disk_data.get("config_version"),
        )


def discover_config(explicit_path: str | None = None) -> str | None:
    """Find the first existing config file.

    If *explicit_path* is given, return it unconditionally (caller is
    responsible for handling missing files).  Otherwise search
    ``PATHS_TO_TRY`` in order and return the first hit, or ``None``.
    """
    if explicit_path is not None:
        return explicit_path
    for path in PATHS_TO_TRY:
        if os.path.isfile(path):
            return path
    return None


# ---------------------------------------------------------------------------
# Routing loop detection
# ---------------------------------------------------------------------------


def detect_routing_loops(
    models: dict[str, ModelRoute],
    upstream_names: dict[str, str],
) -> list[str]:
    """Detect potential routing loops in the model configuration.

    A loop risk exists when model A's upstream_model is also a registered
    model name — if the upstream provider's base_url points back at this
    gateway, the request would be re-routed internally.  Also detects
    transitive chains (A → B → C → A).

    Returns a list of human-readable warning strings (empty if clean).
    """
    warnings: list[str] = []

    # Collect all edges: model_name → upstream_model (where upstream is
    # also a model key).  Include both model-level and per-entry overrides.
    edges: dict[str, str] = {}
    for model_name, route in models.items():
        up = upstream_names.get(model_name)
        if up and up in models:
            edges[model_name] = up
            continue
        for entry in route.providers:
            if entry.upstream_model and entry.upstream_model in models:
                edges[model_name] = entry.upstream_model
                break

    # Report every edge as a potential loop — even non-cyclic chains like
    # A→B are dangerous when the upstream provider points back at this
    # gateway.  True cycles (A→B→A) get a stronger message.
    reported: set[str] = set()
    for source, target in edges.items():
        if source in reported:
            continue
        # Walk forward to detect cycles.
        chain = [source]
        node: str | None = target
        chain_set = {source}
        while node and node not in chain_set:
            chain.append(node)
            chain_set.add(node)
            node = edges.get(node)
        if node and node in chain_set:
            cycle_start = chain.index(node)
            cycle = chain[cycle_start:] + [node]
            warnings.append(f"routing loop: {' → '.join(cycle)}")
            reported.update(cycle)
        else:
            warnings.append(
                f"upstream_model collision: '{source}' maps to"
                f" '{target}' which is also a routable model"
            )
            reported.add(source)

    return warnings


# ---------------------------------------------------------------------------
# Config class
# ---------------------------------------------------------------------------


def load_gateway_config(path: str) -> GatewayConfig:
    """Read a config document and build the config the gateway serves.

    The load-time steps run as part of the read, and the config carries the
    registry they ran — which is what the admin surface reports the config's
    maintenance from.  That is the whole difference between this and
    ``GatewayConfig(load_config(path))``, which has run the steps without
    keeping any trace of them.
    """
    reader = JsoncConfigIO()
    raw = reader.load(path)
    return GatewayConfig(raw, maintenance=reader.last_load_registry)


class GatewayConfig:
    """Parsed and validated gateway configuration."""

    # Default capabilities when not specified in config.
    DEFAULT_CAPABILITIES: list[str] = ["text"]

    # Typed (non-LLM) endpoint pools.  The registry-driven loop in __init__
    # fills one set per registered type; the built-in three are declared here
    # so their attributes are visible to a type checker.  A further type reads
    # its pools through :meth:`typed_pool`.
    embedding_providers: dict[str, dict[str, Any]]
    embedding_provider_infos: dict[str, ProviderInfo]
    embedding_models: dict[str, str]
    rerank_providers: dict[str, dict[str, Any]]
    rerank_provider_infos: dict[str, ProviderInfo]
    rerank_models: dict[str, str]
    decision_providers: dict[str, dict[str, Any]]
    decision_provider_infos: dict[str, ProviderInfo]
    decision_models: dict[str, str]
    default_embedding_format: str
    default_rerank_format: str
    default_decision_format: str

    @staticmethod
    def _formats_for_type(type_name: str) -> list[str]:
        """Return supported formats for a model type from the registry.

        Falls back to an empty list if the type is not registered.
        """
        from .model_types import get_model_type

        desc = get_model_type(type_name)
        return list(desc.formats) if desc else []

    @staticmethod
    def embedding_formats() -> list[str]:
        """Return supported embedding formats from the model type registry."""
        return GatewayConfig._formats_for_type("embedding")

    @staticmethod
    def rerank_formats() -> list[str]:
        """Return supported rerank formats from the model type registry."""
        return GatewayConfig._formats_for_type("rerank")

    def __init__(
        self, raw: dict[str, Any], *, maintenance: Registry | None = None
    ) -> None:
        # What the load of *this* document ran, when a reader ran it: the
        # load-time steps' results, for the surface that reports them.  Passed
        # in by the caller that read the document rather than claimed from a
        # slot, so a config built from a literal reports nothing.
        self.maintenance: Registry | None = maintenance

        # Declarative external shim sources (see gateway/shim_sources.py).
        # Kept verbatim so create_app can apply them for programmatic callers
        # and get_config can expose the section to the admin UI.
        self.shims: dict[str, Any] = dict(raw.get("shims") or {})

        all_providers: dict[str, dict[str, str]] = raw.get("providers", {})

        # The unfiltered view, for consumers that have to see a *disabled*
        # provider too: the shim-removal advisory warns about a config that would
        # break the moment it is switched back on.  It is the caller's mapping,
        # so `_resolve_provider_types` correcting legacy names in place is
        # visible here as well.
        self.raw_providers: dict[str, dict[str, str]] = all_providers

        # Filter out disabled providers (enabled defaults to True)
        self._raw_providers: dict[str, dict[str, str]] = {
            name: cfg
            for name, cfg in all_providers.items()
            if cfg.get("enabled", True) is not False
        }

        self.provider_identities, self.provider_shim_names = (
            self._resolve_provider_identities(self._raw_providers)
        )
        # ``provider_types`` keeps meaning the wire format for existing
        # consumers (routing, proxy, observability, admin).  Read the pair from
        # ``provider_identities`` when the category matters.
        self.provider_types: dict[str, str] = {
            name: ident.format for name, ident in self.provider_identities.items()
        }

        self.provider_supports_custom_tools = self._parse_custom_tools_overrides(
            self._raw_providers
        )

        self.provider_hoist_system_messages = self._parse_hoist_system_overrides(
            self._raw_providers
        )

        self.provider_preflight_token_count = (
            self._parse_preflight_token_count_overrides(self._raw_providers)
        )

        self.provider_max_tool_description_length = (
            self._parse_max_tool_description_length_overrides(self._raw_providers)
        )

        self.provider_key_affinity = self._parse_key_affinity_overrides(
            self._raw_providers
        )

        self.models, self.model_capabilities, self.model_upstream_names = (
            self._parse_models(raw.get("models", {}), self._raw_providers)
        )

        raw_models = raw.get("models", {})
        (
            self.model_url_templates,
            self.model_stream_url_templates,
            self.model_reasoning_overrides,
            self.model_flatten_system,
            self.model_timeouts,
            self.model_max_tool_description_length,
        ) = self._parse_model_overrides(raw_models)

        _server = raw.get("server", {})
        self._apply_server_settings(_server)
        self._apply_auth_settings(_server)
        self._apply_rate_limit_settings(_server)
        self._apply_circuit_breaker_settings(_server)

        self._apply_debug_settings(raw.get("debug", {}))

        self._validate(raw.get("models", {}))

        # Build ProviderInfo objects (with key rotation support)
        self.providers: dict[str, ProviderInfo] = {
            name: build_provider_info(
                self.provider_types[name],
                cfg,
                global_proxy=self.proxy,
                shim_name=self.provider_shim_names.get(name),
            )
            for name, cfg in self._raw_providers.items()
        }

        # Attach the process-wide quota state: the key ring learns which
        # credentials a probe or an in-band 429 has demoted.  State outlives a
        # config reload, so it is bound here rather than owned by the config.
        from .quota.service import bind_provider_quota

        for provider_name, provider_info in self.providers.items():
            bind_provider_quota(provider_name, provider_info)

        # Upstream-account identity per provider, derived from the credential
        # source at build time.  Two providers sharing a token front one
        # account — the key tier-2 same-format routing compares on.  Computed
        # here because the env var *name* is already expanded away by then.
        self.provider_account_tokens: dict[str, str] = {
            name: pinfo.account_token(name) for name, pinfo in self.providers.items()
        }

        # Provider selection, as stages over one request's selection context.
        # The order is checked when the chain is built, so a stage added later
        # fails at construction rather than on the first request.
        self._same_format = SameFormatStage()
        stages: list[RoutingStage] = [AnchorStage()]
        if self.prefer_same_format:
            stages.append(self._same_format)
        self._selection = RoutingChain(stages)

        # --- Typed (non-LLM) endpoint pools ---
        # One loop over the model-type registry: every non-LLM type gets its
        # providers / provider_infos / models pools from the unified provider
        # entries plus its legacy ``<type>_providers`` section, and its
        # default format from the registry.  A newly registered type needs no
        # edit here.
        for descriptor in _typed_descriptors():
            name = descriptor.name
            pools = self._parse_typed_config(
                self._raw_providers,
                {},
                descriptor=descriptor,
                require_declared_format=True,
                map_models=False,
                global_proxy=self.proxy,
            )
            legacy = self._parse_typed_config(
                raw.get(f"{name}_providers", {}),
                raw.get(f"{name}_models", {}),
                descriptor=descriptor,
                format_key="format",
                global_proxy=self.proxy,
            )
            # Unified entries win; the legacy section fills the gaps.
            providers, provider_infos, models = pools
            for key, value in legacy[0].items():
                providers.setdefault(key, value)
            for key, value in legacy[1].items():
                provider_infos.setdefault(key, value)
            for key, value in legacy[2].items():
                models.setdefault(key, value)
            setattr(self, f"{name}_providers", providers)
            setattr(self, f"{name}_provider_infos", provider_infos)
            setattr(self, f"{name}_models", models)
            setattr(
                self,
                f"default_{name}_format",
                raw.get(f"default_{name}_format", descriptor.default_format or ""),
            )

        # --- Extract type-tagged models from unified models pool ---
        self._distribute_typed_models(raw.get("models", {}))

    def _distribute_typed_models(self, raw_models: dict[str, Any]) -> None:
        """Move models with non-LLM ``type`` from the main models pool
        into the corresponding specialized models dicts.

        Uses the model type registry to determine which types are valid
        non-LLM types.  Models whose provider lacks the corresponding
        endpoint config are silently skipped.
        """
        from .model_types import registered_type_names

        # Build a mapping from type name -> (providers dict, models dict)
        # so the distribution loop is data-driven.
        type_pools: dict[str, tuple[dict[str, Any], dict[str, str]]] = {}
        for tname in registered_type_names():
            if tname == "llm":
                continue
            providers_attr = f"{tname}_providers"
            models_attr = f"{tname}_models"
            if hasattr(self, providers_attr) and hasattr(self, models_attr):
                type_pools[tname] = (
                    getattr(self, providers_attr),
                    getattr(self, models_attr),
                )

        to_remove: list[str] = []
        for name, value in raw_models.items():
            if not isinstance(value, dict):
                continue
            model_type = value.get("type", "llm")
            if model_type == "llm" or value.get("enabled") is False:
                continue

            pool = type_pools.get(model_type)
            if pool is None:
                self._warn_unpooled_type(name, model_type)
                continue

            type_providers, type_models = pool
            provider_name = self._first_pool_provider(value, type_providers)
            if provider_name:
                type_models.setdefault(name, provider_name)
                to_remove.append(name)
            else:
                self._warn_skipped_typed_model(name, model_type, value)

        # Remove from main LLM pool (they belong in the specialized pool)
        for name in to_remove:
            self.models.pop(name, None)
            self.model_capabilities.pop(name, None)

    @staticmethod
    def _first_pool_provider(value: Any, type_providers: dict[str, Any]) -> str:
        """The first provider a model names that serves the typed endpoint."""
        return next(
            (
                provider
                for provider in GatewayConfig._declared_provider_names(value)
                if provider in type_providers
            ),
            "",
        )

    @staticmethod
    def _warn_unpooled_type(name: str, model_type: str) -> None:
        """Log a typed model whose type has no providers/models pool."""
        from .model_types import get_model_type

        if get_model_type(model_type) is not None:
            # Type is registered but no providers/models attributes exist yet
            # (e.g. a new type without config integration).
            logger.debug(
                "Model %r has type=%s which is registered but has no "
                "config pool — skipped",
                name,
                model_type,
            )

    @staticmethod
    def _warn_skipped_typed_model(
        name: str, model_type: str, value: dict[str, Any]
    ) -> None:
        """Log a typed model none of whose providers serve the endpoint."""
        from .model_types import get_model_type

        desc = get_model_type(model_type)
        fmt_key = desc.config_format_key if desc else f"{model_type}_format"
        provider_names = GatewayConfig._declared_provider_names(value)
        declared = provider_names[0] if len(provider_names) == 1 else provider_names
        logger.warning(
            "Model %r has type=%s but provider %r lacks %s — skipped",
            name,
            model_type,
            declared,
            fmt_key,
        )

    @staticmethod
    def _parse_custom_tools_overrides(
        raw_providers: dict[str, dict[str, str]],
    ) -> dict[str, bool]:
        """Extract per-provider supports_custom_tools overrides."""
        result: dict[str, bool] = {}
        for pname, pcfg in raw_providers.items():
            if isinstance(pcfg, dict) and "supports_custom_tools" in pcfg:
                result[pname] = bool(pcfg["supports_custom_tools"])
        return result

    @staticmethod
    def _parse_hoist_system_overrides(
        raw_providers: dict[str, dict[str, str]],
    ) -> dict[str, bool]:
        """Extract per-provider hoist_system_messages overrides."""
        result: dict[str, bool] = {}
        for pname, pcfg in raw_providers.items():
            if isinstance(pcfg, dict) and "hoist_system_messages" in pcfg:
                result[pname] = bool(pcfg["hoist_system_messages"])
        return result

    @staticmethod
    def _parse_preflight_token_count_overrides(
        raw_providers: dict[str, dict[str, str]],
    ) -> dict[str, bool]:
        """Extract per-provider preflight_token_count overrides."""
        result: dict[str, bool] = {}
        for pname, pcfg in raw_providers.items():
            if isinstance(pcfg, dict) and "preflight_token_count" in pcfg:
                result[pname] = bool(pcfg["preflight_token_count"])
        return result

    @staticmethod
    def _parse_key_affinity_overrides(
        raw_providers: dict[str, dict[str, str]],
    ) -> dict[str, bool]:
        """Extract per-provider key_affinity overrides."""
        result: dict[str, bool] = {}
        for pname, pcfg in raw_providers.items():
            if isinstance(pcfg, dict) and "key_affinity" in pcfg:
                result[pname] = bool(pcfg["key_affinity"])
        return result

    @staticmethod
    def _parse_max_tool_description_length_overrides(
        raw_providers: dict[str, dict[str, str]],
    ) -> dict[str, int]:
        """Extract per-provider max_tool_description_length overrides."""
        result: dict[str, int] = {}
        for pname, pcfg in raw_providers.items():
            if isinstance(pcfg, dict) and "max_tool_description_length" in pcfg:
                val = pcfg["max_tool_description_length"]
                if val is not None:
                    result[pname] = int(val)
        return result

    @staticmethod
    def _parse_model_overrides(
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

    def _apply_server_settings(self, _server: dict[str, Any]) -> None:
        """Parse server section: host, port, proxy, timeouts, CORS, etc."""
        self.host: str = _server.get("host", "0.0.0.0")
        self.port: int = _server.get("port", 8765)
        self.proxy: str | None = _server.get("proxy")
        self.socket: str | None = _server.get("socket")
        self.credential_visible: bool = _server.get("credential_visible", False)
        # Force off when admin panel has no password protection
        if not _server.get("admin_password"):
            self.credential_visible = False

        # When no API keys are configured, ``open_on_no_keys`` decides whether
        # the standalone gateway serves /v1/* anonymously (True) or rejects
        # every request with 403 (False).  Defaults to False (secure by
        # default).  Set to True for trusted, localhost-only deployments.
        self.open_on_no_keys: bool = bool(_server.get("open_on_no_keys", False))

        # When a model has providers in multiple API formats, prefer the
        # one matching the client's format to avoid unnecessary cross-format
        # conversion.  The preference is tiered and relative to an anchor
        # provider, and lives in the selection chain as a stage.  Off by
        # default: enabling changes which upstream a multi-format model
        # reaches.
        self.prefer_same_format: bool = bool(_server.get("prefer_same_format", False))

        self.admin_password: str | None = _server.get("admin_password")
        if self.admin_password and _ENV_VAR_RE.search(self.admin_password):
            raise ValueError(
                "config: admin_password contains an unresolved ${...} placeholder. "
                "Set the environment variable or use a literal password."
            )

        # CORS allow-list for /admin/* endpoints.
        # Default [] means same-origin only (no Access-Control-Allow-Origin).
        self.admin_cors_origins: list[str] = _server.get("admin_cors_origins", []) or []

        # Optional root redirect: GET / returns 307 to the given path.
        self.root_redirect: str | None = _server.get("root_redirect")

        # upstream_timeout: entire upstream HTTP lifecycle (connect + headers
        # + per-chunk streaming reads).  read_timeout: how long the inbound
        # httpserver waits for the client to send a complete request.
        self.upstream_timeout: float = max(
            1.0, float(_server.get("upstream_timeout", 300.0))
        )
        self.read_timeout: float = max(1.0, float(_server.get("read_timeout", 300.0)))

        # Request-log retention knobs (consumed by setup_admin).
        self.request_log: dict[str, Any] = _server.get("request_log", {}) or {}
        self.ops_log: dict[str, Any] = _server.get("ops_log", {}) or {}
        self.data_dir: str | None = _server.get("data_dir")

        # TLS termination: paths to PEM cert/key files.
        self.tls_cert: str | None = _server.get("tls_cert")
        self.tls_key: str | None = _server.get("tls_key")
        if bool(self.tls_cert) != bool(self.tls_key):
            raise ValueError(
                "config: tls_cert and tls_key must both be set or both be omitted"
            )

    def _apply_auth_settings(self, _server: dict[str, Any]) -> None:
        """Parse API key storage settings from the server section."""
        # Custom SQLite DB path for API key storage (default: alongside config)
        self.api_keys_db: str | None = _server.get("api_keys_db")

    _VALID_RL_ALGORITHMS = frozenset(
        {"token_bucket", "fixed_window", "sliding_window", "gcra"}
    )

    def _apply_circuit_breaker_settings(self, _server: dict[str, Any]) -> None:
        """Parse circuit breaker settings from the server section.

        Supports global defaults under ``server.circuit_breaker`` and
        per-provider overrides under each provider's ``circuit_breaker``
        sub-object.  The circuit breaker is disabled by default.
        """
        from .middleware.circuit_breaker import (
            CircuitBreakerConfig,
            CircuitBreakerRegistry,
        )

        cb = _server.get("circuit_breaker", {}) or {}
        default_cfg = CircuitBreakerConfig(
            enabled=bool(cb.get("enabled", False)),
            error_threshold=int(cb.get("error_threshold", 5)),
            cooldown_seconds=float(cb.get("cooldown_seconds", 30.0)),
            half_open_max_probes=int(cb.get("half_open_max_probes", 1)),
        )
        self.circuit_breaker_registry = CircuitBreakerRegistry(default_cfg)

        # Per-provider overrides
        for pname, pcfg in self._raw_providers.items():
            if not isinstance(pcfg, dict):
                continue
            pcb = pcfg.get("circuit_breaker")
            if not isinstance(pcb, dict):
                continue
            override = CircuitBreakerConfig(
                enabled=bool(pcb.get("enabled", default_cfg.enabled)),
                error_threshold=int(
                    pcb.get("error_threshold", default_cfg.error_threshold)
                ),
                cooldown_seconds=float(
                    pcb.get("cooldown_seconds", default_cfg.cooldown_seconds)
                ),
                half_open_max_probes=int(
                    pcb.get("half_open_max_probes", default_cfg.half_open_max_probes)
                ),
            )
            self.circuit_breaker_registry.set_override(pname, override)

    def _apply_rate_limit_settings(self, _server: dict[str, Any]) -> None:
        """Parse rate limiting settings from the server section."""
        rl = _server.get("rate_limit", {}) or {}
        self.rate_limit_enabled: bool = bool(rl.get("enabled", False))
        algorithm = rl.get("algorithm", "sliding_window")
        if algorithm not in self._VALID_RL_ALGORITHMS:
            logger.warning(
                "config: invalid rate_limit algorithm %r, falling back to 'sliding_window'",
                algorithm,
            )
            algorithm = "sliding_window"
        self.rate_limit_algorithm: str = algorithm
        self.rate_limit_global: str | list[str] | None = rl.get("global")
        self.rate_limit_per_ip: str | list[str] | None = rl.get("per_ip")
        self.rate_limit_per_key: str | list[str] | None = rl.get("per_key")
        self.rate_limit_per_model: str | list[str] | None = rl.get("per_model")
        self.rate_limit_exclude: list[str] = rl.get(
            "exclude_paths", ["/health", "/admin"]
        )
        # When True, trust X-Forwarded-For / X-Real-IP for per-IP limiting.
        # Default False — use direct peer address (safe when exposed directly).
        self.rate_limit_trust_proxy: bool = bool(rl.get("trust_proxy", False))

    def _apply_debug_settings(self, _debug: dict[str, Any]) -> None:
        """Parse debug/logging settings with env-var overrides.

        Env vars ``LLM_ROSETTA_VERBOSE``, ``LLM_ROSETTA_LOG_BODIES``,
        and ``LLM_ROSETTA_LOG_FORMAT`` override their config counterparts.
        """
        self.verbose: bool = _debug.get("verbose", False) or os.environ.get(
            "LLM_ROSETTA_VERBOSE", ""
        ).lower() in ("1", "true", "yes")
        self.log_bodies: bool = _debug.get("log_bodies", False) or os.environ.get(
            "LLM_ROSETTA_LOG_BODIES", ""
        ).lower() in ("1", "true", "yes")
        self.error_dumps_enabled: bool = _debug.get("error_dumps", True)

        # Log format: "text" (colourised, default for TTY), "json" (structured
        # one-line JSON, default for non-TTY), or "auto" (detect from stderr
        # TTY status).  Env var LLM_ROSETTA_LOG_FORMAT overrides config.
        _env_log_format = os.environ.get("LLM_ROSETTA_LOG_FORMAT", "").strip().lower()
        _cfg_log_format = _debug.get("log_format", "auto")
        raw_log_format = _env_log_format or _cfg_log_format
        if raw_log_format not in ("json", "text", "auto"):
            logger.warning(
                "config: invalid log_format %r, falling back to 'auto'",
                raw_log_format,
            )
            raw_log_format = "auto"
        self.log_format: str = raw_log_format  # resolved later by setup_logging

    def _validate(self, raw_models: dict[str, Any]) -> None:
        if not self._raw_providers:
            logger.warning(
                "config: no enabled providers — all providers may be disabled"
            )
            return
        if not self.models:
            logger.warning(
                "config: no routable models — models may reference disabled providers"
            )
            return
        for model, route in self.models.items():
            entry = raw_models.get(model)
            # Typed models are moved out of the LLM pool later; an embedding
            # model legitimately points at an embedding provider.
            declared_type = (
                entry.get("type", "llm") if isinstance(entry, dict) else "llm"
            )
            for pname in route.provider_names:
                if pname not in self._raw_providers:
                    raise ValueError(
                        f"config: model '{model}' references unknown provider '{pname}'"
                    )
                if declared_type != "llm":
                    continue
                identity = self.provider_identities.get(pname)
                if identity is not None and identity.model_type != "llm":
                    raise ValueError(
                        f"config: model '{model}' is an LLM model but provider "
                        f"'{pname}' serves {identity.model_type} "
                        f"({identity.format})"
                    )

        # Warn about potential routing loops.
        for warning in detect_routing_loops(self.models, self.model_upstream_names):
            logger.warning("config: %s", warning)

    @staticmethod
    def _resolve_endpoint_shim(
        cfg: dict[str, Any], *, type_name: str, shim_key: str
    ) -> Any:
        """Return the shim a non-LLM endpoint of *type_name* is configured through.

        An explicit ``cfg[shim_key]`` must name a registered shim serving
        *type_name*; a mismatch is a configuration error.  Otherwise the
        provider-wide ``shim`` / ``type`` is used only when its category
        matches, so a chat provider named ``openai`` keeps its chat identity
        while still serving embeddings through ``embedding_format``.

        Args:
            cfg: Raw provider config entry.
            type_name: Registry category the endpoint belongs to.
            shim_key: Provider config key naming the endpoint shim
                (e.g. ``"embedding_shim"``).

        Returns:
            The effective :class:`~llm_rosetta.shims.provider_shim.ProviderShim`,
            or ``None``.

        Raises:
            ValueError: If ``cfg[shim_key]`` is unknown or serves another type.
        """
        from llm_rosetta.shims import get_shim

        declared = cfg.get(shim_key)
        if declared:
            shim = get_shim(declared)
            if shim is None:
                raise ValueError(f"config: {shim_key} names unknown shim '{declared}'")
            if shim.model_type != type_name:
                raise ValueError(
                    f"config: {shim_key} shim '{declared}' serves "
                    f"{shim.model_type}, not {type_name}"
                )
            return shim
        for key in ("shim", "type"):
            value = cfg.get(key)
            if not value:
                continue
            shim = get_shim(value)
            if shim is not None and shim.model_type == type_name:
                return shim
        return None

    @staticmethod
    def _identity_category(wire_format: str, cfg: dict[str, Any]) -> str | None:
        """Return the category a bare *wire_format* names, or ``None`` if unknown.

        Chat formats resolve to ``llm``.  A non-chat format name is known but
        may span categories (``cohere``, ``jina`` and ``voyage`` are both
        embedding and rerank formats), so prefer a category the provider
        declares an endpoint for and otherwise the registry order.  A name that
        appears in no category stays unknown, which lets the caller keep the
        legacy ``llm`` default for a provider with no declared type.
        """
        from llm_rosetta.model_vocabulary import FORMATS_BY_TYPE

        from .model_types import all_model_types

        if wire_format in FORMATS_BY_TYPE["llm"]:
            return "llm"
        candidates = [
            model_type
            for model_type, formats in FORMATS_BY_TYPE.items()
            if wire_format in formats
        ]
        if not candidates:
            return None
        declared = {
            desc.name
            for desc in all_model_types()
            if not desc.is_llm
            and (
                (desc.config_format_key and cfg.get(desc.config_format_key))
                or (desc.config_shim_key and cfg.get(desc.config_shim_key))
            )
        }
        for candidate in candidates:
            if candidate in declared:
                return candidate
        return candidates[0]

    @classmethod
    def _endpoint_identity(
        cls,
        cfg: dict[str, Any],
        endpoint_types: list[tuple[str, str, str | None]],
        provider_shim_names: dict[str, str | None],
        name: str,
    ) -> tuple[str, str] | None:
        """Return the ``(model_type, format)`` a non-LLM endpoint declares.

        A populated ``<type>_format`` key declares the endpoint just as a
        ``<type>_shim`` does.  Reading only the shim keys left such a provider
        with the provider *name* as its format and an ``llm`` identity, which
        hid it from the admin picker and from the LLM-model validator.
        """
        from llm_rosetta.provider_names import canonical_provider_name

        for type_name, shim_key, format_key in endpoint_types:
            declared_format = (cfg.get(format_key) if format_key else "") or ""
            if not declared_format and not cfg.get(shim_key):
                continue
            shim = cls._resolve_endpoint_shim(
                cfg, type_name=type_name, shim_key=shim_key
            )
            if shim is None and not declared_format:
                continue
            if declared_format:
                wire_format = canonical_provider_name(declared_format)
            else:
                assert shim is not None
                wire_format = shim.format
            if shim is not None:
                # The LLM-side ProviderInfo still needs the shim's connection
                # defaults (base_url, key env) so an endpoint-only provider
                # loads without duplicating them in the config.
                provider_shim_names[name] = shim.name
            return type_name, wire_format
        return None

    @classmethod
    def _resolve_provider_identities(
        cls,
        raw_providers: dict[str, dict[str, str]],
    ) -> tuple[dict[str, ProviderIdentity], dict[str, str | None]]:
        """Resolve each provider's ``(model_type, format)`` identity.

        ``GatewayConfig.provider_types`` is derived from these values as the
        format, so the historical field keeps its meaning for existing
        consumers; read this mapping when the capability category matters.
        For the decision shims the format is the dialect (``systemone`` /
        ``openai_decisions``) rather than the category the old ``base`` field
        happened to carry.

        Resolution order per provider:
          1. ``shim`` field → the shim's pair
          2. ``type`` field → the shim's pair, or the format's category
          3. provider name itself (backward-compatible fallback)
          4. a non-LLM ``*_shim`` when the provider has no chat identity

        Step 4 matters for a provider configured only as an endpoint, e.g.
        ``{"embedding_shim": "jina--embedding"}``.  It never overrides a chat
        identity: a provider named ``openai`` that also declares an embedding
        shim stays an LLM provider and serves embeddings through the endpoint
        config.

        Legacy provider/format names are resolved to canonical spellings
        (see :func:`llm_rosetta.provider_names.canonical_provider_name`), and
        legacy shim names to their canonical ``<provider>--<base>`` spelling.
        A config declaring e.g. ``type: google`` is migrated to
        ``google_generate``; a ``shim:`` / ``type:`` naming a legacy shim (e.g.
        ``volcengine--openai_chat``) is migrated to ``volcengine--chat_completions``.
        Each emits one ``DeprecationWarning`` per distinct legacy name so
        operators migrate before 0.16.0 (#919).

        Returns:
            Tuple of (provider_identities, provider_shim_names).
        """
        from llm_rosetta.provider_names import (
            canonical_provider_name,
            warn_legacy_provider_names,
            warn_legacy_shim_names,
        )
        from llm_rosetta.shims import resolve_format, resolve_model_type

        from .model_types import all_model_types
        from .maintenance.names import correct_legacy_names, legacy_name_error

        # Migrate legacy names in place.  File-backed configs are also
        # corrected and written back by ``JsoncConfigIO``; this covers
        # programmatic configs.  A name that survives was one we could not map,
        # so reject it rather than resolving a legacy alias at runtime.
        correct_legacy_names({"providers": raw_providers})
        error = legacy_name_error(raw_providers)
        if error is not None:
            raise ValueError(error)

        endpoint_types = [
            (desc.name, desc.config_shim_key, desc.config_format_key)
            for desc in all_model_types()
            if not desc.is_llm and desc.config_shim_key
        ]

        identities: dict[str, ProviderIdentity] = {}
        provider_shim_names: dict[str, str | None] = {}
        for name, cfg in raw_providers.items():
            # ``shim:`` / ``type:`` values were already canonicalised above, so
            # no deprecation warning fires for them — the gateway *corrects*
            # the config (and a file-backed one is rewritten on disk), logging
            # each rewrite at INFO.  Only a legacy provider *key* still warns:
            # keys are user identifiers and are not rewritten.
            if "shim" in cfg:
                source = cfg["shim"]
                provider_shim_names[name] = source
            elif "type" in cfg:
                source = cfg["type"]
                provider_shim_names[name] = source
            else:
                warn_legacy_provider_names(name)
                if name not in BASE_FORMATS:
                    warn_legacy_shim_names(name)
                # Resolve through the shim registry too: a provider named after
                # a shim (e.g. ``openai--open_responses``, or the legacy
                # ``openai_responses`` alias of it) uses that shim's base and
                # keeps its identity, rather than being read as a bare format
                # name.
                source = name
                provider_shim_names[name] = name

            wire_format = canonical_provider_name(resolve_format(source))
            model_type = resolve_model_type(source)

            # A provider with no chat identity may be defined by a non-LLM
            # endpoint alone.  The provider-wide shim/type wins when it names
            # a category, so this only fills in the unknown case.
            if model_type is None:
                endpoint = cls._endpoint_identity(
                    cfg, endpoint_types, provider_shim_names, name
                )
                if endpoint is not None:
                    model_type, wire_format = endpoint

            if model_type is None:
                # A known non-chat format still names a category even when the
                # bare name is ambiguous.  Never turn that "I do not know the
                # chat format" into a confident ``llm`` identity.
                model_type = cls._identity_category(wire_format, cfg)

            identities[name] = ProviderIdentity(
                model_type=model_type or "llm", format=wire_format
            )
        return identities, provider_shim_names

    @classmethod
    def _parse_models(
        cls,
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
            route = cls._parse_model_route(name, value, raw_providers)
            if route is None:
                continue
            models[name] = route
            caps, upstream = cls._extract_model_metadata(value)
            model_capabilities[name] = caps
            if upstream:
                model_upstream_names[name] = upstream
        return models, model_capabilities, model_upstream_names

    @classmethod
    def _parse_model_route(
        cls,
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
            return cls._parse_multi_provider(name, value, raw_providers)
        provider_name = value["provider"]
        if provider_name not in raw_providers:
            return None
        return ModelRoute([ProviderEntry(provider_name)], name)

    @staticmethod
    def _declared_provider_names(value: Any) -> list[str]:
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

    @classmethod
    def _extract_model_metadata(cls, value: Any) -> tuple[list[str], str | None]:
        """Extract capabilities and upstream_model from a model entry."""
        if isinstance(value, str):
            return list(cls.DEFAULT_CAPABILITIES), None
        caps = value.get("capabilities", list(cls.DEFAULT_CAPABILITIES))
        return caps, value.get("upstream_model")

    @classmethod
    def _parse_multi_provider(
        cls,
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
            if pname not in raw_providers:
                logger.warning(
                    "Model %r: provider %r not found, skipping",
                    model_name,
                    pname,
                )
                continue
            upstream = item.get("upstream_model") if isinstance(item, dict) else None
            entries.append(ProviderEntry(pname, weight, upstream_model=upstream))

        if not entries:
            return None

        strategy_name = value.get("strategy", DEFAULT_STRATEGY)
        strategy = create_strategy(strategy_name)
        return ModelRoute(entries, model_name, strategy)

    def selection_context(self, model: str, source_format: str) -> SelectionContext:
        """Build one request's selection state, carrying this config's tables.

        This is the one place the two halves of selection are assembled: the
        context is created here, used to choose a provider before any pipeline
        exists, and carried on to the proxy, which fills in what it learns from
        the converted request.

        Args:
            model: Model name as specified by the client.
            source_format: API standard the request arrived in.  Legacy
                spellings normalise here.

        Returns:
            A context holding the model's route and this config's provider
            tables.

        Raises:
            KeyError: If the model is not in the routing table.
        """
        ctx = SelectionContext(model, source_format)
        ctx.install_route(
            model_route=self.models[model],
            providers=self.providers,
            provider_types=self.provider_types,
            provider_shim_names=self.provider_shim_names,
            provider_account_tokens=self.provider_account_tokens,
        )
        return ctx

    def resolve(self, ctx: SelectionContext) -> tuple[ResolvedRoute, ProviderInfo]:
        """Resolve *ctx* to a :class:`ResolvedRoute` and :class:`ProviderInfo`.

        Consolidates model lookup, provider type resolution, shim binding,
        capability detection, and reasoning overrides into a single typed
        result.

        Args:
            ctx: The request's selection state, as built by
                :meth:`selection_context`.  Its ``chosen`` provider is set —
                or left unset, which is reported as not-ready.

        Returns:
            ``(route, provider_info)`` — the route contains all
            pipeline-relevant fields; ``provider_info`` is the
            transport-level connection config.

        Raises:
            ProviderNotReady: If no provider for the model is ready.
        """
        from typing import cast

        from .maintenance.startup import ProviderInitState, ProviderNotReady

        model = ctx.model
        source_provider = cast(ProviderType, ctx.source_format)

        # The anchor is what routing would pick with the preference off, and the
        # full route's counter advances exactly as it would there.  Tier
        # selection can still move the served request onto another account
        # (tier 3), so what is preserved is the anchor selection, not the
        # per-account served share.
        self._selection.run(ctx)
        model_route = ctx.model_route
        assert model_route is not None  # install_route() put it there
        if ctx.chosen is None:
            deferred = getattr(self, "_deferred_startup", None)
            all_failed = deferred is not None and all(
                deferred.provider_state(p.name) == ProviderInitState.FAILED
                for p in model_route.providers
                if not self.providers[p.name].ready
            )
            raise ProviderNotReady(model, failed=all_failed)

        entry = ctx.chosen
        provider_name = entry.name
        provider_type = self.provider_types[provider_name]
        shim_name = self.provider_shim_names.get(provider_name)
        upstream_model = entry.upstream_model or self.model_upstream_names.get(model)
        caps = self.model_capabilities.get(model, list(self.DEFAULT_CAPABILITIES))
        reasoning = self.model_reasoning_overrides.get(model)
        flatten_system = self.model_flatten_system.get(model, False)
        from llm_rosetta.shims import resolve_shim

        _shim = resolve_shim(shim_name) if shim_name else None
        custom_tools = self.provider_supports_custom_tools.get(
            provider_name, _shim.tools.custom_tools if _shim else False
        )
        hoist_system = self.provider_hoist_system_messages.get(
            provider_name, _shim.hoist_system_messages if _shim else True
        )
        preflight = self.provider_preflight_token_count.get(provider_name, False)

        _mtdl_model = self.model_max_tool_description_length.get(model)
        _mtdl_prov = self.provider_max_tool_description_length.get(provider_name)
        _mtdl_shim = _shim.tools.max_description_length if _shim else None
        max_tool_desc = (
            _mtdl_model
            if _mtdl_model is not None
            else (_mtdl_prov if _mtdl_prov is not None else _mtdl_shim)
        )

        route = ResolvedRoute(
            source_provider=source_provider,
            target_provider=cast(ProviderType, provider_type),
            provider_name=provider_name,
            shim_name=shim_name,
            upstream_model=upstream_model,
            model_capabilities=caps,
            reasoning_override=reasoning,
            flatten_system=flatten_system,
            supports_custom_tools=custom_tools,
            hoist_system_messages=hoist_system,
            preflight_token_count=preflight,
            max_tool_description_length=max_tool_desc,
            soft_error_patterns=_shim.soft_error_patterns if _shim else (),
        )

        pinfo = self.providers[provider_name]

        # Per-model URL template override: create a shallow copy of the
        # provider info with modified template(s) so the transport layer
        # hits the right endpoint.
        model_url_tpl = self.model_url_templates.get(model)
        model_stream_tpl = self.model_stream_url_templates.get(model)
        if model_url_tpl or model_stream_tpl:
            pinfo = pinfo.with_url_templates(model_url_tpl, model_stream_tpl)

        model_timeout = self.model_timeouts.get(model)
        if model_timeout is not None:
            pinfo = pinfo.with_timeout(model_timeout)

        return route, pinfo

    # ---- Unified endpoint extraction ------------------------------------

    @staticmethod
    # ---- Typed (non-LLM) endpoint parsing --------------------------------

    @staticmethod
    def _parse_typed_config(
        raw_providers: dict[str, Any],
        raw_models: dict[str, Any],
        *,
        descriptor: ModelTypeDescriptor,
        format_key: str | None = None,
        require_declared_format: bool = False,
        map_models: bool = True,
        global_proxy: str | None = None,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, ProviderInfo], dict[str, str]]:
        """Parse one typed endpoint out of a provider section.

        The unified ``providers:`` entries and the legacy
        ``<type>_providers`` section both come through here, so embedding,
        rerank and decision cannot drift apart.  :meth:`_parse_typed_provider`
        documents the shim / format / path / connection precedence.

        Args:
            raw_providers: Provider entries to read.
            raw_models: Model entries to map onto the surviving providers.
            descriptor: The model type's registry descriptor; it supplies the
                category, the config keys, the default format and the default
                and per-format paths.
            format_key: Config key naming the endpoint format, when it is not
                the descriptor's (the legacy sections spell it ``format``).
            require_declared_format: Skip an entry that declares neither a
                format nor a shim — the unified pool's participation rule.
            map_models: Return the model→provider mapping; ``False`` when the
                caller distributes typed models separately.
            global_proxy: Server-level proxy URL.

        Returns:
            ``(providers, provider_infos, models)``.
        """
        keys = _TypedKeys.for_descriptor(descriptor, format_key)
        providers: dict[str, dict[str, Any]] = {}
        provider_infos: dict[str, ProviderInfo] = {}
        for name, cfg in raw_providers.items():
            parsed = GatewayConfig._parse_typed_provider(
                name,
                cfg,
                keys,
                descriptor=descriptor,
                require_declared_format=require_declared_format,
                global_proxy=global_proxy,
            )
            if parsed is not None:
                providers[name] = parsed[0]
                provider_infos[name] = parsed[1]

        models: dict[str, str] = {}
        if map_models:
            for model_name, value in raw_models.items():
                provider_name = GatewayConfig._first_pool_provider(value, providers)
                if provider_name:
                    models[model_name] = provider_name
        return providers, provider_infos, models

    @staticmethod
    def _parse_typed_provider(
        name: str,
        cfg: Any,
        keys: _TypedKeys,
        *,
        descriptor: ModelTypeDescriptor,
        require_declared_format: bool,
        global_proxy: str | None,
    ) -> tuple[dict[str, Any], ProviderInfo] | None:
        """Parse one provider entry, or ``None`` when it does not participate.

        Precedence:

        1. shim: *shim_key*, else the provider-wide ``shim`` / ``type`` when
           its category matches;
        2. format: an explicit *format_key* > the shim's format > the
           provider-wide ``type`` > *default_format*, canonicalised;
        3. path: an explicit *path_key* > the shim's ``default_path`` when the
           effective format is the shim's > the descriptor's per-format path >
           *default_path*;
        4. connection: a non-empty config value > the shim's default, and a
           per-provider ``proxy`` > the server proxy;
        5. auth: Google's header for a ``google_generate`` upstream, OpenAI
           Bearer otherwise.
        """
        if not isinstance(cfg, dict) or cfg.get("enabled") is False:
            return None
        shim = GatewayConfig._resolve_endpoint_shim(
            cfg, type_name=keys.type_name, shim_key=keys.shim_key
        )
        explicit_format = cfg.get(keys.format_key) or ""
        if require_declared_format and not explicit_format and shim is None:
            return None

        wire_format = GatewayConfig._typed_wire_format(cfg, keys, explicit_format, shim)
        path = GatewayConfig._typed_path(cfg, keys, descriptor, shim, wire_format)
        info = GatewayConfig._typed_provider_info(
            name, cfg, keys, wire_format, path, shim, global_proxy
        )
        return {"format": wire_format, keys.path_key: path}, info

    @staticmethod
    def _typed_wire_format(
        cfg: dict[str, Any], keys: _TypedKeys, explicit_format: str, shim: Any
    ) -> str:
        """The wire format an entry resolves to, canonicalised."""
        from llm_rosetta.provider_names import canonical_provider_name

        if explicit_format:
            return canonical_provider_name(explicit_format)
        if shim is not None:
            return shim.format
        return canonical_provider_name(cfg.get("type") or keys.default_format)

    @staticmethod
    def _typed_path(
        cfg: dict[str, Any],
        keys: _TypedKeys,
        descriptor: ModelTypeDescriptor,
        shim: Any,
        wire_format: str,
    ) -> str:
        """The upstream endpoint path an entry resolves to."""
        path = cfg.get(keys.path_key) or ""
        if not path and shim is not None and wire_format == shim.format:
            path = shim.default_path or ""
        if not path:
            path = descriptor.path_for(wire_format) or ""
        return path or keys.default_path

    @staticmethod
    def _typed_provider_info(
        name: str,
        cfg: dict[str, Any],
        keys: _TypedKeys,
        wire_format: str,
        path: str,
        shim: Any,
        global_proxy: str | None,
    ) -> ProviderInfo:
        """The connection a parsed entry forwards through."""
        from .providers import _apply_shim_defaults
        from .transport.provider_info import google_auth, openai_auth

        effective = _apply_shim_defaults(cfg, shim) if shim is not None else cfg
        return ProviderInfo(
            name=f"{keys.type_name}:{name}",
            api_key=effective.get("api_key", ""),
            base_url=effective.get("base_url", "").rstrip("/"),
            auth_header_fn=(
                google_auth if wire_format == "google_generate" else openai_auth
            ),
            url_template="{base_url}" + path,
            proxy_url=cfg.get("proxy") or global_proxy,
        )

    def typed_pool(
        self, type_name: str
    ) -> tuple[dict[str, dict[str, Any]], dict[str, ProviderInfo], dict[str, str]]:
        """The ``(providers, provider_infos, models)`` pools of *type_name*.

        The registry-driven counterpart of the per-type attributes, for a
        caller that only has the type's name.
        """
        return (
            getattr(self, f"{type_name}_providers"),
            getattr(self, f"{type_name}_provider_infos"),
            getattr(self, f"{type_name}_models"),
        )

    def resolve_typed(self, type_name: str, model: str) -> TypedRoute:
        """Resolve *model* to its provider config for *type_name*.

        The one implementation behind ``resolve_embedding``, ``resolve_rerank``
        and ``resolve_decision``; a newly registered type resolves through it
        without a new method.

        Raises:
            KeyError: If the model is not configured for that type.
        """
        from .model_types import get_model_type

        descriptor = get_model_type(type_name)
        path_key = (
            descriptor.config_path_key if descriptor is not None else None
        ) or f"{type_name}_path"
        models: dict[str, str] = getattr(self, f"{type_name}_models")
        providers: dict[str, dict[str, Any]] = getattr(self, f"{type_name}_providers")
        infos: dict[str, ProviderInfo] = getattr(self, f"{type_name}_provider_infos")
        provider_name = models[model]
        pcfg = providers[provider_name]
        return TypedRoute(
            provider_name=provider_name,
            format=pcfg["format"],
            path=pcfg[path_key],
            provider_info=infos[provider_name],
        )

    def resolve_embedding(self, model: str) -> TypedRoute:
        """Resolve an embedding model to its provider config.

        Raises:
            KeyError: If the model is not in embedding_models.
        """
        return self.resolve_typed("embedding", model)

    def resolve_rerank(self, model: str) -> TypedRoute:
        """Resolve a rerank model to its provider config.

        Raises:
            KeyError: If the model is not in rerank_models.
        """
        return self.resolve_typed("rerank", model)

    def resolve_decision(self, model: str) -> TypedRoute:
        """Resolve a decision model to its provider config.

        Raises:
            KeyError: If the model is not in decision_models.
        """
        return self.resolve_typed("decision", model)
