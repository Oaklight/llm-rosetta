"""Provider shim definitions with a global registry.

A **ProviderShim** is a lightweight identity card that declares which API
standard (converter) a provider uses, along with connection defaults and
optional transforms to bridge schema differences.

The global registry (``_SHIM_REGISTRY``) is a plain dict populated at
import time by ``shims/__init__.py``.  Registration functions
(``register_shim``, ``load_providers_from_dir``) write to it; query
functions (``get_shim``, ``list_shims``) read from it.
"""

from __future__ import annotations

import contextlib
import fnmatch
import functools
import logging
import re
import threading
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

from ..types.ir.configs import IREffort, IRMode  # re-exported
from llm_rosetta.model_vocabulary import FORMATS_BY_TYPE, MODEL_TYPES
from llm_rosetta.provider_names import (
    canonical_provider_name,
    canonical_shim_name,
)
from llm_rosetta.transforms import IRTransform, Transform

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reasoning capability config
# ---------------------------------------------------------------------------

# Provider-side types — strategy enums for shim behavior.

#: How outbound unsigned reasoning blocks are handled.
UnsignedBlocks = Literal["as_is", "preserve"]

#: What to do when a replayed history omits the reasoning of an assistant turn.
ReplayMissing = Literal["as_is", "fill_empty"]

#: Tool search capability mode.
ToolSearchMode = Literal["disabled", "native", "bridge"]

# Backward-compat aliases (deprecated — will be removed in a future version)
EffortLevel = IREffort
DisabledStrategy = Literal["omit", "thinking_disabled", "thinking_budget_zero"]
ThinkingType = Literal["enabled", "adaptive"]
UnsignedReasoningBlocks = UnsignedBlocks
EffortMap = dict[str, str]
EffortField = str


@dataclass(frozen=True)
class ReasoningCapability:
    """How a provider handles reasoning / thinking configuration.

    Each field group controls one dimension of reasoning behavior.
    IR-side values use the fixed vocabulary from ``types.ir.configs``;
    provider-side values are free strings specific to each upstream API.

    Reference: https://llm-rosetta.readthedocs.io/en/latest/api/reasoning/

    Naming convention:
    - ``_modes``: IR value → provider value mapping (dict)
    - ``_field``: provider-side field path (str)
    - ``_range``: IR-side constraint interval (tuple)
    """

    # ── Thinking toggle ──────────────────────────────────────────────
    # Maps IR mode → provider thinking type value.
    # None = provider does not support a thinking block.
    # Example: {"auto": "adaptive", "enabled": "enabled", "disabled": "disabled"}
    # IR modes not present in the map are silently dropped.
    thinking_modes: dict[str, str] | None = None

    # Default IR mode when the request has no explicit mode.
    # Must be a key in thinking_modes.
    thinking_default: IRMode | None = None

    # ── Effort ───────────────────────────────────────────────────────
    # Provider-side field path for the effort value.
    # "reasoning_effort"       → {reasoning_effort: v}
    # "reasoning.effort"       → {reasoning: {effort: v}}
    # "output_config.effort"   → {output_config: {effort: v}}
    # "thinking_level"         → {thinking_config: {thinking_level: v}}
    # "none"                   → provider does not accept effort
    effort_field: str = "reasoning_effort"

    # Supported IR effort range [floor, ceiling].
    # Values outside are clamped to the nearest boundary.
    # None = full IR ladder (minimal–max).
    effort_range: tuple[IREffort, IREffort] | None = None

    # ── Budget ───────────────────────────────────────────────────────
    # Derive budget_tokens as max(1024, int(max_tokens × ratio)),
    # clamped to max_tokens − 1.  None = no automatic derivation.
    budget_ratio: float | None = None

    # ── Visibility ───────────────────────────────────────────────────
    # Maps IR summary value → provider visibility value.
    # None = use converter default (hardcoded per API standard).
    # IR values not in the map → field is omitted from the request.
    # Example (Anthropic):  {"auto": "summarized", "none": "omitted"}
    # Example (OpenAI):     {"auto": "auto", "concise": "concise", "detailed": "detailed"}
    visibility_modes: dict[str, str] | None = None

    # ── Replay (request-side) ────────────────────────────────────────
    # Some thinking-mode upstreams reject a replayed history whose assistant
    # turns omit the reasoning field once tools are in play:
    #
    #   400 The `reasoning_content` in the thinking mode must be passed back
    #       to the API
    #
    # The requirement belongs to the *model*, not to one vendor: the same
    # DeepSeek model is fronted by many compatible endpoints, and a client
    # that replays history across a format boundary drops the reasoning for
    # some turns.  ``"fill_empty"`` declares the requirement and lets the
    # capability pass fill an empty value (accepted by the upstream), so the
    # request can still round-trip.  See #948 and epic #1143.
    replay_missing: ReplayMissing = "as_is"

    # ── Response handling ────────────────────────────────────────────
    # How to handle unsigned (non-redacted) reasoning blocks in responses.
    unsigned_blocks: UnsignedBlocks = "as_is"


# ---------------------------------------------------------------------------
# Grouped config objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConnectionConfig:
    """How to reach the upstream provider.

    Attributes:
        keyless: When ``True``, the upstream accepts requests without a
            credential.  A keyless provider sends no auth header as long as
            no API key is configured; supplying one switches it back to the
            base type's auth (Bearer for OpenAI-compatible upstreams), so a
            paid account can use the same shim.
        session_header: Name of a per-conversation session header the gateway
            injects on every inference request (see :attr:`session_header`).
    """

    base_url: str | None = None
    api_key_env: str | None = None
    models_path: str | None = None
    default_path: str | None = None
    """Upstream endpoint path this shim serves for its ``model_type``.

    Used when a non-LLM endpoint is configured through the shim but declares no
    explicit ``*_path``.  Chat shims leave it ``None``: their path comes from
    the chat format and URL template.
    """
    model_id_field: str | None = None
    auth_header: str | None = None
    keyless: bool = False
    session_header: str | None = None
    """Name of a per-conversation session header the gateway must add.

    Some upstreams (e.g. OpenCode Go) require a stable session identifier on
    every inference request for routing and prompt-cache locality.  When set,
    the gateway injects this header with a value derived per conversation
    (see ``gateway.proxy``); a client-supplied value is passed through
    verbatim.  ``None`` means the provider needs no session header.
    """
    headers: dict[str, str] | None = None
    """Static headers this upstream requires on every request.

    A client fingerprint (a specific ``User-Agent`` and a client marker) is a
    vendor fact, so it is declared here rather than in operator config.  The
    gateway merges these into every upstream request, including the models
    listing and the quota probe.
    """
    request_header: str | None = None
    """Name of a per-request identifier header the upstream validates.

    Unlike :attr:`session_header`, the value is minted for each upstream
    request.  ``request_id_format`` selects the encoding.
    """
    session_id_format: str | None = None
    """Encoding for the session id: ``"opencode"`` or ``None`` (sha256 hex)."""
    request_id_format: str | None = None
    """Encoding for the per-request id: ``"opencode"`` or ``None``."""
    stream_required: bool = False
    """Whether the upstream refuses non-streaming inference requests.

    If set, the gateway forces ``stream: true`` upstream and reassembles the
    SSE reply for a client that asked for a non-streaming response.
    """
    token_command: tuple[str, ...] | None = None
    """Command that mints the upstream credential, declared by the shim.

    Some upstreams do not accept an account API key on their inference
    endpoints — AskSage takes a short-lived access token, ALCF a Globus bearer
    token — and the exchange is a fact about the provider, not about each
    deployment.  A shim that declares the command here is usable without any
    ``token_command`` in the operator's config; a provider that names its own
    command still wins.

    A relative first argument is resolved against the shim's own directory, so
    a shim can ship the script beside its ``provider.yaml``.  A bare name that
    is not a file there (``python3``) stays a normal ``PATH`` lookup.
    """
    manage_usage_url: str | None = None
    """Where the operator changes this account's plan usage, if the vendor has
    a page for it.

    A plan's per-app cap lives in the vendor's own settings, not behind an
    API, so the admin panel can only link to it.  Declaring the URL here keeps
    the vendor fact in the shim rather than in the panel's JavaScript.
    """
    token_refresh_interval: int | None = None
    """Scheduled refresh interval for :attr:`token_command`, in seconds.

    ``None`` leaves the gateway's default (3600) in place.
    """
    token_command_env: str | None = None
    """Environment variable the command reads the operator's key from.

    Some exchanges need the account key as *input* (AskSage mints an access
    token from it), others read credentials of their own (ALCF's
    ``~/.globus``).  Declaring the variable name says which: the gateway passes
    the key the operator configured under that name, and the command's stdout —
    not the key — becomes the credential sent upstream.  It also lifts the
    ``api_key`` / ``token_command`` mutual exclusion for this shim, because the
    key is an input rather than a second credential.
    """


@dataclass(frozen=True)
class QuotaDeclaration:
    """A shim's declaration of where its allowance can be read.

    Attributes:
        probe: Identifier of a probe implemented in
            :mod:`llm_rosetta.shims.quota_probe`.  ``None`` lets the
            gateway resolve one from the base URL's host and lane.
        url: Explicit usage URL, for a vendor whose endpoint is not derivable
            from ``base_url``.
        interval: Background probe interval in seconds.  ``0`` (the default)
            reads on demand only.
        demote_at: Used share at which a window demotes the credential.
            ``100`` keeps the vendor's own exhaustion rule.
        in_band: Declare that this lane has **no** usage endpoint to read —
            its allowance is known only from what an upstream refusal says, and
            the probe is used for that classifier rather than for polling.
            A plan lane whose per-app cap lives in the vendor's own settings is
            the case: the app can be told the window is spent, and can never
            see how much of it is left.
    """

    probe: str | None = None
    url: str | None = None
    interval: int = 0
    demote_at: float = 100.0
    in_band: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return the non-``None`` fields as a plain dict."""
        out: dict[str, Any] = {"demote_at": self.demote_at, "interval": self.interval}
        if self.probe:
            out["probe"] = self.probe
        if self.url:
            out["url"] = self.url
        if self.in_band:
            out["in_band"] = True
        return out


@dataclass(frozen=True)
class ToolsConfig:
    """Tool handling behavior for this provider."""

    custom_tools: bool = False
    max_description_length: int | None = None
    search_mode: ToolSearchMode = "disabled"
    multimodal_result: bool | None = None
    tool_groups: bool = False
    """Whether the provider carries a named group of client-declared tools.

    A client may declare tools inside a named container and address them by
    ``(name, group)``.  The Responses family spells that
    ``{"type": "namespace", …}`` and returns ``namespace`` on the call; other
    formats have no grouping concept at all, and Anthropic and Google group
    only the tools *their own* MCP servers bring, not ones the client sent.

    ``False`` (the default) means the converter emits the flat list it always
    has, with group names folded into the tool names.  This is a per-provider
    declaration rather than something derivable from ``base``: OpenRouter's
    Chat Completions lane accepts the container while OpenAI's does not.
    """

    intrinsic_tools: tuple[str, ...] = ()
    """Intrinsic (provider-hosted) server tools this provider supports.

    Each entry is a canonical kind (``"web_search"``, ``"code_execution"``,
    ``"google_search"``, …).  Empty (the default) means no intrinsic tool
    support: a provider must opt in.  Declaring a kind here lets clients
    request it (by name in their tools JSON) and lets cross-format
    conversion map it to this provider's native server tool.
    """


# Legacy flat kwarg names → (grouped field, sub-field)
_LEGACY_CONNECTION_FIELDS: dict[str, str] = {
    "default_base_url": "base_url",
    "default_api_key_env": "api_key_env",
    "models_path": "models_path",
    "default_path": "default_path",
    "model_id_field": "model_id_field",
}
_LEGACY_TOOLS_FIELDS: dict[str, str] = {
    "supports_custom_tools": "custom_tools",
    "max_tool_description_length": "max_description_length",
    "tool_search_mode": "search_mode",
    "multimodal_tool_result": "multimodal_result",
}


# ---------------------------------------------------------------------------
# Soft-error detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SoftErrorPattern:
    """A pattern that detects upstream 200-but-error responses.

    When a provider returns HTTP 200 with an error embedded in the body
    (e.g. an auth warning instead of real LLM output), the gateway uses
    these patterns to detect and convert them to proper HTTP errors.

    Patterns are matched against the **full JSON-serialized** response
    body (``json.dumps(body)``), not just text content — JSON keys and
    structural tokens are included.  Write sufficiently specific regexes
    to avoid false positives against JSON plumbing.

    Attributes:
        pattern: Regex source string matched against the serialized
            response body (case-insensitive).
        status_code: HTTP status code to return to the client.
        message: Error message included in the client-facing response.
    """

    pattern: str
    status_code: int
    message: str

    @functools.cached_property  # works on frozen dataclasses (writes to __dict__ directly)
    def compiled(self) -> re.Pattern[str]:
        return re.compile(self.pattern, re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


def _normalize_format_kwargs(kwargs: dict[str, Any]) -> None:
    """Fold the deprecated ``base`` keyword into ``format``.

    ``format`` is the canonical spelling; ``base`` is its deprecated alias.
    Both are canonicalised, so the stored pair never carries a legacy spelling.
    """
    if "base" in kwargs:
        warnings.warn(
            "ProviderShim(base=...) is deprecated, use format=...",
            DeprecationWarning,
            stacklevel=3,
        )
        legacy_base = canonical_provider_name(kwargs.pop("base"))
        if "format" in kwargs:
            canonical = canonical_provider_name(kwargs["format"])
            if canonical != legacy_base:
                raise ValueError(
                    f"ProviderShim: base={legacy_base!r} disagrees with "
                    f"format={canonical!r}"
                )
            kwargs["format"] = canonical
        else:
            kwargs["format"] = legacy_base
    if "format" in kwargs:
        # Legacy spellings (openai_chat, anthropic, google) normalise to
        # their canonical format; the stored pair is always canonical.
        kwargs["format"] = canonical_provider_name(kwargs["format"])


@dataclass(frozen=True)
class ProviderShim:
    """Provider identity card with optional transforms.

    Attributes:
        name: Canonical provider identifier (e.g. ``"deepseek"``).
        model_type: Capability category this shim serves (``"llm"``,
            ``"embedding"``, ``"rerank"``, ``"decision"``).  A format name is
            scoped by this category, so the same spelling may name a chat
            format and an embedding format.
        format: Wire format within :attr:`model_type` (e.g.
            ``"chat_completions"``, ``"messages"``, ``"openai"``,
            ``"systemone"``).  The deprecated ``base`` keyword and property
            alias this field.
        logo: URL to the provider's logo image (SVG preferred).
        display_name: This shim's own human-readable label (e.g. ``"cluster_a"``
            on ``"acme_cluster_a--chat_completions"``).  The admin UI renders it
            as the shim's row inside its group — for a shim that also declares
            ``group`` or ``group_label``: one declaring neither was written
            against the older schema, where ``display_name`` titled the group
            instead, and that reading is kept (its row label still comes from
            ``name``).  Falls back to the segment after ``--`` in ``name``.  The
            vendor's own casing, unlike ``name``.
        group: Top-level group this shim appears under in the admin UI's
            provider-type picker.  Falls back to the segment before ``--`` in
            ``name``, so a shim usually needs no declaration; declare it when
            one vendor spans several name prefixes (``acme_cluster_a`` and
            ``acme_cluster_b`` both declaring ``"acme"``).
            The group's title — which is also the brand on free-source
            affordances — falls back to the capitalized group key, so a vendor
            that wants more than that declares ``group_label``.
        group_label: Title of the shim's top-level group.  Falls back to the
            group key.  Keep it identical across the members of one group; the
            last shim loaded wins, as with ``logo``.
        free_source: Marks the provider as a source with a free model pool.
            The admin UI groups its free (keyless) entry under a "Free
            Resource" section, adds a refresh action, and shows the
            third-party disclosure.
        connection: Upstream connection config (URL, API key env,
            models endpoint, model ID field).
        auth: Credential exchange the shim ships in ``auth.py``, for a vendor
            whose credential is not one argv away (an OAuth profile).  See
            :class:`~llm_rosetta.shims.auth_exchange.AuthExchange`.
        tools: Tool handling config (custom tools, description length,
            search mode, multimodal tool results).
        pre_ir_transforms: Body-level transforms applied BEFORE IR
            conversion (normalise provider dialect → standard).
            Aliased as ``from_transforms`` for backward compatibility.
        post_ir_transforms: Body-level transforms applied AFTER IR
            conversion (standard → provider dialect).
            Aliased as ``to_transforms`` for backward compatibility.
        reasoning: Reasoning capability config for this provider.
            When ``None``, the converter uses its built-in default.
        model_reasoning: Per-model reasoning overrides keyed by
            **upstream model ID** (post-alias).  Each entry inherits
            from the provider-level ``reasoning`` for unset fields.
        response_id_prefix: Provider-specific prefix for response IDs
            (e.g. ``"resp_"`` for OpenAI Responses, ``"chatcmpl-"`` for
            OpenAI Chat).  Default ``""`` means passthrough (no
            prefix stripping or adding).
        preserve_fields: Extra response/echo fields this provider adds to
            the base format's preserve set (e.g. the OpenAI-only lifecycle
            fields ``access_programs``, ``billing``, ``content_filters``,
            ``moderation``, ``tool_usage``).  Additive; the base format's
            spec-level set always applies.
        store_default: Value for the ``store`` field injected into a
            response resource when the upstream echo omits it.  ``None``
            (default) leaves the base format's spec behavior (no forced
            ``store``); ``True`` selects OpenAI's server-side-storage
            default.
        hoist_system_messages: Whether to hoist late system messages
            for prompt cache prefix stability.  Default ``True``.
    """

    name: str
    format: str
    model_type: str = "llm"
    logo: str | None = None
    display_name: str | None = None
    group: str | None = None
    group_label: str | None = None
    free_source: bool = False
    connection: ConnectionConfig = ConnectionConfig()
    quota: QuotaDeclaration | None = None
    quota_probes: tuple[Any, ...] = ()
    """Probe implementations the shim ships in ``quota.py``.

    Vendor facts live with the shim: the endpoint, the auth header, the body
    decoder and the error wording.  The gateway only holds the contract and
    the runtime state; it resolves the probe named by ``quota.probe`` against
    this tuple.
    """
    auth: Any | None = None
    """Credential exchange the shim ships in ``auth.py``.

    An exchange that does not fit in an argv (an OAuth profile with an
    interactive first run, a rotating refresh token and a stored record)
    implements :class:`~llm_rosetta.shims.auth_exchange.AuthExchange` and is
    attached here.  A provider-level ``token_command`` still wins; a shim
    declares this form or ``connection.token_command``, never both.
    """
    tools: ToolsConfig = ToolsConfig()
    pre_ir_transforms: tuple[Transform, ...] = ()
    post_ir_transforms: tuple[Transform, ...] = ()
    ir_transforms: tuple[IRTransform, ...] = ()
    response_body_transforms: tuple[Transform, ...] = ()
    reasoning: ReasoningCapability | None = None
    model_reasoning: dict[str, ReasoningCapability] | None = None
    response_id_prefix: str = ""
    preserve_fields: tuple[str, ...] = ()
    store_default: bool | None = None
    hoist_system_messages: bool = True
    soft_error_patterns: tuple[SoftErrorPattern, ...] = ()

    def __init__(self, **kwargs: Any) -> None:  # type: ignore[override]
        """Accept both new and legacy kwarg names.

        Legacy ``from_transforms`` maps to ``pre_ir_transforms``;
        ``to_transforms`` maps to ``post_ir_transforms``.
        Legacy flat connection/tool kwargs are merged into their
        respective grouped config objects.
        New names take precedence if both are provided.
        """
        # ── format / model_type ──────────────────────────────────────
        _normalize_format_kwargs(kwargs)

        # ── Transform renames ────────────────────────────────────────
        if "from_transforms" in kwargs:
            warnings.warn(
                "ProviderShim(from_transforms=...) is deprecated, "
                "use pre_ir_transforms instead",
                DeprecationWarning,
                stacklevel=2,
            )
            if "pre_ir_transforms" not in kwargs:
                kwargs["pre_ir_transforms"] = kwargs.pop("from_transforms")
            else:
                kwargs.pop("from_transforms")
        if "to_transforms" in kwargs:
            warnings.warn(
                "ProviderShim(to_transforms=...) is deprecated, "
                "use post_ir_transforms instead",
                DeprecationWarning,
                stacklevel=2,
            )
            if "post_ir_transforms" not in kwargs:
                kwargs["post_ir_transforms"] = kwargs.pop("to_transforms")
            else:
                kwargs.pop("to_transforms")

        # ── Legacy flat connection kwargs → ConnectionConfig ─────────
        conn_legacy = {
            new: kwargs.pop(old)
            for old, new in _LEGACY_CONNECTION_FIELDS.items()
            if old in kwargs
        }
        if conn_legacy:
            if "connection" not in kwargs:
                kwargs["connection"] = ConnectionConfig(**conn_legacy)
            else:
                warnings.warn(
                    "ProviderShim: flat connection kwargs "
                    f"({', '.join(conn_legacy)}) ignored because "
                    "'connection' was also provided",
                    DeprecationWarning,
                    stacklevel=2,
                )

        # ── Legacy flat tool kwargs → ToolsConfig ────────────────────
        tools_legacy = {
            new: kwargs.pop(old)
            for old, new in _LEGACY_TOOLS_FIELDS.items()
            if old in kwargs
        }
        if tools_legacy:
            if "tools" not in kwargs:
                kwargs["tools"] = ToolsConfig(**tools_legacy)
            else:
                warnings.warn(
                    "ProviderShim: flat tool kwargs "
                    f"({', '.join(tools_legacy)}) ignored because "
                    "'tools' was also provided",
                    DeprecationWarning,
                    stacklevel=2,
                )

        # ── Apply defaults ───────────────────────────────────────────
        _FIELD_DEFAULTS: dict[str, Any] = {
            "model_type": "llm",
            "logo": None,
            "display_name": None,
            "group": None,
            "group_label": None,
            "free_source": False,
            "connection": ConnectionConfig(),
            "quota": None,
            "quota_probes": (),
            "auth": None,
            "tools": ToolsConfig(),
            "pre_ir_transforms": (),
            "post_ir_transforms": (),
            "ir_transforms": (),
            "response_body_transforms": (),
            "reasoning": None,
            "model_reasoning": None,
            "response_id_prefix": "",
            "preserve_fields": (),
            "store_default": None,
            "hoist_system_messages": True,
            "soft_error_patterns": (),
        }
        _VALID_FIELDS = {"name", "format", "base"} | _FIELD_DEFAULTS.keys()
        for k, v in _FIELD_DEFAULTS.items():
            kwargs.setdefault(k, v)

        for required in ("name", "format"):
            if required not in kwargs:
                raise TypeError(
                    f"ProviderShim.__init__() missing required argument: {required!r}"
                )

        # Reject unknown kwargs (match frozen dataclass behavior)
        unknown = set(kwargs) - _VALID_FIELDS
        if unknown:
            raise TypeError(
                f"ProviderShim.__init__() got unexpected keyword argument(s): "
                f"{', '.join(sorted(unknown))}"
            )

        for k, v in kwargs.items():
            object.__setattr__(self, k, v)

    # ── Backward-compatible aliases (read-only) ─────────────────────

    @property
    def base(self) -> str:
        """Deprecated alias for :attr:`format`."""
        return self.format

    @property
    def from_transforms(self) -> tuple[Transform, ...]:
        """Alias for ``pre_ir_transforms`` (deprecated)."""
        return self.pre_ir_transforms

    @property
    def to_transforms(self) -> tuple[Transform, ...]:
        """Alias for ``post_ir_transforms`` (deprecated)."""
        return self.post_ir_transforms

    # Legacy flat connection accessors
    @property
    def default_base_url(self) -> str | None:
        """Alias for ``connection.base_url`` (deprecated)."""
        return self.connection.base_url

    @property
    def default_api_key_env(self) -> str | None:
        """Alias for ``connection.api_key_env`` (deprecated)."""
        return self.connection.api_key_env

    @property
    def models_path(self) -> str | None:
        """Alias for ``connection.models_path`` (deprecated)."""
        return self.connection.models_path

    @property
    def default_path(self) -> str | None:
        """Alias for ``connection.default_path``."""
        return self.connection.default_path

    @property
    def model_id_field(self) -> str | None:
        """Alias for ``connection.model_id_field`` (deprecated)."""
        return self.connection.model_id_field

    # Legacy flat tool accessors
    @property
    def supports_custom_tools(self) -> bool:
        """Alias for ``tools.custom_tools`` (deprecated)."""
        return self.tools.custom_tools

    @property
    def max_tool_description_length(self) -> int | None:
        """Alias for ``tools.max_description_length`` (deprecated)."""
        return self.tools.max_description_length

    @property
    def tool_search_mode(self) -> ToolSearchMode:
        """Alias for ``tools.search_mode`` (deprecated)."""
        return self.tools.search_mode

    @property
    def multimodal_tool_result(self) -> bool | None:
        """Alias for ``tools.multimodal_result`` (deprecated)."""
        return self.tools.multimodal_result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_SHIM_REGISTRY: dict[str, ProviderShim] = {}

# Per-thread view override for :func:`staging_registry`.  Set only on the
# thread that is building a replacement registry, so every other thread keeps
# reading ``_SHIM_REGISTRY`` and never observes a half-built one.
_registry_staging = threading.local()


def _current_registry() -> dict[str, ProviderShim]:
    """Return the registry *this thread* reads and writes.

    Normally the process-wide ``_SHIM_REGISTRY``; while the thread is inside
    :func:`staging_registry` it is that thread's private staging dict instead.
    """
    staged = getattr(_registry_staging, "registry", None)
    return staged if staged is not None else _SHIM_REGISTRY


@contextlib.contextmanager
def staging_registry() -> Iterator[dict[str, ProviderShim]]:
    """Build a replacement registry off to the side, then swap it in at once.

    Registration made inside the block lands in a private staging dict; other
    threads keep reading the live registry untouched, so they never see a
    partially-populated one.  On a clean exit the staging dict replaces the
    global in a **single reference assignment**; if the block raises, the live
    registry is left exactly as it was and the staging dict is discarded.
    """
    previous = getattr(_registry_staging, "registry", None)
    staging: dict[str, ProviderShim] = {}
    _registry_staging.registry = staging
    try:
        yield staging
    except BaseException:
        _registry_staging.registry = previous
        raise
    _registry_staging.registry = previous
    global _SHIM_REGISTRY
    _SHIM_REGISTRY = staging


# Chat formats and model-type names are not shims.  Non-chat format names are
# deliberately absent: ``openai``, ``cohere``, ``jina`` and ``voyage`` are also
# vendor names, so a bare name is only resolved within a model-type context.
_CHAT_FORMATS: frozenset[str] = frozenset(FORMATS_BY_TYPE["llm"])
_NON_SHIM_NAMES: frozenset[str] = _CHAT_FORMATS | frozenset(MODEL_TYPES)


def register_shim(shim: ProviderShim) -> None:
    """Register (or replace) a :class:`ProviderShim` in the global registry.

    The shim is keyed by its **exact** declared name.  Only canonical names are
    stored: :func:`get_shim` resolves legacy spellings on lookup (see
    :data:`llm_rosetta.provider_names.LEGACY_SHIM_ALIASES`), so the registry
    never holds an alias key that could shadow a real shim.

    If a shim with the same name is already registered, it is silently
    replaced and an INFO-level log is emitted.  This allows plugin shims
    to override built-in defaults without raising errors.

    Inside :func:`staging_registry` the write goes to that thread's staging
    dict instead of the live registry.
    """
    registry = _current_registry()
    key = shim.name
    if key in registry:
        logger.info("Shim %r overridden (format: %s)", key, shim.format)
    registry[key] = shim


def unregister_shim(name: str) -> ProviderShim | None:
    """Remove and return a shim by name.  Returns ``None`` if not found.

    The exact name is tried first; only if that misses is *name* resolved to
    its canonical *shim* spelling.  This internal lookup never warns.
    """
    registry = _current_registry()
    shim = registry.pop(name, None)
    if shim is None:
        shim = registry.pop(canonical_shim_name(name), None)
    return shim


def get_shim(name: str) -> ProviderShim | None:
    """Look up a registered :class:`ProviderShim` by *name*.

    The exact name is tried first, so a shim whose name is itself a legacy
    alias (a plugin case) is reachable under that name.  A **bare chat format**
    (``chat_completions``, ``open_responses``, ``messages``, ``google_generate``,
    ``google_interactions``) or a **model-type name** (``llm``, ``embedding``,
    ``rerank``, ``decision``) is never a shim: it resolves to ``None`` so a
    format name can never silently inherit a vendor shim's connection defaults
    (#917).  Any other name falls back to its canonical *shim* spelling
    (``LEGACY_SHIM_ALIASES``) and then to its canonical provider spelling, so a
    shim registered under its canonical ``<provider>--<format>`` name stays
    reachable via a legacy alias (e.g. ``"google"`` →
    ``"google--google_generate"``, ``"openai_responses"`` →
    ``"openai--open_responses"``).  This is an internal lookup and never warns.
    """
    registry = _current_registry()
    shim = registry.get(name)
    if shim is not None:
        return shim
    if name in _NON_SHIM_NAMES:
        return None
    legacy_shim = canonical_shim_name(name)
    if legacy_shim in _NON_SHIM_NAMES:
        return None
    shim = registry.get(legacy_shim)
    if shim is not None:
        return shim
    provider = canonical_provider_name(name)
    if provider in _NON_SHIM_NAMES:
        return None
    return registry.get(provider)


def resolve_shim(shim: ProviderShim | str | None) -> ProviderShim | None:
    """Resolve a shim argument to a :class:`ProviderShim` instance.

    Accepts a :class:`ProviderShim` (returned as-is), a registered name
    (looked up via :func:`get_shim`), or ``None`` (returns ``None``).
    """
    if shim is None:
        return None
    if isinstance(shim, ProviderShim):
        return shim
    return get_shim(shim)


def match_model_reasoning(
    shim: ProviderShim | str | None,
    model: str | None,
) -> ReasoningCapability | None:
    """Return the reasoning override declared for *model*, if any.

    ``model_overrides`` keys are matched against the **upstream model ID**.
    An exact key wins; otherwise the keys are tried in declaration order as
    shell-style patterns (:func:`fnmatch.fnmatchcase`), so a vendor can cover
    a model family with one entry (``deepseek*``) while an exact ID still
    overrides it.  ``fnmatch``'s ``*`` matches ``/`` too, so the same
    pattern covers vendor-prefixed IDs such as ``deepseek/deepseek-v3``.

    Args:
        shim: A shim object or registered name.
        model: The upstream model ID.  ``None`` or empty matches nothing.

    Returns:
        The matching per-model capability, or ``None`` when the model has no
        override (the caller then falls back to the provider-level one).
    """
    resolved = resolve_shim(shim)
    if resolved is None or not resolved.model_reasoning or not model:
        return None
    overrides = resolved.model_reasoning
    exact = overrides.get(model)
    if exact is not None:
        return exact
    for pattern, capability in overrides.items():
        if fnmatch.fnmatchcase(model, pattern):
            return capability
    return None


def list_shims() -> list[ProviderShim]:
    """Return all registered provider shims, in registration order.

    The registry holds canonical names only (see :func:`register_shim`), so
    each shim appears exactly once.
    """
    seen: set[int] = set()
    out: list[ProviderShim] = []
    for shim in _current_registry().values():
        if id(shim) in seen:
            continue
        seen.add(id(shim))
        out.append(shim)
    return out


def resolve_format(name: str) -> str:
    """Resolve a provider/shim *name* to its wire format.

    An exact registered shim name is tried first, so a shim whose name is also
    a legacy alias resolves to *its* declared format rather than being rewritten
    (the OpenAI Responses profile ``"openai_responses"`` now normalises to the
    ``"openai--open_responses"`` shim, whose format is ``"open_responses"``).
    Otherwise legacy *name* inputs are resolved to canonical spellings (see
    :func:`llm_rosetta.provider_names.canonical_provider_name`); this is an
    internal lookup and never warns.  A shim's stored format is already
    canonical.  If the normalised *name* is a known chat format or a model-type
    name, it is returned unchanged.  If the name is not found in either, it is
    returned as-is (caller decides how to handle unknown names).
    """
    registry = _current_registry()
    shim = registry.get(name)
    if shim is not None:
        return shim.format
    resolved = canonical_provider_name(name)
    if resolved in _NON_SHIM_NAMES:
        return resolved
    shim = registry.get(resolved)
    if shim is not None:
        return shim.format
    shim = registry.get(canonical_shim_name(name))
    if shim is not None:
        return shim.format
    return resolved


def resolve_model_type(name: str) -> str | None:
    """Resolve a provider/shim *name* to its capability category.

    A registered shim declares the category.  A bare chat format or a
    model-type name resolves without the registry.  A non-chat format name is
    ambiguous at this boundary (``openai`` is both a vendor and an embedding
    format), so it must come from a shim or from a caller that already knows the
    category; this function returns ``None`` for it.

    Returns:
        ``"llm"``, ``"embedding"``, ``"rerank"`` or ``"decision"``, or
        ``None`` when the name does not identify a category.
    """
    registry = _current_registry()
    shim = registry.get(name)
    if shim is None:
        shim = registry.get(canonical_shim_name(name))
    if shim is not None:
        return shim.model_type
    resolved = canonical_provider_name(name)
    if resolved in MODEL_TYPES:
        return resolved
    if resolved in _CHAT_FORMATS:
        return "llm"
    return None


def resolve_base(name: str) -> str:
    """Deprecated alias for :func:`resolve_format`."""
    warnings.warn(
        "resolve_base() is deprecated, use resolve_format()",
        DeprecationWarning,
        stacklevel=2,
    )
    return resolve_format(name)


def _clear_registry_caches() -> None:
    """Drop everything derived from the registry.

    Converter instances, convention model-list transforms and plugin transform
    namespaces all cache per-shim state, so they must not outlive a rebuild.
    """
    # Also clear the converter cache since shim resolution may have
    # cached converters for names that are now unregistered.
    from llm_rosetta.auto_detect import _converter_cache

    _converter_cache.clear()

    # Clear convention-based model list transforms.
    from llm_rosetta.shims.providers import _model_list_transforms

    _model_list_transforms.clear()

    # Drop the synthetic packages registered for plugin shim transforms.
    from llm_rosetta.shims.providers import _clear_plugin_namespaces

    _clear_plugin_namespaces()


def _reset_registry() -> None:
    """Clear the registry and everything derived from it.  Intended for testing."""
    _SHIM_REGISTRY.clear()
    _clear_registry_caches()
