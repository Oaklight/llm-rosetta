"""The shim data classes: what a provider shim declares.

A **ProviderShim** is a lightweight identity card that declares which API
standard (converter) a provider uses, along with connection defaults and
optional transforms to bridge schema differences.  The classes here are the
values a shim author builds (in ``provider.yaml`` or in Python) and a host
reads; they hold no state.

The registry that stores shims, and the functions that read and write it
(``register_shim``, ``get_shim``, ``list_shims``, ...), are not part of this
contract: a host reaches them through :mod:`llm_rosetta.api`.  The one
exception is :func:`load_providers_from_dir`, the call that registers a shim
tree from a directory.

Every module under :mod:`llm_rosetta.contracts` is importable while
``llm_rosetta`` itself is still initialising: extensions load during
``import llm_rosetta``, so a sidecar that imports a contract
must not pull in :mod:`llm_rosetta.api` or a name defined by
``llm_rosetta/__init__.py``.

These classes lived in :mod:`llm_rosetta.shims.provider_shim` in 0.13.0; that
module re-exports them until 0.15.0.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable, Collection
from typing import TYPE_CHECKING, Any, Literal

from llm_rosetta import retired_names
from llm_rosetta.contracts.transform import IRTransform, Transform
from llm_rosetta.types.ir.configs import IREffort, IRMode
from llm_rosetta.types.ir.constraints import RequestConstraints


#: Request parameters a shim may declare for its own model type, as the IR
#: field names the target converter already writes.  A declared value is a
#: **default**: the gateway fills it only when the client omitted the field.
#: Only fields measured against the live endpoint belong here — a field the
#: upstream ignores is worse than no field.
TYPE_PARAM_KEYS: dict[str, frozenset[str]] = {
    "llm": frozenset(),
    "embeddings": frozenset(
        {"task_type", "dimensions", "encoding_format", "truncation"}
    ),
    "rerank": frozenset(
        {"top_n", "return_documents", "max_tokens_per_doc", "truncation"}
    ),
    "decisions": frozenset(),
}


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
    transport: str = "http"
    """How the gateway reaches this upstream.

    ``"http"`` (the default) posts a JSON body to a URL.  ``"acp"`` runs a
    local agent CLI and speaks the Agent Client Protocol to it over stdio —
    no URL is dialed, and the request body is the lane's own turn shape.  The
    gateway refuses a name it has no transport for at config load rather than
    on the first request.
    """
    acp_argv: tuple[str, ...] | None = None
    """The agent command an ``acp`` transport starts.

    The argv a vendor's CLI ships in (``["agy", "acp"]`` for Antigravity),
    declared by the shim because it is a vendor fact.  ``None`` takes the
    transport's own default.
    """
    acp_env: dict[str, str] | None = None
    """Environment overrides for the agent an ``acp`` transport starts.

    The gateway layers these over its own environment, so ``PATH`` and the
    agent's login store keep working.  A vendor declares a variable its CLI
    needs to expose tools correctly here (Claude Code needs
    ``ENABLE_TOOL_SEARCH=false`` for direct MCP tool calls).  Only read when
    ``transport`` is ``"acp"``.
    """
    acp_session_meta: dict[str, Any] | None = None
    """Extra ``_meta`` for the agent's ``session/new``, ``session/load`` and
    ``session/resume`` requests.

    Agent implementations differ in how they turn off their own tools, and
    some read a vendor-specific switch from the request's ``_meta`` rather
    than from a profile on disk.  The Claude adapter, for example, keeps
    Claude Code's file and shell tools unless ``session/new`` carries
    ``{"disableBuiltInTools": true}``; a resumed session must repeat the same
    value because the adapter includes it in its session fingerprint.  A
    vendor that already disables its tools through its own profile (for
    example through a ``--gemini-dir``) leaves this ``None``.  Only read when
    ``transport`` is ``"acp"``.
    """
    acp_env_unset: tuple[str, ...] = ()
    """Environment variable names removed from the agent's inherited set.

    The gateway copies its own environment so the agent keeps ``PATH`` and the
    login store, but a variable that redirects the vendor CLI would override
    the account the operator signed in with.  Claude Code, for example, bills
    an ``ANTHROPIC_API_KEY`` instead of the stored subscription and follows
    ``ANTHROPIC_BASE_URL``.  A shim names those variables here; the spawn and
    the admin sign-in both remove them before the vendor's own ``acp_env``
    values are applied.  Only read when ``transport`` is ``"acp"``.
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
    cli_argv: tuple[str, ...] | None = None
    """The model-client command a ``claude_cli`` transport starts.

    The lane reaches its model by running a vendor CLI that builds and signs
    the upstream request itself; the gateway never holds the credential.  The
    command is a vendor fact, so the shim declares it rather than the operator
    (``["claude"]``, or an absolute path when the binary is not on ``PATH``).
    ``None`` takes the transport's own default.  Only read when ``transport``
    is ``"claude_cli"``.
    """
    cli_env: dict[str, str] | None = None
    """Environment overrides for the process a ``claude_cli`` transport starts.

    Layered over the gateway's own environment, so ``PATH`` and the CLI's
    login store keep working.  A vendor declares here the variables its CLI
    needs for a correct, non-interactive lane (the caller name a subscription
    meters by, a switch that keeps the CLI from offering its own tools).  Only
    read when ``transport`` is ``"claude_cli"``.
    """
    cli_env_unset: tuple[str, ...] = ()
    """Environment variable names removed before :attr:`cli_env` is applied.

    The gateway copies its environment so the CLI keeps its login, but a
    variable that redirects it (``ANTHROPIC_API_KEY`` bills the API instead of
    the stored subscription; ``ANTHROPIC_BASE_URL`` points it at another host)
    would silently change what the lane does.  Removed from the child before
    the shim's own overrides are layered on.  Only read when ``transport`` is
    ``"claude_cli"``.
    """
    cli_args: tuple[str, ...] = ()
    """Extra arguments appended to the ``claude_cli`` command.

    The transport owns the flags that keep the CLI a model client (no tools,
    no session persistence, one turn).  A lane that needs one more — the
    ``--setting-sources user`` that lets a host's ``apiKeyHelper`` supply the
    credential, a ``--settings`` file, a ``--permission-prompts`` mode — names
    it here.  Appended after the transport's own flags, so a repeated option
    takes this value.  Only read when ``transport`` is ``"claude_cli"``.
    """
    cli_upstream: str | None = None
    """Where the relay forwards the CLI's request.

    ``None`` means Anthropic's own API.  An organization that fronts the API
    with its own gateway names that host here; the CLI's request is forwarded
    to it with the CLI's own authorization headers.  ``https://`` only, except
    for a loopback host (tests and a local gateway).  Only read when
    ``transport`` is ``"claude_cli"``.
    """


@dataclass(frozen=True)
class VendorManifest:
    """Vendor-level defaults shared by every shim in one directory.

    A provider's credential is a fact about the provider, not about each wire
    format it serves: ALCF mints one Globus token for three clusters, AskSage
    one access token for four dialects.  Declaring it once in
    ``<vendor>/provider.yaml`` keeps the copies from drifting.

    The file is optional and may declare any subset of
    :class:`ConnectionConfig`: a leaf overrides any field it names, so a
    vendor whose lanes differ (a different ``base_url`` per cluster) keeps the
    differing field on the leaf.  It may also declare the three fields that
    describe the *vendor* rather than a lane — ``logo``, ``group`` and
    ``group_label`` — on the same terms: a leaf that names one wins.  The
    manifest carries no shim identity: a file that declares ``name``,
    ``format`` or ``base`` is a shim, not a manifest.

    Attributes:
        vendor: Directory name the manifest sits in, for error messages.
        path: The ``provider.yaml`` itself.  A relative ``token_command``
            argument resolves against its directory.
        connection: Raw ``connection:`` mapping, before leaf merging.
        auth: The exchange the vendor ships in ``auth.py`` beside the
            manifest, attached to every leaf.
        logo: The vendor's mark, inherited by every leaf that names none.
        group: The vendor's top-level grouping key, inherited the same way.
            Without it the admin derives one from each shim's name, so a
            vendor whose shim names carry a deployment (``alcf_metis``)
            declares it once here.
        group_label: The vendor's group title, inherited the same way.
        fields: The provider-form inputs this shim declares, as data.  Each is
            a mapping with ``key`` and ``label`` and optionally
            ``placeholder``, ``help`` and ``required``; the gateway renders
            them beside the key and stores what the operator types in the
            entry's ``auth_fields``.  Declaring them here keeps a UI change a
            shim edit, never a gateway edit — and an exchange in ``auth.py``
            may declare the same inputs in code, which is merged with these by
            key.  A leaf that names a key it inherits from its vendor merges
            over that entry attribute by attribute, so rewording a label keeps
            the vendor's ``help``.
        quota: Raw ``quota:`` mapping, inherited by every leaf that names none
            and merged per field on a leaf that does.  An allowance is an
            account fact — one control plane, one workspace — so a vendor with
            several lanes declares it once here.
    """

    vendor: str
    path: Path
    connection: dict[str, Any] = field(default_factory=dict)
    auth: Any | None = None
    logo: str | None = None
    group: str | None = None
    group_label: str | None = None
    fields: tuple[dict[str, Any], ...] = ()
    quota: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuotaDeclaration:
    """A shim's declaration of where its allowance can be read.

    Attributes:
        probe: Identifier of a probe implemented in
            :mod:`llm_rosetta.contracts.quota`.  ``None`` lets the
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
        local_argv: Command that prints the lane's allowance as JSON.  A lane
            whose vendor CLI is the only way to read the account (an agent
            process, a subscription store) declares the command here instead
            of a URL.  The gateway runs it with the lane's profile, and the
            probe's ``parse_local`` decodes its stdout.  ``{acp_home}``,
            ``{acp_state_dir}`` and ``{acp_argv0}`` are expanded from the
            provider entry.
        local_env: Extra environment for :attr:`local_argv`.  The lane's own
            ``acp_env`` is applied first, so a declaration can override a
            vendor default for the probe alone.
        local_timeout: Seconds the command may run before it is stopped.
    """

    probe: str | None = None
    url: str | None = None
    interval: int = 0
    demote_at: float = 100.0
    in_band: bool = False
    local_argv: tuple[str, ...] = ()
    local_env: dict[str, str] = field(default_factory=dict)
    local_timeout: float = 30.0

    def as_dict(self) -> dict[str, Any]:
        """Return the non-``None`` fields as a plain dict."""
        out: dict[str, Any] = {"demote_at": self.demote_at, "interval": self.interval}
        if self.probe:
            out["probe"] = self.probe
        if self.url:
            out["url"] = self.url
        if self.in_band:
            out["in_band"] = True
        if self.local_argv:
            out["local"] = {
                "argv": list(self.local_argv),
                "env": dict(self.local_env),
                "timeout": self.local_timeout,
            }
        return out


@dataclass(frozen=True)
class BatchDeclaration:
    """A shim's declaration of its batch surface.

    Batch capability is a provider fact and cannot be inferred from the wire
    family: an OpenAI-shaped vendor is not automatically a batch vendor.  A
    shim that declares nothing has none — an undeclared lane answers
    ``unsupported`` rather than being downgraded to a gateway-run emulation.

    Attributes:
        dialect: Codec id from :data:`llm_rosetta.formats.batch.DECLARABLE_DIALECTS`
            (``openai_file``, ``anthropic_inline``, ``gemini_inline``,
            ``gemini_file``), or ``"emulated"`` for an upstream with no batch
            API that the gateway may run itself.  It must be one this shim's
            own ``(model_type, format)`` batch record offers, so a lane cannot
            name another format's container by mistake.
        supports_files: Whether the upstream's Files API is usable for uploads.
            OpenAI's batch surface is built on it; Anthropic's inline create is
            not.
        create_path: Path override for job creation, when the vendor serves it
            somewhere other than the dialect's documented path.  The same
            precedence applies to the three overrides below: the shim's value
            wins over the dialect's.
        retrieve_path: Path override for reading a job's state.
        cancel_path: Path override for cancelling a job.
        results_path: Path override for fetching results.
    """

    dialect: str
    supports_files: bool = False
    create_path: str | None = None
    retrieve_path: str | None = None
    cancel_path: str | None = None
    results_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return the fields as a plain dict, omitting the unset paths."""
        out: dict[str, Any] = {
            "dialect": self.dialect,
            "supports_files": self.supports_files,
        }
        for key in ("create_path", "retrieve_path", "cancel_path", "results_path"):
            value = getattr(self, key)
            if value:
                out[key] = value
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


#: Constructor keywords an earlier release accepted, and what replaced each.
#: Passing one raises :class:`TypeError` naming the replacement; the unknown-key
#: check alone would say only that the keyword is unexpected.
_REMOVED_KWARGS: dict[str, str] = {
    "base": "format",
    "from_transforms": "pre_ir_transforms",
    "to_transforms": "post_ir_transforms",
    "default_base_url": "connection=ConnectionConfig(base_url=...)",
    "default_api_key_env": "connection=ConnectionConfig(api_key_env=...)",
    "models_path": "connection=ConnectionConfig(models_path=...)",
    "default_path": "connection=ConnectionConfig(default_path=...)",
    "model_id_field": "connection=ConnectionConfig(model_id_field=...)",
    "supports_custom_tools": "tools=ToolsConfig(custom_tools=...)",
    "max_tool_description_length": "tools=ToolsConfig(max_description_length=...)",
    "tool_search_mode": "tools=ToolsConfig(search_mode=...)",
    "multimodal_tool_result": "tools=ToolsConfig(multimodal_result=...)",
}

#: Read-only attributes an earlier release offered, and what to read instead.
#: Reading one raises :class:`AttributeError` naming the replacement.
_REMOVED_ATTRIBUTES: dict[str, str] = {
    "base": "format",
    "default_path": "connection.default_path",
    "max_tool_description_length": "tools.max_description_length",
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


def _normalize_model_type_kwargs(kwargs: dict[str, Any]) -> None:
    """Reject a retired ``model_type`` spelling (``embedding``, ``decision``)."""
    if "model_type" in kwargs:
        retired_names.check_not_retired(kwargs["model_type"], retired_names.MODEL_TYPE)


def _normalize_format_kwargs(kwargs: dict[str, Any]) -> None:
    """Reject a retired format spelling.

    A retired spelling (``openai_chat``, ``anthropic``) raises a
    :class:`ValueError` that names its replacement.
    """
    if "format" in kwargs:
        # A retired spelling (openai_chat, anthropic, google) is an error that
        # names its replacement, not a name the shim could be stored under.
        retired_names.check_not_retired(kwargs["format"], retired_names.PROVIDER_TYPE)


@dataclass(frozen=True)
class ProviderShim:
    """Provider identity card with optional transforms.

    Attributes:
        name: Canonical provider identifier (e.g. ``"deepseek"``).
        model_type: Capability category this shim serves (``"llm"``,
            ``"embeddings"``, ``"rerank"``, ``"decisions"``).  A format name is
            scoped by this category, so the same spelling may name a chat
            format and an embedding format.
        format: Wire format within :attr:`model_type` (e.g.
            ``"chat_completions"``, ``"messages"``, ``"openai"``,
            ``"systemone"``).
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
        hint_msg: Note the admin panel shows beside the Base URL field when
            this shim is selected — the place to explain a configuration step
            the connection defaults cannot express (e.g. "the Zen lane uses a
            different base URL").  A mapping of language code to text
            (``{"en": ..., "zh": ...}``); a YAML string is stored as
            ``{"en": ...}``.  Plain text: the panel inserts it as text, never
            as HTML.  Display-only; no proxy behavior reads it.
        free_source: Marks the provider as a source with a free model pool.
            The admin UI groups its free (keyless) entry under a "Free
            Resource" section, adds a refresh action, and shows the
            third-party disclosure.
        connection: Upstream connection config (URL, API key env,
            models endpoint, model ID field).
        batch: This lane's batch surface (codec id, Files API support, path
            overrides).  ``None`` means none.  See :class:`BatchDeclaration`.
        auth: Credential exchange the shim ships in ``auth.py``, for a vendor
            whose credential is not one argv away (an OAuth profile).  See
            :class:`~llm_rosetta.contracts.auth.AuthExchange`.
        tools: Tool handling config (custom tools, description length,
            search mode, multimodal tool results).
        pre_ir_transforms: Body-level transforms applied BEFORE IR
            conversion (normalise provider dialect → standard).
        post_ir_transforms: Body-level transforms applied AFTER IR
            conversion (standard → provider dialect).
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
        request_fields: Request-bound fields this provider accepts beyond
            the base format's own set (e.g. the OpenAI-only ``logit_bias``,
            ``prompt_cache_options``, ``moderation``, ``access_programs``).
            A reader forwards a client's value for one of these only when the
            **target** declares it, so a neutral upstream never receives a
            field its spec does not define.
        default_capabilities: Capabilities (``text``, ``vision``, ``tools``,
            ``reasoning``) a model served by this shim has when its config
            entry declares none.  Empty (default) leaves the gateway default
            (``["text"]``); an explicit per-model ``capabilities`` list always
            wins.
        store_default: Value for the ``store`` field injected into a
            response resource when the upstream echo omits it.  ``None``
            (default) leaves the base format's spec behavior (no forced
            ``store``); ``True`` selects OpenAI's server-side-storage
            default.
        hoist_system_messages: Whether to hoist late system messages
            for prompt cache prefix stability.  Default ``True``.
        params: Defaults this shim declares for its own model type's request
            fields (an ``embeddings:``, ``rerank:`` or ``decisions:`` section in
            the YAML).  Keys are IR field names; the gateway fills a field the
            client omitted, so a declared value never overrides the caller.
            Chat requests do not use this field: what a chat dialect requires
            or restricts is declared in :attr:`request` (the ``request:``
            section), so a keyed-by-IR-path constraint never shares a spelling
            with a typed parameter name.
        request: What this vendor requires or restricts on the chat request,
            by IR path.  Overlaid on the converter's own dialect declaration
            per path; see
            :class:`~llm_rosetta.types.ir.constraints.RequestConstraints`.
    """

    name: str
    format: str
    model_type: str = "llm"
    logo: str | None = None
    display_name: str | None = None
    group: str | None = None
    group_label: str | None = None
    hint_msg: dict[str, str] | None = None
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
    batch: BatchDeclaration | None = None
    """This lane's batch surface, when it has one.

    ``None`` (the default) means the lane has no batch API this repository
    declares: a batch client asking for it answers ``unsupported``.  The
    declaration names a codec and path overrides only; the dialect's own
    limits, state words and item key come from the batch record in
    :mod:`llm_rosetta.formats.batch`.
    """
    auth: Any | None = None
    """Credential exchange the shim ships in ``auth.py``.

    An exchange that does not fit in an argv (an OAuth profile with an
    interactive first run, a rotating refresh token and a stored record)
    implements :class:`~llm_rosetta.contracts.auth.AuthExchange` and is
    attached here.  A provider-level ``token_command`` still wins; a shim
    declares this form or ``connection.token_command``, never both.
    """
    manifest: VendorManifest | None = None
    """Vendor manifest whose defaults this shim was merged with.

    ``None`` for a shim that stands alone (a flat leaf, or a vendor that ships
    no ``provider.yaml`` at the vendor level).  The admin panel reads
    :attr:`VendorManifest.path` to point an operator at the file that owns the
    shared connection defaults; the merged values themselves live in
    :attr:`connection`.
    """
    fields: tuple[dict[str, Any], ...] = ()
    """Provider-form inputs this shim declares in its ``provider.yaml``.

    Data, not code: the gateway renders one field per entry beside the key and
    stores the values in the entry's ``auth_fields``, so a UI change is a shim
    edit rather than a gateway edit.  A vendor manifest declares the inputs its
    lanes share and a leaf merges over any of them by key, attribute by
    attribute; an exchange in ``auth.py`` may declare more in code, merged by
    key at render time.
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
    request_fields: tuple[str, ...] = ()
    default_capabilities: tuple[str, ...] = ()
    store_default: bool | None = None
    hoist_system_messages: bool = True
    soft_error_patterns: tuple[SoftErrorPattern, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)
    request: RequestConstraints = field(default_factory=RequestConstraints)

    def __init__(self, **kwargs: Any) -> None:  # type: ignore[override]
        """Build a shim from keyword arguments.

        Every field is keyword-only.  A keyword an earlier release accepted
        (``base``, ``from_transforms``, ``to_transforms`` and the flat
        connection and tool keywords) is rejected with a :class:`TypeError`
        that names its replacement.

        Raises:
            TypeError: A removed keyword, an unknown keyword, or a missing
                ``name`` / ``format``.
        """
        removed = {k: _REMOVED_KWARGS[k] for k in kwargs if k in _REMOVED_KWARGS}
        if removed:
            hints = "; ".join(f"{k}= -> {new}" for k, new in sorted(removed.items()))
            raise TypeError(
                f"ProviderShim() no longer accepts {', '.join(sorted(removed))} "
                f"(removed in 0.14.0): {hints}"
            )

        # ── format / model_type ──────────────────────────────────────
        _normalize_format_kwargs(kwargs)
        _normalize_model_type_kwargs(kwargs)

        # ── Apply defaults ───────────────────────────────────────────
        _FIELD_DEFAULTS: dict[str, Any] = {
            "model_type": "llm",
            "logo": None,
            "display_name": None,
            "group": None,
            "group_label": None,
            "hint_msg": None,
            "free_source": False,
            "connection": ConnectionConfig(),
            "quota": None,
            "quota_probes": (),
            "batch": None,
            "auth": None,
            "manifest": None,
            "tools": ToolsConfig(),
            "pre_ir_transforms": (),
            "post_ir_transforms": (),
            "ir_transforms": (),
            "response_body_transforms": (),
            "reasoning": None,
            "model_reasoning": None,
            "response_id_prefix": "",
            "preserve_fields": (),
            "request_fields": (),
            "default_capabilities": (),
            "store_default": None,
            "hoist_system_messages": True,
            "soft_error_patterns": (),
            "params": {},
            "request": RequestConstraints(),
            "fields": (),
        }
        _VALID_FIELDS = {"name", "format"} | _FIELD_DEFAULTS.keys()
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

    if not TYPE_CHECKING:
        # Hidden from the type checker, so a misspelt attribute stays a
        # diagnostic instead of resolving through this hook.

        def __getattr__(self, name: str) -> Any:
            """Name the replacement of a removed read-only attribute."""
            replacement = _REMOVED_ATTRIBUTES.get(name)
            if replacement is None:
                raise AttributeError(
                    f"{type(self).__name__!r} object has no attribute {name!r}"
                )
            raise AttributeError(
                f"ProviderShim.{name} was removed in 0.14.0; read {replacement}"
            )


def load_providers_from_dir(
    providers_dir: Path,
    *,
    group: str | None = None,
    on_error: str = "raise",
    only_paths: Collection[str] | None = None,
    keep_names: Callable[[str], bool] | None = None,
) -> list[ProviderShim]:
    """Scan *providers_dir* for provider shims and register them.

    The call that registers a shim tree from a directory.  It delegates to the shim loader, imported on call, so this
    module stays importable while the loader itself is starting up.

    Args:
        providers_dir: The root of the plugin's shim tree
            (``<vendor>[/<lane>]/<type>/provider.yaml``).
        group: Group prefix for a tree nested under one vendor directory.
        on_error: ``"raise"`` (default) or ``"warn"`` for a broken directory.
        only_paths: Load only these directories (relative to
            *providers_dir*), when given.
        keep_names: Register only the shims whose name this predicate keeps.

    Returns:
        The shims this call registered.
    """
    from llm_rosetta.shims._loader import load_providers_from_dir as _load

    return _load(
        providers_dir,
        group=group,
        on_error=on_error,
        only_paths=only_paths,
        keep_names=keep_names,
    )


__all__ = [
    "TYPE_PARAM_KEYS",
    "BatchDeclaration",
    "ConnectionConfig",
    "ProviderShim",
    "QuotaDeclaration",
    "ReasoningCapability",
    "ReplayMissing",
    "SoftErrorPattern",
    "ToolSearchMode",
    "ToolsConfig",
    "UnsignedBlocks",
    "VendorManifest",
    "load_providers_from_dir",
]
