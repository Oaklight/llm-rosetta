"""The per-request selection state, and who the request is.

Selection happens at two points in a request's life that a pipeline sits
between: the provider is chosen *before* the request body has been converted,
and the upstream credential and session identity *after* it has been, because
the latter are keyed on the converted request.  The two cannot be one call, so
they share this object instead — created at the edge of the gateway, filled by
the stages as the request advances, and read by the transport.

Nothing here survives the request.  In particular the layer reads no history:
every value below is derived from the request in hand, which is what lets the
gateway run without a store.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from llm_rosetta.provider_names import canonical_provider_name

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..transport.provider_info import ProviderInfo
    from .strategy import ModelRoute, ProviderEntry


@dataclass(frozen=True, slots=True)
class Identity:
    """Who a request is, as far as selection is concerned.

    Filled in two steps, because the two halves become knowable at different
    times: the client is known once the request has been authenticated, the
    conversation only once the request body has been converted to the IR.

    Attributes:
        client: SHA-256 of the client's gateway API key, or ``None`` when the
            request carried no key at all (a keyless deployment, or auth
            switched off).  Stable for the life of that credential.
        conversation: A value that is the same on every turn of one
            conversation and different between conversations, or ``None`` when
            the request carries nothing to key a conversation on.  Derived
            from the first message, which appending history leaves alone, so
            it does not drift as the conversation grows.
    """

    client: str | None = None
    conversation: str | None = None


class SelectionContext:
    """One request's selection state, filled stage by stage.

    Attributes are set through the methods rather than assigned directly, so
    that the order the stages run in is the only way the object changes shape.
    """

    __slots__ = (
        # Fixed inputs, set when the context is created.
        "model",
        "source_format",
        # The config's view of the world, installed by the router.
        "model_route",
        "providers",
        "provider_types",
        "provider_shim_names",
        "provider_account_tokens",
        # Filled by the stages.
        "identity",
        "chosen",
        "provider_info",
        "ir_request",
        "quota_headroom",
        "quota_excluded",
        "failover_excluded",
        "pinned",
        "key_affinity",
        "upstream_session_id",
    )

    def __init__(self, model: str, source_format: str) -> None:
        """Create the context for one request.

        Args:
            model: The model name the client asked for.
            source_format: The API standard the request arrived in.  Legacy
                spellings are accepted and normalised here, so a request that
                names its format the old way is compared against the config's
                canonical names rather than against the string it sent.
        """
        self.model = model
        self.source_format = canonical_provider_name(str(source_format))

        self.model_route: ModelRoute | None = None
        self.providers: Mapping[str, ProviderInfo] = {}
        self.provider_types: Mapping[str, str] = {}
        self.provider_shim_names: Mapping[str, str | None] = {}
        self.provider_account_tokens: Mapping[str, str] = {}

        self.identity = Identity()
        self.chosen: ProviderEntry | None = None
        self.provider_info: ProviderInfo | None = None

        self.ir_request: dict[str, Any] | None = None
        # Smallest remaining share per provider, 0..1; absent means "not
        # measured", which is deliberately not the same as "spent".
        self.quota_headroom: Mapping[str, float] = {}
        # Providers whose every fresh reading says the allowance is spent.
        # Distinct from a provider block: this is a reading, not a decision.
        self.quota_excluded: frozenset[str] = frozenset()
        # Providers this request has already been refused by.  They are out
        # for the rest of *this* request only — a failed-over request must not
        # bounce back to the provider that just said no.
        self.failover_excluded: frozenset[str] = frozenset()
        # One provider this request must be served by, if the caller asked
        # for a specific one.  Every other candidate is excluded, so neither
        # the strategy nor failover can leave the pinned provider.
        self.pinned: str | None = None
        self.key_affinity = True
        self.upstream_session_id: str | None = None

    # -- the config's view -------------------------------------------------

    def install_route(
        self,
        *,
        model_route: ModelRoute,
        providers: Mapping[str, ProviderInfo],
        provider_types: Mapping[str, str],
        provider_shim_names: Mapping[str, str | None],
        provider_account_tokens: Mapping[str, str],
    ) -> None:
        """Hand the context the router's own tables.

        Passed in rather than looked up through a config back-reference: the
        stages then need nothing from the gateway's config object, and a test
        can drive them with plain dictionaries.
        """
        self.model_route = model_route
        self.providers = providers
        self.provider_types = provider_types
        self.provider_shim_names = provider_shim_names
        self.provider_account_tokens = provider_account_tokens

    # -- what the stages record -------------------------------------------

    def with_client(self, key_hash: str | None) -> None:
        """Record the authenticated client.

        Args:
            key_hash: The client key's hash, or ``None`` when there was no key.
        """
        self.identity = replace(self.identity, client=key_hash)

    def with_conversation(self, conversation: str | None) -> None:
        """Record the conversation the request belongs to.

        Args:
            conversation: The derived conversation value, or ``None`` when the
                request carried nothing to derive one from.
        """
        self.identity = replace(self.identity, conversation=conversation)

    def choose(self, entry: ProviderEntry) -> None:
        """Record the provider this request will be served by."""
        self.chosen = entry

    def with_provider_info(self, provider_info: ProviderInfo) -> None:
        """Record the transport configuration the request will be sent with."""
        self.provider_info = provider_info

    def with_quota_headroom(
        self,
        headroom: Mapping[str, float],
        excluded: frozenset[str] = frozenset(),
    ) -> None:
        """Record how much allowance each provider has left, and who has none."""
        self.quota_headroom = dict(headroom)
        self.quota_excluded = excluded

    def with_failover_excluded(self, names: frozenset[str] | set[str]) -> None:
        """Record providers this request must not be served by again.

        The previous choice is voided along with it: the caller is about to
        re-run selection, and a stage that finds nothing must not leave the
        refused provider standing.
        """
        self.failover_excluded = frozenset(names)
        self.chosen = None
        self.provider_info = None

    def with_pinned_provider(self, name: str | None) -> None:
        """Serve this request from *name* only.

        Used by the admin test path to reach one provider of a multi-provider
        model.  ``None`` clears the pin and restores ordinary routing.
        """
        self.pinned = name or None

    def excluded_providers(self) -> frozenset[str]:
        """Every provider selection must skip: spent readings plus failovers.

        A pin excludes every other candidate, so a pinned request cannot be
        served by a provider the caller did not name.
        """
        excluded = self.quota_excluded | self.failover_excluded
        if not self.pinned:
            return excluded
        return frozenset(
            {p.name for p in self.candidates if p.name != self.pinned} | excluded
        )

    def with_ir_request(self, ir_request: dict[str, Any]) -> None:
        """Record the converted request, once the pipeline has produced one.

        The stages that run after this read it for the fingerprints they key
        their decisions on; nothing before it can, which is why the two halves
        of selection cannot be one call.
        """
        self.ir_request = ir_request

    def with_key_affinity(self, enabled: bool) -> None:
        """Record whether the selected provider wants its keys pinned per client.

        A provider-level setting, so it is known only once a provider has been
        chosen.
        """
        self.key_affinity = enabled

    def with_upstream_session_id(self, session_id: str | None) -> None:
        """Record the session identity the upstream will be told."""
        self.upstream_session_id = session_id

    def is_ready(self, provider_name: str) -> bool:
        """Whether *provider_name* can take traffic right now."""
        return self.providers[provider_name].ready

    @property
    def candidates(self) -> list[ProviderEntry]:
        """Every provider the model is configured with, in order."""
        return list(self.model_route.providers) if self.model_route else []
