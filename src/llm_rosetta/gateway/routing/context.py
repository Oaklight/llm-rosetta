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
        "key_affinity",
        "upstream_session_id",
    )

    def __init__(self, model: str, source_format: str) -> None:
        """Create the context for one request.

        Args:
            model: The model name the client asked for.
            source_format: The API standard the request arrived in, already
                canonical (``chat_completions``, not ``openai_chat``).
        """
        self.model = model
        self.source_format = source_format

        self.model_route: ModelRoute | None = None
        self.providers: Mapping[str, ProviderInfo] = {}
        self.provider_types: Mapping[str, str] = {}
        self.provider_shim_names: Mapping[str, str | None] = {}
        self.provider_account_tokens: Mapping[str, str] = {}

        self.identity = Identity()
        self.chosen: ProviderEntry | None = None
        self.provider_info: ProviderInfo | None = None

        self.ir_request: dict[str, Any] | None = None
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
