"""The stateless facts the admin panel reads: registries, constants, validators.

``admin/`` renders and validates against the gateway's model-type and
provider-type registries and a few constants.  Those are process-wide and need
no app, so they sit behind one object that ``create_app`` binds with
``admin_api.bind_catalogue`` instead of travelling through every helper.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import re


class GatewayCatalogue:
    """Answer the admin panel's questions about the gateway's registries."""

    def model_types(self) -> list[Any]:
        """Every registered model type descriptor, in registration order."""
        from ..model_types import all_model_types

        return all_model_types()

    def registered_type_names(self) -> list[str]:
        """The names of the registered model types."""
        from ..model_types import registered_type_names

        return registered_type_names()

    def retired_type_message(self, name: object) -> str | None:
        """The retired-names message when *name* is a retired model type."""
        from ..model_types import retired_type_message

        return retired_type_message(name)

    def entry_capabilities(self) -> frozenset[str]:
        """The capability names a ``models`` entry may declare."""
        from ..config.parse import ENTRY_CAPABILITIES

        return ENTRY_CAPABILITIES

    def declared_model_type(self, entry: Any, where: str | None = None) -> str:
        """The model type a ``models`` entry declares."""
        from ..admin_api import RetiredNameError
        from ..config.errors import RetiredConfigName
        from ..config.parse import declared_model_type

        try:
            return declared_model_type(entry, where)
        except RetiredConfigName as exc:
            raise RetiredNameError(exc.detail, where=exc.where, path=exc.path) from exc

    def face_value(self, entry: dict[str, Any], key: str, type_name: str) -> Any:
        """One endpoint setting: the face block first, then the flat key."""
        from ..config.typed import face_value

        return face_value(entry, key, type_name)

    def base_formats(self) -> tuple[str, ...]:
        """The base formats a provider may use, in picker order."""
        from ..config.providers import BASE_FORMATS

        return BASE_FORMATS

    def base_type_hints(self, provider_type: str) -> dict[str, str]:
        """The placeholder ``base_url`` / ``api_key_env`` hints of a base format."""
        from ..config.providers import get_base_type_hints

        return get_base_type_hints(provider_type)

    def default_supports_custom_tools(self, provider_type: str) -> bool:
        """Whether a base format supports custom tools by default."""
        from ..config.providers import get_default_supports_custom_tools

        return get_default_supports_custom_tools(provider_type)

    def known_provider_types(self) -> list[str]:
        """The built-in provider type names."""
        from ..config.providers import known_provider_types

        return known_provider_types()

    def default_models_path(self, provider_type: str) -> str:
        """The model-listing path appended to a provider's ``base_url``."""
        from ..config.providers import default_models_path

        return default_models_path(provider_type)

    def version_suffix_search(self, base_url: str) -> re.Match[str] | None:
        """Match a trailing ``/v1`` style version segment of *base_url*."""
        from ..transport.provider_info import VERSION_SUFFIXES

        return VERSION_SUFFIXES.search(base_url)

    def max_custom_css(self) -> int:
        """The most characters ``server.admin_custom_css`` may hold."""
        from ..config.settings import MAX_ADMIN_CUSTOM_CSS

        return MAX_ADMIN_CUSTOM_CSS

    def gateway_version(self) -> str:
        """The gateway distribution's version."""
        from .. import __version__

        return __version__

    def parse_quota(self, quota: str) -> dict[str, Any]:
        """Parse a rate quota such as ``"100/s"``; raises ``ValueError`` if bad."""
        from .._vendor.ratelimit import parse_quota

        return parse_quota(quota)

    def detect_routing_loops(
        self, models: Mapping[str, Any], upstream_names: Mapping[str, str]
    ) -> list[str]:
        """Warnings for models whose upstream names route back into the gateway."""
        from ..config import detect_routing_loops

        return detect_routing_loops(dict(models), dict(upstream_names))

    def embeddings_formats(self) -> list[str]:
        """The wire formats an ``embeddings`` provider may use."""
        from ..config import GatewayConfig

        return GatewayConfig.embeddings_formats()

    def rerank_formats(self) -> list[str]:
        """The wire formats a ``rerank`` provider may use."""
        from ..config import GatewayConfig

        return GatewayConfig.rerank_formats()

    def config_lock(self, path: str) -> AbstractAsyncContextManager[None]:
        """Hold the in-process and cross-process lock on the config file *path*."""
        from ..config import async_config_lock

        return async_config_lock(path)

    def provider_preflight_error(
        self, providers: Mapping[str, Any], data_dir: str | None = None
    ) -> str | None:
        """Build each provider in *providers* and return the first refusal."""
        from ..config.providers import build_provider_info, validate_provider_profiles
        from ..config.resolve import resolve_provider_identities

        enabled = dict(providers)
        try:
            identities, shim_names = resolve_provider_identities(enabled)
            built: dict[str, Any] = {}
            for name, cfg in enabled.items():
                identity = identities.get(name)
                if identity is None:  # pragma: no cover - resolver covers every entry
                    continue
                built[name] = build_provider_info(
                    identity.format,
                    cfg,
                    shim_name=shim_names.get(name),
                    config_name=name,
                    data_dir=data_dir,
                    # Every enabled provider is rebuilt on every save; its
                    # startup notices would otherwise repeat once per save.
                    quiet=True,
                )
            validate_provider_profiles(built)
        except Exception as exc:  # noqa: BLE001 - the message is the caller's to read
            return str(exc)
        return None

    def admin_cookie_name(self) -> str:
        """The name of the admin session cookie."""
        from ..middleware.auth import ADMIN_COOKIE_NAME

        return ADMIN_COOKIE_NAME

    def request_is_https(self, request: Any) -> bool:
        """Whether *request* arrived over HTTPS, as the trusted-proxy policy sees it."""
        from ..middleware.auth import request_is_https

        return request_is_https(request)

    def public_token_status(self, status: Any) -> dict[str, Any]:
        """A credential-minting provider's token status, without any secret."""
        from ..transport.token_refresh import public_token_status

        return public_token_status(status)

    async def refresh_credential(self, pinfo: Any, *, debounce: bool = True) -> bool:
        """Re-mint the credential of *pinfo*; ``False`` when it cannot be."""
        from ..transport.token_refresh import force_refresh

        return await force_refresh(pinfo, debounce=debounce)

    def token_pending_sentinel(self) -> str:
        """The key a credential-minting provider holds until its first mint."""
        from ..transport.provider_info import TOKEN_PENDING_SENTINEL

        return TOKEN_PENDING_SENTINEL

    def account_token_for(self, account_id: str) -> str | None:
        """The opaque token that names an account in the panel's URLs."""
        from ..config.providers import account_token_for

        return account_token_for(account_id)

    async def read_provider_quota(
        self, config: Any, name: str, *, force: bool = False
    ) -> dict[str, Any]:
        """Probe (or read the cached probe of) one provider's usage allowance."""
        from ..quota.service import read_provider_quota

        return await read_provider_quota(config, name, force=force)

    def provider_allowance(self, pinfo: Any) -> Any:
        """One provider's allowance as a yes/no and a comparable number."""
        from ..quota.service import provider_allowance

        return provider_allowance(pinfo)

    def provider_cooldown(self, name: str, pinfo: Any) -> dict[str, Any] | None:
        """A provider-wide block, or ``None``."""
        from ..quota.service import provider_cooldown

        return provider_cooldown(name, pinfo)

    def lane_eligibility(
        self, pinfo: Any, name: str | None = None, *, lane: str
    ) -> Any:
        """Whether one lane of *pinfo* may serve, and why not."""
        from ..routing import lane_eligibility

        return lane_eligibility(pinfo, name, lane=lane)

    def provider_eligibility(self, pinfo: Any, name: str | None = None) -> Any:
        """Whether *pinfo* may serve as a provider, and why not."""
        from ..routing import provider_eligibility

        return provider_eligibility(pinfo, name)

    def rate_limited_reason(self) -> str:
        """The eligibility reason code for a rate-limited lane."""
        from ..routing.eligibility import RATE_LIMITED

        return RATE_LIMITED

    def parse_model_routing(self, raw: Any, *, label: str) -> Any:
        """Parse one model's ``routing`` block; raises ``ValueError`` if invalid."""
        from ..routing import parse_model_routing

        return parse_model_routing(raw, label=label)

    def shims_signature(self, raw: Mapping[str, Any] | None) -> str:
        """A stable hash of the ``shims`` section of *raw*."""
        from ..shims.shim_sources import shims_signature

        return shims_signature(raw)

    def resolve_source_config(
        self, section: Mapping[str, Any] | None, config_path: str | None
    ) -> str:
        """Resolve a ``shims`` section into its cache root; raises if invalid."""
        from ..shims.shim_sources import resolve_source_config

        return resolve_source_config(section, config_path)

    def apply_shim_sources(
        self,
        raw: Mapping[str, Any],
        config_path: str | None = None,
        *,
        force_refresh: bool = False,
    ) -> Any:
        """Apply the declared sources and return the load report.  Blocking."""
        from ..shims.shim_sources import apply_shim_sources

        return apply_shim_sources(raw, config_path, force_refresh=force_refresh)

    def check_source_updates(self, report: Any, *, proxy: str | None = None) -> Any:
        """Compare each installed source with what it publishes.  Blocking."""
        from ..shims.sources import check_source_updates

        return check_source_updates(report, proxy=proxy)

    def http_client(self, **options: Any) -> Any:
        """An async HTTP client; use it as ``async with``."""
        from .._vendor import httpclient

        return httpclient.AsyncClient(**options)

    def is_http_response(self, value: Any) -> bool:
        """Whether *value* is a buffered response of :meth:`http_client`."""
        from .._vendor import httpclient

        return isinstance(value, httpclient.Response)

    def yaml_load(self, text: str) -> Any:
        """Parse a YAML document."""
        from .._vendor import yaml

        return yaml.load(text)

    def scrub_credential_patterns(self, text: str) -> str:
        """*text* with anything shaped like a credential removed."""
        from ..observability.sanitize import scrub_credential_patterns

        return scrub_credential_patterns(text)

    def decompress_error_dump_body(self, data: bytes) -> dict[str, Any]:
        """Decode one stored error-dump body."""
        from ..observability.error_dump import decompress_body

        return decompress_body(data)

    def ops_event_types(self) -> list[str]:
        """Every ops-log event type."""
        from ..observability.ops_log import ALL_EVENT_TYPES

        return ALL_EVENT_TYPES

    def ops_sources(self) -> list[str]:
        """Every ops-log source."""
        from ..observability.ops_log import ALL_SOURCES

        return ALL_SOURCES

    def internal_key_label(self) -> str:
        """The label under which the gateway's own requests are counted."""
        from ..keystore import INTERNAL_LABEL

        return INTERNAL_LABEL

    def internal_usage_bucket(self) -> str:
        """The usage subject of the gateway's own requests."""
        from ..keystore import INTERNAL_USAGE_BUCKET

        return INTERNAL_USAGE_BUCKET

    async def check_for_update(
        self, proxy: str | None = None, *, force: bool = False
    ) -> dict[str, Any]:
        """What PyPI offers and whether this process may update itself."""
        from .. import updates

        return await updates.check(proxy, force=force)

    def read_changelog(self, language: str) -> str:
        """The changelog bundled with this build, as markdown."""
        from .. import updates

        return updates.read_changelog(language)

    async def install_update(self, version: str, proxy: str | None) -> dict[str, Any]:
        """Install *version* with pip; ``{"ok", "output", "dists"}``."""
        from .. import updates

        return await updates.install(version, proxy)

    def restart_process(self) -> None:
        """Replace this process image with a fresh interpreter."""
        from .. import updates

        updates.restart_process()
