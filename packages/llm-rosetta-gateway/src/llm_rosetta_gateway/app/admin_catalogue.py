"""The stateless facts the admin panel reads: registries, constants, validators.

``admin/`` renders and validates against the gateway's model-type and
provider-type registries and a few constants.  Those are process-wide and need
no app, so they sit behind one object that ``create_app`` binds with
``admin_api.bind_catalogue`` instead of travelling through every helper.
"""

from __future__ import annotations

from collections.abc import Mapping
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
