"""Constants shared by the provider-shim loader modules."""

from __future__ import annotations

import dataclasses
import logging


from llm_rosetta.contracts.shim import ConnectionConfig


# The literal keeps log records under the pre-split name: the flat module used
# ``getLogger(__name__)``, which resolved to ``llm_rosetta.shims.providers``,
# and ``__name__`` here would be ``...providers._constants``.  Do not change it
# back to ``__name__``; log filters key on the old name.
logger = logging.getLogger("llm_rosetta.shims.providers")

#: The spelling follows the endpoint a type serves (``/v1/embeddings``,
#: ``/v1/rerank``, ``/v1/decisions``) so a path reads like the route it
#: configures; ``llm`` keeps its registry name because chat has no single
#: endpoint.  A plural type name cannot collide with a format name
#: (``embedding`` is a format; ``embeddings`` is not), so legacy layouts that
#: use the format name as a directory keep loading unchanged.
_TYPE_DIRS: dict[str, str] = {
    "llm": "llm",
    "embeddings": "embeddings",
    "rerank": "rerank",
    "decisions": "decisions",
}

#: How many lane levels a vendor may nest below its own directory.  A lane is
#: a named instance of one vendor — ALCF's ``metis``, ``minerva`` and
#: ``sophia`` are three clusters of one account — and it holds type
#: directories of its own.  The cap exists only so a mis-shaped tree cannot
#: recurse without end.
_MAX_LANE_DEPTH = 3

#: ``connection:`` keys a vendor manifest may declare, read off the config
#: object so the two lists cannot drift.
_CONNECTION_KEYS: tuple[str, ...] = tuple(
    f.name for f in dataclasses.fields(ConnectionConfig)
)

#: Top-level keys of a shim ``provider.yaml`` that an earlier release read and
#: the loader now refuses, each with the nested key that replaced it.  A
#: connection key written flat (``base_url:``, ``keyless:`` ...) is refused
#: the same way, and :func:`._parse._reject_removed_keys` builds that half of
#: the table from :data:`_CONNECTION_KEYS`.
_REMOVED_LEAF_KEYS: dict[str, str] = {
    "default_base_url": "connection.base_url",
    "default_api_key_env": "connection.api_key_env",
    "supports_custom_tools": "tools.custom_tools",
    "max_tool_description_length": "tools.max_description_length",
    "tool_search_mode": "tools.search_mode",
    "multimodal_tool_result": "tools.multimodal_result",
    "tool_groups": "tools.tool_groups",
    "intrinsic_tools": "tools.intrinsic_tools",
}

#: Keys that make a ``provider.yaml`` a shim rather than a vendor manifest.
#: ``base`` stays on the list although the loader no longer reads it: a file
#: that still declares it is a shim to be refused with the ``format:`` message,
#: not a vendor manifest to be refused for an unknown key.
_IDENTITY_KEYS: tuple[str, ...] = ("name", "format", "base")

#: Keys a vendor manifest may declare: the shared credential block, and the
#: three fields that describe the vendor itself rather than one lane.
_MANIFEST_KEYS: tuple[str, ...] = (
    "connection",
    "logo",
    "group",
    "group_label",
    "fields",
    "quota",
)
