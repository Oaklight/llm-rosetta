"""Constants shared by the provider-shim loader modules."""

from __future__ import annotations

import dataclasses
import logging


from ..provider_shim import (
    ConnectionConfig,
)


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

#: Pre-``connection:`` spellings still read from a flat shim declaration.
_FLAT_CONNECTION_ALIASES: dict[str, str] = {
    "default_base_url": "base_url",
    "default_api_key_env": "api_key_env",
}

#: Keys that make a ``provider.yaml`` a shim rather than a vendor manifest.
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
