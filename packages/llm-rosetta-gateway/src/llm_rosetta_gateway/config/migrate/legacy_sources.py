"""The old ``manifest.json`` addresses of ``shims.sources`` (#1818).

Before the extension index (#1704) a source published a versioned
``manifest.json`` on the ``dist`` branch of its repository, and a config named
it by URL.  The gateway reads only ``index.json`` now.  This module is the one
place that still knows the old addresses, so ``llm-rosetta gateway migrate`` can
point a known one at the index of the same source and report one it cannot map.

Three hosts served the file, for the repository's ``dist`` branch or a
``dist/<commit>/`` directory of it:

- ``https://cdn.jsdelivr.net/gh/<owner>/<repo>@<ref>/.../manifest.json`` (the
  primary URL of the catalogue; ``fastly.jsdelivr.net`` is the same service),
- ``https://raw.githubusercontent.com/<owner>/<repo>/<ref>/.../manifest.json``
  (the catalogue's fallback URL),
- ``https://github.com/<owner>/<repo>/raw/<ref>/.../manifest.json``.

A repository that was renamed answers to its earlier names too.
"""

from __future__ import annotations

import posixpath
from urllib.parse import urlsplit

from ...shims.catalog import get_shim_source

__all__ = ["index_for", "is_old_manifest"]

#: The repositories that published a manifest, by name, with the source whose
#: index replaces each.  The names are the current ones and the earlier ones
#: (GitHub redirected them, so a config may hold either).
_REPO_SOURCES: dict[str, str] = {
    "llm-rosetta-ext-argonne": "argonne",
    "llm-rosetta-shim-argonne": "argonne",
    "llm-rosetta-shim-argo": "argonne",
    "llm-rosetta-ext-contrib": "contrib",
    "llm-rosetta-shim-contrib": "contrib",
    "llm-rosetta-shim-free": "contrib",
}

_OWNER = "oaklight"
_JSDELIVR = ("cdn.jsdelivr.net", "fastly.jsdelivr.net")


def _owner_repo(url: str) -> tuple[str, str] | None:
    """The ``(owner, repo)`` an old manifest URL addresses, or ``None``."""
    parts = urlsplit(url.strip())
    segments = [s for s in parts.path.split("/") if s]
    host = (parts.hostname or "").lower()
    if host in _JSDELIVR and len(segments) >= 3 and segments[0] == "gh":
        # ``/gh/<owner>/<repo>@<ref>/...``
        return segments[1], segments[2].split("@", 1)[0]
    if host == "raw.githubusercontent.com" and len(segments) >= 2:
        return segments[0], segments[1]
    if host == "github.com" and len(segments) >= 3 and segments[2] == "raw":
        return segments[0], segments[1]
    return None


def is_old_manifest(url: str) -> bool:
    """Whether *url* names a ``manifest.json`` file, which the gateway cannot read."""
    path = urlsplit(url.strip()).path
    return posixpath.basename(path).lower() == "manifest.json"


def index_for(url: str) -> str | None:
    """The ``index.json`` URL that replaces the old manifest *url*, or ``None``.

    ``None`` means *url* is not an old manifest, or it is one of a repository
    this tool does not know (a third party's), which only its owner can map.
    """
    if not is_old_manifest(url):
        return None
    owner_repo = _owner_repo(url)
    if owner_repo is None or owner_repo[0].lower() != _OWNER:
        return None
    key = _REPO_SOURCES.get(owner_repo[1].lower())
    source = get_shim_source(key) if key else None
    return source.manifest_url if source is not None else None
