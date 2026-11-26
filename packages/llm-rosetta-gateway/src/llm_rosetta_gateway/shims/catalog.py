"""Manifest sources, and the shims that stand-alone vendors provide.

A provider shim set that left core lives in a shim source repository that
publishes a versioned ``manifest.json`` (:data:`SHIM_SOURCES`: ``argonne`` and
``contrib``).  The gateway reads those manifests through ``shims.sources`` and
installs vendors through ``shims.install``; the Available view of the admin
Shims page lists what a source's manifest publishes.

:data:`STANDALONE_VENDORS` records the five vendor groups that the 0.14.x
``shims.presets`` keys named.  Two things still read it: the sentence a missing
standalone shim gets (:func:`source_package_advice`, and
:func:`legacy_shim_migration_advice` for an older spelling), which names the
``shims.install`` entry to declare, and ``llm-rosetta migrate``, which rewrites
an old ``shims.presets`` key into that entry.  The retired-names table
(``llm_rosetta.retired_names``) names the replacement of the older spellings;
a test here keeps the two lists in step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StandaloneVendor:
    """A vendor group a 0.14.x ``shims.presets`` key named.

    Attributes:
        key: The retired preset key (``"alcf"``).
        label: Human-readable name, for a message.
        source: The manifest source that publishes it.
        install: The ``shims.install`` selector that installs it.
        vendors: The vendors the selector installs.
        shims: Canonical names of the shims those vendors provide.
        repos: The repository URLs that publish the package.
    """

    key: str
    label: str
    source: str
    install: str
    vendors: tuple[str, ...]
    shims: tuple[str, ...]
    repos: tuple[str, ...]

    @property
    def repo_keys(self) -> set[str]:
        """The normalised form of every URL in :attr:`repos`."""
        return {repo_key(url) for url in self.repos}


_GH = "https://github.com/Oaklight/"

#: The five groups ``shims.presets`` named in 0.14.x.
STANDALONE_VENDORS: tuple[StandaloneVendor, ...] = (
    StandaloneVendor(
        "alcf",
        "ALCF",
        "argonne",
        "argonne:alcf",
        ("alcf",),
        (
            "alcf_metis--chat_completions",
            "alcf_minerva--chat_completions",
            "alcf_sophia--chat_completions",
        ),
        (_GH + "llm-rosetta-ext-argonne",),
    ),
    StandaloneVendor(
        "argo",
        "Argo",
        "argonne",
        "argonne:argo",
        ("argo",),
        ("argo--chat_completions", "argo--messages", "argo--open_responses"),
        (_GH + "llm-rosetta-ext-argonne",),
    ),
    StandaloneVendor(
        "free",
        "Free Endpoints",
        "contrib",
        "contrib:tag:tier:free",
        ("kilo_free", "llm7", "opencode_zen_free", "ovhcloud"),
        (
            "kilo_free--chat_completions",
            "llm7--chat_completions",
            "opencode_zen_free--chat_completions",
            "ovhcloud--chat_completions",
        ),
        (_GH + "llm-rosetta-ext-contrib",),
    ),
    StandaloneVendor(
        "asksage",
        "AskSage",
        "argonne",
        "argonne:asksage",
        ("asksage",),
        (
            "asksage--chat_completions",
            "asksage--messages",
            "asksage--open_responses",
            "asksage--google_generate",
            "asksage--embeddings",
        ),
        (_GH + "llm-rosetta-ext-argonne",),
    ),
    StandaloneVendor(
        "opendesign",
        "OpenDesign",
        "contrib",
        "contrib:opendesign",
        ("opendesign",),
        (
            "opendesign--chat_completions",
            "opendesign--messages",
            "opendesign--open_responses",
        ),
        (_GH + "llm-rosetta-ext-contrib",),
    ),
)


def get_standalone(key: str) -> StandaloneVendor | None:
    """The vendor group the retired preset *key* named, or ``None``."""
    for vendor in STANDALONE_VENDORS:
        if vendor.key == key:
            return vendor
    return None


def find_standalone(shim_name: str) -> StandaloneVendor | None:
    """The vendor group whose package provides *shim_name*, or ``None``."""
    for vendor in STANDALONE_VENDORS:
        if shim_name in vendor.shims:
            return vendor
    return None


def legacy_shim_migration_advice(shim_name: str | None) -> str:
    """The sentence to append when a config still spells a renamed shim.

    A shim that moved to a standalone package keeps its name, so a config that
    names it only needs the install declared; a shim that was *renamed* needs
    the canonical spelling too.

    Args:
        shim_name: The shim the provider names, if any.

    Returns:
        The sentence to append, or ``""`` when the name is not a recorded
        older spelling of a standalone vendor's shim.
    """
    if not shim_name:
        return ""
    from llm_rosetta.api import retired_names

    canonical = retired_names.replacement_for(shim_name, retired_names.SHIM)
    vendor = find_standalone(canonical) if canonical else None
    if vendor is None:
        return ""
    return (
        f" Use the canonical shim {canonical!r} with "
        f"shims.install: ['{vendor.install}'] and configure its connection "
        "defaults."
    )


def source_package_advice(shim_name: str | None) -> str:
    """The sentence to append when a shim belongs to a standalone package.

    Covers the name that is already canonical, where only the install is
    missing; :func:`legacy_shim_migration_advice` covers the older spelling.
    """
    vendor = find_standalone(shim_name) if shim_name else None
    if vendor is None:
        return ""
    return (
        f" The standalone package {vendor.label!r} provides it; declare "
        f"shims.install: ['{vendor.install}']."
    )


def repo_key(url: str | None) -> str:
    """Normalise a repository URL so two spellings of it compare equal.

    A declaration may carry a trailing slash or a ``.git`` suffix, and GitHub
    answers with or without them; the source list, the update check and the
    catalog all name the same repository, so they key on this form.

    Args:
        url: A repository URL, or ``None``.

    Returns:
        The lowercased URL without a trailing slash or ``.git`` suffix.
    """
    if not url:
        return ""
    out = url.strip().rstrip("/")
    if out.endswith(".git"):
        out = out[:-4]
    return out.lower()


@dataclass(frozen=True)
class ShimSource:
    """A shim source repository that publishes a versioned manifest (#1660).

    Attributes:
        key: The source id, as ``shims.sources`` and the ``<source>:`` prefix
            of a ``shims.install`` selector spell it.  It equals the
            manifest's ``source`` field.
        label: Human-readable name.
        repo: The repository URL.
    """

    key: str
    label: str
    repo: str

    @property
    def _owner_repo(self) -> tuple[str, str]:
        owner, name = self.repo.rstrip("/").split("/")[-2:]
        return owner, name

    @property
    def manifest_url(self) -> str:
        """The index URL: ``index.json`` of the repository's latest release."""
        owner, name = self._owner_repo
        return f"https://github.com/{owner}/{name}/releases/latest/download/index.json"

    @property
    def fallback_url(self) -> str | None:
        """The second index URL, or ``None``: a release asset has one home."""
        return None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "key": self.key,
            "label": self.label,
            "repo": self.repo,
            "manifest_url": self.manifest_url,
            "fallback_url": self.fallback_url,
        }


#: The source repositories that publish a manifest.  ``shims.sources`` names
#: them by key; a URL names any other source.  A vendor's mark and name come
#: from the manifest's release records (``logo`` / ``label``), not from here.
SHIM_SOURCES: tuple[ShimSource, ...] = (
    ShimSource(
        key="argonne",
        label="Argonne",
        repo="https://github.com/Oaklight/llm-rosetta-ext-argonne",
    ),
    ShimSource(
        key="contrib",
        label="Contrib",
        repo="https://github.com/Oaklight/llm-rosetta-ext-contrib",
    ),
)

#: The sources ``shims.install`` reads when ``shims.sources`` is not declared.
DEFAULT_SOURCES: tuple[str, ...] = ("contrib", "argonne")


def get_shim_source(key: str) -> ShimSource | None:
    """Return the manifest source with *key*, if any.

    Args:
        key: A source id, e.g. ``"contrib"``.

    Returns:
        The source, or ``None`` for an unknown key.
    """
    for source in SHIM_SOURCES:
        if source.key == key:
            return source
    return None
