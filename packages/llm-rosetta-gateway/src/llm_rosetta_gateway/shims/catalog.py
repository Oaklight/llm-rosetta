"""Standalone shim packages, and where the gateway finds them.

A provider shim set that has moved out of core lives in its own repository and
is consumed through the gateway's top-level ``shims`` section.  An operator
should not have to remember the repository URL and the directory inside it, so
this module is the single place that records both: the admin panel turns each
entry into a one-click preset, and the removal manifest reuses the same remote
for the shims it schedules to leave core.

The catalogue is independent of the removal schedule.  All three packages now
ship standalone: the panel offers each of them as a preset, and the config's
``shims.presets`` key resolves the same entries by name, so a provider config
that still spells ``argo--*`` (or ``alcf--*``, ``asksage--*``) keeps working
once the operator declares the preset.

It is also the record of *which* package replaces a name that no longer
resolves, including the older spellings of its shims: a config that still says
``alcf--sophia`` or ``asksage--openai_chat`` is told the canonical name and the
preset to declare, instead of only that the shim is unknown.  The core alias
table (``llm_rosetta.provider_names.LEGACY_SHIM_ALIASES``) canonicalises those
spellings; the two lists are kept in step by a test here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ShimSourcePackage:
    """A standalone shim package the gateway knows how to declare.

    Attributes:
        key: Short identifier, e.g. ``"alcf"``.
        label: Human-readable name for the admin picker.
        repo: Repository URL (the ``shims.remotes`` entry's ``url``).
        package: Import package name inside the repository; the shims live
            under ``src/<package>/providers``.
        shims: Canonical names of the shims the package provides.
        legacy_names: Older spellings of those shims, as they appeared before a
            rename.  They no longer resolve on their own; the canonical name is
            what a config should say, and the preset is what makes it loadable.
        token_env: Environment variable holding the forge token, for a private
            repository or mirror.  The repositories this project publishes are
            public, so the presets carry no token by default.
    """

    key: str
    label: str
    repo: str
    package: str
    shims: tuple[str, ...]
    legacy_names: tuple[str, ...] = ()
    token_env: str | None = None

    @property
    def remote(self) -> dict[str, Any]:
        """Return the ``shims.remotes`` entry that loads this package.

        No ``ref``: none of the repositories is tagged yet, and an omitted ref
        is resolved to a concrete commit and pinned in the cache stamp, which is
        both reproducible and fetchable today.  Pin a tag here once there is one.
        """
        remote: dict[str, Any] = {
            "url": self.repo,
            "path": f"src/{self.package}/providers",
        }
        if self.token_env:
            remote["token_env"] = self.token_env
        return remote

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view (the admin payload shape)."""
        return {
            "key": self.key,
            "label": self.label,
            "source_repo": self.repo,
            "remote": self.remote,
            "shims": list(self.shims),
        }


def get_source_package(key: str) -> ShimSourcePackage | None:
    """Return the standalone package with *key*, if any.

    This is the lookup ``shims.presets`` uses: a config names a package
    (``"argo"``) rather than spelling out its repository and path.

    Args:
        key: A catalogue key, e.g. ``"argo"``.

    Returns:
        The matching package, or ``None`` for an unknown key.
    """
    for package in SHIM_SOURCE_PACKAGES:
        if package.key == key:
            return package
    return None


def find_source_package(shim_name: str) -> ShimSourcePackage | None:
    """Return the standalone package that provides *shim_name*, if any.

    Used where a config names a shim that is no longer registered: the caller
    can then say which package supplies it and what to declare, instead of
    reporting an unknown provider type.

    Args:
        shim_name: A canonical shim name, e.g. ``"alcf_sophia--chat_completions"``.

    Returns:
        The matching package, or ``None`` when no standalone package provides
        that name.
    """
    for package in SHIM_SOURCE_PACKAGES:
        if shim_name in package.shims:
            return package
    return None


def find_legacy_source_package(shim_name: str) -> ShimSourcePackage | None:
    """Return the package whose shim *shim_name* used to name.

    A config written before a rename spells a shim that no longer resolves;
    the package that provides the canonical replacement is the one to name in
    the error, because declaring its preset is the whole fix.

    Args:
        shim_name: An older spelling, e.g. ``"alcf--sophia"``.

    Returns:
        The matching package, or ``None`` when no package records that
        spelling.
    """
    for package in SHIM_SOURCE_PACKAGES:
        if shim_name in package.legacy_names:
            return package
    return None


def legacy_shim_migration_advice(shim_name: str | None) -> str:
    """The sentence to append when a config still spells a renamed shim.

    A shim that moved to a standalone package keeps its name, so a config that
    names it only needs the preset declared; a shim that was *renamed* needs
    the canonical spelling too.  Both callers — provider validation, and the
    warning a provider with a base_url gets while it runs on generic defaults
    — want the same sentence, so it is written once, here.

    Args:
        shim_name: The shim the provider names, if any.

    Returns:
        The sentence to append, or ``""`` when the name is not a recorded
        older spelling.
    """
    if not shim_name:
        return ""
    package = find_legacy_source_package(shim_name)
    if package is None:
        return ""
    canonical = canonical_shim_for(package, shim_name)
    if canonical:
        return (
            f" Use the canonical shim {canonical!r} with "
            f"shims.presets: ['{package.key}'] and configure its connection defaults."
        )
    return (
        f" The standalone package {package.label!r} provides this shim; declare "
        f"shims.presets: ['{package.key}']."
    )


def source_package_advice(shim_name: str | None) -> str:
    """The sentence to append when a shim belongs to a standalone package.

    Covers the name that is already canonical, where only the preset is
    missing; :func:`legacy_shim_migration_advice` covers the older spelling.
    """
    if not shim_name:
        return ""
    package = find_source_package(shim_name)
    if package is None:
        return ""
    return (
        f" The standalone package {package.label!r} provides it; declare "
        f"shims.presets: ['{package.key}']."
    )


def canonical_shim_for(package: ShimSourcePackage, legacy_name: str) -> str:
    """Return the canonical shim of *package* that replaces *legacy_name*.

    Derived rather than hand-listed, so a rename that follows either pattern
    is answered without a second table: the shim whose name carries the same
    variant (``alcf--sophia`` → ``alcf_sophia--chat_completions``) or the same
    base after canonicalisation (``asksage--openai_chat`` →
    ``asksage--chat_completions``).

    Args:
        package: The package that records *legacy_name*.
        legacy_name: The older spelling.

    Returns:
        The canonical name, or ``""`` when nothing in the package matches it —
        which a test here treats as a broken catalogue entry.
    """
    from llm_rosetta.provider_names import canonical_shim_name

    canonical = canonical_shim_name(legacy_name)
    if canonical in package.shims:
        return canonical
    head, _, variant = legacy_name.partition("--")
    if variant:
        for shim in package.shims:
            if shim.startswith(f"{head}_{variant}--"):
                return shim
    return ""


#: The standalone packages this project publishes.  Adding an entry here makes
#: the Settings panel offer it as a preset; it does not imply a removal.
SHIM_SOURCE_PACKAGES: tuple[ShimSourcePackage, ...] = (
    ShimSourcePackage(
        key="alcf",
        label="ALCF",
        repo="https://github.com/Oaklight/llm-rosetta-shim-alcf",
        package="llm_rosetta_shim_alcf",
        shims=(
            "alcf_metis--chat_completions",
            "alcf_minerva--chat_completions",
            "alcf_sophia--chat_completions",
        ),
        # Cluster-qualified in #1035; the unqualified spellings still ship in
        # configs written for the 0.14.0 dev builds.
        legacy_names=("alcf--metis", "alcf--minerva", "alcf--sophia"),
    ),
    ShimSourcePackage(
        key="argo",
        label="Argo",
        repo="https://github.com/Oaklight/llm-rosetta-shim-argo",
        package="llm_rosetta_shim_argo",
        shims=(
            "argo--chat_completions",
            "argo--messages",
            "argo--open_responses",
        ),
        # The base renames (#917/#923): ``openai_chat`` → ``chat_completions``,
        # ``anthropic`` → ``messages``, ``openai_responses`` →
        # ``open_responses``.
        legacy_names=(
            "argo--openai_chat",
            "argo--anthropic",
            "argo--openai_responses",
        ),
    ),
    ShimSourcePackage(
        key="asksage",
        label="AskSage",
        repo="https://github.com/Oaklight/llm-rosetta-shim-asksage",
        package="llm_rosetta_shim_asksage",
        shims=(
            "asksage--chat_completions",
            "asksage--messages",
            "asksage--open_responses",
            "asksage--google_generate",
        ),
        legacy_names=(
            "asksage--openai_chat",
            "asksage--anthropic",
            "asksage--openai_responses",
        ),
    ),
)
