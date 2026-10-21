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
        token_env: Environment variable holding the forge token, for a private
            repository or mirror.  The repositories this project publishes are
            public, so the presets carry no token by default.
    """

    key: str
    label: str
    repo: str
    package: str
    shims: tuple[str, ...]
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
    ),
)
