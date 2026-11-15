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

from dataclasses import dataclass, field
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
        free: Whether the package serves a keyless free resource.  The panel
            groups these apart from the paid packages.
        subscription: Whether the package serves a plan the operator already
            pays for elsewhere (a ChatGPT plan, say) rather than metered API
            credit.  A separate axis from ``free``: it still needs a credential,
            just not a per-request one.  Nothing declares it yet.
        note: An optional sentence the operator should read before selecting
            the package, keyed by language code like a shim's ``hint_msg``.
            It travels with the preset so a warning cannot be left behind in
            the docs.
        ref: The tag the package's archives are published at, when it has one.
            Setting it pins the preset: the mirror's bytes are the ones built
            for that tag, so a primary that followed ``main`` could drift away
            from them.  ``None`` keeps the pre-archive behaviour (an omitted
            ref is resolved to a commit and pinned in the cache stamp).
        logos: The vendor marks the package's leaves declare, keyed by the
            vendor name (the shim name before ``--``).  The Available view
            draws them before the package is fetched, when there is no
            ``provider.yaml`` on disk to read a mark from; a vendor absent here
            falls back to the initials tile the Installed view uses.
        mirror: The ``shims.remotes[].mirror`` entry, once the package
            publishes a tagged archive.  ``sha256`` is a per-release constant:
            bump it here when the archive is rebuilt.  ``path`` is the
            directory to scan inside *that* archive, which differs from the
            repository's when the archive is built from the providers
            directory.
    """

    key: str
    label: str
    repo: str
    package: str
    shims: tuple[str, ...]
    legacy_names: tuple[str, ...] = ()
    token_env: str | None = None
    free: bool = False
    subscription: bool = False
    note: dict[str, str] | None = None
    ref: str | None = None
    mirror: dict[str, Any] | None = None
    logos: dict[str, str] = field(default_factory=dict)

    @property
    def tier(self) -> str:
        """The resource tier the panel filters and labels on.

        Returns:
            ``"subscription"`` when the package serves a plan, ``"free"`` when
            it serves a keyless pool, else ``"paid"``.
        """
        if self.subscription:
            return "subscription"
        return "free" if self.free else "paid"

    @property
    def remote(self) -> dict[str, Any]:
        """Return the ``shims.remotes`` entry that loads this package.

        Once the package publishes tagged archives the preset pins the tag
        (:attr:`ref`) and names the mirror those archives are served from.
        The two legs name the same version on purpose: the mirror's bytes are
        the ones built for that tag, so a primary that followed ``main`` could
        drift away from them.  Before the first tag an omitted ref is resolved
        to a concrete commit and pinned in the cache stamp, which is both
        reproducible and fetchable.

        The mirror's ``sha256`` is a per-release constant kept beside each
        entry — never computed at run time, which would make the preset depend
        on the network.  Its ``url_template`` is just a template: any host that
        serves the same archive works, so an internal mirror can replace the
        published CDN without touching the loader.
        """
        remote: dict[str, Any] = {
            "url": self.repo,
            "path": f"src/{self.package}/providers",
        }
        if self.ref:
            remote["ref"] = self.ref
        if self.mirror:
            remote["mirror"] = dict(self.mirror)
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
            "free": self.free,
            "tier": self.tier,
            "logos": dict(self.logos),
            "note": self.note,
        }


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
        # The archive lives on the orphan ``dist`` branch, one directory
        # per tag (#1412); the CDN URL is a template, not the only host.
        # ``sha256`` is the published digest of the v0.3.0 archive: bump it
        # together with ``ref`` when a new archive is released.
        # ``path`` names ``providers`` rather than ``"."``: the archive
        # strips its single top-level directory, and a future release that
        # wraps the providers tree must not stop this config loading.
        # The mark is the vendor manifest's, repeated so Available can draw
        # it before the package is fetched.
        logos={
            "alcf": (
                "https://cdn.jsdelivr.net/gh/Oaklight/llm-rosetta-shim-alcf"
                "@main/assets/alcf.svg"
            ),
        },
        ref="v0.3.1",
        mirror={
            "url_template": (
                "https://cdn.jsdelivr.net/gh/Oaklight/llm-rosetta-shim-alcf"
                "@dist/{ref}/alcf.tar.gz"
            ),
            "sha256": "2db618fb85826a2f71ba210ea37f3c471d24e05293fc1c9af310604e717c6b8b",
            "path": "providers",
        },
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
        # The archive lives on the orphan ``dist`` branch, one directory
        # per tag (#1412); the CDN URL is a template, not the only host.
        # ``sha256`` is the published digest of the v0.3.0 archive: bump it
        # together with ``ref`` when a new archive is released.
        # ``path`` names ``providers`` rather than ``"."``: the archive
        # strips its single top-level directory, and a future release that
        # wraps the providers tree must not stop this config loading.
        # The vendor's own wordmark, cut from the banner it publishes and
        # traced to a path (it is a 24px raster: no font to embed).
        logos={
            "argo": (
                "https://cdn.jsdelivr.net/gh/Oaklight/llm-rosetta-shim-argo"
                "@c1925c14f35170aa4f839c7fb8b97de58c2d4bee/assets/argo.svg"
            ),
        },
        ref="v0.3.0",
        mirror={
            "url_template": (
                "https://cdn.jsdelivr.net/gh/Oaklight/llm-rosetta-shim-argo"
                "@dist/{ref}/argo.tar.gz"
            ),
            "sha256": "caeeb7c07957df86600fdf573c448865a85fe87cca7a42e1d404c2ceea21c26e",
            "path": "providers",
        },
    ),
    ShimSourcePackage(
        key="free",
        # Every keyless pool, in one package: four vendors with no contract
        # behind them.  They live outside core so a pool that closes, or
        # tightens its rate limit, can be withdrawn or retuned without a
        # release.  This entry replaces the former ``opencode_zen_free`` one —
        # that repository shipped a single lane, was renamed, and now ships it
        # beside the others under an unchanged shim name.
        label="Free Endpoints",
        repo="https://github.com/Oaklight/llm-rosetta-shim-free",
        package="llm_rosetta_shim_free",
        shims=(
            "kilo_free--chat_completions",
            "llm7--chat_completions",
            "opencode_zen_free--chat_completions",
            "ovhcloud--chat_completions",
        ),
        # Keyless, so the picker groups them under free resources.
        free=True,
        # LLM7 declares no mark of its own, so its tile falls back to initials.
        logos={
            "kilo_free": (
                "https://cdn.jsdelivr.net/npm/@lobehub/icons-static-svg"
                "@latest/icons/kilocode.svg"
            ),
            "opencode_zen_free": (
                "https://cdn.jsdelivr.net/npm/@lobehub/icons-static-svg"
                "@latest/icons/opencode.svg"
            ),
            "ovhcloud": "https://cdn.jsdelivr.net/npm/simple-icons@latest/icons/ovh.svg",
        },
        note={
            "en": (
                "Anonymous free tiers with no contract: Kilo serves only its "
                "free slice, LLM7 only its turbo models, OVHcloud 2 requests "
                "per minute per IP, and OpenCode's gate fingerprints the "
                "client transport. Prompts may be logged by any of them."
            ),
            "zh": (
                "无合同的匿名免费额度：Kilo 只服务免费子集，LLM7 只服务 turbo 模型，"
                "OVHcloud 每 IP 每分钟 2 次请求，OpenCode 的门禁会识别客户端传输栈。"
                "任何一家的提示词都可能被上游记录。"
            ),
        },
        # The archive lives on the orphan ``dist`` branch, one directory per
        # tag; v0.1.0 there is the pre-rename OpenCode-only archive, kept so a
        # released gateway still resolves its old preset.
        ref="v0.3.0",
        mirror={
            "url_template": (
                "https://cdn.jsdelivr.net/gh/Oaklight/llm-rosetta-shim-free"
                "@dist/{ref}/free.tar.gz"
            ),
            "sha256": "397cc846adb14e0fb814dcd632057abb30480ff0ae7e281f4043f9c6f188e6fb",
            "path": "providers",
        },
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
            # The archive's embeddings lane (#1433).  It loads — the library
            # can convert with it — but a gateway embeddings provider on it is
            # auto-disabled by the static-key gate, because the leaf inherits
            # the vendor's token_command (#1401).  Listed anyway: the package
            # ships it, and the migration diagnostics must be able to name it.
            "asksage--embeddings",
        ),
        legacy_names=(
            "asksage--openai_chat",
            "asksage--anthropic",
            "asksage--openai_responses",
        ),
        # The vendor manifest carries the owl; the catalogue repeats it so the
        # Available card, and a loaded package whose archive predates the
        # declaration, both draw a mark instead of initials.
        logos={
            "asksage": (
                "https://cdn.jsdelivr.net/gh/Oaklight/llm-rosetta-shim-asksage"
                "@main/assets/asksage.png"
            ),
        },
        # The archive lives on the orphan ``dist`` branch, one directory
        # per tag (#1412); the CDN URL is a template, not the only host.
        # ``sha256`` is the published digest of the v0.3.0 archive: bump it
        # together with ``ref`` when a new archive is released.
        # ``path`` names ``providers`` rather than ``"."``: the archive
        # strips its single top-level directory, and a future release that
        # wraps the providers tree must not stop this config loading.
        ref="v0.3.0",
        mirror={
            "url_template": (
                "https://cdn.jsdelivr.net/gh/Oaklight/llm-rosetta-shim-asksage"
                "@dist/{ref}/asksage.tar.gz"
            ),
            "sha256": "0fe376890d57d0f1dea2684cd7af031e44319f9be7c83eb39d87ead7e34d7721",
            "path": "providers",
        },
    ),
    ShimSourcePackage(
        key="opendesign",
        label="Open Design Cloud",
        repo="https://github.com/Oaklight/llm-rosetta-shim-opendesign",
        package="llm_rosetta_shim_opendesign",
        shims=(
            "opendesign--chat_completions",
            "opendesign--messages",
            "opendesign--open_responses",
        ),
        # The vendor manifest carries the mark, so every lane shows it; the
        # package needs a core that reads manifest identity (0.14.0.dev31+).
        logos={
            "opendesign": (
                "https://cdn.jsdelivr.net/gh/Oaklight/llm-rosetta-shim-opendesign"
                "@main/assets/opendesign.svg"
            ),
        },
        # ``sha256`` is the published digest of the v0.1.1 archive: bump it
        # together with ``ref`` when a new archive is released.
        ref="v0.1.1",
        mirror={
            "url_template": (
                "https://cdn.jsdelivr.net/gh/Oaklight/llm-rosetta-shim-opendesign"
                "@dist/{ref}/opendesign.tar.gz"
            ),
            "sha256": "428d333d7e91ef3e2bfce9930f18b3c3e5b5a219064a3bdd1f7d83d38656ade8",
            "path": "providers",
        },
    ),
)
