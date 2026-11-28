"""Resolving one provider entry's agent lane (#1830 A8).

``build_provider_info`` asks :func:`resolve_acp_lane` for everything the lane
needs beyond the credential and the URL: which transport carries it, the command
that starts the agent and whether the gateway hardens it, the environment and
options it starts with, the profile directory it signs in to (and whether the
gateway owns that directory), and the process and session limits.  The answer is
:class:`ResolvedLane`, one flat record whose fields are the ``ProviderInfo``
attributes of the same meaning; the provider registry reads it and nothing else
of the lane.

The functions here read a provider's config block and its shim.  They need no
credential, no URL and no HTTP, so the transport package owns them: the config
layer calls in and the transport never imports config.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from .profile import fallback_root, gemini_dir_values, is_agy, state_dir_for

logger = logging.getLogger("llm-rosetta-gateway")


@dataclass(frozen=True)
class ResolvedLane:
    """What one provider entry says about its lane, resolved and checked.

    The names follow the ``ProviderInfo`` attributes they become.

    Attributes:
        transport: The transport name that carries the lane (``http``, ``acp``,
            or one an extension provides).
        transport_owner: The extension that owns the transport, ``None`` for a
            built-in.
        acp_argv: The command that starts the agent; ``None`` leaves it to the
            transport's default.
        acp_argv_custom: Whether the entry names the command (so the shim's
            default is not in force).
        acp_harden: Whether the gateway gives the agent its own profile.
        acp_env: The environment the agent starts with, ``None`` when empty.
        acp_env_unset: Variables the agent's environment must not inherit.
        acp_session_meta: Metadata the shim sends when opening a session.
        transport_options: The shim's options overlaid by the entry's.
        acp_state_dir: The lane's persistent profile directory.
        managed_acp_state_dir: The same directory when the gateway derived it
            (and may move or remove it), else ``None``.
        limits: The ``ProviderInfo`` keyword arguments for the process and
            session limits the entry sets (``acp_processes``,
            ``acp_max_sessions``, ``acp_session_ttl``); empty when it sets none.
    """

    transport: str
    transport_owner: str | None
    acp_argv: tuple[str, ...] | None
    acp_argv_custom: bool
    acp_harden: bool
    acp_env: dict[str, str] | None
    acp_env_unset: tuple[str, ...]
    acp_session_meta: Any
    transport_options: dict[str, Any]
    acp_state_dir: str
    managed_acp_state_dir: str | None
    limits: dict[str, Any] = field(default_factory=dict)


def resolve_acp_lane(
    cfg: dict[str, Any],
    provider_key: str,
    shim: Any | None,
    data_dir: str | None,
) -> ResolvedLane:
    """Resolve one provider entry's lane from its config block and its shim.

    A config key wins over the shim's declaration throughout.  A transport name
    with nothing behind it is refused here, beside the provider it belongs to,
    rather than on the lane's first request.  An HTTP lane starts no agent, so
    it keeps its ``acp_env`` map as written and ignores the profile settings.

    Args:
        cfg: The provider's own config block (shim defaults already applied).
        provider_key: The config key, used in error messages.
        shim: The provider's shim, or ``None``.
        data_dir: The gateway's data directory, already resolved to an
            absolute path, or ``None``.

    Returns:
        The resolved lane.

    Raises:
        ValueError: A setting of the lane is malformed; the message names the
            provider and the setting.
    """
    transport, owner = _resolve_lane_transport(cfg, shim, provider_key)
    limits = _resolve_acp_limits(cfg, provider_key)
    # The command names an install; the home names the profile.  They are read
    # separately so that moving to another vendor CLI version (another argv)
    # keeps the login store, and so the transport knows which command it may
    # harden.  The effective environment is built once: the agent's spawn, the
    # vendor sign-in driver and the admin's status probe all name one profile.
    argv, harden = _resolve_acp_install(cfg, shim, provider_key, transport)
    home, env = _lane_environment(
        cfg, shim, provider_key, data_dir, transport, argv, harden
    )
    block = cfg.get("acp")
    return ResolvedLane(
        transport=transport,
        transport_owner=owner,
        acp_argv=argv,
        acp_argv_custom=(
            cfg.get("acp_argv") is not None
            or (isinstance(block, dict) and block.get("argv") is not None)
        ),
        acp_harden=harden,
        acp_env=env,
        acp_env_unset=(
            tuple(shim.connection.acp_env_unset) if shim is not None else ()
        ),
        acp_session_meta=(
            shim.connection.acp_session_meta if shim is not None else None
        ),
        transport_options=_transport_options(cfg, shim, provider_key, home),
        acp_state_dir=home,
        managed_acp_state_dir=(
            managed_acp_state_dir(provider_key, cfg, data_dir)
            if transport == "acp"
            else None
        ),
        limits=limits,
    )


def _acp_state_dir(
    data_dir: str | None, provider: str, cfg: dict[str, Any] | None = None
) -> str:
    """Return the persistent home for this provider's ACP lane.

    The default is one directory per configured provider,
    ``<data_dir>/acp/<slug>``, so two entries are two profiles, two vendor
    login stores and two processes.  Because it lives under the data
    directory, a deployment that mounts that directory keeps the vendor's
    login across restarts and across image upgrades.

    The ``acp.home`` provider setting names another directory.  An absolute
    path is used as written; a relative one is resolved under the data
    directory (or the per-user fallback when the gateway has no data
    directory).  A tilde is expanded.  A relative path may not contain ``..``:
    a profile that escapes the data directory silently stops surviving the
    image upgrade this home exists for, and an absolute path is the explicit
    way to point elsewhere.

    Raises:
        ValueError: ``acp`` is present but is not a mapping, ``acp.home`` is
            present but is not a non-empty path, a relative home climbs out
            of the data directory, or the named path is not a directory.
    """
    raw = (cfg or {}).get("acp")
    if raw is None:
        return state_dir_for(data_dir, provider)
    if not isinstance(raw, dict):
        raise ValueError(
            f"Provider '{provider}': acp must be a mapping of settings, got {raw!r}"
        )
    home = raw.get("home")
    if home is None:
        return state_dir_for(data_dir, provider)
    if not isinstance(home, str) or not home.strip():
        raise ValueError(
            f"Provider '{provider}': acp.home must be a non-empty path, got {home!r}"
        )
    path = os.path.expanduser(home.strip())
    if not os.path.isabs(path):
        if ".." in re.split(r"[\\/]", path):
            raise ValueError(
                f"Provider '{provider}': acp.home must stay under the data "
                f"directory; use an absolute path to point elsewhere, got {home!r}"
            )
        root = os.path.abspath(data_dir) if data_dir else str(fallback_root().parent)
        path = os.path.join(root, path)
    path = os.path.abspath(path)
    if os.path.exists(path) and not os.path.isdir(path):
        raise ValueError(f"Provider '{provider}': acp.home is not a directory: {path}")
    return path


def lane_profile_is_managed(provider: str, cfg: dict[str, Any]) -> bool:
    """Whether this entry's ACP home is derived by the gateway.

    A managed home (``<data>/acp/<slug>``) is the entry's own profile: the
    gateway may move it when the entry is renamed and remove it when the entry
    is deleted.  An ``acp.home`` or a ``--gemini-dir`` in the agent command is
    the operator's path; lifecycle actions never touch it.
    """
    acp = cfg.get("acp")
    if isinstance(acp, dict) and acp.get("home") is not None:
        return False
    argv = cfg.get("acp_argv")
    if argv is None and isinstance(acp, dict):
        argv = acp.get("argv")
    if isinstance(argv, (list, tuple)) and argv:
        try:
            if gemini_dir_values([str(item) for item in argv]):
                return False
        except (TypeError, ValueError):
            # A malformed command already fails the build.  Treat it as
            # operator-owned here so a cleanup never guesses a path.
            return False
    return True


def managed_acp_state_dir(
    provider: str, cfg: dict[str, Any], data_dir: str | None
) -> str | None:
    """The ACP profile the gateway owns for *provider*, or ``None``.

    ``None`` covers both "there is no managed profile" (an explicit
    ``acp.home`` or ``--gemini-dir``) and "there is no data directory to put
    one under".
    """
    if not data_dir or not lane_profile_is_managed(provider, cfg):
        return None
    return state_dir_for(data_dir, provider)


def _resolve_lane_transport(
    cfg: dict[str, Any], shim: Any | None, provider_key: str
) -> tuple[str, str | None]:
    """Return the transport that carries one provider's lane, and who owns it.

    A config key wins over the shim's declaration.  The name resolves from the
    shim's own scope (its extension's transports, then the gateway's, then the
    extensions it requires), so a name the shim cannot reach is refused here,
    naming the provider it belongs to, rather than on that provider's first
    request.  Resolution reads manifests only: no extension code is imported.

    Args:
        cfg: The provider's own config block.
        shim: The provider's shim, whose ``connection.transport`` supplies the
            default when the config names none.
        provider_key: The config key, used in the error message.

    Returns:
        ``(transport name, owning extension id or None for a built-in)``.

    Raises:
        ValueError: The name is not reachable from the shim's scope.
    """
    from llm_rosetta.contracts.extension import UnresolvedTransport

    from ..registry import reachable_transport_names, resolve_lane

    lane_transport = str(
        cfg.get("transport")
        or (shim.connection.transport if shim is not None else "http")
    )
    shim_name = shim.name if shim is not None else None
    try:
        lane = resolve_lane(lane_transport, shim_name)
    except UnresolvedTransport as exc:
        available = reachable_transport_names(shim_name)
        raise ValueError(
            f"Provider '{provider_key}': unknown transport {lane_transport!r}: "
            f"{exc}. Reachable from this shim: {', '.join(available)}."
        ) from exc
    return lane.name, lane.owner


def _argv_value(value: Any, provider_key: str, label: str) -> tuple[str, ...] | None:
    """Check a declared agent command and normalise it to a tuple.

    A command is a non-empty list of non-empty strings.  Refusing a typo here
    keeps it beside the provider that declared it, rather than surfacing as a
    start failure on the lane's first request.

    Raises:
        ValueError: The value is not a list/tuple of non-empty strings.
    """
    if value is None:
        return None
    if (
        not isinstance(value, (list, tuple))
        or not value
        or not all(isinstance(part, str) and part for part in value)
    ):
        raise ValueError(
            f"Provider '{provider_key}': {label} must be a non-empty list of "
            f"strings, got {value!r}"
        )
    return tuple(value)


def _resolve_acp_argv(
    cfg: dict[str, Any], shim: Any | None, provider_key: str
) -> tuple[str, ...] | None:
    """The command that starts this lane's agent, and the version it selects.

    Three sources, in order: ``acp.argv`` in the provider entry, the older
    top-level ``acp_argv``, then the shim's ``connection.acp_argv``.  The
    command names an *install*, not a profile: pointing it at another version
    of the vendor CLI leaves ``acp.home`` and the vendor's login store where
    they are, so an image upgrade or a version switch does not cost a sign-in.
    """
    block = cfg.get("acp")
    if isinstance(block, dict):
        declared = _argv_value(block.get("argv"), provider_key, "acp.argv")
        if declared is not None:
            return declared
    declared = _argv_value(cfg.get("acp_argv"), provider_key, "acp_argv")
    if declared is not None:
        return declared
    if shim is not None:
        return _argv_value(
            shim.connection.acp_argv, provider_key, "connection.acp_argv"
        )
    return None


def _expand_home(value: Any, home: str) -> Any:
    """Expand the ``{acp_home}`` placeholder in every string of a JSON value."""
    if isinstance(value, str):
        return value.replace("{acp_home}", home).replace("{acp_state_dir}", home)
    if isinstance(value, list):
        return [_expand_home(item, home) for item in value]
    if isinstance(value, dict):
        return {key: _expand_home(item, home) for key, item in value.items()}
    return value


def _transport_options(
    cfg: dict[str, Any],
    shim: Any | None,
    provider_key: str,
    home: str | None = None,
) -> dict[str, Any]:
    """One lane's options: the shim's, overlaid by the provider entry's.

    The gateway does not know any transport's option names.  The entry wins
    per key and a nested value is replaced whole.  Only the shape is checked
    here (an object); the transport named by the lane validates the content.

    *home* is the lane's persistent directory.  A string anywhere in the
    options may name it as ``{acp_home}`` (``{acp_state_dir}`` is kept as an
    alias), so a lane keeps its login in the gateway-owned profile.

    Args:
        cfg: The provider's own config block.
        shim: The provider's shim, whose ``connection.transport_options``
            supplies the defaults.
        provider_key: The config key, used in the error message.
        home: The lane's persistent directory, when it has one.

    Returns:
        The effective options; empty when neither side declares any.

    Raises:
        ValueError: The entry's ``transport_options`` is not an object.
    """
    options = dict(shim.connection.transport_options or {}) if shim is not None else {}
    override = cfg.get("transport_options")
    if override is not None:
        if not isinstance(override, dict):
            raise ValueError(
                f"Provider '{provider_key}': transport_options must be an "
                f"object, got {override!r}"
            )
        options.update(override)
    if home:
        options = _expand_home(options, home)
    return options


def _resolve_acp_harden(cfg: dict[str, Any], provider_key: str) -> bool | None:
    """The entry's explicit hardening choice, or ``None`` for "by name".

    The three states are deliberate.  Absent (``None``) hardens a command
    whose executable is named ``agy`` — the ordinary install.  ``True``
    forces the gateway-owned profile on whatever command the entry names, for
    a renamed binary (``agy-1.7``) or a wrapper that forwards agy's flags.
    ``False`` runs the command as written.

    The effective answer is :func:`_effective_harden`; this function only
    reads and validates the operator's declaration.

    Raises:
        ValueError: ``acp.harden`` is present but is not a boolean.
    """
    block = cfg.get("acp")
    if not isinstance(block, dict):
        return None
    value = block.get("harden")
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(
            f"Provider '{provider_key}': acp.harden must be true or false, "
            f"got {value!r}"
        )
    return value


def _effective_harden(declared: bool | None, argv: tuple[str, ...] | None) -> bool:
    """Whether the gateway gives this lane's agent its own profile.

    An absent ``acp.harden`` follows the command's name, so the shim's default
    ``agy`` command is hardened without the operator repeating it.  An
    explicit ``True`` forces the profile on a renamed or wrapped command; an
    explicit ``False`` runs it as written.
    """
    if declared is not None:
        return declared
    if argv is None:
        # An ACP lane with no command of its own runs the transport's default
        # agent, which is ``agy`` and therefore hardened.
        return True
    return is_agy(list(argv))


#: Provider/argv pairs already warned about.  ``build_provider_info`` runs on
#: the request path, so an unhardened wrapper must not log on every turn.
_NOT_HARDENED_WARNED: set[tuple[str, tuple[str, ...]]] = set()


def _warn_not_hardened(provider_key: str, argv: tuple[str, ...]) -> None:
    """Say once that a replaced agent command does not get the profile.

    A shim that starts ``agy`` declares the profile the lane expects.  When
    the entry replaces that command with something not named ``agy`` — a
    wrapper, a renamed download — the name-based default cannot tell it is
    the same agent, so the tool gate is silently absent.  The operator either
    forces the profile with ``acp.harden: true`` or accepts the command as
    written with ``false``.
    """
    key = (provider_key, tuple(argv))
    if key in _NOT_HARDENED_WARNED:
        return
    _NOT_HARDENED_WARNED.add(key)
    logger.warning(
        "Provider '%s': the agent command %r is not named 'agy', so the "
        "gateway-owned profile is not applied; set acp.harden: true to force "
        "it on this command, or false to accept it.",
        provider_key,
        list(argv),
    )


def _resolve_acp_install(
    cfg: dict[str, Any],
    shim: Any | None,
    provider_key: str,
    lane_transport: str,
) -> tuple[tuple[str, ...] | None, bool]:
    """Return the lane's install command and whether its profile is ours.

    Only an ACP lane has an install to resolve.  A shim that starts ``agy``
    but is overridden by a command of another name gets one warning: the
    name-based default cannot see that the unknown command is the same agent.
    """
    if lane_transport != "acp":
        return None, True
    argv = _resolve_acp_argv(cfg, shim, provider_key)
    declared = _resolve_acp_harden(cfg, provider_key)
    if (
        declared is None
        and shim is not None
        and shim.connection.acp_argv
        and argv is not None
    ):
        if is_agy(list(shim.connection.acp_argv)) and not is_agy(list(argv)):
            _warn_not_hardened(provider_key, argv)
    return argv, _effective_harden(declared, argv)


def _acp_env(
    cfg: dict[str, Any],
    shim: Any | None,
    provider_key: str,
    home: str | None = None,
    *,
    inject_home: bool = False,
) -> dict[str, str] | None:
    """Environment overrides for this provider's ACP agent process.

    The shim declares a vendor switch (Claude Code needs
    ``ENABLE_TOOL_SEARCH=false`` and its config directory); the provider entry
    may add deployment facts without repeating the vendor's.  The operator's
    values win.

    *home* is the lane's persistent directory.  A shim may name it as
    ``{acp_home}`` (``{acp_state_dir}`` is kept as an alias), which keeps a
    vendor's login in the gateway's data directory instead of the process's
    ``HOME``.  With *inject_home*, the gateway publishes it as
    ``LLM_ROSETTA_ACP_HOME`` for a shim that drives the vendor CLI itself.
    That variable is gateway-owned: an ``acp_env`` value is overwritten, so
    the agent and the sign-in driver cannot be split across two profiles.
    ``acp.home`` is the supported way to move the profile.
    """
    env = dict(shim.connection.acp_env or {}) if shim is not None else {}
    override = cfg.get("acp_env")
    if override is not None:
        if not isinstance(override, dict):
            raise ValueError(
                f"Provider '{provider_key}': acp_env must be a mapping of "
                f"environment names to strings, got {override!r}"
            )
        for name, value in override.items():
            if not isinstance(name, str) or not isinstance(value, str):
                raise ValueError(
                    f"Provider '{provider_key}': acp_env must map strings to "
                    f"strings; {name!r} has value {value!r}"
                )
        env.update(override)
    if home:
        env = {
            name: value.replace("{acp_home}", home).replace("{acp_state_dir}", home)
            for name, value in env.items()
        }
        if inject_home:
            env["LLM_ROSETTA_ACP_HOME"] = home
    return env or None


def _lane_environment(
    cfg: dict[str, Any],
    shim: Any | None,
    provider_key: str,
    data_dir: str | None,
    lane_transport: str,
    argv: tuple[str, ...] | None = None,
    harden: bool = False,
) -> tuple[str, dict[str, str] | None]:
    """Return the persistent home and the environment one lane starts with.

    An ACP lane gets a home under the data directory, its ``acp.home``
    override and the ``{acp_home}`` expansion.  A lane that speaks HTTP starts
    no agent, so it ignores those settings and keeps its ``acp_env`` map as
    written.

    An ``agy`` command may already name ``--gemini-dir``.  That directory is
    the profile the transport writes into, so it is also the home the sign-in
    and the environment must name.  The flag is validated here — one
    occurrence, a non-empty absolute path — because a malformed one would
    otherwise let hardening consume the next injected flag as its value.  When
    ``acp.home`` names a different directory, the two are refused rather than
    split silently across the agent and its sign-in.

    An unhardened ``agy`` command with no ``--gemini-dir`` is refused: the
    agent would read the CLI's own profile while the gateway publishes a home
    nobody signs in to.  Naming the directory, or leaving hardening on, keeps
    the contract true on every branch.
    """
    if lane_transport != "acp":
        return _acp_state_dir(data_dir, provider_key), _acp_env(cfg, shim, provider_key)
    home = _acp_state_dir(data_dir, provider_key, cfg)
    if argv:
        agy = is_agy(list(argv))
        named: str | None = None
        # Any command may already name the profile it reads, a wrapper
        # included.  The home the sign-in and the panel use must be that
        # directory whether or not the gateway injects the flag itself.
        values = gemini_dir_values(list(argv))
        if len(values) > 1:
            raise ValueError(
                f"Provider '{provider_key}': the agent command names "
                f"--gemini-dir more than once ({values!r}); keep one "
                f"profile"
            )
        if values:
            named = values[0]
            if not named or named.startswith("-"):
                raise ValueError(
                    f"Provider '{provider_key}': --gemini-dir in the agent "
                    f"command must be followed by a directory, not a flag "
                    f"or nothing"
                )
        if agy and not harden and named is None:
            raise ValueError(
                f"Provider '{provider_key}': acp.harden: false runs agy as "
                f"written, so the profile it reads is not knowable here. Name "
                f"it with --gemini-dir <dir>, or leave hardening on."
            )
        if named is not None:
            expanded = os.path.expanduser(named)
            if not os.path.isabs(expanded):
                raise ValueError(
                    f"Provider '{provider_key}': the --gemini-dir in the agent "
                    f"command must be an absolute path (or move the profile "
                    f"with acp.home), got {named!r}"
                )
            resolved = os.path.abspath(expanded)
            block = cfg.get("acp")
            explicit = isinstance(block, dict) and block.get("home") is not None
            if explicit and home != resolved:
                raise ValueError(
                    f"Provider '{provider_key}': acp.home and the "
                    f"--gemini-dir in the agent command name different "
                    f"directories ({home!r} and {resolved!r}); keep one, or "
                    f"make them equal"
                )
            home = resolved
    return home, _acp_env(cfg, shim, provider_key, home, inject_home=True)


def _resolve_acp_limits(cfg: dict[str, Any], provider_key: str) -> dict[str, Any]:
    """Read one provider's agent-process and session limits.

    The ``acp`` block is the operator's, not the shim's: how many processes to
    run is a fact about this deployment's memory, and how long to keep a
    session is a fact about how its clients work.  Every key is optional, and
    an absent one leaves the transport's default in place.

    Args:
        cfg: The provider's own config block.
        provider_key: The config key, used in error messages.

    Returns:
        The keyword arguments :class:`ProviderInfo` takes for these limits.

    Raises:
        ValueError: A value is present but is not a positive number.
    """
    raw = cfg.get("acp")
    block: dict[str, Any] = raw if isinstance(raw, dict) else {}
    limits: dict[str, Any] = {}
    processes = block.get("processes")
    if processes is not None:
        if not isinstance(processes, int) or isinstance(processes, bool):
            raise ValueError(
                f"Provider '{provider_key}': acp.processes must be an integer, "
                f"got {processes!r}"
            )
        if processes < 1:
            raise ValueError(
                f"Provider '{provider_key}': acp.processes must be >= 1, "
                f"got {processes}"
            )
        limits["acp_processes"] = processes
    max_sessions = block.get("max_sessions")
    if max_sessions is not None:
        if not isinstance(max_sessions, int) or isinstance(max_sessions, bool):
            raise ValueError(
                f"Provider '{provider_key}': acp.max_sessions must be an "
                f"integer, got {max_sessions!r}"
            )
        limits["acp_max_sessions"] = max_sessions
    ttl = block.get("session_ttl")
    if ttl is not None:
        if not isinstance(ttl, (int, float)) or isinstance(ttl, bool):
            raise ValueError(
                f"Provider '{provider_key}': acp.session_ttl must be a number "
                f"of seconds, got {ttl!r}"
            )
        limits["acp_session_ttl"] = float(ttl)
    return limits
