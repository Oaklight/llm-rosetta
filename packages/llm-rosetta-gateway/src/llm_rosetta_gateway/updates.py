"""What this gateway runs, and what PyPI has for it.

The console asks two questions of this module: is there a newer release than
the one in memory, and — when the operator presses update — may this process
replace itself.  Both are facts about the running installation, so they live
beside the gateway rather than in the admin layer.

Nothing here talks to PyPI on import.  :func:`check` caches a successful
answer for :data:`CACHE_SECONDS` and a failed one for
:data:`FAILURE_CACHE_SECONDS`, so a panel left open costs one request per six
hours, and a gateway that cannot reach the index costs one timeout per five
minutes rather than one per page load.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

#: The JSON API of the package index.  Both distributions are asked for; the
#: gateway one did not exist before the split (epic #1127), and a 404 from a
#: not-yet-published distribution is an ordinary answer, not an error.
PYPI_JSON = "https://pypi.org/pypi/{name}/json"
PYPI_RELEASE_JSON = "https://pypi.org/pypi/{name}/{version}/json"
CORE_DIST = "llm-rosetta"
GATEWAY_DIST = "llm-rosetta-gateway"

#: How long one probe of PyPI is reused.  A release is not news for six hours.
CACHE_SECONDS = 6 * 3600

#: How long a *failed* probe is reused.  A machine with no route to PyPI must
#: not pay a timeout on every panel load, and five minutes is short enough
#: that a fixed network is noticed without a restart.
FAILURE_CACHE_SECONDS = 5 * 60

#: A self-update runs ``pip`` and then restarts; a slow index must not hold an
#: admin request open forever.
INSTALL_TIMEOUT = 600.0

_VERSION_RE = re.compile(
    r"^(?:(\d+)!)?(\d+(?:\.\d+)*)(?:(a|b|rc)(\d+))?(?:\.post(\d+))?(?:\.dev(\d+))?$",
    re.IGNORECASE,
)


def parse_version(text: str) -> tuple[Any, ...]:
    """Return a sort key for a PEP 440 version string.

    The key orders every form the index serves: ``0.14.0.dev22`` sorts before
    ``0.14.0a1``, which sorts before ``0.14.0``, which sorts before
    ``0.14.0.post1``.  A local segment (``+g1234abc``) is dropped: it names a
    build, not a release order, and pip ignores it when resolving.  Text the
    pattern does not know sorts below every real version.

    Args:
        text: A version string, normally from ``__version__`` or PyPI.

    Returns:
        A tuple usable with ``<`` and ``>``.
    """
    raw = (text or "").strip().split("+", 1)[0]
    match = _VERSION_RE.match(raw)
    if not match:
        return (-1,)
    epoch = int(match.group(1) or 0)
    release = [int(part) for part in match.group(2).split(".")]
    if len(release) < 4:
        release += [0] * (4 - len(release))
    pre_kind = (match.group(3) or "").lower()
    pre_num = int(match.group(4) or 0)
    dev_num = match.group(6)
    # A bare ``.dev`` release is the first step towards its final version, so
    # it sorts below every alpha.  A pre-release sorts below the final.
    if pre_kind:
        pre_key: tuple[int, int] = ({"a": 1, "b": 2, "rc": 3}[pre_kind], pre_num)
    elif dev_num is not None:
        pre_key = (0, 0)
    else:
        pre_key = (4, 0)
    post_key = (0, 0) if match.group(5) is None else (1, int(match.group(5)))
    # A dev build sorts *below* the version it leads to, so ``0.15.0rc1.dev0``
    # is older than ``0.15.0rc1`` and the console never offers a downgrade.
    dev_key = (0, int(dev_num)) if dev_num is not None else (1, 0)
    return (epoch, *release, *pre_key, *post_key, *dev_key)


def is_newer(candidate: str, current: str) -> bool:
    """Whether ``candidate`` is a later release than ``current``."""
    return parse_version(candidate) > parse_version(current)


def running_versions() -> dict[str, str]:
    """The core and gateway versions this process imported."""
    from llm_rosetta import __version__ as core_version

    from . import __version__ as gateway_version

    return {"core": core_version, "gateway": gateway_version}


def is_editable(dist_name: str) -> bool:
    """Whether a distribution is installed from a source checkout.

    An editable install is a working tree someone is changing; replacing it
    with an index release would silently detach their edits, so the console
    does not offer to update such a process.
    """
    try:
        distribution = importlib.metadata.distribution(dist_name)
        raw = distribution.read_text("direct_url.json")
    except Exception:
        return False
    if not raw:
        return False
    try:
        direct = json.loads(raw)
    except ValueError:
        return False
    return bool((direct.get("dir_info") or {}).get("editable"))


def pip_available() -> bool:
    """Whether this interpreter can run ``pip`` as a module."""
    return importlib.util.find_spec("pip") is not None


def self_update_status(target: str | None) -> dict[str, Any]:
    """Whether this process may install ``target`` and restart into it.

    Args:
        target: The version the probe found, or ``None`` when there is none.

    Returns:
        ``{"allowed": bool, "reason": str | None}``; ``reason`` is a stable
        word the admin panel translates.
    """
    if not target:
        return {"allowed": False, "reason": "no_update"}
    for name in (GATEWAY_DIST, CORE_DIST):
        if is_editable(name):
            return {"allowed": False, "reason": "editable_install"}
    if not pip_available():
        return {"allowed": False, "reason": "no_pip"}
    return {"allowed": True, "reason": None}


def read_changelog(lang: str) -> str:
    """Return the bundled changelog, or ``""`` when the build has none.

    The file is written into the wheel by the release workflow from the docs
    branch, so a source checkout or a wheel built without it simply has no
    changelog to show — the panel then hides the "what changed" entry instead
    of inventing one.
    """
    name = (
        "CHANGELOG.zh.md"
        if (lang or "").lower().startswith("zh")
        else "CHANGELOG.en.md"
    )
    try:
        return (Path(__file__).with_name(name)).read_text(encoding="utf-8")
    except OSError:
        return ""


async def _fetch_latest(client: Any, name: str) -> str | None:
    """The latest version PyPI lists for ``name``, or ``None``."""
    try:
        response = await client.get(PYPI_JSON.format(name=name))
    except Exception:
        return None
    if response.status_code != 200:
        return None
    try:
        info = response.json().get("info") or {}
    except ValueError:
        return None
    version = info.get("version")
    return version if isinstance(version, str) and version else None


def _pick_update(current: dict[str, str], latest: dict[str, str | None]) -> str | None:
    """The highest published version that is newer than what is running.

    The two distributions are released together, so either one may carry the
    answer; the higher of the newer ones wins.
    """
    candidates = [
        version
        for key, version in latest.items()
        if version and is_newer(version, current.get(key, "0"))
    ]
    if not candidates:
        return None
    return max(candidates, key=parse_version)


_cache: dict[str, Any] = {"at": 0.0, "payload": None, "ttl": CACHE_SECONDS}


async def check(proxy: str | None = None, *, force: bool = False) -> dict[str, Any]:
    """Probe PyPI for a newer release, reusing the last answer for six hours.

    Args:
        proxy: The gateway's proxy URL, when one is configured.
        force: Ignore the cache (the panel's own refresh button).

    Returns:
        The payload the admin endpoint serves: what is running, what PyPI
        lists, and the one update that would apply to both.
    """
    now = time.time()
    if not force and _cache["payload"] and now - _cache["at"] < _cache["ttl"]:
        return _cache["payload"]

    from ._vendor.httpclient import AsyncClient

    core_latest: str | None = None
    gateway_latest: str | None = None
    try:
        async with AsyncClient(timeout=6, proxy=proxy) as client:
            # One round trip, not two: an index that hangs costs the panel one
            # timeout rather than a queue of them.
            core_latest, gateway_latest = await asyncio.gather(
                _fetch_latest(client, CORE_DIST),
                _fetch_latest(client, GATEWAY_DIST),
            )
    except Exception:
        core_latest = gateway_latest = None

    current = running_versions()
    latest = {"core": core_latest, "gateway": gateway_latest}
    target = _pick_update(current, latest)
    payload = {
        "checked": core_latest is not None or gateway_latest is not None,
        "checked_at": int(now),
        "current": current,
        "latest": latest,
        "update": (
            {
                "version": target,
                "url": f"https://pypi.org/project/{CORE_DIST}/{target}/",
            }
            if target
            else None
        ),
        "self_update": self_update_status(target),
    }
    # A success is reused for six hours; a failure for five minutes, so an
    # offline gateway costs one timeout per window rather than one per panel
    # load, and a fixed network is picked up without a restart.
    _cache["payload"] = payload
    _cache["at"] = now
    _cache["ttl"] = CACHE_SECONDS if payload["checked"] else FAILURE_CACHE_SECONDS
    return payload


async def published_dists(version: str, proxy: str | None = None) -> list[str]:
    """The distributions that publish ``version`` on PyPI.

    A release before the two-distribution split has only the core package;
    asking for the gateway one would make pip fail the whole install.
    """
    from ._vendor.httpclient import AsyncClient

    found: list[str] = []
    try:
        async with AsyncClient(timeout=8, proxy=proxy) as client:
            for name in (CORE_DIST, GATEWAY_DIST):
                try:
                    response = await client.get(
                        PYPI_RELEASE_JSON.format(name=name, version=version)
                    )
                except Exception:
                    continue
                if response.status_code == 200:
                    found.append(name)
    except Exception:
        return found
    return found


async def install(
    version: str, proxy: str | None = None, timeout: float = INSTALL_TIMEOUT
) -> dict[str, Any]:
    """Install ``version`` with pip, without restarting.

    The caller decides whether to restart: a failed install leaves the old
    code running and the output says why.

    Args:
        version: The release to install, from the update probe.
        proxy: The gateway's proxy URL, when one is configured.
        timeout: Seconds before pip is killed.

    Returns:
        ``{"ok": bool, "output": str, "dists": list[str]}`` — ``output`` is
        the tail of pip's combined stdout and stderr.
    """
    dists = await published_dists(version, proxy=proxy)
    if not dists:
        return {
            "ok": False,
            "output": f"PyPI publishes no {version} release to install",
            "dists": [],
        }
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--no-input",
        "--disable-pip-version-check",
    ]
    if proxy:
        command += ["--proxy", proxy]
    command += [f"{name}=={version}" for name in dists]
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except OSError as exc:
        return {"ok": False, "output": f"cannot start pip: {exc}", "dists": dists}
    try:
        output, _ = await asyncio.wait_for(process.communicate(), timeout)
    except asyncio.TimeoutError:
        process.kill()
        return {
            "ok": False,
            "output": f"pip did not finish within {timeout:.0f}s",
            "dists": dists,
        }
    text = (output or b"").decode("utf-8", "replace")
    return {"ok": process.returncode == 0, "output": text[-4000:], "dists": dists}


def _command_line() -> list[str]:
    """This process's own argv, for re-executing it unchanged.

    ``/proc/self/cmdline`` is the argument vector the kernel actually started,
    which is what a restart must repeat; ``sys.argv`` loses the interpreter
    wrapper (``python -m``) and the executable path.
    """
    try:
        raw = Path("/proc/self/cmdline").read_bytes()
    except OSError:
        return [sys.executable, "-m", "llm_rosetta_gateway", *sys.argv[1:]]
    parts = [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
    return parts or [sys.executable]


def restart_process() -> None:
    """Replace this process image with the same command line.

    ``execv`` keeps the PID, so a supervisor (systemd, Docker) needs no restart
    policy for this to work, and a process without one still comes back.
    Called from a loop callback after the update response has been written.
    """
    os.execv(sys.executable, _command_line())
