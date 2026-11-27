"""Install the external CLI an extension drives, at one exact version (#1820).

``llm-rosetta ext install-cli <id>`` reads ``external[].install`` of the
extension's manifest and puts the program under ``<tools>/<id>/`` (see
:func:`~llm_rosetta.extensions._dirs.tools_dir`)::

    <tools>/bin/<link>                 one directory to put on PATH (the entry
                                       links of every extension, side by side)
    <tools>/<id>/bin/<link>            the entry link ``check`` finds
    <tools>/node/<version>/            the Node.js (and npm) the npm programs use
    <tools>/<id>/npm-<version>/        an npm install (its own prefix)
    <tools>/<id>/release-<version>/    an unpacked release archive
    <tools>/<id>/installed.json        what is installed, written last

Nothing is installed from a source that was not checked: a release archive
(and the Node.js, see :mod:`~llm_rosetta.extensions._node`) must match the
SHA-256 pinned for the platform before it is unpacked, and a member that would
land outside the directory refuses the whole archive.  An npm package is
fetched by the pinned Node.js's ``npm`` at the pinned version, and ``npm``
checks it against the registry's integrity record.  The install runs as the
calling user, never as root, writes only under the tools directory (npm's cache
and its ``HOME`` included), and a run that finds the pinned version in place
changes nothing.  Nothing is read from the image: a program installed here
runs after the image changed, or is reported ``broken`` with the command that
reinstalls it.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from llm_rosetta.contracts.extension import ExternalProgram

from ._dirs import ExtensionInstallError, tools_dir
from ._install import check_external
from ._node import NODE_BUILDS, NODE_VERSION

#: The largest download accepted, in bytes.
MAX_DOWNLOAD = 1 << 30
_STATE = "installed.json"
_MACHINES = {
    "x86_64": "x64",
    "amd64": "x64",
    "aarch64": "arm64",
    "arm64": "arm64",
}

Fetch = Callable[[str, Path], None]


class CliInstallError(ExtensionInstallError):
    """A program could not be installed; nothing was left installed."""


@dataclass(frozen=True)
class CliInstallResult:
    """What one ``install-cli`` run did for one program.

    Attributes:
        program: The program's name.
        state: ``installed``, ``unchanged`` or ``would_install``.
        version: The pinned version.
        link: The entry link.
        found: The version the program reported after the install.
    """

    program: str
    state: str
    version: str
    link: str
    found: str | None = None


def _is_musl() -> bool:
    return any(Path(d).glob("ld-musl-*") for d in ("/lib", "/usr/lib"))


def platform_key() -> str:
    """Name this machine as the release spec does: ``linux_x64_musl``, ``darwin_arm64``.

    Raises:
        CliInstallError: The system or the machine is not one a spec names.
    """
    machine = _MACHINES.get(platform.machine().lower())
    if machine is None:
        raise CliInstallError(
            f"no build is known for the machine {platform.machine()!r}"
        )
    if sys.platform.startswith("linux"):
        return f"linux_{machine}_{'musl' if _is_musl() else 'gnu'}"
    if sys.platform == "darwin":
        return f"darwin_{machine}"
    raise CliInstallError(f"no build is known for the system {sys.platform!r}")


# --------------------------------------------------------------------------- #
# download and unpack
# --------------------------------------------------------------------------- #


class _HttpsOnly(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise CliInstallError(f"a redirect to {newurl!r} leaves https")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _fetch(url: str, dest: Path) -> None:
    """Stream an https URL into *dest*; at most :data:`MAX_DOWNLOAD` bytes."""
    if urllib.parse.urlsplit(url).scheme != "https":
        raise CliInstallError(f"{url!r} is not an https URL")
    opener = urllib.request.build_opener(_HttpsOnly)
    try:
        with opener.open(url, timeout=60) as response, dest.open("wb") as out:
            size = 0
            while chunk := response.read(1 << 16):
                size += len(chunk)
                if size > MAX_DOWNLOAD:
                    raise CliInstallError(f"{url} is larger than {MAX_DOWNLOAD} bytes")
                out.write(chunk)
    except (urllib.error.URLError, OSError) as exc:
        raise CliInstallError(f"cannot download {url}: {exc}") from None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root: Path, name: str) -> bool:
    target = (root / name).resolve()
    return target == root.resolve() or root.resolve() in target.parents


def _unpack(archive: Path, dest: Path, url: str) -> None:
    """Unpack *archive* into *dest*, refusing any member that lands outside it."""
    outside = CliInstallError(
        f"the archive holds a member outside its directory ({url})"
    )
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            if not all(_inside(dest, name) for name in bundle.namelist()):
                raise outside
            bundle.extractall(dest)
        return
    try:
        bundle_tar = tarfile.open(archive)
    except tarfile.TarError as exc:
        raise CliInstallError(f"the archive of {url} cannot be read: {exc}") from None
    with bundle_tar as tar:
        members = tar.getmembers()
        for member in members:
            if member.isdev() or member.isfifo():
                raise CliInstallError(
                    f"the archive holds a device or pipe ({member.name})"
                )
            if not _inside(dest, member.name):
                raise outside
            if member.issym():
                link = os.path.join(os.path.dirname(member.name), member.linkname)
                if os.path.isabs(member.linkname) or not _inside(dest, link):
                    raise outside
            if member.islnk() and (
                os.path.isabs(member.linkname) or not _inside(dest, member.linkname)
            ):
                raise outside
        for member in members:
            tar.extract(member, dest)


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #


def _read_state(root: Path) -> dict[str, object]:
    try:
        return json.loads((root / _STATE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_state(root: Path, state: Mapping[str, object]) -> None:
    tmp = root / f".{_STATE}.tmp"
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, root / _STATE)


def _point_link(link: Path, target: Path) -> None:
    """Make *link* a relative symlink to *target*, replacing what was there."""
    link.parent.mkdir(parents=True, exist_ok=True)
    tmp = link.parent / f".{link.name}.new"
    with contextlib.suppress(FileNotFoundError):
        tmp.unlink()
    tmp.symlink_to(os.path.relpath(target, link.parent))
    os.replace(tmp, link)


def _prune(root: Path, keep: Path) -> None:
    for entry in root.iterdir():
        if entry != keep and entry.name.startswith(("npm-", "release-")):
            shutil.rmtree(entry, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Node.js
# --------------------------------------------------------------------------- #

_NODE_LINKS = ("node", "npm", "npx")


def _node_problem(directory: Path) -> str | None:
    """Why *directory*'s ``node`` does not run at the pinned version, or ``None``."""
    try:
        run = subprocess.run(
            [str(directory / "bin" / "node"), "--version"],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)
    if run.returncode != 0:
        tail = (run.stderr or run.stdout).strip().splitlines()[:1]
        return f"exit {run.returncode}: {tail[0] if tail else 'no output'}"
    if run.stdout.strip() != f"v{NODE_VERSION}":
        return f"it reports {run.stdout.strip()!r}"
    return None


def install_node(
    *, platform: str | None = None, fetch: Fetch | None = None
) -> tuple[Path, str]:
    """Install the pinned Node.js into ``<tools>/node/<version>/``.

    The archive's SHA-256 is checked before it is unpacked.  A run that finds
    that directory holding a ``node`` that runs at the pinned version changes
    nothing; one that finds it broken (another libc, a damaged file) installs
    it again.  The shared links ``<tools>/bin/{node,npm,npx}`` follow it.

    Args:
        platform: The platform key the build is chosen for; this machine's by
            default.
        fetch: ``fetch(url, dest)``; an https download by default.

    Returns:
        The Node.js directory and ``installed`` or ``unchanged``.

    Raises:
        CliInstallError: No build is pinned for the platform, the download or
            its checksum fails, or the unpacked ``node`` does not run.  Nothing
            is left installed by a failed run.
    """
    root = tools_dir() / "node"
    final = root / NODE_VERSION
    shared = tools_dir() / "bin"
    links_ok = all((shared / name).exists() for name in _NODE_LINKS)
    if links_ok and _node_problem(final) is None:
        return final, "unchanged"
    key = platform or platform_key()
    build = NODE_BUILDS.get(key)
    if build is None:
        known = ", ".join(sorted(NODE_BUILDS))
        raise CliInstallError(f"no Node.js build is pinned for {key}; pinned: {known}")
    root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".stage-", dir=root))
    try:
        archive = stage / "download"
        (fetch or _fetch)(build.url, archive)
        got = _sha256(archive)
        if got != build.sha256:
            raise CliInstallError(
                f"the archive of {build.url} does not match the pinned sha256 "
                f"(pinned {build.sha256}, got {got}); nothing was installed"
            )
        unpacked = stage / "unpacked"
        unpacked.mkdir()
        _unpack(archive, unpacked, build.url)
        tops = [entry for entry in unpacked.iterdir() if entry.is_dir()]
        if len(tops) != 1 or not (tops[0] / "bin" / "node").is_file():
            raise CliInstallError(f"the archive of {build.url} holds no bin/node")
        # A node runs from wherever it is: the unpacked copy is tried before it
        # replaces anything, so a failed run leaves what was there.
        problem = _node_problem(tops[0])
        if problem is not None:
            raise CliInstallError(
                f"Node.js {NODE_VERSION} was unpacked but does not run on this "
                f"system ({problem}); on Alpine it needs the libstdc++ package"
            )
        shutil.rmtree(final, ignore_errors=True)
        os.replace(tops[0], final)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    for name in _NODE_LINKS:
        _point_link(shared / name, final / "bin" / name)
    for entry in root.iterdir():
        if entry != final and not entry.name.startswith("."):
            shutil.rmtree(entry, ignore_errors=True)
    return final, "installed"


# --------------------------------------------------------------------------- #
# the two sources
# --------------------------------------------------------------------------- #


def _npm_install(
    program: ExternalProgram,
    root: Path,
    npm: str | None,
    key: str | None,
    fetch: Fetch | None,
) -> tuple[Path, Path]:
    install = program.install
    assert install is not None and install.npm is not None
    path = os.environ.get("PATH", "")
    if npm is None:
        node, _ = install_node(platform=key, fetch=fetch)
        exe = str(node / "bin" / "npm")
        path = os.pathsep.join([str(node / "bin"), path])
    else:
        found = shutil.which(npm)
        if found is None:
            raise CliInstallError(f"{npm!r} is not installed")
        exe = found
    root.mkdir(parents=True, exist_ok=True)
    prefix = root / f"npm-{install.version}"
    shutil.rmtree(prefix, ignore_errors=True)
    env = {
        **os.environ,
        "PATH": path,
        # Everything npm writes goes under the tools directory: a uid with no
        # home must not send its cache or its config to ``/``.
        "HOME": str(root / ".home"),
        "npm_config_cache": str(root / ".npm-cache"),
        "npm_config_update_notifier": "false",
        "npm_config_fund": "false",
        "npm_config_audit": "false",
    }
    command = [
        exe,
        "install",
        "--prefix",
        str(prefix),
        f"{install.npm}@{install.version}",
    ]
    try:
        run = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=env,
            stdin=subprocess.DEVNULL,
            timeout=900,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        shutil.rmtree(prefix, ignore_errors=True)
        raise CliInstallError(f"npm could not run: {exc}") from None
    if run.returncode != 0:
        shutil.rmtree(prefix, ignore_errors=True)
        tail = (run.stderr or run.stdout).strip().splitlines()[-3:]
        raise CliInstallError(f"npm exited with {run.returncode}: {' | '.join(tail)}")
    entry = install.bin or program.name
    binary = prefix / "node_modules" / ".bin" / entry
    if not binary.exists():
        shutil.rmtree(prefix, ignore_errors=True)
        raise CliInstallError(
            f"{install.npm}@{install.version} does not provide the program {entry!r}"
        )
    return prefix, binary


def _release_install(
    program: ExternalProgram, root: Path, key: str, fetch: Fetch
) -> tuple[Path, Path]:
    install = program.install
    assert install is not None and install.release is not None
    release = install.release
    want = release.sha256.get(key)
    if want is None:
        known = ", ".join(sorted(release.sha256))
        raise CliInstallError(
            f"no build of {program.name} {install.version} is recorded for {key}; "
            f"recorded: {known}"
        )
    url = release.url.format(version=install.version, platform=key)
    root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".stage-", dir=root))
    try:
        archive = stage / "download"
        fetch(url, archive)
        got = _sha256(archive)
        if got != want:
            raise CliInstallError(
                f"the archive of {url} does not match the recorded sha256 "
                f"(recorded {want}, got {got}); nothing was installed"
            )
        unpacked = stage / "unpacked"
        unpacked.mkdir()
        _unpack(archive, unpacked, url)
        binary = unpacked / release.binary
        if not binary.is_file():
            raise CliInstallError(f"the archive of {url} holds no {release.binary!r}")
        binary.chmod(binary.stat().st_mode | 0o111)
        final = root / f"release-{install.version}"
        shutil.rmtree(final, ignore_errors=True)
        os.replace(unpacked, final)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return final, final / release.binary


# --------------------------------------------------------------------------- #
# the verb
# --------------------------------------------------------------------------- #


def _link_name(program: ExternalProgram) -> str:
    install = program.install
    assert install is not None
    if install.release is not None:
        return install.release.link or program.name
    return install.bin or program.name


def install_cli(
    ext_id: str,
    program: ExternalProgram,
    *,
    platform: str | None = None,
    fetch: Fetch | None = None,
    npm: str | None = None,
    dry_run: bool = False,
) -> CliInstallResult:
    """Install *program* of extension *ext_id* at its pinned version.

    Args:
        ext_id: The extension id; the program goes under ``<tools>/<ext_id>/``.
        program: The declaration, with an ``install`` spec.
        platform: The platform key a release is chosen for; this machine's by
            default.
        fetch: ``fetch(url, dest)``, how a release archive (or the Node.js) is
            downloaded; an https download by default.
        npm: An npm command to use instead of the one in the pinned Node.js
            this installs under ``<tools>/node``.
        dry_run: Report what would happen and change nothing.

    Returns:
        What was done: ``installed``, ``unchanged`` or ``would_install``.

    Raises:
        CliInstallError: There is no spec, the caller is root, the source is
            missing, a download or its checksum fails, or the installed program
            does not run.  Nothing is left installed by a failed run.
    """
    install = program.install
    if install is None:
        hint = f"; {program.install_hint}" if program.install_hint else ""
        raise CliInstallError(f"{program.name!r} declares no install spec{hint}")
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        raise CliInstallError(
            "run as a regular user, not root: the tools directory is the user's"
        )
    root = tools_dir() / ext_id
    link = root / "bin" / _link_name(program)
    source = "npm" if install.npm is not None else "release"
    state = _read_state(root)
    current = {"program": program.name, "version": install.version, "source": source}
    if (
        {k: state.get(k) for k in current} == current
        and link.exists()
        and check_external(program, search=[root / "bin", tools_dir() / "bin"]).state
        == "ok"
    ):
        return CliInstallResult(
            program.name, "unchanged", install.version, str(link), install.version
        )
    if dry_run:
        return CliInstallResult(
            program.name, "would_install", install.version, str(link)
        )
    if install.npm is not None:
        key = platform
        if key is None and npm is None:
            key = platform_key()
        directory, binary = _npm_install(program, root, npm, key, fetch)
    else:
        key = platform or platform_key()
        directory, binary = _release_install(program, root, key, fetch or _fetch)
    _point_link(link, binary)
    _point_link(tools_dir() / "bin" / link.name, link)
    status = check_external(program, search=[root / "bin", tools_dir() / "bin"])
    if status.state != "ok":
        for failed in (link, tools_dir() / "bin" / link.name):
            with contextlib.suppress(FileNotFoundError):
                failed.unlink()
        shutil.rmtree(directory, ignore_errors=True)
        (root / _STATE).unlink(missing_ok=True)
        raise CliInstallError(
            f"{program.name} {install.version} was installed but its check says "
            f"{status.state}: {status.detail or 'no detail'}"
        )
    _write_state(root, current)
    _prune(root, directory)
    return CliInstallResult(
        program.name, "installed", install.version, str(link), status.found
    )
