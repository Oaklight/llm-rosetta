"""The Node.js that ``ext install-cli`` installs for npm programs (#1820).

An npm CLI (Claude Code, Codex, OpenCode) needs ``node`` and ``npm``.  The image
carries neither: they are installed into the tools directory beside the CLIs, so
a changed image keeps them (``<tools>/node/<version>/``).

The version and the SHA-256 of each build are pinned here, in one place, so a
bump is a reviewed diff.  glibc and macOS builds come from nodejs.org.  Node
does not publish musl builds; those come from ``unofficial-builds.nodejs.org``,
which the Node.js team runs and which publishes a ``SHASUMS256.txt`` but no GPG
signature, so the checksum recorded below is the only check there.  Every
download is https and is checked against its checksum before it is unpacked.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The Node.js version installed (an LTS line; ``npm`` is the one it bundles).
NODE_VERSION = "24.22.3"

_OFFICIAL = "https://nodejs.org/dist/v{version}/node-v{version}-{slug}.tar.gz"
_UNOFFICIAL = (
    "https://unofficial-builds.nodejs.org/download/release/"
    "v{version}/node-v{version}-{slug}.tar.gz"
)


@dataclass(frozen=True)
class NodeBuild:
    """One platform's Node.js archive.

    Attributes:
        url: The archive's https URL, with ``{version}`` already filled in.
        sha256: The archive's SHA-256, lower-case hex.
    """

    url: str
    sha256: str


def _build(template: str, slug: str, sha256: str) -> NodeBuild:
    return NodeBuild(template.format(version=NODE_VERSION, slug=slug), sha256)


#: The pinned builds, by platform key (see ``platform_key``).
NODE_BUILDS: dict[str, NodeBuild] = {
    "linux_x64_gnu": _build(
        _OFFICIAL,
        "linux-x64",
        "061ec3e3062115705b56b51fa9cc330ab50d34b6969387d4e4b1a50c6809acc2",
    ),
    "linux_arm64_gnu": _build(
        _OFFICIAL,
        "linux-arm64",
        "261af2af250dfd7e9f25cf151d1b2713a0adf8f10a9ad116c4947f68627affe8",
    ),
    "linux_x64_musl": _build(
        _UNOFFICIAL,
        "linux-x64-musl",
        "2db4f8a40abe5f5b5d119964223fedc9c4c7c5e7c632901ae860af522c7844b2",
    ),
    "linux_arm64_musl": _build(
        _UNOFFICIAL,
        "linux-arm64-musl",
        "248c96a84f6c8338646b9fd73bf125124678f8fb08db5f23023f462055e0eedc",
    ),
    "darwin_x64": _build(
        _OFFICIAL,
        "darwin-x64",
        "977614af0504c81045e7fd85596dbca0e5f951b96681516dadafedcca8c447f7",
    ),
    "darwin_arm64": _build(
        _OFFICIAL,
        "darwin-arm64",
        "ff6e17d4874b76d272212b908cc020d6e815fba5a6c7eaaa06add725840307fd",
    ),
}
