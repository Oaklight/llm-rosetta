"""``llm-rosetta ext install-cli``: a pinned external CLI in the tools directory (#1820).

No test reaches a network.  A fake ``npm`` (a script that writes what the real
one would) stands for the registry, and a fake fetch hands out release
archives built here.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import stat
import sys
import tarfile
import textwrap
import urllib.parse
from pathlib import Path

import pytest

from llm_rosetta.contracts import IR_VERSION, SURFACE_VERSION
from llm_rosetta.contracts.extension import (
    ExternalInstall,
    ExternalProgram,
    ReleaseArchive,
    parse_extension_manifest,
)
from llm_rosetta.extensions import _cli, _tools
from llm_rosetta.extensions._dirs import tool_bin_dir, tools_dir
from llm_rosetta.extensions._install import check_external
from llm_rosetta.extensions._node import NODE_BUILDS as PINNED_BUILDS
from llm_rosetta.extensions._node import NODE_VERSION, NodeBuild
from llm_rosetta.extensions._tools import CliInstallError, install_cli, platform_key

MUSL_X64 = "linux_x64_musl"
URL = "https://example.test/agy/{version}/agy_{platform}.tar.gz"


@pytest.fixture
def tools(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "config" / "tools"
    monkeypatch.setenv("LLM_ROSETTA_TOOLS_DIR", str(root))
    monkeypatch.setenv("HOME", str(tmp_path / "no-such-home"))
    monkeypatch.setattr(os, "geteuid", lambda: 1000, raising=False)
    return root


@pytest.fixture
def fake_npm(tmp_path, monkeypatch) -> Path:
    """An ``npm`` that installs ``<package>@<version>`` into ``--prefix``."""
    log = tmp_path / "npm.log"
    script = tmp_path / "npm"
    script.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import json, os, sys
            args = sys.argv[1:]
            with open({str(log)!r}, "a") as out:
                out.write(json.dumps({{"args": args, "cache": os.environ.get("npm_config_cache"),
                                       "home": os.environ.get("HOME"),
                                       "path": os.environ.get("PATH"), "self": sys.argv[0]}}) + "\\n")
            if os.environ.get("FAKE_NPM_FAIL"):
                print("npm ERR! 404", file=sys.stderr)
                sys.exit(1)
            prefix = args[args.index("--prefix") + 1]
            spec = args[-1]
            name, _, version = spec.rpartition("@")
            binary = os.environ.get("FAKE_NPM_BIN") or name.split("/")[-1]
            bin_dir = os.path.join(prefix, "node_modules", ".bin")
            os.makedirs(bin_dir)
            path = os.path.join(bin_dir, binary)
            with open(path, "w") as handle:
                guard = "node --version >/dev/null || exit $?\\n" if os.environ.get("FAKE_NPM_NEEDS_NODE") else ""
                handle.write("#!/bin/sh\\n" + guard + "echo " + binary + " " + version + "\\n")
            os.chmod(path, 0o755)
            """
        )
    )
    script.chmod(0o755)
    monkeypatch.setenv("FAKE_NPM_BIN", "claude")
    return script


REAL_FETCH = _tools._fetch


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Nothing here may download: a test that needs a download supplies one."""

    def refuse(url, dest):
        raise AssertionError(f"a test reached the network: {url}")

    monkeypatch.setattr(_tools, "_fetch", refuse)


NODE_URL = "https://example.test/node/v{version}/node-{platform}.tar.gz"


def _node_archive(
    fake_npm: Path | None = None, *, node_body: str | None = None
) -> bytes:
    """A tar.gz shaped like a Node.js release: ``node-vX/bin/{node,npm,npx}``."""
    node = node_body or f"#!/bin/sh\necho v{NODE_VERSION}\n"
    npm = fake_npm.read_bytes() if fake_npm else b"#!/bin/sh\nexit 0\n"
    top = f"node-v{NODE_VERSION}-linux-x64-musl"
    return _archive(
        {
            f"{top}/bin/node": node.encode(),
            f"{top}/bin/npm": npm,
            f"{top}/bin/npx": b"#!/bin/sh\nexit 0\n",
        }
    )


def _pin_node(monkeypatch, blob: bytes, *, sha: str | None = None) -> _Net:
    """Pin *blob* as the Node.js of ``linux_x64_musl`` and serve it."""
    net = _Net(blob)
    url = NODE_URL.format(version=NODE_VERSION, platform="linux-x64-musl")
    digest = sha if sha is not None else hashlib.sha256(blob).hexdigest()
    monkeypatch.setitem(_tools.NODE_BUILDS, MUSL_X64, NodeBuild(url, digest))
    monkeypatch.setattr(_tools, "_fetch", net)
    return net


@pytest.fixture
def managed_node(fake_npm, monkeypatch) -> _Net:
    """The pinned Node.js is an archive whose ``npm`` is the fake one."""
    monkeypatch.setattr(_tools, "platform_key", lambda: MUSL_X64)
    return _pin_node(monkeypatch, _node_archive(fake_npm))


def _npm_calls(fake_npm: Path) -> list[dict]:
    log = fake_npm.parent / "npm.log"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]


def _npm_program(version: str = "2.1.299", **extra) -> ExternalProgram:
    return ExternalProgram(
        name="claude",
        check=("claude", "--version"),
        minimum="2.1.299",
        install=ExternalInstall(
            version=version, npm="@anthropic-ai/claude-code", **extra
        ),
    )


def _archive(files: dict[str, bytes | tuple[str, str]], *, mode: int = 0o755) -> bytes:
    """A tar.gz of *files*; a ``(kind, target)`` value is a link."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, body in files.items():
            info = tarfile.TarInfo(name)
            if isinstance(body, tuple):
                info.type = tarfile.SYMTYPE if body[0] == "symlink" else tarfile.LNKTYPE
                info.linkname = body[1]
                tar.addfile(info)
                continue
            info.size = len(body)
            info.mode = mode
            tar.addfile(info, io.BytesIO(body))
    return buffer.getvalue()


AGY_BODY = b"#!/bin/sh\necho agy 1.5.15\n"


def _release_program(
    blob: bytes, *, sha: str | None = None, binary: str = "antigravity"
) -> ExternalProgram:
    digest = sha if sha is not None else hashlib.sha256(blob).hexdigest()
    return ExternalProgram(
        name="agy",
        check=("agy", "--version"),
        install=ExternalInstall(
            version="1.5.15",
            release=ReleaseArchive(
                url=URL, sha256={MUSL_X64: digest}, binary=binary, link="agy"
            ),
        ),
    )


class _Net:
    """A fetch that serves *blob* for one URL and counts its calls."""

    def __init__(self, blob: bytes) -> None:
        self.blob = blob
        self.urls: list[str] = []

    def __call__(self, url: str, dest: Path) -> None:
        self.urls.append(url)
        dest.write_bytes(self.blob)


def _install(program, *, fake_npm=None, net=None, **kwargs):
    return install_cli(
        "demo",
        program,
        platform=MUSL_X64,
        npm=str(fake_npm) if fake_npm else None,
        fetch=net,
        **kwargs,
    )


class TestToolsDirectory:
    def test_the_environment_names_it(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LLM_ROSETTA_TOOLS_DIR", str(tmp_path / "t"))
        assert tools_dir() == tmp_path / "t"

    def test_by_default_it_sits_beside_the_extensions_directory(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("LLM_ROSETTA_TOOLS_DIR", raising=False)
        monkeypatch.setenv(
            "LLM_ROSETTA_EXTENSIONS_DIR", str(tmp_path / "config" / "extensions")
        )
        assert tools_dir() == tmp_path / "config" / "tools"
        assert (
            tool_bin_dir("claude") == tmp_path / "config" / "tools" / "claude" / "bin"
        )


class TestPlatformKey:
    @pytest.mark.parametrize(
        ("machine", "libc", "key"),
        [
            ("x86_64", "musl", "linux_x64_musl"),
            ("aarch64", "musl", "linux_arm64_musl"),
            ("x86_64", "gnu", "linux_x64_gnu"),
            ("arm64", "gnu", "linux_arm64_gnu"),
        ],
    )
    def test_linux_is_arch_and_libc(self, monkeypatch, machine, libc, key):
        monkeypatch.setattr(_tools.sys, "platform", "linux")
        monkeypatch.setattr(_tools.platform, "machine", lambda: machine)
        monkeypatch.setattr(_tools, "_is_musl", lambda: libc == "musl")
        assert platform_key() == key

    def test_macos_has_no_libc(self, monkeypatch):
        monkeypatch.setattr(_tools.sys, "platform", "darwin")
        monkeypatch.setattr(_tools.platform, "machine", lambda: "arm64")
        assert platform_key() == "darwin_arm64"

    def test_an_unknown_machine_is_refused(self, monkeypatch):
        monkeypatch.setattr(_tools.sys, "platform", "linux")
        monkeypatch.setattr(_tools.platform, "machine", lambda: "riscv64")
        with pytest.raises(CliInstallError, match="riscv64"):
            platform_key()


class TestNpmInstall:
    def test_it_installs_the_exact_version_and_links_the_entry(self, tools, fake_npm):
        result = _install(_npm_program(), fake_npm=fake_npm)
        assert (result.state, result.version) == ("installed", "2.1.299")
        (call,) = _npm_calls(fake_npm)
        assert call["args"][-1] == "@anthropic-ai/claude-code@2.1.299"
        prefix = Path(call["args"][call["args"].index("--prefix") + 1])
        assert prefix == tools / "demo" / "npm-2.1.299"
        link = tools / "demo" / "bin" / "claude"
        assert link.is_symlink() and link.exists()
        assert (
            check_external(_npm_program(), search=[tool_bin_dir("demo")]).state == "ok"
        )

    def test_the_program_is_found_outside_path_only_by_the_tools_directory(
        self, tools, fake_npm, tmp_path, monkeypatch
    ):
        _install(_npm_program(), fake_npm=fake_npm)
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        assert check_external(_npm_program()).state == "missing"

    def test_everything_npm_writes_stays_in_the_tools_directory(self, tools, fake_npm):
        _install(_npm_program(), fake_npm=fake_npm)
        (call,) = _npm_calls(fake_npm)
        assert Path(call["cache"]).is_relative_to(tools)
        assert not (fake_npm.parent / "no-such-home").exists()

    def test_a_second_run_changes_nothing(self, tools, fake_npm):
        _install(_npm_program(), fake_npm=fake_npm)
        before = sorted(p for p in tools.rglob("*"))
        result = _install(_npm_program(), fake_npm=fake_npm)
        assert result.state == "unchanged"
        assert len(_npm_calls(fake_npm)) == 1
        assert sorted(p for p in tools.rglob("*")) == before

    def test_a_new_pinned_version_replaces_the_old_one(self, tools, fake_npm):
        _install(_npm_program("2.1.299"), fake_npm=fake_npm)
        result = _install(_npm_program("2.1.300"), fake_npm=fake_npm)
        assert result.state == "installed"
        assert (tools / "demo" / "npm-2.1.300").is_dir()
        assert not (tools / "demo" / "npm-2.1.299").exists()
        assert "2.1.300" in os.path.realpath(tools / "demo" / "bin" / "claude")

    def test_a_missing_link_is_repaired(self, tools, fake_npm):
        _install(_npm_program(), fake_npm=fake_npm)
        (tools / "demo" / "bin" / "claude").unlink()
        assert _install(_npm_program(), fake_npm=fake_npm).state == "installed"
        assert (
            check_external(_npm_program(), search=[tool_bin_dir("demo")]).state == "ok"
        )

    def test_the_npm_bin_names_the_entry(self, tools, fake_npm, monkeypatch):
        monkeypatch.setenv("FAKE_NPM_BIN", "opencode")
        program = ExternalProgram(
            name="oc",
            check=("opencode", "--version"),
            install=ExternalInstall(
                version="1.19.0", npm="opencode-ai", bin="opencode"
            ),
        )
        _install(program, fake_npm=fake_npm)
        assert (tools / "demo" / "bin" / "opencode").exists()

    def test_a_failed_npm_leaves_nothing_behind(self, tools, fake_npm, monkeypatch):
        monkeypatch.setenv("FAKE_NPM_FAIL", "1")
        with pytest.raises(CliInstallError, match="npm"):
            _install(_npm_program(), fake_npm=fake_npm)
        assert not (tools / "demo" / "bin" / "claude").exists()
        assert not (tools / "demo" / "installed.json").exists()
        assert not (tools / "demo" / "npm-2.1.299").exists()

    def test_an_npm_override_that_is_not_there_is_named(self, tools, tmp_path):
        with pytest.raises(CliInstallError, match="no-such-npm"):
            install_cli("demo", _npm_program(), platform=MUSL_X64, npm="no-such-npm")
        assert not (tools / "demo" / "installed.json").exists()

    def test_a_package_that_provides_no_bin_is_a_failure(
        self, tools, fake_npm, monkeypatch
    ):
        monkeypatch.setenv("FAKE_NPM_BIN", "other")
        with pytest.raises(CliInstallError, match="claude"):
            _install(_npm_program(), fake_npm=fake_npm)
        assert not (tools / "demo" / "installed.json").exists()


class TestNodeInstall:
    def test_it_unpacks_the_pinned_build_and_links_the_commands(
        self, tools, monkeypatch
    ):
        net = _pin_node(monkeypatch, _node_archive())
        directory, state = _tools.install_node(platform=MUSL_X64)
        assert state == "installed"
        assert directory == tools / "node" / NODE_VERSION
        assert net.urls == [
            NODE_URL.format(version=NODE_VERSION, platform="linux-x64-musl")
        ]
        for name in ("node", "npm", "npx"):
            link = tools / "bin" / name
            assert link.is_symlink()
            assert link.resolve() == (directory / "bin" / name).resolve()

    def test_a_second_run_changes_nothing(self, tools, monkeypatch):
        net = _pin_node(monkeypatch, _node_archive())
        _tools.install_node(platform=MUSL_X64)
        _, state = _tools.install_node(platform=MUSL_X64)
        assert state == "unchanged"
        assert len(net.urls) == 1

    def test_a_node_that_no_longer_runs_is_installed_again(self, tools, monkeypatch):
        net = _pin_node(monkeypatch, _node_archive())
        directory, _ = _tools.install_node(platform=MUSL_X64)
        (directory / "bin" / "node").write_text("#!/bin/sh\nexit 127\n")
        _, state = _tools.install_node(platform=MUSL_X64)
        assert state == "installed"
        assert len(net.urls) == 2
        assert (directory / "bin" / "node").read_text().endswith(f"v{NODE_VERSION}\n")

    def test_a_missing_shared_link_is_repaired(self, tools, monkeypatch):
        _pin_node(monkeypatch, _node_archive())
        _tools.install_node(platform=MUSL_X64)
        (tools / "bin" / "npm").unlink()
        assert _tools.install_node(platform=MUSL_X64)[1] == "installed"
        assert (tools / "bin" / "npm").is_symlink()

    def test_a_wrong_checksum_installs_nothing(self, tools, monkeypatch):
        _pin_node(monkeypatch, _node_archive(), sha="0" * 64)
        with pytest.raises(CliInstallError, match="sha256"):
            _tools.install_node(platform=MUSL_X64)
        assert not (tools / "node" / NODE_VERSION).exists()
        assert not (tools / "bin").exists()
        assert [p.name for p in (tools / "node").iterdir()] == []

    def test_a_member_outside_its_directory_refuses_the_archive(
        self, tools, monkeypatch
    ):
        blob = _archive(
            {"../escape": b"x", f"node-v{NODE_VERSION}/bin/node": b"#!/bin/sh\n"}
        )
        _pin_node(monkeypatch, blob)
        with pytest.raises(CliInstallError, match="outside"):
            _tools.install_node(platform=MUSL_X64)
        assert not (tools / "node" / NODE_VERSION).exists()
        assert not (tools / "node" / "escape").exists()

    def test_an_archive_without_bin_node_is_refused(self, tools, monkeypatch):
        _pin_node(monkeypatch, _archive({"node-v1/README": b"x"}))
        with pytest.raises(CliInstallError, match="bin/node"):
            _tools.install_node(platform=MUSL_X64)
        assert not (tools / "node" / NODE_VERSION).exists()

    def test_a_node_that_does_not_run_here_leaves_nothing(self, tools, monkeypatch):
        blob = _node_archive(
            node_body="#!/bin/sh\necho 'Error loading shared library libstdc++.so.6' >&2\nexit 1\n"
        )
        _pin_node(monkeypatch, blob)
        with pytest.raises(CliInstallError, match="libstdc\\+\\+.so.6") as seen:
            _tools.install_node(platform=MUSL_X64)
        assert "exit 1" in str(seen.value)
        assert not (tools / "node" / NODE_VERSION).exists()
        assert not (tools / "bin" / "node").exists()

    def test_a_node_that_does_not_run_does_not_replace_the_one_there(
        self, tools, monkeypatch
    ):
        kept = tools / "node" / NODE_VERSION
        (kept / "bin").mkdir(parents=True)
        (kept / "marker").write_text("old")
        _pin_node(monkeypatch, _node_archive(node_body="#!/bin/sh\nexit 1\n"))
        with pytest.raises(CliInstallError, match="does not run"):
            _tools.install_node(platform=MUSL_X64)
        assert (kept / "marker").read_text() == "old"

    def test_another_version_is_pruned(self, tools, monkeypatch):
        stale = tools / "node" / "20.0.0"
        stale.mkdir(parents=True)
        _pin_node(monkeypatch, _node_archive())
        _tools.install_node(platform=MUSL_X64)
        assert not stale.exists()

    def test_a_platform_with_no_pin_names_the_pinned_ones(self, tools):
        with pytest.raises(CliInstallError, match="linux_riscv64_gnu") as seen:
            _tools.install_node(platform="linux_riscv64_gnu")
        assert "linux_x64_musl" in str(seen.value)
        assert not tools.exists()


class TestNodePins:
    def test_every_supported_platform_is_pinned(self):
        assert set(PINNED_BUILDS) == {
            "linux_x64_gnu",
            "linux_arm64_gnu",
            "linux_x64_musl",
            "linux_arm64_musl",
            "darwin_x64",
            "darwin_arm64",
        }

    @pytest.mark.parametrize("key", sorted(PINNED_BUILDS))
    def test_a_pin_is_an_https_archive_of_the_pinned_version(self, key):
        build = PINNED_BUILDS[key]
        assert build.url.startswith("https://")
        assert f"/v{NODE_VERSION}/node-v{NODE_VERSION}-" in build.url
        assert re.fullmatch(r"[0-9a-f]{64}", build.sha256)

    @pytest.mark.parametrize("key", sorted(PINNED_BUILDS))
    def test_only_musl_comes_from_the_unofficial_builds(self, key):
        host = urllib.parse.urlsplit(PINNED_BUILDS[key].url).hostname
        assert host == (
            "unofficial-builds.nodejs.org" if key.endswith("musl") else "nodejs.org"
        )

    def test_the_version_is_exact(self):
        assert re.fullmatch(r"\d+\.\d+\.\d+", NODE_VERSION)


class TestNpmUsesTheManagedNode:
    def test_it_runs_the_npm_of_the_node_in_the_tools_directory(
        self, tools, managed_node, fake_npm
    ):
        result = install_cli("demo", _npm_program(), platform=MUSL_X64)
        assert result.state == "installed"
        (call,) = _npm_calls(fake_npm)
        node_bin = tools / "node" / NODE_VERSION / "bin"
        assert Path(call["self"]) == node_bin / "npm"
        assert call["path"].split(os.pathsep)[0] == str(node_bin)
        assert len(managed_node.urls) == 1

    def test_the_node_is_installed_once_for_many_programs(
        self, tools, managed_node, fake_npm
    ):
        install_cli("demo", _npm_program(), platform=MUSL_X64)
        install_cli("other", _npm_program(), platform=MUSL_X64)
        assert len(managed_node.urls) == 1
        assert len(_npm_calls(fake_npm)) == 2

    def test_npm_keeps_its_home_and_cache_in_the_programs_directory(
        self, tools, managed_node, fake_npm
    ):
        install_cli("demo", _npm_program(), platform=MUSL_X64)
        (call,) = _npm_calls(fake_npm)
        assert call["home"] == str(tools / "demo" / ".home")
        assert call["cache"] == str(tools / "demo" / ".npm-cache")

    def test_a_failed_node_download_installs_no_program(
        self, tools, fake_npm, monkeypatch
    ):
        _pin_node(monkeypatch, _node_archive(fake_npm), sha="0" * 64)
        with pytest.raises(CliInstallError, match="sha256"):
            install_cli("demo", _npm_program(), platform=MUSL_X64)
        assert _npm_calls(fake_npm) == []
        assert not (tools / "demo" / "installed.json").exists()

    def test_a_program_whose_node_is_gone_is_broken_then_repaired(
        self, tools, managed_node, monkeypatch
    ):
        monkeypatch.setenv("FAKE_NPM_NEEDS_NODE", "1")
        program = _npm_program()
        install_cli("demo", program, platform=MUSL_X64)
        search = [tool_bin_dir("demo"), tools / "bin"]
        assert check_external(program, search=search).state == "ok"
        shutil.rmtree(tools / "node")
        (tools / "bin" / "node").unlink()
        # this host's own node must not stand in for the one that is gone
        monkeypatch.setenv("PATH", str(tools.parent / "empty"))
        assert check_external(program, search=search).state == "broken"
        again = install_cli("demo", program, platform=MUSL_X64)
        assert again.state == "installed"
        assert check_external(program, search=search).state == "ok"


class TestReleaseInstall:
    def test_it_downloads_checks_and_links(self, tools):
        blob = _archive({"antigravity": AGY_BODY})
        net = _Net(blob)
        result = _install(_release_program(blob), net=net)
        assert result.state == "installed"
        assert net.urls == ["https://example.test/agy/1.5.15/agy_linux_x64_musl.tar.gz"]
        link = tools / "demo" / "bin" / "agy"
        assert link.is_symlink() and os.access(link, os.X_OK)
        assert (
            check_external(_release_program(blob), search=[tool_bin_dir("demo")]).state
            == "ok"
        )

    def test_a_second_run_does_not_download_again(self, tools):
        blob = _archive({"antigravity": AGY_BODY})
        net = _Net(blob)
        _install(_release_program(blob), net=net)
        assert _install(_release_program(blob), net=net).state == "unchanged"
        assert len(net.urls) == 1

    def test_a_wrong_checksum_installs_nothing(self, tools):
        blob = _archive({"antigravity": AGY_BODY})
        with pytest.raises(CliInstallError, match="sha256"):
            _install(_release_program(blob, sha="0" * 64), net=_Net(blob))
        assert not any(p.is_file() or p.is_symlink() for p in tools.rglob("*"))

    def test_a_tampered_archive_installs_nothing(self, tools):
        good = _archive({"antigravity": AGY_BODY})
        evil = _archive({"antigravity": b"#!/bin/sh\necho pwned\n"})
        program = _release_program(good)
        with pytest.raises(CliInstallError, match="sha256"):
            _install(program, net=_Net(evil))
        assert not (tools / "demo" / "bin" / "agy").exists()

    def test_a_platform_without_a_checksum_is_refused_before_any_download(self, tools):
        blob = _archive({"antigravity": AGY_BODY})
        net = _Net(blob)
        with pytest.raises(CliInstallError, match="linux_x64_musl"):
            install_cli(
                "demo", _release_program(blob), platform="linux_arm64_gnu", fetch=net
            )
        assert net.urls == []

    @pytest.mark.parametrize(
        "members",
        [
            {"../escape": b"x", "antigravity": AGY_BODY},
            {"/abs/path": b"x", "antigravity": AGY_BODY},
            {"link": ("symlink", "../../outside"), "antigravity": AGY_BODY},
            {"link": ("hardlink", "/etc/passwd"), "antigravity": AGY_BODY},
        ],
        ids=["dotdot", "absolute", "symlink-out", "hardlink-out"],
    )
    def test_an_archive_that_leaves_its_directory_is_refused(self, tools, members):
        blob = _archive(members)
        with pytest.raises(CliInstallError, match="outside"):
            _install(_release_program(blob), net=_Net(blob))
        assert not (tools / "demo" / "bin" / "agy").exists()
        assert not (tools.parent / "escape").exists()

    def test_an_archive_without_the_declared_binary_is_a_failure(self, tools):
        blob = _archive({"something-else": AGY_BODY})
        with pytest.raises(CliInstallError, match="antigravity"):
            _install(_release_program(blob), net=_Net(blob))
        assert not (tools / "demo" / "installed.json").exists()

    def test_the_binary_is_made_executable(self, tools):
        blob = _archive({"antigravity": AGY_BODY}, mode=0o644)
        _install(_release_program(blob), net=_Net(blob))
        mode = (tools / "demo" / "release-1.5.15" / "antigravity").stat().st_mode
        assert mode & stat.S_IXUSR

    def test_the_default_fetch_refuses_a_URL_that_is_not_https(self, tmp_path):
        with pytest.raises(CliInstallError, match="https"):
            REAL_FETCH("http://example.test/a.tar.gz", tmp_path / "a")
        with pytest.raises(CliInstallError, match="https"):
            REAL_FETCH("file:///etc/passwd", tmp_path / "a")


class TestRefusals:
    def test_a_program_without_a_spec_names_its_hint(self, tools):
        program = ExternalProgram(
            "claude", ("claude", "--version"), install_hint="install it by hand"
        )
        with pytest.raises(CliInstallError, match="install it by hand"):
            _install(program)

    def test_root_is_refused(self, tools, monkeypatch):
        monkeypatch.setattr(os, "geteuid", lambda: 0, raising=False)
        with pytest.raises(CliInstallError, match="root"):
            _install(_npm_program())
        assert not tools.exists()

    def test_a_dry_run_changes_nothing(self, tools, fake_npm):
        result = _install(_npm_program(), fake_npm=fake_npm, dry_run=True)
        assert result.state == "would_install"
        assert _npm_calls(fake_npm) == []
        assert not tools.exists()


class TestTheCommand:
    @pytest.fixture
    def declared(self, monkeypatch):
        program = _npm_program()

        class _Manifest:
            external = (program,)

        monkeypatch.setattr(
            _cli,
            "_manifest_of",
            lambda ext_id: _Manifest() if ext_id == "demo" else None,
        )
        return program

    def _run(self, *args):
        return _cli.ext_main(["install-cli", *args])

    def test_it_installs_and_exits_zero(
        self, tools, managed_node, declared, monkeypatch, capsys
    ):
        assert self._run("demo") == 0
        assert "installed" in capsys.readouterr().out
        assert self._run("demo") == 0
        assert "unchanged" in capsys.readouterr().out

    def test_an_unknown_extension_exits_one(self, tools, declared, capsys):
        assert self._run("nope") == 1
        assert "nope" in capsys.readouterr().err

    def test_a_failure_exits_non_zero_and_says_why(
        self, tools, managed_node, declared, monkeypatch, capsys
    ):
        monkeypatch.setenv("FAKE_NPM_FAIL", "1")
        assert self._run("demo") == 1
        assert "npm" in capsys.readouterr().err

    def test_json_reports_the_state(
        self, tools, managed_node, declared, monkeypatch, capsys
    ):
        assert self._run("demo", "--json") == 0
        report = json.loads(capsys.readouterr().out)
        assert report["programs"][0]["state"] == "installed"
        assert report["programs"][0]["version"] == "2.1.299"

    def test_an_extension_with_no_install_spec_exits_one(
        self, tools, monkeypatch, capsys
    ):
        class _Manifest:
            external = (ExternalProgram("x", ("x", "--version")),)

        monkeypatch.setattr(_cli, "_manifest_of", lambda ext_id: _Manifest())
        assert self._run("demo") == 1
        assert "install" in capsys.readouterr().err


class TestDoctorSeesTheInstalledProgram:
    def test_a_program_installed_outside_path_is_ready_for_doctor(
        self, tools, fake_npm, tmp_path, monkeypatch
    ):
        program = _npm_program()
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        manifest = parse_extension_manifest(
            {
                "id": "demo",
                "name": "Demo",
                "version": "0.1.0",
                "maturity": "experimental",
                "requires": {"surface": SURFACE_VERSION, "ir": IR_VERSION},
                "provides": {"shims": ["demo--acp"]},
                "external": [
                    {
                        "name": "claude",
                        "check": ["claude", "--version"],
                        "install": {
                            "npm": "@anthropic-ai/claude-code",
                            "version": "2.1.299",
                        },
                    }
                ],
            }
        )
        args = ("demo", manifest, "loaded", None, {}, False)
        assert _cli._examine(*args)["external"][0]["state"] == "missing"
        _install(program, fake_npm=fake_npm)
        assert _cli._examine(*args)["external"][0]["state"] == "ok"


class TestOneDirectoryForPath:
    def test_every_entry_is_also_linked_in_the_shared_bin(
        self, tools, fake_npm, tmp_path, monkeypatch
    ):
        _install(_npm_program(), fake_npm=fake_npm)
        shared = tools / "bin" / "claude"
        assert shared.is_symlink() and shared.exists()
        monkeypatch.setenv("PATH", str(tools / "bin"))
        assert check_external(_npm_program()).state == "ok"

    def test_a_failed_check_removes_the_shared_link_too(
        self, tools, fake_npm, monkeypatch
    ):
        monkeypatch.setenv("FAKE_NPM_BIN", "claude")
        broken = ExternalProgram(
            name="claude",
            check=("claude", "--version"),
            minimum="9.9.9",
            install=ExternalInstall(version="2.1.299", npm="@anthropic-ai/claude-code"),
        )
        with pytest.raises(CliInstallError, match="too_old"):
            _install(broken, fake_npm=fake_npm)
        assert not (tools / "bin" / "claude").exists()
        assert not (tools / "bin" / "claude").is_symlink()
