"""The standard-library extension installer and directory manager (#1703 E5)."""

from __future__ import annotations

import hashlib
import http.server
import importlib
import os
import stat
import sys
import textwrap
import threading
import zipfile
from pathlib import Path

import pytest

from llm_rosetta.contracts.extension import ExternalProgram, parse_extension_manifest
from llm_rosetta.contracts import SURFACE_VERSION
from llm_rosetta.extensions import (
    ExtensionInstallError,
    _dirs,
    _install,
    activate_extensions_dir,
    check_external,
    configure_extensions_dir,
    extensions_dir,
    install_with_pip,
    install_wheel,
    install_wheel_url,
    installed_extensions,
    list_extensions,
    load_extensions,
    missing_dependencies,
    remove_extension,
)
from llm_rosetta.shims import list_shims
from llm_rosetta.shims._loader import _PLUGIN_SOURCES
from llm_rosetta.shims.provider_shim import unregister_shim


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """Leave sys.path, the namespace cache and the shim registry as found."""
    from llm_rosetta.extensions import _discovery

    before = {shim.name for shim in list_shims()}
    sources = dict(_PLUGIN_SOURCES)
    path = list(sys.path)
    monkeypatch.setattr(_discovery, "_entry_points", lambda group: [])
    monkeypatch.delenv("LLM_ROSETTA_EXTENSIONS_DIR", raising=False)
    configure_extensions_dir(None)
    yield
    configure_extensions_dir(None)
    sys.path[:] = path
    for shim in list_shims():
        if shim.name not in before:
            unregister_shim(shim.name)
    _PLUGIN_SOURCES.clear()
    _PLUGIN_SOURCES.update(sources)
    for name in [n for n in sys.modules if n.split(".")[0] == "llm_rosetta_ext"]:
        del sys.modules[name]
    importlib.invalidate_caches()


def _manifest_text(ext_id="acme", version="0.1.0", surface=None, **extra) -> str:
    surface = SURFACE_VERSION if surface is None else surface
    requires = extra.pop("requires_extra", "")
    return textwrap.dedent(f"""\
        id: {extra.pop("id", ext_id)}
        name: {ext_id} test
        version: {version}
        maturity: experimental
        requires:
          surface: {surface}
          ir: 1
        {requires}
        provides:
          shims: [{ext_id}--chat_completions]
        """)


def _wheel(
    tmp_path: Path,
    ext_id="acme",
    version="0.1.0",
    *,
    manifest=None,
    extra=None,
    name=None,
    dist_info=True,
    with_manifest=True,
) -> Path:
    """Build a real wheel; *extra* maps member name to bytes (or ZipInfo, bytes)."""
    path = tmp_path / (name or f"{ext_id}-{version}-py3-none-any.whl")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        base = f"llm_rosetta_ext/{ext_id}"
        if with_manifest:
            zf.writestr(
                f"{base}/plugin.yaml",
                manifest if manifest is not None else _manifest_text(ext_id, version),
            )
        zf.writestr(
            f"{base}/provider.yaml", "connection:\n  base_url: https://x.invalid\n"
        )
        zf.writestr(
            f"{base}/llm/chat_completions/provider.yaml",
            f"name: {ext_id}--chat_completions\nmodel_type: llm\n"
            "format: chat_completions\n",
        )
        zf.writestr(f"{base}/marker.txt", f"version {version}")
        if dist_info:
            zf.writestr(f"{ext_id}-{version}.dist-info/METADATA", "Name: x\n")
            zf.writestr(
                f"{ext_id}-{version}.dist-info/RECORD", f"{base}/marker.txt,,\n"
            )
        for member, payload in (extra or {}).items():
            zf.writestr(member, payload)
    return path


def _symlink_member(zf_path: Path, name: str) -> None:
    info = zipfile.ZipInfo(name)
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(zf_path, "a") as zf:
        zf.writestr(info, "/etc/passwd")


def _empty(dest: Path) -> bool:
    return not dest.exists() or not any(dest.iterdir())


class TestInstall:
    def test_happy_path(self, tmp_path):
        dest = tmp_path / "ext"
        wheel = _wheel(tmp_path)
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()

        item = install_wheel(wheel, sha256=digest, dest=dest)

        assert (item.id, item.version) == ("acme", "0.1.0")
        assert item.path == dest / "llm_rosetta_ext" / "acme"
        assert item.manifest is not None and item.manifest.id == "acme"
        assert (item.path / "marker.txt").read_text() == "version 0.1.0"
        assert (dest / "acme-0.1.0.dist-info" / "METADATA").is_file()
        assert not [p for p in dest.iterdir() if p.name.startswith(".")]

    def test_bytes_source(self, tmp_path):
        dest = tmp_path / "ext"
        item = install_wheel(_wheel(tmp_path).read_bytes(), dest=dest)
        assert item.id == "acme"

    def test_a_sha_mismatch_writes_nothing(self, tmp_path):
        dest = tmp_path / "ext"
        with pytest.raises(ExtensionInstallError, match="sha256"):
            install_wheel(_wheel(tmp_path), sha256="0" * 64, dest=dest)
        assert _empty(dest)

    def test_not_a_zip(self, tmp_path):
        with pytest.raises(ExtensionInstallError, match="not a valid wheel"):
            install_wheel(b"nope", dest=tmp_path / "ext")

    @pytest.mark.parametrize(
        "member",
        [
            "/etc/evil.py",
            "C:/evil.py",
            "llm_rosetta_ext/acme/../../evil.py",
            "llm_rosetta_ext\\acme\\..\\..\\evil.py",
            "evil.py",
            "acme-0.1.0.data/purelib/evil.py",
            "llm_rosetta_ext/evil.py",
            "llm_rosetta_ext/_private/x.py",
        ],
    )
    def test_a_bad_member_path_is_refused_and_nothing_is_written(
        self, tmp_path, member
    ):
        dest = tmp_path / "ext"
        wheel = _wheel(tmp_path, extra={member: "x"})
        with pytest.raises(ExtensionInstallError):
            install_wheel(wheel, dest=dest)
        assert _empty(dest)
        assert not (tmp_path / "evil.py").exists()

    @pytest.mark.parametrize(
        ("member", "word"),
        [
            ("/etc/evil.py", "absolute"),
            ("C:/evil.py", "absolute"),
            ("llm_rosetta_ext/acme/../../evil.py", "traversal"),
        ],
    )
    def test_the_refusal_names_its_reason(self, tmp_path, member, word):
        wheel = _wheel(tmp_path, extra={member: "x"})
        with pytest.raises(ExtensionInstallError, match=word):
            install_wheel(wheel, dest=tmp_path / "ext")

    def test_a_symlink_member_is_refused(self, tmp_path):
        dest = tmp_path / "ext"
        wheel = _wheel(tmp_path)
        _symlink_member(wheel, "llm_rosetta_ext/acme/link")
        with pytest.raises(ExtensionInstallError, match="symlink"):
            install_wheel(wheel, dest=dest)
        assert _empty(dest)

    def test_a_duplicate_member_is_refused(self, tmp_path):
        with pytest.warns(UserWarning):  # zipfile warns on the duplicate name
            wheel = _wheel(tmp_path, extra={"llm_rosetta_ext/acme/marker.txt": "again"})
        with pytest.raises(ExtensionInstallError, match="twice"):
            install_wheel(wheel, dest=tmp_path / "ext")

    def test_two_extension_ids_in_one_wheel(self, tmp_path):
        dest = tmp_path / "ext"
        wheel = _wheel(tmp_path, extra={"llm_rosetta_ext/other/plugin.yaml": "x"})
        with pytest.raises(ExtensionInstallError, match="exactly one extension"):
            install_wheel(wheel, dest=dest)
        assert _empty(dest)

    def test_no_extension_in_the_wheel(self, tmp_path):
        path = tmp_path / "x.whl"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("x-1.dist-info/METADATA", "")
        with pytest.raises(ExtensionInstallError, match="exactly one extension"):
            install_wheel(path, dest=tmp_path / "ext")

    def test_two_dist_infos_are_refused(self, tmp_path):
        wheel = _wheel(tmp_path, extra={"zzz-1.0.dist-info/METADATA": ""})
        with pytest.raises(ExtensionInstallError, match="dist-info"):
            install_wheel(wheel, dest=tmp_path / "ext")

    def test_a_missing_plugin_yaml(self, tmp_path):
        dest = tmp_path / "ext"
        with pytest.raises(ExtensionInstallError, match="plugin.yaml"):
            install_wheel(_wheel(tmp_path, with_manifest=False), dest=dest)
        assert _empty(dest)

    def test_an_invalid_manifest(self, tmp_path):
        wheel = _wheel(tmp_path, manifest="id: acme\n")
        with pytest.raises(ExtensionInstallError, match="invalid extension manifest"):
            install_wheel(wheel, dest=tmp_path / "ext")

    def test_an_unparsable_manifest(self, tmp_path):
        wheel = _wheel(tmp_path, manifest=b"\xff\xfe not utf-8")
        with pytest.raises(ExtensionInstallError, match="plugin.yaml"):
            install_wheel(wheel, dest=tmp_path / "ext")

    def test_an_incompatible_surface_raises_before_extraction(self, tmp_path):
        dest = tmp_path / "ext"
        wheel = _wheel(tmp_path, manifest=_manifest_text(surface=SURFACE_VERSION + 50))
        with pytest.raises(ExtensionInstallError, match="cannot run here"):
            install_wheel(wheel, dest=dest)
        assert _empty(dest)

    def test_the_manifest_id_must_equal_the_folder(self, tmp_path):
        dest = tmp_path / "ext"
        wheel = _wheel(tmp_path, manifest=_manifest_text("other"))
        with pytest.raises(ExtensionInstallError, match="folder"):
            install_wheel(wheel, dest=dest)
        assert _empty(dest)

    def test_the_unpacked_size_cap(self, tmp_path, monkeypatch):
        dest = tmp_path / "ext"
        wheel = _wheel(tmp_path, extra={"llm_rosetta_ext/acme/big.bin": "x" * 200000})
        monkeypatch.setattr(_install, "MAX_UNPACKED_BYTES", wheel.stat().st_size + 10)

        def unexpected(*args):
            raise AssertionError("the cap must refuse before anything is extracted")

        monkeypatch.setattr(_install, "_extract", unexpected)
        with pytest.raises(ExtensionInstallError, match="size cap"):
            install_wheel(wheel, dest=dest)
        assert _empty(dest)

    def test_the_member_count_cap(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_install, "MAX_MEMBERS", 3)
        with pytest.raises(ExtensionInstallError, match="too many members"):
            install_wheel(_wheel(tmp_path), dest=tmp_path / "ext")

    def test_replacing_an_older_version_leaves_exactly_one(self, tmp_path):
        dest = tmp_path / "ext"
        install_wheel(_wheel(tmp_path, version="0.1.0"), dest=dest)
        extra = {"llm_rosetta_ext/acme/new.txt": "n"}
        install_wheel(_wheel(tmp_path, version="0.2.0", extra=extra), dest=dest)

        folder = dest / "llm_rosetta_ext" / "acme"
        assert (folder / "marker.txt").read_text() == "version 0.2.0"
        assert (folder / "new.txt").is_file()
        assert sorted(p.name for p in dest.glob("*.dist-info")) == [
            "acme-0.2.0.dist-info"
        ]
        assert [i.version for i in installed_extensions(dest)] == ["0.2.0"]

    def test_a_replacement_drops_files_the_new_version_lacks(self, tmp_path):
        dest = tmp_path / "ext"
        install_wheel(
            _wheel(tmp_path, extra={"llm_rosetta_ext/acme/old.txt": "o"}), dest=dest
        )
        install_wheel(_wheel(tmp_path, version="0.2.0"), dest=dest)
        assert not (dest / "llm_rosetta_ext" / "acme" / "old.txt").exists()

    def test_a_failed_swap_restores_the_previous_version(self, tmp_path, monkeypatch):
        dest = tmp_path / "ext"
        install_wheel(_wheel(tmp_path, version="0.1.0"), dest=dest)
        real = os.rename
        calls = []

        def flaky(src, dst):
            calls.append(dst)
            if str(dst).endswith("acme-0.2.0.dist-info"):
                raise OSError("disk says no")
            return real(src, dst)

        monkeypatch.setattr(_install.os, "rename", flaky)
        with pytest.raises(OSError):
            install_wheel(_wheel(tmp_path, version="0.2.0"), dest=dest)
        monkeypatch.undo()
        assert [i.version for i in installed_extensions(dest)] == ["0.1.0"]
        assert (dest / "acme-0.1.0.dist-info").is_dir()

    def test_concurrent_installs_of_one_id_serialize(self, tmp_path, monkeypatch):
        dest = tmp_path / "ext"
        wheels = [
            _wheel(tmp_path, version="0.1.0"),
            _wheel(tmp_path, version="0.2.0"),
        ]
        state = {"inside": 0, "max": 0}
        real_swap = _install._swap_in
        guard = threading.Lock()

        def slow_swap(*args):
            import time

            with guard:
                state["inside"] += 1
                state["max"] = max(state["max"], state["inside"])
            time.sleep(0.2)
            try:
                return real_swap(*args)
            finally:
                with guard:
                    state["inside"] -= 1

        monkeypatch.setattr(_install, "_swap_in", slow_swap)
        errors = []

        def run(wheel):
            try:
                install_wheel(wheel, dest=dest)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=run, args=(w,)) for w in wheels]
        [t.start() for t in threads]
        [t.join() for t in threads]

        assert not errors and state["max"] == 1
        assert len(list(dest.glob("*.dist-info"))) == 1
        assert len(installed_extensions(dest)) == 1

    def test_a_held_lock_blocks_then_times_out(self, tmp_path, monkeypatch):
        dest = tmp_path / "ext"
        dest.mkdir()
        (dest / _install._LOCK_NAME).write_text("1")
        monkeypatch.setattr(_install, "LOCK_WAIT_SECONDS", 0.1)
        with pytest.raises(ExtensionInstallError, match="holds"):
            install_wheel(_wheel(tmp_path), dest=dest)

    def test_a_stale_lock_is_taken_over(self, tmp_path, monkeypatch):
        dest = tmp_path / "ext"
        dest.mkdir()
        lock = dest / _install._LOCK_NAME
        lock.write_text("1")
        os.utime(lock, (1, 1))
        assert install_wheel(_wheel(tmp_path), dest=dest).id == "acme"
        assert not lock.exists()


class _Handler(http.server.BaseHTTPRequestHandler):
    payload = b""
    redirect_to: str | None = None

    def do_GET(self):  # noqa: N802
        if self.path == "/redirect" and self.redirect_to:
            self.send_response(302)
            self.send_header("Location", self.redirect_to)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, format, *args):
        pass


@pytest.fixture
def server():
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


class TestInstallUrl:
    def test_downloads_and_installs(self, tmp_path, server):
        httpd, base = server
        data = _wheel(tmp_path).read_bytes()
        _Handler.payload = data
        dest = tmp_path / "ext"
        item = install_wheel_url(
            f"{base}/a.whl", sha256=hashlib.sha256(data).hexdigest(), dest=dest
        )
        assert item.id == "acme" and item.path.is_dir()

    def test_a_sha_is_required(self, tmp_path):
        with pytest.raises(ExtensionInstallError, match="sha256 is required"):
            install_wheel_url("https://example.com/a.whl", sha256="", dest=tmp_path)

    def test_a_wrong_sha_writes_nothing(self, tmp_path, server):
        _Handler.payload = _wheel(tmp_path).read_bytes()
        dest = tmp_path / "ext"
        with pytest.raises(ExtensionInstallError, match="sha256 mismatch"):
            install_wheel_url(f"{server[1]}/a.whl", sha256="0" * 64, dest=dest)
        assert _empty(dest)

    def test_plain_http_to_a_remote_host_is_refused(self, tmp_path):
        with pytest.raises(ExtensionInstallError, match="plain http"):
            install_wheel_url("http://example.com/a.whl", sha256="0" * 64)

    def test_another_scheme_is_refused(self):
        with pytest.raises(ExtensionInstallError, match="scheme"):
            install_wheel_url("file:///etc/passwd", sha256="0" * 64)

    def test_the_download_is_capped_while_streaming(
        self, tmp_path, server, monkeypatch
    ):
        data = _wheel(tmp_path).read_bytes()
        _Handler.payload = data
        monkeypatch.setattr(_install, "MAX_UNPACKED_BYTES", 100)
        with pytest.raises(ExtensionInstallError, match="cap"):
            install_wheel_url(
                f"{server[1]}/a.whl",
                sha256=hashlib.sha256(data).hexdigest(),
                dest=tmp_path / "ext",
            )

    def test_a_redirect_to_plain_http_elsewhere_is_refused(self, tmp_path, server):
        _Handler.redirect_to = "http://example.com/a.whl"
        try:
            with pytest.raises(ExtensionInstallError, match="refused redirect"):
                install_wheel_url(
                    f"{server[1]}/redirect", sha256="0" * 64, dest=tmp_path / "ext"
                )
        finally:
            _Handler.redirect_to = None

    def test_a_connection_failure_is_an_install_error(self, tmp_path):
        with pytest.raises(ExtensionInstallError, match="cannot download"):
            install_wheel_url(
                "http://127.0.0.1:9/a.whl", sha256="0" * 64, dest=tmp_path, timeout=2
            )


class TestManage:
    def test_remove(self, tmp_path):
        dest = tmp_path / "ext"
        install_wheel(_wheel(tmp_path), dest=dest)
        keep = dest / "unrelated.txt"
        keep.write_text("keep")

        assert remove_extension("acme", dest=dest) is True

        assert not (dest / "llm_rosetta_ext" / "acme").exists()
        assert not list(dest.glob("*.dist-info"))
        assert keep.read_text() == "keep"
        assert installed_extensions(dest) == []

    def test_removing_what_is_not_installed(self, tmp_path):
        dest = tmp_path / "ext"
        assert remove_extension("acme", dest=dest) is False
        install_wheel(_wheel(tmp_path), dest=dest)
        assert remove_extension("ghost", dest=dest) is False
        assert (dest / "llm_rosetta_ext" / "acme").is_dir()

    @pytest.mark.parametrize("bad", ["../x", "a/b", "", "..", "_x", "/etc"])
    def test_remove_refuses_a_path_shaped_id(self, tmp_path, bad):
        outside = tmp_path / "x"
        outside.mkdir()
        with pytest.raises(ExtensionInstallError, match="not an extension id"):
            remove_extension(bad, dest=tmp_path / "ext")
        assert outside.is_dir()

    def test_remove_never_follows_a_symlinked_folder(self, tmp_path):
        dest = tmp_path / "ext"
        (dest / "llm_rosetta_ext").mkdir(parents=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "keep.txt").write_text("k")
        (dest / "llm_rosetta_ext" / "acme").symlink_to(outside)
        assert remove_extension("acme", dest=dest) is True
        assert (outside / "keep.txt").read_text() == "k"

    def test_installed_extensions_reports_a_broken_manifest(self, tmp_path):
        dest = tmp_path / "ext"
        install_wheel(_wheel(tmp_path), dest=dest)
        bad = dest / "llm_rosetta_ext" / "broken"
        bad.mkdir()
        (bad / "plugin.yaml").write_text("id: broken\n")
        nomanifest = dest / "llm_rosetta_ext" / "bare"
        nomanifest.mkdir()
        (dest / "llm_rosetta_ext" / "__pycache__").mkdir()

        found = {i.id: i for i in installed_extensions(dest)}

        assert set(found) == {"acme", "broken", "bare"}
        assert found["acme"].manifest is not None and found["acme"].reason is None
        assert found["broken"].manifest is None and "invalid" in (
            found["broken"].reason or ""
        )
        assert found["bare"].manifest is None and "plugin.yaml" in (
            found["bare"].reason or ""
        )
        assert found["broken"].version is None

    def test_a_missing_directory_lists_nothing(self, tmp_path):
        assert installed_extensions(tmp_path / "nope") == []


class TestActivation:
    def test_a_fresh_install_is_importable_and_loadable_without_a_restart(
        self, tmp_path
    ):
        dest = tmp_path / "ext"
        dest.mkdir()
        configure_extensions_dir(dest)
        # Import the (empty) namespace first so a stale cache would hide the install.
        load_extensions()
        assert not list_extensions()

        install_wheel(_wheel(tmp_path), dest=dest)
        activate_extensions_dir()

        assert sys.path[0] == str(dest)
        assert importlib.import_module("llm_rosetta_ext.acme").__path__
        sys.modules.pop("llm_rosetta_ext.acme")
        load_extensions()
        status = next(s for s in list_extensions() if s.id == "acme")
        assert status.state == "loaded"

    def test_load_extensions_activates_an_existing_directory(self, tmp_path):
        dest = tmp_path / "ext"
        install_wheel(_wheel(tmp_path), dest=dest)
        configure_extensions_dir(dest)
        load_extensions()
        assert next(s for s in list_extensions() if s.id == "acme").state == "loaded"

    def test_load_extensions_leaves_a_missing_directory_alone(self, tmp_path):
        configure_extensions_dir(tmp_path / "absent")
        before = list(sys.path)
        load_extensions()
        assert sys.path == before

    def test_activation_is_idempotent(self, tmp_path):
        dest = tmp_path / "ext"
        dest.mkdir()
        activate_extensions_dir(dest)
        activate_extensions_dir(dest)
        assert sys.path.count(str(dest)) == 1 and sys.path[0] == str(dest)


class TestDirectory:
    def test_precedence_configured_then_env_then_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
        monkeypatch.setattr(_dirs.sys, "platform", "linux")
        assert extensions_dir() == tmp_path / "xdg" / "llm-rosetta" / "extensions"
        monkeypatch.setenv("LLM_ROSETTA_EXTENSIONS_DIR", str(tmp_path / "env"))
        assert extensions_dir() == tmp_path / "env"
        configure_extensions_dir(tmp_path / "explicit")
        assert extensions_dir() == tmp_path / "explicit"

    def test_the_linux_default_is_under_the_home(self, tmp_path, monkeypatch):
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(_dirs.sys, "platform", "linux")
        assert extensions_dir() == (
            tmp_path / ".local" / "share" / "llm-rosetta" / "extensions"
        )

    def test_the_macos_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(_dirs.sys, "platform", "darwin")
        assert extensions_dir() == (
            tmp_path / "Library" / "Application Support" / "llm-rosetta" / "extensions"
        )

    def test_the_windows_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        monkeypatch.setattr(_dirs.sys, "platform", "win32")
        assert extensions_dir() == tmp_path / "llm-rosetta" / "extensions"

    def test_home_slash_fails_with_a_message_naming_the_variable(self, monkeypatch):
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.setenv("HOME", "/")
        monkeypatch.setattr(_dirs.sys, "platform", "linux")
        with pytest.raises(ExtensionInstallError, match="LLM_ROSETTA_EXTENSIONS_DIR"):
            extensions_dir()

    def test_load_extensions_survives_an_unusable_home(self, monkeypatch):
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.setenv("HOME", "/")
        monkeypatch.setattr(_dirs.sys, "platform", "linux")
        load_extensions()


def _stub(tmp_path: Path, name: str, body: str) -> tuple[str, ...]:
    path = tmp_path / f"{name}.py"
    path.write_text(body)
    return (sys.executable, str(path))


class TestDoctor:
    def test_ok(self, tmp_path):
        check = _stub(tmp_path, "tool", "print('tool v2.10.1 (build 7)')")
        status = check_external(ExternalProgram("tool", check, minimum="2.9"))
        assert status.state == "ok" and status.found == "2.10.1"

    def test_the_comparison_is_numeric_not_textual(self, tmp_path):
        check = _stub(tmp_path, "tool", "print('2.10.0')")
        assert (
            check_external(ExternalProgram("t", check, minimum="2.9.0")).state == "ok"
        )

    def test_too_old(self, tmp_path):
        check = _stub(tmp_path, "tool", "import sys; print('1.4.2', file=sys.stderr)")
        status = check_external(ExternalProgram("tool", check, minimum="1.5"))
        assert status.state == "too_old" and status.found == "1.4.2"
        assert "1.5" in (status.detail or "")

    def test_missing(self, tmp_path):
        program = ExternalProgram("nope", (str(tmp_path / "no-such-binary"),))
        assert check_external(program).state == "missing"

    def test_a_timeout_is_missing(self, tmp_path):
        check = _stub(tmp_path, "slow", "import time; time.sleep(5)")
        status = check_external(ExternalProgram("slow", check), timeout=0.2)
        assert status.state == "missing" and "timed out" in (status.detail or "")

    def test_no_version_with_a_minimum_is_unknown(self, tmp_path):
        check = _stub(tmp_path, "tool", "print('hello')")
        assert (
            check_external(ExternalProgram("t", check, minimum="1.0")).state
            == "unknown"
        )
        assert check_external(ExternalProgram("t", check)).state == "ok"

    def test_no_shell_is_used(self, tmp_path):
        marker = tmp_path / "pwned"
        program = ExternalProgram("t", (f"echo hi; touch {marker}",))
        assert check_external(program).state == "missing"
        assert not marker.exists()

    def test_missing_dependencies(self):
        manifest = parse_extension_manifest(
            {
                "id": "acme",
                "name": "a",
                "version": "1.0.0",
                "maturity": "experimental",
                "requires": {
                    "surface": SURFACE_VERSION,
                    "ir": 1,
                    "extensions": ["base", "other"],
                },
                "provides": {"transports": ["acme_t"]},
            }
        )
        base = parse_extension_manifest(
            {
                "id": "base",
                "name": "b",
                "version": "1.0.0",
                "maturity": "experimental",
                "requires": {"surface": SURFACE_VERSION, "ir": 1},
                "provides": {"transports": ["base_t"]},
            }
        )
        assert missing_dependencies(manifest, [base, None]) == ["other"]
        assert missing_dependencies(manifest, {"base": base, "other": base}) == []
        assert missing_dependencies(manifest, []) == ["base", "other"]


class TestPip:
    def test_without_pip_it_refuses(self, monkeypatch, tmp_path):
        monkeypatch.setattr(_install, "pip_available", lambda: False)
        with pytest.raises(ExtensionInstallError, match="pip is not available"):
            install_with_pip("x", dest=tmp_path)

    def test_the_command_line(self, monkeypatch, tmp_path):
        seen = {}

        class Done:
            returncode = 3

        monkeypatch.setattr(_install, "pip_available", lambda: True)
        monkeypatch.setattr(
            _install.subprocess, "run", lambda cmd, **kw: seen.update(cmd=cmd) or Done()
        )
        assert install_with_pip("pkg==1", dest=tmp_path) == 3
        assert seen["cmd"] == [
            sys.executable, "-m", "pip", "install", "--no-deps", "--upgrade",
            "--target", str(tmp_path), "pkg==1",
        ]  # fmt: skip
