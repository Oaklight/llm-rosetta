"""The extension loader (#1703 E4): discovery, gating, registration, isolation."""

from __future__ import annotations

import dataclasses
import importlib
import logging
import sys
import textwrap
from pathlib import Path

import pytest

from llm_rosetta.contracts import SURFACE_VERSION
from llm_rosetta.extensions import (
    configure_extensions_dir,
    extensions_current,
    register_kind_handler,
    KindLoadError,
    _discovery,
    _load,
    list_extensions,
    load_extensions,
)
from llm_rosetta.shims import get_shim, list_shims
from llm_rosetta.shims._loader import plugin_source
from llm_rosetta.shims._loader import _PLUGIN_SOURCES
from llm_rosetta.shims.provider_shim import unregister_shim


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    """Leave the live registry, provenance map and namespace cache as found."""
    before = {shim.name: shim for shim in list_shims()}
    sources = dict(_PLUGIN_SOURCES)
    monkeypatch.setattr(_discovery, "_entry_points", lambda group: [])
    yield
    for shim in list_shims():
        if shim.name not in before:
            unregister_shim(shim.name)
    _PLUGIN_SOURCES.clear()
    _PLUGIN_SOURCES.update(sources)
    for name in [n for n in sys.modules if n.split(".")[0] == "llm_rosetta_ext"]:
        del sys.modules[name]
    importlib.invalidate_caches()


def _manifest(ext_id: str, **override) -> str:
    shims = override.pop("shims", [f"{ext_id}--chat_completions"])
    surface = override.pop("surface", SURFACE_VERSION)
    return textwrap.dedent(f"""\
        id: {override.pop("id", ext_id)}
        name: {ext_id} test
        version: 0.1.0
        maturity: experimental
        requires:
          surface: {surface}
          ir: 1
        provides:
          shims: [{", ".join(shims)}]
        """)


def _extension(root: Path, ext_id: str, *, manifest: str | None = None, leaf=None):
    folder = root / "llm_rosetta_ext" / ext_id
    (folder / "llm" / "chat_completions").mkdir(parents=True)
    (folder / "plugin.yaml").write_text(
        manifest if manifest is not None else _manifest(ext_id)
    )
    (folder / "provider.yaml").write_text(
        "connection:\n  base_url: https://example.invalid\n"
    )
    (folder / "llm" / "chat_completions" / "provider.yaml").write_text(
        leaf
        if leaf is not None
        else f"name: {ext_id}--chat_completions\nmodel_type: llm\nformat: chat_completions\n"
    )
    return folder


@pytest.fixture
def path(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    return tmp_path


def _status(ext_id: str):
    return next(s for s in list_extensions() if s.id == ext_id)


class TestKindHandlers:
    """The loader knows no shim: the shim layer registers a handler for its kind."""

    def test_the_shim_layer_registers_the_shims_kind(self):
        assert "shims" in _load._KIND_HANDLERS

    def test_an_extension_is_skipped_when_no_handler_owns_its_kind(
        self, path, monkeypatch
    ):
        _extension(path, "acme")
        monkeypatch.delitem(_load._KIND_HANDLERS, "shims")

        assert load_extensions() == []

        status = _status("acme")
        assert status.state == "skipped"
        assert "no shim layer" in status.reason
        assert get_shim("acme--chat_completions") is None

    def test_a_handler_refusal_fails_that_extension_with_its_message(
        self, path, monkeypatch
    ):
        _extension(path, "acme")
        _extension(path, "other")

        def refuse(manifest, folder):
            if manifest.id == "acme":
                raise KindLoadError("nope, by handler")
            return []

        monkeypatch.setitem(_load._KIND_HANDLERS, "shims", refuse)

        load_extensions()

        assert _status("acme").state == "failed"
        assert _status("acme").reason == "nope, by handler"
        assert _status("other").state == "loaded"

    def test_a_handler_error_is_named_and_does_not_stop_the_rest(
        self, path, monkeypatch
    ):
        _extension(path, "acme")

        def explode(manifest, folder):
            raise RuntimeError("boom")

        monkeypatch.setitem(_load._KIND_HANDLERS, "shims", explode)

        load_extensions()

        assert _status("acme").reason == "RuntimeError: boom"


def _commands_extension(root: Path, ext_id: str) -> Path:
    folder = _extension(root, ext_id)
    manifest = (folder / "plugin.yaml").read_text()
    (folder / "plugin.yaml").write_text(
        manifest + "  commands:\n    run: {en: Run it}\n"
    )
    return folder


class TestKindDispatch:
    """Every kind an extension provides reaches the handler its layer registered."""

    @pytest.fixture(autouse=True)
    def _handlers(self):
        handlers, undo = dict(_load._KIND_HANDLERS), dict(_load._KIND_UNDO)
        yield
        _load._KIND_HANDLERS.clear()
        _load._KIND_HANDLERS.update(handlers)
        _load._KIND_UNDO.clear()
        _load._KIND_UNDO.update(undo)

    def test_a_handler_runs_only_for_a_kind_the_extension_provides(self, path):
        _extension(path, "plain")
        _commands_extension(path, "cmds")
        seen = []
        register_kind_handler("commands", lambda m, f: seen.append(m.id) or [])

        load_extensions()

        assert seen == ["cmds"]

    def test_a_kind_without_a_handler_needs_none(self, path):
        _commands_extension(path, "cmds")
        _load._KIND_HANDLERS.pop("commands", None)

        load_extensions()

        assert _status("cmds").state == "loaded"

    def test_a_later_kind_that_fails_undoes_the_shims_it_follows(self, path):
        _commands_extension(path, "cmds")

        def refuse(manifest, folder):
            raise KindLoadError("no area for you")

        register_kind_handler("commands", refuse)

        load_extensions()

        status = _status("cmds")
        assert (status.state, status.reason) == ("failed", "no area for you")
        assert status.shims == ()
        assert get_shim("cmds--chat_completions") is None
        assert plugin_source("cmds--chat_completions") is None

    def test_an_undo_that_raises_does_not_hide_the_failure(self, path):
        _commands_extension(path, "cmds")

        def undo_fails(manifest):
            raise ZeroDivisionError("division by zero")

        def handler_fails(manifest, folder):
            raise ZeroDivisionError("division by zero")

        register_kind_handler("shims", _load._KIND_HANDLERS["shims"], undo=undo_fails)
        register_kind_handler("commands", handler_fails)

        load_extensions()

        assert _status("cmds").reason == "ZeroDivisionError: division by zero"

    def test_a_handler_registered_after_the_load_is_applied_to_loaded_extensions(
        self, path
    ):
        _commands_extension(path, "cmds")
        load_extensions()
        seen = []

        register_kind_handler(
            "commands", lambda m, f: seen.append((m.id, f.name)) or []
        )

        assert seen == [("cmds", "cmds")]
        assert _status("cmds").state == "loaded"

    def test_a_late_handler_that_refuses_fails_the_extension_and_undoes_its_shims(
        self, path
    ):
        _commands_extension(path, "cmds")
        load_extensions()
        assert get_shim("cmds--chat_completions") is not None

        def refuse(manifest, folder):
            raise KindLoadError("too late")

        register_kind_handler("commands", refuse)

        status = _status("cmds")
        assert (status.state, status.reason) == ("failed", "too late")
        assert get_shim("cmds--chat_completions") is None

    def test_a_late_handler_does_not_touch_an_extension_that_is_not_loaded(self, path):
        _commands_extension(path, "cmds")
        folder = path / "llm_rosetta_ext" / "cmds"
        text = (folder / "plugin.yaml").read_text()
        text = text.replace(f"surface: {SURFACE_VERSION}", "surface: 9999")
        (folder / "plugin.yaml").write_text(text)
        load_extensions()
        assert _status("cmds").state == "skipped"
        seen = []

        register_kind_handler("commands", lambda m, f: seen.append(m.id) or [])

        assert seen == []

    def test_an_unknown_kind_is_refused(self):
        with pytest.raises(ValueError, match="not an extension kind"):
            register_kind_handler("widgets", lambda m, f: [])

    def test_the_loader_reports_that_it_has_run(self, path):
        load_extensions()
        assert extensions_current() is True

    def test_a_directory_chosen_after_the_load_makes_it_stale(self, path, tmp_path):
        load_extensions()
        other = tmp_path / "elsewhere"
        other.mkdir()
        configure_extensions_dir(other)
        try:
            assert extensions_current() is False
            load_extensions()
            assert extensions_current() is True
        finally:
            configure_extensions_dir(None)


class TestLoading:
    def test_an_extension_registers_its_shims_and_is_attributed(self, path):
        _extension(path, "acme")

        loaded = load_extensions()

        assert [s.name for s in loaded] == ["acme--chat_completions"]
        assert get_shim("acme--chat_completions") is not None
        assert plugin_source("acme--chat_completions") == "acme 0.1.0"
        status = _status("acme")
        assert status.state == "loaded" and status.reason is None
        assert status.shims == ("acme--chat_completions",)
        assert status.source == "namespace"

    def test_the_vendor_manifest_reaches_the_leaf(self, path):
        _extension(path, "acme")
        load_extensions()
        shim = get_shim("acme--chat_completions")
        assert shim is not None
        assert shim.connection.base_url == "https://example.invalid"

    def test_a_transport_only_extension_loads_without_shims(self, path):
        folder = path / "llm_rosetta_ext" / "lane"
        folder.mkdir(parents=True)
        (folder / "plugin.yaml").write_text(
            textwrap.dedent("""\
                id: lane
                name: lane
                version: 0.1.0
                maturity: beta
                requires: {surface: 1, ir: 1}
                provides: {transports: [lane]}
                """).replace("surface: 1", f"surface: {SURFACE_VERSION}")
        )
        assert load_extensions() == []
        assert _status("lane").state == "loaded"

    def test_no_extension_installed_is_not_an_error(self):
        assert load_extensions() == []
        assert list_extensions() == ()

    def test_the_status_is_rebuilt_on_every_load(self, path):
        _extension(path, "acme")
        load_extensions()
        assert [s.id for s in list_extensions()] == ["acme"]
        (path / "llm_rosetta_ext" / "acme" / "plugin.yaml").unlink()
        (path / "llm_rosetta_ext" / "acme" / "provider.yaml").unlink()
        unregister_shim("acme--chat_completions")
        for name in [n for n in sys.modules if n.split(".")[0] == "llm_rosetta_ext"]:
            del sys.modules[name]
        importlib.invalidate_caches()
        load_extensions()
        assert list_extensions() == ()


class TestReload:
    def test_loading_twice_replaces_the_extensions_own_shims(self, path):
        _extension(path, "acme")

        load_extensions()
        again = load_extensions()

        assert [s.name for s in again] == ["acme--chat_completions"]
        status = _status("acme")
        assert status.state == "loaded" and status.reason is None
        assert get_shim("acme--chat_completions") is not None
        assert plugin_source("acme--chat_completions") == "acme 0.1.0"

    def test_a_reload_after_the_extension_broke_removes_its_shims(self, path):
        folder = _extension(path, "acme")
        load_extensions()
        (folder / "plugin.yaml").write_text("id: acme\nname: x\n")

        load_extensions()

        assert _status("acme").state == "failed"
        # the manifest could not be read, so the old registration is not kept
        # on its word; the next successful load registers it again
        (folder / "plugin.yaml").write_text(_manifest("acme"))
        load_extensions()
        assert _status("acme").state == "loaded"

    def test_a_reload_keeps_a_name_that_another_extension_holds(self, path):
        # ``aaa_x--chat`` belongs to ``aaa`` (a lane form) and to ``aaa_x``
        # (the id itself): both may declare it, and only one can own it.
        _extension(
            path,
            "aaa",
            manifest=_manifest("aaa", shims=["aaa_x--chat_completions"]),
            leaf="name: aaa_x--chat_completions\nmodel_type: llm\nformat: chat_completions\n",
        )
        _extension(
            path,
            "aaa_x",
            manifest=_manifest("aaa_x", shims=["aaa_x--chat_completions"]),
            leaf="name: aaa_x--chat_completions\nmodel_type: llm\nformat: chat_completions\n",
        )

        for _ in range(2):
            load_extensions()
            assert _status("aaa").state == "loaded"
            second = _status("aaa_x")
            assert second.state == "failed"
            assert "already registered" in (second.reason or "")
            assert plugin_source("aaa_x--chat_completions") == "aaa 0.1.0"


class TestGating:
    def test_an_incompatible_surface_is_skipped_and_registers_nothing(self, path):
        _extension(
            path, "acme", manifest=_manifest("acme", surface=SURFACE_VERSION + 1)
        )

        assert load_extensions() == []

        status = _status("acme")
        assert status.state == "skipped"
        assert "surface" in (status.reason or "")
        assert get_shim("acme--chat_completions") is None

    def test_an_invalid_manifest_fails_with_the_problem(self, path):
        _extension(path, "acme", manifest="id: acme\nname: x\n")
        load_extensions()
        status = _status("acme")
        assert status.state == "failed"
        assert "invalid extension manifest" in (status.reason or "")

    def test_a_manifest_id_that_differs_from_the_folder_fails(self, path):
        _extension(
            path, "acme", manifest=_manifest("acme", id="other", shims=["other--x"])
        )
        load_extensions()
        status = next(s for s in list_extensions() if s.path.name == "acme")
        assert status.state == "failed" and "folder" in (status.reason or "")
        assert get_shim("other--x") is None

    def test_a_dir_with_provider_yaml_but_no_manifest_is_named(self, path):
        folder = _extension(path, "acme")
        (folder / "plugin.yaml").unlink()
        load_extensions()
        status = _status("acme")
        assert status.state == "failed" and "no plugin.yaml" in (status.reason or "")
        assert get_shim("acme--chat_completions") is None

    def test_one_bad_extension_does_not_stop_the_others(self, path):
        _extension(path, "aaa", manifest="not: [valid")
        _extension(path, "bbb")
        load_extensions()
        assert _status("aaa").state == "failed"
        assert _status("bbb").state == "loaded"
        assert get_shim("bbb--chat_completions") is not None


class TestIsolation:
    def test_a_declared_shim_that_is_not_provided_rolls_the_extension_back(self, path):
        _extension(
            path,
            "acme",
            manifest=_manifest(
                "acme", shims=["acme--chat_completions", "acme--missing"]
            ),
        )
        load_extensions()
        status = _status("acme")
        assert status.state == "failed"
        assert "acme--missing" in (status.reason or "")
        assert get_shim("acme--chat_completions") is None
        assert plugin_source("acme--chat_completions") is None

    def test_a_sidecar_that_raises_rolls_the_extension_back(self, path):
        folder = _extension(
            path,
            "acme",
            manifest=_manifest(
                "acme", shims=["acme--chat_completions", "acme--messages"]
            ),
        )
        second = folder / "llm" / "messages"
        second.mkdir()
        (second / "provider.yaml").write_text(
            "name: acme--messages\nmodel_type: llm\nformat: messages\n"
        )
        (second / "transforms.py").write_text("raise RuntimeError('boom')\n")

        load_extensions()

        status = _status("acme")
        assert status.state == "failed"
        assert "RuntimeError" in (status.reason or "") and "boom" in (
            status.reason or ""
        )
        assert get_shim("acme--chat_completions") is None
        assert plugin_source("acme--chat_completions") is None

    def test_an_undeclared_shim_in_the_tree_is_not_registered(self, path):
        folder = _extension(path, "acme")
        extra = folder / "llm" / "messages"
        extra.mkdir()
        (extra / "provider.yaml").write_text(
            "name: acme--messages\nmodel_type: llm\nformat: messages\n"
        )
        load_extensions()
        assert _status("acme").state == "loaded"
        assert get_shim("acme--messages") is None

    def test_a_core_built_in_is_never_replaced(self, path):
        builtin = next(s for s in list_shims() if "--" in s.name)
        vendor = builtin.name.partition("--")[0]
        _extension(
            path,
            vendor,
            manifest=_manifest(vendor, shims=[builtin.name]),
            leaf=f"name: {builtin.name}\nmodel_type: llm\nformat: chat_completions\n",
        )

        load_extensions()

        status = _status(vendor)
        assert status.state == "failed"
        assert "already registered" in (status.reason or "")
        assert get_shim(builtin.name) is builtin

    def test_the_second_folder_with_the_same_id_is_refused(self, tmp_path, monkeypatch):
        first, second = tmp_path / "one", tmp_path / "two"
        _extension(first, "acme")
        _extension(second, "acme")
        monkeypatch.syspath_prepend(str(second))
        monkeypatch.syspath_prepend(str(first))
        importlib.invalidate_caches()

        load_extensions()

        states = sorted(s.state for s in list_extensions() if s.id == "acme")
        assert states == ["failed", "loaded"]
        failed = next(s for s in list_extensions() if s.state == "failed")
        assert "already provided" in (failed.reason or "")


class TestDiscovery:
    def test_an_entry_point_and_the_namespace_find_one_extension_once(
        self, path, monkeypatch
    ):
        _extension(path, "acme")

        class _Entry:
            name = "acme"
            value = "llm_rosetta_ext.acme"

        monkeypatch.setattr(
            _discovery,
            "_entry_points",
            lambda group: [_Entry()] if group == _discovery.ENTRY_POINT_GROUP else [],
        )
        load_extensions()
        assert [s.id for s in list_extensions()] == ["acme"]
        assert _status("acme").source == "entry point"

    def test_an_entry_point_that_cannot_be_located_is_reported(self, monkeypatch):
        class _Entry:
            name = "ghost"
            value = "llm_rosetta_ext.ghost"

        monkeypatch.setattr(
            _discovery,
            "_entry_points",
            lambda group: [_Entry()] if group == _discovery.ENTRY_POINT_GROUP else [],
        )
        load_extensions()
        status = _status("ghost")
        assert status.state == "failed"

    def test_the_retired_group_is_reported_and_never_run(self, monkeypatch, caplog):
        class _Entry:
            name = "old_plugin"
            value = "old_package:register"

            def load(self):  # pragma: no cover - must not be called
                raise AssertionError("a retired entry point was run")

        monkeypatch.setattr(
            _discovery,
            "_entry_points",
            lambda group: (
                [_Entry()] if group == _discovery.LEGACY_ENTRY_POINT_GROUP else []
            ),
        )
        with caplog.at_level(logging.ERROR, logger="llm_rosetta.shims.providers"):
            assert load_extensions() == []
        assert any(
            "old_plugin" in r.getMessage()
            and "llm_rosetta.extensions" in r.getMessage()
            for r in caplog.records
        ), caplog.text


class TestImportExtensionModule:
    def test_it_imports_a_module_of_an_installed_extension(self, path):
        from llm_rosetta.extensions import import_extension_module

        folder = _extension(path, "acme")
        (folder / "transport.py").write_text("VALUE = 7\n")

        module = import_extension_module("acme", "transport")

        assert module.__name__ == "llm_rosetta_ext.acme.transport"
        assert module.VALUE == 7

    def test_a_module_that_does_not_exist_raises_import_error(self, path):
        from llm_rosetta.extensions import import_extension_module

        _extension(path, "acme")

        with pytest.raises(ModuleNotFoundError):
            import_extension_module("acme", "transport")

    @pytest.mark.parametrize(
        ("ext_id", "submodule"),
        [
            ("../x", ""),
            ("Acme", ""),
            ("", ""),
            ("acme", "a/b"),
            ("acme", "..x"),
            ("acme", "a..b"),
        ],
    )
    def test_a_name_that_is_not_a_plain_module_path_is_refused(self, ext_id, submodule):
        from llm_rosetta.extensions import import_extension_module

        with pytest.raises(ValueError):
            import_extension_module(ext_id, submodule)


class _Dist:
    name = "llm-rosetta-shim-argo"
    version = "0.3.1"


class _Retired:
    def __init__(self, name="argo", value="argo_shim:register", dist=None):
        self.name = name
        self.value = value
        self.dist = dist

    def load(self):  # pragma: no cover - must not be called
        raise AssertionError("a retired entry point was run")


def _only_retired(monkeypatch, entries):
    monkeypatch.setattr(
        _discovery,
        "_entry_points",
        lambda group: (
            list(entries) if group == _discovery.LEGACY_ENTRY_POINT_GROUP else []
        ),
    )


class TestRetiredEntryPoints:
    def test_it_reads_name_value_and_distribution(self, monkeypatch):
        _only_retired(monkeypatch, [_Retired(dist=_Dist())])

        [found] = _discovery.retired_entry_points()

        assert found == _discovery.RetiredEntryPoint(
            "argo", "argo_shim:register", "llm-rosetta-shim-argo 0.3.1"
        )

    def test_a_missing_distribution_is_none(self, monkeypatch):
        _only_retired(monkeypatch, [_Retired()])

        assert _discovery.retired_entry_points()[0].distribution is None

    def test_it_is_frozen(self):
        found = _discovery.RetiredEntryPoint("a", "b")
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(found, "name", "c")

    def test_describe_names_the_distribution(self):
        found = _discovery.RetiredEntryPoint("argo", "v", "pkg 1.0")
        assert found.describe() == (
            "retired entry point argo from pkg 1.0: the group "
            "llm_rosetta.shim_providers was retired in 0.14.0; its shims are not "
            "loaded; publish them as an extension (llm-rosetta ext new)"
        )

    def test_describe_without_a_distribution(self):
        found = _discovery.RetiredEntryPoint("argo", "v")
        assert "from an unknown distribution:" in found.describe()

    def test_the_loader_log_names_the_distribution(self, monkeypatch, caplog):
        _only_retired(monkeypatch, [_Retired(dist=_Dist())])
        with caplog.at_level(logging.ERROR, logger="llm_rosetta.shims.providers"):
            load_extensions()
        assert "from llm-rosetta-shim-argo 0.3.1 in the group" in caplog.text

    def test_the_loader_log_without_a_distribution(self, monkeypatch, caplog):
        _only_retired(monkeypatch, [_Retired()])
        with caplog.at_level(logging.ERROR, logger="llm_rosetta.shims.providers"):
            load_extensions()
        assert "from an unknown distribution in the group" in caplog.text
