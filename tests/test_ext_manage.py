"""Removing an extension with the programs it installed (#1829)."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_rosetta.extensions import (
    ExtensionInstallError,
    extension_origin,
    plan_removal,
    remove_with_programs,
)

MANIFEST = (
    "id: {id}\nname: {id} test\nversion: {version}\nmaturity: experimental\n"
    "requires:\n  surface: 1\n  ir: 1\nprovides:\n  shims: [{id}--chat_completions]\n"
)


@pytest.fixture
def layout(tmp_path):
    """An extensions directory and a tools directory side by side."""
    dest = tmp_path / "config" / "extensions"
    tools = tmp_path / "config" / "tools"
    (dest / "llm_rosetta_ext").mkdir(parents=True)
    tools.mkdir(parents=True)
    return dest, tools


def _extension(dest: Path, ext_id: str, version: str = "0.1.0") -> Path:
    folder = dest / "llm_rosetta_ext" / ext_id
    folder.mkdir()
    (folder / "plugin.yaml").write_text(MANIFEST.format(id=ext_id, version=version))
    info = dest / f"{ext_id}-{version}.dist-info"
    info.mkdir()
    (info / "LLM_ROSETTA_EXTENSION").write_text(ext_id)
    return folder


def _program(tools: Path, ext_id: str, *, npm: bool = False) -> Path:
    """What ``install_cli`` leaves: the program, its entry link, a shared link."""
    root = tools / ext_id
    kind = "npm-1.0.0" if npm else "release-1.0.0"
    (root / kind / "bin").mkdir(parents=True)
    binary = root / kind / "bin" / ext_id
    binary.write_text("x" * 100)
    (root / "bin").mkdir()
    (root / "bin" / ext_id).symlink_to(binary)
    (tools / "bin").mkdir(exist_ok=True)
    (tools / "bin" / ext_id).symlink_to(root / "bin" / ext_id)
    return root


def _node(tools: Path) -> Path:
    node = tools / "node" / "24.22.3"
    (node / "bin").mkdir(parents=True)
    (node / "bin" / "node").write_text("n" * 10)
    (tools / "bin").mkdir(exist_ok=True)
    for name in ("node", "npm", "npx"):
        (tools / "bin" / name).symlink_to(node / "bin" / "node")
    return tools / "node"


class TestOrigin:
    def test_a_folder_in_the_extensions_directory_is_a_directory_install(self, layout):
        dest, _ = layout
        assert extension_origin(_extension(dest, "acme"), dest) == "directory"

    def test_a_folder_elsewhere_is_the_images(self, layout, tmp_path):
        dest, _ = layout
        site = tmp_path / "site-packages" / "llm_rosetta_ext" / "acme"
        site.mkdir(parents=True)
        assert extension_origin(site, dest) == "image"

    def test_a_link_out_of_the_directory_is_the_images(self, layout, tmp_path):
        dest, _ = layout
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        link = dest / "llm_rosetta_ext" / "acme"
        link.symlink_to(outside)
        assert extension_origin(link, dest) == "image"


class TestPlan:
    def test_it_lists_the_extension_its_programs_and_its_links(self, layout):
        dest, tools = layout
        folder = _extension(dest, "acme")
        _program(tools, "acme")
        plan = plan_removal("acme", dest=dest, tools=tools)
        paths = [e.path for e in plan.delete]
        assert plan.removable and plan.origin == "directory" and plan.version == "0.1.0"
        assert paths[0] == str(folder)
        assert str(dest / "acme-0.1.0.dist-info") in paths
        assert str(tools / "bin" / "acme") in paths
        assert str(tools / "acme") in paths
        assert next(e for e in plan.delete if e.path == str(tools / "acme")).size == 100

    def test_nothing_changes_when_only_planned(self, layout):
        dest, tools = layout
        _extension(dest, "acme")
        _program(tools, "acme")
        plan_removal("acme", dest=dest, tools=tools)
        assert (dest / "llm_rosetta_ext" / "acme").is_dir()
        assert (tools / "acme").is_dir()

    def test_the_last_user_of_node_takes_it_along(self, layout):
        dest, tools = layout
        _extension(dest, "acme")
        _program(tools, "acme", npm=True)
        _node(tools)
        plan = plan_removal("acme", dest=dest, tools=tools)
        paths = [e.path for e in plan.delete]
        assert str(tools / "node") in paths
        assert {str(tools / "bin" / n) for n in ("node", "npm", "npx")} <= set(paths)
        assert plan.keep == ()

    def test_node_stays_while_another_npm_program_needs_it(self, layout):
        dest, tools = layout
        _extension(dest, "acme")
        _extension(dest, "other")
        _program(tools, "acme", npm=True)
        _program(tools, "other", npm=True)
        _node(tools)
        plan = plan_removal("acme", dest=dest, tools=tools)
        paths = [e.path for e in plan.delete]
        assert str(tools / "node") not in paths
        assert str(tools / "bin" / "node") not in paths
        assert [(e.path, e.note) for e in plan.keep] == [
            (str(tools / "node"), "still used by: other")
        ]

    def test_a_release_program_elsewhere_does_not_hold_node(self, layout):
        dest, tools = layout
        _extension(dest, "acme")
        _program(tools, "acme", npm=True)
        _program(tools, "other", npm=False)
        _node(tools)
        assert str(tools / "node") in [
            e.path for e in plan_removal("acme", dest=dest, tools=tools).delete
        ]

    def test_an_extension_without_programs_leaves_node_alone(self, layout):
        dest, tools = layout
        _extension(dest, "acme")
        _program(tools, "other", npm=True)
        _node(tools)
        plan = plan_removal("acme", dest=dest, tools=tools)
        assert str(tools / "node") not in [e.path for e in plan.delete]

    def test_an_image_extension_is_not_removable_and_names_the_command(
        self, layout, tmp_path
    ):
        dest, tools = layout
        site = tmp_path / "site" / "acme"
        site.mkdir(parents=True)
        plan = plan_removal("acme", path=site, dest=dest, tools=tools)
        assert not plan.removable and plan.origin == "image" and plan.delete == ()
        assert "rebuild the image" in (plan.hint or "")

    def test_an_unknown_extension_is_missing(self, layout):
        dest, tools = layout
        plan = plan_removal("ghost", dest=dest, tools=tools)
        assert plan.origin == "missing" and not plan.removable

    @pytest.mark.parametrize("bad", ["../x", "a/b", "", "_x"])
    def test_a_path_shaped_id_is_refused(self, layout, bad):
        dest, tools = layout
        with pytest.raises(ExtensionInstallError, match="not an extension id"):
            plan_removal(bad, dest=dest, tools=tools)


class TestRemove:
    def test_it_deletes_what_the_plan_names_and_nothing_else(self, layout):
        dest, tools = layout
        _extension(dest, "acme")
        _extension(dest, "other")
        _program(tools, "acme", npm=True)
        _program(tools, "other", npm=True)
        _node(tools)
        plan = plan_removal("acme", dest=dest, tools=tools)
        result = remove_with_programs("acme", dest=dest, tools=tools)
        assert set(result.deleted) == {e.path for e in plan.delete}
        assert result.failed == ()
        assert not (dest / "llm_rosetta_ext" / "acme").exists()
        assert not (tools / "acme").exists()
        assert not (tools / "bin" / "acme").is_symlink()
        assert (dest / "llm_rosetta_ext" / "other").is_dir()
        assert (tools / "other" / "npm-1.0.0").is_dir()
        assert (tools / "node" / "24.22.3" / "bin" / "node").is_file()
        assert (tools / "bin" / "node").is_symlink()

    def test_the_last_npm_extension_takes_node_with_it(self, layout):
        dest, tools = layout
        _extension(dest, "acme")
        _program(tools, "acme", npm=True)
        _node(tools)
        remove_with_programs("acme", dest=dest, tools=tools)
        assert not (tools / "node").exists()
        assert not any((tools / "bin" / n).is_symlink() for n in ("node", "npm", "npx"))

    def test_an_image_extension_is_refused_and_untouched(self, layout, tmp_path):
        dest, tools = layout
        site = tmp_path / "site" / "acme"
        site.mkdir(parents=True)
        (site / "plugin.yaml").write_text("x")
        _program(tools, "acme")
        with pytest.raises(ExtensionInstallError, match="rebuild the image"):
            remove_with_programs("acme", path=site, dest=dest, tools=tools)
        assert site.is_dir() and (tools / "acme").is_dir()

    def test_a_missing_extension_is_refused(self, layout):
        dest, tools = layout
        _program(tools, "ghost")
        with pytest.raises(ExtensionInstallError):
            remove_with_programs("ghost", dest=dest, tools=tools)
        assert (tools / "ghost").is_dir()

    def test_a_program_that_cannot_be_deleted_is_reported_not_hidden(
        self, layout, monkeypatch
    ):
        import llm_rosetta.extensions._manage as manage

        dest, tools = layout
        _extension(dest, "acme")
        _program(tools, "acme")
        real = manage._delete

        def refuse(path):
            if path == tools / "acme":
                return "PermissionError: denied"
            return real(path)

        monkeypatch.setattr(manage, "_delete", refuse)
        result = remove_with_programs("acme", dest=dest, tools=tools)
        assert result.failed == ((str(tools / "acme"), "PermissionError: denied"),)
        assert str(tools / "acme") not in result.deleted
        assert not (dest / "llm_rosetta_ext" / "acme").exists()
