"""``ext new`` and ``ext check``: an extension folder is created and validated."""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from llm_rosetta.authoring import ext_scaffold
from llm_rosetta.authoring.ext_scaffold import (
    ExtensionScaffoldError,
    chat_formats,
    check_extension,
    render_extension,
)
from llm_rosetta.shims.provider_shim import list_shims

_BUILDS = (
    importlib.util.find_spec("pip") is not None
    and ext_scaffold.local_backend_available()
)
needs_build = pytest.mark.skipif(
    not _BUILDS,
    reason="pip and a local wheel backend (setuptools with bdist_wheel) are required",
)
needs_toml = pytest.mark.skipif(
    ext_scaffold._toml_loader() is None, reason="no TOML parser"
)


def _new(dest: Path, ext_id: str, *flags: str) -> int:
    parser = argparse.ArgumentParser()
    ext_scaffold.add_ext_arguments(parser)
    return ext_scaffold.run_ext_new(
        parser.parse_args([ext_id, "--dest", str(dest), *flags])
    )


def _make(tmp_path: Path, ext_id: str = "acme", *flags: str) -> Path:
    assert _new(tmp_path, ext_id, *flags) == 0
    return tmp_path / ext_id


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, f"{old!r} not in {path}"
    path.write_text(text.replace(old, new), encoding="utf-8")


def _problems(folder: Path, *, build: bool = False) -> list[str]:
    return check_extension(folder, build=build).problems


def _has(problems: list[str], *needles: str) -> bool:
    return any(all(n in p for n in needles) for p in problems)


# ---------------------------------------------------------------------------
# ext new
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flags", [(), ("--transport",)])
def test_a_new_extension_passes_check_without_edits(tmp_path, flags):
    folder = _make(tmp_path, "acme_free", *flags)
    result = check_extension(folder, build=False)
    assert result.problems == []
    # The wheel check did not run, and it says so.
    assert any("wheel contents" in line for line in result.skipped)


@needs_build
@pytest.mark.parametrize("flags", [(), ("--transport",)])
def test_a_new_extension_builds_a_wheel_that_carries_its_files(tmp_path, flags):
    folder = _make(tmp_path, "acme_free", *flags)
    result = check_extension(folder, build=True)
    assert result.problems == []
    assert not any("wheel" in line for line in result.skipped)
    # A build leaves the folder as it was: it ran in a copy.
    assert not (folder / "build").exists()
    assert not list(folder.glob("*.egg-info"))


def test_the_folder_holds_the_files_the_layout_names(tmp_path):
    folder = _make(tmp_path, "acme", "--transport")
    files = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    assert files == {
        "plugin.yaml",
        "pyproject.toml",
        "README.md",
        "__init__.py",
        "provider.yaml",
        "llm/chat_completions/provider.yaml",
        "transport/__init__.py",
        "transport/transport.py",
        "tests/test_extension.py",
    }
    assert "transports: [acme]" in (folder / "plugin.yaml").read_text()
    assert (
        "transport: acme" in (folder / "llm/chat_completions/provider.yaml").read_text()
    )
    plain = _make(tmp_path / "plain", "acme")
    assert "transport" not in (plain / "plugin.yaml").read_text()
    assert not (plain / "transport").exists()


def test_todo_markers_stay_in_prose(tmp_path):
    folder = _make(tmp_path, "acme", "--transport")
    marked = sorted(
        p.relative_to(folder).as_posix()
        for p in folder.rglob("*")
        if p.is_file() and "TODO" in p.read_text(encoding="utf-8")
    )
    assert marked == ["README.md", "transport/transport.py"]


@pytest.mark.parametrize("fmt", chat_formats())
def test_every_chat_format_scaffolds_a_loadable_lane(tmp_path, fmt):
    folder = _make(tmp_path, "acme", "--format", fmt)
    assert (folder / "llm" / fmt / "provider.yaml").is_file()
    assert _problems(folder) == []


def test_the_name_reaches_the_manifest_and_the_prose(tmp_path):
    folder = _make(tmp_path, "acme", "--name", "Acme: Free Tier")
    assert 'name: "Acme: Free Tier"' in (folder / "plugin.yaml").read_text()
    assert "Acme: Free Tier" in (folder / "README.md").read_text()
    assert _problems(folder) == []


@pytest.mark.parametrize(
    "ext_id", ["", "Acme", "1acme", "acme-free", "acme free", "_acme", "acme.x"]
)
def test_a_bad_id_is_refused(tmp_path, ext_id, capsys):
    with pytest.raises(ExtensionScaffoldError, match="lowercase letters"):
        render_extension(ext_id, format_="chat_completions")
    assert _new(tmp_path, ext_id) == 2
    assert "lowercase letters" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize(
    "ext_id",
    ["scripts", "skills", "docs", "tests", "packages", "core", "contrib"],
)
def test_a_reserved_id_is_refused(tmp_path, ext_id, capsys):
    assert _new(tmp_path, ext_id) == 2
    assert "reserved" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


def test_a_format_outside_the_chat_vocabulary_is_refused(tmp_path, capsys):
    assert _new(tmp_path, "acme", "--format", "nonsense") == 2
    assert "not a chat format" in capsys.readouterr().err
    # An embeddings format is not a chat format.
    assert _new(tmp_path, "acme", "--format", "cohere") == 2
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("name", ['say "hi"', "back\\slash", "two\nlines"])
def test_a_name_that_breaks_the_files_is_refused(tmp_path, name):
    with pytest.raises(ExtensionScaffoldError, match="name may not"):
        render_extension("acme", format_="chat_completions", name=name)


def test_an_existing_folder_is_not_overwritten(tmp_path, capsys):
    folder = _make(tmp_path, "acme")
    (folder / "README.md").write_text("mine", encoding="utf-8")
    assert _new(tmp_path, "acme") == 2
    assert "already exists" in capsys.readouterr().err
    assert (folder / "README.md").read_text() == "mine"


def test_the_template_files_are_shipped_as_package_data():
    pyproject = Path(ext_scaffold.__file__).parents[3] / "pyproject.toml"
    loader = ext_scaffold._toml_loader()
    if loader is None:
        pytest.skip("no TOML parser")
    data = loader.loads(pyproject.read_text(encoding="utf-8"))
    patterns = data["tool"]["setuptools"]["package-data"]["llm_rosetta"]
    package = Path(ext_scaffold.__file__).parents[1]
    shipped = {p for pattern in patterns for p in package.glob(pattern)}
    templates = set(ext_scaffold.EXT_TEMPLATE_DIR.rglob("*.tmpl"))
    assert templates
    assert templates <= shipped


def test_the_template_has_no_orphan_file(tmp_path):
    used = {src for src, _ in ext_scaffold._BASE_TEMPLATES}
    used |= {src for src, _ in ext_scaffold._TRANSPORT_TEMPLATES}
    used |= {src for src, _ in ext_scaffold._COMMAND_TEMPLATES}
    used |= {"pyproject.toml.tmpl", "llm/format/provider.yaml.tmpl"}
    found = {
        p.relative_to(ext_scaffold.EXT_TEMPLATE_DIR).as_posix()
        for p in ext_scaffold.EXT_TEMPLATE_DIR.rglob("*.tmpl")
    }
    assert found == used


# ---------------------------------------------------------------------------
# ext check: each defect, named in a problem string
# ---------------------------------------------------------------------------


def test_the_id_in_plugin_yaml_must_be_the_folder_name(tmp_path):
    folder = _make(tmp_path, "acme")
    _edit(folder / "plugin.yaml", "id: acme\n", "id: other\n")
    _edit(folder / "plugin.yaml", "[acme--", "[other--")
    problems = _problems(folder)
    assert _has(problems, "'acme'", "'other'", "same string")


def test_a_shim_declared_but_absent_from_the_tree_is_reported(tmp_path):
    folder = _make(tmp_path, "acme")
    _edit(
        folder / "plugin.yaml",
        "shims: [acme--chat_completions]",
        "shims: [acme--chat_completions, acme--messages]",
    )
    problems = _problems(folder)
    assert problems == [
        "provides.shims lists 'acme--messages' but the tree does not register it"
    ]


def test_a_shim_in_the_tree_but_not_declared_is_reported(tmp_path):
    folder = _make(tmp_path, "acme")
    lane = folder / "llm" / "messages"
    lane.mkdir()
    (lane / "provider.yaml").write_text(
        "name: acme--messages\nmodel_type: llm\nformat: messages\n"
        "connection:\n  base_url: https://example.invalid/v1\n",
        encoding="utf-8",
    )
    problems = _problems(folder)
    assert problems == [
        "the tree registers 'acme--messages' but provides.shims does not list it"
    ]


def test_a_tree_the_loader_refuses_is_reported_not_raised(tmp_path):
    folder = _make(tmp_path, "acme")
    _edit(
        folder / "llm/chat_completions/provider.yaml",
        "format: chat_completions",
        "format: no_such",
    )
    problems = _problems(folder)
    assert _has(problems, "shim tree does not load")


@needs_toml
def test_the_version_must_match_plugin_yaml(tmp_path):
    folder = _make(tmp_path, "acme")
    _edit(folder / "plugin.yaml", "version: 0.1.0", "version: 0.2.0")
    assert _problems(folder) == [
        "pyproject.toml project.version is '0.1.0', plugin.yaml says '0.2.0'"
    ]


@needs_toml
def test_the_distribution_name_follows_the_id(tmp_path):
    folder = _make(tmp_path, "acme_free")
    _edit(
        folder / "pyproject.toml",
        "llm-rosetta-ext-acme-free",
        "llm-rosetta-ext-acme_free",
    )
    assert _has(_problems(folder), "project.name", "'llm-rosetta-ext-acme-free'")


@needs_toml
def test_package_dir_must_map_the_namespace_to_the_folder(tmp_path):
    folder = _make(tmp_path, "acme")
    _edit(
        folder / "pyproject.toml",
        '{ "llm_rosetta_ext.acme" = "." }',
        '{ "llm_rosetta_ext.acme" = "src" }',
    )
    assert _has(_problems(folder), "package-dir", "'llm_rosetta_ext.acme'")


@needs_toml
def test_packages_must_list_the_extension_package(tmp_path):
    folder = _make(tmp_path, "acme")
    _edit(
        folder / "pyproject.toml",
        'packages = ["llm_rosetta_ext.acme"]',
        "packages = []",
    )
    assert _has(_problems(folder), "tool.setuptools.packages", "'llm_rosetta_ext.acme'")


@needs_toml
def test_the_entry_point_must_name_the_package(tmp_path):
    folder = _make(tmp_path, "acme")
    _edit(folder / "pyproject.toml", 'acme = "llm_rosetta_ext.acme"', 'acme = "acme"')
    assert _has(_problems(folder), "entry point", "llm_rosetta.extensions")


def test_a_reserved_folder_name_is_reported(tmp_path):
    folder = _make(tmp_path, "acme")
    renamed = tmp_path / "docs"
    folder.rename(renamed)
    problems = _problems(renamed)
    assert _has(problems, "reserved")


def test_a_bad_folder_name_is_reported(tmp_path):
    folder = _make(tmp_path, "acme")
    renamed = tmp_path / "Acme-Free"
    folder.rename(renamed)
    problems = _problems(renamed)
    assert _has(problems, "folder 'Acme-Free'", "lowercase letters")


def test_a_missing_manifest_is_reported(tmp_path):
    folder = _make(tmp_path, "acme")
    (folder / "plugin.yaml").unlink()
    assert _problems(folder) == ["no plugin.yaml"]


def test_an_invalid_manifest_lists_its_problems(tmp_path):
    folder = _make(tmp_path, "acme")
    _edit(folder / "plugin.yaml", "maturity: experimental", "maturity: shiny")
    problems = _problems(folder)
    assert problems and all(p.startswith("plugin.yaml: ") for p in problems)
    assert _has(problems, "maturity")


def test_a_manifest_that_is_not_yaml_is_reported(tmp_path):
    folder = _make(tmp_path, "acme")
    (folder / "plugin.yaml").write_text("- just\n- a list\n", encoding="utf-8")
    assert _problems(folder)


@pytest.mark.parametrize("missing", ["README.md", "__init__.py"])
def test_a_missing_required_file_is_reported(tmp_path, missing):
    folder = _make(tmp_path, "acme")
    (folder / missing).unlink()
    assert _has(_problems(folder), f"missing {missing}")


def test_a_missing_tests_folder_is_reported(tmp_path):
    import shutil

    folder = _make(tmp_path, "acme")
    shutil.rmtree(folder / "tests")
    assert _has(_problems(folder), "missing tests/")


def test_a_missing_pyproject_is_reported(tmp_path):
    folder = _make(tmp_path, "acme")
    (folder / "pyproject.toml").unlink()
    assert _has(_problems(folder), "missing pyproject.toml")


def test_a_declared_transport_needs_a_module(tmp_path):
    import shutil

    folder = _make(tmp_path, "acme", "--transport")
    shutil.rmtree(folder / "transport")
    assert _has(_problems(folder), "provides.transports", "acme")


def test_the_scaffolded_transport_exposes_the_mapping_the_gateway_reads(tmp_path):
    folder = _make(tmp_path, "acme", "--transport")
    init = (folder / "transport" / "__init__.py").read_text()
    assert 'TRANSPORTS = {"acme": ' in init
    assert _problems(folder) == []


def test_a_transport_module_without_the_mapping_is_a_problem(tmp_path):
    folder = _make(tmp_path, "acme", "--transport")
    init = folder / "transport" / "__init__.py"
    init.write_text(init.read_text().replace("TRANSPORTS = ", "NOT_THE_MAPPING = "))
    assert _has(_problems(folder), "no TRANSPORTS mapping")


def test_a_mapping_that_disagrees_with_the_manifest_is_a_problem(tmp_path):
    folder = _make(tmp_path, "acme", "--transport")
    init = folder / "transport" / "__init__.py"
    init.write_text(init.read_text().replace('{"acme": ', '{"other": '))
    assert _has(_problems(folder), "TRANSPORTS has", "other")


def test_a_transport_module_that_does_not_parse_is_a_problem(tmp_path):
    folder = _make(tmp_path, "acme", "--transport")
    (folder / "transport" / "__init__.py").write_text("def broken(:\n")
    assert _has(_problems(folder), "does not parse")


def test_a_transport_module_file_also_counts(tmp_path):
    import shutil

    folder = _make(tmp_path, "acme", "--transport")
    shutil.rmtree(folder / "transport")
    (folder / "transport.py").write_text(
        'TRANSPORTS = {"acme": object}\n', encoding="utf-8"
    )
    assert _problems(folder) == []


@needs_build
def test_a_data_file_missing_from_package_data_is_caught_in_the_wheel(tmp_path):
    folder = _make(tmp_path, "acme")
    leaf = folder / "llm" / "chat_completions"
    (leaf / "transforms.py").write_text(
        "response_body_transforms = ()\n", encoding="utf-8"
    )
    # The shipped globs cover llm/*/*.py; a pyproject that omits it loses the file.
    assert _problems(folder, build=True) == []
    _edit(folder / "pyproject.toml", ', "llm/*/*.py"', "")
    problems = _problems(folder, build=True)
    assert problems == [
        "wheel lacks llm_rosetta_ext/acme/llm/chat_completions/transforms.py"
    ]
    # The build ran on a copy: the folder holds no build output.
    assert not (folder / "build").exists()


@needs_build
def test_a_provider_yaml_missing_from_package_data_is_caught(tmp_path):
    folder = _make(tmp_path, "acme")
    _edit(folder / "pyproject.toml", ', "llm/*/provider.yaml"', "")
    assert _problems(folder, build=True) == [
        "wheel lacks llm_rosetta_ext/acme/llm/chat_completions/provider.yaml"
    ]


@needs_build
def test_a_leftover_build_directory_does_not_change_the_wheel_check(tmp_path):
    folder = _make(tmp_path, "acme")
    stale = folder / "build" / "lib" / "llm_rosetta_ext" / "acme"
    stale.mkdir(parents=True)
    (stale / "ghost.py").write_text("", encoding="utf-8")
    (folder / "acme.egg-info").mkdir()
    assert _problems(folder, build=True) == []


@needs_build
def test_a_file_the_wheel_ships_but_the_folder_lacks_is_reported(tmp_path):
    folder = _make(tmp_path, "acme")
    _edit(
        folder / "pyproject.toml",
        'packages = ["llm_rosetta_ext.acme"]',
        'packages = ["llm_rosetta_ext.acme", "llm_rosetta_ext.acme.tests"]',
    )
    (folder / "tests" / "__init__.py").write_text("", encoding="utf-8")
    problems = _problems(folder, build=True)
    assert _has(problems, "wheel ships llm_rosetta_ext/acme/tests/")


@needs_build
def test_a_pyproject_the_build_refuses_is_reported(tmp_path):
    folder = _make(tmp_path, "acme")
    _edit(
        folder / "pyproject.toml",
        'build-backend = "setuptools.build_meta"',
        'build-backend = "no.such.backend"',
    )
    assert _has(_problems(folder, build=True), "wheel build failed")


def test_without_pip_the_wheel_check_is_skipped_not_passed(tmp_path, monkeypatch):
    folder = _make(tmp_path, "acme")
    monkeypatch.setattr(ext_scaffold, "pip_available", lambda: False)
    result = check_extension(folder, build=True)
    assert result.problems == []
    assert result.skipped == ["wheel contents: pip is not available"]


def test_without_a_toml_parser_the_pyproject_check_is_skipped(tmp_path, monkeypatch):
    folder = _make(tmp_path, "acme")
    _edit(folder / "plugin.yaml", "version: 0.1.0", "version: 9.9.9")
    monkeypatch.setattr(ext_scaffold, "_toml_loader", lambda: None)
    result = check_extension(folder, build=False)
    assert result.problems == []
    assert any("pyproject.toml content" in line for line in result.skipped)


def test_a_pyproject_that_does_not_parse_is_reported(tmp_path):
    if ext_scaffold._toml_loader() is None:
        pytest.skip("no TOML parser")
    folder = _make(tmp_path, "acme")
    (folder / "pyproject.toml").write_text("[project\n", encoding="utf-8")
    assert _has(_problems(folder), "pyproject.toml does not parse")


def test_checking_leaves_the_live_registry_alone(tmp_path):
    before = sorted(shim.name for shim in list_shims())
    folder = _make(tmp_path, "acme")
    assert _problems(folder) == []
    assert sorted(shim.name for shim in list_shims()) == before
    assert "acme--chat_completions" not in before
    # ...and a tree that fails to load leaves it alone as well.
    _edit(
        folder / "llm/chat_completions/provider.yaml",
        "format: chat_completions",
        "format: no_such",
    )
    assert _problems(folder)
    assert sorted(shim.name for shim in list_shims()) == before


# ---------------------------------------------------------------------------
# run_ext_check: the exit code and the report
# ---------------------------------------------------------------------------


def test_run_ext_check_returns_zero_for_a_sound_folder(tmp_path, capsys):
    folder = _make(tmp_path, "acme")
    assert ext_scaffold.run_ext_check(folder, build=False) == 0
    out = capsys.readouterr().out
    assert "ok      acme" in out
    assert "skipped acme: wheel contents" in out


def test_run_ext_check_returns_one_and_prints_the_problem(tmp_path, capsys):
    folder = _make(tmp_path, "acme")
    (folder / "README.md").unlink()
    assert ext_scaffold.run_ext_check(folder, build=False) == 1
    assert "problem acme: missing README.md" in capsys.readouterr().out


def test_run_ext_check_walks_the_packages_of_a_root(tmp_path, capsys):
    packages = tmp_path / "packages"
    _make(packages, "good")
    bad = _make(packages, "bad")
    (bad / "tests" / "test_extension.py").unlink()
    (bad / "tests").rmdir()
    assert ext_scaffold.run_ext_check(tmp_path, build=False) == 1
    out = capsys.readouterr().out
    assert "ok      good" in out
    assert "problem bad: missing tests/" in out


def test_a_folder_in_packages_without_a_manifest_fails_the_root(tmp_path, capsys):
    packages = tmp_path / "packages"
    _make(packages, "good")
    (packages / "stray").mkdir()
    assert ext_scaffold.run_ext_check(tmp_path, build=False) == 1
    assert "problem stray: no plugin.yaml" in capsys.readouterr().out


def test_run_ext_check_takes_the_parsed_flags(tmp_path, capsys):
    folder = _make(tmp_path, "acme")
    parser = argparse.ArgumentParser()
    ext_scaffold.add_ext_arguments(parser, command="check")
    args = parser.parse_args([str(folder), "--no-build"])
    assert args.build is False
    assert ext_scaffold.run_ext_check(args) == 0
    assert "wheel contents: build not requested" in capsys.readouterr().out
    assert parser.parse_args([str(folder)]).build is True


def test_add_ext_arguments_refuses_an_unknown_command():
    with pytest.raises(ValueError, match="unknown ext command"):
        ext_scaffold.add_ext_arguments(argparse.ArgumentParser(), command="nope")


def test_the_new_flags_default_to_chat_completions_in_packages():
    parser = argparse.ArgumentParser()
    ext_scaffold.add_ext_arguments(parser)
    args = parser.parse_args(["acme"])
    assert (args.format, args.dest, args.transport) == (
        "chat_completions",
        "packages",
        False,
    )


def test_the_generated_test_runs_the_check(tmp_path):
    folder = _make(tmp_path, "acme")
    text = (folder / "tests" / "test_extension.py").read_text()
    assert re.search(r"run_ext_check\(.*build=False\) == 0", text)


@pytest.mark.parametrize(
    "flags",
    [[], ["--transport", "--command", "hello"]],
    ids=["shim-only", "shim-transport-command"],
)
def test_the_generated_extension_passes_its_own_tests(tmp_path, flags):
    """A new extension is not born with a red test (``run_ext_check`` once left the api).

    The generated ``tests/test_extension.py`` runs in a child interpreter that
    imports the core under test, as an author's checkout would.
    """
    import llm_rosetta

    folder = _make(tmp_path, "acme", *flags)
    source = Path(llm_rosetta.__file__).resolve().parent.parent
    env = {**os.environ, "PYTHONPATH": str(source)}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
        cwd=folder,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-800:] + result.stderr[-400:]
