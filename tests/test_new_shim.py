"""Tests for ``scripts/dev/new_shim.py`` — the shim scaffolder.

The script's value is that its output loads: every case here finishes by asking
the real loader whether the files it wrote make a shim, rather than trusting the
template.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from llm_rosetta.shims.provider_shim import _reset_registry
from llm_rosetta.shims.providers import load_providers, load_providers_from_dir

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/dev/new_shim.py"


def _script():
    spec = importlib.util.spec_from_file_location("new_shim", SCRIPT)
    assert spec is not None and spec.loader is not None, SCRIPT
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _clean_registry():
    _reset_registry()
    yield
    _reset_registry()
    load_providers()


@pytest.fixture
def new_shim():
    return _script()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    providers = tmp_path / "providers"
    providers.mkdir()
    return providers


def _fake_repository(monkeypatch, repo: Path) -> None:
    """Make *repo* answer like a GitHub shim package."""
    from llm_rosetta.shims import scaffold

    def fake_git(*args: str, cwd: Path) -> str | None:
        if args[0] == "rev-parse":
            return str(repo)
        if args[0] == "remote":
            return "git@github.com:acme/shim-acme.git"
        if args[0] == "symbolic-ref":
            return "origin/main"
        return None

    monkeypatch.setattr(scaffold, "_git", fake_git)


def test_the_typed_layout_creates_a_vendor_manifest_and_a_leaf(new_shim, root):
    code = new_shim.main(
        [
            "acme",
            "--root",
            str(root),
            "--base-url",
            "https://api.acme.example/v1",
            "--logo",
            "https://example.com/acme.svg",
            "--group-label",
            "ACME",
        ]
    )

    leaf = root / "acme" / "llm" / "chat_completions" / "provider.yaml"
    assert code == 0
    assert leaf.is_file()
    text = leaf.read_text(encoding="utf-8")
    assert "name: acme--chat_completions" in text
    assert "model_type: llm" in text
    assert "format: chat_completions" in text
    assert "base_url: https://api.acme.example/v1" in text
    # The credential and the vendor identity live one level up.
    manifest = (root / "acme" / "provider.yaml").read_text(encoding="utf-8")
    assert "api_key_env: ACME_API_KEY" in manifest
    assert "logo: https://example.com/acme.svg" in manifest
    assert "group: acme" in manifest
    assert "group_label: ACME" in manifest
    assert "api_key_env" not in text


def test_the_scaffold_loads_with_what_it_inherits(new_shim, root):
    new_shim.main(
        [
            "acme",
            "--root",
            str(root),
            "--base-url",
            "https://api.acme.example/v1",
            "--logo",
            "https://example.com/acme.svg",
            "--group-label",
            "ACME",
        ]
    )

    shims = load_providers_from_dir(root)

    assert [s.name for s in shims] == ["acme--chat_completions"]
    shim = shims[0]
    assert shim.connection.base_url == "https://api.acme.example/v1"
    assert shim.connection.api_key_env == "ACME_API_KEY"
    assert (shim.group, shim.group_label) == ("acme", "ACME")
    assert shim.logo == "https://example.com/acme.svg"


def test_the_flat_layout_puts_the_format_under_the_vendor(new_shim, root):
    new_shim.main(["acme", "--root", str(root), "--layout", "flat"])

    assert (root / "acme" / "chat_completions" / "provider.yaml").is_file()
    assert [s.name for s in load_providers_from_dir(root)] == ["acme--chat_completions"]


def test_a_non_chat_type_names_its_own_directory(new_shim, root):
    new_shim.main(
        [
            "acme",
            "--root",
            str(root),
            "--model-type",
            "embeddings",
            "--format",
            "openai",
        ]
    )

    leaf = root / "acme" / "embeddings" / "provider.yaml"
    assert leaf.is_file()
    names = [s.name for s in load_providers_from_dir(root)]
    assert names == ["acme--openai"]


def test_an_existing_leaf_is_refused(new_shim, root):
    new_shim.main(["acme", "--root", str(root)])

    code = new_shim.main(["acme", "--root", str(root)])

    assert code == 1


def test_the_package_manifest_gains_the_new_entry(new_shim, root):
    new_shim.main(["acme", "--root", str(root)])
    (root / "shims.yaml").write_text(
        "version: 1\nshims:\n  - name: acme--chat_completions\n"
        "    path: acme/llm/chat_completions\n",
        encoding="utf-8",
    )

    new_shim.main(["acme", "--root", str(root), "--format", "messages"])

    text = (root / "shims.yaml").read_text(encoding="utf-8")
    assert text.count("- name:") == 2
    assert "path: acme/llm/messages" in text
    assert [s.name for s in load_providers_from_dir(root)] == [
        "acme--chat_completions",
        "acme--messages",
    ]


def test_the_package_manifest_can_be_left_alone(new_shim, root):
    original = "version: 1\nshims: []\n"
    (root / "shims.yaml").write_text(original, encoding="utf-8")

    new_shim.main(["acme", "--root", str(root), "--no-registry-manifest"])

    assert (root / "shims.yaml").read_text(encoding="utf-8") == original


def test_a_manifest_that_does_not_end_with_the_list_is_left_to_the_operator(
    new_shim, root, capsys
):
    """Appending there would move the key; the script says so instead."""
    (root / "shims.yaml").write_text(
        "version: 1\nshims:\n  - name: acme--chat_completions\n"
        "    path: acme/llm/chat_completions\nstrict: true\n",
        encoding="utf-8",
    )

    new_shim.main(["acme", "--root", str(root), "--format", "messages"])

    assert "add this entry by hand" in capsys.readouterr().out
    assert "- name: acme--messages" not in (root / "shims.yaml").read_text(
        encoding="utf-8"
    )


def test_the_transform_stub_is_picked_up_by_the_loader(new_shim, root):
    new_shim.main(["acme", "--root", str(root), "--transform"])

    shims = load_providers_from_dir(root)

    # A tuple of callables, not a function that returns one: the pipeline
    # iterates this attribute, so a callable export would load and then fail
    # on the first request.
    assert isinstance(shims[0].response_body_transforms, tuple)
    (transform,) = shims[0].response_body_transforms
    body = {"choices": []}
    assert transform(body) == body


def test_check_reports_what_the_leaf_loads_as(new_shim, root, capsys):
    code = new_shim.main(["acme", "--root", str(root), "--check"])

    assert code == 0
    assert "check   acme--chat_completions loads" in capsys.readouterr().out


def test_a_vendor_with_a_manifest_keeps_state_its_own_identity(new_shim, root):
    """A second lane does not restate what the vendor already declares."""
    new_shim.main(
        ["acme", "--root", str(root), "--logo", "https://example.com/acme.svg"]
    )

    new_shim.main(["acme", "--root", str(root), "--format", "messages"])

    text = (root / "acme" / "llm" / "messages" / "provider.yaml").read_text(
        encoding="utf-8"
    )
    assert "logo:" not in text
    assert "group:" not in text
    manifest = (root / "acme" / "provider.yaml").read_text(encoding="utf-8")
    assert manifest.count("logo:") == 1


class TestInteractiveMode:
    """Interactive is the default: every field the flags left out is asked for."""

    @staticmethod
    def _answers(monkeypatch, answers: list[str]):
        from llm_rosetta.shims import scaffold

        monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
        # The base URL probe and the icon lookup are real HTTP requests; no
        # test may make one.
        monkeypatch.setattr(
            scaffold,
            "probe_base_url",
            lambda url: (True, f"{url}/models answers 200"),
        )
        monkeypatch.setattr(scaffold, "find_lobehub_logo", lambda vendor: None)
        queue = iter(answers)
        monkeypatch.setattr("builtins.input", lambda prompt="": next(queue))

    def test_it_asks_for_every_field_and_checks_the_result(
        self, monkeypatch, root, capsys
    ):
        from llm_rosetta.shims import scaffold

        self._answers(
            monkeypatch,
            [
                str(root),  # scan root, asked first so the shape is known
                "acme",  # vendor
                "llm",  # model type
                "chat_completions",  # format
                "typed",  # layout
                "https://api.acme.example/v1",  # base URL
                "ACME_API_KEY",  # credential env var
                "",  # models path
                "acme",  # display name
                "https://example.com/acme.svg",  # logo
                "n",  # do not save the mark / do not reference the CDN
                "ACME",  # group title
                "n",  # transforms stub
            ],
        )

        code = scaffold.main([])

        out = capsys.readouterr().out
        assert code == 0
        assert "Creating a shim" in out
        assert "check   acme--chat_completions loads" in out
        leaf = root / "acme" / "llm" / "chat_completions" / "provider.yaml"
        assert leaf.is_file()
        assert "display_name: acme" in leaf.read_text(encoding="utf-8")

    def test_answers_given_as_flags_are_not_asked_for_again(self, monkeypatch, root):
        from llm_rosetta.shims import scaffold

        self._answers(
            monkeypatch,
            [
                str(root),  # scan root
                "llm",  # model type
                "chat_completions",  # format
                "typed",  # layout
                "https://api.acme.example/v1",
                "ACME_API_KEY",
                "",
                "",
                "",
                "",
                "n",
            ],
        )

        code = scaffold.main(["acme"])

        assert code == 0
        assert (root / "acme" / "provider.yaml").is_file()

    def test_no_input_requires_a_vendor(self, root, capsys):
        from llm_rosetta.shims import scaffold

        code = scaffold.main(["--no-input", "--root", str(root)])

        assert code == 2
        assert "vendor name is required" in capsys.readouterr().err

    def test_a_logo_url_offers_to_save_it_into_the_package(
        self, monkeypatch, root, capsys
    ):
        from llm_rosetta.shims import scaffold

        _fake_repository(monkeypatch, root.parent)
        monkeypatch.setattr(scaffold, "_download_logo", lambda url: (b"<svg/>", ".svg"))
        self._answers(
            monkeypatch,
            [
                str(root),  # scan root
                "acme",
                "llm",
                "chat_completions",
                "typed",
                "https://api.acme.example/v1",
                "ACME_API_KEY",
                "",
                "",
                "https://example.com/acme.svg",
                "y",  # save the mark and use its CDN URL
                "ACME",
                "n",
            ],
        )

        code = scaffold.main([])

        assert code == 0
        assert (root.parent / "assets" / "acme.svg").read_bytes() == b"<svg/>"
        text = (root / "acme" / "provider.yaml").read_text(encoding="utf-8")
        assert (
            "logo: https://cdn.jsdelivr.net/gh/acme/shim-acme@main/assets/acme.svg"
            in (text)
        )
        assert "saved   " in capsys.readouterr().out


class TestLogoLocalization:
    """A mark a package ships lives in the repository, not on someone's host."""

    def test_it_saves_the_mark_and_returns_the_cdn_url(self, monkeypatch, root):
        from llm_rosetta.shims import scaffold

        _fake_repository(monkeypatch, root.parent)
        monkeypatch.setattr(scaffold, "_download_logo", lambda url: (b"<svg/>", ".svg"))

        url, note = scaffold.localize_logo("https://example.com/acme.svg", root, "acme")

        assert (root.parent / "assets" / "acme.svg").read_bytes() == b"<svg/>"
        assert url == "https://cdn.jsdelivr.net/gh/acme/shim-acme@main/assets/acme.svg"
        assert note.startswith("saved   ")

    def test_an_unchanged_mark_is_not_rewritten(self, monkeypatch, root):
        from llm_rosetta.shims import scaffold

        _fake_repository(monkeypatch, root.parent)
        assets = root.parent / "assets"
        assets.mkdir()
        (assets / "acme.svg").write_bytes(b"<svg/>")
        monkeypatch.setattr(scaffold, "_download_logo", lambda url: (b"<svg/>", ".svg"))

        _url, note = scaffold.localize_logo(
            "https://example.com/acme.svg", root, "acme"
        )

        assert note.startswith("kept existing")

    def test_outside_a_repository_the_url_is_kept(self, monkeypatch, root):
        from llm_rosetta.shims import scaffold

        monkeypatch.setattr(scaffold, "_git", lambda *args, cwd: None)

        url, note = scaffold.localize_logo("https://example.com/acme.svg", root, "acme")

        assert url == "https://example.com/acme.svg"
        assert "not inside a git repository" in note

    def test_a_download_failure_keeps_the_url(self, monkeypatch, root):
        from llm_rosetta.shims import scaffold

        _fake_repository(monkeypatch, root.parent)

        def boom(_url):
            raise ValueError("the mark URL must use https")

        monkeypatch.setattr(scaffold, "_download_logo", boom)

        url, note = scaffold.localize_logo("http://example.com/acme.svg", root, "acme")

        assert url == "http://example.com/acme.svg"
        assert "must use https" in note

    def test_the_flag_mode_localizes_without_prompting(self, monkeypatch, root):
        from llm_rosetta.shims import scaffold

        _fake_repository(monkeypatch, root.parent)
        monkeypatch.setattr(scaffold, "_download_logo", lambda url: (b"<svg/>", ".svg"))

        code = scaffold.main(
            [
                "acme",
                "--no-input",
                "--root",
                str(root),
                "--logo",
                "https://example.com/acme.svg",
                "--localize-logo",
            ]
        )

        assert code == 0
        assert (root.parent / "assets" / "acme.svg").is_file()


class TestGatewayCommand:
    """``llm-rosetta-gateway shims new`` is the installed entry point."""

    @staticmethod
    def _argv(root: Path, *extra: str) -> list[str]:
        return ["llm-rosetta-gateway", "shims", "new", *extra, "--root", str(root)]

    def test_the_subcommand_scaffolds(self, monkeypatch, root, tmp_path):
        from llm_rosetta_gateway import cli

        monkeypatch.setattr(
            "sys.argv",
            self._argv(
                root,
                "acme",
                "--no-input",
                "--base-url",
                "https://api.acme.example/v1",
            ),
        )
        config = tmp_path / "config.jsonc"
        config.write_text("{}\n", encoding="utf-8")
        monkeypatch.setattr(cli, "discover_config", lambda _path: str(config))

        with pytest.raises(SystemExit) as exit_info:
            cli.main()

        assert exit_info.value.code == 0
        assert (root / "acme" / "llm" / "chat_completions" / "provider.yaml").is_file()

    def test_a_failed_load_check_reaches_the_shell(self, monkeypatch, root, tmp_path):
        from llm_rosetta.shims import providers as providers_module
        from llm_rosetta_gateway import cli

        def refuse(*_args, **_kwargs):
            raise ValueError("'chat_completions' is not an embeddings format")

        monkeypatch.setattr(providers_module, "load_providers_from_dir", refuse)
        monkeypatch.setattr(
            "sys.argv", self._argv(root, "acme", "--no-input", "--check")
        )
        config = tmp_path / "config.jsonc"
        config.write_text("{}\n", encoding="utf-8")
        monkeypatch.setattr(cli, "discover_config", lambda _path: str(config))

        with pytest.raises(SystemExit) as exit_info:
            cli.main()

        assert exit_info.value.code == 1


class TestFormatChoice:
    """The format is a closed list, taken from the converter registry."""

    def test_the_prompt_offers_every_registered_chat_format(
        self, monkeypatch, root, capsys
    ):
        TestInteractiveMode._answers(
            monkeypatch,
            [
                str(root),  # scan root
                "acme",
                "llm",
                "rpc",  # not a format
                "chat_completions",
                "typed",
                "",
                "",
                "",
                "",
                "",
                "",
                "n",
            ],
        )

        code = _script().main([])

        out = capsys.readouterr().out
        assert code == 0
        assert "'rpc' is not one of" in out
        for fmt in ("chat_completions", "open_responses", "google_interactions"):
            assert fmt in out

    def test_an_unknown_flag_format_is_refused_before_anything_is_written(
        self, root, capsys
    ):
        code = _script().main(
            ["acme", "--no-input", "--root", str(root), "--format", "rpc"]
        )

        assert code == 2
        err = capsys.readouterr().err
        assert "unknown format 'rpc' for llm" in err
        assert "available: " in err
        assert not (root / "acme").exists()

    def test_the_format_help_lists_the_choices(self, capsys):
        with pytest.raises(SystemExit) as exit_info:
            _script().main(["--help"])

        assert exit_info.value.code == 0
        out = capsys.readouterr().out
        assert "llm: chat_completions" in out
        assert "embeddings: openai" in out


class TestBaseUrlProbe:
    """A pasted base URL is normalised, then asked whether it answers."""

    @staticmethod
    def _status(monkeypatch, statuses: dict[str, tuple[int | None, str]]):
        from llm_rosetta.shims import scaffold

        monkeypatch.setattr(
            scaffold, "_http_status", lambda url, **_kw: statuses.get(url, (404, ""))
        )

    def test_a_models_200_passes(self, monkeypatch):
        from llm_rosetta.shims import scaffold

        self._status(monkeypatch, {"https://api.acme.example/v1/models": (200, "")})

        ok, detail = scaffold.probe_base_url("https://api.acme.example/v1")

        assert ok is True
        assert "answers 200" in detail

    def test_an_authentication_refusal_proves_reachability(self, monkeypatch):
        from llm_rosetta.shims import scaffold

        self._status(monkeypatch, {"https://api.acme.example/v1/models": (401, "")})

        ok, detail = scaffold.probe_base_url("https://api.acme.example/v1")

        assert ok is True
        assert "without a key" in detail

    def test_a_404_asks_for_the_suffix(self, monkeypatch):
        from llm_rosetta.shims import scaffold

        self._status(
            monkeypatch,
            {
                "https://api.acme.example/v1/models": (404, ""),
                "https://api.acme.example/v1": (200, ""),
            },
        )

        ok, detail = scaffold.probe_base_url("https://api.acme.example/v1")

        assert ok is False
        assert "right suffix" in detail

    def test_an_unreachable_host_says_why(self, monkeypatch):
        from llm_rosetta.shims import scaffold

        self._status(
            monkeypatch, {"https://api.acme.example/v1/models": (None, "timed out")}
        )

        ok, detail = scaffold.probe_base_url("https://api.acme.example/v1")

        assert ok is False
        assert "cannot reach" in detail and "timed out" in detail

    def test_a_pasted_models_suffix_is_dropped(self):
        from llm_rosetta.shims import scaffold

        url, note = scaffold.normalize_base_url(" https://api.acme.example/v1/models/ ")

        assert url == "https://api.acme.example/v1"
        assert "/models suffix" in note

    def test_a_clean_url_is_left_alone(self):
        from llm_rosetta.shims import scaffold

        url, note = scaffold.normalize_base_url("https://api.acme.example/v1")

        assert (url, note) == ("https://api.acme.example/v1", "")

    def test_the_probe_flag_reports_the_result(self, monkeypatch, root, capsys):

        self._status(monkeypatch, {"https://api.acme.example/v1/models": (200, "")})

        code = _script().main(
            [
                "acme",
                "--no-input",
                "--root",
                str(root),
                "--base-url",
                "https://api.acme.example/v1/models",
                "--probe",
            ]
        )

        out = capsys.readouterr().out
        assert code == 0
        assert "dropped the pasted /models suffix" in out
        assert "ok:" in out
        leaf = root / "acme" / "llm" / "chat_completions" / "provider.yaml"
        assert "base_url: https://api.acme.example/v1" in leaf.read_text(
            encoding="utf-8"
        )


class TestCtrlC:
    """One Ctrl+C re-asks; a second one ends the run without a traceback."""

    @staticmethod
    def _interrupting(monkeypatch, answers: list[str], at: set[int]):
        from llm_rosetta.shims import scaffold

        monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
        monkeypatch.setattr(
            scaffold, "probe_base_url", lambda url: (True, f"{url}/models answers 200")
        )
        monkeypatch.setattr(scaffold, "find_lobehub_logo", lambda vendor: None)
        queue = iter(answers)
        calls = {"n": 0}

        def fake_input(prompt: str = "") -> str:
            calls["n"] += 1
            if calls["n"] in at:
                assert scaffold._guard is not None
                scaffold._guard.presses += 1
                raise KeyboardInterrupt
            return next(queue)

        monkeypatch.setattr("builtins.input", fake_input)

    def test_the_first_ctrl_c_re_asks_the_same_question(
        self, monkeypatch, root, capsys
    ):
        from llm_rosetta.shims import scaffold

        self._interrupting(
            monkeypatch,
            [
                str(root),  # scan root
                "acme",
                "llm",
                "chat_completions",
                "typed",
                "",
                "",
                "",
                "",
                "",
                "",
                "n",
            ],
            at={1},
        )

        code = scaffold.main([])

        out = capsys.readouterr().out
        assert code == 0
        assert "Ctrl+C again to quit" in out
        assert (root / "acme" / "llm" / "chat_completions" / "provider.yaml").is_file()

    def test_a_second_ctrl_c_cancels_without_writing(self, monkeypatch, root, capsys):
        from llm_rosetta.shims import scaffold

        self._interrupting(monkeypatch, [], at={1, 2})

        code = scaffold.main([])

        err = capsys.readouterr().err
        assert code == 130
        assert "cancelled" in err
        assert "no files were written" in err
        assert not (root / "acme").exists()


def _leaf_file(
    root: Path,
    relative: str,
    *,
    name: str,
    format_: str = "chat_completions",
) -> Path:
    """Write a loadable leaf at *relative* (a directory below *root*)."""
    path = root / relative / "provider.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"name: {name}\nmodel_type: llm\nformat: {format_}\n"
        "connection:\n  base_url: https://api.example/v1\n  api_key_env: X_API_KEY\n",
        encoding="utf-8",
    )
    return path


class TestExistingTree:
    """A root that already holds shims decides the shape a new one takes."""

    def test_a_bundle_of_single_lane_vendors_gets_flat_leaves(self, root, capsys):
        _leaf_file(root, "kilo", name="kilo--chat_completions")

        code = _script().main(["acme", "--no-input", "--root", str(root)])

        out = capsys.readouterr().out
        assert code == 0
        assert "single-lane vendor" in out
        leaf = root / "acme" / "chat_completions" / "provider.yaml"
        assert leaf.is_file()
        assert not (root / "acme" / "provider.yaml").exists()
        assert "api_key_env: ACME_API_KEY" in leaf.read_text(encoding="utf-8")

    def test_a_manifest_group_gets_the_typed_layout(self, root):
        (root / "google").mkdir()
        (root / "google" / "provider.yaml").write_text(
            "connection:\n  api_key_env: GOOGLE_API_KEY\n", encoding="utf-8"
        )
        _leaf_file(root, "google/llm/chat_completions", name="google--chat_completions")

        code = _script().main(["acme", "--no-input", "--root", str(root)])

        assert code == 0
        assert (root / "acme" / "llm" / "chat_completions" / "provider.yaml").is_file()

    def test_a_lane_below_a_single_lane_vendor_is_refused(self, root, capsys):
        _leaf_file(root, "kilo", name="kilo--chat_completions")

        code = _script().main(["kilo", "--no-input", "--root", str(root)])

        err = capsys.readouterr().err
        assert code == 1
        assert "is a shim, not a vendor manifest" in err
        assert "never looks below it" in err
        assert not (root / "kilo" / "chat_completions").exists()

    def test_a_name_another_leaf_declares_is_refused(self, root, capsys):
        _leaf_file(root, "acme/llm/chat_completions", name="acme--chat_completions")

        code = _script().main(
            ["acme", "--no-input", "--root", str(root), "--layout", "flat"]
        )

        err = capsys.readouterr().err
        assert code == 1
        assert "'acme--chat_completions' is already declared by" in err
        assert not (root / "acme" / "chat_completions").exists()

    def test_a_second_shape_inside_one_vendor_is_noted(self, root, capsys):
        _leaf_file(root, "acme/llm/chat_completions", name="acme--chat_completions")

        code = _script().main(
            [
                "acme",
                "--no-input",
                "--root",
                str(root),
                "--format",
                "messages",
                "--layout",
                "flat",
            ]
        )

        out = capsys.readouterr().out
        assert code == 0
        assert "two shapes" in out
        assert (root / "acme" / "messages" / "provider.yaml").is_file()

    def test_a_model_type_name_is_refused_as_a_vendor(self, root, capsys):
        code = _script().main(["llm", "--no-input", "--root", str(root)])

        assert code == 2
        assert "model-type directory name" in capsys.readouterr().err

    def test_a_name_the_loader_skips_is_refused(self, root, capsys):
        code = _script().main(["_acme", "--no-input", "--root", str(root)])

        assert code == 2
        assert "the loader skips such directories" in capsys.readouterr().err

    def test_the_interactive_mode_asks_the_root_first_and_follows_it(
        self, monkeypatch, root, capsys
    ):
        from llm_rosetta.shims import scaffold

        _leaf_file(root, "kilo", name="kilo--chat_completions")
        monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
        monkeypatch.setattr(scaffold, "probe_base_url", lambda url: (True, "ok"))
        monkeypatch.setattr(scaffold, "find_lobehub_logo", lambda vendor: None)
        queue = iter(
            [
                str(root),  # scan root
                "acme",  # vendor
                "llm",  # model type
                "chat_completions",  # format
                "",  # layout: the detected default
                "",  # base URL
                "",  # credential env var
                "",  # models path
                "",  # display name
                "",  # logo
                "",  # group title
                "n",  # transforms stub
            ]
        )
        prompts: list[str] = []

        def answering(prompt: str = "") -> str:
            prompts.append(prompt)
            return next(queue)

        monkeypatch.setattr("builtins.input", answering)

        code = _script().main([])

        out = capsys.readouterr().out
        assert code == 0
        assert "found 1 single-lane vendor" in out
        assert prompts[0].startswith("Scan root")
        layout_prompt = next(p for p in prompts if p.startswith("Layout"))
        assert layout_prompt.endswith("[flat]: ")
        assert (root / "acme" / "chat_completions" / "provider.yaml").is_file()


class TestCtrlD:
    """Ctrl+D ends the questionnaire; it never silently keeps the default."""

    @staticmethod
    def _input(monkeypatch, *, answers: list[str] | None = None):
        from llm_rosetta.shims import scaffold

        monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
        monkeypatch.setattr(scaffold, "probe_base_url", lambda url: (True, "ok"))
        monkeypatch.setattr(scaffold, "find_lobehub_logo", lambda vendor: None)
        queue = iter(answers or [])

        def eof(prompt: str = "") -> str:
            try:
                return next(queue)
            except StopIteration as exc:
                raise EOFError from exc

        monkeypatch.setattr("builtins.input", eof)

    def test_end_of_input_at_the_first_question_cancels(
        self, monkeypatch, root, capsys
    ):
        from llm_rosetta.shims import scaffold

        self._input(monkeypatch)

        code = scaffold.main([])

        err = capsys.readouterr().err
        assert code == 130
        assert "input ended (Ctrl+D)" in err
        assert "no files were written" in err
        assert not (root / "acme").exists()

    def test_end_of_input_mid_questionnaire_cancels(self, monkeypatch, root, capsys):
        from llm_rosetta.shims import scaffold

        self._input(
            monkeypatch,
            answers=[str(root), "acme", "llm", "chat_completions", "typed"],
        )

        code = scaffold.main([])

        err = capsys.readouterr().err
        assert code == 130
        assert "input ended (Ctrl+D)" in err
        assert not (root / "acme").exists()


class TestLobehubLogo:
    """A provider's mark is looked up in the icon set the panel already uses."""

    ICON = (
        "https://cdn.jsdelivr.net/npm/@lobehub/icons-static-svg@latest/icons/"
        "opencode.svg"
    )

    @staticmethod
    def _recording_answers(monkeypatch, answers: list[str], icon: str | None):
        from llm_rosetta.shims import scaffold

        monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
        monkeypatch.setattr(scaffold, "probe_base_url", lambda url: (True, "ok"))
        monkeypatch.setattr(scaffold, "find_lobehub_logo", lambda vendor: icon)
        queue = iter(answers)
        prompts: list[str] = []

        def answering(prompt: str = "") -> str:
            prompts.append(prompt)
            return next(queue)

        monkeypatch.setattr("builtins.input", answering)
        return prompts

    def test_the_candidates_walk_from_the_exact_name_to_the_provider(self):
        from llm_rosetta.shims import scaffold

        candidates = scaffold.lobehub_candidates("opencode_zen_free")

        assert candidates[0] == "opencode_zen_free"
        assert "opencode" in candidates

    def test_a_name_that_is_not_a_slug_offers_nothing(self):
        from llm_rosetta.shims import scaffold

        assert scaffold.lobehub_candidates("a/b c") == ()

    def test_the_first_icon_that_answers_wins(self, monkeypatch):
        from llm_rosetta.shims import scaffold

        asked: list[tuple[str, str | None]] = []

        def status(url: str, **kwargs):
            asked.append((url, kwargs.get("method")))
            if url.endswith("/icons/opencode_zen_free.svg"):
                return 404, ""
            return 200, ""

        monkeypatch.setattr(scaffold, "_http_status", status)

        url = scaffold.find_lobehub_logo("opencode_zen_free")

        assert url == self.ICON
        assert asked[0][1] == "HEAD"
        assert len(asked) == 2

    def test_an_unreachable_cdn_stops_the_search(self, monkeypatch):
        from llm_rosetta.shims import scaffold

        asked: list[str] = []

        def status(url: str, **_kwargs):
            asked.append(url)
            return None, "timed out"

        monkeypatch.setattr(scaffold, "_http_status", status)

        assert scaffold.find_lobehub_logo("opencode_zen_free") is None
        assert len(asked) == 1

    def test_no_icon_under_any_candidate_is_none(self, monkeypatch):
        from llm_rosetta.shims import scaffold

        monkeypatch.setattr(scaffold, "_http_status", lambda url, **kw: (404, ""))

        assert scaffold.find_lobehub_logo("acme") is None

    def test_a_jsdelivr_url_is_kept_without_a_copy(self, monkeypatch, root):
        from llm_rosetta.shims import scaffold

        _fake_repository(monkeypatch, root.parent)

        def boom(_url):
            raise AssertionError("a jsDelivr URL must not be downloaded")

        monkeypatch.setattr(scaffold, "_download_logo", boom)

        url, note = scaffold.localize_logo(self.ICON, root, "opencode")

        assert url == self.ICON
        assert "already served from jsDelivr" in note
        assert not (root.parent / "assets").exists()

    def test_the_fastly_host_counts_as_jsdelivr_too(self):
        from llm_rosetta.shims import scaffold

        assert scaffold.is_jsdelivr("https://fastly.jsdelivr.net/gh/x/y@main/a.svg")
        assert scaffold.is_jsdelivr("https://cdn.jsdelivr.net/npm/x/a.svg")
        assert not scaffold.is_jsdelivr("https://example.com/a.svg")

    def test_the_icon_is_offered_as_the_default(self, monkeypatch, root, capsys):
        prompts = self._recording_answers(
            monkeypatch,
            [
                str(root),  # scan root
                "acme",  # vendor
                "llm",  # model type
                "chat_completions",  # format
                "typed",  # layout
                "",  # base URL
                "",  # credential env var
                "",  # models path
                "",  # display name
                "",  # logo: accept the lobehub default
                "",  # group title
                "n",  # transforms stub
            ],
            self.ICON,
        )

        code = _script().main([])

        out = capsys.readouterr().out
        assert code == 0
        assert "lobehub has an icon for acme" in out
        assert any(prompt.endswith(f"[{self.ICON}]: ") for prompt in prompts)
        manifest = (root / "acme" / "provider.yaml").read_text(encoding="utf-8")
        assert f"logo: {self.ICON}" in manifest
        assert not any(prompt.startswith("Save the mark") for prompt in prompts)

    def test_a_pasted_jsdelivr_url_is_not_offered_for_saving(
        self, monkeypatch, root, capsys
    ):
        prompts = self._recording_answers(
            monkeypatch,
            [
                str(root),
                "acme",
                "llm",
                "chat_completions",
                "typed",
                "",
                "",
                "",
                "",
                self.ICON,  # logo typed by hand
                "",  # group title
                "n",  # transforms stub
            ],
            None,
        )

        code = _script().main([])

        out = capsys.readouterr().out
        assert code == 0
        assert "nothing to copy into the package" in out
        assert not any(prompt.startswith("Save the mark") for prompt in prompts)
        manifest = (root / "acme" / "provider.yaml").read_text(encoding="utf-8")
        assert f"logo: {self.ICON}" in manifest

    def test_a_non_jsdelivr_url_still_offers_to_save_it(self, monkeypatch, root):
        prompts = self._recording_answers(
            monkeypatch,
            [
                str(root),
                "acme",
                "llm",
                "chat_completions",
                "typed",
                "",
                "",
                "",
                "",
                "https://example.com/acme.svg",
                "n",  # do not save it
                "",  # group title
                "n",  # transforms stub
            ],
            None,
        )

        code = _script().main([])

        assert code == 0
        assert any(prompt.startswith("Save the mark") for prompt in prompts)

    def test_the_flag_mode_takes_the_icon(self, monkeypatch, root, capsys):
        from llm_rosetta.shims import scaffold

        monkeypatch.setattr(scaffold, "find_lobehub_logo", lambda vendor: self.ICON)

        code = _script().main(
            ["acme", "--no-input", "--root", str(root), "--auto-logo"]
        )

        out = capsys.readouterr().out
        assert code == 0
        assert "lobehub, acme" in out
        manifest = (root / "acme" / "provider.yaml").read_text(encoding="utf-8")
        assert f"logo: {self.ICON}" in manifest

    def test_the_flag_mode_fails_when_there_is_no_icon(self, monkeypatch, root, capsys):
        from llm_rosetta.shims import scaffold

        monkeypatch.setattr(scaffold, "find_lobehub_logo", lambda vendor: None)

        code = _script().main(
            ["acme", "--no-input", "--root", str(root), "--auto-logo"]
        )

        assert code == 2
        assert "no lobehub icon found for 'acme'" in capsys.readouterr().err
        assert not (root / "acme").exists()
