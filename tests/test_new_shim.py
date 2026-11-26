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
from llm_rosetta.shims import _loader as providers_module
from llm_rosetta.shims._loader import load_providers, load_providers_from_dir
from llm_rosetta.devtools import scaffold
from llm_rosetta.devtools.scaffold import check_tree
from llm_rosetta.devtools.scaffold import apply_migration, plan_migration
from llm_rosetta.devtools.scaffold import run_migrate

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

    leaf = root / "acme" / "llm" / "provider.yaml"
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


def test_a_second_lane_moves_the_first_into_a_format_directory(new_shim, root):
    new_shim.main(["acme", "--root", str(root)])

    code = new_shim.main(["acme", "--root", str(root), "--format", "open_responses"])

    assert code == 0
    assert (root / "acme" / "llm" / "chat_completions" / "provider.yaml").is_file()
    assert (root / "acme" / "llm" / "open_responses" / "provider.yaml").is_file()
    assert not (root / "acme" / "llm" / "provider.yaml").exists()
    names = sorted(s.name for s in load_providers_from_dir(root))
    assert names == ["acme--chat_completions", "acme--open_responses"]


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


@pytest.mark.parametrize(
    "flag",
    [
        ["--package", "llm-rosetta-shim-acme"],
        ["--localize-logo"],
        ["--no-registry-manifest"],
    ],
)
def test_the_external_package_flags_are_gone(new_shim, root, flag, capsys):
    """An extension folder is the shipped form; the scaffold writes only a tree."""
    with pytest.raises(SystemExit) as raised:
        new_shim.main(["acme", "--root", str(root), *flag])

    assert raised.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
    assert not (root / "acme").exists()


def test_a_stray_shims_yaml_is_neither_read_nor_written(new_shim, root):
    original = "version: 1\nshims: []\n"
    (root / "shims.yaml").write_text(original, encoding="utf-8")

    assert new_shim.main(["acme", "--root", str(root)]) == 0

    assert (root / "shims.yaml").read_text(encoding="utf-8") == original


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

        self._answers(
            monkeypatch,
            [
                str(root),  # scan root, asked first so the shape is known
                "acme",  # vendor
                "llm",  # model type
                "chat_completions",  # format
                "",  # lane
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
        leaf = root / "acme" / "llm" / "provider.yaml"
        assert leaf.is_file()
        assert "display_name: acme" in leaf.read_text(encoding="utf-8")

    def test_answers_given_as_flags_are_not_asked_for_again(self, monkeypatch, root):

        self._answers(
            monkeypatch,
            [
                str(root),  # scan root
                "llm",  # model type
                "chat_completions",  # format
                "",  # lane
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

        code = scaffold.main(["--no-input", "--root", str(root)])

        assert code == 2
        assert "vendor name is required" in capsys.readouterr().err


class TestScriptExitCode:
    """``scripts/dev/new_shim.py`` is the one command that writes a shim tree.

    The gateway's ``shims new`` is gone (``llm-rosetta ext new`` creates an
    extension folder instead), so the wrapper is where the scaffolder's exit
    code reaches a shell.
    """

    def test_it_scaffolds(self, new_shim, root):
        code = new_shim.main(
            [
                "acme",
                "--no-input",
                "--base-url",
                "https://api.acme.example/v1",
                "--root",
                str(root),
            ]
        )

        assert code == 0
        assert (root / "acme" / "llm" / "provider.yaml").is_file()

    def test_a_failed_load_check_reaches_the_shell(self, new_shim, monkeypatch, root):
        def refuse(*_args, **_kwargs):
            raise ValueError("'chat_completions' is not an embeddings format")

        # Undone before the registry fixture reloads the built-ins on teardown.
        with monkeypatch.context() as patch:
            patch.setattr(providers_module, "load_providers_from_dir", refuse)
            code = new_shim.main(["acme", "--no-input", "--check", "--root", str(root)])

        assert code == 1


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
                "",  # lane
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

        monkeypatch.setattr(
            scaffold, "_http_status", lambda url, **_kw: statuses.get(url, (404, ""))
        )

    def test_a_models_200_passes(self, monkeypatch):

        self._status(monkeypatch, {"https://api.acme.example/v1/models": (200, "")})

        ok, detail = scaffold.probe_base_url("https://api.acme.example/v1")

        assert ok is True
        assert "answers 200" in detail

    def test_an_authentication_refusal_proves_reachability(self, monkeypatch):

        self._status(monkeypatch, {"https://api.acme.example/v1/models": (401, "")})

        ok, detail = scaffold.probe_base_url("https://api.acme.example/v1")

        assert ok is True
        assert "without a key" in detail

    def test_a_404_asks_for_the_suffix(self, monkeypatch):

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

        self._status(
            monkeypatch, {"https://api.acme.example/v1/models": (None, "timed out")}
        )

        ok, detail = scaffold.probe_base_url("https://api.acme.example/v1")

        assert ok is False
        assert "cannot reach" in detail and "timed out" in detail

    def test_a_pasted_models_suffix_is_dropped(self):

        url, note = scaffold.normalize_base_url(" https://api.acme.example/v1/models/ ")

        assert url == "https://api.acme.example/v1"
        assert "/models suffix" in note

    def test_a_clean_url_is_left_alone(self):

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
        leaf = root / "acme" / "llm" / "provider.yaml"
        assert "base_url: https://api.acme.example/v1" in leaf.read_text(
            encoding="utf-8"
        )


class TestCtrlC:
    """One Ctrl+C re-asks; a second one ends the run without a traceback."""

    @staticmethod
    def _interrupting(monkeypatch, answers: list[str], at: set[int]):

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

        self._interrupting(
            monkeypatch,
            [
                str(root),  # scan root
                "acme",
                "llm",
                "chat_completions",
                "",  # lane
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
        assert (root / "acme" / "llm" / "provider.yaml").is_file()

    def test_a_second_ctrl_c_cancels_without_writing(self, monkeypatch, root, capsys):

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

    def test_a_bundle_of_single_lane_vendors_gets_a_typed_leaf(self, root, capsys):
        _leaf_file(root, "kilo", name="kilo--chat_completions")

        code = _script().main(["acme", "--no-input", "--root", str(root)])

        assert code == 0
        assert (root / "acme" / "llm" / "provider.yaml").is_file()
        # The credential still lives one level up, in the vendor manifest.
        manifest = (root / "acme" / "provider.yaml").read_text(encoding="utf-8")
        assert "api_key_env: ACME_API_KEY" in manifest

    def test_a_manifest_group_gets_the_typed_layout(self, root):
        (root / "google").mkdir()
        (root / "google" / "provider.yaml").write_text(
            "connection:\n  api_key_env: GOOGLE_API_KEY\n", encoding="utf-8"
        )
        _leaf_file(root, "google/llm/chat_completions", name="google--chat_completions")

        code = _script().main(["acme", "--no-input", "--root", str(root)])

        assert code == 0
        assert (root / "acme" / "llm" / "provider.yaml").is_file()

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

        code = _script().main(["acme", "--no-input", "--root", str(root)])

        err = capsys.readouterr().err
        assert code == 1
        assert "'acme--chat_completions' is already declared by" in err

    def test_the_second_lane_moves_the_existing_one(self, root, capsys):
        _leaf_file(root, "acme/llm", name="acme--chat_completions")

        code = _script().main(
            ["acme", "--no-input", "--root", str(root), "--format", "messages"]
        )

        out = capsys.readouterr().out
        assert code == 0
        assert "moved the existing lane into" in out
        assert (root / "acme" / "llm" / "chat_completions" / "provider.yaml").is_file()
        assert (root / "acme" / "llm" / "messages" / "provider.yaml").is_file()

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

        assert code == 0
        assert prompts[0].startswith("Scan root")
        assert any(p.startswith("Lane directory") for p in prompts)
        assert (root / "acme" / "llm" / "provider.yaml").is_file()


class TestCtrlD:
    """Ctrl+D ends the questionnaire; it never silently keeps the default."""

    @staticmethod
    def _input(monkeypatch, *, answers: list[str] | None = None):

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

        self._input(monkeypatch)

        code = scaffold.main([])

        err = capsys.readouterr().err
        assert code == 130
        assert "input ended (Ctrl+D)" in err
        assert "no files were written" in err
        assert not (root / "acme").exists()

    def test_end_of_input_mid_questionnaire_cancels(self, monkeypatch, root, capsys):

        self._input(
            monkeypatch,
            answers=[str(root), "acme", "llm", "chat_completions", ""],
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

        candidates = scaffold.lobehub_candidates("opencode_zen_free")

        assert candidates[0] == "opencode_zen_free"
        assert "opencode" in candidates

    def test_a_name_that_is_not_a_slug_offers_nothing(self):

        assert scaffold.lobehub_candidates("a/b c") == ()

    def test_the_first_icon_that_answers_wins(self, monkeypatch):

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

        asked: list[str] = []

        def status(url: str, **_kwargs):
            asked.append(url)
            return None, "timed out"

        monkeypatch.setattr(scaffold, "_http_status", status)

        assert scaffold.find_lobehub_logo("opencode_zen_free") is None
        assert len(asked) == 1

    def test_no_icon_under_any_candidate_is_none(self, monkeypatch):

        monkeypatch.setattr(scaffold, "_http_status", lambda url, **kw: (404, ""))

        assert scaffold.find_lobehub_logo("acme") is None

    def test_the_icon_is_offered_as_the_default(self, monkeypatch, root, capsys):
        prompts = self._recording_answers(
            monkeypatch,
            [
                str(root),  # scan root
                "acme",  # vendor
                "llm",  # model type
                "chat_completions",  # format
                "",  # lane
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
                "",  # lane
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

        assert code == 0
        assert not any(prompt.startswith("Save the mark") for prompt in prompts)
        manifest = (root / "acme" / "provider.yaml").read_text(encoding="utf-8")
        assert f"logo: {self.ICON}" in manifest

    def test_the_flag_mode_takes_the_icon(self, monkeypatch, root, capsys):

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

        monkeypatch.setattr(scaffold, "find_lobehub_logo", lambda vendor: None)

        code = _script().main(
            ["acme", "--no-input", "--root", str(root), "--auto-logo"]
        )

        assert code == 2
        assert "no lobehub icon found for 'acme'" in capsys.readouterr().err
        assert not (root / "acme").exists()


class TestTemplate:
    """The template directory, and what the scaffold derives from it."""

    _VALUES = {
        "vendor": "acme",
        "group": "acme",
        "group_label": "Acme",
        "logo": "https://example.invalid/acme.svg",
        "api_key_env": "ACME_API_KEY",
        "name": "acme--chat_completions",
        "model_type": "llm",
        "format": "chat_completions",
        "base_url": "https://api.acme.example/v1",
        "display_name": "Acme",
        "models_path": "https://api.acme.example/v1/models",
    }

    def test_every_template_renders(self):
        """A placeholder the scaffold does not fill must fail here, not at use."""

        for template in sorted(scaffold.TEMPLATE_DIR.rglob("*.tmpl")):
            rendered = scaffold._render(
                template.read_text(encoding="utf-8"), self._VALUES
            )
            assert "{{" not in rendered, template

    def test_the_tree_carries_exactly_the_modules_the_loader_reads(self):
        """The template set and the loader's allow-list are the same list."""

        names = {path.name for path in scaffold.TREE_DIR.rglob("*.tmpl")}
        assert names == {
            "provider.yaml.tmpl",
            "transforms.py.tmpl",
            "auth.py.tmpl",
            "quota.py.tmpl",
        }

    def test_a_line_with_an_empty_placeholder_is_dropped(self):
        """Optional fields stay on one line in the template."""

        text = "a: {{one}}\nb: {{two}}\nc: x\n"

        assert scaffold._render(text, {"one": "1", "two": ""}) == "a: 1\nc: x\n"

    def test_the_second_lane_moves_the_leaf_modules_with_it(self, new_shim, root):
        new_shim.main(["acme", "--root", str(root), "--transform"])
        new_shim.main(["acme", "--root", str(root), "--format", "messages"])

        assert (root / "acme" / "llm" / "chat_completions" / "transforms.py").is_file()
        assert not (root / "acme" / "llm" / "transforms.py").exists()


class TestCheckTree:
    """`shims check` reports what the loader would not read, and nothing else."""

    @staticmethod
    def _manifest(vendor_dir: Path) -> None:
        vendor_dir.mkdir(parents=True, exist_ok=True)
        (vendor_dir / "provider.yaml").write_text(
            "connection:\n  api_key_env: ACME_API_KEY\n", encoding="utf-8"
        )

    @staticmethod
    def _leaf(directory: Path, name: str) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "provider.yaml"
        path.write_text(
            f"name: {name}\nmodel_type: llm\nformat: chat_completions\n"
            "connection:\n  base_url: https://api.acme.example/v1\n",
            encoding="utf-8",
        )
        return path

    def test_a_canonical_tree_has_no_findings(self, tmp_path: Path):

        self._manifest(tmp_path / "acme")
        self._leaf(tmp_path / "acme" / "llm", "acme--chat_completions")

        assert check_tree(tmp_path) == []

    def test_a_vendor_directory_used_as_a_leaf_is_reported(self, tmp_path: Path):

        self._leaf(tmp_path / "acme", "acme--chat_completions")

        findings = check_tree(tmp_path)
        assert len(findings) == 1
        assert "a vendor directory is a leaf" in findings[0].message

    def test_a_relative_token_command_without_its_file_is_reported(
        self, tmp_path: Path
    ):
        """The command runs in the leaf directory, so the file must be there.

        A moved leaf leaves `../token.py` one level short, and nothing fails
        until the first credential exchange.
        """

        self._manifest(tmp_path / "acme")
        leaf = tmp_path / "acme" / "llm"
        leaf.mkdir(parents=True)
        (leaf / "provider.yaml").write_text(
            "name: acme--chat_completions\nmodel_type: llm\n"
            "format: chat_completions\nconnection:\n"
            "  token_command: [python3, ../token.py]\n",
            encoding="utf-8",
        )

        findings = check_tree(tmp_path)

        assert len(findings) == 1
        assert "connection.token_command" in findings[0].message

        (tmp_path / "acme" / "token.py").write_text("", encoding="utf-8")
        assert check_tree(tmp_path) == []

    def test_a_broken_fields_block_is_reported_before_it_is_installed(
        self, tmp_path: Path
    ):
        """The loader refuses the whole root, so the check must say which file.

        The panel renders those entries verbatim; a typo in one would otherwise
        surface as a gateway that will not load the tree at all.
        """

        (tmp_path / "acme").mkdir()
        (tmp_path / "acme" / "provider.yaml").write_text(
            "fields:\n  - key: email\n", encoding="utf-8"
        )
        self._leaf(tmp_path / "acme" / "llm", "acme--chat_completions")

        findings = check_tree(tmp_path)

        assert [f.path.name for f in findings] == ["provider.yaml"]
        assert "declares no label" in findings[0].message
        # The finding carries the path, so the sentence does not repeat it.
        assert str(tmp_path) not in findings[0].message

    def test_a_well_formed_fields_block_is_not_a_finding(self, tmp_path: Path):

        (tmp_path / "acme").mkdir()
        (tmp_path / "acme" / "provider.yaml").write_text(
            "fields:\n  - key: email\n    label: Account email\n", encoding="utf-8"
        )
        self._leaf(tmp_path / "acme" / "llm", "acme--chat_completions")

        assert check_tree(tmp_path) == []

    def test_a_retired_exchange_in_a_private_directory_is_not_a_second_one(
        self, tmp_path: Path
    ):
        """``_old/`` is where a package parks what the loader must not read.

        No scan reaches into it, so an ``auth.py`` there is not a competing
        exchange — reporting it would send the author looking for a conflict
        that does not exist.
        """

        self._manifest(tmp_path / "acme")
        (tmp_path / "acme" / "auth.py").write_text(
            "EXCHANGE = None\n", encoding="utf-8"
        )
        self._leaf(tmp_path / "acme" / "llm", "acme--chat_completions")
        retired = tmp_path / "acme" / "_old"
        retired.mkdir()
        (retired / "auth.py").write_text("EXCHANGE = None\n", encoding="utf-8")

        assert check_tree(tmp_path) == []

    def test_a_type_directory_holding_a_leaf_and_a_lane_is_reported(
        self, tmp_path: Path
    ):
        """The half-migrated shape: the flat leaf is the one that moves.

        Reporting the nested lane as "the only lane ... it belongs directly
        under <type>/" names a move that would collide with the leaf already
        there, so the finding has to name the flat leaf and send it down.
        """

        self._manifest(tmp_path / "acme")
        flat = self._leaf(tmp_path / "acme" / "llm", "acme--chat_completions")
        self._leaf(tmp_path / "acme" / "llm" / "messages", "acme--messages")

        findings = check_tree(tmp_path)

        assert len(findings) == 1
        assert findings[0].path == flat
        assert "holds a leaf and format directories at once" in findings[0].message
        assert "Move this leaf down" in findings[0].message

    def test_a_leaf_without_a_type_directory_is_reported(self, tmp_path: Path):

        self._manifest(tmp_path / "acme")
        self._leaf(tmp_path / "acme" / "chat_completions", "acme--chat_completions")

        findings = check_tree(tmp_path)
        assert len(findings) == 1
        assert "no <type>/ directory" in findings[0].message

    def test_the_only_lane_at_a_format_level_is_reported(self, tmp_path: Path):

        self._manifest(tmp_path / "acme")
        self._leaf(
            tmp_path / "acme" / "llm" / "chat_completions", "acme--chat_completions"
        )

        findings = check_tree(tmp_path)
        assert len(findings) == 1
        assert "the only lane of llm" in findings[0].message

    def test_two_lanes_at_a_format_level_conform(self, tmp_path: Path):

        self._manifest(tmp_path / "acme")
        self._leaf(
            tmp_path / "acme" / "llm" / "chat_completions", "acme--chat_completions"
        )
        second = tmp_path / "acme" / "llm" / "messages"
        second.mkdir()
        (second / "provider.yaml").write_text(
            "name: acme--messages\nmodel_type: llm\nformat: messages\n"
            "connection:\n  base_url: https://api.acme.example/v1\n",
            encoding="utf-8",
        )

        assert check_tree(tmp_path) == []

    def test_a_vendor_level_transforms_file_is_reported(self, tmp_path: Path):

        self._manifest(tmp_path / "acme")
        self._leaf(tmp_path / "acme" / "llm", "acme--chat_completions")
        (tmp_path / "acme" / "transforms.py").write_text("", encoding="utf-8")

        findings = check_tree(tmp_path)
        assert len(findings) == 1
        assert "vendor-level one is never read" in findings[0].message

    def test_a_second_exchange_below_the_vendor_is_reported(self, tmp_path: Path):

        self._manifest(tmp_path / "acme")
        self._leaf(tmp_path / "acme" / "llm", "acme--chat_completions")
        (tmp_path / "acme" / "auth.py").write_text("", encoding="utf-8")
        (tmp_path / "acme" / "llm" / "auth.py").write_text("", encoding="utf-8")

        findings = check_tree(tmp_path)
        assert len(findings) == 1
        assert "already ships its exchange" in findings[0].message

    def test_a_private_helper_module_is_not_reported(self, tmp_path: Path):

        self._manifest(tmp_path / "acme")
        self._leaf(tmp_path / "acme" / "llm", "acme--chat_completions")
        (tmp_path / "acme" / "model_utils.py").write_text("", encoding="utf-8")

        assert check_tree(tmp_path) == []


class TestMigrate:
    """`shims migrate` moves a tree onto the contract, and only what it can."""

    @staticmethod
    def _leaf(
        directory: Path, name: str, *, model_type: str = "llm", fmt="chat_completions"
    ):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "provider.yaml").write_text(
            f"name: {name}\nmodel_type: {model_type}\nformat: {fmt}\n"
            "connection:\n  base_url: https://api.acme.example/v1\n",
            encoding="utf-8",
        )

    @staticmethod
    def _manifest(vendor_dir: Path) -> None:
        vendor_dir.mkdir(parents=True, exist_ok=True)
        (vendor_dir / "provider.yaml").write_text(
            "connection:\n  api_key_env: ACME_API_KEY\n", encoding="utf-8"
        )

    def test_a_vendor_used_as_a_leaf_gets_a_manifest_and_a_type_level(
        self, tmp_path: Path
    ):

        self._leaf(tmp_path / "acme", "acme--chat_completions")

        moves, notes = plan_migration(tmp_path)
        assert notes == []
        apply_migration(tmp_path, moves)

        assert (tmp_path / "acme" / "llm" / "provider.yaml").is_file()
        manifest = (tmp_path / "acme" / "provider.yaml").read_text(encoding="utf-8")
        assert manifest.endswith("connection: {}\n")

    def test_a_format_directory_under_the_vendor_moves_under_its_type(
        self, tmp_path: Path
    ):

        self._manifest(tmp_path / "acme")
        for fmt in ("chat_completions", "messages"):
            self._leaf(tmp_path / "acme" / fmt, f"acme--{fmt}", fmt=fmt)

        apply_migration(tmp_path, plan_migration(tmp_path)[0])

        assert (
            tmp_path / "acme" / "llm" / "chat_completions" / "provider.yaml"
        ).is_file()
        assert (tmp_path / "acme" / "llm" / "messages" / "provider.yaml").is_file()

    def test_a_lane_gets_its_type_directory_below_it(self, tmp_path: Path):

        self._manifest(tmp_path / "alcf")
        for cluster in ("metis", "minerva"):
            self._leaf(tmp_path / "alcf" / cluster, f"alcf_{cluster}--chat_completions")
            (tmp_path / "alcf" / cluster / "transforms.py").write_text(
                "", encoding="utf-8"
            )

        apply_migration(tmp_path, plan_migration(tmp_path)[0])

        for cluster in ("metis", "minerva"):
            leaf = tmp_path / "alcf" / cluster / "llm"
            assert (leaf / "provider.yaml").is_file()
            assert (leaf / "transforms.py").is_file()
        assert not (tmp_path / "alcf" / "metis" / "llm" / "llm").exists()

    def test_the_only_lane_at_a_format_level_moves_up(self, tmp_path: Path):

        self._manifest(tmp_path / "acme")
        self._leaf(
            tmp_path / "acme" / "llm" / "chat_completions", "acme--chat_completions"
        )

        apply_migration(tmp_path, plan_migration(tmp_path)[0])

        assert (tmp_path / "acme" / "llm" / "provider.yaml").is_file()
        assert not (tmp_path / "acme" / "llm" / "chat_completions").exists()

    def test_a_dry_run_moves_nothing(self, tmp_path: Path, capsys):

        self._leaf(tmp_path / "acme", "acme--chat_completions")

        code = run_migrate(tmp_path)

        out = capsys.readouterr().out
        assert code == 1
        assert "rerun with --write" in out
        assert (tmp_path / "acme" / "provider.yaml").is_file()
        assert not (tmp_path / "acme" / "llm").exists()

    def test_a_second_lane_leaves_the_sibling_lane_where_it_is(self, tmp_path: Path):
        """The flat lane moves down; the lane already at a format level stays.

        The type directory is both the flat lane's home and the parent of a
        sibling lane, so moving it wholesale would drag the sibling one level
        deeper — a tree this migration's own `check` then rejects.
        """

        self._manifest(tmp_path / "acme")
        lane = tmp_path / "acme" / "llm"
        self._leaf(lane, "acme--messages", fmt="messages")
        (lane / "transforms.py").write_text("", encoding="utf-8")
        self._leaf(lane / "chat_completions", "acme--chat_completions")

        apply_migration(tmp_path, plan_migration(tmp_path)[0])

        assert (lane / "messages" / "provider.yaml").is_file()
        assert (lane / "messages" / "transforms.py").is_file()
        assert (lane / "chat_completions" / "provider.yaml").is_file()
        assert not (lane / "messages" / "chat_completions").exists()
        assert check_tree(tmp_path) == []

    def test_a_vendor_leaf_with_a_sibling_lane_keeps_the_sibling(self, tmp_path: Path):
        """The vendor leaf moves down without taking its sibling with it.

        Dragging the sibling would leave the second move with no source, so the
        migration would fail halfway through.
        """

        self._leaf(tmp_path / "acme", "acme--chat_completions")
        self._leaf(tmp_path / "acme" / "messages", "acme--messages", fmt="messages")

        apply_migration(tmp_path, plan_migration(tmp_path)[0])

        assert (
            tmp_path / "acme" / "llm" / "chat_completions" / "provider.yaml"
        ).is_file()
        assert (tmp_path / "acme" / "llm" / "messages" / "provider.yaml").is_file()
        assert not (
            tmp_path / "acme" / "llm" / "chat_completions" / "messages"
        ).exists()
        assert check_tree(tmp_path) == []

    def test_a_conforming_tree_has_nothing_to_do(self, tmp_path: Path, capsys):

        self._manifest(tmp_path / "acme")
        self._leaf(tmp_path / "acme" / "llm", "acme--chat_completions")

        code = run_migrate(tmp_path, write=True)

        assert code == 0
        assert "nothing to migrate" in capsys.readouterr().out
