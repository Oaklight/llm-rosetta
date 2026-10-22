"""The docs-build helper runs the branch's own Read the Docs steps.

The value of the helper is that it reads the branch's ``.readthedocs.yaml``
rather than describing it: a CI job that repeats the copy keeps passing while
the branch's yaml has rotted, which is exactly how both doc sites stayed down.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "dev" / "rtd_build.py"


def _load():
    spec = importlib.util.spec_from_file_location("rtd_build", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["rtd_build"] = module
    spec.loader.exec_module(module)
    return module


def test_a_single_command_entry_stays_one_step():
    module = _load()

    assert module._commands("zensical build") == ["zensical build"]


def test_a_command_list_is_flattened_in_order():
    module = _load()

    assert module._commands(["pip install x", "zensical build"]) == [
        "pip install x",
        "zensical build",
    ]


def test_the_script_reads_the_branchs_own_yaml(tmp_path, monkeypatch):
    """The regression case: a branch yaml that copies a path that is gone.

    The helper must fail on the branch's step, not on its own idea of where
    the source lives.
    """
    module = _load()
    branch = tmp_path / "docs_en"
    branch.mkdir()
    (branch / ".readthedocs.yaml").write_text(
        "version: 2\n"
        "build:\n"
        "  jobs:\n"
        "    pre_build:\n"
        "      - cp -r src_repo/src/llm_rosetta src/\n"
        "    build:\n"
        "      html:\n"
        "        - zensical build\n"
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()  # placeholders: git is not reached for these steps

    monkeypatch.setattr(module, "_redirect_env", lambda root: {})

    assert module.main([str(branch)]) == 1
    assert not (branch / "src_repo").exists()
