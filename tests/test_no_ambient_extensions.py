"""The suite never sees the developer's own installed extensions."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def test_the_per_user_extensions_directory_is_not_on_sys_path() -> None:
    # `import llm_rosetta` activates that directory when it exists; the root
    # conftest points the variable at an empty path before the first import.
    real = Path("~/.local/share/llm-rosetta/extensions").expanduser()
    assert str(real) not in sys.path


def test_the_session_variable_names_a_directory_that_does_not_exist() -> None:
    value = os.environ["LLM_ROSETTA_EXTENSIONS_DIR"]
    assert not Path(value).exists()
