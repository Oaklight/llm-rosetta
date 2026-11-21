"""Every shipped and fixture ``provider.yaml`` is valid YAML (#1660).

The loader reads shim declarations with the vendored zero-dependency parser
(``llm_rosetta._vendor.yaml``), which is lenient: it accepted a double-quoted
``hint_msg`` with unescaped ``"`` inside, which a strict parser rejects.  A
producer repository that validates its tree with a strict parser would then
disagree with core about the same file.  Each file must therefore parse with
PyYAML and give the same value as the vendored parser.  PyYAML is a test-only
dependency here; the check is skipped where it is not installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _moved_vendors import MOVED_VENDORS_DIR
from llm_rosetta._vendor.yaml import load as vendored_load
from llm_rosetta.shims import providers as providers_pkg


def _load_pyyaml():
    """PyYAML itself, even when ``yaml`` names the vendored parser.

    Another test module can leave a ``_vendor`` directory on ``sys.path``, after
    which ``import yaml`` finds the vendored zero-dependency parser; the strict
    half of this file would then compare that parser with itself.  PyYAML is
    therefore located with every ``_vendor`` entry left out, loaded, and taken
    back out of ``sys.modules`` so nothing else sees the swap.
    """
    import importlib.machinery
    import importlib.util
    import sys

    paths = [p for p in sys.path if "_vendor" not in p]
    spec = importlib.machinery.PathFinder.find_spec("yaml", paths)
    if spec is None or spec.loader is None:
        return None
    mine = lambda key: key == "yaml" or key.startswith("yaml.")  # noqa: E731
    saved = {k: v for k, v in sys.modules.items() if mine(k)}
    for key in saved:
        del sys.modules[key]
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules["yaml"] = module
        spec.loader.exec_module(module)
    finally:
        for key in [k for k in sys.modules if mine(k)]:
            del sys.modules[key]
        sys.modules.update(saved)
    return module


yaml = _load_pyyaml()
if yaml is None or not hasattr(yaml, "YAMLError"):
    pytest.skip("PyYAML is not installed", allow_module_level=True)

CORE = Path(providers_pkg.__file__).parent
ROOTS = (CORE, MOVED_VENDORS_DIR)
FILES = sorted(p for root in ROOTS for p in root.rglob("*.yaml"))


def _id(path: Path) -> str:
    for root in ROOTS:
        if root in path.parents:
            return f"{root.name}/{path.relative_to(root)}"
    return str(path)


def test_there_are_files_to_check():
    assert any(CORE in p.parents for p in FILES)
    assert any(MOVED_VENDORS_DIR in p.parents for p in FILES)


@pytest.mark.parametrize("path", FILES, ids=_id)
def test_a_shim_yaml_parses_strictly_and_agrees_with_the_loader(path: Path):
    text = path.read_text(encoding="utf-8")
    strict = yaml.safe_load(text)
    assert strict == vendored_load(text)


def test_the_check_catches_an_unescaped_quote():
    """The defect this file exists for, so the strict half is proven to bite."""
    text = 'hint_msg:\n  en: "declare shims.presets: ["x"] to load it."\n'
    with pytest.raises(yaml.YAMLError):
        yaml.safe_load(text)
