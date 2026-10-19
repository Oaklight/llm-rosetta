"""Tests for the shim-removal gate (``scripts/check_shim_removals.py``).

The gate is the half of the sunset that cannot be forgotten: it passes while the
version is below a removal and fails on its own once the version reaches it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from llm_rosetta.shims import SHIM_REMOVALS, ShimRemoval

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_shim_removals.py"


def _load_script():
    """Import the script by path (it is not an installed module)."""
    spec = importlib.util.spec_from_file_location("check_shim_removals", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestGateOnThisCheckout:
    """What the gate says about the repository as it stands."""

    def test_the_checkout_is_consistent(self):
        """Nothing is due yet, so the repository may still name these shims."""
        assert _load_script().main([]) == 0

    def test_the_whole_manifest_arms_at_the_removal_version(self):
        assert _load_script().main(["--version", "0.16.0", "--quiet"]) == 1

    def test_it_names_the_mockup_that_needs_migrating(self):
        """The 0.16.0 failure list is actionable, not just red."""
        problems = _load_script().check("0.16.0", ["design"], [])
        assert any("provider-list-responsive.html" in problem for problem in problems)

    def test_the_default_roots_reach_the_code_that_has_to_change(self):
        """The gate's own cleanup list has to be inside what it scans.

        A removal owes more than prose: the ``LEGACY_SHIM_ALIASES`` entry and the
        built-in ``provider.yaml`` go with the shim.  The registration half
        cannot stand in for scanning them, because a shim that is already gone
        resolves to ``None`` whether or not its alias is still there.
        """
        module = _load_script()
        assert {"src", "scripts", "design"} <= set(module.DEFAULT_ROOTS)
        # The manifest names every scheduled shim by design, so it exempts itself.
        assert "src/llm_rosetta/shims/removals.py" in module.DEFAULT_EXEMPT_FILES
        # The source catalogue names the shims each package provides.
        assert "src/llm_rosetta/shims/catalog.py" in module.DEFAULT_EXEMPT_FILES

    def test_the_shipped_scripts_and_mock_ups_are_in_reach(self):
        """Every root that names a scheduled shim today is actually scanned.

        Named explicitly because these are the ones a reader would not think to
        add: a config *example* inside a helper script, and a UI mock-up.  Both
        fail here only once a removal is due, which is the point.
        """
        module = _load_script()
        assert module.check("0.16.0", ["scripts", "design"], [])

    def test_a_leftover_alias_under_src_is_reported(self, tmp_path):
        module = _load_script()
        alias = tmp_path / "src" / "llm_rosetta" / "provider_names.py"
        alias.parent.mkdir(parents=True)
        alias.write_text('    "argo--openai_chat": "argo--chat_completions",\n')

        problems = module.check("0.16.0", [str(tmp_path / "src")], [])
        assert any("provider_names.py" in problem for problem in problems)


class TestReferenceHalf:
    """Stale mentions under the scanned roots."""

    def test_a_mention_is_ignored_while_the_removal_is_not_due(self, tmp_path):
        (tmp_path / "config.jsonc").write_text('{"shim": "argo--anthropic"}\n')
        assert _load_script().check("0.15.0", [str(tmp_path)], []) == []

    def test_a_mention_fails_once_the_removal_is_due(self, tmp_path):
        (tmp_path / "config.jsonc").write_text('{"shim": "argo--anthropic"}\n')
        problems = _load_script().check("0.16.0", [str(tmp_path)], [])
        assert any("config.jsonc:1" in problem for problem in problems)

    def test_an_exempt_file_is_not_reported(self, tmp_path):
        changelog = tmp_path / "changelog.md"
        changelog.write_text("`argo--anthropic` was removed in 0.16.0\n")
        problems = _load_script().check("0.16.0", [str(tmp_path)], [str(changelog)])
        assert not any("changelog.md" in problem for problem in problems)


class TestManifestHalf:
    """The two ways the manifest and the registry can disagree."""

    def test_a_shim_removed_before_its_version_is_reported(self, monkeypatch):
        module = _load_script()
        early = ShimRemoval(
            shim="not-a-real-shim--chat_completions",
            removed_in="0.16.0",
            remote={"url": "https://github.com/x/y", "path": "src/p/providers"},
            source_repo="https://github.com/x/y",
        )
        monkeypatch.setattr(module, "SHIM_REMOVALS", (early,))
        problems = module.check("0.15.0", [], [])
        assert len(problems) == 1
        assert "is not registered" in problems[0]

    def test_a_shim_still_present_at_its_version_is_reported(self):
        module = _load_script()
        problems = module.check("0.16.0", [], [])
        still_registered = [
            problem for problem in problems if "is still registered" in problem
        ]
        assert len(still_registered) == len(SHIM_REMOVALS)
