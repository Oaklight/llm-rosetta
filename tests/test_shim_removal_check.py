"""Tests for the shim-removal gate (``scripts/check_shim_removals.py``).

The gate is the half of the sunset that cannot be forgotten: it passes while the
version is below a removal and fails on its own once the version reaches it.

Nothing is scheduled in the shipped manifest today — ALCF, AskSage and Argo all
ship standalone from 0.14.0 — so the tests that exercise the gate pass a
synthetic manifest of a shim that is still built in.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from llm_rosetta.shims import SHIM_REMOVALS, ShimRemoval

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_shim_removals.py"

SAMPLE = ShimRemoval(
    shim="minimax--messages",
    removed_in="0.16.0",
    remote={
        "url": "https://github.com/Oaklight/llm-rosetta-shim-minimax",
        "path": "src/llm_rosetta_shim_minimax/providers",
    },
    source_repo="https://github.com/Oaklight/llm-rosetta-shim-minimax",
    legacy_names=("minimax--anthropic",),
)
SAMPLES = (SAMPLE,)


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
        """No removals are scheduled, so nothing can be stale."""
        assert _load_script().main([]) == 0

    def test_the_empty_manifest_stays_green_at_any_version(self):
        assert _load_script().main(["--version", "0.16.0", "--quiet"]) == 0

    def test_a_synthetic_manifest_arms_at_its_version(self):
        module = _load_script()
        assert module.check("0.15.0", [], [], removals=SAMPLES) == []
        problems = module.check("0.16.0", [], [], removals=SAMPLES)
        assert len(problems) == 1
        assert "is still registered" in problems[0]
        assert "minimax--messages" in problems[0]

    def test_it_names_the_file_that_needs_migrating(self, tmp_path):
        """The failure list is actionable, not just red."""
        config = tmp_path / "config.jsonc"
        config.write_text('{"shim": "minimax--anthropic"}\n')
        problems = _load_script().check("0.16.0", [str(tmp_path)], [], removals=SAMPLES)
        assert any("config.jsonc:1" in problem for problem in problems)

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

    def test_the_gate_scans_the_real_roots_without_false_positives(self):
        """The shipped roots stay clean for a shim that is not scheduled.

        At the removal version the only problem left is the manifest half (the
        shim is still registered in this synthetic scenario); a real reference
        under the scanned roots would add a line here.  Nothing in these roots
        mentions ``minimax--*``, so the scan proves it reached them and found
        no spurious hits.
        """
        module = _load_script()
        problems = module.check("0.16.0", ["examples", "docker"], [], removals=SAMPLES)
        assert all("is still registered" in problem for problem in problems)

    def test_a_leftover_alias_under_src_is_reported(self, tmp_path):
        module = _load_script()
        alias = tmp_path / "src" / "llm_rosetta" / "provider_names.py"
        alias.parent.mkdir(parents=True)
        alias.write_text('    "minimax--anthropic": "minimax--messages",\n')

        problems = module.check("0.16.0", [str(tmp_path / "src")], [], removals=SAMPLES)
        assert any("provider_names.py" in problem for problem in problems)


class TestReferenceHalf:
    """Stale mentions under the scanned roots."""

    def test_a_mention_is_ignored_while_the_removal_is_not_due(self, tmp_path):
        (tmp_path / "config.jsonc").write_text('{"shim": "minimax--anthropic"}\n')
        assert (
            _load_script().check("0.15.0", [str(tmp_path)], [], removals=SAMPLES) == []
        )

    def test_a_mention_fails_once_the_removal_is_due(self, tmp_path):
        (tmp_path / "config.jsonc").write_text('{"shim": "minimax--anthropic"}\n')
        problems = _load_script().check("0.16.0", [str(tmp_path)], [], removals=SAMPLES)
        assert any("config.jsonc:1" in problem for problem in problems)

    def test_an_exempt_file_is_not_reported(self, tmp_path):
        changelog = tmp_path / "changelog.md"
        changelog.write_text("`minimax--anthropic` was removed in 0.16.0\n")
        problems = _load_script().check(
            "0.16.0",
            [str(tmp_path)],
            [str(changelog)],
            removals=SAMPLES,
        )
        assert not any("changelog.md" in problem for problem in problems)


class TestManifestHalf:
    """The two ways the manifest and the registry can disagree."""

    def test_a_shim_removed_before_its_version_is_reported(self):
        module = _load_script()
        early = ShimRemoval(
            shim="not-a-real-shim--chat_completions",
            removed_in="0.16.0",
            remote={"url": "https://github.com/x/y", "path": "src/p/providers"},
            source_repo="https://github.com/x/y",
        )
        problems = module.check("0.15.0", [], [], removals=(early,))
        assert len(problems) == 1
        assert "is not registered" in problems[0]

    def test_a_shim_still_present_at_its_version_is_reported(self):
        module = _load_script()
        problems = module.check("0.16.0", [], [], removals=SAMPLES)
        still_registered = [
            problem for problem in problems if "is still registered" in problem
        ]
        assert len(still_registered) == len(SAMPLES)

    def test_the_shipped_manifest_is_empty(self):
        """Removals are deliberate: adding one is a reviewed edit to the manifest."""
        assert SHIM_REMOVALS == ()
