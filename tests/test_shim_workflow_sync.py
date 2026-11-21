"""The five shim packages must carry the workflow this repository masters."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: The script under test, loaded by path for the reason given in
#: ``test_shim_catalog_sync.py``: a ``sys.path`` insert is invisible to the
#: type checker, which then cannot resolve the import at all.
SCRIPT = ROOT / "scripts" / "dev" / "check_shim_workflow.py"

spec = importlib.util.spec_from_file_location("check_shim_workflow", SCRIPT)
assert spec is not None and spec.loader is not None, SCRIPT
check = importlib.util.module_from_spec(spec)
# Registered before exec: a dataclass resolves its own module through
# ``sys.modules``, and an unregistered module fails there.
sys.modules[spec.name] = check
spec.loader.exec_module(check)

sys.path.insert(0, str(ROOT / "packages/llm-rosetta-gateway/src"))
sys.path.insert(0, str(ROOT / "packages/llm-rosetta/src"))

from llm_rosetta_gateway.shims.catalog import SHIM_SOURCE_PACKAGES  # noqa: E402

MASTER = check.MASTER.read_bytes()


def _copy(key: str, content: bytes | None) -> object:
    return check.Copy(
        key=key,
        repo=f"https://github.com/Oaklight/llm-rosetta-shim-{key}",
        content=content,
    )


def test_the_master_copy_is_a_workflow_and_still_does_the_two_things():
    """The master copy is the file the packages carry, not a second draft.

    Read as text rather than through a YAML parser: this repository does not
    depend on one, and the two lines that matter are exact strings.
    """
    text = MASTER.decode("utf-8")

    assert "\nname: Publish branch archive\n" in text
    # The gates the drift guard exists to keep from disappearing.
    assert "sha256sum -c" in text
    assert 'cp "$archive" "${archive}.sha256" "${sha}/"' in text
    assert "on:\n  push:\n    branches: [main]" in text


def test_a_matching_copy_is_ok():
    agreeing, missing, differing = check.compare_copies(
        MASTER, [_copy("argo", MASTER), _copy("alcf", MASTER)]
    )

    assert [c.key for c in agreeing] == ["argo", "alcf"]
    assert missing == [] and differing == []


def test_a_repository_without_a_copy_is_missing():
    _, missing, differing = check.compare_copies(MASTER, [_copy("argo", None)])

    assert [c.key for c in missing] == ["argo"]
    assert differing == []


def test_a_copy_that_dropped_the_digest_gate_differs():
    """The drift this exists for: a copy that silently stops publishing .sha256."""
    edited = MASTER.replace(
        b'cp "$archive" "${archive}.sha256" "${sha}/"', b'cp "$archive" "${sha}/"'
    )
    assert edited != MASTER

    _, _, differing = check.compare_copies(MASTER, [_copy("free", edited)])

    assert [c.key for c in differing] == ["free"]


def test_the_difference_names_the_line(capsys):
    edited = MASTER.replace(b"cancel-in-progress: false", b"cancel-in-progress: true")
    agreeing, missing, differing = check.compare_copies(MASTER, [_copy("argo", edited)])

    code = check.report(MASTER, agreeing, missing, differing, require_all=True)

    out = capsys.readouterr().out
    assert code == 1
    assert "cancel-in-progress: false" in out
    assert "cancel-in-progress: true" in out


def test_a_missing_copy_warns_until_require_all(capsys):
    agreeing, missing, differing = check.compare_copies(MASTER, [_copy("argo", None)])

    assert check.report(MASTER, agreeing, missing, differing, require_all=False) == 0
    assert "warning argo" in capsys.readouterr().out

    assert check.report(MASTER, agreeing, missing, differing, require_all=True) == 1


def test_an_offline_run_reads_saved_copies(tmp_path, capsys):
    """The offline mode must rehearse the same verdict as the network run."""
    (tmp_path / "asksage.yml").write_bytes(MASTER)
    (tmp_path / "alcf.yml").write_bytes(
        MASTER.replace(
            b"permissions:\n  contents: write", b"permissions:\n  contents: read"
        )
    )

    code = check.main(["--from-dir", str(tmp_path), "--require-all"])

    out = capsys.readouterr().out
    assert code == 1
    assert "ok      asksage" in out
    assert "problem alcf" in out
    assert "problem opendesign" in out  # no saved file for it
    # Packages of a manifest source are named and skipped (#1660).
    assert "skip    argo" in out
    assert "skip    free" in out


def test_manifest_source_packages_are_skipped():
    """argonne and contrib release a manifest; their packages carry no copy."""
    from llm_rosetta_gateway.shims.catalog import SHIM_SOURCES

    checked, skipped = check.split_packages(
        SHIM_SOURCE_PACKAGES, [source.repo for source in SHIM_SOURCES]
    )

    assert [p.key for p in skipped] == ["argo", "free"]
    assert [p.key for p in checked] == ["alcf", "asksage", "opendesign"]


def test_every_catalogue_package_is_checked(tmp_path):
    """A package added to the catalogue is checked without touching this script."""
    saved = tmp_path
    for package in SHIM_SOURCE_PACKAGES:
        (saved / f"{package.key}.yml").write_bytes(MASTER)

    assert check.main(["--from-dir", str(saved), "--require-all"]) == 0
