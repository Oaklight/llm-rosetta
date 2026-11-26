"""Symbolic checks of two small pure functions (CrossHair).

CrossHair explores inputs with an SMT solver and reports a counterexample
when a postcondition can fail.  A run that finds none inside its budget is
*not* a proof: ``CANNOT_CONFIRM`` is accepted, a failing or raising case is
not.  The properties:

* ``version_key`` orders ``MAJOR.MINOR.PATCH`` like the integer triple, ranks a
  pre-release below its release, and compares numeric pre-release identifiers
  as numbers.

``_member_parts`` (the wheel member name rule of the extension installer) is
checked by bounded exhaustive enumeration instead: CrossHair's search missed a
mutant that drops the ``..`` rule, so every name up to four characters over an
alphabet of the separators and special characters is tried.  The rule: an
accepted name never yields an empty part, a ``..`` part, a part holding a path
separator or a NUL, and its parts joined onto a POSIX or a Windows stage stay
inside it.
"""

from __future__ import annotations

import itertools
import ntpath
import posixpath
from pathlib import PurePosixPath, PureWindowsPath

import pytest

pytest.importorskip("crosshair")

from crosshair.core_and_libs import MessageType, analyze_function  # noqa: E402
from crosshair.options import AnalysisOptionSet  # noqa: E402

from llm_rosetta.api import version_key  # noqa: E402
from llm_rosetta.extensions._install import (  # noqa: E402
    ExtensionInstallError,
    _member_parts,
)

#: Outcomes that mean a counterexample, or a property CrossHair could not run.
_FAILURES = {
    MessageType.POST_FAIL,
    MessageType.POST_ERR,
    MessageType.EXEC_ERR,
    MessageType.SYNTAX_ERR,
    MessageType.IMPORT_ERR,
    MessageType.PRE_UNSAT,
}


def _no_counterexample(fn, seconds: float = 10.0) -> None:
    options = AnalysisOptionSet(per_condition_timeout=seconds)
    messages = [m for c in analyze_function(fn, options) for m in c.analyze()]
    failed = [m for m in messages if m.state in _FAILURES]
    assert not failed, "\n".join(f"{m.state.name}: {m.message}" for m in failed)


def release_order(a: int, b: int, c: int, d: int, e: int, f: int) -> bool:
    """
    pre: 0 <= a <= 3 and 0 <= b <= 3 and 0 <= c <= 3
    pre: 0 <= d <= 3 and 0 <= e <= 3 and 0 <= f <= 3
    post: _
    """
    below = version_key(f"{a}.{b}.{c}") < version_key(f"{d}.{e}.{f}")
    return below == ((a, b, c) < (d, e, f))


def prerelease_below_release(a: int, b: int, c: int) -> bool:
    """
    pre: 0 <= a <= 9 and 0 <= b <= 9 and 0 <= c <= 9
    post: _
    """
    return version_key(f"{a}.{b}.{c}-rc") < version_key(f"{a}.{b}.{c}")


def numeric_identifiers_compare_as_numbers(x: int, y: int) -> bool:
    """
    pre: 0 <= x <= 30 and 0 <= y <= 30
    post: _
    """
    below = version_key(f"1.0.0-rc.{x}") < version_key(f"1.0.0-rc.{y}")
    return below == (x < y)


@pytest.mark.parametrize(
    "fn",
    [
        release_order,
        prerelease_below_release,
        numeric_identifiers_compare_as_numbers,
    ],
    ids=lambda fn: fn.__name__,
)
def test_crosshair_finds_no_counterexample(fn) -> None:
    _no_counterexample(fn)


#: Characters that matter to the rule: a name, a dot, both separators, a drive
#: colon, a drive letter and NUL.
_ALPHABET = ("a", "C", ".", "/", "\\", ":", "\x00")


def _names(max_length: int = 4):
    for size in range(max_length + 1):
        for chars in itertools.product(_ALPHABET, repeat=size):
            yield "".join(chars)


def test_an_accepted_member_name_stays_inside_the_stage() -> None:
    stage = PurePosixPath("/stage")
    accepted = 0
    for name in _names():
        try:
            parts = _member_parts(name)
        except ExtensionInstallError:
            continue
        accepted += 1
        assert parts, repr(name)
        for part in parts:
            assert part not in ("", ".."), repr(name)
            assert not any(ch in part for ch in "/\\\x00"), repr(name)
        joined = posixpath.normpath(str(stage.joinpath(*parts)))
        assert joined.startswith("/stage/"), repr(name)
        win = ntpath.normpath(str(PureWindowsPath("C:/stage").joinpath(*parts)))
        assert win.lower().startswith("c:\\stage\\"), repr(name)
    assert accepted, "the alphabet accepts no name: the check is vacuous"
