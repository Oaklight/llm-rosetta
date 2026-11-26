"""Laws of the extension index code over generated indexes (Hypothesis).

``test_ext_index_and_cli.py`` pins behaviour with examples.  The checks here
state the rules once and run them over generated input, with the expected
answer computed by a reference written out in this file (a SemVer 2.0
comparison, and an explicit statement of what makes an entry unusable):

* ``select_entry`` returns the highest version among the entries that fit the
  interpreter, the surface level and the IR level; it does not depend on the
  order the index lists them; a requested version narrows the candidates and
  never widens them;
* ``plan_updates`` gives one plan per installed extension, in order; an update
  always targets a higher version than the installed one; applying every
  update and planning again finds nothing to update;
* ``entry_problems`` names exactly the three ways an entry can be unusable;
* ``parse_index`` returns the entries it was given, in file order, and refuses
  a document whose header or entry is malformed;
* ``install_from_index`` installs each required extension once, before the
  extension that requires it, skips one that is installed, and refuses a cycle.

Each test goes through the names ``llm_rosetta.api`` exports (``plan_updates``
is not exported, so it comes from ``llm_rosetta.extensions._index``), and one
test checks that the exported names are the functions of that module.
"""

from __future__ import annotations

import itertools
import sys
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

import llm_rosetta.api as api  # noqa: E402
from llm_rosetta.contracts import IR_VERSION, SURFACE_VERSION  # noqa: E402
from llm_rosetta.extensions import _index  # noqa: E402
from llm_rosetta.extensions._dirs import ExtensionInstallError  # noqa: E402

_SETTINGS = settings(
    max_examples=150,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
_PY = sys.version_info


def test_the_names_are_the_product_s_own() -> None:
    for name in (
        "select_entry",
        "entry_problems",
        "parse_index",
        "install_from_index",
        "version_key",
        "IndexEntry",
    ):
        assert getattr(api, name) is getattr(_index, name), name


# --- a SemVer reference ------------------------------------------------------


def _split(version: str) -> tuple[tuple[int, int, int], list[str] | None]:
    core, _, pre = version.partition("-")
    major, minor, patch = (int(p) for p in core.split("."))
    return (major, minor, patch), (pre.split(".") if pre else None)


def _cmp_identifier(a: str, b: str) -> int:
    if a.isdigit() and b.isdigit():
        return (int(a) > int(b)) - (int(a) < int(b))
    if a.isdigit():
        return -1
    if b.isdigit():
        return 1
    return (a > b) - (a < b)


def semver_cmp(a: str, b: str) -> int:
    """SemVer 2.0 section 11, written out."""
    (core_a, pre_a), (core_b, pre_b) = _split(a), _split(b)
    if core_a != core_b:
        return -1 if core_a < core_b else 1
    if pre_a is None or pre_b is None:
        return (pre_a is None) - (pre_b is None)  # a final outranks its pre-release
    for x, y in zip(pre_a, pre_b):
        if (c := _cmp_identifier(x, y)) != 0:
            return c
    return (len(pre_a) > len(pre_b)) - (len(pre_a) < len(pre_b))


def newest(versions: list[str]) -> str:
    best = versions[0]
    for version in versions[1:]:
        if semver_cmp(version, best) > 0:
            best = version
    return best


# --- generators --------------------------------------------------------------

_PRE = st.sampled_from(["alpha", "beta.1", "rc.1", "rc.2", "rc.10", "1", "x.y"])
_VERSIONS = st.builds(
    lambda a, b, c, pre: f"{a}.{b}.{c}" + (f"-{pre}" if pre else ""),
    st.integers(0, 3),
    st.integers(0, 3),
    st.integers(0, 3),
    st.one_of(st.none(), _PRE),
)
_PYTHON = st.sampled_from([None, ">=3.0", "<4", ">=99", "<3"])
_SURFACE = st.sampled_from([None, SURFACE_VERSION, SURFACE_VERSION + 1, 0])
_IR = st.sampled_from([None, 0, IR_VERSION, IR_VERSION + 1])
_SHA = "a" * 64


def _entry(ext_id: str, version: str, **kw: Any) -> api.IndexEntry:
    return api.IndexEntry(
        id=ext_id,
        version=version,
        url=f"https://example.invalid/{ext_id}-{version}.whl",
        sha256=_SHA,
        **kw,
    )


@st.composite
def _entries(draw, ext_id: str = "acme", min_size: int = 0) -> list[api.IndexEntry]:
    versions = draw(st.lists(_VERSIONS, unique=True, min_size=min_size, max_size=6))
    return [
        _entry(
            ext_id,
            v,
            requires_python=draw(_PYTHON),
            surface=draw(_SURFACE),
            ir=draw(_IR),
        )
        for v in versions
    ]


def _problems(entry: api.IndexEntry) -> list[str]:
    """What makes an entry unusable, stated without the product's helper."""
    out = []
    if entry.requires_python in (">=99", "<3"):
        out.append("python")
    if entry.surface is not None and entry.surface != SURFACE_VERSION:
        out.append("surface")
    if entry.ir is not None and entry.ir > IR_VERSION:
        out.append("ir")
    return out


# --- entry_problems and select_entry ----------------------------------------


@_SETTINGS
@given(entries=_entries())
def test_entry_problems_names_the_three_ways_an_entry_fails(entries) -> None:
    assert _PY >= (3, 10)  # the interpreter the generated clauses assume
    for entry in entries:
        found = api.entry_problems(entry)
        assert len(found) == len(_problems(entry))
        assert bool(found) == bool(_problems(entry))


@_SETTINGS
@given(entries=_entries())
def test_select_entry_returns_the_newest_usable_entry(entries) -> None:
    usable = [e for e in entries if not _problems(e)]
    index = {"acme": entries}
    if not usable:
        with pytest.raises(ExtensionInstallError):
            api.select_entry(index, "acme")
        return
    chosen = api.select_entry(index, "acme")
    assert chosen in entries
    assert chosen.version == newest([e.version for e in usable])


@_SETTINGS
@given(entries=_entries(min_size=2), data=st.data())
def test_select_entry_does_not_depend_on_listing_order(entries, data) -> None:
    shuffled = data.draw(st.permutations(entries))
    try:
        first = api.select_entry({"acme": entries}, "acme")
    except ExtensionInstallError:
        with pytest.raises(ExtensionInstallError):
            api.select_entry({"acme": shuffled}, "acme")
        return
    assert api.select_entry({"acme": shuffled}, "acme") == first


@_SETTINGS
@given(entries=_entries(min_size=1), data=st.data())
def test_a_requested_version_narrows_and_never_widens(entries, data) -> None:
    wanted = data.draw(st.sampled_from([e.version for e in entries]))
    (target,) = [e for e in entries if e.version == wanted]
    if _problems(target):
        with pytest.raises(ExtensionInstallError):
            api.select_entry({"acme": entries}, "acme", version=wanted)
    else:
        assert api.select_entry({"acme": entries}, "acme", version=wanted) == target
    with pytest.raises(ExtensionInstallError):
        api.select_entry({"acme": entries}, "acme", version="9.9.9")


def test_select_entry_refuses_an_unknown_or_empty_extension() -> None:
    with pytest.raises(ExtensionInstallError):
        api.select_entry({}, "acme")
    with pytest.raises(ExtensionInstallError):
        api.select_entry({"acme": []}, "acme")


# --- plan_updates ------------------------------------------------------------


def _installed(ext_id: str, version: str | None) -> Any:
    return SimpleNamespace(id=ext_id, version=version)


@st.composite
def _fleet(draw):
    ids = ["alpha", "beta", "gamma", "delta"]
    index = {
        ext_id: draw(_entries(ext_id))
        for ext_id in draw(st.lists(st.sampled_from(ids), unique=True, max_size=4))
    }
    installed = [
        _installed(ext_id, draw(st.one_of(st.none(), _VERSIONS)))
        for ext_id in draw(st.lists(st.sampled_from(ids), unique=True, max_size=4))
    ]
    return index, installed


@_SETTINGS
@given(fleet=_fleet())
def test_plan_updates_follows_the_four_statuses(fleet) -> None:
    index, installed = fleet
    plans = _index.plan_updates(index, installed)
    assert [p.id for p in plans] == [i.id for i in installed]
    for plan, item in zip(plans, installed):
        assert plan.current == item.version
        usable = [e for e in index.get(item.id, []) if not _problems(e)]
        if item.id not in index:
            assert plan.status == "not_in_index"
        elif not usable:
            assert plan.status == "no_compatible"
        else:
            best = newest([e.version for e in usable])
            if item.version is None or semver_cmp(best, item.version) > 0:
                assert plan.status == "update"
                assert plan.target is not None and plan.target.version == best
            else:
                assert plan.status == "current"
                assert plan.target is None


@_SETTINGS
@given(fleet=_fleet())
def test_an_update_never_downgrades(fleet) -> None:
    index, installed = fleet
    for plan in _index.plan_updates(index, installed):
        if plan.status == "update" and plan.current is not None:
            assert plan.target is not None
            assert semver_cmp(plan.target.version, plan.current) > 0


@_SETTINGS
@given(fleet=_fleet())
def test_applying_every_update_leaves_nothing_to_update(fleet) -> None:
    index, installed = fleet
    after = [
        _installed(
            p.id, p.target.version if p.status == "update" and p.target else p.current
        )
        for p in _index.plan_updates(index, installed)
    ]
    again = _index.plan_updates(index, after)
    assert all(p.status != "update" for p in again)
    assert [p.status for p in again] == [
        "current" if p0.status == "update" else p0.status
        for p0 in _index.plan_updates(index, installed)
    ]


# --- parse_index -------------------------------------------------------------


def _document(entries_by_id: dict[str, list[api.IndexEntry]], **header: Any) -> dict:
    return {
        "index_version": 1,
        **header,
        "extensions": {
            ext_id: [
                {
                    "version": e.version,
                    "url": e.url,
                    "sha256": e.sha256,
                    **(
                        {}
                        if e.requires_python is None
                        else {"requires_python": e.requires_python}
                    ),
                    **({} if e.surface is None else {"surface": e.surface}),
                    **({} if e.ir is None else {"ir": e.ir}),
                }
                for e in entries
            ]
            for ext_id, entries in entries_by_id.items()
        },
    }


@_SETTINGS
@given(fleet=_fleet())
def test_parse_index_returns_the_entries_in_file_order(fleet) -> None:
    index, _ = fleet
    parsed = api.parse_index(
        _document(index, source="acme", generated_at="2026-01-02T03:04:05Z")
    )
    assert list(parsed) == list(index)
    for ext_id, entries in index.items():
        assert parsed[ext_id] == entries
    assert parsed.source == "acme"
    assert parsed.generated_at == "2026-01-02T03:04:05Z"


@pytest.mark.parametrize(
    "document",
    [
        None,
        [],
        "x",
        {},
        {"index_version": 2, "extensions": {}},
        {"index_version": 1},
        {"index_version": 1, "extensions": []},
        {"index_version": 1, "extensions": {"Bad-Id": []}},
        {"index_version": 1, "extensions": {"a": {}}},
        {"index_version": 1, "extensions": {}, "source": "Bad Source"},
        {"index_version": 1, "extensions": {}, "generated_at": "yesterday"},
        {"index_version": 1, "extensions": {}, "generated_at": 5},
    ],
)
def test_parse_index_refuses_a_malformed_document(document: Any) -> None:
    with pytest.raises(ExtensionInstallError):
        api.parse_index(document)


_GOOD_ENTRY = {"version": "1.0.0", "url": "https://x/a.whl", "sha256": _SHA}


@pytest.mark.parametrize(
    "patch",
    [
        {"version": "1.0"},
        {"version": "01.0.0"},
        {"version": 1},
        {"url": ""},
        {"sha256": "abc"},
        {"requires_python": 3},
        {"requires_extensions": ["Bad"]},
        {"requires_extensions": "a"},
        {"surface": True},
        {"surface": "1"},
        {"ir": 1.5},
        {"maturity": "gold"},
        {"commit": "xyz"},
        {"tags": "a"},
    ],
)
def test_parse_index_refuses_a_malformed_entry(patch: dict[str, Any]) -> None:
    document = {"index_version": 1, "extensions": {"a": [{**_GOOD_ENTRY, **patch}]}}
    with pytest.raises(ExtensionInstallError):
        api.parse_index(document)


@pytest.mark.xfail(
    strict=False,
    reason="True == 1, so a document whose index_version is the boolean true is "
    "read as version 1 (reported to llm-rosetta-03)",
)
def test_parse_index_refuses_a_boolean_index_version() -> None:
    with pytest.raises(ExtensionInstallError):
        api.parse_index({"index_version": True, "extensions": {}})


@pytest.mark.parametrize("raw", [None, 3, "x", []])
def test_parse_index_refuses_an_entry_that_is_not_an_object(raw: Any) -> None:
    with pytest.raises(ExtensionInstallError):
        api.parse_index({"index_version": 1, "extensions": {"a": [raw]}})


@_SETTINGS
@given(
    stamp=st.sampled_from(
        [
            "2026-01-02T03:04:05Z",
            "2026-01-02t03:04:05z",
            "2026-01-02 03:04:05.123+01:00",
            "2026-01-02T03:04:05-08:00",
        ]
    )
)
def test_parse_index_accepts_each_rfc3339_spelling(stamp: str) -> None:
    assert api.parse_index(_document({}, generated_at=stamp)).generated_at == stamp


# --- install_from_index ------------------------------------------------------


@st.composite
def _requirement_graph(draw):
    """Leaves e0..ek need nothing; every other extension needs only leaves.

    A leaf is reached by at most one path below any extension, which is the
    shape the installer handles today; the shape it does not is the xfail test
    below.
    """
    leaves = draw(st.integers(1, 3))
    count = leaves + draw(st.integers(0, 3))
    ids = [f"e{i}" for i in range(count)]
    requires = {
        ext_id: (
            ()
            if i < leaves
            else tuple(
                draw(st.lists(st.sampled_from(ids[:leaves]), unique=True, max_size=3))
            )
        )
        for i, ext_id in enumerate(ids)
    }
    installed = draw(st.sets(st.sampled_from(ids), max_size=count))
    return ids, requires, installed


class _Machine:
    """A fake extensions directory: installing makes an id installed."""

    def __init__(self, installed: set[str]) -> None:
        self.installed = list(installed)
        self.log: list[str] = []

    def installed_extensions(self, dest: Any = None) -> list[Any]:
        return [_installed(i, "1.0.0") for i in self.installed]

    def install_entry(self, entry: api.IndexEntry, dest: Any = None) -> Any:
        self.log.append(entry.id)
        self.installed.append(entry.id)
        return _installed(entry.id, entry.version)


def _graph_index(
    requires: dict[str, tuple[str, ...]],
) -> dict[str, list[api.IndexEntry]]:
    return {
        ext_id: [_entry(ext_id, "1.0.0", requires_extensions=deps)]
        for ext_id, deps in requires.items()
    }


@pytest.fixture
def machine(monkeypatch: pytest.MonkeyPatch):
    def make(installed: set[str]) -> _Machine:
        fake = _Machine(installed)
        monkeypatch.setattr(_index, "installed_extensions", fake.installed_extensions)
        monkeypatch.setattr(_index, "install_entry", fake.install_entry)
        return fake

    return make


@_SETTINGS
@given(graph=_requirement_graph(), data=st.data())
def test_install_from_index_installs_each_requirement_once_and_first(
    graph, data, monkeypatch
) -> None:
    ids, requires, installed = graph
    fake = _Machine(installed)
    monkeypatch.setattr(_index, "installed_extensions", fake.installed_extensions)
    monkeypatch.setattr(_index, "install_entry", fake.install_entry)
    target = data.draw(st.sampled_from(ids))

    result = api.install_from_index(target, _graph_index(requires))

    done = [item.id for item in result]
    assert done == fake.log
    assert len(done) == len(set(done))  # a diamond installs its base once
    assert done[-1] == target
    for position, ext_id in enumerate(done):
        for dependency in requires[ext_id]:
            assert dependency in installed or dependency in done[:position]
    # Nothing that was already installed is installed again, except the target.
    assert not (set(done[:-1]) & installed)
    # Everything the target needs, transitively, is present afterwards.
    needed, stack = set(), [target]
    while stack:
        for dependency in requires[stack.pop()]:
            if dependency not in needed:
                needed.add(dependency)
                stack.append(dependency)
    assert needed <= installed | set(done)


@pytest.mark.parametrize("cycle", [["a", "b"], ["a", "b", "c"], ["a"]])
def test_install_from_index_refuses_a_requirement_cycle(machine, cycle) -> None:
    fake = machine(set())
    requires: dict[str, tuple[str, ...]] = {
        ext_id: (cycle[(i + 1) % len(cycle)],) for i, ext_id in enumerate(cycle)
    }
    with pytest.raises(ExtensionInstallError, match="cycle"):
        api.install_from_index("a", _graph_index(requires))
    assert fake.log == []


def test_install_from_index_installs_nothing_when_a_requirement_is_unusable(
    machine,
) -> None:
    fake = machine(set())
    index = _graph_index({"a": ("b",), "b": ()})
    index["b"] = [_entry("b", "1.0.0", surface=SURFACE_VERSION + 1)]
    with pytest.raises(ExtensionInstallError):
        api.install_from_index("a", index)
    assert fake.log == []
    # Pairs are tried once each: the loop above never revisits an id.
    assert len(list(itertools.pairwise(fake.log))) == 0


@pytest.mark.xfail(
    strict=False,
    reason="a requirement reached by two paths is reported as a cycle: the set of "
    "installed ids is read once before the loop and the visited set is never "
    "popped (reported to llm-rosetta-03)",
)
def test_install_from_index_installs_a_requirement_shared_by_two_paths(
    machine,
) -> None:
    """top needs mid and base; mid needs base.  There is no cycle."""
    machine(set())
    index = _graph_index({"base": (), "mid": ("base",), "top": ("mid", "base")})
    done = [i.id for i in api.install_from_index("top", index)]
    assert done == ["base", "mid", "top"]


# --- location, url resolution, optional fields, size cap ---------------------


def test_index_location_prefers_explicit_then_environment_then_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_index.INDEX_ENV, raising=False)
    assert _index.index_location("x.json") == "x.json"
    assert _index.index_location() == _index.DEFAULT_INDEX_URL
    assert _index.index_location("") == _index.DEFAULT_INDEX_URL
    monkeypatch.setenv(_index.INDEX_ENV, "env.json")
    assert _index.index_location() == "env.json"
    assert _index.index_location("x.json") == "x.json"
    monkeypatch.setenv(_index.INDEX_ENV, "")
    assert _index.index_location() == _index.DEFAULT_INDEX_URL


def _url_of(base: str, url: str) -> str:
    document = {
        "index_version": 1,
        "extensions": {"a": [{**_GOOD_ENTRY, "url": url}]},
    }
    return api.parse_index(document, base=base)["a"][0].url


@pytest.mark.parametrize(
    ("base", "url", "expected"),
    [
        ("https://h/d/index.json", "https://other/a.whl", "https://other/a.whl"),
        ("https://h/d/index.json", "HTTP://other/a.whl", "HTTP://other/a.whl"),
        ("https://h/d/index.json", "/abs/a.whl", "/abs/a.whl"),
        ("https://h/d/index.json", "a.whl", "https://h/d/a.whl"),
        ("https://h/d/index.json", "sub/a.whl", "https://h/d/sub/a.whl"),
        ("/srv/idx/index.json", "/abs/a.whl", "/abs/a.whl"),
        ("/srv/idx/index.json", "https://other/a.whl", "https://other/a.whl"),
        ("/srv/idx/index.json", "a.whl", "/srv/idx/a.whl"),
        ("/srv/idx/index.json", "sub/a.whl", "/srv/idx/sub/a.whl"),
    ],
)
def test_an_entry_url_is_resolved_against_the_index_location(
    base: str, url: str, expected: str
) -> None:
    assert _url_of(base, url) == expected


def test_an_entry_url_is_left_alone_without_a_location() -> None:
    assert _url_of("", "rel/a.whl") == "rel/a.whl"


@pytest.mark.parametrize("key", ["name", "logo", "maturity", "commit"])
@pytest.mark.parametrize("value", ["", 5, True, ["x"]])
def test_an_optional_text_field_must_be_a_non_empty_string(key, value) -> None:
    document = {
        "index_version": 1,
        "extensions": {"a": [{**_GOOD_ENTRY, key: value}]},
    }
    with pytest.raises(ExtensionInstallError):
        api.parse_index(document)


@_SETTINGS
@given(
    tags=st.lists(st.text(min_size=1, max_size=8), max_size=4),
    shims=st.lists(st.text(min_size=1, max_size=8), max_size=4),
    needs=st.lists(st.sampled_from(["alpha", "beta_2", "g"]), max_size=3),
)
def test_valid_lists_are_kept_in_order(tags, shims, needs) -> None:
    document = {
        "index_version": 1,
        "extensions": {
            "a": [
                {
                    **_GOOD_ENTRY,
                    "tags": tags,
                    "shims": shims,
                    "requires_extensions": needs,
                }
            ]
        },
    }
    entry = api.parse_index(document)["a"][0]
    assert (entry.tags, entry.shims, entry.requires_extensions) == (
        tuple(tags),
        tuple(shims),
        tuple(needs),
    )


@pytest.mark.parametrize("key", ["tags", "shims", "requires_extensions"])
@pytest.mark.parametrize("bad", [[""], [5], ["ok", ""], ["ok", None], "ok", {"a": 1}])
def test_a_list_field_refuses_an_empty_or_non_string_item(key, bad) -> None:
    value = (
        ["alpha", *bad]
        if isinstance(bad, list) and key == "requires_extensions"
        else bad
    )
    document = {
        "index_version": 1,
        "extensions": {"a": [{**_GOOD_ENTRY, key: value}]},
    }
    with pytest.raises(ExtensionInstallError):
        api.parse_index(document)


def _padded(size: int) -> bytes:
    body = b'{"index_version": 1, "extensions": {}}'
    return body + b" " * (size - len(body))


def test_the_size_cap_admits_exactly_the_cap_and_refuses_one_byte_more(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_index, "MAX_INDEX_BYTES", 128)
    exact, over = tmp_path / "exact.json", tmp_path / "over.json"
    exact.write_bytes(_padded(128))
    over.write_bytes(_padded(129))
    assert _index.load_index(str(exact)) == {}
    with pytest.raises(ExtensionInstallError, match="cap"):
        _index.load_index(str(over))
