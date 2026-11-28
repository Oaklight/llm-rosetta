"""Laws of the manifest-source selection and freshness code (Hypothesis).

The existing tests (``test_shim_manifest_sources.py``, ``test_shim_releases.py``,
``test_shim_install_units.py``, ``test_shim_source_catalog.py``) pin each rule
with an example.  The checks here state the rules once and run them over
generated input, against a reference written out in this file:

* ``choose_release(prefer_stable=True)``: the newest installable *stable* release wins, a
  pre-release wins only when no stable release is installable, the skipped
  list is the incompatible releases newer than the choice, and none of it
  depends on the order the index lists them;
* ``pick_selector``: an exclusion beats everything, a pin beats an unpinned
  vendor entry beats a tag or ``*``, two pins with different versions are a
  conflict, and none of these depends on the order of the list;
* ``parse_selector``: every structured selector round-trips through its
  spelling, and the combinations the grammar forbids are refused;
* ``_keeps``: unpinned, an install is never downgraded; pinned, only the pinned
  version stays;
* ``_generated_at`` / ``_older``: the stamp is read as an instant whatever its
  offset and fraction;
* ``_load_manifest``: a manifest older than the cached one never replaces it,
  whichever URL served it, and the second URL is consulted as documented.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from llm_rosetta.api import IndexEntry, choose_release, parse_index  # noqa: E402
from llm_rosetta.contracts import IR_VERSION, SURFACE_VERSION  # noqa: E402
from llm_rosetta_gateway.shims.manifest_sources import (  # noqa: E402
    freshness,
    plan,
    selection,
)

_SETTINGS = settings(
    max_examples=150,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def test_the_targets_are_the_product_s_own() -> None:
    assert selection.pick_selector is plan.pick_selector
    assert freshness.parse_index is parse_index


# --- a SemVer reference (see tests/test_extension_index_laws.py) ------------


def _cmp_identifier(a: str, b: str) -> int:
    if a.isdigit() and b.isdigit():
        return (int(a) > int(b)) - (int(a) < int(b))
    if a.isdigit():
        return -1
    if b.isdigit():
        return 1
    return (a > b) - (a < b)


def semver_cmp(a: str, b: str) -> int:
    def split(v: str):
        core, _, pre = v.partition("-")
        return tuple(int(p) for p in core.split(".")), (pre.split(".") if pre else None)

    (ca, pa), (cb, pb) = split(a), split(b)
    if ca != cb:
        return -1 if ca < cb else 1
    if pa is None or pb is None:
        return (pa is None) - (pb is None)
    for x, y in zip(pa, pb):
        if (c := _cmp_identifier(x, y)) != 0:
            return c
    return (len(pa) > len(pb)) - (len(pa) < len(pb))


def newest(versions: list[str]) -> str:
    best = versions[0]
    for v in versions[1:]:
        if semver_cmp(v, best) > 0:
            best = v
    return best


_PRE = st.sampled_from(["alpha", "beta.1", "rc.1", "rc.2", "rc.10"])
_VERSIONS = st.builds(
    lambda a, b, c, pre: f"{a}.{b}.{c}" + (f"-{pre}" if pre else ""),
    st.integers(0, 3),
    st.integers(0, 3),
    st.integers(0, 3),
    st.one_of(st.none(), _PRE),
)
_SHA = "b" * 64


def _entry(version: str, *, usable: bool = True, ext_id: str = "acme") -> IndexEntry:
    return IndexEntry(
        id=ext_id,
        version=version,
        url=f"https://example.invalid/{version}.whl",
        sha256=_SHA,
        surface=SURFACE_VERSION if usable else SURFACE_VERSION + 1,
        ir=IR_VERSION,
    )


@st.composite
def _releases(draw, min_size: int = 0) -> list[IndexEntry]:
    versions = draw(st.lists(_VERSIONS, unique=True, min_size=min_size, max_size=7))
    return [_entry(v, usable=draw(st.booleans())) for v in versions]


def _usable(entry: IndexEntry) -> bool:
    return entry.surface == SURFACE_VERSION


# --- the stable-first reading of choose_release -------------------------------


def _select_release(items):
    return choose_release(items, prefer_stable=True)


@_SETTINGS
@given(items=_releases())
def test_select_release_prefers_the_newest_stable_then_the_newest_pre(items) -> None:
    chosen, skipped = _select_release(items)
    usable = [e for e in items if _usable(e)]
    stable = [e.version for e in usable if "-" not in e.version]
    pre = [e.version for e in usable if "-" in e.version]
    if stable:
        expected = newest(stable)
    elif pre:
        expected = newest(pre)
    else:
        assert chosen is None
        assert {e.version for e, _ in skipped} == {
            e.version for e in items if not _usable(e)
        }
        return
    assert chosen is not None and chosen.version == expected
    assert {e.version for e, _ in skipped} == {
        e.version
        for e in items
        if not _usable(e) and semver_cmp(e.version, expected) > 0
    }
    assert all(why for _, why in skipped)


@_SETTINGS
@given(items=_releases(min_size=2), data=st.data())
def test_select_release_does_not_depend_on_listing_order(items, data) -> None:
    shuffled = data.draw(st.permutations(items))
    first, skipped = _select_release(items)
    again, skipped_again = _select_release(shuffled)
    assert again == first
    assert {e.version for e, _ in skipped_again} == {e.version for e, _ in skipped}


# --- selectors ---------------------------------------------------------------

_SOURCES = ["contrib", "argonne"]
_VENDORS = ["acme", "alcf", "zeta"]
_TAGS = ["tier:free", "region", "beta"]
_PINS = [None, None, "1.0.0", "2.0.0"]


@st.composite
def _selector_text(draw) -> str:
    source = draw(st.one_of(st.none(), st.sampled_from(_SOURCES)))
    prefix = f"{source}:" if source else ""
    kind = draw(st.sampled_from(["vendor", "vendor", "tag", "all", "exclude"]))
    if kind == "all":
        return f"{prefix}*"
    if kind == "tag":
        return f"{prefix}tag:{draw(st.sampled_from(_TAGS))}"
    vendor = draw(st.sampled_from(_VENDORS))
    if kind == "exclude":
        return f"!{prefix}{vendor}"
    pin = draw(st.sampled_from(_PINS))
    return f"{prefix}{vendor}" + (f"@{pin}" if pin else "")


@st.composite
def _selector_set(draw):
    texts = draw(st.lists(_selector_text(), min_size=1, max_size=6, unique=True))
    return [plan.parse_selector(t) for t in texts]


def _applicable(selectors, source_id, vendor, tags):
    return [s for s in selectors if s.applies_to(source_id) and s.matches(vendor, tags)]


@_SETTINGS
@given(
    selectors=_selector_set(),
    source_id=st.sampled_from(_SOURCES),
    vendor=st.sampled_from(_VENDORS),
    tags=st.lists(st.sampled_from(_TAGS), unique=True, max_size=3),
)
def test_pick_selector_follows_the_documented_rule(
    selectors, source_id, vendor, tags
) -> None:
    got = plan.pick_selector(selectors, source_id, vendor, tags)
    hits = _applicable(selectors, source_id, vendor, tags)
    if any(s.negate for s in hits):
        assert got is None
        return
    if not hits:
        assert got is None
        return
    assert got is not None and got.raw in {s.raw for s in hits}
    explicit = [s for s in hits if s.kind == "vendor"]
    pins = [s for s in explicit if s.version is not None]
    if len({s.version for s in pins}) > 1:
        assert set(got.pin_conflict) == {s.raw for s in pins}
    elif pins:
        assert got.version == pins[0].version and got.kind == "vendor"
    elif explicit:
        assert got.kind == "vendor"
    assert not got.pin_conflict or len({s.version for s in pins}) > 1


@_SETTINGS
@given(
    selectors=_selector_set(),
    source_id=st.sampled_from(_SOURCES),
    vendor=st.sampled_from(_VENDORS),
    tags=st.lists(st.sampled_from(_TAGS), unique=True, max_size=3),
    data=st.data(),
)
def test_a_selector_set_decides_the_same_whatever_its_order(
    selectors, source_id, vendor, tags, data
) -> None:
    """What is installed, and at which pin, does not depend on list order."""
    shuffled = data.draw(st.permutations(selectors))
    a = plan.pick_selector(selectors, source_id, vendor, tags)
    b = plan.pick_selector(shuffled, source_id, vendor, tags)
    assert (a is None) == (b is None)
    if a is not None and b is not None:
        assert set(a.pin_conflict) == set(b.pin_conflict)
        # A conflict names no version to install (the installer refuses it), so
        # which of the conflicting pins comes back first is not a decision; and
        # among tag and ``*`` entries the first listed speaks, so only whether
        # an explicit vendor entry decided is order-independent.
        assert (a.kind == "vendor") == (b.kind == "vendor")
        assert a.negate == b.negate
        if not a.pin_conflict:
            assert a.version == b.version


@_SETTINGS
@given(selectors=_selector_set(), source_id=st.sampled_from(_SOURCES))
def test_an_exclusion_is_found_whatever_the_order_of_the_other_entries(
    selectors, source_id
) -> None:
    for vendor in _VENDORS:
        found = plan.excluded_by(selectors, source_id, vendor)
        expected = any(
            s.negate and s.applies_to(source_id) and s.value == vendor
            for s in selectors
        )
        assert (found is not None) == expected
        if expected:
            assert plan.pick_selector(selectors, source_id, vendor, []) is None


@_SETTINGS
@given(
    source=st.one_of(st.none(), st.sampled_from(_SOURCES)),
    vendor=st.sampled_from(_VENDORS),
    version=st.sampled_from(_PINS),
    negate=st.booleans(),
    pad=st.sampled_from(["", " ", "  "]),
)
def test_a_vendor_selector_round_trips_through_its_spelling(
    source, vendor, version, negate, pad
) -> None:
    assume(not (negate and version))  # the grammar forbids a pinned exclusion
    text = ("!" if negate else "") + (f"{source}:" if source else "") + vendor
    text += f"@{version}" if version else ""
    parsed = plan.parse_selector(pad + text + pad)
    assert (parsed.source, parsed.kind, parsed.value) == (source, "vendor", vendor)
    assert (parsed.negate, parsed.version) == (negate, version)
    assert parsed.raw == text


@_SETTINGS
@given(
    source=st.one_of(st.none(), st.sampled_from(_SOURCES)),
    tag=st.sampled_from(_TAGS),
)
def test_tag_and_star_selectors_round_trip(source, tag) -> None:
    prefix = f"{source}:" if source else ""
    parsed_tag = plan.parse_selector(f"{prefix}tag:{tag}")
    assert (parsed_tag.source, parsed_tag.kind, parsed_tag.value) == (
        source,
        "tag",
        tag,
    )
    parsed_all = plan.parse_selector(f"{prefix}*")
    assert (parsed_all.source, parsed_all.kind, parsed_all.value) == (source, "all", "")


@pytest.mark.parametrize(
    "text",
    [
        None,
        3,
        "",
        "  ",
        "!tag:free",
        "!*",
        "tag:free@1.0.0",
        "*@1.0.0",
        "!acme@1.0.0",
        "acme@",
        "acme@v1",
        "Acme",
        "bad source:acme",
        "tag:",
        "tag:Bad",
    ],
)
def test_the_grammar_refuses_what_it_forbids(text: Any) -> None:
    with pytest.raises(ValueError):
        plan.parse_selector(text)


# --- _keeps ------------------------------------------------------------------


@_SETTINGS
@given(installed=_VERSIONS, chosen=_VERSIONS, pin=st.one_of(st.none(), _VERSIONS))
def test_an_unpinned_install_is_never_downgraded_and_a_pin_holds_exactly(
    installed, chosen, pin
) -> None:
    selector = plan.InstallSelector(
        raw="acme", source=None, kind="vendor", value="acme", version=pin
    )
    keeps = selection._keeps(_entry(installed), _entry(chosen), selector)
    if pin is None:
        assert keeps == (semver_cmp(installed, chosen) >= 0)
    else:
        assert keeps == (installed == chosen)


# --- generated_at ------------------------------------------------------------


@st.composite
def _stamps(draw) -> tuple[str, datetime]:
    """A stamp in one of RFC 3339's spellings, with the instant it means."""
    moment = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(
        seconds=draw(st.integers(0, 3 * 365 * 86400))
    )
    offset = draw(st.sampled_from([0, 0, 60, -480, 330]))
    local = moment + timedelta(minutes=offset)
    digits = draw(st.sampled_from(["", ".5", ".123", ".123456", ".1234567"]))
    sep = draw(st.sampled_from(["T", "t", " "]))
    if offset == 0:
        zone = draw(st.sampled_from(["Z", "z", "+00:00", "-00:00"]))
    else:
        sign = "+" if offset > 0 else "-"
        zone = f"{sign}{abs(offset) // 60:02d}:{abs(offset) % 60:02d}"
    text = f"{local:%Y-%m-%d}{sep}{local:%H:%M:%S}{digits}{zone}"
    fraction = timedelta(microseconds=int((digits[1:7] or "0").ljust(6, "0")))
    return text, moment + fraction


def _manifest(stamp: str, vendors: tuple[str, ...] = ()) -> Any:
    return parse_index(
        {
            "index_version": 1,
            "source": "contrib",
            "generated_at": stamp,
            "extensions": {v: [] for v in vendors},
        }
    )


@_SETTINGS
@given(stamp=_stamps())
def test_generated_at_is_read_as_an_instant(stamp) -> None:
    text, moment = stamp
    assert freshness._generated_at(_manifest(text)) == moment


@_SETTINGS
@given(a=_stamps(), b=_stamps())
def test_older_compares_instants(a, b) -> None:
    assert freshness._older(_manifest(a[0]), _manifest(b[0])) == (a[1] < b[1])


def test_an_unknown_time_is_never_older() -> None:
    known = _manifest("2026-01-01T00:00:00Z")
    unknown = _manifest("")
    assert not freshness._older(unknown, known)
    assert not freshness._older(known, unknown)
    assert not freshness._older(unknown, unknown)


# --- the rollback guard ------------------------------------------------------

_URL = "https://example.invalid/dist/index.json"
_MIRROR = "https://mirror.invalid/dist/index.json"


class _Http:
    """Serves one stamp per URL; counts the fetches of each."""

    def __init__(self) -> None:
        self.stamps: dict[str, str] = {}
        self.sources: dict[str, str | None] = {}
        self.calls: list[str] = []

    def __call__(self, url: str, *, headers=None, timeout=None, proxy=None) -> bytes:
        self.calls.append(url)
        if url not in self.stamps:
            raise ConnectionError(url)
        document: dict[str, Any] = {
            "index_version": 1,
            "generated_at": self.stamps[url],
            "extensions": {},
        }
        source = self.sources.get(url, "contrib")
        if source is not None:
            document["source"] = source
        return json.dumps(document).encode()


def _load(root: Path, http: _Http, spec: Any, *, force: bool = True):
    return freshness._load_manifest(
        spec,
        root,
        http_get=http,
        now=lambda: 1_000.0,
        ttl=3600,
        proxy=None,
        force=force,
    )


def _spec(*, mirror: bool = False):
    return plan.ManifestSourceSpec(
        key="contrib", url=_URL, fallback_url=_MIRROR if mirror else None
    )


@_SETTINGS
@given(served=st.lists(_stamps(), min_size=1, max_size=6))
def test_the_rollback_guard_never_accepts_an_older_generated_at(served) -> None:
    """Whatever the server answers in turn, the manifest never goes back."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        http = _Http()
        best: datetime | None = None
        for text, moment in served:
            http.stamps[_URL] = text
            state = _load(root, http, _spec())
            assert state.manifest is not None
            got = freshness._generated_at(state.manifest)
            assert got is not None
            if best is not None and moment < best:
                assert got == best
                assert state.stale and state.from_cache and state.stale_reason
            else:
                assert got == moment
                assert not state.stale
                best = moment
            # What is on disk is what was returned.
            cached = json.loads((root / "index.json").read_text())["generated_at"]
            assert freshness._generated_at(_manifest(cached)) == got


@_SETTINGS
@given(cache=_stamps(), primary=_stamps(), mirror=_stamps())
def test_the_second_url_is_consulted_only_when_the_first_is_older_than_the_cache(
    cache, primary, mirror
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        http = _Http()
        http.stamps[_URL] = cache[0]
        _load(root, http, _spec(mirror=True))
        http.calls.clear()

        http.stamps[_URL] = primary[0]
        http.stamps[_MIRROR] = mirror[0]
        state = _load(root, http, _spec(mirror=True))

        assert state.manifest is not None
        got = freshness._generated_at(state.manifest)
        if primary[1] >= cache[1]:
            # The first answer is fresh enough: no second opinion, and it wins
            # even when the mirror is newer.
            assert http.calls == [_URL]
            assert got == primary[1]
            return
        # Older than the cache: the mirror is asked once, and the newer of the
        # two is taken unless that is still older than the cache.
        assert http.calls == [_URL, _MIRROR]
        chosen = mirror[1] if mirror[1] > primary[1] else primary[1]
        if chosen < cache[1]:
            assert got == cache[1] and state.stale
        else:
            assert got == chosen and not state.stale


# The ``source`` header is the id the index gives itself.  A catalogue id
# (``argonne``, ``contrib``) names one repository, and the wheels' sha256 comes
# from the same document, so an index that calls itself something else is a
# different repository's trust root: it is refused, never used.  A source given
# as a bare URL has no id to compare, and an index with no header makes no claim.


def test_a_manifest_that_names_another_source_is_refused() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        http = _Http()
        http.stamps[_URL] = "2026-01-01T00:00:00Z"
        spec = plan.ManifestSourceSpec(key="argonne", url=_URL)

        state = _load(Path(tmp), http, spec)

        assert state.manifest is None
        assert state.error is not None
        assert "'contrib'" in state.error and "'argonne'" in state.error
        assert not list(Path(tmp).rglob("index.json")), "a refused index is not cached"


def test_a_refused_primary_falls_back_to_a_mirror_that_names_the_source() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        http = _Http()
        http.stamps[_URL] = http.stamps[_MIRROR] = "2026-01-01T00:00:00Z"
        http.sources[_URL] = "someone-else"
        spec = plan.ManifestSourceSpec(key="contrib", url=_URL, fallback_url=_MIRROR)

        state = _load(Path(tmp), http, spec)

        assert state.manifest is not None and state.manifest.source == "contrib"
        assert http.calls == [_URL, _MIRROR]


def test_a_cached_manifest_that_names_another_source_is_not_used() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        http = _Http()
        http.stamps[_URL] = "2026-01-01T00:00:00Z"
        wrong = plan.ManifestSourceSpec(key="argonne", url=_URL)
        document = json.loads(http(_URL))
        path = Path(tmp) / "index.json"
        path.write_text(json.dumps(document), encoding="utf-8")

        assert freshness._read_cached_manifest(path, wrong) is None
        right = plan.ManifestSourceSpec(key="contrib", url=_URL)
        assert freshness._read_cached_manifest(path, right) is not None


@pytest.mark.parametrize("key", ["contrib", None])
def test_an_index_with_no_source_header_or_a_url_source_is_used(key) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        http = _Http()
        http.stamps[_URL] = "2026-01-01T00:00:00Z"
        if key is not None:
            http.sources[_URL] = None
        else:
            http.sources[_URL] = "anything"
        spec = plan.ManifestSourceSpec(key=key, url=_URL)

        state = _load(Path(tmp), http, spec)

        assert state.manifest is not None and state.error is None


# --- the selector's own predicates, and which entry speaks ------------------


@_SETTINGS
@given(
    selector=_selector_text().map(plan.parse_selector),
    source_id=st.one_of(st.none(), st.sampled_from(_SOURCES)),
    vendor=st.sampled_from(_VENDORS),
    tags=st.lists(st.sampled_from(_TAGS), unique=True, max_size=3),
)
def test_applies_to_and_matches_follow_their_definitions(
    selector, source_id, vendor, tags
) -> None:
    assert selector.applies_to(source_id) == (
        selector.source is None or selector.source == source_id
    )
    expected = {
        "all": True,
        "tag": selector.value in tags,
        "vendor": selector.value == vendor,
    }[selector.kind]
    assert selector.matches(vendor, tags) == expected


def test_a_parsed_selector_defaults_to_a_plain_install() -> None:
    parsed = plan.parse_selector("acme")
    assert (parsed.negate, parsed.version, parsed.origin) == (False, None, "install")
    assert parsed.pin_conflict == ()
    assert plan.parse_selector("acme", origin="moved:acme").origin == "moved:acme"


@_SETTINGS
@given(
    texts=st.lists(
        st.one_of(st.sampled_from(_TAGS).map(lambda t: f"tag:{t}"), st.just("*")),
        min_size=1,
        max_size=4,
        unique=True,
    ),
    vendor=st.sampled_from(_VENDORS),
)
def test_among_tag_and_star_entries_the_first_one_listed_speaks(texts, vendor) -> None:
    selectors = [plan.parse_selector(t) for t in texts]
    tags = [s.value for s in selectors if s.kind == "tag"]
    hits = [s for s in selectors if s.matches(vendor, tags)]
    chosen = plan.pick_selector(selectors, None, vendor, tags)
    assert chosen is not None and chosen.raw == hits[0].raw


@_SETTINGS
@given(
    selectors=_selector_set(),
    source_id=st.sampled_from(_SOURCES),
    vendor=st.sampled_from(_VENDORS),
    tags=st.lists(st.sampled_from(_TAGS), unique=True, max_size=3),
)
def test_each_entry_that_did_not_install_the_vendor_says_why(
    selectors, source_id, vendor, tags
) -> None:
    winner = plan.pick_selector(selectors, source_id, vendor, tags)
    reasons = plan._overridden_reason(selectors, source_id, vendor, tags, winner)
    hits = [s for s in _applicable(selectors, source_id, vendor, tags) if not s.negate]
    expected_keys = {s.raw for s in hits if winner is None or s.raw != winner.raw}
    assert set(reasons) == expected_keys
    exclusion = plan.excluded_by(selectors, source_id, vendor)
    for raw, why in reasons.items():
        if exclusion is not None:
            assert "excluded by shims.install" in why and exclusion.raw in why
        elif winner is not None and raw in winner.pin_conflict:
            assert why.startswith("pinned more than once")
        else:
            assert winner is not None
            assert why == f"already selected by {winner.raw}"


# --- picking the release a selector asks for ---------------------------------


def _selector(version: str | None = None, conflict: tuple[str, ...] = ()):
    return plan.InstallSelector(
        raw="acme" if version is None else f"acme@{version}",
        source=None,
        kind="vendor",
        value="acme",
        version=version,
        pin_conflict=conflict,
    )


@_SETTINGS
@given(items=_releases(), pin=_VERSIONS)
def test_a_pin_names_one_release_and_never_falls_back(items, pin) -> None:
    chosen, passed_over, problem = selection._choose_release(
        items, _selector(pin), "contrib", "acme"
    )
    listed = [e for e in items if e.version == pin]
    assert passed_over == []
    if not listed:
        assert chosen is None and problem == "the index does not list"
    elif not _usable(listed[0]):
        assert chosen is None and problem is not None
        assert problem.startswith("is not installable on this core (")
    else:
        assert chosen == listed[0] and problem is None


@_SETTINGS
@given(items=_releases())
def test_without_a_pin_the_newest_installable_release_is_chosen(items) -> None:
    chosen, passed_over, problem = selection._choose_release(
        items, _selector(), "contrib", "acme"
    )
    expected, skipped = _select_release(items)
    assert problem is None
    assert chosen == expected
    assert passed_over == list(skipped)


def test_two_pins_of_one_vendor_choose_nothing() -> None:
    chosen, passed_over, problem = selection._choose_release(
        [_entry("1.0.0")], _selector("1.0.0", ("acme@1.0.0", "acme@2.0.0")), "c", "acme"
    )
    assert (chosen, passed_over, problem) == (None, [], "pinned more than once")


@_SETTINGS
@given(items=_releases(), pin=st.one_of(st.none(), _VERSIONS), conflict=st.booleans())
def test_the_reason_nothing_installs_names_what_the_selector_asked(
    items, pin, conflict
) -> None:
    selector = _selector(pin, ("acme@1.0.0", "acme@2.0.0") if conflict else ())
    _, passed_over, problem = selection._choose_release(
        items, selector, "contrib", "acme"
    )
    text = selection._no_release_error("acme", selector, problem, passed_over)
    if conflict:
        assert text == "acme is pinned more than once: acme@1.0.0, acme@2.0.0"
    elif problem:
        assert text == f"pinned acme@{pin} {problem}"
    else:
        reason = "; ".join(why for _, why in passed_over) or "no release"
        assert text == f"no version of acme is installable on this core ({reason})"


# --- the manifest cache: a reference model ----------------------------------

_INSTANTS = [
    "2026-01-01T00:00:00Z",
    "2026-01-02T00:00:00Z",
    "2026-01-03T00:00:00Z",
]
_RECHECK = freshness.VENDOR_RECHECK_SECONDS
_TTL = 3600


class _Server:
    def __init__(self) -> None:
        self.mode = "ok"
        self.stamp = _INSTANTS[0]
        self.vendors: tuple[str, ...] = ()
        self.calls = 0

    def __call__(self, url: str, *, headers=None, timeout=None, proxy=None) -> bytes:
        self.calls += 1
        if self.mode == "down":
            raise ConnectionError(url)
        if self.mode == "garbage":
            return b"not json"
        return json.dumps(
            {
                "index_version": 1,
                "source": "contrib",
                "generated_at": self.stamp,
                "extensions": {
                    v: [
                        {
                            "version": "1.0.0",
                            "url": f"https://example.invalid/{v}.whl",
                            "sha256": _SHA,
                        }
                    ]
                    for v in self.vendors
                },
            }
        ).encode()


class _Model:
    """What ``_load_manifest`` should do, written from its docstring."""

    def __init__(self, expects: bool) -> None:
        self.expects = expects  # a selector ``contrib:acme`` is installed
        self.cache: tuple[int, tuple[str, ...]] | None = None  # (instant, vendors)
        self.fetched_at: float | None = None
        self.stale_reason = False
        self.check: tuple[tuple[str, ...], float] | None = None  # (missing, at)

    def missing(self, vendors: tuple[str, ...]) -> tuple[str, ...]:
        return ("acme",) if self.expects and "acme" not in vendors else ()

    def cooling(self, missing: tuple[str, ...], now: float) -> bool:
        if not missing or self.check is None:
            return False
        seen, at = self.check
        return set(missing) <= set(seen) and 0 <= now - at < _RECHECK

    def step(self, now: float, force: bool, mode: str, stamp: int, vendors):
        """Return ``(fetched, from_cache, stale, instant, vendors_seen)``."""
        within = self.fetched_at is not None and now - self.fetched_at <= _TTL
        if self.cache is not None and not force and within:
            gap = self.missing(self.cache[1])
            if not self.stale_reason and (not gap or self.cooling(gap, now)):
                return False, True, False, self.cache[0], self.cache[1]
        if mode != "ok":
            if self.cache is not None:
                return True, True, True, self.cache[0], self.cache[1]
            return True, False, False, None, ()
        gap = self.missing(vendors)
        cooling = not force and self.cooling(gap, now)
        if self.cache is not None and stamp < self.cache[0]:
            self.stale_reason = True
            return True, True, True, self.cache[0], self.cache[1]
        self.cache = (stamp, vendors)
        self.fetched_at = now
        self.stale_reason = False
        if gap:
            self.check = self.check if cooling and self.check else (gap, now)
        else:
            self.check = None
        return True, False, False, stamp, vendors


_STEP = st.tuples(
    st.sampled_from([0, 10, _RECHECK - 1, _RECHECK, _RECHECK + 1, _TTL, _TTL + 1]),
    st.booleans(),
    st.sampled_from(["ok", "ok", "ok", "down", "garbage"]),
    st.integers(0, len(_INSTANTS) - 1),
    st.booleans(),
)


@_SETTINGS
@given(expects=st.booleans(), steps=st.lists(_STEP, min_size=1, max_size=9))
def test_the_manifest_cache_behaves_like_its_reference(expects, steps) -> None:
    selectors = [plan.parse_selector("contrib:acme")] if expects else []
    model = _Model(expects)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        server = _Server()
        now = 1_000.0
        for advance, force, mode, stamp, has_vendor in steps:
            now += advance
            server.mode, server.stamp = mode, _INSTANTS[stamp]
            server.vendors = ("acme",) if has_vendor else ()
            before = server.calls
            state = freshness._load_manifest(
                _spec(),
                root,
                http_get=server,
                now=lambda: now,
                ttl=_TTL,
                proxy=None,
                force=force,
                selectors=selectors,
            )
            fetched, from_cache, stale, instant, seen = model.step(
                now, force, mode, stamp, server.vendors
            )
            assert (server.calls > before) == fetched
            assert state.from_cache == from_cache
            assert state.stale == stale
            if instant is None:
                assert state.manifest is None
                assert state.error and state.unreachable == (mode == "down")
                continue
            assert state.manifest is not None
            assert freshness._generated_at(state.manifest) == datetime.fromisoformat(
                _INSTANTS[instant].replace("Z", "+00:00")
            )
            assert set(state.manifest) == set(seen)
            assert bool(state.error) == (stale)
            assert bool(state.stale_reason) == (stale and mode == "ok")


@_SETTINGS
@given(
    expects=st.lists(st.sampled_from(_VENDORS), unique=True, max_size=3),
    present=st.lists(st.sampled_from(_VENDORS), unique=True, max_size=3),
    source_id=st.sampled_from(_SOURCES),
)
def test_only_an_explicit_vendor_of_this_source_is_expected(
    expects, present, source_id
) -> None:
    selectors = [plan.parse_selector(f"{source_id}:{v}") for v in expects] + [
        plan.parse_selector("tag:beta"),
        plan.parse_selector("*"),
        plan.parse_selector(f"!{source_id}:zeta"),
        plan.parse_selector("other:alcf"),
    ]
    assert freshness._expected_vendors(selectors, source_id) == set(expects)
    manifest = _manifest("2026-01-01T00:00:00Z", tuple(present))
    spec = plan.ManifestSourceSpec(key=source_id, url=_URL)
    assert freshness._missing_vendors(manifest, spec, selectors) == sorted(
        set(expects) - set(present)
    )


def test_two_different_pins_are_a_conflict_in_either_order() -> None:
    low, high = plan.parse_selector("acme@1.0.0"), plan.parse_selector("acme@2.0.0")
    for order in ([low, high], [high, low]):
        got = plan.pick_selector(order, None, "acme", [])
        assert got is not None
        assert set(got.pin_conflict) == {"acme@1.0.0", "acme@2.0.0"}
