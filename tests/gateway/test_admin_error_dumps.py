"""Query parsing for the admin error-dump routes.

The panel filters and pages on the server, so these parameters decide what
an operator sees.  Each case pins a value that used to be resolved in the
browser over one 500-row page, and a malformed one that must answer 400
rather than silently widening the query.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_rosetta_gateway.admin.routes.observability import (
    MAX_DUMP_IDS,
    _dump_cursor,
    _dump_filters,
    _dump_ids,
    _dump_status_bounds,
    get_error_dumps,
)


def _req(**query: str) -> SimpleNamespace:
    return SimpleNamespace(query_params={k: [v] for k, v in query.items()})


def test_status_class_becomes_a_range() -> None:
    assert _dump_status_bounds(_req(status="4xx")) == (400, 499)
    assert _dump_status_bounds(_req(status="5xx")) == (500, 599)


def test_status_code_is_its_own_range() -> None:
    assert _dump_status_bounds(_req(status="429")) == (429, 429)


def test_a_missing_status_filters_nothing() -> None:
    assert _dump_status_bounds(_req()) == (None, None)
    assert _dump_status_bounds(_req(status="")) == (None, None)


@pytest.mark.parametrize("raw", ["abc", "99", "600", "4x", "xx"])
def test_a_malformed_status_is_refused(raw: str) -> None:
    with pytest.raises(ValueError):
        _dump_status_bounds(_req(status=raw))


def test_time_bounds_are_normalised_to_utc() -> None:
    filters = _dump_filters(_req(start="2026-11-01T00:00:00Z"))
    assert filters["start"] == "2026-11-01T00:00:00+00:00"
    assert filters["end"] is None


def test_a_time_bound_without_an_offset_is_refused() -> None:
    with pytest.raises(ValueError):
        _dump_filters(_req(start="2026-11-01T00:00:00"))


def test_each_filter_reaches_its_own_key() -> None:
    filters = _dump_filters(
        _req(
            model="gpt-5",
            model_search="haiku",
            error_phase="upstream",
            provider="argo",
            status="5xx",
            q="abc",
        )
    )
    assert filters["model"] == "gpt-5"
    assert filters["model_search"] == "haiku"
    assert filters["error_phase"] == "upstream"
    assert filters["provider"] == "argo"
    assert (filters["status_min"], filters["status_max"]) == (500, 599)
    assert filters["q"] == "abc"


def test_ids_are_split_and_trimmed() -> None:
    assert _dump_ids(_req(ids="a, b ,,c")) == ["a", "b", "c"]


def test_no_ids_means_no_id_filter() -> None:
    assert _dump_ids(_req()) is None
    assert _dump_ids(_req(ids="  ")) is None


def test_ids_are_capped_at_one_statement() -> None:
    many = ",".join(f"d{i}" for i in range(MAX_DUMP_IDS + 1))
    with pytest.raises(ValueError):
        _dump_ids(_req(ids=many))


def test_a_cursor_is_normalised_like_every_other_bound() -> None:
    # The stored timestamps are `isoformat()`; a `Z` bound would compare
    # above `+00:00` and drop the row exactly at the boundary.
    assert _dump_cursor(
        _req(before_timestamp="2026-11-01T00:00:00Z", before_id="d9")
    ) == ("2026-11-01T00:00:00+00:00", "d9")


def test_no_cursor_means_the_first_page() -> None:
    assert _dump_cursor(_req()) is None


@pytest.mark.parametrize(
    "query",
    [
        {"before_timestamp": "2026-11-01T00:00:00+00:00"},
        {"before_id": "d9"},
        {"before_timestamp": "2026-11-01T00:00:00", "before_id": "d9"},
    ],
)
def test_half_a_cursor_is_refused(query: dict[str, str]) -> None:
    # One half does not name a row, and guessing the other (the head, or the
    # newest id) would silently restart or skip the walk.
    with pytest.raises(ValueError):
        _dump_cursor(_req(**query))


@pytest.mark.asyncio
async def test_a_bad_status_answers_400_not_500() -> None:
    req = _req(status="nope")
    req.app = SimpleNamespace(persistence=object())
    response = await get_error_dumps(req)
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_the_cursor_reaches_the_read() -> None:
    # Parsing the cursor is not the same as handing it to the query: without
    # this the walk would silently page from the newest row every time.
    seen: dict[str, object] = {}

    class _Persistence:
        async def query_error_dumps(self, **kwargs: object) -> tuple[list[object], int]:
            seen.update(kwargs)
            return [], 0

    req = _req(
        before_timestamp="2026-11-01T00:00:00+00:00",
        before_id="d9",
        limit="500",
    )
    req.app = SimpleNamespace(persistence=_Persistence())
    response = await get_error_dumps(req)
    assert response.status_code == 200
    assert seen["before"] == ("2026-11-01T00:00:00+00:00", "d9")
    assert seen["limit"] == 500
