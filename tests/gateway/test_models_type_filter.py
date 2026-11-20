"""The ``?type=`` filter of the model listings after the spelling removal."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from llm_rosetta_gateway.app.routes_models import _type_filter


def _request(**query: str) -> SimpleNamespace:
    return SimpleNamespace(query_params=dict((k, [v]) for k, v in query.items()))


@pytest.mark.parametrize("value", ["llm", "embeddings", "rerank", "decisions"])
def test_a_current_type_is_the_filter(value: str) -> None:
    assert _type_filter(_request(type=value)) == (value, None)


@pytest.mark.parametrize(
    "request_", [SimpleNamespace(), _request(), _request(type="all")]
)
def test_no_filter(request_: SimpleNamespace) -> None:
    assert _type_filter(request_) == (None, None)


@pytest.mark.parametrize(
    ("old", "new"), [("embedding", "embeddings"), ("decision", "decisions")]
)
def test_a_retired_type_is_a_400_naming_the_replacement(old: str, new: str) -> None:
    value, response = _type_filter(_request(type=old))
    assert value is None
    assert response is not None and response.status_code == 400
    message = json.loads(response.body)["error"]
    assert f"'{old}' was renamed to '{new}'" in message
    assert "llm-rosetta-migrate" in message
