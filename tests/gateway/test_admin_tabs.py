"""Every rail tab is a tab the server knows about.

`_VALID_TABS` is what the gateway validates a downstream `disabled_tabs`
list against: an id missing from it is dropped with a warning and the tab
stays visible.  The shims page was added to the rail without being added
there, so a deployment could not hide a page it was serving.  This walks the
markup in the direction the server validates, which the JS-side check
(`tests/js/tabs-panels.test.mjs`, added with the jobs page) does not.
"""

from __future__ import annotations

import re
from pathlib import Path

from llm_rosetta_gateway.admin import _VALID_TABS

ADMIN_HTML = (
    Path(__file__).resolve().parents[2]
    / "packages"
    / "llm-rosetta-gateway"
    / "src"
    / "llm_rosetta_gateway"
    / "admin"
    / "admin.html"
)

#: The markup spells the observability tab with its display name; the server
#: knows it as ``logs``, the id the routes and the config use.
RAIL_ALIASES = {"observability": "logs"}


def _rail_tabs() -> set[str]:
    text = ADMIN_HTML.read_text(encoding="utf-8")
    return set(re.findall(r'id="tabBtn-([a-z]+)"', text))


def test_every_rail_tab_can_be_disabled_from_the_server() -> None:
    missing = {RAIL_ALIASES.get(t, t) for t in _rail_tabs()} - set(_VALID_TABS)
    assert not missing, f"rail tabs missing from _VALID_TABS: {sorted(missing)}"


def test_the_walk_finds_the_tabs() -> None:
    """Guard the guard: a regex that matches nothing proves nothing."""
    tabs = _rail_tabs()
    assert len(tabs) >= 6, tabs
    assert "shims" in tabs
