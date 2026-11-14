"""Structural checks on the admin shell.

A missing ``</div>`` in ``admin.html`` does not fail anything that reads the
panel: class names still resolve, text still matches, and every browser shows
the later panels nested inside the broken one — which is how the Settings page
went blank while a test that only read its text stayed green.  These cases
parse the file the way the browser does and fail on the two shapes that
produce that: unbalanced markup, and a panel inside another panel.
"""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

ADMIN_HTML = (
    Path(__file__).resolve().parents[2]
    / "packages"
    / "llm-rosetta-gateway"
    / "src"
    / "llm_rosetta_gateway"
    / "admin"
    / "admin.html"
)

#: Elements that never take a closing tag.
_VOID = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "source",
    "track",
    "wbr",
    "path",
    "circle",
    "rect",
}


class _Shell(HTMLParser):
    """Track element nesting and which tab panel, if any, holds each panel."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, str | None]] = []
        self.stray_closes: list[int] = []
        #: panel id -> the panel it is nested inside (``None`` at top level).
        self.panels: dict[str, str | None] = {}
        self._open_panels: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _VOID:
            return
        attr = dict(attrs)
        panel = None
        if "tab-panel" in (attr.get("class") or "").split():
            panel = attr.get("id") or "?"
            self.panels[panel] = self._open_panels[-1] if self._open_panels else None
            self._open_panels.append(panel)
        self.stack.append((tag, panel))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # ``<path/>`` and friends: opened and closed in one token.
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID:
            return
        match = next(
            (i for i in range(len(self.stack) - 1, -1, -1) if self.stack[i][0] == tag),
            None,
        )
        if match is None:
            self.stray_closes.append(self.getpos()[0])
            return
        while len(self.stack) > match:
            _, panel = self.stack.pop()
            if panel is not None:
                self._open_panels.pop()

    @property
    def open_elements(self) -> list[str]:
        return [tag for tag, _ in self.stack]


def _parse(text: str) -> _Shell:
    shell = _Shell()
    shell.feed(text)
    shell.close()
    return shell


def test_admin_html_markup_is_balanced() -> None:
    shell = _parse(ADMIN_HTML.read_text(encoding="utf-8"))
    assert not shell.stray_closes, f"stray closing tags on lines {shell.stray_closes}"
    assert shell.open_elements == [], f"unclosed elements at EOF: {shell.open_elements}"


def test_no_tab_panel_is_nested_in_another() -> None:
    """A panel inside a panel hides the inner one for the whole session."""
    shell = _parse(ADMIN_HTML.read_text(encoding="utf-8"))
    nested = {pid: outer for pid, outer in shell.panels.items() if outer}
    assert not nested, f"panels nested inside another: {nested}"


def test_the_parser_sees_every_panel() -> None:
    """Guard the guard: a parser that finds no panels proves nothing."""
    shell = _parse(ADMIN_HTML.read_text(encoding="utf-8"))
    assert len(shell.panels) >= 8, shell.panels


def test_a_panel_pushed_inside_another_is_caught() -> None:
    """Mutate a copy into the Settings bug's shape: an unbalanced opener."""
    text = ADMIN_HTML.read_text(encoding="utf-8")
    broken = text.replace(
        '<div class="tab-panel" id="tab-keys" role="tabpanel"',
        '<div class="tab-panel" id="tab-keys" role="tabpanel"><div>',
        1,
    )
    assert broken != text
    shell = _parse(broken)
    nested = {pid: outer for pid, outer in shell.panels.items() if outer}
    assert nested, "a panel pushed inside another must be visible to the parser"
