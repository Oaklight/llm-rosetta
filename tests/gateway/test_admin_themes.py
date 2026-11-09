"""The admin theme registry and the theme files it names (#1471).

The registry in ``admin/themes.py`` is the single place a theme is named: the
page's boot script, the Settings picker, ``tools/theme_matrix`` and the wheel
packaging all derive from it.  Two directions must hold, so these tests check
both — every registered theme has its file, and every file under
``css/themes/`` is registered — and then that the files stay interchangeable:
each theme declares the same token names as the default, or a component that
uses a token goes unstyled in one scheme.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_rosetta_gateway.admin import themes as registry
from llm_rosetta_gateway.admin.routes.auth import serve_admin_html

ADMIN_DIR = Path(registry.__file__).parent
THEME_DIR = ADMIN_DIR / "css" / "themes"

_BLOCK = re.compile(
    r'\[data-scheme="(?P<scheme>[^"]+)"\](?:\[data-mode="(?P<mode>[^"]+)"\])?\s*\{(?P<body>[^}]*)\}'
)
_TOKEN = re.compile(r"(--[A-Za-z0-9-]+)\s*:")
_COMMENT = re.compile(r"/\*[\s\S]*?\*/")


def _theme_text(theme: registry.Theme) -> str:
    """One theme's stylesheet, comments removed (they may sit inside a block)."""
    return _COMMENT.sub("", (ADMIN_DIR / theme.file).read_text(encoding="utf-8"))


def _blocks(theme: registry.Theme) -> dict[str | None, set[str]]:
    """Return ``{mode: token names}`` for one theme, ``None`` for its shared block."""
    blocks: dict[str | None, set[str]] = {}
    for match in _BLOCK.finditer(_theme_text(theme)):
        blocks[match.group("mode")] = set(_TOKEN.findall(match.group("body")))
    return blocks


class TestRegistry:
    def test_every_registered_theme_has_its_file(self):
        for theme in registry.THEMES:
            assert (ADMIN_DIR / theme.file).is_file(), theme.file

    def test_every_theme_file_is_registered(self):
        on_disk = sorted(path.stem for path in THEME_DIR.glob("*.css"))
        assert on_disk == sorted(registry.theme_names())

    def test_the_file_name_is_the_theme_name(self):
        # The file stem is the data-scheme value, which is what a selector and
        # a stored preference name; a mismatch would load one theme under
        # another's name.
        for theme in registry.THEMES:
            assert Path(theme.file).stem == theme.name
            assert theme.file == f"css/themes/{theme.name}.css"

    def test_names_are_unique_and_labelled(self):
        names = [theme.name for theme in registry.THEMES]
        assert len(names) == len(set(names))
        assert all(theme.label.strip() for theme in registry.THEMES)

    def test_exactly_one_default_and_it_exists(self):
        defaults = [theme for theme in registry.THEMES if theme.default]
        assert len(defaults) == 1
        assert registry.default_theme() is defaults[0]

    def test_url_shape(self):
        for theme in registry.THEMES:
            url = registry.theme_url(theme)
            assert url == f"/admin/static/{theme.file}"
            assert url.startswith(registry.STATIC_URL_PREFIX)
            # A same-origin path: no scheme, no host, no traversal.
            assert not url.startswith("//") and ".." not in url

    def test_payload_is_the_page_contract(self):
        payload = registry.registry_payload()
        assert payload["default"] == registry.default_theme().name
        assert payload["themes"] == [
            {"name": t.name, "label": t.label, "url": registry.theme_url(t)}
            for t in registry.THEMES
        ]
        # It crosses the wire as JSON inside a script tag.
        assert json.loads(json.dumps(payload)) == payload

    def test_an_unknown_name_is_not_an_error(self):
        # A stored preference can name a theme an upgrade removed; the panel
        # falls back rather than failing.
        assert registry.theme_by_name("no-such-theme") is None
        assert registry.theme_by_name(None) is None
        assert registry.theme_by_name(registry.default_theme().name) is not None


class TestThemeFiles:
    def test_every_theme_declares_shared_light_and_dark(self):
        for theme in registry.THEMES:
            assert set(_blocks(theme)) == {None, "light", "dark"}, theme.name

    def test_every_theme_offers_the_same_tokens_per_mode(self):
        # The tokens a component can resolve are its theme's shared block plus
        # the mode block.  A theme may override a shared token in one mode
        # (emerald re-declares --dot/--grid for the dark grid grey), but the
        # *available* set must match the default's: a token missing from one
        # theme leaves that component unstyled there, and a token only one
        # theme declares is dead weight in the others.
        reference = _blocks(registry.default_theme())
        for mode in ("light", "dark"):
            expected = reference[None] | reference[mode]
            for theme in registry.THEMES:
                if theme is registry.default_theme():
                    continue
                blocks = _blocks(theme)
                available = blocks[None] | blocks[mode]
                assert available == expected, (
                    f"{theme.name} ({mode}): "
                    f"missing {sorted(expected - available)}, "
                    f"extra {sorted(available - expected)}"
                )

    def test_a_theme_declares_values_only(self):
        for theme in registry.THEMES:
            for match in _BLOCK.finditer(_theme_text(theme)):
                for declaration in match.group("body").split(";"):
                    if not declaration.strip():
                        continue
                    prop = declaration.split(":", 1)[0].strip()
                    assert prop.startswith("--"), (
                        f"{theme.file}: {prop!r} is not a theme token"
                    )

    def test_the_default_themes_structural_block_is_not_empty(self):
        # A structural block that parsed as empty would make the completeness
        # check above pass vacuously.
        shared = _blocks(registry.default_theme())[None]
        assert len(shared) >= 20


class TestInjection:
    @pytest.mark.asyncio
    async def test_the_page_receives_the_registry_once(self):
        app = SimpleNamespace(admin_custom_head="")
        response = await serve_admin_html(SimpleNamespace(app=app))

        html = (
            response.body if isinstance(response.body, str) else response.body.decode()
        )
        assert "<!--admin-themes-->" not in html
        assert html.count('id="adminThemes"') == 1
        injected = re.search(
            r'<script id="adminThemes" type="application/json">(.*?)</script>',
            html,
        )
        assert injected is not None
        assert json.loads(injected.group(1)) == registry.registry_payload()

    @pytest.mark.asyncio
    async def test_a_custom_head_does_not_duplicate_the_registry(self):
        app = SimpleNamespace(admin_custom_head="<style>.downstream{}</style>")
        response = await serve_admin_html(SimpleNamespace(app=app))

        html = (
            response.body if isinstance(response.body, str) else response.body.decode()
        )
        assert html.count('id="adminThemes"') == 1
        assert html.index('id="adminThemes"') < html.index(".downstream")

    @pytest.mark.asyncio
    async def test_the_static_link_names_the_default_theme(self):
        # The static href is the no-JavaScript fallback, and it is a second
        # hand-written copy of the default theme's URL: if the registry's
        # default changes and this link does not, a JS-disabled page silently
        # shows the old theme while the comment next to the link promises the
        # default.  Bound here so the two cannot drift.
        link = re.search(
            r'<link[^>]+id="themeSheet"[^>]+href="([^"]+)"',
            (ADMIN_DIR / "admin.html").read_text(encoding="utf-8"),
        )
        assert link is not None, "expected the static themeSheet link"
        assert link.group(1) == registry.theme_url(registry.default_theme())

    @pytest.mark.asyncio
    async def test_a_label_cannot_close_the_script_tag(self, monkeypatch):
        # The registry is embedded in an inline JSON script, and the HTML
        # parser ends that element at the first `</script` it sees.  Today's
        # labels are fixed literals; the escaping is what keeps a future one
        # from ending the element and leaving the rest as markup.
        hostile = registry.Theme(
            "hostile", "x</script><script>alert(1)</script>", "css/themes/hostile.css"
        )
        monkeypatch.setattr(registry, "THEMES", (*registry.THEMES, hostile))
        app = SimpleNamespace(admin_custom_head="<style>.unique-hostile{}</style>")

        response = await serve_admin_html(SimpleNamespace(app=app))
        html = (
            response.body if isinstance(response.body, str) else response.body.decode()
        )

        injected = re.search(
            r'<script id="adminThemes" type="application/json">(.*?)</script>',
            html,
        )
        assert injected is not None
        payload = injected.group(1)
        assert "</script" not in payload
        assert "<script" not in payload
        # The payload is still the registry: escaping must not change what the
        # page reads back.
        labels = [theme["label"] for theme in json.loads(payload)["themes"]]
        assert labels[-1] == hostile.label
