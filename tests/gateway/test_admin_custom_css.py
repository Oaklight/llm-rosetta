"""The operator's admin theme override (``server.admin_custom_css``).

It is CSS only, appended after the built-in schemes and after any downstream
``custom_head`` fragment, so an operator can restyle the panel from the config
file without touching code — and without shipping JavaScript into every
session.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from llm_rosetta_gateway.admin import _admin_theme_style, apply_admin_theme
from llm_rosetta_gateway.admin.routes.config.settings import put_server_settings
from llm_rosetta_gateway.config import GatewayConfig, JsoncConfigIO
from llm_rosetta_gateway.config.settings import MAX_ADMIN_CUSTOM_CSS


def _config(server: dict | None = None) -> GatewayConfig:
    return GatewayConfig({"providers": {}, "models": {}, "server": server or {}})


class TestConfigParsing:
    def test_absent_is_empty(self):
        assert _config().admin_custom_css == ""

    def test_value_is_stripped(self):
        cfg = _config({"admin_custom_css": "  :root{--accent:red}  "})
        assert cfg.admin_custom_css == ":root{--accent:red}"

    def test_wrong_type_is_rejected(self):
        with pytest.raises(ValueError, match="admin_custom_css"):
            _config({"admin_custom_css": 5})

    def test_oversize_is_rejected(self):
        with pytest.raises(ValueError, match="limited to"):
            _config({"admin_custom_css": "a" * (MAX_ADMIN_CUSTOM_CSS + 1)})


class TestHeadInjection:
    def test_empty_produces_no_block(self):
        assert _admin_theme_style("") == ""
        assert _admin_theme_style(None) == ""

    def test_a_stray_close_tag_cannot_end_the_block(self):
        block = _admin_theme_style("a{} /* </style><script>x()</script> */")

        assert block.count("</style>") == 1
        assert "<\\/style>" in block

    def test_the_override_goes_after_any_downstream_fragment(self):
        app = SimpleNamespace()
        app._admin_head_prefix = "<style>.downstream{}</style>"

        apply_admin_theme(app, ":root{--accent:red}")

        html = app.admin_custom_head
        assert html.index(".downstream") < html.index("admin-custom-css")

    def test_clearing_removes_the_block(self):
        app = SimpleNamespace()
        app._admin_head_prefix = "<style>.downstream{}</style>"
        apply_admin_theme(app, ":root{--accent:red}")

        apply_admin_theme(app, "")

        assert "admin-custom-css" not in app.admin_custom_head


def _request(tmp_path, raw: dict, body: dict) -> SimpleNamespace:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    app = SimpleNamespace(
        config_path=str(path),
        config_io=JsoncConfigIO(),
        gateway_config=None,
        provider_validation_warnings=[],
        shim_source_signature=None,
        rate_limit_state=None,
    )
    return SimpleNamespace(
        app=app, path_params={}, method="PUT", headers={}, json=lambda: body
    )


@pytest.mark.asyncio
async def test_put_saves_and_clears_the_override(tmp_path):
    request = _request(
        tmp_path,
        {"providers": {}, "models": {}},
        {"admin_custom_css": ":root{--accent:red}"},
    )

    saved = json.loads((await put_server_settings(request)).body)
    assert saved["ok"] is True
    on_disk = json.loads((tmp_path / "config.json").read_text())
    assert on_disk["server"]["admin_custom_css"] == ":root{--accent:red}"

    request.json = lambda: {"admin_custom_css": ""}
    await put_server_settings(request)
    on_disk = json.loads((tmp_path / "config.json").read_text())
    assert "admin_custom_css" not in on_disk["server"]


@pytest.mark.asyncio
async def test_put_rejects_a_non_string(tmp_path):
    request = _request(
        tmp_path, {"providers": {}, "models": {}}, {"admin_custom_css": 5}
    )

    response = await put_server_settings(request)

    assert response.status_code == 400
