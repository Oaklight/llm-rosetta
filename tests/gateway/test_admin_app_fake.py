"""The fake app the admin tests share answers like the real app's admin context."""

from __future__ import annotations

from _admin_app import attach_admin, fake_app
from llm_rosetta_gateway.admin_api import AdminContext, admin_context
from llm_rosetta_gateway.app.admin_context import GatewayAdminContext
from types import SimpleNamespace


def test_the_fake_app_serves_the_config_it_holds():
    cfg = object()
    request = SimpleNamespace(app=fake_app(gateway_config=cfg))
    assert isinstance(request.app.admin, GatewayAdminContext)
    assert isinstance(admin_context(request), AdminContext)
    assert admin_context(request).served_config() is cfg


def test_the_fake_app_follows_a_config_assigned_later():
    request = SimpleNamespace(app=fake_app())
    assert admin_context(request).served_config() is None
    cfg = object()
    request.app.gateway_config = cfg
    assert admin_context(request).served_config() is cfg


def test_attach_admin_works_on_a_class_based_fake():
    class _App:
        gateway_config = None

    app = attach_admin(_App())
    cfg = object()
    app.gateway_config = cfg
    assert admin_context(SimpleNamespace(app=app)).served_config() is cfg
