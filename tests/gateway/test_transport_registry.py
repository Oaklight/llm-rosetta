"""Transport scope: an extension's transports reach only the shims that may name them.

Every test installs real extension wheels into a temporary extensions directory
and loads them with core's loader, so the manifests the registry reads are the
ones the loader accepted.  Resolution reads manifests only; the transport module
is imported once, at app creation, for the extensions a shim or a provider uses.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from _ext_wheels import build_wheel
from llm_rosetta.api import (
    configure_extensions_dir,
    get_shim,
    install_wheel,
    list_extensions,
    load_providers,
    register_kind_handler,
)
from llm_rosetta.contracts.extension import UnresolvedTransport
from llm_rosetta_gateway.config.providers import build_provider_info
from llm_rosetta_gateway.transport import registry

TRANSPORT_SOURCE = textwrap.dedent(
    """\
    class Pigeon:
        def __init__(self, *, timeout: float = 1.0) -> None:
            self.timeout = timeout
            self.owner = OWNER

    OWNER = {owner!r}
    TRANSPORTS = {{{ids}}}
    """
)


def _transport_module(owner: str, *ids: str) -> str:
    table = ", ".join(f"{name!r}: Pigeon" for name in ids)
    return TRANSPORT_SOURCE.format(owner=owner, ids=table)


@pytest.fixture
def ext_dir(tmp_path: Path) -> Path:
    path = tmp_path / "extensions"
    path.mkdir()
    configure_extensions_dir(path)
    return path


def _install(
    ext_dir: Path,
    ext_id: str,
    *,
    transports: tuple[str, ...] = (),
    uses: str | None = None,
    requires: tuple[str, ...] = (),
    transport_module: str | None = None,
) -> None:
    """Install extension *ext_id* with one shim that names transport *uses*."""
    shim = f"{ext_id}--chat_completions"
    members: dict[str, str] = {}
    if transports:
        members["transport.py"] = (
            transport_module
            if transport_module is not None
            else _transport_module(ext_id, *transports)
        )
    wheel = build_wheel(
        ext_id,
        "0.1.0",
        [shim],
        members=members,
        transport_of={shim: uses} if uses else None,
        transports=transports,
        requires_extensions=requires,
    )
    install_wheel(wheel, dest=ext_dir)


def _load() -> None:
    load_providers()
    for item in list_extensions():
        assert item.state == "loaded", (item.id, item.reason)


def _status(ext_id: str):
    return next(item for item in list_extensions() if item.id == ext_id)


def _imported(ext_id: str) -> bool:
    return f"llm_rosetta_ext.{ext_id}.transport" in sys.modules


class TestResolvingFromTheShimsScope:
    def test_an_extension_reaches_its_own_transport(self, ext_dir):
        _install(ext_dir, "acme", transports=("pigeon",), uses="pigeon")
        _load()

        lane = registry.resolve_lane("pigeon", "acme--chat_completions")

        assert (lane.name, lane.owner, lane.key) == ("pigeon", "acme", "acme:pigeon")

    def test_a_builtin_is_reachable_from_an_extensions_shim(self, ext_dir):
        _install(ext_dir, "acme", transports=("pigeon",))
        _load()

        lane = registry.resolve_lane("acp", "acme--chat_completions")

        assert (lane.owner, lane.key) == (None, "acp")

    def test_a_required_extensions_transport_is_reachable(self, ext_dir):
        _install(ext_dir, "lanes", transports=("pigeon",))
        _install(ext_dir, "acme", requires=("lanes",), uses="pigeon")
        _load()

        lane = registry.resolve_lane("pigeon", "acme--chat_completions")

        assert lane.owner == "lanes"

    def test_an_unrequired_extensions_transport_is_unreachable(self, ext_dir):
        _install(ext_dir, "lanes", transports=("pigeon",))
        _install(ext_dir, "acme")
        _load()

        with pytest.raises(UnresolvedTransport, match="does not list it under"):
            registry.resolve_lane("pigeon", "acme--chat_completions")

    def test_a_shim_outside_every_extension_reaches_the_builtins_only(self, ext_dir):
        _install(ext_dir, "acme", transports=("pigeon",))
        _load()

        with pytest.raises(UnresolvedTransport, match="only the shims of an"):
            registry.resolve_lane("pigeon", "deepseek--chat_completions")
        with pytest.raises(UnresolvedTransport):
            registry.resolve_lane("pigeon", None)
        assert registry.resolve_lane("http", None).key == "http"

    def test_two_extensions_may_declare_the_same_id_and_each_gets_its_own(
        self, ext_dir
    ):
        _install(ext_dir, "acme", transports=("pigeon",), uses="pigeon")
        _install(ext_dir, "beta", transports=("pigeon",), uses="pigeon")
        _load()

        assert registry.resolve_lane("pigeon", "acme--chat_completions").owner == "acme"
        assert registry.resolve_lane("pigeon", "beta--chat_completions").owner == "beta"
        built: Any = registry.build_lane_transports(timeout=5.0)
        assert built["acme:pigeon"] is not built["beta:pigeon"]
        assert (built["acme:pigeon"].owner, built["beta:pigeon"].owner) == (
            "acme",
            "beta",
        )

    def test_reachable_names_come_from_manifests_without_importing(self, ext_dir):
        _install(ext_dir, "lanes", transports=("far",))
        _install(ext_dir, "acme", transports=("pigeon",), requires=("lanes",))
        _install(ext_dir, "other", transports=("hidden",))
        _load()

        names = registry.reachable_transport_names("acme--chat_completions")

        assert names == ("acp", "far", "http", "pigeon")
        assert registry.reachable_transport_names(None) == ("acp", "http")
        assert not _imported("acme") and not _imported("lanes")

    def test_a_builtin_id_claimed_by_a_shims_extension_is_refused(self, ext_dir):
        _install(ext_dir, "acme", transports=("acp",))
        load_providers()

        status = _status("acme")
        assert status.state == "failed"
        assert "claims transport 'acp'" in status.reason
        # Its shim is not registered, so it has no scope of its own to resolve in.
        assert get_shim("acme--chat_completions") is None
        assert registry.owning_extension("acme--chat_completions") is None


class TestTheTransportsKind:
    """The gateway registers the ``transports`` kind with core's loader."""

    def test_a_late_registration_fails_an_extension_that_is_already_loaded(
        self, ext_dir
    ):
        register_kind_handler("transports", lambda manifest, folder: [])
        try:
            _install(ext_dir, "acme", transports=("acp",))
            load_providers()
            assert _status("acme").state == "loaded"

            register_kind_handler("transports", registry._claim_check)

            assert _status("acme").state == "failed"
            assert get_shim("acme--chat_completions") is None
        finally:
            register_kind_handler("transports", registry._claim_check)

    def test_an_extension_with_its_own_ids_is_loaded(self, ext_dir):
        _install(ext_dir, "acme", transports=("pigeon",))
        _load()
        assert _status("acme").state == "loaded"


class TestBuildingTheLaneTable:
    def test_http_is_not_a_lane_transport(self):
        built: Any = registry.build_lane_transports(timeout=60.0)

        assert set(built) == {"acp"}
        assert isinstance(built["acp"], registry.AcpTransport)

    def test_an_extension_transport_receives_the_server_timeout(self, ext_dir):
        _install(ext_dir, "acme", transports=("pigeon",), uses="pigeon")
        _load()

        built: Any = registry.build_lane_transports(timeout=42.0)

        assert built["acme:pigeon"].timeout == 42.0

    def test_no_timeout_leaves_the_factory_its_own_default(self, ext_dir):
        _install(ext_dir, "acme", transports=("pigeon",), uses="pigeon")
        _load()

        built: Any = registry.build_lane_transports(timeout=None)

        assert built["acme:pigeon"].timeout == 1.0

    def test_an_unreferenced_transport_is_never_imported(self, ext_dir):
        _install(ext_dir, "acme", transports=("pigeon",))
        _load()

        built: Any = registry.build_lane_transports(timeout=1.0)

        assert set(built) == {"acp"}
        assert not _imported("acme")

    def test_a_configured_provider_makes_its_transport_wanted(self, ext_dir):
        _install(ext_dir, "acme", transports=("pigeon",))
        _load()

        built: Any = registry.build_lane_transports(
            timeout=1.0, wanted={("acme", "pigeon"), ("ghost", "pigeon")}
        )

        assert set(built) == {"acp", "acme:pigeon"}

    def test_a_dependencys_transport_is_built_for_the_dependent(self, ext_dir):
        _install(ext_dir, "lanes", transports=("pigeon",))
        _install(ext_dir, "acme", requires=("lanes",), uses="pigeon")
        _load()

        built: Any = registry.build_lane_transports(timeout=1.0)

        assert built["lanes:pigeon"].owner == "lanes"
        assert not _imported("acme")

    def test_a_builtin_id_claimed_by_an_extension_does_not_reach_the_lane_table(
        self, ext_dir
    ):
        _install(ext_dir, "acme", transports=("acp",))
        load_providers()

        built: Any = registry.build_lane_transports(timeout=60.0)

        assert set(built) == {"acp"}
        assert isinstance(built["acp"], registry.AcpTransport)
        assert not _imported("acme")

    @pytest.mark.parametrize(
        ("source", "message"),
        [
            (
                "raise ImportError('no module named acme_lib')\n",
                r"failed to import: ImportError",
            ),
            ("X = 1\n", "does not expose a TRANSPORTS mapping"),
            ("TRANSPORTS = {'other': object}\n", "has no entry for it"),
            ("TRANSPORTS = {'pigeon': 42}\n", "not callable"),
        ],
    )
    def test_a_broken_transport_module_names_the_extension(
        self, ext_dir, source, message
    ):
        _install(
            ext_dir,
            "acme",
            transports=("pigeon",),
            uses="pigeon",
            transport_module=source,
        )
        _load()

        with pytest.raises(
            registry.TransportPluginError, match=rf"Extension 'acme'.*{message}"
        ):
            registry.build_lane_transports(timeout=60.0)

    def test_a_factory_that_cannot_start_names_the_extension_and_transport(
        self, ext_dir
    ):
        # a constructor that does not take ``timeout=``, as an unedited
        # scaffold once generated
        _install(
            ext_dir,
            "acme",
            transports=("pigeon",),
            uses="pigeon",
            transport_module="class Pigeon:\n    pass\n\nTRANSPORTS = {'pigeon': Pigeon}\n",
        )
        _load()

        with pytest.raises(
            registry.TransportPluginError,
            match=r"'pigeon' of extension 'acme' failed to start: TypeError",
        ):
            registry.build_lane_transports(timeout=60.0)

    def test_an_override_covers_every_lane_without_importing_an_extension(
        self, ext_dir
    ):
        _install(ext_dir, "acme", transports=("pigeon",), uses="pigeon")
        _install(ext_dir, "claims", transports=("pigeon",))
        _load()
        override = object()

        built: Any = registry.build_lane_transports(timeout=60.0, override=override)

        assert set(built) == {"acp", "acme:pigeon", "claims:pigeon"}
        assert all(value is override for value in built.values())
        assert not _imported("acme") and not _imported("claims")


class TestWhatConfigAccepts:
    CFG = {"base_url": "https://acme.example/v1", "api_key": "k"}

    def test_an_own_transport_carries_the_lane(self, ext_dir):
        _install(ext_dir, "acme", transports=("pigeon",), uses="pigeon")
        _load()

        info = build_provider_info(
            "chat_completions", dict(self.CFG), shim_name="acme--chat_completions"
        )

        assert (info.transport, info.transport_owner) == ("pigeon", "acme")
        assert info.transport_key == "acme:pigeon"
        assert not _imported("acme")

    def test_a_builtin_transport_keeps_its_bare_key(self, ext_dir):
        _install(ext_dir, "acme", transports=("pigeon",))
        _load()

        info = build_provider_info(
            "chat_completions",
            {**self.CFG, "transport": "acp"},
            shim_name="acme--chat_completions",
        )

        assert (info.transport, info.transport_owner, info.transport_key) == (
            "acp",
            None,
            "acp",
        )

    def test_an_unreachable_name_is_refused_with_the_scope_message(self, ext_dir):
        _install(ext_dir, "lanes", transports=("pigeon",))
        _install(ext_dir, "acme")
        _load()

        with pytest.raises(ValueError, match="unknown transport 'pigeon'") as caught:
            build_provider_info(
                "chat_completions",
                {**self.CFG, "transport": "pigeon"},
                shim_name="acme--chat_completions",
                config_name="mine",
            )

        assert isinstance(caught.value.__cause__, UnresolvedTransport)
        assert "Provider 'mine'" in str(caught.value)
        assert "requires.extensions" in str(caught.value)

    def test_the_old_entry_point_group_is_gone(self):
        assert not hasattr(registry, "TRANSPORT_ENTRY_POINT_GROUP")
        assert not hasattr(registry, "installed_transport_names")
