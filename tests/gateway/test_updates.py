"""The console's update probe: version order, the cached probe, the gate.

The probe answers two questions with different stakes.  "Is there a newer
release" only reads; "may this process replace itself" decides whether a
click can run pip over the code that is answering.  Both are tested here,
and pressed against a fake index so no test reaches PyPI.
"""

from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from llm_rosetta_gateway import updates


# -- Version order ---------------------------------------------------------


@pytest.mark.parametrize(
    ("older", "newer"),
    [
        ("0.14.0.dev22", "0.14.0a1"),
        ("0.14.0a1", "0.14.0b1"),
        ("0.14.0b1", "0.14.0rc1"),
        ("0.14.0rc1", "0.14.0"),
        ("0.14.0", "0.14.0.post1"),
        ("0.14.0.post1.dev0", "0.14.0.post1"),
        ("0.9.0", "0.14.0"),
        ("0.14.0", "0.14.1"),
        ("0.14.9", "0.15.0"),
        ("0.14.0", "1.0.0"),
    ],
)
def test_the_order_matches_pep_440(older: str, newer: str) -> None:
    assert updates.parse_version(older) < updates.parse_version(newer)


def test_a_local_build_is_the_same_release() -> None:
    """``+g1234abc`` names a build of a version, not a later version."""
    assert updates.parse_version("0.14.0.dev22+g1234abc") == updates.parse_version(
        "0.14.0.dev22"
    )


def test_a_dev_build_of_a_pre_release_sorts_before_it() -> None:
    """PEP 440 puts ``rc1.dev0`` before ``rc1``; offering a downgrade would
    follow from getting this branch wrong."""
    assert updates.parse_version("0.15.0rc1.dev0") < updates.parse_version("0.15.0rc1")
    assert not updates.is_newer("0.15.0rc1.dev0", "0.15.0rc1")


def test_unknown_text_sorts_below_every_release() -> None:
    assert updates.parse_version("not-a-version") < updates.parse_version("0.0.1")


def test_newer_compares_against_the_running_version() -> None:
    assert updates.is_newer("0.14.0", "0.14.0.dev22")
    assert not updates.is_newer("0.13.0", "0.14.0.dev22")
    assert not updates.is_newer("0.14.0", "0.14.0")


# -- The probe -------------------------------------------------------------


class _Response:
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeClient:
    """A client that answers from a table and records what was asked."""

    calls: list[str] = []
    table: dict[str, str] = {}

    def __init__(self, **_: Any) -> None:
        pass

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *_: Any) -> bool:
        return False

    async def get(self, url: str) -> _Response:
        _FakeClient.calls.append(url)
        for name, version in _FakeClient.table.items():
            if url == f"https://pypi.org/pypi/{name}/json":
                return _Response(200, {"info": {"version": version}})
            # A release page exists for every version of a listed name.
            if url.startswith(f"https://pypi.org/pypi/{name}/"):
                return _Response(200, {"info": {"version": version}})
        return _Response(404, {})


@pytest.fixture
def fake_index(monkeypatch: pytest.MonkeyPatch) -> type[_FakeClient]:
    _FakeClient.calls = []
    _FakeClient.table = {}
    monkeypatch.setattr(
        "llm_rosetta_gateway._vendor.httpclient.AsyncClient", _FakeClient
    )
    monkeypatch.setattr(
        updates,
        "_cache",
        {"at": 0.0, "payload": None, "ttl": updates.CACHE_SECONDS},
    )
    return _FakeClient


def test_the_probe_offers_the_newer_release(
    fake_index: type[_FakeClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_index.table = {"llm-rosetta": "0.15.0"}
    monkeypatch.setattr(
        updates, "running_versions", lambda: {"core": "0.14.0", "gateway": "0.14.0"}
    )
    payload = asyncio.run(updates.check())
    assert payload["checked"] is True
    assert payload["update"]["version"] == "0.15.0"


def test_a_release_published_only_as_the_gateway_still_counts(
    fake_index: type[_FakeClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two distributions can land minutes apart; either may carry the news."""
    fake_index.table = {"llm-rosetta-gateway": "0.15.0"}
    monkeypatch.setattr(
        updates, "running_versions", lambda: {"core": "0.14.0", "gateway": "0.14.0"}
    )
    payload = asyncio.run(updates.check())
    assert payload["update"]["version"] == "0.15.0"


def test_the_probe_reports_no_update_when_running_the_newest(
    fake_index: type[_FakeClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_index.table = {"llm-rosetta": "0.13.0"}
    monkeypatch.setattr(
        updates, "running_versions", lambda: {"core": "0.14.0", "gateway": "0.14.0"}
    )
    payload = asyncio.run(updates.check())
    assert payload["update"] is None


def test_a_failed_probe_is_retried_after_a_short_window(
    fake_index: type[_FakeClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An offline index is not asked on every panel load, but it is retried."""
    monkeypatch.setattr(
        updates, "running_versions", lambda: {"core": "0.14.0", "gateway": "0.14.0"}
    )
    first = asyncio.run(updates.check())
    assert first["checked"] is False
    assert updates._cache["ttl"] == updates.FAILURE_CACHE_SECONDS

    # Inside the window the failure is reused, not re-asked.
    calls = len(fake_index.calls)
    asyncio.run(updates.check())
    assert len(fake_index.calls) == calls

    # Past it, the probe asks again and can succeed.
    fake_index.table = {"llm-rosetta": "0.15.0"}
    updates._cache["at"] -= updates.FAILURE_CACHE_SECONDS
    second = asyncio.run(updates.check())
    assert second["checked"] is True


def test_a_good_probe_is_reused(
    fake_index: type[_FakeClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_index.table = {"llm-rosetta": "0.15.0"}
    monkeypatch.setattr(
        updates, "running_versions", lambda: {"core": "0.14.0", "gateway": "0.14.0"}
    )
    asyncio.run(updates.check())
    calls = len(fake_index.calls)
    asyncio.run(updates.check())
    assert len(fake_index.calls) == calls


# -- The self-update gate --------------------------------------------------


def test_an_editable_install_may_not_update_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        updates,
        "is_editable",
        lambda name: name == "llm-rosetta-gateway",
    )
    status = updates.self_update_status("0.15.0")
    assert status == {"allowed": False, "reason": "editable_install"}


def test_an_interpreter_without_pip_may_not_update_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(updates, "is_editable", lambda name: False)
    monkeypatch.setattr(updates, "pip_available", lambda: False)
    status = updates.self_update_status("0.15.0")
    assert status == {"allowed": False, "reason": "no_pip"}


def test_a_plain_install_may_update_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(updates, "is_editable", lambda name: False)
    monkeypatch.setattr(updates, "pip_available", lambda: True)
    assert updates.self_update_status("0.15.0") == {"allowed": True, "reason": None}


def test_no_offered_version_means_nothing_to_do() -> None:
    assert updates.self_update_status(None)["reason"] == "no_update"


def test_only_published_distributions_are_installed(
    fake_index: type[_FakeClient],
) -> None:
    """Before the split a release has a core package and no gateway one."""
    fake_index.table = {"llm-rosetta": "0.15.0"}
    assert asyncio.run(updates.published_dists("0.15.0")) == ["llm-rosetta"]

    fake_index.table = {"llm-rosetta": "0.15.0", "llm-rosetta-gateway": "0.15.0"}
    assert asyncio.run(updates.published_dists("0.15.0")) == [
        "llm-rosetta",
        "llm-rosetta-gateway",
    ]


class _Proc:
    def __init__(self, returncode: int, output: bytes) -> None:
        self.returncode = returncode
        self._output = output

    async def communicate(self) -> tuple[bytes, None]:
        return self._output, None

    def kill(self) -> None:
        pass


def test_install_pins_every_published_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _published(version: str, proxy: str | None = None) -> list[str]:
        return ["llm-rosetta", "llm-rosetta-gateway"]

    seen: dict[str, tuple[str, ...]] = {}

    async def _spawn(*command: str, **_: Any) -> _Proc:
        seen["command"] = command
        return _Proc(0, b"Successfully installed llm-rosetta-0.15.0")

    monkeypatch.setattr(updates, "published_dists", _published)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _spawn)
    result = asyncio.run(updates.install("0.15.0"))
    assert result["ok"] is True
    command = seen["command"]
    assert command[1:4] == ("-m", "pip", "install")
    assert "llm-rosetta==0.15.0" in command
    assert "llm-rosetta-gateway==0.15.0" in command


def test_a_failing_pip_reports_its_output(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _published(version: str, proxy: str | None = None) -> list[str]:
        return ["llm-rosetta"]

    async def _spawn(*_: str, **__: Any) -> _Proc:
        return _Proc(1, b"ERROR: Could not find a version")

    monkeypatch.setattr(updates, "published_dists", _published)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _spawn)
    result = asyncio.run(updates.install("0.15.0"))
    assert result["ok"] is False
    assert "Could not find a version" in result["output"]


def test_nothing_to_install_when_no_distribution_has_the_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _published(version: str, proxy: str | None = None) -> list[str]:
        return []

    monkeypatch.setattr(updates, "published_dists", _published)
    result = asyncio.run(updates.install("0.15.0"))
    assert result["ok"] is False
    assert "no 0.15.0 release" in result["output"]


def test_the_command_line_is_the_process_argv() -> None:
    argv = updates._command_line()
    assert argv
    assert all(isinstance(part, str) and part for part in argv)


def test_the_command_line_falls_back_without_proc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _missing(self: Any, *_: Any, **__: Any) -> bytes:
        raise OSError("no /proc here")

    monkeypatch.setattr(updates.Path, "read_bytes", _missing)
    argv = updates._command_line()
    assert argv[:3] == [sys.executable, "-m", "llm_rosetta_gateway"]


# -- The changelog ---------------------------------------------------------


def test_a_build_without_a_changelog_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source checkout or a wheel built without the file has no dialog."""

    def _missing(self: Any, *_: Any, **__: Any) -> str:
        raise OSError("no such file")

    monkeypatch.setattr(updates.Path, "read_text", _missing)
    assert updates.read_changelog("en") == ""


# -- The routes ------------------------------------------------------------


def _request(body: dict[str, Any] | None = None, config: Any = None) -> Any:
    app = SimpleNamespace(config=config, ops_ctx=None)

    def _json() -> dict[str, Any]:
        if body is None:
            raise ValueError("no body")
        return body

    return SimpleNamespace(app=app, json=_json, query_params={})


def _config(update_check: bool = True, self_update: bool = True) -> Any:
    return SimpleNamespace(
        update_check=update_check, self_update=self_update, proxy=None
    )


def test_the_probe_route_honours_the_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_rosetta_gateway.admin.routes.updates import get_updates

    async def _never(*_: Any, **__: Any) -> dict[str, Any]:
        raise AssertionError("the probe must not run when the switch is off")

    monkeypatch.setattr(updates, "check", _never)
    response = asyncio.run(get_updates(_request(config=_config(update_check=False))))
    assert json.loads(response.body) == {"checked": False, "reason": "disabled"}


def test_the_apply_route_refuses_when_self_update_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_rosetta_gateway.admin.routes.updates import apply_update

    async def _never(*_: Any, **__: Any) -> dict[str, Any]:
        raise AssertionError("no probe and no install when the switch is off")

    monkeypatch.setattr(updates, "check", _never)
    response = asyncio.run(
        apply_update(_request({"version": "0.15.0"}, config=_config(self_update=False)))
    )
    assert response.status_code == 403
    assert json.loads(response.body)["reason"] == "disabled"


def test_the_apply_route_installs_only_the_version_on_offer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_rosetta_gateway.admin.routes import updates as routes

    async def _check(*_: Any, **__: Any) -> dict[str, Any]:
        return {
            "update": {"version": "0.15.0"},
            "self_update": {"allowed": True, "reason": None},
        }

    installed: list[str] = []

    async def _install(version: str, *_: Any, **__: Any) -> dict[str, Any]:
        installed.append(version)
        return {"ok": True, "output": "", "dists": ["llm-rosetta"]}

    restarts: list[int] = []
    monkeypatch.setattr(updates, "check", _check)
    monkeypatch.setattr(updates, "install", _install)
    monkeypatch.setattr(updates, "restart_process", lambda: restarts.append(1))

    async def _click() -> Any:
        # The restart is scheduled, not immediate: the response has to leave
        # first.  Both happen on one loop, as they do in the gateway.
        response = await routes.apply_update(
            _request({"version": "0.15.0"}, config=_config())
        )
        await asyncio.sleep(0.9)
        return response

    response = asyncio.run(_click())
    body = json.loads(response.body)
    assert response.status_code == 200
    assert body["status"] == "restarting"
    assert installed == ["0.15.0"]
    assert restarts == [1]


def test_the_apply_route_refuses_a_version_nobody_offered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_rosetta_gateway.admin.routes.updates import apply_update

    async def _check(*_: Any, **__: Any) -> dict[str, Any]:
        return {
            "update": {"version": "0.15.0"},
            "self_update": {"allowed": True, "reason": None},
        }

    monkeypatch.setattr(updates, "check", _check)
    response = asyncio.run(
        apply_update(_request({"version": "9.9.9"}, config=_config()))
    )
    assert response.status_code == 409
    assert json.loads(response.body)["reason"] == "no_update"


def test_a_failed_install_keeps_the_old_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_rosetta_gateway.admin.routes.updates import apply_update

    async def _check(*_: Any, **__: Any) -> dict[str, Any]:
        return {
            "update": {"version": "0.15.0"},
            "self_update": {"allowed": True, "reason": None},
        }

    async def _install(*_: Any, **__: Any) -> dict[str, Any]:
        return {"ok": False, "output": "ERROR: no matching distribution", "dists": []}

    restarts: list[int] = []
    monkeypatch.setattr(updates, "check", _check)
    monkeypatch.setattr(updates, "install", _install)
    monkeypatch.setattr(updates, "restart_process", lambda: restarts.append(1))
    response = asyncio.run(
        apply_update(_request({"version": "0.15.0"}, config=_config()))
    )
    assert response.status_code == 502
    assert "no matching distribution" in json.loads(response.body)["output"]
    assert restarts == []
