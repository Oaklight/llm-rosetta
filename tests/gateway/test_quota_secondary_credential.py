"""A probe behind a second credential, and the reading it may not use.

A vendor can report its allowance on a control plane that refuses the key the
model requests send.  The probe contract carries two optional capabilities for
that — ``credential_role`` and a multi-step ``fetch`` — and both have one rule
attached: the reading is *advisory*.  It is shown, it may weight a share, and
it never demotes, excludes or dates a model credential, because none of those
decisions is about the control key.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from llm_rosetta_gateway.quota import service
from llm_rosetta_gateway.quota.models import (
    QuotaReading,
    QuotaWindow,
    credential_fingerprint,
)
from llm_rosetta_gateway.quota.service import (
    provider_exhausted_until,
    provider_spent,
)
from llm_rosetta_gateway.quota.state import QUOTA_STATE
from llm_rosetta.contracts.auth import (
    DEFAULT_CREDENTIAL_ROLE,
    AuthContext,
    BaseAuthExchange,
    InteractiveSetupRequired,
    credential_for,
)
from llm_rosetta.contracts.quota import (
    PROBE_ROLE_API_KEY,
    ProbeRefused,
    QuotaProbeError,
    QuotaWindow as ProbeWindow,
    parse_iso,
)

_PROBE = "p-secondary"
_RUNTIME = "sk-runtime-0000000000000000000000000000"
_CONTROL = "vela_ctrl_00000000000000000000000000000000"
_URL = "https://control.example.test"


@pytest.fixture(autouse=True)
def _clean_quota_state():
    QUOTA_STATE.clear()
    yield
    QUOTA_STATE.clear()


class _Response:
    """The minimum of the vendored ``Response`` the probe path touches."""

    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body) if not isinstance(body, str) else body

    def json(self) -> Any:
        return self._body


class _Client:
    """A client that answers by URL suffix and records what it was sent."""

    def __init__(self, answers: dict[str, _Response]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> _Client:
        return self

    async def __aenter__(self) -> _Client:
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    async def get(self, url: str, headers: Any = None) -> _Response:
        self.calls.append((url, dict(headers or {})))
        for suffix, response in self.answers.items():
            if url.endswith(suffix):
                return response
        return _Response(404, {"error": {"message": "no such route"}})


def _install(monkeypatch: pytest.MonkeyPatch, client: _Client) -> None:
    monkeypatch.setattr(service, "AsyncClient", client)
    monkeypatch.setattr(service, "HttpResponse", _Response)


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


class _ControlProbe:
    """A probe that reads the allowance with the vendor's control credential."""

    id = _PROBE
    default_url = _URL
    credential_role = "control"

    def __init__(self, *, windows: list[ProbeWindow] | None = None) -> None:
        self._windows = windows or [
            ProbeWindow(name="5 hours", used_percent=40.0, span_seconds=18000)
        ]

    def match(self, base_url: str) -> str | None:
        return _URL

    def headers(self, key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {key}"}

    def parse(self, body: Any) -> tuple[str | None, list[ProbeWindow]]:
        return None, list(self._windows)

    def describe_error(self, status_code: int, body: Any) -> str:
        return f"refused ({status_code})"

    def inband(self, status_code: int, body: Any) -> Any | None:
        return None


class _ModelProbe(_ControlProbe):
    """The same probe reading the credential the requests themselves send."""

    credential_role = PROBE_ROLE_API_KEY


class _Exchange(BaseAuthExchange):
    """An exchange that serves a control credential beside the model one."""

    def __init__(self, *, control: str | None = _CONTROL) -> None:
        self._control = control

    async def credential(self, ctx: AuthContext) -> str:
        return _RUNTIME

    async def credential_for(self, ctx: AuthContext, role: str) -> str:
        if role == "control":
            if self._control is None:
                raise InteractiveSetupRequired("no sign-in yet")
            return self._control
        return await self.credential(ctx)


class _NoRoles(BaseAuthExchange):
    """An exchange that predates the role member."""

    async def credential(self, ctx: AuthContext) -> str:
        return _RUNTIME


def _pinfo(exchange: Any = None, *, key: str = _RUNTIME) -> Any:
    from llm_rosetta_gateway.transport.provider_info import KeyRing

    ring = KeyRing(key)
    return SimpleNamespace(
        name="od",
        config_name="od",
        keyless=False,
        base_url="https://model.example.test/v1",
        proxy_url=None,
        quota_probe=_PROBE,
        key_ring=ring,
        auth_exchange=exchange,
        auth_context=AuthContext(provider="od") if exchange is not None else None,
        # A vendor that stamps the credential into its own header too, so a
        # branch that stops skipping it is visible in the request.
        auth_headers_for=lambda k: {"Authorization": f"Bearer {k}", "x-api-key": k},
        active_key=lambda: key,
    )


def _run(pinfo: Any, probe: Any) -> Any:
    return asyncio.run(service._probe_one(pinfo, probe, _URL, pinfo.active_key()))


# ---------------------------------------------------------------------------
# Which credential the read sends
# ---------------------------------------------------------------------------


class TestThePrimaryCredential:
    def test_a_probe_with_no_role_reads_the_model_key(self, monkeypatch):
        client = _Client({"control.example.test": _Response(200, {"windows": []})})
        _install(monkeypatch, client)
        probe = _ModelProbe()

        reading = _run(_pinfo(), probe)

        assert reading.status == "ok"
        assert reading.advisory is False
        assert client.calls[0][1]["Authorization"] == f"Bearer {_RUNTIME}"
        # The lane's own header is the model credential's to carry.
        assert client.calls[0][1]["x-api-key"] == _RUNTIME


class TestTheSecondaryCredential:
    def test_the_read_uses_the_role_the_probe_names(self, monkeypatch):
        client = _Client({"control.example.test": _Response(200, {})})
        _install(monkeypatch, client)

        reading = _run(_pinfo(_Exchange()), _ControlProbe())

        assert reading.status == "ok"
        assert reading.advisory is True
        # The whole header set, not one key: the probe's own headers are the
        # only ones on the request, so neither the model credential nor the
        # lane's header can travel to the control plane.
        headers = client.calls[0][1]
        assert headers["Authorization"] == f"Bearer {_CONTROL}"
        assert "x-api-key" not in headers
        assert _RUNTIME not in str(headers)

    def test_an_exchange_without_the_role_reports_unsupported(self, monkeypatch):
        client = _Client({})
        _install(monkeypatch, client)

        reading = _run(_pinfo(_NoRoles()), _ControlProbe())

        assert reading.status == "unsupported"
        assert "control" in reading.error
        assert client.calls == []

    def test_no_sign_in_means_no_request(self, monkeypatch):
        client = _Client({})
        _install(monkeypatch, client)

        reading = _run(_pinfo(_Exchange(control=None)), _ControlProbe())

        assert reading.status == "unsupported"
        assert "no sign-in yet" in reading.error
        assert client.calls == []

    def test_no_exchange_means_no_request(self, monkeypatch):
        client = _Client({})
        _install(monkeypatch, client)

        reading = _run(_pinfo(), _ControlProbe())

        assert reading.status == "unsupported"
        assert client.calls == []

    def test_the_reading_is_filed_under_the_model_key(self, monkeypatch):
        """The panel looks a lane up by the key requests use, not the control one."""
        client = _Client({"control.example.test": _Response(200, {})})
        _install(monkeypatch, client)

        reading = _run(_pinfo(_Exchange()), _ControlProbe())

        assert reading.credential == credential_fingerprint(_RUNTIME)


# ---------------------------------------------------------------------------
# More than one request
# ---------------------------------------------------------------------------


class _MultiStepProbe(_ControlProbe):
    """A probe whose allowance takes three reads, as the control plane does."""

    async def fetch(self, read: Any, credential: str, url: str) -> Any:
        status, body = await read(f"{url}/workspaces/current", None)
        assert status == 200
        workspace = body["activeWorkspaceId"]
        status, body = await read(f"{url}/workspaces/{workspace}/preflight", None)
        if status >= 400:
            raise ProbeRefused(status, body)
        plan_windows = body["codingPlan"]["windows"]
        status, summary = await read(f"{url}/billing/summary", None)
        plan = summary.get("tier") if status < 400 else None
        return plan, [
            ProbeWindow(
                name="5 hours",
                used_percent=window["used"] / window["limit"] * 100,
                resets_at=parse_iso(window.get("resetsAt")),
            )
            for window in plan_windows
        ]


class TestAMultiStepRead:
    @staticmethod
    def _answers() -> dict[str, _Response]:
        return {
            "/workspaces/current": _Response(200, {"activeWorkspaceId": "w-1"}),
            "/workspaces/w-1/preflight": _Response(
                200,
                {
                    "codingPlan": {
                        "windows": [
                            {
                                "used": 30,
                                "limit": 100,
                                "resetsAt": "2026-11-16T12:00:00Z",
                            }
                        ]
                    }
                },
            ),
            "/billing/summary": _Response(200, {"tier": "pro"}),
        }

    def test_the_probe_drives_every_step(self, monkeypatch):
        client = _Client(self._answers())
        _install(monkeypatch, client)

        reading = _run(_pinfo(_Exchange()), _MultiStepProbe())

        assert reading.status == "ok"
        assert reading.plan == "pro"
        assert [window.name for window in reading.windows] == ["5 hours"]
        assert reading.windows[0].used_percent == pytest.approx(30.0)
        assert [url.removeprefix(_URL) for url, _ in client.calls] == [
            "/workspaces/current",
            "/workspaces/w-1/preflight",
            "/billing/summary",
        ]
        # Every step names the control credential, never the model one.
        assert {headers["Authorization"] for _, headers in client.calls} == {
            f"Bearer {_CONTROL}"
        }

    def test_a_refused_step_is_described_by_the_probe(self, monkeypatch):
        answers = self._answers()
        answers["/workspaces/w-1/preflight"] = _Response(403, {"error": "nope"})
        client = _Client(answers)
        _install(monkeypatch, client)

        reading = _run(_pinfo(_Exchange()), _MultiStepProbe())

        assert reading.status == "error"
        assert reading.error == "refused (403)"

    def test_an_answer_without_a_window_is_an_error(self, monkeypatch):
        answers = self._answers()
        answers["/workspaces/w-1/preflight"] = _Response(200, {"codingPlan": {}})
        client = _Client(answers)
        _install(monkeypatch, client)

        class _Strict(_MultiStepProbe):
            async def fetch(self, read: Any, credential: str, url: str) -> Any:
                status, body = await read(f"{url}/workspaces/w-1/preflight", None)
                if not body.get("codingPlan", {}).get("windows"):
                    raise QuotaProbeError("no window in the answer")
                return None, []

        reading = _run(_pinfo(_Exchange()), _Strict())

        assert reading.status == "error"
        assert "no window" in reading.error


# ---------------------------------------------------------------------------
# What an advisory reading may not do
# ---------------------------------------------------------------------------


def _advisory_reading(key: str = _RUNTIME) -> QuotaReading:
    return QuotaReading(
        probe=_PROBE,
        credential=credential_fingerprint(key),
        windows=[
            QuotaWindow(
                name="5 hours",
                used_percent=100.0,
                span_seconds=18000,
                resets_at=datetime.now(timezone.utc) + timedelta(hours=1),
            )
        ],
        advisory=True,
    )


class TestAnAdvisoryReadingNeverDecidesTheLane:
    def test_the_key_stays_usable(self):
        QUOTA_STATE.put_reading(_advisory_reading())

        assert QUOTA_STATE.is_usable(_PROBE, _RUNTIME) is True

    def test_a_spent_control_window_does_not_mark_the_provider_spent(self):
        pinfo = _pinfo()
        pinfo.quota_probe = _PROBE
        QUOTA_STATE.put_reading(_advisory_reading())

        assert provider_spent(pinfo) is False
        assert provider_exhausted_until(pinfo) is None


# ---------------------------------------------------------------------------
# The role rule itself
# ---------------------------------------------------------------------------


class TestTheCredentialRoleRule:
    def test_the_default_role_is_the_model_credential(self):
        ctx = AuthContext(provider="od")

        assert DEFAULT_CREDENTIAL_ROLE == "api_key"
        assert asyncio.run(credential_for(_NoRoles(), ctx)) == _RUNTIME
        assert asyncio.run(credential_for(_NoRoles(), ctx, "api_key")) == _RUNTIME

    def test_an_exchange_that_predates_the_member_refuses_another_role(self):
        ctx = AuthContext(provider="od")

        with pytest.raises(Exception, match="serves no 'control' credential"):
            asyncio.run(credential_for(_NoRoles(), ctx, "control"))

    def test_an_exchange_that_serves_the_role_answers(self):
        ctx = AuthContext(provider="od")

        assert asyncio.run(credential_for(_Exchange(), ctx, "control")) == _CONTROL
        assert asyncio.run(credential_for(_Exchange(), ctx)) == _RUNTIME


class TestAnAdvisoryReadingDoesNotClearADemotion:
    """A fresh control reading is not evidence that a demotion has passed.

    The demotion was written against the serving credential after an in-band
    refusal; clearing it from a read made with another key would hand the lane
    back on evidence about a different credential — through the round trip,
    not only through the predicate.
    """

    @staticmethod
    def _config():
        from llm_rosetta_gateway.config import GatewayConfig

        return GatewayConfig(
            {
                "providers": {
                    "od": {
                        "type": "chat_completions",
                        "api_key": _RUNTIME,
                        "base_url": "https://model.example.test/v1",
                    }
                },
                "models": {"m": {"provider": "od"}},
            }
        )

    def _read(self, monkeypatch, config, *, advisory: bool):
        pinfo = config.providers["od"]
        pinfo.quota_probe = _PROBE
        pinfo.quota_probe_impl = _ControlProbe()
        pinfo.quota_usage_url = _URL
        fingerprint = credential_fingerprint(_RUNTIME)

        async def fake_probe_one(*args: Any, **kwargs: Any) -> QuotaReading:
            return QuotaReading(
                probe=_PROBE,
                credential=fingerprint,
                windows=[QuotaWindow(name="5 hours", used_percent=10.0)],
                advisory=advisory,
            )

        monkeypatch.setattr(service, "_probe_one", fake_probe_one)
        asyncio.run(service.read_provider_quota(config, "od", force=True))

    def test_the_demotion_survives_an_advisory_read(self, monkeypatch):
        config = self._config()
        until = datetime.now(timezone.utc) + timedelta(hours=2)
        QUOTA_STATE.demote(_PROBE, credential_fingerprint(_RUNTIME), until, "429")

        self._read(monkeypatch, config, advisory=True)

        assert QUOTA_STATE.is_usable(_PROBE, _RUNTIME) is False

    def test_a_reading_with_the_key_itself_still_clears_it(self, monkeypatch):
        """The control: the same round trip with the serving credential."""
        config = self._config()
        until = datetime.now(timezone.utc) + timedelta(hours=2)
        QUOTA_STATE.demote(_PROBE, credential_fingerprint(_RUNTIME), until, "429")

        self._read(monkeypatch, config, advisory=False)

        assert QUOTA_STATE.is_usable(_PROBE, _RUNTIME) is True
