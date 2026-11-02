"""A provider entry's own capabilities narrow the route for one request.

A model may be served by several upstreams that do not all do the same things
— one text-only, one with vision.  Before this slice the route handed an image
to whichever lane the strategy picked.  Now an entry may declare what it serves
for that model, and the eligibility stage drops it for a request whose needs it
does not cover.  The filter is per request: the same entry stays a candidate
for everything else.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from llm_rosetta_gateway.config import GatewayConfig
from llm_rosetta_gateway.maintenance.startup import ProviderNotReady
from llm_rosetta_gateway.routing import (
    CAPABILITY,
    Eligibility,
    EligibilityStage,
    RuleFacts,
    SelectionContext,
    classify_unavailable,
    required_capabilities,
)
from llm_rosetta_gateway.routing.rules import facts_from_ir
from llm_rosetta_gateway.config.parse import (
    declared_provider_entries,
    parse_entry_capabilities,
)
from llm_rosetta_gateway.routing.strategy import ModelRoute, ProviderEntry
from llm_rosetta_gateway._vendor.httpserver import Response
from llm_rosetta_gateway.transport.provider_info import ProviderInfo


def _pinfo(name: str) -> ProviderInfo:
    return ProviderInfo(
        name,
        api_key="k1",
        base_url="https://example.test",
        auth_header_fn=lambda key: {"Authorization": f"Bearer {key}"},
        url_template="{base_url}/x",
    )


def _bare_config(providers: list[str]) -> GatewayConfig:
    """A config with *providers* and one model, for the classifier's table."""
    return GatewayConfig(
        {
            "providers": {
                name: {"type": "openai", "base_url": f"https://{name}.test"}
                for name in providers
            },
            "models": {"m": {"providers": providers}},
        }
    )


def _context(
    entries: list[ProviderEntry], required: frozenset[str] = frozenset()
) -> SelectionContext:
    ctx = SelectionContext("m", "chat_completions")
    ctx.install_route(
        model_route=ModelRoute(providers=entries, route_id="m"),
        providers={entry.name: _pinfo(entry.name) for entry in entries},
        provider_types={},
        provider_shim_names={},
        provider_account_tokens={},
    )
    ctx.with_required_capabilities(required)
    return ctx


class TestTheParsedEntry:
    """The field is read from a ``providers[]`` object, and only there."""

    def test_capabilities_land_on_the_entry(self):
        config = GatewayConfig(
            {
                "providers": {
                    "a": {"type": "openai", "base_url": "https://a.test"},
                    "b": {"type": "openai", "base_url": "https://b.test"},
                },
                "models": {
                    "m": {
                        "providers": [
                            {"name": "a", "capabilities": ["text"]},
                            {"name": "b", "capabilities": ["text", "vision"]},
                        ]
                    }
                },
            }
        )

        entries = config.models["m"].providers

        assert entries[0].capabilities == ("text",)
        assert entries[1].capabilities == ("text", "vision")

    def test_an_entry_without_the_field_declares_nothing(self):
        config = GatewayConfig(
            {
                "providers": {"a": {"type": "openai", "base_url": "https://a.test"}},
                "models": {"m": {"providers": ["a"]}},
            }
        )

        assert config.models["m"].providers[0].capabilities is None

    def test_a_non_list_is_a_config_error(self):
        with pytest.raises(ValueError, match="capabilities"):
            GatewayConfig(
                {
                    "providers": {
                        "a": {"type": "openai", "base_url": "https://a.test"}
                    },
                    "models": {
                        "m": {"providers": [{"name": "a", "capabilities": "vision"}]}
                    },
                }
            )


class TestTheEligibilityStage:
    """The comparison is per request, and only for entries that declare."""

    def test_an_image_skips_a_text_only_entry(self):
        ctx = _context(
            [
                ProviderEntry(name="a", capabilities=("text",)),
                ProviderEntry(name="b", capabilities=("text", "vision")),
            ],
            required=frozenset({"vision"}),
        )

        EligibilityStage().run(ctx)

        assert ctx.eligibility["a"].eligible is False
        assert ctx.eligibility["a"].reason == CAPABILITY
        assert ctx.eligibility["b"].eligible is True

    def test_the_same_entry_stays_a_candidate_for_other_requests(self):
        entries = [
            ProviderEntry(name="a", capabilities=("text",)),
            ProviderEntry(name="b", capabilities=("text", "vision")),
        ]
        ctx = _context(entries)

        EligibilityStage().run(ctx)

        assert ctx.eligibility["a"].eligible is True
        assert ctx.eligibility["b"].eligible is True

    def test_an_undeclared_entry_is_never_filtered(self):
        ctx = _context(
            [ProviderEntry(name="a"), ProviderEntry(name="b", capabilities=("text",))],
            required=frozenset({"vision"}),
        )

        EligibilityStage().run(ctx)

        assert ctx.eligibility["a"].eligible is True
        assert ctx.eligibility["b"].eligible is False

    def test_the_answer_keeps_what_the_predicate_said(self):
        ctx = _context(
            [ProviderEntry(name="a", capabilities=("text",))],
            required=frozenset({"vision"}),
        )

        EligibilityStage().run(ctx)

        answer = ctx.eligibility["a"]
        assert isinstance(answer, Eligibility)
        assert answer.headroom is None


class TestTheDeclaredCapabilities:
    """The field is validated where it is read, not where it is used."""

    def test_names_are_lowercased(self):
        assert parse_entry_capabilities(["VISION", " text "], label="x") == (
            "vision",
            "text",
        )

    def test_an_empty_list_means_inherit(self):
        """The admin deletes the key for ``[]``; the file agrees."""
        assert parse_entry_capabilities([], label="x") is None

    def test_a_bare_string_is_refused(self):
        with pytest.raises(ValueError, match="must be a list of names"):
            parse_entry_capabilities("vision", label="x")

    def test_a_non_string_member_is_refused(self):
        with pytest.raises(ValueError, match="must be a list of names"):
            parse_entry_capabilities(["text", 5], label="x")

    def test_an_unknown_name_is_refused(self):
        with pytest.raises(ValueError, match="unknown capability visoin"):
            parse_entry_capabilities(["visoin"], label="x")

    def test_typed_entries_use_the_same_rule(self):
        """A typed model's pool is parsed by the same function, so a typo fails there too."""
        with pytest.raises(ValueError, match="unknown capabilit"):
            declared_provider_entries(
                {"providers": [{"name": "a", "capabilities": ["text", "nonsense"]}]}
            )


class TestWhatARequestNeeds:
    """Three facts map to a capability; a plain request needs nothing."""

    def test_an_image_needs_vision(self):
        assert required_capabilities(RuleFacts(images=True)) == frozenset({"vision"})

    def test_tools_need_tools(self):
        assert required_capabilities(RuleFacts(tools=True)) == frozenset({"tools"})

    def test_an_effort_needs_reasoning(self):
        assert required_capabilities(RuleFacts(effort="high")) == frozenset(
            {"reasoning"}
        )

    def test_anthropic_thinking_needs_reasoning(self):
        """``thinking: enabled`` carries no effort, and still needs the lane."""
        facts = facts_from_ir(
            {"messages": [], "reasoning": {"mode": "enabled", "budget_tokens": 1500}}
        )

        assert required_capabilities(facts) == frozenset({"reasoning"})

    def test_adaptive_thinking_needs_reasoning(self):
        facts = facts_from_ir({"messages": [], "reasoning": {"mode": "auto"}})

        assert required_capabilities(facts) == frozenset({"reasoning"})

    def test_thinking_turned_off_needs_nothing(self):
        facts = facts_from_ir({"messages": [], "reasoning": {"mode": "disabled"}})

        assert required_capabilities(facts) == frozenset()

    def test_a_budget_without_a_mode_needs_reasoning(self):
        facts = facts_from_ir({"messages": [], "reasoning": {"budget_tokens": 800}})

        assert required_capabilities(facts) == frozenset({"reasoning"})

    def test_a_plain_request_needs_nothing(self):
        assert required_capabilities(RuleFacts(tokens=100)) == frozenset()

    def test_no_facts_means_nothing(self):
        assert required_capabilities(None) == frozenset()


class TestNothingServesTheRequest:
    """Every candidate declaring the wrong thing is its own state."""

    def test_the_classifier_says_capability(self):
        ctx = _context(
            [ProviderEntry(name="a", capabilities=("text",))],
            required=frozenset({"vision"}),
        )
        EligibilityStage().run(ctx)

        unavailable = classify_unavailable(_bare_config(["a"]), ctx)

        assert unavailable.reason == "capability"
        assert unavailable.unsupported == ("vision",)

    def test_the_exception_names_the_missing_capability(self):
        error = ProviderNotReady("m", unsupported=("vision",))

        assert error.unsupported == ("vision",)
        assert "vision" in str(error)
        assert error.retry_after is None

    def test_a_route_with_one_usable_entry_is_not_capability(self):
        ctx = _context(
            [
                ProviderEntry(name="a", capabilities=("text",)),
                ProviderEntry(name="b"),
            ],
            required=frozenset({"vision"}),
        )
        EligibilityStage().run(ctx)

        unavailable = classify_unavailable(_bare_config(["a", "b"]), ctx)

        assert unavailable.reason != "capability"


class TestTheEdge:
    """The chat route reports a capability refusal as the client's problem."""

    def test_a_capability_refusal_is_a_400_with_no_retry_after(self, monkeypatch):
        import json

        from llm_rosetta_gateway.app.routes_llm import _resolve_or_error

        config = GatewayConfig(
            {
                "providers": {"a": {"type": "openai", "base_url": "https://a.test"}},
                "models": {"m": {"providers": ["a"]}},
            }
        )
        selection = SelectionContext("m", "chat_completions")
        selection.install_route(
            model_route=config.models["m"],
            providers=config.providers,
            provider_types=config.provider_types,
            provider_shim_names=config.provider_shim_names,
            provider_account_tokens=config.provider_account_tokens,
        )

        def _refuse(self, ctx):
            raise ProviderNotReady("m", unsupported=("vision",))

        monkeypatch.setattr(GatewayConfig, "resolve", _refuse)

        response = _resolve_or_error(config, selection, "rid")

        assert isinstance(response, Response)
        assert response.status_code == 400
        assert "Retry-After" not in response.headers
        assert "vision" in json.dumps(json.loads(response.body))


class TestTheAdminWrite:
    """The drawer's field writes the entry, and a blank clears it."""

    def test_setting_capabilities_writes_the_entry(self):
        from llm_rosetta_gateway.admin.routes.config.model_providers import (
            _set_provider_attrs,
        )

        plist = [{"name": "a"}]

        assert _set_provider_attrs(plist, "a", None, None, ["text", "vision"])
        assert plist[0]["capabilities"] == ["text", "vision"]

    def test_a_blank_field_removes_the_declaration(self):
        from llm_rosetta_gateway.admin.routes.config.model_providers import (
            _set_provider_attrs,
        )

        plist = [{"name": "a", "capabilities": ["text"]}]

        assert _set_provider_attrs(plist, "a", None, None, [])
        assert "capabilities" not in plist[0]

    def test_an_unsent_field_leaves_the_entry_alone(self):
        from llm_rosetta_gateway.admin.routes.config.model_providers import (
            _set_provider_attrs,
        )

        plist = [{"name": "a", "capabilities": ["text"]}]

        assert _set_provider_attrs(plist, "a", 3, None)
        assert plist[0]["capabilities"] == ["text"]


class _Upstream:
    """A chat upstream that answers 200 and counts that it was called."""

    def __init__(self) -> None:
        self.hits = 0
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> int:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        return self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, reader: Any, writer: Any) -> None:
        self.hits += 1
        head = await reader.readuntil(b"\r\n\r\n")
        length = 0
        for line in head.decode().split("\r\n"):
            if line.lower().startswith("content-length:"):
                length = int(line.split(":", 1)[1])
        if length:
            await reader.readexactly(length)
        payload = json.dumps(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        ).encode()
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode()
            + payload
        )
        await writer.drain()
        writer.close()


_IMAGE_BODY = {
    "model": "m",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            "data:image/png;base64,"
                            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
                            "2mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
                        )
                    },
                },
            ],
        }
    ],
    "max_tokens": 16,
}


async def _serve_model(tmp_path: Any, entries: list[dict[str, Any]]):
    """A real gateway in front of two recording upstreams."""
    from llm_rosetta_gateway.app import create_app
    from llm_rosetta_gateway.observability import PersistenceManager, RequestLog

    text, vision = _Upstream(), _Upstream()
    text_port, vision_port = await text.start(), await vision.start()
    pm = await PersistenceManager.create(str(tmp_path / "gateway.db"))
    config = GatewayConfig(
        {
            "providers": {
                "textlane": {
                    "type": "chat_completions",
                    "base_url": f"http://127.0.0.1:{text_port}/v1",
                    "api_key": "x",
                },
                "visionlane": {
                    "type": "chat_completions",
                    "base_url": f"http://127.0.0.1:{vision_port}/v1",
                    "api_key": "x",
                },
            },
            "models": {
                "m": {
                    "providers": entries,
                    "routing": {"strategy": "ordered", "affinity": "off", "sink": True},
                }
            },
            "server": {"data_dir": str(tmp_path / "data"), "open_on_no_keys": True},
        }
    )
    app = await create_app(config)
    wired: Any = app
    wired.request_log = RequestLog(persistence=pm)
    wired.persistence = pm
    listener = await asyncio.start_server(app._handle_connection, "127.0.0.1", 0)
    return text, vision, pm, listener


async def _post(port: int, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    raw = json.dumps(body).encode()
    request = (
        f"POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nConnection: close\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(raw)}\r\n\r\n"
    ).encode() + raw
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(request)
    await writer.drain()
    writer.write_eof()
    data = await reader.read()
    writer.close()
    head, _, payload = data.partition(b"\r\n\r\n")
    return int(head.decode().split("\r\n")[0].split()[1]), json.loads(payload or b"{}")


class TestTheEdgeAsksForTheCapabilities:
    """The gate that makes the filter run: the chat handler builds the needs.

    Every unit test above sets ``required_capabilities`` itself, so deleting
    the one line in the chat handler that fills it leaves them green.  This
    goes through the real handler with two declared lanes and asserts which
    upstream answered.
    """

    @pytest.mark.asyncio
    async def test_an_image_skips_the_text_only_lane(self, tmp_path):
        text, vision, pm, listener = await _serve_model(
            tmp_path,
            [
                {"name": "textlane", "capabilities": ["text"]},
                {"name": "visionlane", "capabilities": ["text", "vision"]},
            ],
        )
        try:
            status, _body = await _post(
                listener.sockets[0].getsockname()[1], _IMAGE_BODY
            )
            assert status == 200
            assert (text.hits, vision.hits) == (0, 1)
        finally:
            listener.close()
            await listener.wait_closed()
            await text.stop()
            await vision.stop()
            await pm.close()

    @pytest.mark.asyncio
    async def test_without_a_declaration_the_first_lane_serves(self, tmp_path):
        """The control: no capabilities, and the ordered route keeps lane one."""
        text, vision, pm, listener = await _serve_model(
            tmp_path,
            [{"name": "textlane"}, {"name": "visionlane"}],
        )
        try:
            status, _body = await _post(
                listener.sockets[0].getsockname()[1], _IMAGE_BODY
            )
            assert status == 200
            assert (text.hits, vision.hits) == (1, 0)
        finally:
            listener.close()
            await listener.wait_closed()
            await text.stop()
            await vision.stop()
            await pm.close()
