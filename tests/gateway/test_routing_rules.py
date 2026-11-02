"""Per-request rules: one condition, one action, first match wins.

A rule moves a request within the model's own route.  These tests pin the
parse (one condition per documented kind, and the errors for the rest), each
condition's evaluation against facts, and the two actions end to end through
the selection chain.
"""

from __future__ import annotations

import json

from datetime import datetime, timezone

import pytest

from llm_rosetta_gateway.config import GatewayConfig
from llm_rosetta_gateway.routing import (
    EFFORT_ORDER,
    Rule,
    RuleFacts,
    facts_from_ir,
    matches,
    parse_rules,
)

NOW = datetime(2026, 11, 2, 3, 0, tzinfo=timezone.utc)


def _facts(**kwargs) -> RuleFacts:
    return RuleFacts(**kwargs)


def _rule(when, **action) -> Rule:
    return parse_rules([{"when": when, **action}], label="r")[0]


class TestTheParse:
    def test_an_absent_list_is_no_rules(self):
        assert parse_rules(None, label="r") == ()

    def test_every_documented_condition_parses(self):
        rules = parse_rules(
            [
                {"when": {"images": True}, "use": "a"},
                {"when": {"tokens_gte": 120000}, "use": "a"},
                {"when": {"effort_gte": "high"}, "use": "a"},
                {"when": {"agents": ["codex"]}, "use": "a"},
                {
                    "when": {
                        "time": {"tz": "Asia/Shanghai", "from": "09:00", "to": "18:00"}
                    },
                    "pause": "a",
                },
            ],
            label="r",
        )

        assert [rule.use or rule.pause for rule in rules] == [
            "a",
            "a",
            "a",
            "a",
            "a",
        ]
        assert all(isinstance(rule.when, dict) for rule in rules)

    def test_an_unknown_condition_is_an_error_not_a_miss(self):
        with pytest.raises(ValueError, match="unknown condition"):
            parse_rules([{"when": {"colour": "blue"}, "use": "a"}], label="r")

    def test_a_rule_needs_exactly_one_action(self):
        with pytest.raises(ValueError, match="exactly one of 'use' or 'pause'"):
            parse_rules([{"when": {}}], label="r")
        with pytest.raises(ValueError, match="exactly one of 'use' or 'pause'"):
            parse_rules([{"when": {}, "use": "a", "pause": "b"}], label="r")

    def test_an_unknown_rule_key_is_refused(self):
        with pytest.raises(ValueError, match="unknown key"):
            parse_rules([{"when": {}, "use": "a", "after": 1}], label="r")

    @pytest.mark.parametrize(
        "when, message",
        [
            ({"images": "yes"}, "images must be true or false"),
            ({"tokens_gte": -1}, "0 or greater"),
            ({"tokens_gte": "many"}, "must be an integer"),
            ({"effort_gte": "extreme"}, "effort_gte must be one of"),
            ({"agents": []}, "non-empty array"),
            ({"time": {"tz": "Mars/Base"}}, "not a known time zone"),
            ({"time": {"tz": "UTC", "from": "9am"}}, "HH:MM"),
        ],
    )
    def test_a_malformed_condition_is_refused(self, when, message):
        with pytest.raises(ValueError, match=message):
            parse_rules([{"when": when, "use": "a"}], label="r")


class TestTheConditions:
    def test_images_matches_both_ways(self):
        assert matches(_rule({"images": True}, use="a"), _facts(images=True))
        assert not matches(_rule({"images": True}, use="a"), _facts(images=False))
        assert matches(_rule({"images": False}, use="a"), _facts(images=False))

    def test_tokens_gte_is_inclusive(self):
        rule = _rule({"tokens_gte": 100}, use="a")

        assert matches(rule, _facts(tokens=99)) is False
        assert matches(rule, _facts(tokens=100)) is True

    def test_effort_gte_follows_the_ladder(self):
        rule = _rule({"effort_gte": "high"}, use="a")

        assert matches(rule, _facts(effort="high")) is True
        assert matches(rule, _facts(effort="max")) is True
        assert matches(rule, _facts(effort="medium")) is False
        # An unstated effort is not a claim of "high".
        assert matches(rule, _facts(effort=None)) is False

    def test_every_ladder_step_is_ordered(self):
        assert EFFORT_ORDER == ("minimal", "low", "medium", "high", "xhigh", "max")

    def test_agents_matches_a_substring_case_insensitively(self):
        rule = _rule({"agents": ["codex"]}, use="a")

        assert matches(rule, _facts(agents=("codex-cli/1.2",))) is True
        assert matches(rule, _facts(agents=("Mozilla/5.0 Codex",))) is True
        assert matches(rule, _facts(agents=("opencode",))) is False

    def test_time_window_is_local_to_the_zone(self):
        rule = _rule(
            {"time": {"tz": "Asia/Shanghai", "from": "09:00", "to": "18:00"}}, pause="a"
        )

        # 03:00 UTC is 11:00 in Shanghai.
        assert matches(rule, _facts(now=NOW)) is True
        # 16:00 UTC is midnight in Shanghai.
        late = datetime(2026, 11, 2, 16, 0, tzinfo=timezone.utc)
        assert matches(rule, _facts(now=late)) is False

    def test_an_overnight_window_wraps(self):
        rule = _rule({"time": {"tz": "UTC", "from": "22:00", "to": "06:00"}}, pause="a")

        assert matches(rule, _facts(now=datetime(2026, 11, 2, 23, tzinfo=timezone.utc)))
        assert matches(rule, _facts(now=datetime(2026, 11, 2, 5, tzinfo=timezone.utc)))
        assert not matches(
            rule, _facts(now=datetime(2026, 11, 2, 12, tzinfo=timezone.utc))
        )

    def test_a_window_whose_ends_meet_covers_the_whole_day(self):
        """``from: 09:00, to: 09:00`` reads as all day, not as an empty window."""
        rule = _rule({"time": {"tz": "UTC", "from": "09:00", "to": "09:00"}}, pause="a")

        for hour in (0, 8, 9, 12, 23):
            moment = datetime(2026, 11, 2, hour, tzinfo=timezone.utc)
            assert matches(rule, _facts(now=moment)) is True

    def test_effort_gte_matches_a_thinking_request(self):
        """Anthropic-style thinking has no effort string, and is still heavy."""
        rule = _rule({"effort_gte": "high"}, use="a")
        thinking = facts_from_ir(
            {"messages": [], "reasoning": {"mode": "enabled", "budget_tokens": 1500}}
        )

        assert matches(rule, thinking) is True

    def test_effort_gte_does_not_match_thinking_that_is_off(self):
        rule = _rule({"effort_gte": "high"}, use="a")
        off = facts_from_ir({"messages": [], "reasoning": {"mode": "disabled"}})

        assert matches(rule, off) is False

    def test_thinking_does_not_claim_the_top_of_the_ladder(self):
        rule = _rule({"effort_gte": "max"}, use="a")
        thinking = facts_from_ir({"messages": [], "reasoning": {"mode": "enabled"}})

        assert matches(rule, thinking) is False

    def test_the_default_end_covers_the_last_minute(self):
        rule = _rule({"time": {"tz": "UTC"}}, pause="a")
        late = datetime(2026, 11, 2, 23, 59, tzinfo=timezone.utc)

        assert matches(rule, _facts(now=late)) is True

    def test_every_condition_in_one_rule_must_hold(self):
        rule = _rule({"images": True, "tokens_gte": 100}, use="a")

        assert matches(rule, _facts(images=True, tokens=500)) is True
        assert matches(rule, _facts(images=True, tokens=50)) is False

    def test_an_empty_condition_always_matches(self):
        assert matches(_rule({}, use="a"), _facts()) is True


class TestTheFacts:
    def test_images_are_detected_in_message_content(self):
        ir = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {"type": "image", "url": "data:image/png;base64,AAAA"},
                    ],
                }
            ]
        }

        assert facts_from_ir(ir).images is True

    def test_an_image_inside_a_tool_result_counts(self):
        ir = {
            "messages": [
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_call_id": "c1",
                            "result": [{"type": "image", "url": "data:,"}],
                        }
                    ],
                }
            ]
        }

        assert facts_from_ir(ir).images is True

    def test_image_payloads_do_not_inflate_the_estimate(self):
        text_only = {
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        }
        with_image = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hi"},
                        {
                            "type": "image",
                            "url": "data:image/png;base64," + "A" * 100000,
                        },
                    ],
                }
            ]
        }

        assert facts_from_ir(with_image).tokens < facts_from_ir(text_only).tokens + 50

    def test_effort_comes_from_the_ir(self):
        ir = {"messages": [], "reasoning": {"effort": "high"}}

        assert facts_from_ir(ir).effort == "high"

    def test_the_system_prompt_and_tools_count_as_input(self):
        """A rule about input size must see what the upstream bills for."""
        ir = {
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "system_instruction": [{"type": "text", "text": "x" * 40_000}],
            "tools": [{"name": "t", "description": "y" * 20_000, "parameters": {}}],
        }

        facts = facts_from_ir(ir)

        # 60k characters of system prompt and tools is ~15k tokens on its own.
        assert facts.tokens > 10_000

    def test_no_ir_is_small_and_empty(self):
        facts = facts_from_ir(None)

        assert facts == RuleFacts()


class TestTheStage:
    @staticmethod
    def _config(rules, *, names=("a", "b"), strategy=None):
        routing = {"rules": rules}
        if strategy is not None:
            routing["strategy"] = strategy
        return GatewayConfig(
            {
                "providers": {
                    name: {
                        "type": "chat_completions",
                        "api_key": f"sk-{name}",
                        "base_url": f"https://{name}.test",
                    }
                    for name in names
                },
                "models": {"m": {"providers": list(names), "routing": routing}},
            }
        )

    @staticmethod
    def _pick(config, facts):
        selection = config.selection_context("m", "chat_completions")
        selection.with_rule_facts(facts)
        config.resolve(selection)
        assert selection.chosen is not None
        return selection.chosen.name

    def test_a_use_rule_promotes_its_member(self):
        config = self._config([{"when": {"images": True}, "use": "b"}])

        assert self._pick(config, RuleFacts(images=True)) == "b"

    def test_a_rule_that_does_not_match_falls_through(self):
        config = self._config([{"when": {"images": True}, "use": "b"}])

        assert self._pick(config, RuleFacts()) == "a"

    def test_a_pause_rule_drops_its_member(self):
        config = self._config([{"when": {"images": True}, "pause": "a"}])

        assert self._pick(config, RuleFacts(images=True)) == "b"

    def test_the_first_match_wins(self):
        config = self._config(
            [
                {"when": {"images": True}, "use": "a"},
                {"when": {"tokens_gte": 0}, "use": "b"},
            ]
        )

        assert self._pick(config, RuleFacts(images=True)) == "a"

    def test_a_time_rule_describes_itself_as_json(self):
        """The panel reads this through ``json.dumps``: no ``datetime.time``."""
        rule = _rule(
            {"time": {"tz": "Asia/Shanghai", "from": "09:00", "to": "18:00"}},
            pause="a",
        )

        described = json.loads(json.dumps(rule.describe()))

        assert described["when"]["time"] == {
            "tz": "Asia/Shanghai",
            "from": "09:00",
            "to": "18:00",
        }

    def test_a_pause_rule_drops_an_ordered_anchor(self):
        """Ordered always picks lane one, so the exclusion is what moves it."""
        config = self._config([{"when": {}, "pause": "a"}], strategy="ordered")
        selection = config.selection_context("m", "chat_completions")
        selection.with_rule_facts(RuleFacts())

        config.resolve(selection)

        assert selection.chosen is not None and selection.chosen.name == "b"

    def test_a_rule_cannot_introduce_a_provider(self):
        with pytest.raises(ValueError, match="never introduce one"):
            self._config([{"when": {}, "use": "zzz"}])

    def test_a_use_rule_does_not_re_promote_a_failed_over_provider(self):
        """The retry must not spend its budget on the lane that just refused."""
        config = self._config([{"when": {}, "use": "b"}])
        selection = config.selection_context("m", "chat_completions")
        selection.with_rule_facts(RuleFacts())
        selection.with_failover_excluded(frozenset({"b"}))

        config.resolve(selection)

        assert selection.chosen is not None and selection.chosen.name == "a"

    def test_a_use_rule_does_not_promote_an_ineligible_member(self):
        """A rule is a preference; an unavailable provider is not a choice."""
        from llm_rosetta_gateway.quota.state import QUOTA_STATE

        config = self._config([{"when": {"images": True}, "use": "b"}])
        QUOTA_STATE.block_provider("@never-used", 30, "usage limit")
        config.providers["b"].quota_identity = "@rule-ineligible"
        config.providers["b"].quota_provider_blocked = lambda: True

        assert self._pick(config, RuleFacts(images=True)) == "a"

    def test_a_model_without_rules_never_reads_the_facts(self):
        config = self._config([])
        selection = config.selection_context("m", "chat_completions")
        # No facts recorded at all: the stage must not need them.
        config.resolve(selection)
        assert selection.chosen is not None
