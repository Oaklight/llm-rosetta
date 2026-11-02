"""Per-request rules: promote one provider, or drop one, for this request.

A rule is a condition and an action.  The condition reads facts about the
request — does it carry images, how large is it, how much reasoning effort
did it ask for, which agent sent it, what time is it where the operator lives
— and the action is one of:

- ``use``: promote a member of the route, so this request is served by it
  when it can serve at all;
- ``pause``: drop a member of the route for this request only.

The first matching rule wins, and a request no rule matches falls through to
the strategy exactly as it would with no rules.  A rule can only name a
provider the model already lists: it may promote or drop, never introduce.

Conditions read the request's **IR**, never the source body, so one rule means
the same thing whichever dialect the client spoke.  Two of the facts are
approximations and say so:

- ``tokens_gte`` compares against an estimate — the serialized IR's size over
  four, with image payloads excluded.  The gateway has no tokenizer, and
  paying for one on every request would cost more than the rule is worth.
- ``agents`` matches the client's ``x-rosetta-agent`` tags and its
  ``User-Agent``, case-insensitively and by substring: the agents do not agree
  on a spelling, and a rule that misses by one character is worse than a rule
  that matches broadly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, time, timezone
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .stage import RoutingStage

if TYPE_CHECKING:
    from .context import SelectionContext

#: The reasoning ladder, weakest first.  Mirrors ``IREffort``.
EFFORT_ORDER = ("minimal", "low", "medium", "high", "xhigh", "max")

#: The condition keys a rule may carry.
CONDITIONS = ("images", "tokens_gte", "effort_gte", "agents", "time")


@dataclass(frozen=True, slots=True)
class Rule:
    """One condition and the action it triggers.

    Attributes:
        when: The parsed condition, keyed by condition name.  Every key must
            hold; an empty condition always matches.
        use: A provider to promote, or ``None``.
        pause: A provider to drop for this request, or ``None``.
    """

    when: dict[str, Any]
    use: str | None = None
    pause: str | None = None

    def describe(self) -> dict[str, Any]:
        """The rule as the admin panel reports it.

        A ``time`` window keeps ``datetime.time`` for matching, and the panel
        reads JSON: the times are rendered back to ``HH:MM`` here rather than
        making the whole route unloadable.
        """
        when = dict(self.when)
        window = when.get("time")
        if isinstance(window, dict):
            when["time"] = {
                "tz": window["tz"],
                "from": window["from"].strftime("%H:%M"),
                "to": window["to"].strftime("%H:%M"),
            }
        return {"when": when, "use": self.use, "pause": self.pause}


@dataclass(frozen=True, slots=True)
class RuleFacts:
    """What a request looks like to a rule.

    Attributes:
        images: Whether the request carries at least one image.
        tokens: Estimated input size, in tokens.
        effort: The reasoning effort the request asked for, if any.
        thinking: Whether the request asked for thinking at all.  An Anthropic
            or Gemini request says so with a mode and a budget, and no effort
            spelling anywhere.
        agents: Tags the client named itself with, uppercased for matching.
        now: The moment to judge ``time`` conditions against; tests pass one.
    """

    images: bool = False
    tokens: int = 0
    effort: str | None = None
    thinking: bool = False
    agents: tuple[str, ...] = ()
    now: datetime | None = None


def parse_rules(raw: Any, *, label: str) -> tuple[Rule, ...]:
    """Parse a model's rule list.

    Args:
        raw: The ``rules`` array, or ``None``.
        label: The config path, for the error message.

    Returns:
        The rules, in the order they are evaluated.

    Raises:
        ValueError: If the array or one of its rules is malformed.
    """
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"config: {label} must be an array, got {raw!r}")
    return tuple(
        _parse_rule(item, label=f"{label}[{index}]") for index, item in enumerate(raw)
    )


def _parse_rule(raw: Any, *, label: str) -> Rule:
    if not isinstance(raw, dict):
        raise ValueError(f"config: {label} must be an object, got {raw!r}")
    unknown = set(raw) - {"when", "use", "pause"}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"config: {label} has unknown key(s): {names}")

    when = raw.get("when", {})
    if not isinstance(when, dict):
        raise ValueError(f"config: {label}.when must be an object, got {when!r}")
    unknown_conditions = set(when) - set(CONDITIONS)
    if unknown_conditions:
        names = ", ".join(sorted(unknown_conditions))
        raise ValueError(
            f"config: {label}.when has unknown condition(s): {names}; "
            f"known conditions: {', '.join(CONDITIONS)}"
        )

    use, pause = raw.get("use"), raw.get("pause")
    if (use is None) == (pause is None):
        raise ValueError(f"config: {label} must name exactly one of 'use' or 'pause'")
    for name, value in (("use", use), ("pause", pause)):
        if value is not None and not isinstance(value, str):
            raise ValueError(f"config: {label}.{name} must be a provider name")

    return Rule(
        when=_parse_condition_set(when, label=f"{label}.when"),
        use=use,
        pause=pause,
    )


def _parse_condition_set(when: dict[str, Any], *, label: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, value in when.items():
        if key == "images":
            if not isinstance(value, bool):
                raise ValueError(f"config: {label}.images must be true or false")
            parsed[key] = value
        elif key == "tokens_gte":
            parsed[key] = _non_negative_int(value, label=f"{label}.tokens_gte")
        elif key == "effort_gte":
            if value not in EFFORT_ORDER:
                raise ValueError(
                    f"config: {label}.effort_gte must be one of "
                    f"{', '.join(EFFORT_ORDER)}; got {value!r}"
                )
            parsed[key] = value
        elif key == "agents":
            if (
                not isinstance(value, list)
                or not value
                or not all(isinstance(item, str) for item in value)
            ):
                raise ValueError(
                    f"config: {label}.agents must be a non-empty array of strings"
                )
            parsed[key] = tuple(item.upper() for item in value)
        elif key == "time":
            parsed[key] = _parse_time(value, label=f"{label}.time")
    return parsed


def _parse_time(raw: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"config: {label} must be an object, got {raw!r}")
    unknown = set(raw) - {"tz", "from", "to"}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"config: {label} has unknown key(s): {names}")
    tz = raw.get("tz")
    if not isinstance(tz, str) or not tz:
        raise ValueError(f"config: {label}.tz must be a time zone name")
    try:
        ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(
            f"config: {label}.tz is not a known time zone: {tz!r}"
        ) from exc
    return {
        "tz": tz,
        "from": _parse_clock(raw.get("from", "00:00"), label=f"{label}.from"),
        "to": _parse_clock(raw.get("to", "23:59"), label=f"{label}.to"),
    }


def _parse_clock(raw: Any, *, label: str) -> time:
    if not isinstance(raw, str):
        raise ValueError(f"config: {label} must be a time of day, e.g. '09:00'")
    try:
        hour, minute = raw.split(":")
        return time(int(hour), int(minute))
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"config: {label} must be a time of day as HH:MM, got {raw!r}"
        ) from exc


def _non_negative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"config: {label} must be an integer, got {value!r}")
    if value < 0:
        raise ValueError(f"config: {label} must be 0 or greater, got {value!r}")
    return value


def matches(rule: Rule, facts: RuleFacts) -> bool:
    """Whether every condition in *rule* holds for *facts*."""
    return all(
        _condition_matches(key, value, facts) for key, value in rule.when.items()
    )


def _condition_matches(key: str, expected: Any, facts: RuleFacts) -> bool:
    if key == "images":
        return facts.images is expected
    if key == "tokens_gte":
        return facts.tokens >= expected
    if key == "effort_gte":
        if facts.effort is not None and facts.effort in EFFORT_ORDER:
            return EFFORT_ORDER.index(facts.effort) >= EFFORT_ORDER.index(expected)
        # A thinking block carries no effort string; asking for thinking at
        # all is at least a "high" request for routing purposes.
        return facts.thinking and EFFORT_ORDER.index("high") >= EFFORT_ORDER.index(
            expected
        )
    if key == "agents":
        return any(tag in agent for tag in expected for agent in _upper(facts.agents))
    if key == "time":
        return _time_matches(expected, facts)
    return False


def _upper(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(value.upper() for value in values)


def _time_matches(window: dict[str, Any], facts: RuleFacts) -> bool:
    moment = (facts.now or datetime.now(timezone.utc)).astimezone(
        ZoneInfo(window["tz"])
    )
    local = moment.hour * 60 + moment.minute
    start = window["from"].hour * 60 + window["from"].minute
    end = window["to"].hour * 60 + window["to"].minute
    if start == end:
        # ``from: 09:00, to: 09:00`` reads as "all day", not as an empty
        # window that silently disables the rule it is attached to.
        return True
    # Both ends are inclusive at their minute, so the default ``to: 23:59``
    # covers the last minute of the day instead of skipping it.
    if start < end:
        return start <= local <= end
    # An overnight window, e.g. 22:00 → 06:00.
    return local >= start or local <= end


def facts_from_ir(
    ir_request: dict[str, Any] | None,
    *,
    agents: tuple[str, ...] = (),
    now: datetime | None = None,
) -> RuleFacts:
    """Derive a rule's facts from a converted request.

    Args:
        ir_request: The source-format request in IR form, or ``None``.
        agents: Tags the client named itself with (headers).
        now: The moment to judge ``time`` conditions against.

    Returns:
        The facts.  An absent or empty request is small, image-free and
        effort-free rather than an error: a rule must not fail a request.
    """
    from llm_rosetta.types.ir.type_guards import is_image_part, is_tool_result_part

    if not ir_request:
        return RuleFacts(agents=agents, now=now)

    images = 0
    images_stripped: list[Any] = []
    for message in ir_request.get("messages") or ():
        parts = message.get("content") or ()
        if not isinstance(parts, list):
            continue
        kept: list[Any] = []
        for part in parts:
            if is_image_part(part):
                images += 1
                kept.append({"type": "image"})
            elif is_tool_result_part(part):
                result = part.get("result")
                if isinstance(result, list) and any(
                    isinstance(block, dict) and is_image_part(block) for block in result
                ):
                    images += 1
                    kept.append({"type": "image"})
                else:
                    kept.append(part)
            else:
                kept.append(part)
        images_stripped.append({"role": message.get("role"), "content": kept})

    # Four characters per token is the usual rough rule; image payloads are
    # excluded because a data: URI is not a token count.  The system prompt
    # and the tool schemas are input the upstream bills for, so they count
    # too: a rule about input size that only sees the conversation misses
    # however many tokens they add.
    counted: dict[str, Any] = {"messages": images_stripped}
    if ir_request.get("system_instruction"):
        counted["system_instruction"] = ir_request["system_instruction"]
    if ir_request.get("tools"):
        counted["tools"] = ir_request["tools"]
    size = len(json.dumps(counted, ensure_ascii=False, separators=(",", ":")))
    reasoning = ir_request.get("reasoning") or {}
    effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
    mode = reasoning.get("mode") if isinstance(reasoning, dict) else None
    budget = reasoning.get("budget_tokens") if isinstance(reasoning, dict) else None
    # An OpenAI request spells thinking as an effort; a thinking block spells
    # it as a mode and a budget.  Both mean "this request thinks".
    thinking = mode in ("auto", "enabled") or (mode != "disabled" and bool(budget))
    return RuleFacts(
        images=images > 0,
        tokens=size // 4,
        effort=effort if isinstance(effort, str) else None,
        thinking=thinking,
        agents=agents,
        now=now,
    )


class RulesStage(RoutingStage):
    """Apply the model's rules to the provider the earlier stages chose.

    Runs after the anchor and the same-format tiers, so a rule composes with
    them: it starts from the provider routing would have picked and moves the
    request within the model's own route.  A ``use`` rule promotes a member
    that may serve; a ``pause`` rule drops one and re-runs the selection over
    what is left.  Neither can introduce a provider the model does not list.

    The first matching rule wins and the rest are not consulted; no match
    leaves the anchor standing.
    """

    __slots__ = ()

    name = "rules"
    requires = frozenset(
        {
            "model_route",
            "identity",
            "eligibility",
            "quota_excluded",
            "failover_excluded",
            "rule_facts",
        }
    )
    provides = frozenset({"chosen"})

    def run(self, ctx: SelectionContext) -> None:
        """Apply the first matching rule, if any."""
        if ctx.chosen is None or ctx.model_route is None:
            return
        rules = ctx.model_route.routing.rules
        facts = ctx.rule_facts
        if not rules or facts is None:
            return
        for rule in rules:
            if matches(rule, facts):
                self._apply(ctx, rule)
                return

    @staticmethod
    def _apply(ctx: SelectionContext, rule: Rule) -> None:
        from .select import select_ready_entry

        assert ctx.model_route is not None  # checked by run()
        if rule.use is not None:
            # A provider this request already failed over from is not a
            # candidate for the retry: promoting it again spends the failover
            # budget on the lane that just refused.
            if rule.use in ctx.failover_excluded:
                return
            answer = ctx.eligibility.get(rule.use)
            if answer is None or not answer.eligible:
                # The rule named a provider that cannot serve this request;
                # the anchor stands rather than the request failing.
                return
            for entry in ctx.model_route.providers:
                if entry.name == rule.use:
                    ctx.choose(entry)
                    return
            return

        paused = rule.pause
        assert paused is not None  # the parser guarantees exactly one action
        excluded = ctx.excluded_providers() | {paused}
        entry = select_ready_entry(
            ctx.model_route,
            eligibility=ctx.eligibility,
            identity=ctx.identity.client,
            excluded=excluded,
        )
        if entry is not None:
            ctx.choose(entry)
