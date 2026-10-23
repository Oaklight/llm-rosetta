"""Wire suite — one canonical request, round-tripped through every chat format.

The matrix this starts from is the translation one: IR → target body → IR.  A
probe writes one canonical request in one format, reads the body back, and
compares the two IRs.  What the round trip loses is reported in the case's
data; what makes a case **fail** is losing something the conversion did not
report — a difference no conversion warning accounts for.

That distinction is the point.  A loss the caller is told about is a
documented limit of a format; a loss nobody hears about is the bug class the
conversion-warning sink (#1009) exists to end, and it is the one assertion a
suite can make without first knowing what every format supports.

"Accounts for" is decided on the warning's ``subject``, not on its prose: a
warning explains a loss when the subject it carries *is* the field (or the
container holding it).  Prose is a fallback for the reporters that carry no
subject yet, and only for a value that vanished — a changed value needs a
report that names it, because "something mentioned this word" is not
evidence that anything expected the change.

The suite is offline on purpose: it probes the *conversion* surface, so it
needs no credential and is deterministic enough to diff against a baseline.
It probes each format against the same IR (five identity round trips), not
the 25 ordered format pairs; upstream wire behaviour — what a vendor accepts
and echoes — is the next family of cases and needs the scheduled,
credentialled run the epic plans.
"""

from __future__ import annotations

from typing import Any

from llm_rosetta import get_converter_for_provider
from llm_rosetta.conversion_warnings import WarningSink, recording_to
from llm_rosetta.converters.base.context import ConversionContext

from ..model import Probe, ProbeFailure

#: The five chat formats, in catalog order.
FORMATS: tuple[str, ...] = (
    "chat_completions",
    "open_responses",
    "messages",
    "google_generate",
    "google_interactions",
)

#: One request every chat format can express: instructions, a user turn, one
#: function tool, automatic tool choice, and the sampling controls.  Kept to
#: the shared core so a difference in the round trip means the format is not
#: carrying something it should, not that the request asked for an extension.
CANONICAL_REQUEST: dict[str, Any] = {
    "model": "conformance-model",
    "system_instruction": [{"type": "text", "text": "You are terse."}],
    "messages": [
        {
            "role": "user",
            "content": [{"type": "text", "text": "What is the weather in Paris?"}],
        }
    ],
    "tools": [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Look up the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ],
    "tool_choice": {"mode": "auto"},
    "generation": {"temperature": 0.2, "top_p": 0.9, "max_tokens": 64},
}

#: The same request with every sampling knob an IR request can carry.  The
#: canonical case above exercises the shared surface; this one is where a
#: field a format does not have is most likely to vanish unnoticed, so it is
#: the case that most needs the "reported or failed" rule.
SAMPLING_REQUEST: dict[str, Any] = {
    **CANONICAL_REQUEST,
    "generation": {
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 20,
        "max_tokens": 64,
        "stop_sequences": ["END"],
        "seed": 7,
        "logprobs": True,
        "top_logprobs": 2,
    },
}

#: ``(case label, request)`` pairs every format is probed with.
CASES: tuple[tuple[str, dict[str, Any]], ...] = (
    ("request", CANONICAL_REQUEST),
    ("sampling", SAMPLING_REQUEST),
)

#: Values a writer's own defaults materialise on the way back.  A reader that
#: fills in ``tool_name: ""`` or ``metadata: {}`` has lost nothing, so the
#: comparison ignores them on both sides.
_DEFAULTS: tuple[Any, ...] = ("", None, [], {})


def project(shape: Any, value: Any) -> Any:
    """Keep *value*'s fields that *shape* names, recursively.

    A reader materialises fields the request never had — a synthesised
    ``provider_metadata``, a denormalised ``required_parameters``, a
    ``strict`` default — and those are bookkeeping, not information the round
    trip carried.  Comparing against the canonical request's own shape asks
    the question that matters: did every field I sent come back unchanged?
    """
    if isinstance(shape, dict) and isinstance(value, dict):
        return {
            key: project(item, value[key])
            for key, item in shape.items()
            if key in value
        }
    if isinstance(shape, list) and isinstance(value, list):
        if not shape:
            return value
        return [project(shape[0], item) for item in value]
    return value


def normalize(value: Any) -> Any:
    """Drop the empty defaults a round trip materialises, recursively.

    Applied after the children are normalized, so a container that held only
    defaults goes too — ``[{"metadata": {}}]`` becomes ``[]``.
    """
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized = normalize(item)
            if normalized not in _DEFAULTS:
                result[key] = normalized
        return result
    if isinstance(value, list):
        items = [normalize(item) for item in value]
        return [item for item in items if item not in _DEFAULTS]
    return value


def differences(expected: Any, actual: Any, path: str = "") -> list[str]:
    """Return the paths where *actual* differs from *expected*.

    Both sides are normalized first, so only a value that changed or
    disappeared is reported.  A list is compared item by item, which is what
    makes a reordered or truncated tool list visible.
    """
    left, right = normalize(expected), normalize(actual)
    if isinstance(left, dict) and isinstance(right, dict):
        found: list[str] = []
        for key in sorted(set(left) | set(right)):
            where = f"{path}.{key}" if path else key
            if key not in right:
                # Recurse against an empty container so the loss is named at
                # the leaf, not swallowed by the container that held it.
                if isinstance(left[key], (dict, list)):
                    found.extend(differences(left[key], type(left[key])(), where))
                else:
                    found.append(f"{where}: lost")
            elif key not in left:
                if isinstance(right[key], (dict, list)):
                    found.extend(differences(type(right[key])(), right[key], where))
                else:
                    found.append(f"{where}: added by the round trip")
            else:
                found.extend(differences(left[key], right[key], where))
        return found
    if isinstance(left, list) and isinstance(right, list):
        found = []
        if len(left) != len(right):
            found.append(f"{path or '<root>'}: {len(left)} → {len(right)} items")
        for index, (one, other) in enumerate(zip(left, right)):
            found.extend(differences(one, other, f"{path}[{index}]"))
        return found
    if left != right:
        return [f"{path or '<root>'}: {left!r} → {right!r}"]
    return []


def _path_and_change(loss: str) -> tuple[str, bool]:
    """Split a :func:`differences` entry into its path and whether it changed."""
    path, _, what = loss.partition(": ")
    return path, "→" in what


def explained(loss: str, warnings: list[Any]) -> bool:
    """Whether a conversion warning accounts for *loss*.

    The warning's **subject** has to name the field: it explains the loss when
    it is the field's own name, or a prefix of the path — a reporter that
    names the container it emptied has named what went missing.  Prose is a
    fallback, and only for reporters that carry no subject yet.  A *changed*
    value never takes the prose route: a warning that merely mentions the word
    is not evidence the change was expected, so it needs a subject that names
    the field exactly.
    """
    path, changed = _path_and_change(loss)
    leaf = path.rsplit(".", 1)[-1].split("[")[0]
    if not leaf:
        return False
    for warning in warnings:
        subject = str(getattr(warning, "subject", "") or "")
        if subject:
            if subject == leaf or subject == path or path.startswith(subject + "."):
                return True
            continue
        if not changed and leaf in str(warning):
            return True
    return False


def _warnings_data(warnings: list[Any], leg: str) -> list[dict[str, str]]:
    """Render the sink of one direction for the report."""
    return [
        {
            "leg": leg,
            "code": getattr(warning, "code", ""),
            "subject": getattr(warning, "subject", ""),
            "message": str(warning),
        }
        for warning in warnings
    ]


class _RoundTripProbe:
    """One format's canonical round trip."""

    def __init__(
        self, provider: str, request: dict[str, Any], case: str = "request"
    ) -> None:
        self.name = f"{case}/{provider}"
        self._provider = provider
        self._request = request

    def run(self) -> dict[str, Any]:
        converter = get_converter_for_provider(self._provider)
        ctx = ConversionContext()

        # One sink per direction, and the ambient sink installed for each call
        # exactly as ConversionPipeline installs it: most converters report
        # through ``report_drop`` without a ``warnings=`` argument, so a probe
        # that called them bare would see a reported loss as unreported.
        written = WarningSink(leg=self._provider)
        with recording_to(written):
            body, _ = converter.request_to_provider(dict(self._request), context=ctx)
        read = WarningSink(leg=self._provider)
        ctx.warnings = read
        with recording_to(read):
            restored = converter.request_from_provider(body, context=ctx)

        lost = differences(self._request, project(self._request, restored))
        warnings = [*written, *read]
        silent = [loss for loss in lost if not explained(loss, warnings)]
        data = {
            "provider": self._provider,
            "lost": lost,
            "warnings": _warnings_data(written, "write") + _warnings_data(read, "read"),
        }
        if silent:
            raise ProbeFailure(
                f"{self._provider} round trip loses {len(silent)} "
                f"field(s) without a warning: " + "; ".join(silent)
            )
        return data


def build() -> tuple[list[Probe], dict[str, Any]]:
    """Build the suite: every case, in every chat format."""
    probes: list[Probe] = [
        _RoundTripProbe(provider, request, case)
        for case, request in CASES
        for provider in FORMATS
    ]
    return probes, {"formats": list(FORMATS), "cases": [case for case, _ in CASES]}
