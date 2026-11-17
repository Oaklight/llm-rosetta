"""Live suite — what each vendor accepts, and whether its answer reads back.

Every other suite is offline: it compares our own conversion against a
baseline.  This one leaves the machine.  For each of the five chat formats it
builds the request the converter emits, sends it to the vendor that format
belongs to, and asserts the two things a scheduled run can assert without a
baseline:

- the vendor accepts the body we emit (HTTP 200, JSON in the dialect), and
- its answer still reads back through our reader into IR with text in it.

A vendor that rejects the body — a ``400`` naming a field the body carries —
is a **failure**: that is the request this library ships.  A credential the
vendor refuses (``401``/``403``), a ``400`` that names the credential or the
model, a ``404`` for a model that moved, a rate limit, a 5xx or a transport
error is an **error**: the probe could not run, which is a fact about the run
rather than about the dialect.  A lane whose credential, model or base URL is
not configured is **skipped**, so a run that holds two keys out of five is a
partial sweep rather than a broken one.

Two things the first measured run settled, and both are load-bearing:

- The request asks for a few hundred output tokens, not a few dozen.  The
  reasoning models answer with an empty text part when a 32-token budget is
  spent on the thought, which is a truncated answer rather than a rejection.
- ``google_generate`` is probed with ``output_format="rest"``, the shape the
  gateway sends.  The converter's default ``sdk`` shape nests sampling under
  ``config``, which the Generative Language endpoint rejects with HTTP 400
  ``Unknown name "config"``.

Lanes and their environment contract (the CI job maps secrets and repository
variables onto the same names):

| lane | credential | model | base |
| --- | --- | --- | --- |
| ``chat_completions/openai`` | ``OPENAI_API_KEY`` | ``OPENAI_MODEL`` | ``OPENAI_BASE_URL`` |
| ``open_responses/openai`` | ``OPENAI_RESPONSES_API_KEY`` | ``OPENAI_RESPONSES_MODEL`` | ``OPENAI_RESPONSES_BASE_URL`` |
| ``messages/anthropic`` | ``ANTHROPIC_API_KEY`` | ``ANTHROPIC_MODEL`` | ``ANTHROPIC_BASE_URL`` |
| ``google_generate/google`` | ``GOOGLE_API_KEY`` | ``GOOGLE_MODEL`` | ``GOOGLE_BASE_URL`` |
| ``google_interactions/google`` | ``GOOGLE_API_KEY`` | ``GOOGLE_MODEL`` | ``GOOGLE_BASE_URL`` |

The Google lanes default their base URL to
``https://generativelanguage.googleapis.com``; the others have none, because
their base carries an API version and a vendor prefix and a wrong default
would quietly probe the wrong surface.

Run one sweep locally:

.. code-block:: bash

    set -a; . ./.env; set +a
    python -m tools.conformance run --suite live --secret-env OPENAI_API_KEY \\
        --secret-env ANTHROPIC_API_KEY --secret-env GOOGLE_API_KEY \\
        --out /tmp/live-report.json
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from collections.abc import Callable

from llm_rosetta import get_converter_for_provider
from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.formats.catalog import BY_PAIR

from ..model import Probe, ProbeFailure, ProbeSkip

#: A transport: ``(url, headers, body, timeout) -> (status, text)``.  Injected
#: in tests so the suite can be pinned without a network.
Post = Callable[[str, dict[str, str], dict[str, Any], float], tuple[int, str]]

#: Default upstream of the Google lanes.  The other lanes have no default on
#: purpose: their base URL carries an API version and a vendor prefix, and a
#: hard-coded one would quietly probe the wrong surface.
_GOOGLE_BASE = "https://generativelanguage.googleapis.com"

#: The question.  It must be answerable in one word, so an empty answer is the
#: vendor's doing rather than the prompt's.
PROMPT = "Reply with the single word: ok"

#: Output-token budget.  A reasoning model spends its budget on the thought
#: before it writes anything, so a small number yields a text-free answer.
MAX_TOKENS = 256


@dataclass(frozen=True, slots=True)
class Lane:
    """One ``(format, vendor)`` lane the sweep probes.

    Attributes:
        name: Case name in the report, ``<format>/<vendor>``.
        format: Converter format key, also the catalog pair's format.
        key_env: Environment variable holding the credential.
        model_env: Environment variable holding the model name.
        base_env: Environment variable holding the base URL; ``None`` when the
            lane has a fixed default.
        default_base: Base URL used when *base_env* is unset.
    """

    name: str
    format: str
    key_env: str
    model_env: str
    base_env: str | None = None
    default_base: str | None = None


#: The five chat formats, one vendor each — the vendor the format is named for.
LANES: tuple[Lane, ...] = (
    Lane(
        name="chat_completions/openai",
        format="chat_completions",
        key_env="OPENAI_API_KEY",
        model_env="OPENAI_MODEL",
        base_env="OPENAI_BASE_URL",
    ),
    Lane(
        name="open_responses/openai",
        format="open_responses",
        key_env="OPENAI_RESPONSES_API_KEY",
        model_env="OPENAI_RESPONSES_MODEL",
        base_env="OPENAI_RESPONSES_BASE_URL",
    ),
    Lane(
        name="messages/anthropic",
        format="messages",
        key_env="ANTHROPIC_API_KEY",
        model_env="ANTHROPIC_MODEL",
        base_env="ANTHROPIC_BASE_URL",
    ),
    Lane(
        name="google_generate/google",
        format="google_generate",
        key_env="GOOGLE_API_KEY",
        model_env="GOOGLE_MODEL",
        base_env="GOOGLE_BASE_URL",
        default_base=_GOOGLE_BASE,
    ),
    Lane(
        name="google_interactions/google",
        format="google_interactions",
        key_env="GOOGLE_API_KEY",
        model_env="GOOGLE_MODEL",
        base_env="GOOGLE_BASE_URL",
        default_base=_GOOGLE_BASE,
    ),
)


def _auth_headers(auth: str | None, key: str) -> dict[str, str]:
    """The headers a format's auth builder declares, as plain HTTP.

    Args:
        auth: The catalog record's ``auth`` key.
        key: The credential.

    Returns:
        Request headers carrying the credential.

    Raises:
        ValueError: The catalog names an auth builder this probe does not know.
    """
    if auth == "openai":
        return {"Authorization": f"Bearer {key}"}
    if auth == "anthropic":
        return {"x-api-key": key, "anthropic-version": "2023-06-01"}
    if auth == "google":
        return {"x-goog-api-key": key}
    raise ValueError(f"no credential headers for auth builder {auth!r}")


def _post_json(
    url: str, headers: dict[str, str], body: dict[str, Any], timeout: float
) -> tuple[int, str]:
    """Send *body* as JSON and return ``(status, response text)``.

    An HTTP error is a status like any other; only a transport failure raises,
    and the runner reports that as an error case.
    """
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def _answer_text(response: dict[str, Any]) -> str:
    """The text parts of an IR response, joined."""
    parts: list[str] = []
    for choice in response.get("choices", []):
        for part in choice.get("message", {}).get("content") or []:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
    return "".join(parts)


def _id_prefix(identifier: Any) -> str | None:
    """The id family a vendor uses (``resp``, ``chatcmpl``), or ``None``."""
    if not isinstance(identifier, str) or not identifier:
        return None
    head, sep, _ = identifier.partition("_")
    return head if sep else None


#: Words that make a ``400``/``404`` about the run rather than about the body
#: we sent: a revoked or mistyped key, a permission, a model that moved.  The
#: weekly job meets these more often than a wrong field, and reporting them as
#: "the vendor rejected our request" would name the wrong culprit.
_RUN_FAULT_MARKERS: tuple[str, ...] = (
    "api key",
    "api_key",
    "api-key",
    "credential",
    "unauthorized",
    "unauthenticated",
    "authentication",
    "permission",
    "does not exist",
    "not found",
    "not_found",
    "unknown model",
    "model_not_found",
    "invalid model",
)


def _is_request_fault(status: int, text: str, body: dict[str, Any]) -> bool:
    """Whether a non-200 answer blames the body this probe sent.

    Only a ``400`` can: the fail bucket means "the request this library ships
    is wrong", which the vendor says by naming a field the body carries.  A
    ``400`` that names the credential or the model, a ``404``, a rate limit and
    a 5xx are the run's own state, and the error bucket reports those.

    ``model`` is not a field of the body this *library* chooses: the harness
    fills it from the lane's ``*_MODEL`` variable, so "Invalid value for
    'model'" is the run's configuration and not a dialect fault.  It is left
    out of the scan, which sends a message naming it to the error bucket the
    same way a marker does.

    Args:
        status: The HTTP status the vendor answered with.
        text: The response body.
        body: The request body this probe sent.

    Returns:
        ``True`` when the answer is about a field of *body* that the
        converter, not the run, chose.
    """
    if status != 400:
        return False
    lowered = text.lower()
    if any(marker in lowered for marker in _RUN_FAULT_MARKERS):
        return False
    fields = (str(key).lower() for key in body)
    return any(field in lowered for field in fields if field != "model")


def _request_body(lane: Lane, model: str) -> dict[str, Any]:
    """The body the lane's converter emits for the minimal request.

    ``google_generate`` is built in the REST shape the gateway sends; the
    converter's default SDK shape is not a wire format.
    """
    context = ConversionContext()
    if lane.format == "google_generate":
        context.options["output_format"] = "rest"
    request = {
        "model": model,
        "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}],
        "generation": {"max_tokens": MAX_TOKENS},
    }
    converter = get_converter_for_provider(lane.format)
    body, _ = converter.request_to_provider(request, context=context)
    return body


class _LaneProbe:
    """One lane's acceptance sweep."""

    name: str

    def __init__(
        self,
        lane: Lane,
        *,
        post: Post | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._lane = lane
        self._post = post or _post_json
        self._timeout = timeout
        self.name = lane.name

    def _resolve(self) -> tuple[str, str, str]:
        """The credential, model and base URL, or a skip naming what is absent."""
        lane = self._lane
        key = os.environ.get(lane.key_env, "")
        model = os.environ.get(lane.model_env, "")
        base = (
            (os.environ.get(lane.base_env, "") if lane.base_env else "")
            or lane.default_base
            or ""
        )
        missing = [
            label
            for label, value in (
                (lane.key_env, key),
                (lane.model_env, model),
                (lane.base_env or "base_url", base),
            )
            if not value
        ]
        if missing:
            raise ProbeSkip(f"{self.name}: not configured ({', '.join(missing)} unset)")
        return key, model, base.rstrip("/")

    def run(self) -> dict[str, Any]:
        key, model, base = self._resolve()
        spec = BY_PAIR[("llm", self._lane.format)]
        if spec.url_template is None or spec.auth is None:
            raise ValueError(
                f"{self.name}: the catalog entry has no URL or auth builder"
            )
        url = spec.url_template.format(base_url=base, model=model)
        body = _request_body(self._lane, model)
        status, text = self._post(
            url, _auth_headers(spec.auth, key), body, self._timeout
        )
        if status in (401, 403):
            raise RuntimeError(
                f"{self.name}: the credential was refused (HTTP {status})"
            )
        if status != 200:
            if _is_request_fault(status, text, body):
                raise ProbeFailure(
                    f"{self.name}: the vendor rejected the request "
                    f"(HTTP {status}): {text[:300]}"
                )
            raise RuntimeError(
                f"{self.name}: the probe could not run (HTTP {status}): {text[:300]}"
            )
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProbeFailure(
                f"{self.name}: the answer is not JSON: {exc}: {text[:300]}"
            ) from exc
        converter = get_converter_for_provider(self._lane.format)
        try:
            response = converter.response_from_provider(payload)
        except Exception as exc:  # noqa: BLE001 - the answer did not read back
            raise ProbeFailure(
                f"{self.name}: the answer does not read back "
                f"({type(exc).__name__}: {exc})"
            ) from exc
        choices = response.get("choices") or []
        answer = _answer_text(response)
        finish = choices[0].get("finish_reason", {}).get("reason") if choices else None
        if not choices or not answer.strip():
            raise ProbeFailure(
                f"{self.name}: the answer carries no text "
                f"(choices={len(choices)}, finish={finish!r}, "
                f"chars={len(answer)})"
            )
        return {
            "http_status": status,
            "requested_model": model,
            "echoed_model": response.get("model"),
            "id_prefix": _id_prefix(payload.get("id")),
            "finish_reason": finish,
            "text_chars": len(answer),
            "usage_keys": sorted((response.get("usage") or {}).keys()),
        }


def configured_lanes() -> list[str]:
    """The lane names whose credential, model and base are all present."""
    names = []
    for lane in LANES:
        probe = _LaneProbe(lane)
        try:
            probe._resolve()
        except ProbeSkip:
            continue
        names.append(lane.name)
    return names


def build() -> tuple[list[Probe], dict[str, Any]]:
    """Build the suite: one probe per lane, plus the environment it sees."""
    probes: list[Probe] = [_LaneProbe(lane) for lane in LANES]
    return probes, {
        "lanes": [lane.name for lane in LANES],
        "configured": configured_lanes(),
    }
