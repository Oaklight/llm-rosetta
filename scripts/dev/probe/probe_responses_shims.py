#!/usr/bin/env python3
"""Probe ``open_responses`` shims against their live upstreams.

A shim's ``base: open_responses`` claims the vendor-neutral Open Responses
spec.  The OpenAI profile layered on top of that spec is opt-in and declared
per shim through ``response_id_prefix``, ``store_default`` and
``preserve_fields``.  Whether a provider actually behaves like that profile is
a per-provider question that has to be settled against the real endpoint:
copying the OpenAI shim's block is not evidence.

This script sends a minimal ``POST /responses`` to every configured
``open_responses`` shim that has a resolvable credential, and prints what the
upstream actually does next to what the shim claims:

- the response id, and whether it carries ``resp_``
- whether ``store: true`` is accepted, rejected, ignored or never echoed
- which OpenAI lifecycle fields come back, and which of them are non-null

See issue #899.  Providers whose credential is missing, or whose upstream has
no open Responses endpoint for the configured account, are reported as skipped
rather than silently dropped.

Disagreements are split by strength, on one rule: a *mismatch* is a claim the
shim makes that the upstream contradicts, and nothing else.  Two qualify — a
declared id prefix that does not appear, and a ``store_default`` the upstream
rejects, ignores, or never echoes.  Everything else is a *note*, including
every check about a declaration the shim *omits*, because the probe's own
request is not evidence about what a shim should declare: it sends
``store: true`` deliberately, so an upstream echoing it shows that the upstream
supports store, not that the shim ought to inject a default.  Whether either
omission check fires also depends on which model ``discover_model`` happened to
pick, which is not a property of the shim — a multi-vendor proxy legitimately
serves ``resp_...`` from one model family and bare UUIDs from another, so it has
no single correct value to declare.  A declared ``preserve_fields`` entry that
did not appear in this one sample is a note for the same reason: for a
conditional field that is not proof, and an over-declared field is inert
anyway.

Usage::

    python scripts/dev/probe/probe_responses_shims.py
    python scripts/dev/probe/probe_responses_shims.py --shim deepseek--open_responses
    python scripts/dev/probe/probe_responses_shims.py --model deepseek--open_responses=deepseek-chat
    python scripts/dev/probe/probe_responses_shims.py --json results.json
    python scripts/dev/probe/probe_responses_shims.py --proxy socks5://127.0.0.1:1082

Exit status is 1 when any probed shim has a mismatch, so the script is usable
as a manual gate in the #899 follow-up loop.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from llm_rosetta._vendor import httpclient
from llm_rosetta.shims.provider_shim import ProviderShim, list_shims
from llm_rosetta.shims.providers import load_providers

# Lifecycle/echo fields the OpenAI Responses profile emits on top of the spec.
OPENAI_LIFECYCLE_FIELDS: tuple[str, ...] = (
    "access_programs",
    "billing",
    "content_filters",
    "moderation",
    "tool_usage",
)

PROMPT = "Reply with the single word: ok"
TIMEOUT = 60.0
ENV_PATH = Path(__file__).resolve().parents[3] / ".env"

# Model ids that are never usable for a chat probe: prefixes that only audio,
# image and legacy completion models carry, plus substrings that mark a
# non-text modality wherever they appear (``gpt-4o-transcribe``,
# ``gpt-live-transcribe``, ...).
_NON_CHAT_PREFIXES = (
    "babbage",
    "dall-e",
    "davinci",
    "gpt-image",
    "omni-moderation",
    "sora",
    "text-embedding",
    "text-moderation",
    "tts-",
    "whisper",
)
_NON_CHAT_SUBSTRINGS = ("audio", "embed", "realtime", "transcribe")

# Session id sent under a shim-declared ``connection.session_header``.  The
# upstream only needs a stable opaque value for routing.
PROBE_SESSION_ID = "llm-rosetta-responses-probe"

_ABSENT = "<absent>"


def load_env() -> None:
    """Load the repo-root ``.env`` into ``os.environ`` without overwriting."""
    if not ENV_PATH.exists():
        return
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)


def responses_shims(only: str | None = None) -> list[ProviderShim]:
    """Return registered shims whose base is the neutral Responses spec.

    Args:
        only: Optional shim name to restrict the result to.

    Returns:
        The matching shims, in registry order.
    """
    load_providers()
    found = [s for s in list_shims() if s.base == "open_responses"]
    if only:
        found = [s for s in found if s.name == only]
    return found


def resolve_api_key(shim: ProviderShim) -> tuple[str, str | None]:
    """Return the credential for *shim* and the env var it came from.

    Uses the shim's declared ``api_key_env`` first, then falls back to the
    ``<VENDOR>_API_KEY`` spelling, because a local ``.env`` does not always use
    the name the shim publishes.

    Args:
        shim: Shim whose credential is resolved.

    Returns:
        ``(api_key, env_var_name)``.  Both are empty/None when no credential is
        configured.
    """
    declared = shim.connection.api_key_env
    candidates = [declared] if declared else []
    vendor = shim.name.split("--")[0].replace("-", "_").upper()
    candidates.append(f"{vendor}_API_KEY")
    for name in candidates:
        if name and os.environ.get(name):
            return os.environ[name], name
    return "", declared


def auth_headers(shim: ProviderShim, api_key: str) -> dict[str, str]:
    """Build request headers for *shim* from *api_key*.

    Mirrors the gateway convention: a bearer token under ``Authorization``
    unless the shim names a custom ``auth_header``, in which case the key is
    sent under that header verbatim.

    Args:
        shim: Shim whose connection config supplies the header convention.
        api_key: Credential to send, if any.

    Returns:
        Headers to use for the probe requests.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        if shim.connection.auth_header:
            headers[shim.connection.auth_header] = api_key
        else:
            headers["Authorization"] = f"Bearer {api_key}"
    if shim.connection.session_header:
        headers[shim.connection.session_header] = PROBE_SESSION_ID
    return headers


def _usable_model_id(model_id: str) -> bool:
    """Return True when *model_id* can serve a chat probe.

    Args:
        model_id: Candidate model id from the upstream model list.

    Returns:
        ``False`` for embedding, image, audio and moderation models.
    """
    lowered = model_id.lower()
    if any(hint in lowered for hint in _NON_CHAT_SUBSTRINGS):
        return False
    return not lowered.startswith(_NON_CHAT_PREFIXES)


def discover_model(
    client: httpclient.Client, shim: ProviderShim, headers: dict[str, str]
) -> str | None:
    """Return a chat-capable model id from the shim's model list.

    Args:
        client: Shared HTTP client.
        shim: Shim whose models endpoint is queried.
        headers: Auth headers for the upstream.

    Returns:
        The first usable model id, or ``None`` when the list cannot be fetched
        or contains no usable entry.
    """
    base = (shim.connection.base_url or "").rstrip("/")
    path = shim.connection.models_path or f"{base}/models"
    try:
        response = client.get(path, headers=headers)
    except httpclient.HttpClientError:
        return None
    if response.status_code != 200:
        return None
    try:
        data = response.json()
    except ValueError:
        return None
    entries = data.get("data") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        return None
    field = shim.connection.model_id_field or "id"
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get(field)
        if isinstance(model_id, str) and model_id and _usable_model_id(model_id):
            return model_id
    return None


def probe_once(
    client: httpclient.Client,
    base_url: str,
    headers: dict[str, str],
    model: str,
    store: bool,
) -> dict[str, Any]:
    """Send one minimal ``/responses`` request and summarise the reply.

    Args:
        client: Shared HTTP client.
        base_url: Upstream base URL, without a trailing slash.
        headers: Auth headers for the upstream.
        model: Model id to request.
        store: Value of the ``store`` field to send.

    Returns:
        A dict with ``status`` and ``ok``, plus ``id``, ``store`` and
        ``fields`` on success.  Failures carry an ``error`` string instead.
    """
    body = {
        "model": model,
        "input": PROMPT,
        "max_output_tokens": 16,
        "store": store,
    }
    try:
        response = client.post(f"{base_url}/responses", json=body, headers=headers)
    except httpclient.HttpClientError as exc:
        return {"ok": False, "status": None, "error": f"{type(exc).__name__}: {exc}"}

    result: dict[str, Any] = {"ok": False, "status": response.status_code}
    if response.status_code != 200:
        result["error"] = response.text[:200]
        return result
    try:
        data = response.json()
    except ValueError:
        result["error"] = "non-JSON body"
        return result
    if not isinstance(data, dict) or "id" not in data:
        # Some gateways answer 200 with an error envelope; treat that as a
        # failure rather than reading a non-existent response resource.
        result["error"] = f"200 without a response resource: {json.dumps(data)[:200]}"
        return result

    result["ok"] = True
    result["id"] = data.get("id")
    result["store"] = data.get("store", _ABSENT)
    result["fields"] = {k: data.get(k, _ABSENT) for k in OPENAI_LIFECYCLE_FIELDS}
    return result


def analyse(
    shim: ProviderShim, observed: dict[str, Any], store_state: str
) -> tuple[list[str], list[str]]:
    """Compare what the shim declares against what the upstream does.

    Args:
        shim: Shim whose declared profile traits are checked.
        observed: Successful probe result carrying ``id``, ``store`` and
            ``fields``.
        store_state: Classification from :func:`classify_store`.

    Returns:
        ``(mismatches, notes)`` — traits the probe contradicts, then weaker
        observations that one sample cannot settle.
    """
    mismatches: list[str] = []
    notes: list[str] = []
    response_id = str(observed.get("id") or "")
    has_prefix = response_id.startswith("resp_")

    if shim.response_id_prefix and not has_prefix:
        mismatches.append(
            f"declares response_id_prefix {shim.response_id_prefix!r} but the "
            f"upstream id is {response_id!r}"
        )
    if has_prefix and not shim.response_id_prefix:
        # A note, not a mismatch.  A declared prefix is a claim about every
        # response, so contradicting it is a defect; the absence of a
        # declaration is not, because a multi-vendor proxy legitimately serves
        # ``resp_...`` from one model family and bare UUIDs from another and so
        # has no correct value to declare.  Whether this fires also depends on
        # which model ``discover_model`` happened to pick, which is not a
        # property of the shim.
        notes.append(
            f"upstream returned a resp_-prefixed id ({response_id!r}) but the "
            "shim declares no prefix; correct for a multi-vendor proxy, worth "
            "checking otherwise"
        )

    if shim.store_default is True and store_state != "accepted":
        if store_state == "absent":
            mismatches.append(
                "declares store_default: true but the upstream response carries "
                "no store field"
            )
        else:
            mismatches.append(
                f"declares store_default: true but the upstream {store_state} it"
            )
    if shim.store_default is None and store_state == "accepted":
        # A note, for the same reason as the undeclared-prefix branch above.
        # The probe sends ``store: true``, so an upstream echoing it shows the
        # upstream *supports* store, not that the shim ought to inject a
        # default; and whether this fires depends on which model
        # ``discover_model`` picked, not on the shim.  OpenCode is the live
        # example: it serves OpenAI-family models, so a correct shim could be
        # reported here and fail the run.
        notes.append("upstream honors store but no store_default is declared")

    # ``preserve_fields`` is recorded as a note, never a mismatch: an
    # over-declared field is inert (absent fields are never captured, and null
    # ones are filtered out), so it cannot corrupt output the way a wrong
    # ``store_default`` or id prefix does.  It still matters for a reader
    # deciding what the shim claims about the provider.
    fields = observed.get("fields") or {}
    declared = list(shim.preserve_fields)
    if declared:
        absent = [k for k in declared if fields.get(k) == _ABSENT]
        null = [k for k in declared if fields.get(k) is None]
        if absent:
            notes.append(f"declared fields not seen in this sample: {absent}")
        if null:
            notes.append(f"declared fields returned only as null: {null}")
    return mismatches, notes


def classify_store(accepted: dict[str, Any]) -> str:
    """Classify how the upstream handles ``store: true``.

    Args:
        accepted: Probe result for ``store: true``.

    Returns:
        ``"rejected"``, ``"accepted"``, ``"ignored"``, ``"absent"`` or
        ``"unavailable"``.
    """
    if accepted.get("status") == 400:
        return "rejected"
    if not accepted.get("ok"):
        return "unavailable"
    value = accepted.get("store")
    if value is True:
        return "accepted"
    if value is False:
        return "ignored"
    return "absent"


def probe_shim(
    client: httpclient.Client, shim: ProviderShim, override: str
) -> dict[str, Any]:
    """Probe a single shim and return its full result record.

    Args:
        client: Shared HTTP client.
        shim: Shim to probe.
        override: Model id to force, if any.

    Returns:
        A record holding the declared traits, the raw observations, and either
        ``mismatches``/``notes`` or a ``skipped`` explanation.
    """
    record: dict[str, Any] = {
        "shim": shim.name,
        "base_url": shim.connection.base_url,
        "declared": {
            "response_id_prefix": shim.response_id_prefix,
            "store_default": shim.store_default,
            "preserve_fields": list(shim.preserve_fields),
        },
    }

    api_key, env_name = resolve_api_key(shim)
    if not api_key:
        record["skipped"] = f"{env_name or 'credential'} is not set"
        return record
    record["api_key_env"] = env_name

    base_url = (shim.connection.base_url or "").rstrip("/")
    if not base_url:
        record["skipped"] = "shim declares no base_url"
        return record
    headers = auth_headers(shim, api_key)

    model = override or discover_model(client, shim, headers)
    if not model:
        record["skipped"] = "no model id discovered; pass --model"
        return record
    record["model"] = model

    accepted = probe_once(client, base_url, headers, model, store=True)
    baseline = probe_once(client, base_url, headers, model, store=False)
    record["store_true"] = accepted
    record["store_false"] = baseline
    record["store_state"] = classify_store(accepted)

    # A rejected ``store: true`` still leaves the baseline usable for the id
    # and lifecycle-field checks.
    observed = accepted if accepted.get("ok") else baseline
    if not observed.get("ok"):
        record["skipped"] = observed.get("error", "probe failed")
        return record
    mismatches, notes = analyse(shim, observed, record["store_state"])
    record["mismatches"] = mismatches
    record["notes"] = notes
    return record


def main() -> int:
    """Run the probe and print a markdown report.

    Returns:
        ``0`` when every probed shim is clean, ``1`` on any mismatch, ``2``
        when the shim filter matched nothing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shim", help="probe only this shim name")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="SHIM=MODEL",
        help="override the model used for a shim (repeatable)",
    )
    parser.add_argument("--json", dest="json_path", help="write raw results here")
    parser.add_argument(
        "--timeout", type=float, default=TIMEOUT, help="per-request timeout in seconds"
    )
    parser.add_argument(
        "--proxy",
        help=(
            "route requests through this proxy, e.g. "
            "socks5://127.0.0.1:1082 for an 'ssh -N -f ts-build01-socks' "
            "tunnel when an upstream is only reachable from another host"
        ),
    )
    args = parser.parse_args()

    overrides = dict(item.split("=", 1) for item in args.model if "=" in item)
    load_env()

    shims = responses_shims(args.shim)
    if not shims:
        print(f"No open_responses shim matched {args.shim!r}", file=sys.stderr)
        return 2

    with httpclient.Client(timeout=args.timeout, proxy=args.proxy) as client:
        rows = [
            probe_shim(client, shim, overrides.get(shim.name, "")) for shim in shims
        ]

    print("| shim | model | id | store: true | echo fields (non-null) | verdict |")
    print("|---|---|---|---|---|---|")
    for record in rows:
        if "skipped" in record:
            reason = str(record["skipped"]).replace("|", "/").replace("\n", " ")
            print(f"| {record['shim']} | - | - | - | - | skipped: {reason[:70]} |")
            continue
        observed = (
            record["store_true"]
            if record["store_true"].get("ok")
            else record["store_false"]
        )
        fields = observed.get("fields") or {}
        non_null = [k for k, v in fields.items() if v not in (_ABSENT, None)]
        if record["mismatches"]:
            verdict = "MISMATCH"
        elif record["notes"]:
            verdict = "note"
        else:
            verdict = "ok"
        print(
            f"| {record['shim']} | {record['model']} | {observed.get('id')} "
            f"| {record['store_state']} | {', '.join(non_null) or '-'} | {verdict} |"
        )

    for record in rows:
        if not record.get("mismatches") and not record.get("notes"):
            continue
        print(f"\n{record['shim']}:")
        for reason in record.get("mismatches", []):
            print(f"  - MISMATCH: {reason}")
        for note in record.get("notes", []):
            print(f"  - note: {note}")

    if args.json_path:
        Path(args.json_path).write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nRaw results written to {args.json_path}", file=sys.stderr)

    mismatched = [r for r in rows if r.get("mismatches")]
    return 1 if mismatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
