"""Credential sanitization for upstream error bodies and headers.

Framework-agnostic: any proxy that logs upstream errors can redact credentials
without importing the gateway.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_AUTH_HEADER_NAMES = frozenset(
    {
        "authorization",
        "x-api-key",
        "x-goog-api-key",
        "proxy-authorization",
    }
)

_CREDENTIAL_PATTERNS = re.compile(
    r"|".join(
        [
            r"Bearer\s+[A-Za-z0-9_\-./+]+",
            r"sk-ant-[A-Za-z0-9_-]{20,}",
            r"sk-[A-Za-z0-9_-]{20,}",
            r"AIza[A-Za-z0-9_-]{30,}",
            # A JWT: three base64url segments, the first always decoding to
            # ``{"`` — hence the ``eyJ`` prefix.  OAuth access and ID tokens
            # are shaped this way, so the pattern a bearer-token upstream is
            # most likely to echo was the one shape the others missed.
            r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}",
        ]
    )
)


# A credential-named field in JSON, a form or a header-ish line.  Matching the
# text rather than a parsed document is the point: the body it is applied to
# is one that failed to parse.  The name must stand alone (``max_tokens`` is
# not ``token``), and the value is a quoted string or a bare run.
_SECRET_FIELD = re.compile(
    r"""(?P<name>(?<![A-Za-z0-9_-])["']?
        (?:x-)?(?:api[_-]?key|apikey|authorization|access[_-]?token|
        refresh[_-]?token|id[_-]?token|auth[_-]?token|token|secret|
        client[_-]?secret|password|passwd|credentials?|
        (?:set-)?cookie|session[_-]?id)
        ["']?\s*[:=]\s*)
        (?P<value>"(?:[^"\\]|\\.)*"?|'[^']*'?|
        (?:(?:bearer|basic|digest|negotiate|token)[ \t]+)?
        (?!\[REDACTED\])[^\s,;&}\]]+)""",
    re.IGNORECASE | re.VERBOSE,
)

# A cookie header line carries several ``name=value`` pairs after one name, so
# it is redacted to the end of the line rather than to the first separator.
_COOKIE_LINE = re.compile(r"(?im)^(?P<name>[ \t]*(?:set-)?cookie[ \t]*:[ \t]*)[^\r\n]+")

# The same holds for ``Authorization``: a Digest or AWS4-HMAC-SHA256 credential
# is a list of comma-separated parameters, and stopping at the first comma left
# the proof parameter (``response=``, ``Signature=``) in the stored body.  A
# quoted value is taken whole; an unquoted one — a header line pasted into a
# body, or a form field — is taken to the end of its line, because where one
# parameter ends cannot be told without parsing the scheme.
_AUTH_VALUE = re.compile(
    r"""(?im)
    (?P<name>(?<![A-Za-z0-9_-])["']?(?:proxy-)?authorization["']?[ \t]*[:=][ \t]*)
    (?P<value>
        "(?:[^"\\]|\\.)*"
      | [^\r\n]+
    )""",
    re.VERBOSE,
)


def _redact_auth(match: re.Match[str]) -> str:
    """Keep the field name; replace its value, quoting it if it was quoted."""
    value = match.group("value")
    return match.group("name") + ('"[REDACTED]"' if value[:1] == '"' else "[REDACTED]")


def scrub_body_text(text: str, secrets: Iterable[str] = ()) -> str:
    """Redact credentials from the text of a request body that did not parse.

    Three layers, each catching what the others miss: an ``Authorization``,
    ``Proxy-Authorization`` or cookie line redacted whole, because a Digest or
    signature credential is a list of parameters rather than one token; the
    value of any field named like a credential (``api_key``, ``token``,
    ``secret``, ``password``…, with an auth scheme such as ``Bearer`` or
    ``Basic`` taken as part of the value); every string in *secrets* verbatim
    (the keys the gateway itself holds, whatever shape the vendor gave them);
    and the vendor-shaped patterns of :func:`scrub_credential_patterns`.  A
    credential that is none of these — an arbitrary string under an arbitrary
    field name, not configured here — is not recognised.
    """
    text = _AUTH_VALUE.sub(_redact_auth, text)
    text = _SECRET_FIELD.sub(
        lambda m: (
            m.group("name") + '"[REDACTED]"'
            if m.group("value")[:1] in "\"'"
            else m.group("name") + "[REDACTED]"
        ),
        text,
    )
    text = _COOKIE_LINE.sub(lambda m: m.group("name") + "[REDACTED]", text)
    for secret in sorted({s for s in secrets if s}, key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    return scrub_credential_patterns(text)


def sanitize_headers(headers: dict[str, str]) -> dict[str, str]:
    """Return a copy of *headers* with auth values redacted."""
    result = {}
    for k, v in headers.items():
        if k.lower() in _AUTH_HEADER_NAMES:
            result[k] = "[REDACTED]"
        else:
            result[k] = v
    return result


def scrub_credential_patterns(text: str) -> str:
    """Replace credential-shaped strings in *text* with ``[REDACTED]``."""
    return _CREDENTIAL_PATTERNS.sub("[REDACTED]", text)


def sanitize_upstream_error(raw: str | bytes) -> str | bytes:
    """Scrub credentials from an upstream error body, preserving type."""
    if isinstance(raw, bytes):
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw
        return scrub_credential_patterns(decoded).encode("utf-8")
    return scrub_credential_patterns(raw)
