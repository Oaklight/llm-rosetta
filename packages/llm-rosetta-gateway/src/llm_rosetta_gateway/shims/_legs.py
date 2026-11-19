"""Fetch legs (primary and mirror): planning, failure handling and URL helpers."""

from __future__ import annotations

import json
import os
from urllib.parse import quote, urlsplit


from ._types import (
    HTTP_TIMEOUT,
    HttpGet,
    ShimArchiveUnusable,
    ShimSourceError,
    ShimSourceUnreachable,
    _SHA_RE,
    logger,
)
from ._cache import (
    _chosen_legs,
    _chosen_legs_lock,
)
from ._spec import (
    is_supported_archive_url,
    _RemoteSpec,
)


def _mirror_slot(declared: str | None, resolved: str | None) -> str | None:
    """Which ref the mirror leg's ``{ref}`` is filled with.

    A declared ref wins: a tag-addressed mirror publishes one directory per
    release, and that spelling must not depend on api.github.com answering.
    With no declared ref the source tracks its branch, and the mirror is
    addressed by the commit the forge resolved — the pinned commit and the
    mirror's bytes then name the same tree, which the old ``sha256`` pair used
    to guarantee.  ``None`` leaves the placeholder as ``HEAD``, which the
    caller's URL builder turns into a leg that 404s rather than a wrong tree.

    Args:
        declared: The remote's ``ref``, when the operator pinned one.
        resolved: The commit a GitHub primary resolved to, when it could.

    Returns:
        The value for the mirror's ``{ref}``.
    """
    return declared or resolved


def _leg_plan(
    spec: _RemoteSpec,
    primary_url: str,
    *,
    ref: str | None,
    mirror_ref: str | None = None,
) -> list[tuple[str, str, str | None]]:
    """Return the legs to try, in order, as ``(leg, url, sha256 pin)``.

    *ref* fills the primary's ``{ref}`` (the resolved commit when the forge
    resolved one); *mirror_ref* fills the mirror's, defaulting to the same
    value.  The caller passes the declared ref for the mirror so its URL does
    not depend on the forge's API.

    The leg that last served this source is tried first, so a mirror chosen
    because the primary was unreachable is not re-probed on every fetch.  If
    the remembered leg fails with a transport error, the other leg is still
    tried — the mirror may be down while the primary has recovered.
    """
    legs: list[tuple[str, str, str | None]] = [("primary", primary_url, spec.sha256)]
    if spec.mirror_url:
        mirror_slot = ref if mirror_ref is None else mirror_ref
        legs.append(
            (
                "mirror",
                spec.mirror_url.replace("{ref}", mirror_slot or "HEAD"),
                spec.mirror_sha256,
            )
        )
    with _chosen_legs_lock:
        chosen = _chosen_legs.get(spec.source_dir.name)
    if chosen == "mirror" and len(legs) == 2:
        legs.reverse()
    return legs


def _log_leg_failure(
    spec: _RemoteSpec,
    url: str,
    exc: BaseException,
    unreachable: bool,
    next_leg: tuple[str, str, str | None],
) -> None:
    """Say which leg failed and which one is about to be tried."""
    logger.warning(
        "Shim source %s: %s %s (%s); trying the %s %s",
        spec.url,
        url,
        "unreachable" if unreachable else "failed",
        exc,
        next_leg[0],
        next_leg[1],
    )


def _exhausted_error(
    exc: BaseException,
    *,
    spec: _RemoteSpec,
    leg: str,
    url: str,
    failures: list[str],
    all_unreachable: bool,
    unusable: bool = False,
) -> ShimSourceError:
    """The error for a leg that failed and has no untried leg behind it.

    A failure of every leg with every one of them a transport failure is
    :class:`ShimSourceUnreachable`; a body that could not be installed is
    :class:`ShimArchiveUnusable`; anything else is a plain source error.  When
    more than one leg failed the message names each.

    The type is the rule: ``_resolve_remote_source`` re-raises the two
    "the bytes are wrong" classes instead of degrading to the cache, and
    degrades everything else.
    """
    if all_unreachable:
        if len(failures) == 1:
            where = url if url == spec.url else f"{url} (source {spec.url})"
            return ShimSourceUnreachable(
                f"cannot reach {where}: {type(exc).__name__}: {exc}"
            )
        return ShimSourceUnreachable(
            f"no leg reachable for source {spec.url}: " + "; ".join(failures)
        )
    error_type: type[ShimSourceError] = (
        ShimArchiveUnusable if unusable else ShimSourceError
    )
    if len(failures) > 1:
        return error_type(
            f"no usable leg for source {spec.url}: " + "; ".join(failures)
        )
    return error_type(f"{leg} {url}: {type(exc).__name__}: {exc}")


def _is_unreachable(exc: BaseException) -> bool:
    """Whether *exc* is a transport failure — no response from the server.

    Only this class of failure falls back to the mirror.  An HTTP status is a
    response from the source itself, and a bad archive is the source's
    verdict; neither is a network symptom to route around.

    The transport subclasses are named rather than taking ``OSError`` whole:
    what is left of ``OSError`` here is local (a permission error, a path of
    the wrong type), which says nothing about whether the host answers.
    """
    from llm_rosetta_gateway._vendor.httpclient import (
        HTTPError,
        HttpConnectionError,
        HttpTimeoutError,
    )

    if isinstance(exc, HTTPError):
        return False
    return isinstance(
        exc, (HttpConnectionError, HttpTimeoutError, ConnectionError, TimeoutError)
    )


def _leg_headers(
    headers: dict[str, str],
    *,
    leg: str,
    url: str,
    primary_url: str,
    spec: _RemoteSpec,
) -> dict[str, str]:
    """Request headers for one leg.

    A bearer token belongs to the origin that declared it.  The primary's
    ``token_env`` therefore rides only on a leg whose origin is the primary's:
    a mirror on another host never sees it, and one that needs a credential
    declares its own ``token_env``.
    """
    token_env = spec.token_env
    if leg == "mirror":
        if spec.mirror_token_env:
            token_env = spec.mirror_token_env
        elif not _same_origin(url, primary_url):
            token_env = None
    leg_headers = {
        name: value
        for name, value in headers.items()
        if name.lower() != "authorization"
    }
    if token_env:
        leg_headers.update(_bearer_headers(token_env))
    return leg_headers


def _same_origin(first: str, second: str) -> bool:
    """Whether two URLs share a scheme, host and port."""
    left, right = _url_origin(first), _url_origin(second)
    return left is not None and left == right


#: The port a URL scheme implies when it names none.
_DEFAULT_PORTS = {"http": 80, "https": 443}


def _url_origin(url: str) -> tuple[str, str, int | None] | None:
    """The ``(scheme, host, port)`` an Authorization header would reach.

    A scheme's default port is filled in, so ``https://h`` and ``https://h:443``
    are one origin: writing the default explicitly must not cost the credential.
    """
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    if not parts.hostname:
        return None
    scheme = parts.scheme.lower()
    if port is None:
        port = _DEFAULT_PORTS.get(scheme)
    return scheme, parts.hostname.lower(), port


def _archive_url(url: str, ref: str | None, github: tuple[str, str] | None) -> str:
    """Translate a declared URL into the archive URL to GET."""
    if github is not None:
        owner, repo = github
        return f"https://codeload.github.com/{owner}/{repo}/tar.gz/{ref or 'HEAD'}"
    if "{ref}" in url:
        return url.replace("{ref}", ref or "HEAD")
    if is_supported_archive_url(url):
        return url
    raise ShimSourceError(
        "not a GitHub repository URL and not an archive URL; give an explicit "
        "archive URL (optionally containing '{ref}') or a GitHub repo URL"
    )


def _resolve_github_head(
    owner: str,
    repo: str,
    *,
    ref: str | None = None,
    headers: dict[str, str],
    http_get: HttpGet,
    proxy: str | None,
) -> str:
    """Resolve a GitHub ref (default: HEAD) to a concrete commit SHA."""
    target = quote(ref, safe="") if ref else "HEAD"
    api = f"https://api.github.com/repos/{owner}/{repo}/commits/{target}"
    data = http_get(api, headers=headers, timeout=HTTP_TIMEOUT, proxy=proxy)
    payload = json.loads(bytes(data).decode("utf-8"))
    sha = payload.get("sha")
    if not isinstance(sha, str) or not _SHA_RE.match(sha):
        raise ShimSourceError("could not resolve HEAD to a commit SHA")
    return sha


def _auth_headers(token_env: str | None) -> dict[str, str]:
    """Build request headers, adding a bearer token when configured."""
    headers = {"User-Agent": "llm-rosetta-shim-loader", "Accept": "*/*"}
    if token_env:
        headers.update(_bearer_headers(token_env))
    return headers


def _bearer_headers(token_env: str) -> dict[str, str]:
    """The ``Authorization`` header for *token_env*, or an error when it is unset."""
    token = os.environ.get(token_env)
    if not token:
        raise ShimSourceError(f"token_env {token_env!r} is not set in the environment")
    return {"Authorization": f"Bearer {token}"}
