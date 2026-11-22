"""Turn a nightly shim-drift run into tracker state, without becoming noise.

The nightly job exists because a package can drift, or stop publishing, and
nothing in this repository notices.  A red check in the Actions tab is not
noticing, so a failure opens an issue — and the hard part is not opening it,
it is not opening it again every night.

The rule this encodes:

* **One issue, ever.** It is found by its fixed title; a second is never
  created while one is open.
* **Speak only when the verdict changes.** The body carries a fingerprint of
  the problem set, so an unchanged failure costs one API read and no comment.
* **Close only on a green run of the timer.** A one-off ``workflow_dispatch``
  rehearsal must not close an issue that is still real.

The fingerprint lives in the issue body as an HTML comment, so it survives a
lost cache and needs no state file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

#: One open issue carries the whole story; finding it by title needs no label.
ISSUE_TITLE = "shim drift: the published packages disagree with this repository"

#: The line that carries the fingerprint, invisible when the issue is rendered.
FINGERPRINT_MARKER = "<!-- shim-drift-fingerprint: {value} -->"


@dataclass(frozen=True)
class Verdict:
    """What the checks said, in the form an issue can carry.

    Attributes:
        problems: One line per problem, already trimmed and sorted.
        fingerprint: A hash of *problems*, stable across runs and orderings.
    """

    problems: tuple[str, ...]
    fingerprint: str


def read_problems(text: str) -> tuple[str, ...]:
    """Extract the problem lines from a check log.

    Only ``problem`` lines matter: ``warning`` is the pre-rollout state of a
    package that has no copy yet, and ``ok`` is the quiet case.  Duplicates and
    blank lines are dropped, and the result is sorted so two runs that saw the
    same trouble in a different order fingerprint identically.

    Args:
        text: The combined output of the check scripts.

    Returns:
        The problem lines, sorted.
    """
    lines = {
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("problem ")
    }
    return tuple(sorted(lines))


def verdict(problems: tuple[str, ...]) -> Verdict:
    """Fingerprint a problem set.

    Args:
        problems: The problem lines.

    Returns:
        The verdict, with a fingerprint over the sorted lines.
    """
    digest = hashlib.sha256("\n".join(problems).encode("utf-8")).hexdigest()
    return Verdict(problems=problems, fingerprint=digest)


def previous_fingerprint(body: str) -> str | None:
    """The fingerprint an issue body carries, if it carries one."""
    marker = "<!-- shim-drift-fingerprint: "
    start = body.find(marker)
    if start == -1:
        return None
    end = body.find("-->", start)
    if end == -1:
        return None
    return body[start + len(marker) : end].strip()


def action_for(*, current: Verdict, issue_body: str | None, scheduled: bool) -> str:
    """Decide what to do with the tracker, given the current verdict.

    Args:
        current: What the checks said this run.
        issue_body: The open issue's body, or ``None`` when none is open.
        scheduled: Whether this run came from the timer rather than a dispatch.

    Returns:
        One of ``create``, ``update``, ``close``, ``nothing``.
    """
    if not current.problems:
        if issue_body is not None and scheduled:
            return "close"
        return "nothing"
    if issue_body is None:
        return "create"
    if previous_fingerprint(issue_body) == current.fingerprint:
        return "nothing"
    return "update"


def render_body(current: Verdict, *, run_url: str | None) -> str:
    """The issue body: what is wrong now, and how to check it.

    Args:
        current: The verdict to describe.
        run_url: The run that produced it, when the caller knows one.

    Returns:
        The Markdown body, fingerprint included.
    """
    lines = [
        "The nightly `shim-drift` check found the published shim packages out of "
        "step with this repository.",
        "",
        "```",
        *current.problems,
        "```",
        "",
        "Fix: add the missing source or vendor repository to `SHIM_SOURCES` in "
        "`llm_rosetta_gateway/shims/catalog.py`, or update the repository URL "
        "the catalogue names.",
        "",
        FINGERPRINT_MARKER.format(value=current.fingerprint),
    ]
    if run_url:
        lines.insert(1, f"Run: {run_url}")
    return "\n".join(lines) + "\n"


def _api(method: str, url: str, token: str, payload: dict | None = None) -> dict | list:
    """Call the GitHub API and return the decoded answer.

    Args:
        method: The HTTP method.
        url: The full URL.
        token: A token with ``issues: write`` on the repository.
        payload: The JSON body, when the method takes one.

    Returns:
        The decoded response.

    Raises:
        SystemExit: The API refused or could not be reached.  A reporter that
            silently fails is worse than one that fails loudly: the nightly job
            would be green while saying nothing.
    """
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(  # noqa: S310 - https://api.github.com, fixed below
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "llm-rosetta",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=20) as response:  # noqa: S310
            body = response.read()
    except HTTPError as exc:
        raise SystemExit(f"GitHub answered {exc.code} for {method} {url}") from exc
    except URLError as exc:
        raise SystemExit(f"cannot reach GitHub: {exc.reason}") from exc
    return json.loads(body) if body else {}


def open_issue(repo: str, token: str) -> dict | None:
    """The open issue this reporter owns, if there is one.

    Searches by the fixed title rather than by a label, so the reporter works
    on a repository whose label set it does not control.
    """
    query = f'repo:{repo} is:issue is:open in:title "shim drift"'
    answer = _api(
        "GET",
        "https://api.github.com/search/issues?q=" + query.replace(" ", "+"),
        token,
    )
    assert isinstance(answer, dict)
    for item in answer.get("items", []):
        if item.get("title") == ISSUE_TITLE:
            return item
    return None


def main(argv: list[str] | None = None) -> int:
    """Read the logs, decide, and act.

    Args:
        argv: Command-line arguments.

    Returns:
        ``0`` when the tracker now agrees with the checks, ``1`` when the run
        found problems (so the job is red whether or not an issue was written).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--problems",
        nargs="+",
        metavar="FILE",
        required=True,
        help="Check logs to read problem lines from",
    )
    parser.add_argument("--repo", required=True, help="owner/name")
    parser.add_argument(
        "--event",
        default="schedule",
        help="github.event_name; only 'schedule' may close the issue",
    )
    parser.add_argument("--run-url", default=None)
    parser.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN"),
        help="Token with issues:write; defaults to $GITHUB_TOKEN",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the decision and change nothing",
    )
    args = parser.parse_args(argv)

    text = "\n".join(Path(path).read_text(encoding="utf-8") for path in args.problems)
    current = verdict(read_problems(text))
    if args.dry_run:
        issue = None
        print(f"problems: {len(current.problems)} fingerprint: {current.fingerprint}")
        print(
            f"action (no open issue): {action_for(current=current, issue_body=None, scheduled=args.event == 'schedule')}"
        )
        return 1 if current.problems else 0

    if not args.token:
        raise SystemExit("no token: pass --token or set GITHUB_TOKEN")

    issue = open_issue(args.repo, args.token)
    issue_body = issue.get("body") if issue else None
    action = action_for(
        current=current, issue_body=issue_body, scheduled=args.event == "schedule"
    )
    body = render_body(current, run_url=args.run_url)
    base = f"https://api.github.com/repos/{args.repo}/issues"

    if action == "create":
        created = _api("POST", base, args.token, {"title": ISSUE_TITLE, "body": body})
        assert isinstance(created, dict)
        print(f"created #{created.get('number')}: {len(current.problems)} problem(s)")
    elif action == "update":
        assert issue is not None
        _api("PATCH", f"{base}/{issue['number']}", args.token, {"body": body})
        print(f"updated #{issue['number']}: {len(current.problems)} problem(s)")
    elif action == "close":
        assert issue is not None
        _api(
            "POST",
            f"{base}/{issue['number']}/comments",
            args.token,
            {"body": "The nightly check is green again."},
        )
        _api(
            "PATCH",
            f"{base}/{issue['number']}",
            args.token,
            {"state": "closed"},
        )
        print(f"closed #{issue['number']}: the check is green")
    else:
        print(f"nothing to do: {len(current.problems)} problem(s)")

    return 1 if current.problems else 0


if __name__ == "__main__":
    sys.exit(main())
