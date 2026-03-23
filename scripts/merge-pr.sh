#!/usr/bin/env bash
set -euo pipefail

usage() {
	echo "Usage: $(basename "$0") <PR_NUMBER> [--dry-run]"
	echo ""
	echo "Rebase-merge a GitHub PR onto master, stripping Co-authored-by trailers."
	echo ""
	echo "Options:"
	echo "  --dry-run   Show what would happen without pushing or closing the PR"
	exit 1
}

if [[ $# -lt 1 ]]; then
	usage
fi

PR_NUMBER="$1"
DRY_RUN=false
if [[ "${2:-}" == "--dry-run" ]]; then
	DRY_RUN=true
fi

BRANCH="pr-${PR_NUMBER}"
BASE_BRANCH="master"

# Ensure we're in a git repo
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
	echo "Error: not inside a git repository"
	exit 1
fi

# Ensure gh CLI is available
if ! command -v gh &>/dev/null; then
	echo "Error: gh CLI not found"
	exit 1
fi

# Ensure working tree is clean
if ! git diff --quiet || ! git diff --cached --quiet; then
	echo "Error: working tree is not clean, please commit or stash changes first"
	exit 1
fi

# Verify PR exists and is open
PR_STATE=$(gh pr view "$PR_NUMBER" --json state --jq '.state')
if [[ "$PR_STATE" != "OPEN" ]]; then
	echo "Error: PR #${PR_NUMBER} is not open (state: ${PR_STATE})"
	exit 1
fi

PR_TITLE=$(gh pr view "$PR_NUMBER" --json title --jq '.title')
echo "Merging PR #${PR_NUMBER}: ${PR_TITLE}"

# Fetch latest master
echo "=> Fetching latest ${BASE_BRANCH}..."
git fetch origin "${BASE_BRANCH}"
git checkout "${BASE_BRANCH}"
git merge --ff-only "origin/${BASE_BRANCH}"

# Fetch PR into a local branch
echo "=> Fetching PR #${PR_NUMBER}..."
git fetch origin "pull/${PR_NUMBER}/head:${BRANCH}"

# Rebase onto master and strip Co-authored-by in one pass
echo "=> Rebasing and stripping Co-authored-by trailers..."
git rebase "${BASE_BRANCH}" "${BRANCH}" --exec \
	'git commit --amend -m "$(git log --format=%B -n1 | sed "/^Co-authored-by:/d" | sed -e :a -e "/^\n*$/{$d;N;ba;}")"'

# Show result
echo ""
echo "=> Commits to merge:"
git log --oneline "${BASE_BRANCH}..${BRANCH}"
echo ""

if [[ "$DRY_RUN" == true ]]; then
	echo "[dry-run] Would fast-forward ${BASE_BRANCH}, push, and close PR #${PR_NUMBER}"
	echo "[dry-run] Cleaning up local branch ${BRANCH}..."
	git checkout "${BASE_BRANCH}"
	git branch -D "${BRANCH}"
	exit 0
fi

# Fast-forward merge
echo "=> Fast-forward merging into ${BASE_BRANCH}..."
git checkout "${BASE_BRANCH}"
git merge --ff-only "${BRANCH}"

# Push
echo "=> Pushing ${BASE_BRANCH}..."
git push origin "${BASE_BRANCH}"

# Clean up local branch
git branch -d "${BRANCH}"

# Close PR with comment
echo "=> Closing PR #${PR_NUMBER}..."
gh pr close "$PR_NUMBER" --comment "Merged via local rebase onto ${BASE_BRANCH} (Co-authored-by trailers stripped)."

echo ""
echo "Done! PR #${PR_NUMBER} merged and closed."
