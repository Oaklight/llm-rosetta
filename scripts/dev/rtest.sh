#!/usr/bin/env bash
# rtest — run a worktree's pytest suite on a remote machine.
#
# The suite is copied to the remote host and executed *there*, from the copy,
# with PYTHONPATH pointing at the synced packages. The run therefore exercises
# the working tree and not whatever version happens to be installed on the
# remote.
#
# The file list comes from git (`git ls-files -co --exclude-standard`), so
# ignored files — screenshots, scratch databases, caches — never travel, and a
# worktree's rules apply unchanged.
#
# Environment:
#   RTEST_HOST      ssh destination, e.g. user@build-host. Required.
#   RTEST_ROOT      remote directory holding one tree per worktree
#                   (default .cache/rtest under the remote user's home).
#                   rsync does not expand a shell, so this is a plain path:
#                   relative to the remote home, or absolute. No whitespace.
#   RTEST_PYTHON    interpreter on the remote
#                   (default <RTEST_ROOT>/venv/bin/python, i.e.
#                   $HOME/.cache/rtest/venv/bin/python)
#   RTEST_SSH_OPTS  extra ssh options, e.g. "-p 2222 -i ~/.ssh/build"
#
# Usage:
#   scripts/dev/rtest.sh [--worktree DIR] [--dry-run] [pytest args...]
#
#   scripts/dev/rtest.sh                 # this tree, tests/ minus integration
#   scripts/dev/rtest.sh -q --no-cov     # extra pytest args pass through
#   scripts/dev/rtest.sh --worktree ../other-wt tests/gateway -q
#   RTEST_HOST=me@host scripts/dev/rtest.sh --dry-run
#
# The remote host needs the project's dependencies installed in RTEST_PYTHON;
# this script installs nothing.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: rtest.sh [--worktree DIR] [--dry-run] [pytest args...]

  --worktree DIR  tree to test (default: the current directory's git root)
  --dry-run       print the file count, destination and remote command
  -h, --help      this text

Environment:
  RTEST_HOST      ssh destination (required)
  RTEST_ROOT      remote root for the trees (default .cache/rtest, relative
                  to the remote user's home)
  RTEST_PYTHON    remote interpreter (default <RTEST_ROOT>/venv/bin/python)
  RTEST_SSH_OPTS  extra ssh options, passed through verbatim

Without pytest arguments the run is `tests/ --ignore=tests/integration -q`.
EOF
}

die() {
  printf 'rtest: %s\n' "$*" >&2
  exit 2
}

# Single-quote a string for display, escaping embedded single quotes.
shell_quote() {
  printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

# A short, stable hash of a path, without assuming coreutils.
short_hash() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256
  else
    cksum
  fi | cut -c1-8
}

worktree=$PWD
dry_run=0
pytest_args=()

while (($# > 0)); do
  case $1 in
    --worktree)
      (($# >= 2)) || die "--worktree needs a directory"
      worktree=$2
      shift 2
      ;;
    --worktree=*)
      worktree=${1#*=}
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    --)
      shift
      while (($# > 0)); do
        pytest_args+=("$1")
        shift
      done
      ;;
    *)
      pytest_args+=("$1")
      shift
      ;;
  esac
done

if ((${#pytest_args[@]} == 0)); then
  pytest_args=(tests/ --ignore=tests/integration -q)
fi

[[ -n ${RTEST_HOST:-} ]] ||
  die "RTEST_HOST is not set (example: RTEST_HOST=me@build-host $0)"
[[ -d $worktree ]] || die "no such directory: $worktree"

top=$(git -C "$worktree" rev-parse --show-toplevel 2>/dev/null) ||
  die "not inside a git work tree: $worktree"
tree=$(cd "$top" && pwd -P)

# One directory per tree: the basename for a human, the hash for uniqueness.
# Two sessions pointed at different worktrees never share a delete target.
slug="$(basename "$tree")-$(printf '%s' "$tree" | short_hash)"

# rsync runs the destination through no shell, so a relative root resolves
# against the remote user's home there and in the ssh command alike.
if [[ -n ${RTEST_ROOT:-} ]]; then
  root=$RTEST_ROOT
  default_python="$root/venv/bin/python"
else
  root=.cache/rtest
  # shellcheck disable=SC2016  # deliberate: this one is expanded by the remote shell
  default_python='$HOME/.cache/rtest/venv/bin/python'
fi
python=${RTEST_PYTHON:-"$default_python"}
for value in "$root" "$python"; do
  case $value in
    *[[:space:]\']*) die "RTEST_ROOT and RTEST_PYTHON must not contain whitespace or single quotes" ;;
  esac
done
dest="$root/$slug"

# The source roots that live in this tree, so the run imports the synced
# copies rather than whatever is installed remotely.
mapfile -t src_dirs < <(
  {
    git -C "$tree" ls-files 'packages/*/src/**' |
      awk -F/ 'NF >= 3 { print $1 "/" $2 "/" $3 }'
    if git -C "$tree" ls-files 'src/**' | read -r _; then
      printf 'src\n'
    fi
  } | sort -u
)
pythonpath=""
if ((${#src_dirs[@]} > 0)); then
  pythonpath=$(IFS=: && printf '%s' "${src_dirs[*]}")
fi

ssh_command=(ssh)
if [[ -n ${RTEST_SSH_OPTS:-} ]]; then
  read -r -a ssh_opts <<<"$RTEST_SSH_OPTS"
  ssh_command+=("${ssh_opts[@]}")
fi

printf -v quoted_args ' %q' "${pytest_args[@]}"
remote_command="mkdir -p $dest && cd $dest && "
if [[ -n $pythonpath ]]; then
  printf -v quoted_pythonpath '%q' "$pythonpath"
  remote_command+="PYTHONPATH=$quoted_pythonpath "
fi
remote_command+="$python -m pytest$quoted_args"

if ((dry_run)); then
  file_count=$(git -C "$tree" ls-files -co --exclude-standard | wc -l | tr -d ' ')
  printf 'host:    %s\n' "$RTEST_HOST"
  printf 'tree:    %s\n' "$tree"
  printf 'files:   %s\n' "$file_count"
  printf 'dest:    %s\n' "$dest"
  printf 'rsync:   git ls-files -z -co --exclude-standard | '
  printf 'rsync -a --delete --from0 --files-from=- %q %q\n' "$tree/" "$RTEST_HOST:$dest/"
  printf 'command: %s %s %s\n' \
    "${ssh_command[*]}" "$RTEST_HOST" "$(shell_quote "$remote_command")"
  exit 0
fi

git -C "$tree" ls-files -z -co --exclude-standard |
  rsync -a --delete --from0 --files-from=- "$tree/" "$RTEST_HOST:$dest/"

# exec so the remote exit status is this script's, and output streams as it
# arrives.
exec "${ssh_command[@]}" "$RTEST_HOST" "$remote_command"
