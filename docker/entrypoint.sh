#!/bin/sh

# The config volume; the variable lets a test point the script elsewhere.
CONFIG_DIR=${CONFIG_DIR:-/config}
mkdir -p "$CONFIG_DIR" 2>/dev/null || true

# Resolve HOME for the runtime uid.  appuser (uid 1000) has a passwd home; a uid
# that Compose's `user:` directive selects has no passwd entry, and the image
# then leaves HOME at "/", which nothing can write.  Keep the passwd home when
# one exists, otherwise take HOME from the environment when it names one (the
# source compose file passes /config/home) and fall back to $CONFIG_DIR/home.  This
# runs as the target uid, never as root.
passwd_home=$(awk -F: -v uid="$(id -u)" '$3 == uid { print $6; exit }' /etc/passwd 2>/dev/null)
if [ -n "$passwd_home" ] && [ -d "$passwd_home" ]; then
	HOME="$passwd_home"
else
	case "$HOME" in
		"" | "/") HOME=$CONFIG_DIR/home ;;
	esac
fi
export HOME
if ! mkdir -p "$HOME" 2>/dev/null; then
	echo "error: cannot create HOME=$HOME as uid $(id -u); the mounted directory must be writable by that uid (chown it on the host, or set PUID/PGID to its owner)" >&2
	exit 1
fi
if [ ! -w "$HOME" ]; then
	echo "error: HOME=$HOME exists but is not writable by uid $(id -u); chown it on the host, or set PUID/PGID to its owner" >&2
	exit 1
fi

# Extensions installed in the container, and the moved vendors the gateway
# installs by itself at start, survive the container only when they live in the
# config volume.  Unless the operator set the directory, use
# $CONFIG_DIR/extensions.  A run that replaces this script with `--entrypoint`
# gets no default.  The directory has to be writable by this uid, as HOME does.
if [ -z "${LLM_ROSETTA_EXTENSIONS_DIR:-}" ]; then
	if ! mkdir -p "$CONFIG_DIR/extensions" 2>/dev/null || [ ! -w "$CONFIG_DIR/extensions" ]; then
		echo "error: cannot use $CONFIG_DIR/extensions for installed extensions as uid $(id -u); the mounted directory must be writable by that uid (chown it on the host, or set PUID/PGID to its owner), or set LLM_ROSETTA_EXTENSIONS_DIR" >&2
		exit 1
	fi
	LLM_ROSETTA_EXTENSIONS_DIR=$CONFIG_DIR/extensions
fi
export LLM_ROSETTA_EXTENSIONS_DIR

# External CLIs that `llm-rosetta ext install-cli` installs (an extension's
# agent CLI, at a pinned version) live in the config volume too.  Their entry
# links are collected in one directory, which goes on PATH here so a CLI
# installed after the container started is found without a restart.
LLM_ROSETTA_TOOLS_DIR=${LLM_ROSETTA_TOOLS_DIR:-$CONFIG_DIR/tools}
export LLM_ROSETTA_TOOLS_DIR
PATH=$LLM_ROSETTA_TOOLS_DIR/bin:$PATH
export PATH

if [ "${1#-}" != "$1" ]; then
	set -- llm-rosetta-gateway "$@"
fi

# Whether this invocation starts the server: the default CMD, an argument list
# that holds only options, and an explicit `serve` (also `gateway serve`).  Every
# other command (init, add, ext, gateway usage, a shell) is a one-off and must
# not write a config template or tell the operator to restart the container.
starts_server() {
	case "$1" in
		llm-rosetta-gateway | */llm-rosetta-gateway) bare_serves=1 ;;
		llm-rosetta | */llm-rosetta) bare_serves=0 ;;
		*) return 1 ;;
	esac
	shift
	first=""
	second=""
	while [ "$#" -gt 0 ] && [ -z "$second" ]; do
		case "$1" in
			--config | -c | --host | --port | --socket | -S | --tls-cert | --tls-key | --proxy | --data-dir | --log-level)
				[ "$#" -gt 1 ] && shift
				;;
			-*) ;;
			*)
				if [ -z "$first" ]; then first=$1; else second=$1; fi
				;;
		esac
		shift
	done
	case "$first" in
		"") [ "$bare_serves" = 1 ] ;;
		serve) return 0 ;;
		gateway) [ "$second" = serve ] ;;
		*) return 1 ;;
	esac
}

if starts_server "$@" && [ ! -f "$CONFIG_DIR/config.jsonc" ]; then
	echo "No config.jsonc found in $CONFIG_DIR, generating template..."
	llm-rosetta-gateway --config "$CONFIG_DIR/config.jsonc" init
	echo "Edit $CONFIG_DIR/config.jsonc with your API keys and restart the container."
fi

exec "$@"
