#!/bin/sh

mkdir -p /config 2>/dev/null || true

# Resolve HOME for the runtime uid.  appuser (uid 1000) has a passwd home; a uid
# that Compose's `user:` directive selects has no passwd entry, and the image
# then leaves HOME at "/", which nothing can write.  Keep the passwd home when
# one exists, otherwise take HOME from the environment when it names one (the
# source compose file passes /config/home) and fall back to /config/home.  This
# runs as the target uid, never as root.
passwd_home=$(awk -F: -v uid="$(id -u)" '$3 == uid { print $6; exit }' /etc/passwd 2>/dev/null)
if [ -n "$passwd_home" ] && [ -d "$passwd_home" ]; then
	HOME="$passwd_home"
else
	case "$HOME" in
		"" | "/") HOME=/config/home ;;
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

if [ ! -f /config/config.jsonc ]; then
	echo "No config.jsonc found in /config, generating template..."
	llm-rosetta-gateway --config /config/config.jsonc init
	echo "Edit /config/config.jsonc with your API keys and restart the container."
fi

if [ "${1#-}" != "$1" ]; then
	set -- llm-rosetta-gateway "$@"
fi

exec "$@"
