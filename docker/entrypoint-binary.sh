#!/bin/sh

# If running as root (default), handle PUID/PGID and drop privileges.
# If running as non-root (docker run --user), skip privilege management.
if [ "$(id -u)" = "0" ]; then
	PUID=${PUID:-1000}
	PGID=${PGID:-1000}

	if [ "$(id -u appuser)" != "$PUID" ] || [ "$(id -g appuser)" != "$PGID" ]; then
		sed -i "s/^appuser:x:[0-9]*:[0-9]*:/appuser:x:$PUID:$PGID:/" /etc/passwd
		sed -i "s/^appgroup:x:[0-9]*:/appgroup:x:$PGID:/" /etc/group
	fi

	mkdir -p /config
	chown -R appuser:appgroup /config

	if [ ! -f /config/config.jsonc ]; then
		echo "No config.jsonc found in /config, generating template..."
		su-exec appuser llm-rosetta-gateway --config /config/config.jsonc init
		echo "Edit /config/config.jsonc with your API keys and restart the container."
	fi

	if [ "${1#-}" != "$1" ]; then
		set -- llm-rosetta-gateway "$@"
	fi

	exec su-exec appuser "$@"
else
	mkdir -p /config 2>/dev/null || true

	if [ ! -f /config/config.jsonc ]; then
		echo "No config.jsonc found in /config, generating template..."
		llm-rosetta-gateway --config /config/config.jsonc init
		echo "Edit /config/config.jsonc with your API keys and restart the container."
	fi

	if [ "${1#-}" != "$1" ]; then
		set -- llm-rosetta-gateway "$@"
	fi

	exec "$@"
fi
