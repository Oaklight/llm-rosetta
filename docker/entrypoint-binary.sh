#!/bin/sh

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
	su appuser -s /bin/sh -c "llm-rosetta-gateway --config /config/config.jsonc init"
	echo "Edit /config/config.jsonc with your API keys and restart the container."
fi

# Build a properly quoted command string from "$@"
CMD=""
for arg in "$@"; do
	CMD="$CMD '${arg}'"
done
exec su appuser -s /bin/sh -c "exec $CMD"
