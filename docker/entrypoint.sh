#!/bin/sh

# Load PUID and PGID from environment variables
PUID=${PUID:-1000}
PGID=${PGID:-1000}

# Modify the existing user and group to match PUID and PGID
if [ "$(id -u appuser)" != "$PUID" ] || [ "$(id -g appuser)" != "$PGID" ]; then
	sed -i "s/^appuser:x:[0-9]*:[0-9]*:/appuser:x:$PUID:$PGID:/" /etc/passwd
	sed -i "s/^appgroup:x:[0-9]*:/appgroup:x:$PGID:/" /etc/group
fi

# Ensure config directory exists with proper ownership
# (Docker creates bind-mount directories as root if they don't exist on the host)
mkdir -p /config
chown -R appuser:appgroup /config

# Auto-generate a template config if none exists
if [ ! -f /config/config.jsonc ]; then
	echo "No config.jsonc found in /config, generating template..."
	su-exec appuser llm-rosetta-gateway --config /config/config.jsonc init
	echo "Edit /config/config.jsonc with your API keys and restart the container."
fi

# Switch to appuser and execute the command passed as arguments
exec su-exec appuser "$@"
