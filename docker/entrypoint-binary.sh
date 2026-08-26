#!/bin/sh

mkdir -p /config 2>/dev/null || true

if [ ! -f /config/config.jsonc ]; then
	echo "No config.jsonc found in /config, generating template..."
	/usr/local/bin/llm-rosetta-gateway --config /config/config.jsonc init
	echo "Edit /config/config.jsonc with your API keys and restart the container."
fi

if [ "${1#-}" != "$1" ]; then
	set -- /usr/local/bin/llm-rosetta-gateway "$@"
fi

exec "$@"
