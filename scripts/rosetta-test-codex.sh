#!/bin/bash
# Launch Codex CLI through Rosetta Gateway
# Usage: ./scripts/rosetta-test-codex.sh [profile]
#   profile: rosetta-openai (default), rosetta-anthropic, rosetta-google
#   Profiles are defined in ~/.codex/config.toml

PROFILE="${1:-rosetta-openai}"

exec codex --profile "$PROFILE"
