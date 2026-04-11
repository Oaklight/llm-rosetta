#!/bin/bash
# Launch Claude Code through Rosetta Gateway
# Usage: ./scripts/rosetta-test-claude-code.sh [model]
#   model defaults to gpt-4.1-nano
#   Available models: gpt-4.1-nano, anthropic/claude-haiku-4.5, gemini-2.5-flash-lite
#   Gateway maps model name → upstream provider automatically

MODEL="${1:-gpt-4.1-nano}"

ANTHROPIC_BASE_URL=http://localhost:8765 \
	ANTHROPIC_API_KEY=dummy \
	exec claude --model "$MODEL" --verbose
