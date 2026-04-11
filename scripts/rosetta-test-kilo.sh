#!/bin/bash
# Launch Kilo through Rosetta Gateway (via VS Code / IDE)
# Kilo is an IDE extension — select "rosetta" provider in the IDE model picker.
# Available models under rosetta provider:
#   - gpt-4.1-nano              → OpenAI backend
#   - anthropic/claude-haiku-4.5 → Anthropic backend
#   - gemini-2.5-flash-lite     → Google backend
#
# Models are defined in ~/.config/kilo/kilo.jsonc under "rosetta" provider.
echo "Kilo is an IDE extension. Select rosetta/<model> in the model picker."
echo "Available rosetta models:"
echo "  - rosetta/gpt-4.1-nano"
echo "  - rosetta/anthropic/claude-haiku-4.5"
echo "  - rosetta/gemini-2.5-flash-lite"
