#!/bin/bash
# Run all llm_api_simple_tests scripts against the llm-rosetta gateway.
#
# Each SDK format (openai_chat, openai_responses, anthropic, google_genai)
# is tested against every model configured in the gateway, in both
# non-streaming and streaming modes.
#
# Prerequisites:
#   - Gateway running (llm-rosetta-gateway)
#   - Submodule initialised (git submodule update --init)
#
# Usage:
#   ./scripts/run_gateway_integration.sh                  # defaults (both modes)
#   STREAM=false ./scripts/run_gateway_integration.sh     # non-streaming only
#   STREAM=true  ./scripts/run_gateway_integration.sh     # streaming only
#   GATEWAY_URL=http://host:9000 ./scripts/run_gateway_integration.sh
#   MODELS="argo:claude-opus-4.6" ./scripts/run_gateway_integration.sh
#   SDKS=anthropic ./scripts/run_gateway_integration.sh   # single provider
#   SDKS="anthropic google_genai" MODELS=gpt-5-nano ./scripts/run_gateway_integration.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUBMOD="$REPO_ROOT/llm_api_simple_tests"

GATEWAY_URL="${GATEWAY_URL:-http://localhost:8765}"
API_KEY="${API_KEY:-test}"

# STREAM: "false" | "true" | "both" (default)
STREAM="${STREAM:-both}"

# Models to test — override with MODELS env var (space-separated)
if [ -z "${MODELS:-}" ]; then
    MODELS="gpt-5-nano gpt-4.1-nano argo:claude-opus-4.6 gemini-2.5-flash-lite gemini-3.1-flash-lite-preview"
fi

# SDKs/providers to test — override with SDKS env var (space-separated)
if [ -z "${SDKS:-}" ]; then
    SDKS="openai_chat openai_responses anthropic google_genai"
fi

# Determine which stream modes to run
if [ "$STREAM" = "both" ]; then
    STREAM_MODES="false true"
else
    STREAM_MODES="$STREAM"
fi

# Per-SDK base URLs — OpenAI SDKs need /v1, Anthropic and Google do not
base_url_for_sdk() {
    case "$1" in
        openai_chat|openai_responses) echo "$GATEWAY_URL/v1" ;;
        *)                            echo "$GATEWAY_URL" ;;
    esac
}

total=0
passed=0
failed=0
failures=""

for stream_mode in $STREAM_MODES; do
    if [ "$stream_mode" = "false" ]; then
        echo "======== Non-Streaming ========"
    else
        echo ""
        echo "========== Streaming =========="
    fi

    for sdk in $SDKS; do
        for model in $MODELS; do
            base_url="$(base_url_for_sdk "$sdk")"
            for script in "$SUBMOD/scripts/$sdk"/*.py; do
                [ -f "$script" ] || continue
                test_name="$(basename "$script" .py)"
                stream_tag="stream=$stream_mode"
                label="[$sdk] [$model] [$stream_tag] $test_name"
                total=$((total + 1))

                output=$(cd "$SUBMOD" && \
                    BASE_URL="$base_url" API_KEY="$API_KEY" MODEL="$model" STREAM="$stream_mode" \
                    python "scripts/$sdk/$test_name.py" 2>&1) || true

                if echo "$output" | grep -q "PASSED"; then
                    echo "✓ $label"
                    passed=$((passed + 1))
                else
                    echo "✗ $label"
                    error=$(echo "$output" | grep -E 'Error|Traceback|error|FAILED' | head -3)
                    failures="$failures\n--- $label ---\n$error\n"
                    failed=$((failed + 1))
                fi
            done
        done
    done
done

echo ""
echo "============================================"
echo "  TOTAL: $total  |  PASSED: $passed  |  FAILED: $failed"
echo "============================================"

if [ -n "$failures" ]; then
    echo ""
    echo "FAILURES:"
    echo -e "$failures"
fi

[ "$failed" -eq 0 ]
