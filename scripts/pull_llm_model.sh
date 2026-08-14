#!/usr/bin/env bash
#
# Optional. Pulls the LLM used to generate object-category priors. Run in the container:
#   docker compose exec apexnav ./scripts/pull_llm_model.sh
#
# You only need this to regenerate llm/answers/*. The repo ships pre-generated output
# and config/habitat_eval_hm3dv2.yaml already points at it, so evaluation runs fine
# without ever starting ollama.
#
# Models are written to data/ollama (the mounted volume), not into the image.

set -euo pipefail

MODEL="${MODEL:-qwen3:8b}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-/workspace/ApexNav/data/ollama}"
mkdir -p "${OLLAMA_MODELS}"

if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "starting ollama serve in the background..."
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    for _ in $(seq 1 30); do
        curl -sf http://localhost:11434/api/tags >/dev/null 2>&1 && break
        sleep 1
    done
fi

if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "error: ollama did not come up; see /tmp/ollama.log" >&2
    exit 1
fi

echo "pulling ${MODEL} into ${OLLAMA_MODELS} ..."
ollama pull "${MODEL}"
ollama list
