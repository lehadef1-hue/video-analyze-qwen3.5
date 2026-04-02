#!/usr/bin/env bash
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
export HF_HOME="/workspace/hf_cache"
export HF_HUB_CACHE="/workspace/hf_cache/hub"
export TRANSFORMERS_CACHE="/workspace/hf_cache/hub"
export HF_HUB_DISABLE_XET=1
cd "$PROJECT_DIR"
uvicorn model_server:app --host 0.0.0.0 --port 8080
