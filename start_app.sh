#!/usr/bin/env bash
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
export PERFORMER_DB_PATH="/workspace/my_performers.pkl"
cd "$PROJECT_DIR"
API_KEY="incredible" python3 video_processor.py
