#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# RunPod deployment script for videoQwen
# Run once from the project directory: bash setup.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
HF_CACHE="/workspace/hf_cache"

echo "=== videoQwen setup ==="
echo "Project : $PROJECT_DIR"
echo "Venv    : $VENV_DIR"
echo "HF cache: $HF_CACHE"
echo

# ── 1. Python venv ────────────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/4] Creating venv..."
    python3 -m venv "$VENV_DIR" --system-site-packages
else
    echo "[1/4] Venv already exists, skipping."
fi

source "$VENV_DIR/bin/activate"

# ── 2. vllm (latest) ─────────────────────────────────────────────────────────
echo "[2/4] Installing vllm (latest)..."
pip install --upgrade pip --quiet
pip install vllm

# ── 3. Project dependencies ───────────────────────────────────────────────────
echo "[3/4] Installing project dependencies..."
pip install \
    fastapi \
    "uvicorn[standard]" \
    opencv-python-headless \
    pillow \
    requests \
    jinja2 \
    transformers \
    numpy \
    yt-dlp

# ── 4. Optional: InsightFace for performer recognition ───────────────────────
echo
read -r -p "[4/4] Install InsightFace for performer recognition? [y/N] " REPLY
if [[ "$REPLY" =~ ^[Yy]$ ]]; then
    pip install insightface onnxruntime-gpu
    echo "  InsightFace installed."
else
    echo "  Skipped. Run later: pip install insightface onnxruntime-gpu"
fi

# ── HF cache dir ─────────────────────────────────────────────────────────────
mkdir -p "$HF_CACHE/hub"

# ── Generate start scripts ───────────────────────────────────────────────────
echo
echo "Generating start scripts..."

cat > "$PROJECT_DIR/start_model_server.sh" << 'EOF'
#!/usr/bin/env bash
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
export HF_HOME="/workspace/hf_cache"
export HF_HUB_CACHE="/workspace/hf_cache/hub"
export TRANSFORMERS_CACHE="/workspace/hf_cache/hub"
export HF_HUB_DISABLE_XET=1
cd "$PROJECT_DIR"
uvicorn model_server:app --host 0.0.0.0 --port 8080
EOF

cat > "$PROJECT_DIR/start_app.sh" << 'EOF'
#!/usr/bin/env bash
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
export PERFORMER_DB_PATH="/workspace/my_performers.pkl"
cd "$PROJECT_DIR"
uvicorn video_processor:app --host 0.0.0.0 --port 8000
EOF

chmod +x "$PROJECT_DIR/start_model_server.sh" "$PROJECT_DIR/start_app.sh"

echo
echo "=== Done ==="
echo
echo "Start model server : bash start_model_server.sh"
echo "Start app server   : bash start_app.sh"
echo
echo "Build performer DB (optional):"
echo "  export TPDB_API_TOKEN=your_token"
echo "  python build_performer_db.py --auto --count 200"
