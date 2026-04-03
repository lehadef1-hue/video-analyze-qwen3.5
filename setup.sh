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
echo
echo "Выберите модель:"
echo "  1) Qwen3-VL-30B-FP8   (vision-language, grid mode, ~40 GB VRAM)"
echo "  2) Qwen3.5-27B-FP8    (multimodal, video mode, ~28 GB VRAM)"
echo
read -r -p "Введите 1 или 2: " BACKEND_CHOICE

case "$BACKEND_CHOICE" in
    1)
        MODEL_PRESET="qwen3vl"
        TAGGER_MODE="grid"
        echo "→ Qwen3-VL-30B-FP8  (grid mode)"
        ;;
    2)
        MODEL_PRESET="qwen35"
        TAGGER_MODE="video"
        echo "→ Qwen3.5-27B-FP8  (video mode)"
        ;;
    *)
        echo "Неверный выбор. Завершение."
        exit 1
        ;;
esac

echo

# ── 1. Python venv ────────────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/4] Creating venv..."
    python3 -m venv "$VENV_DIR" --system-site-packages
else
    echo "[1/4] Venv already exists, skipping."
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet

# ── 2. vLLM ──────────────────────────────────────────────────────────────────
echo "[2/4] Installing vLLM + transformers..."
pip install vllm transformers

# ── 3. Project dependencies ───────────────────────────────────────────────────
echo "[3/4] Installing project dependencies..."
pip install \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    opencv-python-headless \
    pillow \
    requests \
    jinja2 \
    numpy \
    yt-dlp \
    huggingface_hub

# ── 4. InsightFace ───────────────────────────────────────────────────────────
echo "[4/4] Installing InsightFace for performer recognition..."
pip install insightface onnxruntime-gpu
echo "  InsightFace installed."

# ── Директории ───────────────────────────────────────────────────────────────
mkdir -p "$HF_CACHE/hub"

# ── Генерация start-скриптов ─────────────────────────────────────────────────
echo
echo "Generating start scripts..."

cat > "$PROJECT_DIR/start_model_server.sh" << EOF
#!/usr/bin/env bash
PROJECT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
source "\$PROJECT_DIR/.venv/bin/activate"
export HF_HOME="/workspace/hf_cache"
export HF_HUB_CACHE="/workspace/hf_cache/hub"
export TRANSFORMERS_CACHE="/workspace/hf_cache/hub"
export HF_HUB_DISABLE_XET=1
export MODEL_PRESET="$MODEL_PRESET"
cd "\$PROJECT_DIR"
uvicorn model_server:app --host 0.0.0.0 --port 8080
EOF
chmod +x "$PROJECT_DIR/start_model_server.sh"

cat > "$PROJECT_DIR/start_app.sh" << EOF
#!/usr/bin/env bash
PROJECT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
source "\$PROJECT_DIR/.venv/bin/activate"
export PERFORMER_DB_PATH="/workspace/my_performers.pkl"
export TAGGER_MODE=$TAGGER_MODE
cd "\$PROJECT_DIR"
uvicorn video_processor:app --host 0.0.0.0 --port 8000
EOF
chmod +x "$PROJECT_DIR/start_app.sh"

# ── Итог ─────────────────────────────────────────────────────────────────────
echo
echo "=== Done ==="
echo "Model  : $MODEL_PRESET"
echo "Tagger : $TAGGER_MODE"
echo
echo "Start model server : bash start_model_server.sh"
echo "Start app server   : bash start_app.sh"
echo
echo "Build performer DB (optional):"
echo "  export TPDB_API_TOKEN=your_token"
echo "  python build_performer_db.py --auto --count 200"
