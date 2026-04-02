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
echo "Выберите бэкенд модели:"
echo "  1) vLLM        — Qwen/Qwen3-VL-30B-A3B-Instruct-FP8  (HuggingFace, ~30 GB VRAM)"
echo "  2) llama.cpp   — GGUF модель с mmproj (напр. HauhauCS/Qwen3.5-27B-Uncensored Q8_0, ~27 GB VRAM)"
echo
read -r -p "Введите 1 или 2: " BACKEND_CHOICE

case "$BACKEND_CHOICE" in
    1)
        BACKEND="vllm"
        echo "→ Выбран бэкенд: vLLM"
        ;;
    2)
        BACKEND="llama"
        MODEL_PATH="$HF_CACHE/Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf"
        MMPROJ_PATH="$HF_CACHE/mmproj-Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-f16.gguf"
        echo "→ Выбран бэкенд: llama.cpp"
        echo "  MODEL_PATH  = $MODEL_PATH"
        echo "  MMPROJ_PATH = $MMPROJ_PATH"
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

# ── 2. Зависимости бэкенда ───────────────────────────────────────────────────
if [ "$BACKEND" = "vllm" ]; then
    echo "[2/4] Installing vLLM (latest)..."
    pip install vllm transformers
else
    echo "[2/4] Installing llama-cpp-python with CUDA support..."
    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
fi

# ── 3. Project dependencies ───────────────────────────────────────────────────
echo "[3/4] Installing project dependencies..."
pip install \
    fastapi \
    "uvicorn[standard]" \
    opencv-python-headless \
    pillow \
    requests \
    jinja2 \
    numpy \
    yt-dlp

# ── 4. Optional: InsightFace ─────────────────────────────────────────────────
echo
read -r -p "[4/4] Install InsightFace for performer recognition? [y/N] " REPLY
if [[ "$REPLY" =~ ^[Yy]$ ]]; then
    pip install insightface onnxruntime-gpu
    echo "  InsightFace installed."
else
    echo "  Skipped. Run later: pip install insightface onnxruntime-gpu"
fi

# ── Директории ───────────────────────────────────────────────────────────────
mkdir -p "$HF_CACHE/hub"
if [ "$BACKEND" = "llama" ]; then
    mkdir -p "$(dirname "$MODEL_PATH")"
fi

# ── Генерация start-скриптов ─────────────────────────────────────────────────
echo
echo "Generating start scripts..."

if [ "$BACKEND" = "vllm" ]; then
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
    chmod +x "$PROJECT_DIR/start_model_server.sh"
    MODEL_SERVER_SCRIPT="start_model_server.sh"

else
    # llama.cpp — пишем пути напрямую в скрипт
    cat > "$PROJECT_DIR/start_model_server_llama.sh" << EOF
#!/usr/bin/env bash
PROJECT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
source "\$PROJECT_DIR/.venv/bin/activate"
export MODEL_PATH="$MODEL_PATH"
export MMPROJ_PATH="$MMPROJ_PATH"
export N_GPU_LAYERS=-1
export N_CTX=32768
cd "\$PROJECT_DIR"
uvicorn model_server_llama:app --host 0.0.0.0 --port 8080
EOF
    chmod +x "$PROJECT_DIR/start_model_server_llama.sh"
    MODEL_SERVER_SCRIPT="start_model_server_llama.sh"
fi

cat > "$PROJECT_DIR/start_app.sh" << 'EOF'
#!/usr/bin/env bash
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"
export PERFORMER_DB_PATH="/workspace/my_performers.pkl"
cd "$PROJECT_DIR"
uvicorn video_processor:app --host 0.0.0.0 --port 8000
EOF
chmod +x "$PROJECT_DIR/start_app.sh"

# ── Итог ─────────────────────────────────────────────────────────────────────
echo
echo "=== Done ==="
echo "Backend: $BACKEND"
echo
echo "Start model server : bash $MODEL_SERVER_SCRIPT"
echo "Start app server   : bash start_app.sh"
echo
if [ "$BACKEND" = "llama" ]; then
    echo "Если модель ещё не скачана:"
    echo "  pip install huggingface_hub"
    echo "  huggingface-cli download HauhauCS/Qwen3.5-27B-Uncensored-HauhauCS-Aggressive \\"
    echo "    Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf \\"
    echo "    mmproj-Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-f16.gguf \\"
    echo "    --local-dir $HF_CACHE"
    echo
fi
echo "Build performer DB (optional):"
echo "  export TPDB_API_TOKEN=your_token"
echo "  python build_performer_db.py --auto --count 200"
