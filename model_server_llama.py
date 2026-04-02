"""
Model server using llama.cpp backend (llama-cpp-python).
Drop-in replacement for model_server.py — same /generate API.

Compatible with GGUF models + separate mmproj file for vision.
Tested with: HauhauCS/Qwen3.5-27B-Uncensored-HauhauCS-Aggressive

Install:
    pip install llama-cpp-python[server] fastapi uvicorn

For GPU support build with CUDA:
    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

Run:
    MODEL_PATH=/workspace/models/model.gguf \\
    MMPROJ_PATH=/workspace/models/mmproj.gguf \\
    uvicorn model_server_llama:app --host 0.0.0.0 --port 8080
"""

import gc
import json
import logging
import os
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional


# ─── Логирование ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("model_server_llama")


# ─── ENV ─────────────────────────────────────────────────
HF_REPO_ID       = os.environ.get("HF_REPO_ID",       "HauhauCS/Qwen3.5-27B-Uncensored-HauhauCS-Aggressive")
MODEL_FILENAME   = os.environ.get("MODEL_FILENAME",   "Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf")
MMPROJ_FILENAME  = os.environ.get("MMPROJ_FILENAME",  "mmproj-Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-f16.gguf")
HF_CACHE         = os.environ.get("HF_CACHE",         "/workspace/hf_cache")

MODEL_PATH   = os.environ.get("MODEL_PATH",   os.path.join(HF_CACHE, MODEL_FILENAME))
MMPROJ_PATH  = os.environ.get("MMPROJ_PATH",  os.path.join(HF_CACHE, MMPROJ_FILENAME))
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "-1"))   # -1 = all layers on GPU
N_CTX        = int(os.environ.get("N_CTX",        "32768"))
N_BATCH      = int(os.environ.get("N_BATCH",      "512"))


# ─── Авто-скачивание GGUF файлов ─────────────────────────
def _download_if_missing(filename: str, dest_path: str) -> None:
    if os.path.exists(dest_path):
        return
    logger.info(f"Файл не найден, скачиваю: {filename} → {dest_path}")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.critical("huggingface_hub не установлен. Выполните: pip install huggingface_hub")
        sys.exit(1)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        local_dir=HF_CACHE,
    )
    logger.info(f"✓ Скачано: {filename}")

_download_if_missing(MODEL_FILENAME, MODEL_PATH)
_download_if_missing(MMPROJ_FILENAME, MMPROJ_PATH)

has_vision = os.path.exists(MMPROJ_PATH)
if not has_vision:
    logger.warning(f"mmproj не найден после скачивания: {MMPROJ_PATH} — vision отключён")


# ─── Загрузка модели ─────────────────────────────────────
logger.info(f"Загрузка модели: {MODEL_PATH}")
logger.info(f"GPU layers: {N_GPU_LAYERS} | ctx: {N_CTX} | batch: {N_BATCH}")

from llama_cpp import Llama

try:
    from llama_cpp import LlamaGrammar
    _grammar_available = True
except ImportError:
    _grammar_available = False
    logger.warning("LlamaGrammar недоступен — guided_json не поддерживается")

chat_handler = None
if has_vision:
    try:
        # Qwen2VLChatHandler поддерживает архитектуру Qwen2.5-VL / Qwen3.5-VL
        from llama_cpp.llama_chat_format import Qwen2VLChatHandler
        chat_handler = Qwen2VLChatHandler(clip_model_path=MMPROJ_PATH, verbose=False)
        logger.info("Vision включён (Qwen2VLChatHandler)")
    except (ImportError, Exception) as e:
        logger.warning(f"Qwen2VLChatHandler недоступен ({e}), пробуем LlavaQwen2ChatHandler...")
        try:
            from llama_cpp.llama_chat_format import LlavaQwen2ChatHandler
            chat_handler = LlavaQwen2ChatHandler(clip_model_path=MMPROJ_PATH, verbose=False)
            logger.info("Vision включён (LlavaQwen2ChatHandler)")
        except (ImportError, Exception) as e2:
            logger.error(f"Vision handler не загружен: {e2} — работаем без изображений")
            chat_handler = None

llm = Llama(
    model_path=MODEL_PATH,
    chat_handler=chat_handler,
    n_ctx=N_CTX,
    n_batch=N_BATCH,
    n_gpu_layers=N_GPU_LAYERS,
    verbose=False,
)

logger.info("✓ Модель успешно загружена")


# ─── FastAPI ─────────────────────────────────────────────
app = FastAPI(title="llama.cpp Model Server")


class GenerateRequest(BaseModel):
    prompt: str
    base64_images: List[str] = []
    sampling_params: Optional[Dict[str, Any]] = None
    enable_thinking: bool = False  # /think mode — Qwen3 thinking tokens
    guided_json: Optional[Dict[str, Any]] = None  # JSON schema → grammar-based structured output


@app.post("/generate")
async def generate(request: GenerateRequest):
    try:
        params = request.sampling_params or {
            "temperature": 0.7,
            "top_p":       0.95,
            "max_tokens":  96,
        }

        # Рекомендованные параметры из model card
        if request.enable_thinking:
            params.setdefault("temperature", 0.6)
            params.setdefault("top_k",       20)
            params.setdefault("min_p",       0.0)
        else:
            params.setdefault("top_k", 20)
            params.setdefault("min_p", 0.0)

        # repeat_penalty = аналог repetition_penalty в vLLM
        repeat_penalty = params.pop("repetition_penalty", params.pop("repeat_penalty", 1.15))

        # ── Thinking mode ────────────────────────────────
        # Qwen3: /think = включить, /no_think = выключить (передаётся в начале сообщения)
        prompt_text = request.prompt
        if request.enable_thinking:
            prompt_text = "/think\n" + prompt_text
            logger.info("Thinking mode enabled")
        else:
            prompt_text = "/no_think\n" + prompt_text

        # ── Сборка content ───────────────────────────────
        content: List[Dict] = []

        for b64 in request.base64_images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        content.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content}]

        # ── Grammar (guided_json) ────────────────────────
        grammar = None
        if request.guided_json and _grammar_available:
            try:
                grammar = LlamaGrammar.from_json_schema(json.dumps(request.guided_json))
            except Exception as e:
                logger.warning(f"Не удалось построить grammar: {e} — продолжаем без ограничений")

        # ── Генерация ────────────────────────────────────
        response = llm.create_chat_completion(
            messages=messages,
            temperature=params.get("temperature", 0.7),
            top_p=params.get("top_p", 0.95),
            top_k=params.get("top_k", 20),
            min_p=params.get("min_p", 0.0),
            max_tokens=params.get("max_tokens", 96),
            repeat_penalty=repeat_penalty,
            grammar=grammar,
        )

        choice = response["choices"][0]
        text = (choice["message"]["content"] or "").strip()
        finish_reason = choice.get("finish_reason", "stop")

        if finish_reason == "length":
            logger.warning(f"finish_reason=length — max_tokens hit, ответ обрезан ({len(text)} chars)")

        return {"output": text, "finish_reason": finish_reason}

    except Exception as e:
        logger.exception("Ошибка генерации")
        raise HTTPException(500, detail=str(e))
    finally:
        gc.collect()


@app.get("/")
def health():
    return {"status": "ok", "model": os.path.basename(MODEL_PATH)}
