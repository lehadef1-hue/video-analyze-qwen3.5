import os
import sys
import base64
import io
import gc
import logging
import multiprocessing
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

# vLLM >= 0.12: StructuredOutputsParams + structured_outputs
# vLLM 0.4–0.11: GuidedDecodingParams + guided_decoding
_StructuredOutputsParams = None
_GuidedDecodingParams = None
try:
    from vllm.sampling_params import StructuredOutputsParams as _StructuredOutputsParams
except ImportError:
    try:
        from vllm.sampling_params import GuidedDecodingParams as _GuidedDecodingParams
    except ImportError:
        pass  # guided generation not available in this vLLM build
import torch


# 🔒 ВАЖНО: защита от fork-повторной загрузки
multiprocessing.set_start_method("spawn", force=True)


# ─── Логирование ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("model_server")


# ─── ENV ────────────────────────────────────────────────
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["HF_HUB_CACHE"] = "/workspace/hf_cache/hub"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf_cache/hub"
os.environ["HF_HUB_DISABLE_XET"] = "1"

# ─── Выбор модели ───────────────────────────────────────
# Поддерживаемые модели:
#   qwen3vl   → Qwen/Qwen3-VL-30B-A3B-Instruct-FP8   (vision, image mode)
#   qwen35    → Qwen/Qwen3.5-27B-FP8                  (native multimodal, video mode)
MODEL_PRESET = os.getenv("MODEL_PRESET", "qwen3vl")

_PRESETS = {
    "qwen3vl": {
        "model_name":  "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
        "max_model_len": 32768,
        "limit_mm_per_prompt": {"image": 25},
        "mm_processor_kwargs": {
            "min_pixels": 256 * 28 * 28,
            "max_pixels": 512 * 28 * 28,
        },
    },
    "qwen35": {
        "model_name":  "Qwen/Qwen3.5-27B-FP8",
        "max_model_len": 32768,
        "limit_mm_per_prompt": {"image": 16},
        "mm_processor_kwargs": {
            "min_pixels": 256 * 28 * 28,
            "max_pixels": 512 * 28 * 28,
        },
    },
}

if MODEL_PRESET not in _PRESETS:
    logger.critical(f"Неизвестный MODEL_PRESET={MODEL_PRESET!r}. Допустимые: {list(_PRESETS)}")
    sys.exit(1)

_preset   = _PRESETS[MODEL_PRESET]
MODEL_NAME = _preset["model_name"]
CACHE_DIR  = "/workspace/hf_cache/hub"

logger.info(f"MODEL_PRESET={MODEL_PRESET}  →  {MODEL_NAME}")


# ─── Проверка GPU ───────────────────────────────────────
if not torch.cuda.is_available():
    logger.critical("CUDA НЕ ДОСТУПНА")
    sys.exit(1)

total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
free = torch.cuda.mem_get_info()[0] / (1024**3)

logger.info(f"GPU total: {total:.2f} GB")
logger.info(f"GPU free:  {free:.2f} GB")

if free < 35:
    gpu_util = 0.60
else:
    gpu_util = 0.90

logger.info(f"gpu_memory_utilization = {gpu_util}")


# ─── Загрузка модели ────────────────────────────────────
logger.info("Загрузка процессора...")
processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    cache_dir=CACHE_DIR,
)

logger.info("Загрузка LLM через vLLM...")
llm = LLM(
    model=MODEL_NAME,
    download_dir=CACHE_DIR,
    dtype="auto",
    gpu_memory_utilization=gpu_util,
    max_model_len=_preset["max_model_len"],
    enforce_eager=True,
    tensor_parallel_size=1,
    limit_mm_per_prompt=_preset["limit_mm_per_prompt"],
    mm_processor_kwargs=_preset["mm_processor_kwargs"],
)

logger.info("✓ Модель успешно загружена")


# ─── FastAPI ────────────────────────────────────────────
app = FastAPI(title=f"Model Server [{MODEL_PRESET}]")


class GenerateRequest(BaseModel):
    prompt: str
    base64_images: List[str] = []
    fps: Optional[float] = None    # if set → video mode (frames sent as video sequence)
    sampling_params: Optional[Dict[str, Any]] = None
    enable_thinking: bool = False  # Qwen3 /think mode — outputs <think>…</think> before answer
    guided_json: Optional[Dict[str, Any]] = None  # JSON schema → forced structured output


def decode_base64_image(b64: str) -> Image.Image:
    img_bytes = base64.b64decode(b64)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


@app.post("/generate")
async def generate(request: GenerateRequest):
    images = []
    outputs = None
    # vLLM video mode (multi_modal_data "video") requires VideoMetadata — not supported via PIL.
    # Both grid mode and frame mode use image inputs; fps param is informational only.
    try:
        images = [decode_base64_image(b) for b in request.base64_images]

        if request.fps is not None and images:
            logger.info(f"Frame mode: {len(images)} sequential frames @ {request.fps} fps  preset={MODEL_PRESET}")

        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": request.prompt})

        conversation = [{"role": "user", "content": content}]

        full_prompt = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=request.enable_thinking,
        )

        if request.enable_thinking:
            logger.info("Thinking mode enabled for this request")

        # Default sampling params differ per model
        if MODEL_PRESET == "qwen35":
            default_params = {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "presence_penalty": 1.5,
                "max_tokens": 512,
            }
        else:
            default_params = {
                "temperature": 0.7,
                "top_p": 0.95,
                "max_tokens": 96,
            }

        params = request.sampling_params or default_params
        params.setdefault("repetition_penalty", 1.15)

        if request.guided_json:
            if _StructuredOutputsParams is not None:
                params["structured_outputs"] = _StructuredOutputsParams(json=request.guided_json)
            elif _GuidedDecodingParams is not None:
                params["guided_decoding"] = _GuidedDecodingParams(json=request.guided_json)
            else:
                logger.warning("Structured output not supported by this vLLM build — skipping")
        sampling = SamplingParams(**params)

        if images:
            outputs = llm.generate(
                {
                    "prompt": full_prompt,
                    "multi_modal_data": {"image": images},
                },
                sampling,
            )
        else:
            outputs = llm.generate(full_prompt, sampling)

        out = outputs[0].outputs[0]
        text = out.text.strip()
        finish_reason = out.finish_reason  # "stop" or "length" (tokens exhausted)

        if finish_reason == "length":
            logger.warning(f"⚠️  finish_reason=length — max_tokens hit, response truncated ({len(text)} chars)")

        return {"output": text, "finish_reason": finish_reason}

    except Exception as e:
        logger.exception("Ошибка генерации")
        raise HTTPException(500, detail=str(e))
    finally:
        # Явно освобождаем PIL-изображения и vLLM outputs — иначе накапливаются в RAM
        for img in images:
            img.close()
        del images, outputs
        gc.collect()


@app.get("/")
def health():
    return {"status": "ok", "model": MODEL_NAME}
