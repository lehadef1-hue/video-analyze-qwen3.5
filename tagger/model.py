"""
Model interface — sends requests to model_server.py via HTTP (vLLM backend).
Drop-in replacement for the transformers-based QwenVLModel.
"""

import os
import io
import base64
import logging
import requests
from PIL import Image

logger = logging.getLogger("tagger.model")

MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://localhost:8080/generate")

MAX_NEW_TOKENS = 512
TEMPERATURE    = 0.05


def _pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=82, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class QwenVLModel:
    """HTTP client that talks to model_server.py — same interface as the transformers version."""

    def __init__(self, model_id: str | None = None):
        # model_id ignored — model is managed by model_server
        self.model_server_url = MODEL_SERVER_URL

    def load(self):
        """No-op: model is loaded in model_server.py."""
        logger.info(f"Using remote model server at {self.model_server_url}")

    def analyze(
        self,
        images: list[Image.Image],
        prompt: str,
        segment_info: dict | None = None,
        verbose: bool = False,
    ) -> str:
        base64_images = [_pil_to_base64(img) for img in images]

        payload = {
            "prompt": prompt,
            "base64_images": base64_images,
            "sampling_params": {
                "temperature": TEMPERATURE,
                "top_p": 0.9,
                "max_tokens": MAX_NEW_TOKENS,
                "repetition_penalty": 1.15,
            },
        }

        if verbose:
            print(f"\n  [PROMPT LAST 300]: ...{prompt[-300:]}")

        try:
            resp = requests.post(self.model_server_url, json=payload, timeout=420)
            resp.raise_for_status()
            output = resp.json().get("output", "").strip()
            if verbose:
                print(f"  [RAW OUTPUT]: {output}")
            return output
        except Exception as e:
            logger.exception(f"Model server call failed: {e}")
            return ""
