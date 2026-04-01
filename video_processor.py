import os
import sys
import cv2
import base64
import io
import json
import re
import logging
from datetime import datetime
from pathlib import Path
from glob import glob
from PIL import Image
import requests
from typing import List, Dict, Optional

import shutil
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Optional performer recognition module
try:
    from performer_finder import identify_performers as _identify_performers
    PERFORMER_RECOGNITION_AVAILABLE = True
except ImportError:
    PERFORMER_RECOGNITION_AVAILABLE = False

# ─── Category logic (from local tagger) ─────────────────────────────────────
from tagger.categories import load_categories, build_canonical_map, build_category_prompt
from tagger.validate import validate_categories

_categories    = load_categories()
_canonical_map = build_canonical_map(_categories)
_category_prompt = build_category_prompt(_categories)

# ─── Logging Configuration ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-7s | %(threadName)-10s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("video_analyzer")

# ─── Configuration ──────────────────────────────────────────────────────────────
MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://localhost:8080/generate")

# ANALYSIS_PROMPT uses <<CATEGORIES>> as sentinel replaced at module load time
# so that {frame_count} remains available for per-call .format()
_ANALYSIS_PROMPT_BASE = """You receive {frame_count} key frames sampled evenly across the full video.

TASK: Analyze the video content and return structured metadata.

--- ORIENTATION ---
Choose EXACTLY ONE value: straight | gay | shemale
- straight  = male+female sex, OR all-female (lesbian = category tag, NOT an orientation)
- gay       = ONLY male performers having sex with each other — no women present at all
- shemale   = trans woman (MTF): visibly female body (breasts, feminine shape) WITH a penis visible

CRITICAL: "lesbian" is NOT a valid orientation — use "straight" instead.
CRITICAL: if you see a penis on a femininely-built performer → shemale, not straight.

--- DESCRIPTION ---
Write a vivid, explicit, dirty description (3-5 sentences).
Use raw, vulgar slang and dirty talk style. Describe the performers, their bodies, positions, actions in graphic detail.
Write like a horny human would describe the scene to a friend — be crude, playful and nasty. Do NOT be clinical or polite.
NEVER start with "This video", "In this video", "The video" or similar — jump straight into describing the action.

--- CATEGORIES ---
Tag ONLY what is CLEARLY and VISUALLY CONFIRMED across multiple frames.
Write 5–10 tags. HARD MAXIMUM: 10. WHEN IN DOUBT → OMIT. Fewer correct tags > many hallucinated tags.

Available categories:
<<CATEGORIES>>

--- STUDIO / WATERMARK ---
Look for any STATIC text overlay, watermark, or logo visible across the frames.
This is typically: a website URL (e.g. "brazzers.com"), a studio brand name (e.g. "Brazzers", "Reality Kings"),
or a channel name. It usually appears in a corner (top-left, bottom-right, etc.) and is semi-transparent or white text.

Rules:
- Return the text EXACTLY as it appears (preserve capitalisation and dots)
- If it's a URL like "www.site.com" or "site.com" — return just the domain, e.g. "site.com"
- If you see a logo/brand without readable text, describe it briefly, e.g. "Brazzers logo"
- Return null if no watermark is clearly readable in at least 2 frames

--- OUTPUT ---
Return ONLY valid JSON, no markdown, no extra text:
{{"orientation":"straight","description":"...","studio":null,"categories":[...]}}"""

ANALYSIS_PROMPT = _ANALYSIS_PROMPT_BASE.replace("<<CATEGORIES>>", _category_prompt)

FRAME_PROMPT = """You receive {frame_count} frames (indexed 0 to {last_idx}) from a video.

TASK: Score and select the 5 best frames for display quality.

SCORING — start at 10, subtract:
  -5 if ANY performer has closed eyes
  -4 if image is blurry (motion blur, out-of-focus)
  -3 if any performer's face is not fully visible or cut off
  -2 if image is dark or overexposed
  -1 if no explicit sexual act is clearly visible

SELECTION:
  - Evaluate all frames
  - Pick exactly 5 with HIGHEST scores (or all if fewer than 5 available)
  - For each selected frame: return index (0-based), score (1-10), reason (5–15 words)

--- OUTPUT ---
Return ONLY valid JSON, no markdown, no extra text:
{{"frames":[{{"index":3,"score":8,"reason":"..."}},...]}}"""

FINAL_FRAME_PROMPT = """You receive {frame_count} frames (indexed 0 to {last_idx}).
These are already pre-selected as the BEST candidates from the full video.

TASK: Pick the final 5 frames for display + 1 thumbnail.

SELECTION RULES:
  - Choose exactly 5 frames (or fewer if less than 5 available)
  - VARIETY is the top priority: different acts, positions, angles, moments — avoid similar-looking frames
  - Still reject: closed eyes (-5), blurry (-4), face cut off (-3), dark/overexposed (-2)
  - Re-score each chosen frame 1–10 applying the same penalties
  - For each selected frame: return index (0-based), score (1-10), reason (5–15 words)

THUMBNAIL (1 frame from your 5):
  - Best overall quality: sharp, eye contact, explicit action clearly visible
  - Must be the single most compelling frame for a preview image

--- OUTPUT ---
Return ONLY valid JSON, no markdown, no extra text:
{{"frames":[{{"index":3,"score":9,"reason":"..."}},...],  "thumbnailIndex":3}}"""

VALID_ORIENTATIONS = {"straight", "gay", "shemale"}


def pil_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.convert("RGB").save(buffered, format="JPEG", quality=82, optimize=True)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def call_vision_model(
    prompt: str,
    images: List[Image.Image],
    sampling: Optional[Dict] = None
) -> str:
    base64_images = [pil_to_base64(img) for img in images]
    payload = {
        "prompt": prompt,
        "base64_images": base64_images,
        "sampling_params": sampling or {
            "temperature": 0.65,
            "top_p": 0.90,
            "max_tokens": 1200
        }
    }
    try:
        logger.debug(f"POST → {MODEL_SERVER_URL} | images: {len(images)}")
        resp = requests.post(MODEL_SERVER_URL, json=payload, timeout=420)
        resp.raise_for_status()
        data = resp.json()
        output = data.get("output", "").strip()
        logger.debug(f"Model response: {len(output)} characters")
        return output
    except Exception:
        logger.exception("Model call error")
        return ""


def extract_json_from_response(text: str) -> Optional[Dict]:
    if not text:
        return None

    def _try_parse(s: str) -> Optional[Dict]:
        s = re.sub(r',\s*(?=[}\]])', '', s).strip()
        s = re.sub(r'(?<!\]),\s*"thumbnailIndex":\s*(\d+)\s*\}\s*$', r'], "thumbnailIndex": \1}', s)
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None

    for pattern in (r'```json\s*([\s\S]*?)```', r'```\s*([\s\S]*?)```'):
        m = re.search(pattern, text)
        if m:
            result = _try_parse(m.group(1))
            if result is not None:
                return result

    candidates = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:i + 1])
                start = None

    for candidate in sorted(candidates, key=len, reverse=True):
        result = _try_parse(candidate)
        if result is not None:
            return result

    if depth > 0 and start is not None:
        truncated = text[start:]
        for suffix in ('}' * depth, '}' * depth + '}'):
            result = _try_parse(truncated + suffix)
            if result is not None:
                return result

    result: Dict = {}
    for key in ("orientation", "description", "studio"):
        m = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if m:
            result[key] = m.group(1)
        else:
            m2 = re.search(rf'"{key}"\s*:\s*(null)', text)
            if m2:
                result[key] = None
    cats_m = re.search(r'"categories"\s*:\s*\[([^\]]*)', text)
    if cats_m:
        result["categories"] = re.findall(r'"((?:[^"\\]|\\.)*)"', cats_m.group(1))
    if result:
        logger.warning("Used regex fallback to recover partial JSON")
        return result

    logger.error("Failed to extract valid JSON from model response")
    logger.debug(f"Problem text:\n{text[:800]}")
    return None


def _normalize_cats(raw: list) -> List[str]:
    """Normalize categories via canonical map (handles aliases from categories.json)."""
    result, seen = [], set()
    for cat in raw:
        if not isinstance(cat, str):
            continue
        canonical = _canonical_map.get(cat.lower().strip())
        if canonical and canonical.lower() not in seen:
            result.append(canonical)
            seen.add(canonical.lower())
    return result


def extract_key_frames(
    video_path: str,
    target_count: int = 25,
    start_at: Optional[int] = None,
    end_at: Optional[int] = None,
) -> List[Image.Image]:
    """Extract target_count frames evenly from [start_at, end_at] range."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 8:
        cap.release()
        raise ValueError("Video too short")

    if start_at is None:
        start_at = max(1, int(total_frames * 0.04))
    if end_at is None:
        end_at = total_frames - max(1, int(total_frames * 0.04))

    usable = max(1, end_at - start_at)
    step = max(1, usable // target_count)

    frames = []
    for i in range(start_at, end_at, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
        if len(frames) >= target_count:
            break
    cap.release()
    logger.info(f"Extracted {len(frames)} frames from {video_path} (pos {start_at}–{end_at} / {total_frames})")
    return frames


def extract_frames_for_selection(video_path: str, target_count: int = 25) -> List[Image.Image]:
    """Extract frames from central region (8%–92%) for thumbnail selection."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total_frames < 8:
        raise ValueError("Video too short")
    skip = max(1, int(total_frames * 0.08))
    return extract_key_frames(video_path, target_count=target_count, start_at=skip, end_at=total_frames - skip)


def _parse_frame_candidates(parsed: Optional[Dict], frames: List[Image.Image]) -> List[Dict]:
    """Extract frame candidates from model response in unified format."""
    if not parsed:
        return []
    candidates = []
    for item in parsed.get("frames", []):
        idx = item.get("index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(frames):
            continue
        candidates.append({
            "frame": frames[idx],
            "score": int(item.get("score", 5)),
            "reason": str(item.get("reason", ""))[:80],
        })
    return candidates


def process_video(video_path: str, output_dir: str, base_name: str) -> Dict:
    """Three-pass video analysis with vision model for content extraction and frame selection."""
    logger.info(f"Processing: {base_name}")
    try:
        _cap = cv2.VideoCapture(video_path)
        total_frames_count = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        _cap.release()
        mid = total_frames_count // 2
        skip4  = max(1, int(total_frames_count * 0.04))
        skip8  = max(1, int(total_frames_count * 0.08))
        end92  = total_frames_count - skip8

        # ── Pass 1: Content analysis (orientation, description, categories) ─────
        frames_1a = extract_key_frames(video_path, 25, start_at=skip4, end_at=end92)
        if len(frames_1a) < 4:
            return {"status": "skipped", "reason": "too few frames"}

        raw1a = call_vision_model(
            ANALYSIS_PROMPT.format(frame_count=len(frames_1a)),
            frames_1a,
            {"temperature": 0.45, "top_p": 0.85, "max_tokens": 1000}
        )
        p1a = extract_json_from_response(raw1a)
        if not p1a:
            return {"status": "error", "reason": "pass1a invalid response"}

        description  = p1a.get("description", "").strip()
        orientation  = p1a.get("orientation", "straight")
        if orientation not in VALID_ORIENTATIONS:
            orientation = "straight"
        cats_a       = _normalize_cats(p1a.get("categories", []))[:15]
        studio_raw   = p1a.get("studio")
        studio       = str(studio_raw).strip() if studio_raw and str(studio_raw).strip().lower() not in ("null", "none", "") else None
        logger.info(f"Pass 1: orient={orientation} cats={len(cats_a)} studio={studio!r}")

        final_categories = validate_categories(cats_a, orientation)

        # ── Pass 2a: Frame selection (first half) ────────────────────────────────
        frames_2a = extract_key_frames(video_path, 25, start_at=skip8, end_at=mid)
        raw2a = call_vision_model(
            FRAME_PROMPT.format(frame_count=len(frames_2a), last_idx=len(frames_2a) - 1),
            frames_2a,
            {"temperature": 0.3, "top_p": 0.80, "max_tokens": 1200}
        )
        p2a        = extract_json_from_response(raw2a)
        candidates_a = _parse_frame_candidates(p2a, frames_2a)
        logger.info(f"Pass 2a: candidates={len(candidates_a)}")

        # ── Pass 2b: Frame selection (second half) ────────────────────────────────
        frames_2b = extract_key_frames(video_path, 25, start_at=mid, end_at=end92)
        raw2b = call_vision_model(
            FRAME_PROMPT.format(frame_count=len(frames_2b), last_idx=len(frames_2b) - 1),
            frames_2b,
            {"temperature": 0.3, "top_p": 0.80, "max_tokens": 1200}
        )
        p2b = extract_json_from_response(raw2b)
        candidates_b = _parse_frame_candidates(p2b, frames_2b)
        logger.info(f"Pass 2b: candidates={len(candidates_b)}")

        # ── Merge candidates for Pass 3 ────────────────────────────────────────────
        all_candidates = sorted(
            candidates_a + candidates_b,
            key=lambda x: x["score"],
            reverse=True
        )
        top10 = all_candidates[:10]
        logger.info(f"Merged: {len(all_candidates)} candidates → top {len(top10)} for Pass 3")

        # ── Pass 3: Final selection from top-10 ────────────────────────────────────
        top5 = top10[:5]
        thumb_frame = top10[0]["frame"] if top10 else None

        if top10:
            frames_3 = [c["frame"] for c in top10]
            raw3 = call_vision_model(
                FINAL_FRAME_PROMPT.format(frame_count=len(frames_3), last_idx=len(frames_3) - 1),
                frames_3,
                {"temperature": 0.3, "top_p": 0.80, "max_tokens": 1200}
            )
            p3 = extract_json_from_response(raw3)
            candidates_3 = _parse_frame_candidates(p3, frames_3)
            thumb_idx_3 = (p3 or {}).get("thumbnailIndex")
            logger.info(f"Pass 3: candidates={len(candidates_3)}")

            if candidates_3:
                top5 = sorted(candidates_3, key=lambda x: x["score"], reverse=True)[:5]
                if isinstance(thumb_idx_3, int) and 0 <= thumb_idx_3 < len(frames_3):
                    thumb_frame = frames_3[thumb_idx_3]
                elif top5:
                    thumb_frame = top5[0]["frame"]

        # ── Performer recognition (optional) ────────────────────────────────────────
        performers: List[str] = []
        if PERFORMER_RECOGNITION_AVAILABLE:
            frames_face = extract_key_frames(video_path, 100, start_at=skip4, end_at=end92)
            performers = _identify_performers(frames_face)
            if performers:
                logger.info(f"Performers identified: {performers}")

        # ── Save results ───────────────────────────────────────────────────────────
        os.makedirs(output_dir, exist_ok=True)
        saved_frames = []
        for i, cand in enumerate(top5):
            path = os.path.join(output_dir, f"{base_name}_frame_{i:03d}.jpg")
            cand["frame"].save(path, quality=85, optimize=True)
            saved_frames.append({"score": cand["score"], "path": path, "reason": cand["reason"]})

        thumb_path = None
        if thumb_frame is not None:
            thumb_path = os.path.join(output_dir, f"{base_name}_thumb.jpg")
            thumb_frame.save(thumb_path, quality=88, optimize=True)

        meta = {
            "description": description,
            "categories": final_categories[:15],
            "orientation": orientation,
            "studio": studio,
            "performers": performers,
        }
        with open(os.path.join(output_dir, f"{base_name}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info(f"Done: {base_name} | orient={orientation} studio={studio!r} cats={len(final_categories)} frames={len(saved_frames)}")
        return {
            "status": "ok",
            "thumbnail": thumb_path,
            "top5_frames": saved_frames,
            "categories": final_categories,
            "orientation": orientation,
            "studio": studio,
            "performers": performers,
            "description": description[:400] if description else None,
        }

    except Exception as e:
        logger.exception(f"Critical error processing {video_path}")
        return {"status": "error", "reason": str(e)}


# ─── FastAPI Application ────────────────────────────────────────────────────────
app = FastAPI(
    title="Adult Video Analyzer",
    description="Video analysis: thumbnails, categories, descriptions",
    version="0.1"
)

class ProcessRequest(BaseModel):
    input_dir: str = "/workspace/video/videos"
    output_dir: str = "/workspace/video/result"


templates = Jinja2Templates(directory="templates")
app.mount("/results", StaticFiles(directory="/workspace/video/result"), name="results")


UPLOAD_DIR = "/workspace/video/videos"

@app.post("/upload")
async def upload_videos(files: List[UploadFile] = File(...)):
    """Upload video files to input directory."""
    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    uploaded = []
    for f in files:
        dest = Path(UPLOAD_DIR) / f.filename
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        uploaded.append(f.filename)
    logger.info(f"Uploaded {len(uploaded)} file(s): {uploaded}")
    return {"uploaded": uploaded, "count": len(uploaded)}


@app.get("/browse")
async def browse_results(request: Request):
    base = Path("/workspace/video/result")
    runs = []

    for run_dir in sorted(base.iterdir(), reverse=True):
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
        run_time = run_dir.name.replace("run_", "").replace("_", " ")
        videos = []

        for video_dir in run_dir.iterdir():
            if not video_dir.is_dir():
                continue
            meta_file = video_dir / f"{video_dir.name}_meta.json"
            if not meta_file.exists():
                continue

            with open(meta_file, encoding="utf-8") as f:
                meta = json.load(f)

            thumb_rel = f"/results/{run_dir.name}/{video_dir.name}/{video_dir.name}_thumb.jpg"
            thumb_abs = Path("/workspace/video/result") / run_dir.name / video_dir.name / f"{video_dir.name}_thumb.jpg"
            thumb = thumb_rel if thumb_abs.exists() else None

            frames = []
            for f in sorted(video_dir.glob(f"{video_dir.name}_frame_*.jpg")):
                rel_path = f"/results/{run_dir.name}/{video_dir.name}/{f.name}"
                frames.append(rel_path)

            videos.append({
                "name": video_dir.name,
                "thumbnail": thumb,
                "frames": frames[:8],
                "categories": meta.get("categories", []),
                "orientation": meta.get("orientation", ""),
                "studio": meta.get("studio") or "",
                "performers": meta.get("performers") or [],
                "description": meta.get("description", "—")
            })

        runs.append({
            "name": run_dir.name,
            "time": run_time,
            "videos": videos
        })

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "runs": runs}
    )


@app.post("/process")
def process_videos_endpoint(req: ProcessRequest):
    logger.info(f"Request → input: {req.input_dir} | output: {req.output_dir}")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"run_{run_timestamp}"
    base_output = Path(req.output_dir)
    this_run_dir = base_output / run_dir_name
    this_run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results → {this_run_dir}")

    videos = []
    for pattern in ("*.mp4", "*.mkv", "*.webm", "*.mov"):
        videos.extend(glob(os.path.join(req.input_dir, pattern)))

    if not videos:
        return {"status": "nothing_to_process", "videos_found": 0}

    results = []
    for path in sorted(videos):
        base_name = Path(path).stem
        video_output_dir = this_run_dir / base_name
        video_output_dir.mkdir(exist_ok=True)

        result = process_video(path, str(video_output_dir), base_name)
        results.append({
            "file": base_name,
            "result": result,
            "output_folder": str(video_output_dir)
        })

    return {
        "status": "processed",
        "run_folder": str(this_run_dir),
        "run_timestamp": run_timestamp,
        "results": results
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting video analyzer server on :8000")
    uvicorn.run(
        "video_processor:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        workers=1
    )
