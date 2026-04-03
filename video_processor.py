"""
video_processor_v2.py — Advanced adult video analyzer with SEO output.
Runs on :8001 by default.
"""

import os, sys, cv2, base64, io, json, re, logging, shutil, subprocess, uuid, threading, queue
from dotenv import load_dotenv; load_dotenv()
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import requests
from PIL import Image
from fastapi import FastAPI, Request, UploadFile, File, Form, Header, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ─── Performer recognition ────────────────────────────────────────────────────
try:
    from performer_finder import (
        detect_embeddings, cluster_embeddings, match_centroids,
        load_db as load_performer_db,
    )
    PERFORMER_RECOGNITION_AVAILABLE = True
except ImportError:
    PERFORMER_RECOGNITION_AVAILABLE = False

# ─── Category tagger (scene/grid analysis) ───────────────────────────────────
from tagger import VideoTagger
_tagger = VideoTagger()
_tagger.load_model()

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("analyzer_v2")

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_SERVER_URL  = os.getenv("MODEL_SERVER_URL",  "http://localhost:8080/generate")
UPLOAD_DIR        = os.getenv("UPLOAD_DIR",        "/workspace/video/videos")
OUTPUT_DIR        = os.getenv("OUTPUT_DIR",        "/workspace/video/result")
PERFORMER_DB_PATH = os.getenv("PERFORMER_DB_PATH", "/workspace/my_performers.pkl")
API_KEY           = os.getenv("API_KEY", "")          # empty = auth disabled

VALID_ORIENTATIONS = {"straight", "gay", "shemale"}

# ─── Content blocklist ────────────────────────────────────────────────────────
# Word substitutions — replace problematic terms with neutral synonyms
_SYNONYMS: Dict[str, str] = {
    "abused":       "dominated",
    "asphyxia":     "breath play",
    "behead":       "overpowered",
    "bleed":        "marked",
    "blood":        "passion marks",
    "choke":        "throat grip",
    "choking":      "throat gripping",
    "decapitation": "submission",
    "drugged":      "seduced",
    "forced":       "taken",
    "hidden cam":   "candid",
    "kill":         "destroyed",
    "leaked":       "released",
    "murder":       "dominated hard",
    "rape":         "ravaged",
    "snuff":        "fantasy play",
    "strangle":     "neck grabbed",
    "torture":      "intense play",
    "upskirt":      "voyeur angle",
    "downblouse":   "cleavage view",
    "child":        "teen",
    "children":     "teens",
    "kid":          "teen",
    "15yo":         "teen",
    "16yo":         "teen",
    "17yo":         "teen",
    "cp":           "teen porn",
    "scat":         "shit",
    "loli":         "teen",
    "l.o.l.i":      "teen",
    "shota":        "teen",
    "gore":         "passion marks",
}

_SYNONYM_RE = re.compile(
    "|".join(r"\b" + re.escape(w) + r"\b" for w in sorted(_SYNONYMS, key=len, reverse=True)),
    re.IGNORECASE,
)


def _replace(m: re.Match) -> str:
    return _SYNONYMS.get(m.group(0).lower(), m.group(0))


def _redact_blocked(text: str) -> str:
    return _SYNONYM_RE.sub(_replace, text)


def _filter_blocked_list(items: List[str]) -> List[str]:
    return [_SYNONYM_RE.sub(_replace, item) for item in items]


DESCRIPTION_STYLES = {
    "standard": (
        "Write a vivid, explicit, dirty description (3–4 sentences, MAX 350 characters). "
        "Use raw, vulgar slang and dirty talk style. Describe performers, bodies, positions, actions in graphic detail. "
        "Write like a horny human — be crude, playful, nasty. Do NOT be clinical or polite. "
        "NEVER start with 'This video', 'In this video', or similar — jump straight into describing the action. "
        "Stop after 4 sentences maximum."
    ),
    "clean": (
        "Write a tasteful description (3–4 sentences, MAX 350 characters) suitable for mainstream platforms. "
        "Focus on mood, setting, performers' appearance, and the nature of the encounter. "
        "Avoid explicit sexual terms. NEVER start with 'This video', 'In this video', or similar. "
        "Stop after 4 sentences maximum."
    ),
    "cinematic": (
        "Write a cinematic description (3–4 sentences, MAX 350 characters) in the style of a film critic. "
        "Focus on visual composition, lighting quality, performers' chemistry, camera angles. "
        "Treat it as a review of visual and performative qualities. NEVER start with 'This video' or similar. "
        "Stop after 4 sentences maximum."
    ),
}

# ─── Prompts ──────────────────────────────────────────────────────────────────

def build_analysis_prompt(frame_count: int, ts_map: str, desc_style: str, language: str) -> str:
    return f"""You are watching a porn video. You receive {frame_count} frames sampled evenly across the full video duration.

FRAME TIMESTAMPS (seconds from video start):
{ts_map}

STRICT RULE — applies to ALL sections below:
Describe ONLY what is CLEARLY and UNAMBIGUOUSLY VISIBLE in the frames.
WHEN IN DOUBT → OMIT. Do NOT infer, assume, or invent details not directly visible.
If something is partially visible or unclear — do not mention it.

--- ORIENTATION ---
Choose EXACTLY ONE value: straight | gay | shemale
Base your choice ONLY on performers clearly visible in the frames.
- straight  = male+female sex, OR all-female (lesbian = category tag, NOT an orientation)
- gay       = ONLY male performers having sex with each other — no women present at all
- shemale   = trans woman (MTF): visibly feminine body (breasts) WITH a penis visible

CRITICAL: "lesbian" is NOT a valid orientation → use "straight".
CRITICAL: feminine body + visible penis → "shemale", not "straight".

--- CONTENT TYPE ---
Before writing the description, identify what type of content this is:
• Real human performers only → describe normally
• Animated/CGI content (anime, 3D, cartoon, hentai) → say so explicitly in description
• MIXED: real human performer + animated/CGI elements (tentacles, monster, creature, animated appendages) → describe non-human elements accurately. If penetration is by a tentacle or CGI creature — call it exactly that, NOT "penis" or "cock". Do NOT misidentify non-human objects as human anatomy.

--- DESCRIPTION ---
{desc_style}
IMPORTANT: describe only what is directly visible in the frames. Do not add acts, body details, or setting elements that are not clearly shown. If a detail is not visible — do not include it.

--- WATERMARKS / ON-SCREEN TEXT ---
List ONLY text overlays or logos that are clearly readable across the frames.
Do NOT guess or reconstruct partially visible text.
Return exact text as array (e.g. ["brazzers.com", "pornhub.com"]). Return [] if none.

--- OUTPUT ---
Return ONLY valid JSON, no markdown. All text fields in {language}.
{{"orientation":"straight","description":"...","watermarks":["site.com"]}}"""



def build_seo_prompt(description: str, categories: List[str], orientation: str, language: str,
                     tag_count: int = 5, secondary_tag_count: int = 7) -> str:
    cats_str = ", ".join(categories)
    return f"""You are a professional SEO specialist for an adult content website. Your task is to generate fully optimized SEO metadata based on the provided video information.

VIDEO DESCRIPTION:
{description}

CATEGORIES: {cats_str}
ORIENTATION: {orientation}

GENERATE THE FOLLOWING:

1. META TITLE — 50–60 characters total.
   - Include the most important keyword phrase naturally.
   - Make it compelling and click-worthy. No ALL CAPS, no excessive punctuation.

2. META DESCRIPTION — 140–160 characters total.
   - Natural flowing sentence, not a keyword dump.
   - Include 2–3 relevant keyword phrases. Should entice users to click.

3. PRIMARY TAGS — up to {tag_count} long-tail keyword phrases (3–6 words each). Include as many as are accurate and relevant — do not pad with low-quality tags to reach the limit.
   - Exact search queries real users type. Based on acts, appearance, setting.
   - Format: lowercase phrases, e.g. "amateur outdoor sex video"

4. SECONDARY TAGS — up to {secondary_tag_count} shorter keyword phrases (2–4 words). Include as many as are accurate and relevant — do not pad with low-quality tags to reach the limit.
   - Broader supporting search terms. Mix acts, appearance, category keywords.
   - Format: lowercase phrases

5. SEO DESCRIPTION — 2–3 short paragraphs (80–120 words total).
   - Natural readable prose, not keyword stuffing. Start with main action and performers.
   - Mention key visual details, setting, mood. Avoid the most extreme profanity but explicit terms are acceptable.
   - Keyword-rich but reads naturally. MUST NOT be empty — always generate this field.

--- OUTPUT ---
Return ONLY valid JSON, no markdown, no extra text. Stop immediately after the closing brace. All text in {language}.
IMPORTANT: primary_tags array must contain AT MOST {tag_count} items. secondary_tags array must contain AT MOST {secondary_tag_count} items. Do NOT generate more tags than requested.
{{"meta_title":"...","meta_description":"...","primary_tags":[...],"secondary_tags":[...],"seo_description":"..."}}"""


def build_seo_translate_prompt(meta_title: str, meta_desc: str, seo_description: str, language: str) -> str:
    return f"""Translate the following adult video SEO texts into {language}.
Rules:
- Keep meaning, tone, and keywords accurate.
- seo_description must stay under 150 words. Do NOT add new sentences. Do NOT repeat any phrase.
- Return ONLY valid JSON, no markdown, no extra text.

SOURCE TEXTS:
meta_title: {meta_title}
meta_description: {meta_desc}
seo_description: {seo_description}

{{"meta_title":"...","meta_description":"...","seo_description":"..."}}"""



# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fmt_ts(seconds: float) -> str:
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


_MAX_IMG_SIDE = 640  # max pixels per side before encoding — keeps image tokens ≤ 512/image

def pil_to_base64(img: Image.Image) -> str:
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > _MAX_IMG_SIDE:
        scale = _MAX_IMG_SIDE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def call_vision_model(
    prompt: str,
    images: List[Image.Image],
    sampling: Optional[Dict] = None,
    enable_thinking: bool = False,
    guided_json: Optional[Dict] = None,
    pass_name: str = "?",
    fps: Optional[float] = None,
) -> str:
    payload = {
        "prompt": prompt,
        "base64_images": [pil_to_base64(img) for img in images],
        "sampling_params": sampling or {"temperature": 0.65, "top_p": 0.90, "max_tokens": 2048},
        "enable_thinking": enable_thinking,
    }
    if fps is not None:
        payload["fps"] = fps
    if guided_json:
        payload["guided_json"] = guided_json
    try:
        r = requests.post(MODEL_SERVER_URL, json=payload, timeout=420)
        r.raise_for_status()
        full_resp = r.json()
        out = full_resp.get("output", "").strip()
        finish_reason = full_resp.get("finish_reason", full_resp.get("stop_reason", "unknown"))
        max_tok = payload['sampling_params'].get('max_tokens')
        logger.info(f"[{pass_name}] response: {len(out)} chars, finish_reason={finish_reason}, max_tokens={max_tok}")
        if finish_reason in ("length", "max_tokens"):
            logger.warning(f"[{pass_name}] hit token limit (max_tokens={max_tok}) — response likely truncated")
        return out
    except Exception:
        logger.exception(f"[{pass_name}] model call failed")
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
            r = _try_parse(m.group(1))
            if r is not None:
                return r

    candidates = []
    depth, start = 0, None
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

    for cand in sorted(candidates, key=len, reverse=True):
        r = _try_parse(cand)
        if r is not None:
            return r

    # Truncated JSON — append missing closing braces
    if depth > 0 and start is not None:
        truncated = text[start:]
        for suffix in ('}' * depth, '}' * depth + '}'):
            r = _try_parse(truncated + suffix)
            if r is not None:
                return r

    # Regex fallback for Pass 1 fields
    result: Dict = {}
    for key in ("orientation", "description"):
        m = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if m:
            result[key] = m.group(1)
    if result:
        logger.warning("Used regex fallback for JSON")
        return result

    last_open  = text.rfind('{')
    last_close = text.rfind('}')
    logger.error(
        f"JSON parse failed. len={len(text)}, "
        f"last_open={last_open}, last_close={last_close}, "
        f"tail={repr(text[-80:])}"
    )
    return None


def extract_key_frames_ts(
    video_path: str,
    target_count: int = 25,
    start_at: Optional[int] = None,
    end_at: Optional[int] = None,
) -> Tuple[List[Image.Image], List[float]]:
    """Extract frames and their timestamps (seconds)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    if total < 8:
        cap.release()
        raise ValueError("Video too short")
    if start_at is None:
        start_at = max(1, int(total * 0.04))
    if end_at is None:
        end_at = total - max(1, int(total * 0.04))
    usable = max(1, end_at - start_at)
    step = max(1, usable // target_count)
    frames, timestamps = [], []
    for i in range(start_at, end_at, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        timestamps.append(float(i / fps))
        if len(frames) >= target_count:
            break
    cap.release()
    logger.info(f"Extracted {len(frames)} frames ({start_at}–{end_at}/{total})")
    return frames, timestamps








def _seo_fallback(text: str) -> Dict:
    """Extract SEO fields from truncated/malformed JSON via regex."""
    result: Dict = {}
    for key in ("meta_title", "meta_description", "seo_description"):
        m = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if m:
            result[key] = m.group(1)
    for key in ("primary_tags", "secondary_tags"):
        m = re.search(rf'"{key}"\s*:\s*\[([^\]]*)', text)
        if m:
            items = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
            if items:
                result[key] = items
    if result:
        logger.warning(f"SEO regex fallback used — recovered: {list(result.keys())}")
    return result


# ─── Core processor ───────────────────────────────────────────────────────────

def process_video_v2(
    video_path: str,
    output_dir: str,
    base_name: str,
    language: str = "English",
    style: str = "standard",
    extra_languages: Optional[List[str]] = None,   # additional lang codes ["de","es",...]
    tag_count: int = 5,
    secondary_tag_count: int = 7,
    category_count: int = 10,
) -> Dict:
    logger.info(f"Processing v2: {base_name} | lang={language} style={style}")
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        skip4 = max(1, int(total_frames * 0.04))
        skip8 = max(1, int(total_frames * 0.08))
        end92 = total_frames - skip8

        # ── Pass 1: Analysis + key scenes ──────────────────────────────────────
        frames_1a, ts_1a = extract_key_frames_ts(video_path, 25, start_at=skip4, end_at=end92)
        if len(frames_1a) < 4:
            return {"status": "skipped", "reason": "too few frames"}

        ts_map = "  ".join(f"F{i}={_fmt_ts(t)}" for i, t in enumerate(ts_1a))
        desc_style = DESCRIPTION_STYLES.get(style, DESCRIPTION_STYLES["standard"])

        logger.info(f"Pass1 start: frames={len(frames_1a)} lang={language}")
        raw1 = call_vision_model(
            build_analysis_prompt(len(frames_1a), ts_map, desc_style, language),
            frames_1a,
            {"temperature": 0.45, "top_p": 0.85, "max_tokens": 2000},
            pass_name="Pass1",
        )
        p1 = extract_json_from_response(raw1)
        if not p1:
            return {"status": "error", "reason": "pass1 invalid response"}

        description = _redact_blocked(p1.get("description", "").strip())
        orientation = p1.get("orientation", "straight")
        if orientation not in VALID_ORIENTATIONS:
            orientation = "straight"
        watermarks = [str(w).strip() for w in (p1.get("watermarks") or []) if str(w).strip()]

        # ── Categories via VideoTagger (orientation from Pass1) ──────────────────
        logger.info(f"Cats-tagger start: orient_pass1={orientation}")
        tagger_result = _tagger.tag_video(video_path, orientation=orientation, verbose=True)
        cats_raw = _filter_blocked_list(tagger_result["categories"])
        # Sort by frequency descending, then trim to category_count
        counts = tagger_result.get("category_counts", {})
        cats_raw = sorted(cats_raw, key=lambda c: counts.get(c, 0), reverse=True)
        final_categories = cats_raw

        # Build key_scenes from tagger scene segments (no redundant model pass)
        key_scenes = [
            {"ts": seg["start_sec"], "formatted": _fmt_ts(seg["start_sec"])}
            for seg in tagger_result.get("scene_segments", [])
        ]
        logger.info(f"Pass 1+cats done: orient={orientation} cats={final_categories} watermarks={watermarks} scenes={len(key_scenes)}")

        # ── Pass SEO (multi-language) ────────────────────────────────────────
        final_categories = final_categories[:category_count]

        # Build list of (lang_code, lang_name) — first entry is primary language
        all_langs: List[Tuple[str, str]] = []
        # primary
        primary_code = next(
            (k for k, v in LANG_MAP.items() if v.lower() == language.lower()), "en"
        )
        all_langs.append((primary_code, language))
        # extras
        for code in (extra_languages or []):
            lc = code.lower()
            if lc in LANG_MAP and lc != primary_code:
                all_langs.append((lc, LANG_MAP[lc]))

        seo_by_lang: Dict[str, Dict] = {}

        # Primary language — full SEO (title + desc + tags)
        (p_lang_code, p_lang_name) = all_langs[0]
        _SEO_SCHEMA = {
            "type": "object",
            "properties": {
                "meta_title":       {"type": "string"},
                "meta_description": {"type": "string"},
                "seo_description":  {"type": "string"},
                "primary_tags":     {"type": "array", "items": {"type": "string"}},
                "secondary_tags":   {"type": "array", "items": {"type": "string"}},
            },
            "required": ["meta_title", "meta_description", "seo_description", "primary_tags", "secondary_tags"],
        }
        logger.info(f"SEO-{p_lang_code} start: cats={len(final_categories)}")
        raw_seo = call_vision_model(
            build_seo_prompt(description[:500], final_categories, orientation, p_lang_name,
                             tag_count, secondary_tag_count),
            [],
            {"temperature": 0.3, "top_p": 0.85, "max_tokens": 4096, "repetition_penalty": 1.3},
            guided_json=_SEO_SCHEMA,
            pass_name=f"SEO-{p_lang_code}",
        )
        p_seo = extract_json_from_response(raw_seo) or _seo_fallback(raw_seo or "")
        primary_tags   = _filter_blocked_list([t.strip() for t in p_seo.get("primary_tags", []) if t.strip()][:tag_count])
        secondary_tags = _filter_blocked_list([t.strip() for t in p_seo.get("secondary_tags", []) if t.strip()][:secondary_tag_count])
        seo_desc = _redact_blocked(p_seo.get("seo_description", "").strip())
        # Fallback: if seo_description empty, use meta_description
        if not seo_desc:
            seo_desc = _redact_blocked(p_seo.get("meta_description", "").strip())
            if seo_desc:
                logger.warning("seo_description was empty — using meta_description as fallback")
        seo_by_lang[p_lang_code] = {
            "meta_title":       _redact_blocked(p_seo.get("meta_title", "").strip()),
            "meta_description": _redact_blocked(p_seo.get("meta_description", "").strip()),
            "seo_description":  seo_desc,
        }
        logger.info(f"Pass SEO [{p_lang_code}]: title={len(seo_by_lang[p_lang_code]['meta_title'])}")

        # Extra languages — translate only meta_title, meta_description, seo_description
        base_title    = seo_by_lang[p_lang_code]["meta_title"]
        base_meta     = seo_by_lang[p_lang_code]["meta_description"]
        base_seo_desc = seo_by_lang[p_lang_code]["seo_description"]
        for lang_code, lang_name in all_langs[1:]:
            _SEO_TR_SCHEMA = {
                "type": "object",
                "properties": {
                    "meta_title":       {"type": "string"},
                    "meta_description": {"type": "string"},
                    "seo_description":  {"type": "string"},
                },
                "required": ["meta_title", "meta_description", "seo_description"],
            }
            logger.info(f"SEO-tr-{lang_code} start")
            raw_tr = call_vision_model(
                build_seo_translate_prompt(base_title, base_meta, base_seo_desc, lang_name),
                [],
                {"temperature": 0.2, "top_p": 0.80, "max_tokens": 2048},
                guided_json=_SEO_TR_SCHEMA,
                pass_name=f"SEO-tr-{lang_code}",
            )
            p_tr = extract_json_from_response(raw_tr) or {}
            seo_by_lang[lang_code] = {
                "meta_title":       _redact_blocked(p_tr.get("meta_title", "").strip()),
                "meta_description": _redact_blocked(p_tr.get("meta_description", "").strip()),
                "seo_description":  _redact_blocked(p_tr.get("seo_description", "").strip()),
            }
            logger.info(f"Pass SEO translate [{lang_code}]: title={len(seo_by_lang[lang_code]['meta_title'])}")

        # Flat fields from primary language for backward compat
        primary_seo     = seo_by_lang[p_lang_code]
        meta_title      = primary_seo["meta_title"]
        meta_desc       = primary_seo["meta_description"]
        seo_description = primary_seo["seo_description"]

        # ── Performer recognition ────────────────────────────────────────────────
        performers: List[Dict] = []
        if PERFORMER_RECOGNITION_AVAILABLE:
            try:
                frames_face, _ = extract_key_frames_ts(video_path, 100, start_at=skip4, end_at=end92)
                db = load_performer_db(PERFORMER_DB_PATH)
                if db:
                    centroids = cluster_embeddings(detect_embeddings(frames_face))
                    matches   = match_centroids(centroids, db)
                    performers = [{"name": m["name"], "score": round(m["score"] * 100)} for m in matches]
                    if performers:
                        logger.info(f"Performers: {performers}")
            except Exception as e:
                logger.warning(f"Performer detection failed: {e}")

        # ── Save results ────────────────────────────────────────────────────────
        os.makedirs(output_dir, exist_ok=True)
        saved_frames = []
        thumb_path = None
        thumb_b64 = ""

        meta = {
            "description": description,
            "categories": final_categories,
            "orientation": orientation,
            "watermarks": watermarks,
            "performers": performers,
            "key_scenes": key_scenes,
            "seo": seo_by_lang,
            "meta_title": meta_title,
            "meta_description": meta_desc,
            "primary_tags": primary_tags,
            "secondary_tags": secondary_tags,
            "seo_description": seo_description,
            "language": language,
            "style": style,
        }
        with open(os.path.join(output_dir, f"{base_name}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        def _to_url(p):
            if not p:
                return None
            try:
                rel = Path(p).relative_to(OUTPUT_DIR)
                return f"/v2/results/{rel.as_posix()}"
            except ValueError:
                return None

        logger.info(f"Done: {base_name}")
        return {
            "status": "ok",
            "base_name": base_name,
            "orientation": orientation,
            "description": description,
            "categories": final_categories,
            "watermarks": watermarks,
            "performers": performers,
            "key_scenes": key_scenes,
            "seo": seo_by_lang,
            "meta_title": meta_title,
            "meta_description": meta_desc,
            "primary_tags": primary_tags,
            "secondary_tags": secondary_tags,
            "seo_description": seo_description,
            "thumbnail": _to_url(thumb_path),
            "thumbnail_base64": thumb_b64,
            "frames": [
                {
                    "url": _to_url(f["path"]),
                    "score": f["score"],
                    "reason": f["reason"],
                    "ts": f["ts"],
                    "ts_fmt": f["ts_fmt"],
                }
                for f in saved_frames
            ],
        }

    except Exception as e:
        logger.exception(f"Critical error processing {video_path}")
        return {"status": "error", "reason": str(e)}


def _build_webhook_payload(task_id: str, result: Dict) -> Dict:
    """Convert internal result to the external webhook format."""
    if result.get("status") != "ok":
        return {"success": False, "task_id": task_id, "error": result.get("reason", "unknown")}

    seo = result.get("seo", {})
    primary_code = next(iter(seo), "en")

    performers_out = [
        {"name": p["name"], "confidence": p.get("score", 0)}
        for p in result.get("performers", [])
    ]

    r: Dict = {
        "primary_tags":     result.get("primary_tags", []),
        "secondary_tags":   result.get("secondary_tags", []),
        "categories":       result.get("categories", []),
        "orientation":      result.get("orientation", ""),
        "description":      result.get("description", ""),
        "watermarks":       result.get("watermarks", []),
        "performers":       performers_out,
        "meta_title":       result.get("meta_title", ""),
        "meta_description": result.get("meta_description", ""),
        "seo_description":  result.get("seo_description", ""),
        "preview_thumbnail": ("data:image/jpeg;base64," + result["thumbnail_base64"]) if result.get("thumbnail_base64") else "",
    }

    # Flatten extra languages: meta_title_de, meta_description_de, seo_description_de ...
    for lang_code, lang_data in seo.items():
        if lang_code == primary_code:
            continue
        r[f"meta_title_{lang_code}"]       = lang_data.get("meta_title", "")
        r[f"meta_description_{lang_code}"] = lang_data.get("meta_description", "")
        r[f"seo_description_{lang_code}"]  = lang_data.get("seo_description", "")

    return {"success": True, "task_id": task_id, "result": r}


# ─── Task store ───────────────────────────────────────────────────────────────
# {task_id: {"status": "processing"|"done"|"error", "stage": str, "result": dict}}
_tasks: Dict[str, Dict] = {}
_tasks_lock = threading.Lock()

# ─── Serial task queue ────────────────────────────────────────────────────────
_task_queue: queue.Queue = queue.Queue()

def _queue_worker():
    """Single worker — processes one video at a time, queues the rest."""
    while True:
        task_id, fn, args, kwargs, webhook_url = _task_queue.get()
        try:
            _run_task(task_id, fn, *args, webhook_url=webhook_url, **kwargs)
        except Exception as e:
            logger.error(f"Worker error {task_id}: {e}")
        finally:
            _task_queue.task_done()

threading.Thread(target=_queue_worker, daemon=True, name="task-worker").start()


def _run_task(task_id: str, fn, *args, webhook_url: str = "", **kwargs):
    """Run fn(*args, **kwargs) in a thread; store result in _tasks; fire webhook if set."""
    try:
        result = fn(*args, **kwargs)
    except Exception as e:
        result = {"status": "error", "reason": str(e)}
    with _tasks_lock:
        _tasks[task_id]["status"] = result.get("status", "error")
        _tasks[task_id]["result"] = result
    if webhook_url:
        try:
            payload = _build_webhook_payload(task_id, result)
            log_payload = {**payload, "result": {**payload.get("result", {}), "preview_thumbnail": f"<base64 {len(payload.get('result', {}).get('preview_thumbnail', ''))} chars>"}}
            logger.info(f"Webhook payload → {json.dumps(log_payload, ensure_ascii=False)}")
            wh = requests.post(webhook_url, json=payload, timeout=15,
                               headers={"User-Agent": _CHROME_UA, "Content-Type": "application/json"})
            logger.info(f"Webhook fired → {webhook_url} | status={wh.status_code} | response={wh.text[:5000]}")
        except Exception as e:
            logger.warning(f"Webhook failed: {e}")
        run_dir_to_clean = result.get("_run_dir")
        if run_dir_to_clean:
            shutil.rmtree(run_dir_to_clean, ignore_errors=True)
            logger.info(f"Cleaned up output dir: {run_dir_to_clean}")


# ─── FastAPI app ──────────────────────────────────────────────────────────────

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Video Analyzer v2")
templates = Jinja2Templates(directory="templates")
app.mount("/v2/results", StaticFiles(directory=OUTPUT_DIR), name="v2_results")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("analyzer.html", {"request": request})


@app.get("/v2/status/{task_id}")
async def task_status(task_id: str):
    with _tasks_lock:
        task = _tasks.get(task_id)
    if task is None:
        return JSONResponse({"status": "not_found"}, status_code=404)
    if task["status"] == "processing":
        return JSONResponse({"status": "processing", "stage": task.get("stage", "")})
    return JSONResponse(task["result"])


@app.post("/v2/analyze-upload")
async def analyze_upload(
    files: List[UploadFile] = File(...),
    language: str = Form("English"),
    style: str = Form("standard"),
):
    if not files:
        return JSONResponse({"status": "error", "reason": "no files"}, status_code=400)

    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(OUTPUT_DIR) / f"run_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    f = files[0]
    dest = Path(UPLOAD_DIR) / f.filename
    with open(dest, "wb") as out:
        shutil.copyfileobj(f.file, out)
    base_name = dest.stem
    video_out = run_dir / base_name
    video_out.mkdir(exist_ok=True)

    task_id = str(uuid.uuid4())
    with _tasks_lock:
        _tasks[task_id] = {"status": "processing", "stage": "starting", "result": None}

    t = threading.Thread(
        target=_run_task,
        args=(task_id, process_video_v2, str(dest), str(video_out), base_name, language, style),
        daemon=True,
    )
    t.start()
    return JSONResponse({"status": "processing", "task_id": task_id})


@app.post("/v2/analyze-url")
async def analyze_url(
    url: str = Form(...),
    language: str = Form("English"),
    style: str = Form("standard"),
):
    run_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_tmpl = str(Path(UPLOAD_DIR) / f"ytdl_{run_ts}.%(ext)s")
    try:
        r = subprocess.run(
            ["yt-dlp", url, "-o", out_tmpl, "--no-playlist",
             "-f", "best[ext=mp4]/best", "--merge-output-format", "mp4"],
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode != 0:
            return JSONResponse({"status": "error", "reason": f"yt-dlp: {r.stderr[:300]}"}, status_code=400)
    except FileNotFoundError:
        return JSONResponse({"status": "error", "reason": "yt-dlp not installed"}, status_code=500)
    except subprocess.TimeoutExpired:
        return JSONResponse({"status": "error", "reason": "download timeout"}, status_code=500)

    downloaded = list(Path(UPLOAD_DIR).glob(f"ytdl_{run_ts}.*"))
    if not downloaded:
        return JSONResponse({"status": "error", "reason": "file not found after download"}, status_code=500)

    video_path = str(downloaded[0])
    base_name  = Path(video_path).stem
    run_dir    = Path(OUTPUT_DIR) / f"run_{run_ts}"
    video_out  = run_dir / base_name
    video_out.mkdir(parents=True, exist_ok=True)

    task_id = str(uuid.uuid4())
    with _tasks_lock:
        _tasks[task_id] = {"status": "processing", "stage": "starting", "result": None}

    t = threading.Thread(
        target=_run_task,
        args=(task_id, process_video_v2, video_path, str(video_out), base_name, language, style),
        daemon=True,
    )
    t.start()
    return JSONResponse({"status": "processing", "task_id": task_id})


# ─── JSON API ─────────────────────────────────────────────────────────────────

LANG_MAP = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "ja": "Japanese",
    "zh": "Chinese", "ko": "Korean", "nl": "Dutch", "pl": "Polish",
    "ar": "Arabic",  "tr": "Turkish",  "cs": "Czech",   "sv": "Swedish",
}


class AnalyzeRequest(BaseModel):
    video_url: str
    languages: List[str] = ["en"]
    style: str = "standard"
    client_reference_id: str = ""
    webhook_url: str = ""
    # ignored fields kept for compat
    tag_count: int = 10
    secondary_tag_count: int = 7
    category_count: int = 10


def _check_api_key(x_api_key: Optional[str]):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


_CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


def _download_video(video_url: str, run_ts: str) -> str:
    """Download video via yt-dlp (supports mp4, HLS, most platforms). Returns local path."""
    out_tmpl = str(Path(UPLOAD_DIR) / f"api_{run_ts}.%(ext)s")
    r = subprocess.run(
        ["yt-dlp", video_url, "-o", out_tmpl, "--no-playlist",
         "-f", "best[ext=mp4]/best", "--merge-output-format", "mp4",
         "--user-agent", _CHROME_UA,
         "--add-header", "Accept-Language:en-US,en;q=0.9"],
        capture_output=True, text=True, timeout=600,
    )
    if r.returncode != 0:
        raise RuntimeError(f"yt-dlp error: {r.stderr[:400]}")
    downloaded = list(Path(UPLOAD_DIR).glob(f"api_{run_ts}.*"))
    if not downloaded:
        raise RuntimeError("File not found after download")
    return str(downloaded[0])


def _api_task(task_id: str, req: AnalyzeRequest):
    """Full pipeline: download → process → return result."""
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{task_id[:8]}"
    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

    # Download
    with _tasks_lock:
        _tasks[task_id]["stage"] = "downloading"
    try:
        video_path = _download_video(req.video_url, run_ts)
    except Exception as e:
        return {"status": "error", "reason": str(e),
                "client_reference_id": req.client_reference_id}

    langs     = req.languages or ["en"]
    lang_code = langs[0].lower()
    language  = LANG_MAP.get(lang_code, "English")
    extra     = langs[1:]  # remaining codes for extra SEO passes

    base_name = Path(video_path).stem
    run_dir   = Path(OUTPUT_DIR) / f"api_{run_ts}"
    video_out = run_dir / base_name
    video_out.mkdir(parents=True, exist_ok=True)

    with _tasks_lock:
        _tasks[task_id]["stage"] = "analyzing"

    result = process_video_v2(
        video_path, str(video_out), base_name, language, req.style,
        extra_languages=extra,
        tag_count=req.tag_count,
        secondary_tag_count=req.secondary_tag_count,
        category_count=req.category_count,
    )
    if req.client_reference_id:
        result["client_reference_id"] = req.client_reference_id

    result["_run_dir"] = str(run_dir)

    try:
        os.remove(video_path)
        logger.info(f"Deleted video: {video_path}")
    except Exception as e:
        logger.warning(f"Could not delete video {video_path}: {e}")

    return result


@app.post("/api/v2/analyze")
async def api_analyze(
    body: AnalyzeRequest,
    x_api_key: Optional[str] = Header(default=None),
):
    _check_api_key(x_api_key)
    task_id = str(uuid.uuid4())
    with _tasks_lock:
        _tasks[task_id] = {"status": "processing", "stage": "queued", "result": None}

    _task_queue.put((task_id, _api_task, (task_id, body), {}, body.webhook_url))
    queue_pos = _task_queue.qsize()
    return JSONResponse({"status": "processing", "task_id": task_id,
                         "client_reference_id": body.client_reference_id,
                         "queue_position": queue_pos})


@app.get("/api/v2/status/{task_id}")
async def api_task_status(
    task_id: str,
    x_api_key: Optional[str] = Header(default=None),
):
    _check_api_key(x_api_key)
    with _tasks_lock:
        task = _tasks.get(task_id)
    if task is None:
        return JSONResponse({"status": "not_found"}, status_code=404)
    if task["status"] == "processing":
        return JSONResponse({"status": "processing", "stage": task.get("stage", ""),
                             "queue_pending": _task_queue.qsize()})
    return JSONResponse(task["result"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("video_processor:app", host="0.0.0.0", port=8000, log_level="info", workers=1)