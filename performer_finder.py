"""
Performer face recognition module.

Dependencies (all free):
    pip install insightface onnxruntime-gpu opencv-python-headless numpy

On first run InsightFace will automatically download the buffalo_l model (~200 MB).

Environment variables:
    PERFORMER_DB_PATH   – path to the embeddings database (default: performers_db.pkl)
    PERFORMER_THRESHOLD – cosine similarity threshold 0–1 (default: 0.42)
"""

import os
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image

logger = logging.getLogger("performer_finder")

PERFORMER_DB_PATH  = os.getenv("PERFORMER_DB_PATH", "workspace/my_performers.pkl")
MATCH_THRESHOLD    = float(os.getenv("PERFORMER_THRESHOLD", "0.45"))
CLUSTER_THRESHOLD  = 0.62   # cosine sim above this → same person
MIN_DET_SCORE      = 0.80   # face detection confidence threshold

# ── Lazy-loaded InsightFace app ───────────────────────────────────────────────
_face_app = None


def _get_face_app():
    """Load InsightFace buffalo_l on first call; return None if not installed."""
    global _face_app
    if _face_app is not None:
        return _face_app
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        _face_app = app
        logger.info("InsightFace buffalo_l loaded")
    except Exception as e:
        logger.warning(f"InsightFace not available — performer detection disabled. ({e})")
        _face_app = None
    return _face_app


def _pil_to_bgr(img: Image.Image) -> "np.ndarray":
    import cv2
    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _norm(v: "np.ndarray") -> "np.ndarray":
    return v / (np.linalg.norm(v) + 1e-8)


def _cos(a: "np.ndarray", b: "np.ndarray") -> float:
    return float(np.dot(a, b))


# ── Face embedding extraction ─────────────────────────────────────────────────

def detect_embeddings(frames: List[Image.Image]) -> List["np.ndarray"]:
    """Detect all faces in a list of frames; return list of normalised embeddings."""
    app = _get_face_app()
    if app is None:
        return []

    embeddings: List[np.ndarray] = []
    for img in frames:
        try:
            bgr = _pil_to_bgr(img)
            for face in app.get(bgr):
                if face.embedding is not None and face.det_score >= MIN_DET_SCORE:
                    embeddings.append(_norm(face.embedding))
        except Exception as e:
            logger.debug(f"Frame face-detection error: {e}")

    return embeddings


# ── Clustering (group embeddings by identity) ─────────────────────────────────

def cluster_embeddings(
    embeddings: List["np.ndarray"],
    threshold: float = CLUSTER_THRESHOLD,
) -> List["np.ndarray"]:
    """
    Greedy single-linkage clustering by cosine similarity.
    Returns one centroid (re-normalised mean) per identity cluster.
    """
    if not embeddings:
        return []

    clusters: List[List[np.ndarray]] = []
    for emb in embeddings:
        best_idx, best_sim = -1, -1.0
        for i, cluster in enumerate(clusters):
            # compare against centroid of cluster (first element for speed)
            sim = _cos(emb, cluster[0])
            if sim > threshold and sim > best_sim:
                best_sim, best_idx = sim, i
        if best_idx >= 0:
            clusters[best_idx].append(emb)
        else:
            clusters.append([emb])

    centroids = [_norm(np.mean(c, axis=0)) for c in clusters]
    logger.debug(f"Clustered {len(embeddings)} faces → {len(centroids)} identities")
    return centroids


# ── Database I/O ──────────────────────────────────────────────────────────────

def load_db(path: str = PERFORMER_DB_PATH) -> Dict[str, List["np.ndarray"]]:
    """Load performer DB: {performer_name: [embedding, ...]}"""
    p = Path(path)
    if not p.exists():
        logger.debug(f"Performer DB not found at {path}")
        return {}
    try:
        with open(p, "rb") as f:
            db = pickle.load(f)
        logger.info(f"Performer DB loaded: {len(db)} performers from {path}")
        return db
    except Exception as e:
        logger.error(f"Failed to load performer DB: {e}")
        return {}


def save_db(db: Dict[str, List["np.ndarray"]], path: str = PERFORMER_DB_PATH) -> None:
    with open(path, "wb") as f:
        pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Performer DB saved: {len(db)} performers → {path}")


# ── Matching ──────────────────────────────────────────────────────────────────

def match_centroids(
    centroids: List["np.ndarray"],
    db: Dict[str, List["np.ndarray"]],
    threshold: float = MATCH_THRESHOLD,
    max_results: int = 6,
) -> List[Dict]:
    """
    Compare face centroids against every performer in the DB.
    Returns [{name, score}, ...] sorted by confidence descending.
    """
    if not centroids or not db:
        return []

    matched: Dict[str, float] = {}
    for centroid in centroids:
        for name, db_embs in db.items():
            score = max(_cos(centroid, e) for e in db_embs)
            if score >= threshold:
                if name not in matched or matched[name] < score:
                    matched[name] = score

    results = sorted(matched.items(), key=lambda x: x[1], reverse=True)
    return [{"name": n, "score": round(s, 3)} for n, s in results[:max_results]]


# ── Public entry point ────────────────────────────────────────────────────────

def identify_performers(
    frames: List[Image.Image],
    db_path: str = PERFORMER_DB_PATH,
) -> List[str]:
    """
    Main function: detect faces → cluster → match against DB.
    Returns a list of performer name strings (empty if impossible).
    """
    if _get_face_app() is None:
        return []

    db = load_db(db_path)
    if not db:
        return []

    raw = detect_embeddings(frames)
    if not raw:
        logger.info("No faces detected in provided frames")
        return []

    centroids = cluster_embeddings(raw)
    matches = match_centroids(centroids, db)
    names = [f"{m['name']} ({round(m['score'] * 100)}%)" for m in matches]
    logger.info(f"Identified performers: {names}")
    return names
