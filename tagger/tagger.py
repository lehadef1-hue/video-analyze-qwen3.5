"""
Main video tagger.

Pipeline:
  1. Scene detection via histogram correlation (up to MAX_SCENES=8)
  2. Each scene → PASSES_PER_SCENE=4 temporal windows × FRAMES_PER_PASS=4 frames

  Two modes (TAGGER_MODE env or mode= param):
    "video" — frames sent as video sequence with fps=2 (native Qwen3-VL video input)
    "grid"  — frames composed into 2×2 grids, sent as images (legacy mode)

  Aggregation:
    • Union all detections with frequency counts
    • validate_categories() resolves conflicts using counts + logic rules

Total calls: MAX_SCENES  (e.g. 8 scenes ≈ 25-35s on A40)
"""

import os
import time
from collections import Counter
from pathlib import Path

from .frames import get_video_info, get_scene_segments, make_grid
from .categories import load_categories, build_category_prompt, build_canonical_map, parse_model_output
from .validate import validate_categories
from .model import QwenVLModel

# ── Configuration ────────────────────────────────────────────────────────────

MAX_SCENES       = 8    # max scenes to analyze
PASSES_PER_SCENE = 4    # temporal windows per scene
FRAMES_PER_PASS  = 4    # frames per window
GRID_COLS        = 2    # grid columns for grid mode (2×2)

MIN_SCENE_SEC  = 3.0  # ignore scenes shorter than this
TAGGER_FPS     = 2.0  # fps value sent to model in video mode

# Category must appear in at least this many scenes to be included
MIN_PASS_COUNT = 1

# Default mode: "video" or "grid" (override with TAGGER_MODE env var)
# "grid"  — vLLM backend (model_server.py)   → 2×2 grid images
# "video" — llama.cpp backend (model_server_llama.py) → frames + fps
DEFAULT_MODE = os.getenv("TAGGER_MODE", "grid")


# ── Prompt templates ──────────────────────────────────────────────────────────

_PROMPT_VIDEO = """\
You are analyzing a video scene. The frames are shown in chronological order.

ALLOWED CATEGORY NAMES (you may ONLY use names from this list):
{categories}

TASK: Look at the frames. Pick 5–15 names from the list above that are clearly and unambiguously visible.
RULES:
- Copy names EXACTLY as written above (case-sensitive).
- Do NOT invent names. Do NOT use words not in the list above.
- Do NOT repeat names. No more than 15.
- When in doubt — omit.

Return JSON only:
{{"categories": ["Name1", "Name2"]}}"""

_PROMPT_GRID = """\
You are shown {n_grids} images. Each is a 2×2 grid of video frames (left→right, top→bottom = time order).

ALLOWED CATEGORY NAMES (you may ONLY use names from this list):
{categories}

TASK: Look at the images. Pick 5–15 names from the list above that are clearly and unambiguously visible.
RULES:
- Copy names EXACTLY as written above (case-sensitive).
- Do NOT invent names. Do NOT use words not in the list above.
- Do NOT repeat names. No more than 15.
- When in doubt — omit.

Return JSON only:
{{"categories": ["Name1", "Name2"]}}"""


class VideoTagger:
    def __init__(
        self,
        categories_path: str | Path | None = None,
        model_id: str | None = None,
        passes_per_scene: int = PASSES_PER_SCENE,
        max_scenes: int = MAX_SCENES,
        frames_per_pass: int = FRAMES_PER_PASS,
        min_pass_count: int = MIN_PASS_COUNT,
        mode: str = DEFAULT_MODE,
    ):
        self.categories = load_categories(categories_path) if categories_path else load_categories()
        self.canonical_map = build_canonical_map(self.categories)
        self._category_listing = build_category_prompt(self.categories)

        model_kwargs = {"model_id": model_id} if model_id else {}
        self.model = QwenVLModel(**model_kwargs)

        self.passes_per_scene = passes_per_scene
        self.max_scenes = max_scenes
        self.frames_per_pass = frames_per_pass
        self.min_pass_count = min_pass_count
        self.mode = mode if mode in ("video", "grid") else "video"

    def load_model(self):
        self.model.load()

    def tag_video(self, video_path: str, orientation: str = "straight", verbose: bool = True) -> dict:
        t0 = time.time()
        video_path = str(video_path)

        info = get_video_info(video_path)
        duration = info["duration"]

        if verbose:
            print(f"\n{'='*65}")
            print(f"Video: {Path(video_path).name}")
            print(f"Duration: {duration:.1f}s | {info['fps']:.1f}fps | {info['width']}x{info['height']}")
            print(f"Mode: {self.mode}")

        category_counter: Counter = Counter()
        total_passes = 0

        if verbose:
            print(f"\n[Scenes] Detecting scenes (max {self.max_scenes})...")

        scenes = get_scene_segments(
            video_path,
            passes_per_scene=self.passes_per_scene,
            frames_per_pass=self.frames_per_pass,
            max_scenes=self.max_scenes,
            min_scene_len_sec=MIN_SCENE_SEC,
        )

        n_scenes = len(scenes)

        if verbose:
            print(f"[Scenes] {n_scenes} scenes detected")

        for scene in scenes:
            scene_i = scene["scene_idx"] + 1
            t_start = scene["start_sec"]
            t_end   = scene["end_sec"]
            frames  = scene["frames"]

            if self.mode == "grid":
                # Compose frames into 2×2 grids, one grid per pass
                n_per_grid = GRID_COLS * GRID_COLS  # 4 frames per grid
                grids = []
                for i in range(0, len(frames), n_per_grid):
                    chunk = frames[i:i + n_per_grid]
                    grids.append(make_grid(chunk, cols=GRID_COLS, cell_size=(480, 270), add_index=True))
                n_grids = len(grids)

                if verbose:
                    print(
                        f"  Scene {scene_i} | {n_grids} grids "
                        f"[{t_start:.0f}s–{t_end:.0f}s] → single call",
                        end=" ",
                    )

                scene_prompt = _PROMPT_GRID.format(
                    categories=self._category_listing,
                    n_grids=n_grids,
                )
                raw = self.model.analyze(grids, scene_prompt, fps=None, verbose=verbose)

            else:
                # Video mode: send individual frames with fps
                if verbose:
                    print(
                        f"  Scene {scene_i} | {len(frames)} frames "
                        f"[{t_start:.0f}s–{t_end:.0f}s] → single call",
                        end=" ",
                    )

                scene_prompt = _PROMPT_VIDEO.format(categories=self._category_listing)
                raw = self.model.analyze(frames, scene_prompt, fps=TAGGER_FPS, verbose=verbose)

            _, cats = parse_model_output(raw, self.canonical_map)

            if verbose and cats:
                print(f"\n    → {cats}")
            elif verbose:
                print()

            for c in cats:
                category_counter[c] += 1
            total_passes += 1

        # ══ Aggregate ══════════════════════════════════════════════════════════
        raw_categories = sorted(
            cat for cat, count in category_counter.items()
            if count >= self.min_pass_count
        )

        final_categories = sorted(
            validate_categories(raw_categories, orientation, counts=dict(category_counter))
        )

        processing_time = time.time() - t0

        if verbose:
            print(f"\n{'─'*65}")
            print(f"Category counts (top-30):")
            for cat, cnt in category_counter.most_common(30):
                bar = "█" * cnt
                print(f"  {bar} {cnt}x  {cat}")
            print(f"Categories ({len(final_categories)}): {final_categories}")
            print(f"Total passes: {total_passes} | Time: {processing_time:.1f}s")

        scene_segments = []
        for scene in scenes:
            scene_segments.append({
                "start_sec": float(round(scene["start_sec"], 1)),
                "end_sec":   float(round(scene["end_sec"], 1)),
            })

        return {
            "video": video_path,
            "duration": duration,
            "categories": final_categories,
            "category_counts": dict(category_counter.most_common()),
            "scenes_detected": n_scenes,
            "scene_segments": scene_segments,
            "total_passes": total_passes,
            "processing_time": processing_time,
        }

    def tag_batch(self, video_paths: list[str], orientation: str = "straight", verbose: bool = True) -> list[dict]:
        results = []
        for i, path in enumerate(video_paths):
            if verbose:
                print(f"\n[{i+1}/{len(video_paths)}]")
            results.append(self.tag_video(path, orientation=orientation, verbose=verbose))
        return results
