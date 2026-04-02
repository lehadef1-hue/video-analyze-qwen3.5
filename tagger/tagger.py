"""
Main video tagger.

Pipeline:
  1. Scene detection via histogram correlation (up to MAX_SCENES=8)
  2. Each scene → PASSES_PER_SCENE=4 temporal windows × FRAMES_PER_PASS=4 frames
     Each window → 2×2 grid (GRID_COLS=2)
     16 frames per scene presented as 4 × 2×2 grids in ONE model call
  3. Model returns categories per scene (orientation comes from Pass1)

  Aggregation:
    • Union all detections with frequency counts
    • validate_categories() resolves conflicts using counts + logic rules

Total calls: MAX_SCENES  (e.g. 8 scenes ≈ 25-35s on A40)
"""

import time
from collections import Counter
from pathlib import Path

from .frames import get_video_info, get_scene_segments
from .categories import load_categories, build_category_prompt, build_canonical_map, parse_model_output
from .validate import validate_categories
from .model import QwenVLModel

# ── Configuration ────────────────────────────────────────────────────────────

MAX_SCENES       = 8    # max scenes to analyze
PASSES_PER_SCENE = 4    # temporal windows per scene (all sent in one model call)
FRAMES_PER_PASS  = 4    # frames per window → 2×2 grid
GRID_COLS        = 2    # grid columns (2 cols × 2 rows = 4 cells)

MIN_SCENE_SEC  = 3.0  # ignore scenes shorter than this

# Category must appear in at least this many passes to be included
MIN_PASS_COUNT = 1


# ── Prompt template ───────────────────────────────────────────────────────────

ANALYSIS_PROMPT_TEMPLATE = """\
You are shown {n_grids} images. Each is a 2×2 grid of video frames (left→right, top→bottom = time order).

Look at the images and list ONLY what is clearly and unambiguously visible.
Use ONLY names from the list below. Copy them EXACTLY (case-sensitive).
STRICT RULES:
- Return EXACTLY 5 to 15 names. No more than 15. Stop after 15.
- Only include what is visually confirmed — when in doubt, omit.
- Do NOT repeat names.

{categories}

Return JSON only, no explanation:
{{"categories": ["Name1", "Name2", "Name3"]}}"""


class VideoTagger:
    def __init__(
        self,
        categories_path: str | Path | None = None,
        model_id: str | None = None,
        passes_per_scene: int = PASSES_PER_SCENE,
        max_scenes: int = MAX_SCENES,
        frames_per_pass: int = FRAMES_PER_PASS,
        min_pass_count: int = MIN_PASS_COUNT,
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

        category_counter: Counter = Counter()
        total_passes = 0

        if verbose:
            print(f"\n[Scenes] Detecting scenes (max {self.max_scenes})...")

        segments = get_scene_segments(
            video_path,
            passes_per_scene=self.passes_per_scene,
            frames_per_pass=self.frames_per_pass,
            grid_cols=GRID_COLS,
            max_scenes=self.max_scenes,
            min_scene_len_sec=MIN_SCENE_SEC,
        )

        n_scenes = len(set(s["scene_idx"] for s in segments)) if segments else 0

        if verbose:
            print(f"[Scenes] {n_scenes} scenes detected")

        segments_by_scene: dict[int, list] = {}
        for seg in segments:
            segments_by_scene.setdefault(seg["scene_idx"], []).append(seg)

        for scene_idx, scene_segs in segments_by_scene.items():
            scene_i = scene_idx + 1
            t_start = scene_segs[0]["start_sec"]
            t_end   = scene_segs[-1]["end_sec"]
            n_grids = len(scene_segs)

            if verbose:
                print(
                    f"  Scene {scene_i} | {n_grids} grids "
                    f"[{t_start:.0f}s–{t_end:.0f}s] → single call",
                    end=" ",
                )

            scene_prompt = ANALYSIS_PROMPT_TEMPLATE.format(
                categories=self._category_listing,
                n_grids=n_grids,
            )

            raw = self.model.analyze(
                [seg["grid"] for seg in scene_segs],
                scene_prompt,
                segment_info=scene_segs[0],
                verbose=verbose,
            )
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
        for scene_idx, scene_segs in segments_by_scene.items():
            scene_segments.append({
                "start_sec": float(round(scene_segs[0]["start_sec"], 1)),
                "end_sec":   float(round(scene_segs[-1]["end_sec"], 1)),
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
