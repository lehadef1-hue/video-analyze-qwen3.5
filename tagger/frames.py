"""
Frame extraction, scene detection and grid composition.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


def get_video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "width": width,
        "height": height,
    }


def extract_frames_range(
    video_path: str,
    start_frame: int,
    end_frame: int,
    n_frames: int,
) -> list[Image.Image]:
    """Extract N evenly spaced frames from [start_frame, end_frame]."""
    count = end_frame - start_frame
    if count <= 0:
        return []

    indices = np.linspace(start_frame, end_frame, n_frames, dtype=int)
    frames = []
    cap = cv2.VideoCapture(video_path)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def detect_scenes(
    video_path: str,
    min_scene_len_sec: float = 3.0,
    max_scenes: int = 8,
    threshold: float = 0.35,
) -> list[tuple[int, int]]:
    """
    Detect scene boundaries via histogram correlation.
    Returns list of (start_frame, end_frame) tuples.
    """
    info = get_video_info(video_path)
    fps = max(info["fps"], 1)
    frame_count = info["frame_count"]
    min_scene_frames = int(min_scene_len_sec * fps)

    cap = cv2.VideoCapture(video_path)
    scenes = []
    scene_start = 0
    prev_hist = None

    # Sample at 3 fps to speed up detection
    sample_every = max(1, int(fps / 3))

    frame_idx = 0
    try:
        while frame_idx < frame_count:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            cv2.normalize(hist, hist)

            if prev_hist is not None:
                corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                if corr < threshold and (frame_idx - scene_start) >= min_scene_frames:
                    scenes.append((scene_start, frame_idx - 1))
                    scene_start = frame_idx

            prev_hist = hist
            frame_idx += sample_every
    finally:
        cap.release()

    scenes.append((scene_start, frame_count - 1))

    # If too many scenes — keep the longest ones
    if len(scenes) > max_scenes:
        scenes_by_len = sorted(scenes, key=lambda x: x[1] - x[0], reverse=True)
        scenes = sorted(scenes_by_len[:max_scenes], key=lambda x: x[0])

    return scenes


def split_into_segments(
    start_frame: int,
    end_frame: int,
    n_segments: int,
) -> list[tuple[int, int]]:
    """
    Divide [start_frame, end_frame] into N equal segments.
    Returns list of (seg_start, seg_end).
    """
    total = end_frame - start_frame
    if total <= 0 or n_segments <= 0:
        return [(start_frame, end_frame)]

    seg_len = total / n_segments
    segments = []
    for i in range(n_segments):
        s = int(start_frame + i * seg_len)
        e = int(start_frame + (i + 1) * seg_len) - 1
        e = min(e, end_frame)
        if s <= e:
            segments.append((s, e))
    return segments


def make_grid(
    frames: list[Image.Image],
    cols: int = 2,
    cell_size: tuple[int, int] = (480, 270),
    add_index: bool = True,
) -> Image.Image:
    """
    Compose frames into a temporal grid image.
    Default: 2×2 grid (cols=2, so 4 frames → 2 rows).
    Frames read left→right, top→bottom = time progression.

    Args:
        frames: list of PIL images
        cols: columns in grid
        cell_size: (width, height) per cell
        add_index: overlay frame number for orientation
    """
    if not frames:
        return Image.new("RGB", cell_size, (20, 20, 20))

    rows = (len(frames) + cols - 1) // cols
    cell_w, cell_h = cell_size
    grid_w = cols * cell_w
    grid_h = rows * cell_h

    # Thin separator lines
    sep = 2
    grid = Image.new("RGB", (grid_w + sep * (cols - 1), grid_h + sep * (rows - 1)), (40, 40, 40))

    for i, frame in enumerate(frames):
        row = i // cols
        col = i % cols
        x = col * (cell_w + sep)
        y = row * (cell_h + sep)

        resized = frame.resize(cell_size, Image.LANCZOS)

        if add_index:
            draw = ImageDraw.Draw(resized)
            label = f"{i + 1}"
            draw.rectangle([0, 0, 22, 18], fill=(0, 0, 0, 180))
            draw.text((2, 2), label, fill=(255, 255, 0))

        grid.paste(resized, (x, y))

    return grid


# ── Public API ──────────────────────────────────────────────────────────────


def get_overview_frames(video_path: str, n_frames: int = 8) -> list[Image.Image]:
    """Uniform frames across entire video for global context."""
    info = get_video_info(video_path)
    return extract_frames_range(video_path, 0, info["frame_count"] - 1, n_frames)


def get_scene_segments(
    video_path: str,
    passes_per_scene: int = 4,    # 4 windows → 4 grids 2×2 per scene
    frames_per_pass: int = 4,     # 4 frames per window → 2×2 grid
    grid_cols: int = 2,           # 2×2 grid
    max_scenes: int = 8,          # up to 8 scenes
    min_scene_len_sec: float = 3.0,
) -> list[dict]:
    """
    Detect scenes and divide each scene into `passes_per_scene` temporal segments.
    Default: 4 grids 2×2 per scene, up to 8 scenes.
    Returns ready-to-use grid images per segment.

    Returns:
        list of {
            "scene_idx": int,
            "segment_idx": int,        # 0-based within scene
            "total_segments": int,
            "start_frame": int,
            "end_frame": int,
            "start_sec": float,
            "end_sec": float,
            "grid": PIL.Image,
        }
    """
    info = get_video_info(video_path)
    fps = max(info["fps"], 1)
    frame_count = info["frame_count"]

    scenes = detect_scenes(
        video_path,
        min_scene_len_sec=min_scene_len_sec,
        max_scenes=max_scenes,
    )

    results = []
    for scene_idx, (scene_start, scene_end) in enumerate(scenes):
        segments = split_into_segments(scene_start, scene_end, passes_per_scene)

        for seg_idx, (seg_start, seg_end) in enumerate(segments):
            frames = extract_frames_range(
                video_path, seg_start, seg_end, n_frames=frames_per_pass
            )
            if not frames:
                continue

            grid = make_grid(frames, cols=grid_cols, cell_size=(480, 270), add_index=True)

            results.append({
                "scene_idx": scene_idx,
                "segment_idx": seg_idx,
                "total_segments": len(segments),
                "start_frame": seg_start,
                "end_frame": seg_end,
                "start_sec": seg_start / fps,
                "end_sec": seg_end / fps,
                "grid": grid,
            })

    return results
