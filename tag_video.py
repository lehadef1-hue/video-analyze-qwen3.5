#!/usr/bin/env python3
"""
CLI for video tagging with multi-pass scene analysis.

Usage:
    python tag_video.py video.mp4
    python tag_video.py video.mp4 --passes 8 --scenes 5
    python tag_video.py video1.mp4 video2.mp4 --output results.json
    python tag_video.py video.mp4 --min-count 2  # only cats seen in 2+ passes
"""

import argparse
import json
import sys
from pathlib import Path

from tagger import VideoTagger


def main():
    parser = argparse.ArgumentParser(
        description="Tag videos with categories using Qwen3-VL "
                    "(4 passes/scene × 4 frames → 4× 2×2 grids per scene)"
    )
    parser.add_argument("videos", nargs="+", help="Video file(s) to analyze")
    parser.add_argument("--model", default=None, help="Override model ID")
    parser.add_argument("--categories", default=None, help="Path to categories.json")
    parser.add_argument("--passes", type=int, default=4, help="Temporal passes per scene, each → 2×2 grid (default: 4)")
    parser.add_argument("--scenes", type=int, default=8, help="Max scenes to analyze (default: 8)")
    parser.add_argument("--frames", type=int, default=4, help="Frames per pass → 2×2 grid (default: 4)")
    parser.add_argument("--min-count", type=int, default=1,
                        help="Min pass count to include category (default: 1)")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    parser.add_argument("--json", action="store_true", help="Print results as JSON")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    tagger = VideoTagger(
        categories_path=args.categories,
        model_id=args.model,
        passes_per_scene=args.passes,
        max_scenes=args.scenes,
        frames_per_pass=args.frames,
        min_pass_count=args.min_count,
    )

    tagger.load_model()

    results = []
    for video_path in args.videos:
        if not Path(video_path).exists():
            print(f"ERROR: File not found: {video_path}", file=sys.stderr)
            continue
        try:
            result = tagger.tag_video(video_path, verbose=not args.quiet)
        except Exception as exc:
            print(f"ERROR processing {video_path}: {exc}", file=sys.stderr)
            continue
        results.append(result)

        if not args.json and not args.output:
            total_passes = result["total_passes"]
            counts = result["category_counts"]
            # Sort by count descending
            cats_sorted = sorted(result["categories"], key=lambda c: counts.get(c, 0), reverse=True)

            print(f"\nFile: {result['video']}")
            print(f"Duration: {result['duration']:.1f}s")
            print(f"Scenes: {result['scenes_detected']} | Total passes: {total_passes}")
            print(f"Categories ({len(cats_sorted)}):")
            for cat in cats_sorted:
                count = counts.get(cat, 0)
                bar = "█" * count + "░" * (total_passes - count)
                print(f"  {count:2d}/{total_passes}  {bar}  {cat}")
            print(f"Time: {result['processing_time']:.1f}s")

    if not results:
        sys.exit(1)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
