"""
Load categories.json and build prompts + alias matching.
"""

import json
import re
from pathlib import Path


CATEGORIES_PATH = Path(__file__).parent.parent / "categories.json"


def load_categories(path: str | Path = CATEGORIES_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_category_prompt(categories: dict) -> str:
    """
    Category listing grouped by section with full descriptions.
    """
    lines = []
    for section in categories.values():
        title = section["title"]
        lines.append(f"\n## {title}")
        for cat in section["categories"]:
            name = cat["name"]
            desc = cat.get("description", "")
            if len(desc) > 300:
                desc = desc[:297] + "..."
            lines.append(f'  - "{name}": {desc}')
    return "\n".join(lines)


def build_canonical_map(categories: dict) -> dict[str, str]:
    """
    Build a mapping: any alias/name (lowercased) → canonical category name.
    Used to normalize model output.
    """
    mapping = {}
    for section in categories.values():
        for cat in section["categories"]:
            name = cat["name"]
            canonical = name

            # Map exact name
            mapping[name.lower()] = canonical

            # Map aliases
            for alias in cat.get("aliases", []):
                mapping[alias.lower().strip()] = canonical

    return mapping


def build_guided_schema(categories: dict) -> dict:
    """
    Build a JSON schema for vLLM structured output (guided decoding).
    Forces model to ONLY output names that exist in the category list.
    """
    all_names = []
    for section in categories.values():
        for cat in section["categories"]:
            all_names.append(cat["name"])
    return {
        "type": "object",
        "properties": {
            "categories": {
                "type": "array",
                "items": {"type": "string", "enum": all_names},
                "maxItems": 15,
            }
        },
        "required": ["categories"],
    }


VALID_ORIENTATIONS = {"straight", "gay", "shemale"}


def parse_model_output(
    raw_output: str,
    canonical_map: dict[str, str],
) -> tuple[str | None, list[str]]:
    """
    Parse model JSON output.  Expected format:
        {"orientation": "straight|gay|shemale", "categories": ["cat1", ...]}

    Falls back to plain JSON array (legacy) for backwards compatibility.

    Returns:
        (orientation, categories)
        orientation is None if not present or invalid.
    """
    orientation: str | None = None
    categories: list[str] = []

    # Strip <think>...</think> blocks before parsing
    raw_output = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL).strip()

    # ── Try structured object ─────────────────────────────────────────────────
    obj_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
    if obj_match:
        try:
            obj = json.loads(obj_match.group())
            if isinstance(obj, dict):
                raw_orient = obj.get("orientation", "")
                if isinstance(raw_orient, str) and raw_orient.lower() in VALID_ORIENTATIONS:
                    orientation = raw_orient.lower()
                raw_cats = obj.get("categories", [])
                if isinstance(raw_cats, list):
                    categories = _parse_cat_list(raw_cats, canonical_map)
                    return orientation, categories
        except (json.JSONDecodeError, ValueError):
            pass

    # ── Fallback: plain JSON array ────────────────────────────────────────────
    arr_match = re.search(r'\[.*?\]', raw_output, re.DOTALL)
    if arr_match:
        try:
            items = json.loads(arr_match.group())
            if isinstance(items, list):
                categories = _parse_cat_list(items, canonical_map)
                return orientation, categories
        except json.JSONDecodeError:
            pass

    # ── Last resort: line by line ─────────────────────────────────────────────
    results = set()
    for line in raw_output.split('\n'):
        clean = re.sub(r'["\'\-\*\•\d\.\,]', ' ', line).strip()
        if not clean:
            continue
        canonical = _find_canonical(clean, canonical_map)
        if canonical:
            results.add(canonical)
    return orientation, sorted(results)


def _parse_cat_list(items: list, canonical_map: dict[str, str]) -> list[str]:
    results = set()
    for item in items:
        if isinstance(item, str):
            canonical = _find_canonical(item, canonical_map)
            if canonical:
                results.add(canonical)
    return sorted(results)


def _find_canonical(text: str, canonical_map: dict[str, str]) -> str | None:
    """Try to match text to a canonical category name."""
    text_lower = text.lower().strip()

    # Exact match
    if text_lower in canonical_map:
        return canonical_map[text_lower]

    # Fuzzy: find longest key that appears as a whole word in text
    best = None
    best_len = 0
    for key, canonical in canonical_map.items():
        if len(key) <= best_len:
            continue
        if re.search(r'(?<![a-z])' + re.escape(key) + r'(?![a-z])', text_lower):
            best = canonical
            best_len = len(key)

    return best


