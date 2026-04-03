"""
Post-processing validation for detected categories.
Removes logical contradictions, resolves priority conflicts,
and auto-completes implied categories.

When ``counts`` (category_counter dict) is provided, "which tag is correct"
conflicts are resolved by detection frequency rather than fixed priority.
Absolute anatomical / orientation rules always apply regardless of counts.
"""

from typing import Dict, List, Optional


def validate_categories(
    categories: List[str],
    orientation: Optional[str],
    counts: Optional[Dict[str, int]] = None,
) -> List[str]:
    """
    Enforce logical consistency across detected categories.

    Args:
        categories:  Canonical category names that passed the min_pass_count filter.
        orientation: Top-level orientation label ("straight", "gay", "shemale")
                     — used for absolute anatomical exclusions.
                     "Lesbian" is a category tag, not an orientation value.
        counts:      Raw detection counts per canonical name from category_counter.
                     When supplied, ambiguous conflicts are resolved by frequency
                     rather than fixed priority.

    Rules are applied in dependency order:
      1. Orientation-based anatomical exclusions  (absolute)
      2. Performer count / group logic            (absolute)
      3. Age exclusions                           (absolute + count-aware)
      4. Solo: remove all partner-dependent acts  (absolute)
      5. Role / dominance exclusions              (count-aware)
      6. Body attribute exclusions                (count-aware)
      7. Production format exclusions             (count-aware)
      8. Anal / vaginal exclusion                 (count-aware)
      9. Animation / rendered content             (count-aware)
    """
    cats = list(categories)

    def cl() -> set:
        return {c.lower() for c in cats}

    def remove(*names: str):
        low = {n.lower() for n in names}
        nonlocal cats
        cats = [c for c in cats if c.lower() not in low]

    def add(name: str):
        if name.lower() not in cl():
            cats.append(name)

    # ── Count helpers ─────────────────────────────────────────────────────────

    def cnt(name: str) -> int:
        """Detection count for a category name (case-insensitive). 0 if unknown."""
        if not counts:
            return 0
        name_l = name.lower()
        for k, v in counts.items():
            if k.lower() == name_l:
                return v
        return 0

    def resolve_pair(a: str, b: str, default_keep: str) -> str:
        """Return the name to keep when a and b conflict.
        Uses detection counts when available; otherwise uses default_keep."""
        if counts:
            ca, cb = cnt(a), cnt(b)
            if ca != cb:
                return a if ca > cb else b
        return default_keep

    def resolve_group(names: List[str], default_priority: List[str]) -> str:
        """Return the single name to keep from a set of mutually exclusive names.
        Uses highest detection count; ties broken by default_priority order."""
        if counts:
            best_count = max(cnt(n) for n in names)
            candidates = [n for n in names if cnt(n) == best_count]
            if len(candidates) == 1:
                return candidates[0]
            # Tie — fall through to priority
            names = candidates
        for p in default_priority:
            for n in names:
                if n.lower() == p.lower():
                    return n
        return names[0]

    # ── Anatomy / orientation constant sets ───────────────────────────────────

    requires_penis = {
        "blowjob", "deepthroat", "handjob", "titty fucking", "footjob",
        "cumshot", "facial", "creampie", "big cock", "small cock",
        "cbt",
    }
    requires_vagina = {
        "pussy licking", "squirt",
    }
    female_only = {
        "pregnant", "lactating", "milf",
    }
    real_only = {
        "amateur", "pornstar", "webcam", "onlyfans", "casting", "crossdresser",
    }
    animation_types = {
        "3d", "anime", "cartoon", "hentai", "furry", "futanari", "tentacle",
    }
    real_performer_indicators = {
        "pornstar", "amateur",
        "blonde", "brunette", "red head",
        "big tits", "small tits", "big ass", "big cock", "small cock",
        "teen", "mature", "milf", "granny",
        "oiled", "tattoo", "hairy", "bbw", "chubby", "skinny",
    }

    # ══════════════════════════════════════════════════════════════════════════
    # 1. Orientation-based exclusions  (ABSOLUTE — counts cannot override)
    # ══════════════════════════════════════════════════════════════════════════

    if orientation == "gay":
        # All-male: no vaginal acts, no female-only tags, no strap-on pegging
        remove(*requires_vagina, *female_only, "granny",
               "lesbian", "pegging", "strapon", "futanari",
               "big tits", "small tits")

    elif orientation == "shemale":
        # Trans performer (has penis): no vaginal acts
        remove(*requires_vagina, "pregnant", "lactating",
               "lesbian", "gay", "pegging", "strapon")

    elif orientation == "straight":
        remove("gay")
        if "lesbian" in cl():
            # All-female sub-scene within straight label
            remove(*requires_penis, "pegging", "bisexual male")

    # ══════════════════════════════════════════════════════════════════════════
    # 2. Performer count / group logic  (ABSOLUTE)
    # ══════════════════════════════════════════════════════════════════════════
    cats_l = cl()

    # Gangbang always implies Group Sex
    if "gangbang" in cats_l and "group sex" not in cats_l:
        add("Group Sex")

    cats_l = cl()

    # Threesome (3) vs Group Sex (4+) — keep Group Sex
    if "threesome" in cats_l and "group sex" in cats_l:
        remove("Threesome")

    cats_l = cl()

    # Couple (exactly 2) is incompatible with Threesome / Group / Gangbang
    if "couple" in cats_l and any(x in cats_l for x in ("threesome", "group", "gangbang")):
        remove("Couple")

    cats_l = cl()

    # Double Penetration requires ≥3 performers
    if "double penetration" in cats_l:
        if not any(x in cats_l for x in ("threesome", "group", "gangbang")):
            remove("Double Penetration")

    cats_l = cl()

    # Cuckold requires ≥3 performers (couple + third)
    if "cuckold" in cats_l:
        if not any(x in cats_l for x in ("threesome", "group", "gangbang")):
            remove("Cuckold")

    cats_l = cl()

    # Bisexual Male is incompatible with an all-female scene
    if "bisexual male" in cats_l and "lesbian" in cats_l:
        remove("Lesbian")

    cats_l = cl()

    # Bisexual Male needs ≥3 performers with mixed genders
    if "bisexual male" in cats_l:
        if not any(x in cats_l for x in ("group", "threesome", "gangbang")):
            remove("Bisexual Male")

    cats_l = cl()

    # Interracial requires ≥2 performers — incompatible with Solo
    if "interracial" in cats_l and "solo" in cats_l:
        remove("Interracial")

    cats_l = cl()

    # Track multi-performer context (used for attribute exclusions below)
    is_multi = any(x in cats_l for x in ("couple", "threesome", "group", "gangbang"))

    # ══════════════════════════════════════════════════════════════════════════
    # 3. Age category logic
    # ══════════════════════════════════════════════════════════════════════════
    cats_l = cl()

    age_young = {"teen", "young"}
    age_old   = {"mature", "milf", "granny"}
    has_young = bool(age_young & cats_l)
    has_old   = bool(age_old & cats_l)

    # Auto-tag Old & Young when both age brackets are present together
    if has_young and has_old and "old & young" not in cats_l:
        add("Old & Young")

    cats_l = cl()

    # Remove Old & Young if only one bracket is present (mislabeled)
    if "old & young" in cats_l and not (has_young and has_old):
        remove("Old & Young")

    cats_l = cl()

    # In single-performer context: age brackets are mutually exclusive.
    # Resolve by count (which bracket was seen more consistently).
    if "old & young" not in cats_l and not is_multi:
        present_old = age_old & cats_l

        if has_young and present_old:
            # Determine dominant bracket by total count
            young_score = sum(cnt(t) for t in age_young & cats_l)
            old_score   = sum(cnt(t) for t in present_old)
            if old_score > young_score:
                remove("Teen", "Young")
                # Within old bracket: keep highest-count tag
                if len(present_old) > 1:
                    keep = resolve_group(
                        list(present_old),
                        ["Granny", "Milf", "Mature"],
                    )
                    remove(*[t for t in present_old if t != keep.lower()])
            else:
                remove("Mature", "Milf", "Granny")

        elif present_old and len(present_old) > 1:
            # Multiple old-bracket tags without young — keep the most-detected one
            keep = resolve_group(
                list(present_old),
                ["Granny", "Milf", "Mature"],
            )
            remove(*[t for t in present_old if t != keep.lower()])

    # ══════════════════════════════════════════════════════════════════════════
    # 4. Solo: remove all partner-dependent acts  (ABSOLUTE)
    # ══════════════════════════════════════════════════════════════════════════
    cats_l = cl()

    if "solo" in cats_l:
        # Solo wins only if it appears MORE often than partner-dependent acts.
        # If gangbang/blowjob etc. appear more often → video is not solo, remove Solo.
        solo_count = cnt("Solo")
        partner_acts = ["gangbang", "blowjob", "group sex", "threesome", "couple",
                        "double penetration", "cumshot", "facial", "creampie",
                        "pussy licking", "handjob", "titty fucking", "casting"]
        max_partner_count = max((cnt(a) for a in partner_acts if a in cats_l), default=0)
        if max_partner_count >= solo_count:
            # Partner acts dominate — drop Solo
            remove("Solo")
        else:
            # Solo dominates — drop partner-dependent acts
            remove(
                "Group Sex", "Threesome", "Gangbang", "Couple",
                "Double Penetration", "Bisexual Male", "Lesbian", "Cuckold",
                "Pussy Licking", "Rimjob", "Blowjob", "Deepthroat",
                "Handjob", "Titty Fucking", "Footjob", "CBT",
                "Pegging", "Strapon", "Fisting",
                "Cumshot", "Facial", "Creampie",
                "Casting", "Interracial", "Old & Young",
                "Femdom", "Bondage",
            )

    # ══════════════════════════════════════════════════════════════════════════
    # 5. Role / dominance exclusions  (count-aware)
    # ══════════════════════════════════════════════════════════════════════════
    cats_l = cl()

    if orientation == "gay":
        remove("Femdom")  # gender-coded, no meaning in all-male scene

    # ══════════════════════════════════════════════════════════════════════════
    # 6. Body attribute mutual exclusions  (count-aware)
    # ══════════════════════════════════════════════════════════════════════════
    cats_l = cl()

    # Breast size: mutually exclusive in single-performer context
    if "big tits" in cats_l and "small tits" in cats_l and not is_multi:
        loser = resolve_pair("Big Tits", "Small Tits", default_keep="Big Tits")
        remove("Big Tits" if loser != "Big Tits" else "Small Tits")

    cats_l = cl()

    # Dick size: mutually exclusive
    if "big cock" in cats_l and "small cock" in cats_l:
        loser = resolve_pair("Big Cock", "Small Cock", default_keep="Big Cock")
        remove("Big Cock" if loser != "Big Cock" else "Small Cock")

    cats_l = cl()

    # Body type: mutually exclusive in single-performer context
    # Priority when tied: BBW > Chubby > Skinny
    body_types = {"bbw", "chubby", "skinny"}
    present_body = body_types & cats_l
    if len(present_body) > 1 and not is_multi:
        keep = resolve_group(
            list(present_body),
            ["BBW", "Chubby", "Skinny"],
        )
        remove(*[b for b in present_body if b != keep.lower()])

    cats_l = cl()

    # Hair color: mutually exclusive in single-performer context
    # Priority when tied: Blonde > Brunette > Red Head
    hair_colors = {"blonde", "brunette", "red head"}
    present_hair = hair_colors & cats_l
    if len(present_hair) > 1 and not is_multi:
        keep = resolve_group(
            list(present_hair),
            ["Blonde", "Brunette", "Red Head"],
        )
        remove(*[h for h in present_hair if h != keep.lower()])

    # ══════════════════════════════════════════════════════════════════════════
    # 7. Production / format mutual exclusions  (count-aware)
    # ══════════════════════════════════════════════════════════════════════════
    cats_l = cl()

    # Japanese Censored vs Uncensored
    if "japanese censored" in cats_l and "japanese uncensored" in cats_l:
        loser = resolve_pair(
            "Japanese Censored", "Japanese Uncensored",
            default_keep="Japanese Censored",
        )
        remove("Japanese Censored" if loser != "Japanese Censored" else "Japanese Uncensored")

    cats_l = cl()

    # Amateur vs Pornstar
    if "amateur" in cats_l and "pornstar" in cats_l:
        loser = resolve_pair("Amateur", "Pornstar", default_keep="Pornstar")
        remove("Amateur" if loser != "Amateur" else "Pornstar")

    cats_l = cl()

    # Vintage: incompatible with modern formats (absolute — vintage cannot be HD/VR)
    if "vintage" in cats_l:
        remove("HD", "Virtual Reality", "Vertical Video", "OnlyFans", "Webcam", "4K")

    cats_l = cl()

    # Virtual Reality vs Vertical Video — different capture formats; VR wins
    if "virtual reality" in cats_l and "vertical video" in cats_l:
        remove("Vertical Video")

    cats_l = cl()

    # VR and POV serve the same immersion purpose — VR wins
    if "virtual reality" in cats_l and "pov" in cats_l:
        remove("POV")

    # ══════════════════════════════════════════════════════════════════════════
    # 8. Anal / vaginal exclusion  (count-aware)
    # ══════════════════════════════════════════════════════════════════════════
    cats_l = cl()

    vaginal_indicators = {"creampie", "squirt", "pussy licking"}
    present_vaginal = vaginal_indicators & cats_l

    if "anal" in cats_l and "double penetration" not in cats_l and present_vaginal:
        anal_count    = cnt("anal")
        vaginal_count = max(cnt(v) for v in present_vaginal)

        if counts and anal_count != vaginal_count:
            if anal_count > vaginal_count:
                # Anal is the consistent signal; vaginal tag is the false positive
                remove(*present_vaginal)
            else:
                # Vaginal activity is the consistent signal; anal is the false positive
                remove("Anal")
        else:
            # No counts available — default: vaginal evidence removes Anal
            remove("Anal")

    # ══════════════════════════════════════════════════════════════════════════
    # 9. Animation / rendered content  (count-aware)
    # ══════════════════════════════════════════════════════════════════════════
    cats_l = cl()

    if "gameplay video" in cats_l:
        keep_set = {"gameplay video", "hd", "4k", "vertical video", "virtual reality",
                    "vintage", "anime", "cartoon", "3d"}
        cats = [c for c in cats if c.lower() in keep_set]

    elif any(a in cats_l for a in animation_types):
        present_anim = animation_types & cats_l
        present_real = real_performer_indicators & cats_l

        if present_real:
            anim_score = sum(cnt(a) for a in present_anim)
            real_score = sum(cnt(r) for r in present_real)

            if counts and real_score != anim_score:
                if real_score > anim_score:
                    # Real performers dominate — animation tags are false positives
                    remove("Anime", "Hentai", "Cartoon", "3D",
                           "Furry", "Futanari", "Tentacle")
                else:
                    # Animation dominates — real-performer tags are false positives
                    remove(*real_only)
            else:
                # No counts or tied — default: real evidence removes animation tags
                remove("Anime", "Hentai", "Cartoon", "3D",
                       "Furry", "Futanari", "Tentacle")
        else:
            # No real-performer indicators at all → confirmed animation
            remove(*real_only)

    cats_l = cl()

    # Futanari is an animation-specific concept — remove outside animation context
    if "futanari" in cats_l and not any(a in cats_l for a in animation_types):
        remove("Futanari")

    # ══════════════════════════════════════════════════════════════════════════
    # 10. Hairy: require at least 2 scene detections to avoid head-hair FP
    # ══════════════════════════════════════════════════════════════════════════
    cats_l = cl()

    if "hairy" in cats_l and cnt("Hairy") < 2:
        remove("Hairy")

    return cats
