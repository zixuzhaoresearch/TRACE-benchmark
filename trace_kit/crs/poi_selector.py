"""POI candidate selection strategies for CRS dialogue generation."""

import random
from collections import defaultdict
from typing import Dict, List, Optional

from trace_kit.datasets.yelp.models import YelpPOI


def select_candidate_pois(
    knowledge_base: List[YelpPOI],
    city: str,
    poi_type: str,
    n: int = 8,
) -> List[YelpPOI]:
    """Select candidate POIs for a dialogue scenario.

    Strategy:
    - Filter by city and type
    - Ensure rating diversity (mix of high-rated and mid-rated)
    - Ensure category diversity (avoid all-same-cuisine)
    - Require sufficient review quality

    Args:
        knowledge_base: Full list of POIs
        city: Target city
        poi_type: Target POI type
        n: Number of candidates to select

    Returns:
        List of selected YelpPOI candidates
    """
    # Filter
    pool = [
        p for p in knowledge_base
        if p.city == city and p.poi_type == poi_type and len(p.reviews) >= 3
    ]

    if len(pool) <= n:
        return pool

    # Split into tiers by rating
    pool.sort(key=lambda p: p.stars, reverse=True)
    high = [p for p in pool if p.stars >= 4.0]
    mid = [p for p in pool if 3.0 <= p.stars < 4.0]

    # Ensure at least 1 mid-rated for contrast (makes recommendations more realistic)
    n_high = max(n - 1, n * 2 // 3)
    n_mid = n - n_high

    selected = []

    if high:
        # Pick from high-rated with category diversity
        selected.extend(_diverse_sample(high, n_high))
    if mid and n_mid > 0:
        selected.extend(_diverse_sample(mid, n_mid))

    # Fill remaining from pool if needed
    if len(selected) < n:
        remaining = [p for p in pool if p not in selected]
        random.shuffle(remaining)
        selected.extend(remaining[:n - len(selected)])

    return selected[:n]


def _diverse_sample(pois: List[YelpPOI], n: int) -> List[YelpPOI]:
    """Sample POIs with category diversity."""
    if len(pois) <= n:
        return list(pois)

    # Group by primary category (first category)
    by_cat: Dict[str, List[YelpPOI]] = defaultdict(list)
    for p in pois:
        primary_cat = p.categories[0] if p.categories else 'Other'
        by_cat[primary_cat].append(p)

    selected = []
    categories = list(by_cat.keys())
    random.shuffle(categories)

    # Round-robin through categories
    cat_idx = 0
    while len(selected) < n:
        cat = categories[cat_idx % len(categories)]
        candidates = by_cat[cat]
        if candidates:
            choice = random.choice(candidates)
            selected.append(choice)
            candidates.remove(choice)
        cat_idx += 1
        # Safety: break if we've exhausted all categories
        if cat_idx > n * len(categories):
            break

    return selected[:n]


def expand_candidate_pool(
    current_candidates: List[YelpPOI],
    knowledge_base: List[YelpPOI],
    target_size: int,
    city: str,
    poi_type: str,
) -> List[YelpPOI]:
    """Expand a candidate POI set by sampling additional POIs from the same city/type.

    Used for candidate pool size ablation (e.g., 8 → 16 or 32).
    The original candidates are always included first, with new POIs appended.

    Args:
        current_candidates: Existing candidate POIs (preserved as-is)
        knowledge_base: Full POI knowledge base
        target_size: Desired total candidate pool size
        city: City to filter by
        poi_type: POI type to filter by

    Returns:
        Expanded list of candidate POIs (original + new)
    """
    if len(current_candidates) >= target_size:
        return current_candidates[:target_size]

    current_ids = {p.business_id for p in current_candidates}
    extra_pool = [
        p for p in knowledge_base
        if p.city == city and p.poi_type == poi_type
        and len(p.reviews) >= 3
        and p.business_id not in current_ids
    ]

    n_extra = target_size - len(current_candidates)
    if len(extra_pool) <= n_extra:
        extra = extra_pool
    else:
        extra = _diverse_sample(extra_pool, n_extra)

    return list(current_candidates) + extra


def select_for_dialogue_batch(
    knowledge_base: List[YelpPOI],
    city: str,
    poi_type: str,
    n_dialogues: int,
    pois_per_dialogue: int = 8,
) -> List[List[YelpPOI]]:
    """Select candidate POI sets for multiple dialogues, maximizing coverage.

    Tries to use different POIs across dialogues to avoid repetitive conversations.

    Returns:
        List of POI candidate lists, one per dialogue
    """
    pool = [
        p for p in knowledge_base
        if p.city == city and p.poi_type == poi_type and len(p.reviews) >= 3
    ]

    if not pool:
        return []

    # Track how often each POI has been used
    usage_count: Dict[str, int] = defaultdict(int)
    results = []

    for _ in range(n_dialogues):
        # Prefer less-used POIs
        pool_sorted = sorted(pool, key=lambda p: (usage_count[p.business_id], -p.stars))

        # Take from least-used, then apply diversity
        candidates = pool_sorted[:max(pois_per_dialogue * 3, len(pool))]
        selected = _diverse_sample(candidates, pois_per_dialogue)

        for p in selected:
            usage_count[p.business_id] += 1

        results.append(selected)

    return results
