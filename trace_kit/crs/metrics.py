"""Evaluation metrics for CRS baseline comparison.

Includes:
- Text quality: BLEU-4, ROUGE-L (self-implemented, no external deps)
- Recommendation quality: Recall@K, MRR
- Grounding quality: quote-to-source fuzzy match ratio
- Citation quality: citation density (verbatim token ratio)
- Provenance quality: provenance coverage (aspect-citation alignment)
- Practical value: price alignment (budget match), spatial coherence, itinerary diversity
- Dialogue-level: TSR, T2R, Rejection Recovery Rate
- LLM-as-Judge: 5-dimension scoring rubric
- Aggregation: per-action, overall, and dialogue-level macro-average
"""

import logging
import math
import re
import threading
import time
from collections import Counter
from typing import Dict, List, Optional

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

from trace_kit.datasets.yelp.models import YelpPOI


# ---------------------------------------------------------------------------
# Text metrics
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lowercase word tokenization (separates punctuation)."""
    return re.findall(r'\w+', text.lower())


def _ngrams(tokens: List[str], n: int) -> List[tuple]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def compute_bleu4(reference: str, hypothesis: str) -> float:
    """Compute smoothed BLEU-4 (Chen & Cherry method 1).

    Smoothing: for n-gram orders with zero matches, use 1/(2^k) where k is
    the number of consecutive zero-count orders from that point.
    """
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)

    if not hyp_tokens or not ref_tokens:
        return 0.0

    # Brevity penalty
    bp = math.exp(1 - len(ref_tokens) / len(hyp_tokens)) if len(hyp_tokens) < len(ref_tokens) else 1.0

    log_avg = 0.0
    k = 0  # consecutive zero-count counter for Chen & Cherry Method 1
    for n in range(1, 5):
        ref_ngrams = Counter(_ngrams(ref_tokens, n))
        hyp_ngrams = Counter(_ngrams(hyp_tokens, n))

        if not hyp_ngrams:
            return 0.0

        # Clipped counts
        clipped = 0
        for ng, count in hyp_ngrams.items():
            clipped += min(count, ref_ngrams.get(ng, 0))

        total = sum(hyp_ngrams.values())

        if clipped == 0:
            k += 1
            # Chen & Cherry smoothing method 1: 1 / (2^k) for consecutive zero orders
            precision = 1.0 / (2 ** k)
        else:
            precision = clipped / total

        log_avg += math.log(precision)

    log_avg /= 4
    return bp * math.exp(log_avg)


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L F1 via longest common subsequence."""
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)

    if not ref_tokens or not hyp_tokens:
        return 0.0

    m, n = len(ref_tokens), len(hyp_tokens)

    # DP for LCS length
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr

    lcs_len = prev[n]
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / n
    recall = lcs_len / m
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Recommendation metrics
# ---------------------------------------------------------------------------

def compute_recommendation_metrics(
    gold_poi_ids: List[str],
    predicted_poi_ids: List[str],
) -> Dict[str, float]:
    """Compute Recall@1, Recall@3, and MRR.

    Args:
        gold_poi_ids: Ground-truth recommended POI IDs (unordered set).
        predicted_poi_ids: Model-predicted POI IDs (ordered by confidence).

    Returns:
        Dict with 'recall@1', 'recall@3', 'mrr'.

    Note: MRR is computed as the mean reciprocal rank across all gold
    POI IDs (not standard IR MRR which uses only the first relevant result).
    """
    if not gold_poi_ids:
        return {'recall@1': 0.0, 'recall@3': 0.0, 'mrr': 0.0}

    gold_set = set(gold_poi_ids)

    # Recall@K
    def recall_at_k(k: int) -> float:
        hits = gold_set & set(predicted_poi_ids[:k])
        return len(hits) / len(gold_set)

    # MRR: for each gold POI, find its rank in predicted list
    reciprocal_ranks = []
    for gid in gold_poi_ids:
        try:
            rank = predicted_poi_ids.index(gid) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    return {
        'recall@1': recall_at_k(1),
        'recall@3': recall_at_k(3),
        'mrr': mrr,
    }


# ---------------------------------------------------------------------------
# Grounding metrics
# ---------------------------------------------------------------------------

def build_review_lookup(pois: List[YelpPOI]) -> Dict[str, str]:
    """Build mapping from review reference ID to review text.

    Key format: "{business_id}::R{n}" (1-indexed).
    """
    lookup = {}
    for poi in pois:
        for i, rev in enumerate(poi.reviews, 1):
            key = f"{poi.business_id}::R{i}"
            lookup[key] = rev.text
    return lookup


def compute_grounding_score(
    response_text: str,
    referenced_review_ids: List[str],
    review_lookup: Dict[str, str],
    threshold: float = 0.80,
) -> float:
    """Compute fraction of quoted text that is grounded in source reviews.

    Extracts double-quoted substrings (>=10 chars) from response_text,
    checks each against the referenced reviews using fuzzy partial_ratio.

    Returns:
        Ratio of grounded quotes in [0, 1]. Returns 1.0 if no quotes found
        (no grounding claim → vacuously true).
    """
    quote_pattern = re.compile(r'"([^"]{10,})"')
    quotes = quote_pattern.findall(response_text)

    if not quotes:
        return 1.0

    grounded = 0
    for quote in quotes:
        best_score = 0.0
        # Check referenced reviews first
        for ref_id in referenced_review_ids:
            source = review_lookup.get(ref_id, '')
            if not source:
                continue
            score = fuzz.partial_ratio(quote.lower(), source.lower()) / 100.0
            best_score = max(best_score, score)

        # Fallback: check all reviews
        if best_score < threshold:
            for source in review_lookup.values():
                score = fuzz.partial_ratio(quote.lower(), source.lower()) / 100.0
                best_score = max(best_score, score)
                if best_score >= threshold:
                    break

        if best_score >= threshold:
            grounded += 1

    return grounded / len(quotes)


def compute_composite_grounding(
    grounding_score: float,
    citation_density: float,
    provenance_coverage: float,
) -> float:
    """Composite grounding metric that penalizes responses with no citations.

    Addresses the reviewer concern that GS=1.0 when no quotes are present,
    which can incentivize under-citation. The composite score requires both
    high fuzzy-match grounding AND non-trivial citation density.

    Formula: GS * min(1.0, CD / 0.05) * (0.5 + 0.5 * PC)
    - GS=1.0 with CD=0 → composite=0 (no citations = no grounding credit)
    - GS=1.0 with CD=0.4 → composite depends on PC
    - GS=0.9 with CD=0.3 and PC=0.8 → ~0.81

    Returns:
        Composite score in [0, 1].
    """
    cd_gate = min(1.0, citation_density / 0.05)
    pc_weight = 0.5 + 0.5 * provenance_coverage
    return grounding_score * cd_gate * pc_weight


def compute_citation_density(
    response_text: str,
    review_lookup: Dict[str, str],
    min_span_tokens: int = 5,
) -> float:
    """Compute fraction of response tokens that are verbatim citations from reviews.

    Extracts all double-quoted substrings from the response, checks if each is
    a verbatim substring of any source review (case-insensitive), and counts
    the cited tokens vs total response tokens.

    Args:
        response_text: The baseline's generated response.
        review_lookup: Mapping from review ref ID to review text.
        min_span_tokens: Minimum token length for a quote to count.

    Returns:
        Ratio of verbatim-cited tokens to total response tokens in [0, 1].
        Returns 0.0 if no quotes or no reviews.
    """
    response_tokens = _tokenize(response_text)
    if not response_tokens or not review_lookup:
        return 0.0

    quote_pattern = re.compile(r'"([^"]{5,})"')
    quotes = quote_pattern.findall(response_text)
    if not quotes:
        return 0.0

    # Build a single lowercase corpus of all review text for substring matching
    all_reviews_lower = {rid: text.lower() for rid, text in review_lookup.items()}

    cited_token_count = 0
    for quote in quotes:
        quote_lower = quote.lower()
        quote_tokens = _tokenize(quote)
        if len(quote_tokens) < min_span_tokens:
            continue
        # Check if this quote is a verbatim substring of any review
        for review_text_lower in all_reviews_lower.values():
            if quote_lower in review_text_lower:
                cited_token_count += len(quote_tokens)
                break

    return min(1.0, cited_token_count / len(response_tokens))


def compute_provenance_coverage(
    response_text: str,
) -> float:
    """Compute provenance coverage: fraction of aspect claims backed by citations.

    Detects aspect mentions (food/service/ambiance/value/location keywords)
    in the response, then checks if each mentioned aspect has at least one
    review citation nearby (within the same sentence or adjacent clause).

    Returns:
        Ratio of cited aspects to total mentioned aspects in [0, 1].
        Returns 1.0 if no aspect mentions found (vacuously covered).
    """
    aspect_keywords = {
        'food': {'food', 'dish', 'meal', 'menu', 'taste', 'flavor', 'pasta',
                 'steak', 'pizza', 'sushi', 'seafood', 'cuisine', 'chef',
                 'delicious', 'fresh', 'portion', 'appetizer', 'dessert'},
        'service': {'service', 'staff', 'waiter', 'server', 'host', 'attentive',
                    'friendly', 'rude', 'slow', 'helpful', 'prompt'},
        'ambiance': {'ambiance', 'ambience', 'atmosphere', 'vibe', 'decor',
                     'cozy', 'romantic', 'quiet', 'loud', 'elegant', 'music'},
        'value': {'price', 'value', 'expensive', 'cheap', 'affordable',
                  'overpriced', 'reasonable', 'worth', 'budget'},
        'location': {'location', 'parking', 'walk', 'downtown', 'nearby',
                     'convenient', 'distance', 'central'},
    }

    # Citation pattern: [Rn] or business_id::Rn
    citation_re = re.compile(r'\[R\d+\]|[a-zA-Z0-9_-]+::R\d+')

    # Split response into sentences
    sentences = re.split(r'(?<=[.!?])\s+', response_text)

    aspects_mentioned = set()
    aspects_cited = set()

    for sent in sentences:
        sent_lower = sent.lower()
        sent_tokens = set(sent_lower.split())
        has_citation = bool(citation_re.search(sent))

        for asp, keywords in aspect_keywords.items():
            if sent_tokens & keywords:
                aspects_mentioned.add(asp)
                if has_citation:
                    aspects_cited.add(asp)

    if not aspects_mentioned:
        return 1.0

    return len(aspects_cited) / len(aspects_mentioned)


# ---------------------------------------------------------------------------
# Entailment-based grounding (paraphrase-tolerant)
# ---------------------------------------------------------------------------

# Lazy-loaded singleton for NLI model
_nli_pipeline = None
_nli_lock = threading.Lock()


def _get_nli_pipeline():
    """Lazy-load the NLI cross-encoder model (thread-safe singleton)."""
    global _nli_pipeline
    if _nli_pipeline is not None:
        return _nli_pipeline
    with _nli_lock:
        if _nli_pipeline is not None:
            return _nli_pipeline
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        model_name = "cross-encoder/nli-deberta-v3-base"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading NLI model: {model_name} on {device}")
        # FP16 on GPU halves activation memory (DeBERTa disentangled-attention
        # is memory-hungry at long sequences); CPU stays FP32.
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=dtype,
        ).to(device)
        model.eval()
        _nli_pipeline = {
            'tokenizer': AutoTokenizer.from_pretrained(model_name),
            'model': model,
            'device': device,
        }
        # Label mapping: 0=contradiction, 1=entailment, 2=neutral
        _nli_pipeline['label_map'] = {0: 'contradiction', 1: 'entailment', 2: 'neutral'}
    return _nli_pipeline


def _check_entailment_batch(
    premises: List[str],
    hypotheses: List[str],
    threshold: float = 0.5,
) -> List[bool]:
    """Check entailment for a batch of (premise, hypothesis) pairs.

    Returns list of booleans: True if entailment score >= threshold.
    """
    import torch

    if not premises:
        return []

    nli = _get_nli_pipeline()
    tokenizer = nli['tokenizer']
    model = nli['model']

    # Tokenize all pairs
    device = nli['device']
    encodings = tokenizer(
        premises, hypotheses,
        padding=True, truncation=True, max_length=512,
        return_tensors='pt',
    )
    encodings = {k: v.to(device) for k, v in encodings.items()}

    with torch.no_grad():
        logits = model(**encodings).logits
        probs = torch.softmax(logits, dim=-1)
        # Column 1 = entailment probability
        entailment_probs = probs[:, 1].tolist()

    return [p >= threshold for p in entailment_probs]


def compute_entailment_grounding(
    response_text: str,
    referenced_review_ids: List[str],
    review_lookup: Dict[str, str],
    entailment_threshold: float = 0.5,
    batch_size: int = 32,
) -> Optional[float]:
    """Compute entailment-based grounding: fraction of response claims entailed by reviews.

    Unlike compute_grounding_score (which uses fuzzy string matching),
    this metric uses an NLI model to check semantic entailment,
    tolerating faithful paraphrases that don't match verbatim.

    Algorithm:
        1. Split response into sentences (claims)
        2. Filter to substantive claims (>5 words, not purely conversational)
        3. For each claim, check if ANY referenced review entails it
        4. Return fraction of entailed claims

    Args:
        response_text: The baseline's generated response.
        referenced_review_ids: Review IDs cited in this response.
        review_lookup: Mapping from review ref ID to review text.
        entailment_threshold: Minimum entailment probability to count as grounded.
        batch_size: Batch size for NLI inference.

    Returns:
        Fraction of entailed claims in [0, 1].
        Returns None if no substantive claims found.
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', response_text.strip())

    # Filter to substantive claims (skip greetings and short fragments)
    # Only filter truly conversational filler — NOT recommendation-introducing
    # phrases like "based on" or "i'd recommend" which often contain key claims
    _greeting_prefixes = {
        'hi', 'hello', 'sure', 'great', 'of course', 'thank', 'thanks',
        "you're welcome", 'enjoy', 'have a', 'for you',
    }
    claims = []
    for sent in sentences:
        words = sent.split()
        if len(words) < 5:
            continue
        sent_lower = sent.lower().strip()
        if any(sent_lower.startswith(p) for p in _greeting_prefixes):
            if len(words) < 10:
                continue
        claims.append(sent)

    if not claims:
        return None

    # Collect review texts for entailment checking
    review_texts = []
    for ref_id in referenced_review_ids:
        text = review_lookup.get(ref_id, '')
        if text:
            # Truncate long reviews to fit NLI model context
            review_texts.append(text[:1000])

    # Fallback: if no referenced reviews, use all available (limited)
    if not review_texts:
        for text in list(review_lookup.values())[:5]:
            if text:
                review_texts.append(text[:1000])

    if not review_texts:
        return 0.0

    # Flatten all (review, claim) pairs into a single batch for efficiency
    # This avoids re-tokenizing the same reviews once per claim
    n_reviews = len(review_texts)
    all_premises = []
    all_hypotheses = []
    for claim in claims:
        all_premises.extend(review_texts)
        all_hypotheses.extend([claim] * n_reviews)

    # Run NLI in batches
    all_results = []
    for i in range(0, len(all_premises), batch_size):
        batch_p = all_premises[i:i + batch_size]
        batch_h = all_hypotheses[i:i + batch_size]
        all_results.extend(_check_entailment_batch(batch_p, batch_h, entailment_threshold))

    # Reshape: each claim has n_reviews consecutive results
    entailed_claims = 0
    for ci in range(len(claims)):
        start = ci * n_reviews
        if any(all_results[start:start + n_reviews]):
            entailed_claims += 1

    return entailed_claims / len(claims)


def _extract_claims_and_reviews(
    response_text: str,
    referenced_review_ids: List[str],
    review_lookup: Dict[str, str],
) -> tuple:
    """Shared helper: split response into substantive claims + collect review texts.

    Returns (claims, review_texts) where each is a list of strings, or
    (None, None) if there are no substantive claims.
    """
    sentences = re.split(r'(?<=[.!?])\s+', response_text.strip())
    _greeting_prefixes = {
        'hi', 'hello', 'sure', 'great', 'of course', 'thank', 'thanks',
        "you're welcome", 'enjoy', 'have a', 'for you',
    }
    claims = []
    for sent in sentences:
        words = sent.split()
        if len(words) < 5:
            continue
        sent_lower = sent.lower().strip()
        if any(sent_lower.startswith(p) for p in _greeting_prefixes):
            if len(words) < 10:
                continue
        claims.append(sent)
    if not claims:
        return None, None

    review_texts = []
    for ref_id in referenced_review_ids:
        text = review_lookup.get(ref_id, '')
        if text:
            review_texts.append(text[:1000])
    if not review_texts:
        for text in list(review_lookup.values())[:5]:
            if text:
                review_texts.append(text[:1000])
    return claims, review_texts


def batch_compute_entailment_grounding(
    turns_data: List[tuple],
    entailment_threshold: float = 0.5,
    batch_size: int = 64,
) -> List[Optional[float]]:
    """Vectorised entailment grounding across many turns.

    Args:
        turns_data: list of `(response_text, referenced_review_ids, review_lookup)`
            tuples, one per turn.
        entailment_threshold: NLI prob threshold (default 0.5, same as per-turn fn).
        batch_size: NLI inference batch size; 256 fits comfortably on a 24 GB GPU
            for 512-token DeBERTa-v3-base.

    Returns:
        list of fractions in [0, 1], `None` for turns with no substantive claims.

    Why: the per-turn `compute_entailment_grounding` issues one GPU call per
    turn (5--25 pairs). Each call has ~120 ms of CUDA / Python dispatch
    overhead, so we hit ~8 turns/sec on a 4090 even with the GPU at 80 %.
    By flattening pairs across many turns into one big list and shipping
    256-pair batches, we amortise the dispatch and get 5--10x throughput.
    """
    # Per-turn extract: claims, review texts
    per_turn = []  # list of (claims, review_texts) or (None, None)
    for response_text, ref_ids, review_lookup in turns_data:
        claims, review_texts = _extract_claims_and_reviews(
            response_text, ref_ids, review_lookup,
        )
        per_turn.append((claims, review_texts))

    # Build flat (premise, hypothesis, turn_idx, claim_idx) list across all turns
    pair_records = []  # list of (premise, hypothesis, turn_idx, claim_idx)
    for ti, (claims, reviews) in enumerate(per_turn):
        if claims is None or not reviews:
            continue
        for ci, claim in enumerate(claims):
            for review in reviews:
                pair_records.append((review, claim, ti, ci))

    # Run NLI in big batches; clear GPU cache periodically to avoid fragmentation
    all_results: List[bool] = []
    if pair_records:
        try:
            import torch as _t
            _has_cuda = _t.cuda.is_available()
        except Exception:
            _has_cuda = False
        for i in range(0, len(pair_records), batch_size):
            chunk = pair_records[i:i + batch_size]
            premises = [p[0] for p in chunk]
            hypotheses = [p[1] for p in chunk]
            all_results.extend(
                _check_entailment_batch(premises, hypotheses, entailment_threshold)
            )
            # Periodic cache clear: keep GPU fragmentation under control
            # without paying the full empty_cache() tax on every step.
            if _has_cuda and (i // batch_size) % 16 == 15:
                _t.cuda.empty_cache()
        if _has_cuda:
            _t.cuda.empty_cache()

    # OR-reduce per (turn_idx, claim_idx): claim is grounded if ANY review entails it
    # Use a list-of-sets indexed by (turn_idx, claim_idx)
    claim_entailed: Dict[tuple, bool] = {}
    for (premise, hyp, ti, ci), is_entailed in zip(pair_records, all_results):
        key = (ti, ci)
        claim_entailed[key] = claim_entailed.get(key, False) or is_entailed

    # Compute fraction per turn
    output: List[Optional[float]] = []
    for ti, (claims, reviews) in enumerate(per_turn):
        if claims is None:
            output.append(None)
            continue
        if not reviews:
            output.append(0.0)
            continue
        n_entailed = sum(claim_entailed.get((ti, ci), False) for ci in range(len(claims)))
        output.append(n_entailed / len(claims))
    return output


# ---------------------------------------------------------------------------
# Spatial metrics
# ---------------------------------------------------------------------------

def compute_spatial_coherence(
    recommended_poi_ids: List[str],
    poi_lookup: Dict[str, 'YelpPOI'],
    walkable_km: float = 2.0,
) -> Optional[float]:
    """Compute spatial coherence of recommended POIs.

    Measures: what fraction of recommended POI pairs are within walkable distance?
    High coherence = user can easily visit all recommendations on foot.

    Args:
        recommended_poi_ids: Business IDs of recommended POIs.
        poi_lookup: Mapping from business_id to YelpPOI (with lat/lon).
        walkable_km: Threshold for "walkable" distance (default 2km).

    Returns:
        Fraction of POI pairs within walkable distance, in [0, 1].
        Returns None if fewer than 2 POIs with valid coordinates.
    """
    coords = []
    for pid in recommended_poi_ids:
        poi = poi_lookup.get(pid)
        if poi and not (poi.latitude == 0 and poi.longitude == 0):
            coords.append((poi.latitude, poi.longitude))

    if len(coords) < 2:
        return None

    n_pairs = 0
    n_walkable = 0
    R = 6371.0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            lat1, lon1 = coords[i]
            lat2, lon2 = coords[j]
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = (math.sin(dlat / 2) ** 2 +
                 math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
                 math.sin(dlon / 2) ** 2)
            dist_km = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            n_pairs += 1
            if dist_km <= walkable_km:
                n_walkable += 1

    return n_walkable / n_pairs if n_pairs > 0 else None


def compute_route_efficiency(
    recommended_poi_ids: List[str],
    poi_lookup: Dict[str, 'YelpPOI'],
) -> Optional[float]:
    """Compute route efficiency as total path distance (nearest-neighbor TSP).

    Returns the total walking distance in km if all recommended POIs are visited
    in greedy nearest-neighbor order. Lower is better (tighter spatial clustering).

    Returns None if fewer than 2 POIs with valid coordinates.
    """
    coords = []
    for pid in recommended_poi_ids:
        poi = poi_lookup.get(pid)
        if poi and not (poi.latitude == 0 and poi.longitude == 0):
            coords.append((poi.latitude, poi.longitude))

    if len(coords) < 2:
        return None

    R = 6371.0

    def _haversine(c1, c2):
        lat1, lon1 = c1
        lat2, lon2 = c2
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Greedy nearest-neighbor TSP
    visited = [False] * len(coords)
    visited[0] = True
    current = 0
    total_km = 0.0

    for _ in range(len(coords) - 1):
        best_dist = float('inf')
        best_next = -1
        for j in range(len(coords)):
            if not visited[j]:
                d = _haversine(coords[current], coords[j])
                if d < best_dist:
                    best_dist = d
                    best_next = j
        visited[best_next] = True
        total_km += best_dist
        current = best_next

    return total_km


def compute_itinerary_diversity(
    recommended_poi_ids: List[str],
    poi_lookup: Dict[str, 'YelpPOI'],
) -> Optional[float]:
    """Compute itinerary type diversity of recommended POIs.

    Measures: what fraction of distinct POI types (restaurant/hotel/attraction)
    are represented in the recommendations? Higher = more diverse itinerary.

    Args:
        recommended_poi_ids: Business IDs of recommended POIs.
        poi_lookup: Mapping from business_id to YelpPOI.

    Returns:
        Fraction of unique POI types / 3 (max 3 types), in [0, 1].
        Returns None if no valid POIs.
    """
    types = set()
    for pid in recommended_poi_ids:
        poi = poi_lookup.get(pid)
        if poi:
            types.add(poi.poi_type)

    if not types:
        return None

    # Normalize by max possible types (restaurant, hotel, attraction = 3)
    MAX_POI_TYPES = 3
    return len(types) / MAX_POI_TYPES


# ---------------------------------------------------------------------------
# Price alignment metrics
# ---------------------------------------------------------------------------

# Price preference tiers: keyword → maximum acceptable price level (number of $)
_BUDGET_KEYWORDS = {
    'budget': 2, 'cheap': 2, 'affordable': 2, 'inexpensive': 2,
    'low cost': 2, 'economical': 2, 'frugal': 2, 'thrifty': 2,
    'tight budget': 1, 'backpacker': 1, 'student': 2,
}
_MIDRANGE_KEYWORDS = {
    'mid-range': 3, 'midrange': 3, 'moderate': 3, 'mid range': 3,
    'reasonable': 3, 'middle': 3, 'not too expensive': 3,
    'moderately priced': 3,
}
_UPSCALE_KEYWORDS = {
    'upscale': 4, 'luxury': 4, 'fine dining': 4, 'high-end': 4,
    'premium': 4, 'expensive': 4, 'splurge': 4, 'fancy': 4,
    'elegant': 4,
}
# Explicit dollar-sign patterns
_DOLLAR_SIGN_MAP = {
    '$$$$': 4, '$$$': 3, '$$': 2, '$': 1,
}


def extract_price_preference(dialogue_texts: List[str]) -> Optional[int]:
    """Extract user's price preference from dialogue user turns.

    Scans all user utterances for price-related keywords and returns the
    maximum acceptable price level (1-4, corresponding to $-$$$$).

    Returns None if no price preference is expressed.
    """
    combined = ' '.join(dialogue_texts).lower()

    # Check explicit dollar signs first (most specific)
    for pattern, level in sorted(_DOLLAR_SIGN_MAP.items(), key=lambda x: -len(x[0])):
        if pattern in combined:
            return level

    # Check keyword tiers (budget < midrange < upscale)
    for keywords in [_BUDGET_KEYWORDS, _MIDRANGE_KEYWORDS, _UPSCALE_KEYWORDS]:
        for kw, level in keywords.items():
            if kw in combined:
                return level

    return None


def _price_range_to_level(price_range: Optional[str]) -> Optional[int]:
    """Convert '$'-style price range to numeric level (1-4)."""
    if not price_range:
        return None
    return len(price_range)  # '$' → 1, '$$' → 2, '$$$' → 3, '$$$$' → 4


def compute_price_alignment(
    recommended_poi_ids: List[str],
    poi_lookup: Dict[str, 'YelpPOI'],
    user_price_max: int,
) -> Optional[float]:
    """Compute price alignment: fraction of recommended POIs within user's budget.

    A recommended POI is "aligned" if its price_range level ≤ user_price_max.
    POIs without price_range data are excluded from the calculation.

    Args:
        recommended_poi_ids: Business IDs of recommended POIs.
        poi_lookup: Mapping from business_id to YelpPOI (with price_range).
        user_price_max: Maximum acceptable price level (1-4).

    Returns:
        Fraction of price-known POIs within budget, in [0, 1].
        Returns None if no recommended POIs have price data.
    """
    aligned = 0
    total = 0
    for pid in recommended_poi_ids:
        poi = poi_lookup.get(pid)
        if not poi:
            continue
        poi_level = _price_range_to_level(poi.price_range)
        if poi_level is None:
            continue
        total += 1
        if poi_level <= user_price_max:
            aligned += 1

    return aligned / total if total > 0 else None


# ---------------------------------------------------------------------------
# Dialogue-level metrics
# ---------------------------------------------------------------------------

def compute_dialogue_metrics(
    dialogue_turns: List[Dict],
    gold_poi_ids_per_dialogue: List[str],
) -> Dict[str, float]:
    """Compute dialogue-level success and efficiency metrics.

    Operates on the per-turn results for a single dialogue, plus
    the gold POI IDs that should be recommended in that dialogue.

    Metrics:
        - task_success (TSR): 1.0 if any gold POI was recommended, else 0.0.
        - turns_to_first_correct (T2R): Turn index of first correct recommendation.
            Lower is better. Returns num_system_turns if never correct.
        - rejection_recovery_rate: After a reject_and_refine action, how often
            the next recommendation includes a gold POI. In [0, 1].

    Args:
        dialogue_turns: List of per-turn result dicts for one dialogue.
            Each must have 'action', 'turn_id', and optionally 'predicted_poi_ids'.
        gold_poi_ids_per_dialogue: The set of gold POI IDs for the dialogue.

    Returns:
        Dict with 'task_success', 'turns_to_first_correct', 'rejection_recovery_rate'.
    """
    gold_set = set(gold_poi_ids_per_dialogue)

    # Task Success Rate: did any system turn recommend a gold POI?
    system_turns = [t for t in dialogue_turns if t.get('action') in ('recommend', 'compare')]
    any_hit = False
    first_correct_turn_id = None

    for idx, turn in enumerate(system_turns):
        predicted = set(turn.get('predicted_poi_ids', []))
        if predicted & gold_set:
            any_hit = True
            if first_correct_turn_id is None:
                first_correct_turn_id = turn.get('turn_id', idx)

    task_success = 1.0 if any_hit else 0.0
    turns_to_first_correct = first_correct_turn_id if first_correct_turn_id is not None else len(dialogue_turns)

    # Rejection Recovery Rate: after reject_and_refine, does next recommend hit?
    rejection_recoveries = []
    all_turns = dialogue_turns
    for i, turn in enumerate(all_turns):
        if turn.get('action') != 'reject_and_refine':
            continue
        # Find next recommend/compare turn after this rejection
        for j in range(i + 1, len(all_turns)):
            next_turn = all_turns[j]
            if next_turn.get('action') in ('recommend', 'compare'):
                predicted = set(next_turn.get('predicted_poi_ids', []))
                rejection_recoveries.append(1.0 if predicted & gold_set else 0.0)
                break

    rejection_recovery = (
        sum(rejection_recoveries) / len(rejection_recoveries)
        if rejection_recoveries else None
    )

    result = {
        'task_success': task_success,
        'turns_to_first_correct': turns_to_first_correct,
    }
    if rejection_recovery is not None:
        result['rejection_recovery_rate'] = rejection_recovery

    return result


def aggregate_dialogue_metrics(per_dialogue_results: List[Dict]) -> Dict[str, float]:
    """Aggregate dialogue-level metric dicts into macro-averages.

    Args:
        per_dialogue_results: List of dicts from compute_dialogue_metrics().

    Returns:
        Dict with averaged metrics across all dialogues.
    """
    if not per_dialogue_results:
        return {}

    keys = ['task_success', 'turns_to_first_correct', 'rejection_recovery_rate']
    result = {'n_dialogues': len(per_dialogue_results)}

    for key in keys:
        vals = [d[key] for d in per_dialogue_results if key in d and d[key] is not None]
        if vals:
            result[key] = sum(vals) / len(vals)

    return result


# ---------------------------------------------------------------------------
# LLM-as-Judge
# ---------------------------------------------------------------------------

def compute_llm_judge_score(
    gold_text: str,
    predicted_text: str,
    dialogue_context: str,
    poi_context: str,
    model: str = "gemini/gemini-2.5-flash",
) -> Optional[Dict[str, float]]:
    """Compute LLM-as-Judge scores across 5 dimensions.

    Uses a DSPy Predict call to score the predicted response against the gold
    response on a 1-5 scale for each dimension.

    Dimensions:
        - relevance: Does the response address the user's actual need?
        - informativeness: Does it provide useful, specific information?
        - grounding_quality: Are claims supported by review evidence?
        - natural_flow: Does it read naturally as a dialogue turn?
        - recommendation_justification: Are recommendations well-reasoned?

    Returns:
        Dict with dimension scores (1-5) and 'overall' average.
        Returns None if LLM call fails.
    """
    import dspy
    from pydantic import BaseModel as PydanticBaseModel, Field as PydanticField

    class _JudgeOutput(PydanticBaseModel):
        relevance: int = PydanticField(
            description="1-5: Does the response address the user's stated need?",
        )
        informativeness: int = PydanticField(
            description="1-5: Does it provide useful, specific, actionable information?",
        )
        grounding_quality: int = PydanticField(
            description="1-5: Are claims supported by evidence from reviews?",
        )
        natural_flow: int = PydanticField(
            description="1-5: Does it read naturally as a conversational turn?",
        )
        recommendation_justification: int = PydanticField(
            description="1-5: Are recommendations well-reasoned and explained?",
        )

    class _JudgeSignature(dspy.Signature):
        """Score a CRS response on 5 quality dimensions (1-5 scale)."""
        prompt: str = dspy.InputField(desc="Evaluation context and instructions")
        output: _JudgeOutput = dspy.OutputField(desc="Scores for 5 quality dimensions")

    prompt = (
        "You are evaluating a tourism recommendation system response.\n\n"
        "## Dialogue Context\n"
        f"{dialogue_context}\n\n"
        "## Available POI Information\n"
        f"{poi_context[:2000]}\n\n"
        "## Gold (Reference) Response\n"
        f"{gold_text}\n\n"
        "## Predicted Response (to evaluate)\n"
        f"{predicted_text}\n\n"
        "## Scoring Rubric\n"
        "Rate the predicted response on each dimension (1=poor, 5=excellent):\n"
        "- **relevance**: Does it address what the user asked for?\n"
        "- **informativeness**: Does it provide useful, specific details?\n"
        "- **grounding_quality**: Are claims backed by actual review evidence?\n"
        "- **natural_flow**: Does it read naturally in the conversation?\n"
        "- **recommendation_justification**: Are rec choices well-explained?\n"
    )

    try:
        lm = dspy.LM(model=model)
        with dspy.context(lm=lm, adapter=dspy.JSONAdapter()):
            predictor = dspy.Predict(_JudgeSignature)
            result = predictor(prompt=prompt)
            scores: _JudgeOutput = result.output

        dims = {
            'relevance': max(1, min(5, scores.relevance)),
            'informativeness': max(1, min(5, scores.informativeness)),
            'grounding_quality': max(1, min(5, scores.grounding_quality)),
            'natural_flow': max(1, min(5, scores.natural_flow)),
            'recommendation_justification': max(1, min(5, scores.recommendation_justification)),
        }
        dims['overall'] = sum(dims.values()) / 5.0
        return dims

    except Exception as e:
        logger.warning(f"LLM judge scoring failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Cross-model judge agreement
# ---------------------------------------------------------------------------

def compute_inter_judge_agreement(
    cross_judge_results: Dict[str, Dict[str, List[Dict]]],
) -> Dict[str, float]:
    """Compute inter-judge agreement metrics across multiple LLM judges.

    Args:
        cross_judge_results: {judge_model: {baseline_name: [per_turn_results]}}

    Returns:
        Dict with 'kendall_tau', 'spearman_rho', and per-pair agreements.
    """
    from scipy import stats

    models = list(cross_judge_results.keys())
    if len(models) < 2:
        return {}

    # Extract per-baseline average judge_overall scores for each judge
    judge_scores = {}
    for model_name, baseline_results in cross_judge_results.items():
        model_scores = {}
        for baseline_name, results in baseline_results.items():
            judge_vals = [
                r['judge_overall'] for r in results
                if r.get('role') != 'user' and 'judge_overall' in r
            ]
            if judge_vals:
                model_scores[baseline_name] = sum(judge_vals) / len(judge_vals)
        judge_scores[model_name] = model_scores

    # Compute pairwise agreement
    agreement = {}
    common_baselines = set.intersection(
        *[set(scores.keys()) for scores in judge_scores.values()]
    )
    if len(common_baselines) < 3:
        agreement['warning'] = 'Too few common baselines for agreement computation'
        return agreement

    baselines_sorted = sorted(common_baselines)
    for i, m1 in enumerate(models):
        for m2 in models[i + 1:]:
            scores1 = [judge_scores[m1][b] for b in baselines_sorted]
            scores2 = [judge_scores[m2][b] for b in baselines_sorted]

            tau, tau_p = stats.kendalltau(scores1, scores2)
            rho, rho_p = stats.spearmanr(scores1, scores2)

            pair_key = f"{m1}_vs_{m2}"
            agreement[f'{pair_key}_kendall_tau'] = float(tau)
            agreement[f'{pair_key}_spearman_rho'] = float(rho)

    # Overall averages
    taus = [v for k, v in agreement.items() if 'kendall_tau' in k]
    rhos = [v for k, v in agreement.items() if 'spearman_rho' in k]
    if taus:
        agreement['avg_kendall_tau'] = sum(taus) / len(taus)
    if rhos:
        agreement['avg_spearman_rho'] = sum(rhos) / len(rhos)

    return agreement


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_metrics(per_turn_results: List[Dict]) -> Dict:
    """Aggregate per-turn metric dicts into a summary report.

    Each dict in per_turn_results should contain:
        - 'action': str (the gold action type)
        - 'bleu4': float
        - 'rouge_l': float
        - 'grounding': float
        - Optional recommendation metrics: 'recall@1', 'recall@3', 'mrr'

    Returns:
        Dict with 'overall' and 'per_action' macro-averages.
    """
    if not per_turn_results:
        return {'overall': {}, 'per_action': {}}

    # Group by action
    by_action: Dict[str, List[Dict]] = {}
    for r in per_turn_results:
        action = r.get('action', 'unknown')
        by_action.setdefault(action, []).append(r)

    def _average(results: List[Dict], key: str) -> Optional[float]:
        vals = [r[key] for r in results if key in r and r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    metric_keys = [
        'bleu4', 'rouge_l', 'grounding', 'citation_density',
        'provenance_coverage', 'entailment_grounding',
        'spatial_coherence', 'itinerary_diversity',
        'price_alignment',
        'recall@1', 'recall@3', 'mrr',
        'judge_relevance', 'judge_informativeness', 'judge_grounding_quality',
        'judge_natural_flow', 'judge_recommendation_justification', 'judge_overall',
    ]

    # Per-action averages
    per_action = {}
    for action, results in sorted(by_action.items()):
        summary = {'count': len(results)}
        for key in metric_keys:
            val = _average(results, key)
            if val is not None:
                summary[key] = val
        per_action[action] = summary

    # Overall averages
    overall = {'count': len(per_turn_results)}
    for key in metric_keys:
        val = _average(per_turn_results, key)
        if val is not None:
            overall[key] = val

    return {'overall': overall, 'per_action': per_action}
