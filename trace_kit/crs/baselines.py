"""CRS baseline methods for benchmark evaluation.

Twelve baselines of increasing sophistication:
1.  PopularityBaseline          — rank by stars/review_count, template responses (floor)
2.  TFIDFBaseline               — self-implemented TF-IDF over POI documents (standard IR)
3.  LLMZeroShotBaseline         — DSPy + Gemini with minimal prompt (ceiling)
4.  DSTBaseline                 — Schema-guided DST + constraint filtering + LLM generation
5.  RAGCitationBaseline         — RAG retrieval + citation-constrained generation
6.  AspectRetrievalBaseline     — Aspect-based review decomposition + aspect-weighted matching
7.  DenseRetrievalBaseline      — Sentence Transformer bi-encoder + negative preference
8.  SpatialBaseline             — Location-aware filtering via haversine distance
9.  HybridRRFBaseline           — Reciprocal Rank Fusion of sparse + dense + structured
10. MultiReviewSynthesisBaseline — Multi-review aspect synthesis with provenance tracking
11. MemoryAugmentedBaseline     — Persistent cross-session preference memory + warm-start
12. ItineraryBaseline           — Cross-type itinerary planning with nearby POI suggestions
"""

import logging
import math
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, runtime_checkable

import dspy
from pydantic import BaseModel, Field

from trace_kit.crs.models import DialogueTurn, ScenarioMeta, TourismDialogueState
from trace_kit.crs.prompts import format_poi_cards
from trace_kit.datasets.yelp.models import YelpPOI

logger = logging.getLogger(__name__)


def _build_lm(model: str, max_tokens: int = 600,
              api_key: Optional[str] = None, api_base: Optional[str] = None):
    """Build a dspy.LM with model-specific kwargs.

    Reasoning models (gpt-5.x) require temperature=1.0 and max_tokens >= 16000.
    Non-reasoning models accept the smaller max_tokens (default 600) for
    response-generation efficiency.
    """
    kwargs: Dict[str, object] = {"model": model}
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base
    if "gpt-5" in model:
        kwargs["temperature"] = 1.0
        kwargs["max_tokens"] = max(max_tokens, 16000)
    else:
        kwargs["max_tokens"] = max_tokens
    return dspy.LM(**kwargs)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

@dataclass
class BaselinePrediction:
    """Output of a baseline's predict_turn method."""
    response_text: str
    recommended_poi_ids: List[str] = field(default_factory=list)
    referenced_review_ids: List[str] = field(default_factory=list)


@runtime_checkable
class CRSBaseline(Protocol):
    """Protocol for CRS baselines."""
    name: str

    def predict_turn(
        self,
        dialogue_history: List[DialogueTurn],
        gold_action: str,
        candidate_pois: List[YelpPOI],
        scenario: ScenarioMeta,
    ) -> BaselinePrediction: ...

    def reset_state(self) -> None:
        """Reset dialogue state for a new dialogue. Optional."""
        ...

    def set_knowledge_base(self, pois: List[YelpPOI]) -> None:
        """Set the full POI knowledge base. Optional."""
        ...


_EMPTY_PREDICTION = BaselinePrediction(
    response_text="I don't have any recommendations available at the moment.",
    recommended_poi_ids=[],
    referenced_review_ids=[],
)

# Open-set candidate pool threshold. When candidate_pois exceeds this,
# LLM baselines pre-filter via retrieval to avoid prompt overflow.
OPEN_SET_PREFILTER_THRESHOLD = 16
OPEN_SET_PREFILTER_TOP_K = 16
# Which retriever to use for open-set LLM prefilter. 'tfidf' (default) or 'dense'.
OPEN_SET_PREFILTER_RETRIEVER = 'tfidf'

# Shared prefilter instances (lazy init). Mutated under _prefilter_lock when
# parallel workers are active — DenseRetrievalBaseline encodes on CUDA, which
# is not thread-safe; TFIDFBaseline.build_index also mutates per-instance state
# keyed by candidate set, so concurrent calls with different city/type pools
# corrupt the cache.
_prefilter_tfidf: Optional['TFIDFBaseline'] = None
_prefilter_dense: Optional['DenseRetrievalBaseline'] = None
_prefilter_lock = threading.Lock()


def prefilter_candidates(
    candidate_pois: List[YelpPOI],
    dialogue_history: List[DialogueTurn],
    top_k: int = OPEN_SET_PREFILTER_TOP_K,
    threshold: int = OPEN_SET_PREFILTER_THRESHOLD,
) -> List[YelpPOI]:
    """Pre-filter large candidate sets via retrieval for LLM baselines.

    When candidate_pois <= threshold, returns as-is (closed-set mode).
    When > threshold, builds a retriever index (TF-IDF or Dense per
    OPEN_SET_PREFILTER_RETRIEVER) and retrieves top-k POIs using dialogue
    history as query (open-set retrieve-then-generate).
    """
    if len(candidate_pois) <= threshold:
        return candidate_pois

    # Build query from dialogue history (user turns + last system turn)
    query_parts = []
    for t in dialogue_history:
        if t.role == 'user':
            query_parts.append(t.text)
        elif t.role == 'system' and t == dialogue_history[-1]:
            query_parts.append(t.text)
    query = ' '.join(query_parts[-5:])  # last 5 relevant turns

    global _prefilter_tfidf, _prefilter_dense
    with _prefilter_lock:
        if OPEN_SET_PREFILTER_RETRIEVER == 'dense':
            if _prefilter_dense is None:
                _prefilter_dense = DenseRetrievalBaseline()
            # DenseRetrievalBaseline._rank_pois builds its index internally
            ranked = _prefilter_dense._rank_pois(query, candidate_pois)
        else:
            if _prefilter_tfidf is None:
                _prefilter_tfidf = TFIDFBaseline()
            _prefilter_tfidf.build_index(candidate_pois)
            ranked = _prefilter_tfidf._rank_pois(query, candidate_pois)

    filtered = ranked[:top_k]
    logger.debug(
        f"Prefiltered {len(candidate_pois)} -> {len(filtered)} candidates "
        f"(top-{top_k} by {OPEN_SET_PREFILTER_RETRIEVER})"
    )
    return filtered


# ---------------------------------------------------------------------------
# 1. PopularityBaseline
# ---------------------------------------------------------------------------

class PopularityBaseline:
    """Rank POIs by popularity (stars, review_count). Ignores dialogue history.

    This is the floor baseline — demonstrates what happens when the system
    has no understanding of user preferences.
    """
    name = "popularity"

    def predict_turn(
        self,
        dialogue_history: List[DialogueTurn],
        gold_action: str,
        candidate_pois: List[YelpPOI],
        scenario: ScenarioMeta,
    ) -> BaselinePrediction:
        if not candidate_pois:
            return _EMPTY_PREDICTION

        ranked = sorted(
            candidate_pois,
            key=lambda p: (-p.stars, -p.review_count),
        )

        if gold_action == 'recommend':
            poi = ranked[0]
            review_text, review_id = self._pick_review(poi)
            text = (
                f"I'd recommend {poi.name}. "
                f"Rated {poi.stars}/5 with {poi.review_count} reviews. "
                f'One reviewer said: "{review_text}"'
            )
            return BaselinePrediction(
                response_text=text,
                recommended_poi_ids=[poi.business_id],
                referenced_review_ids=[review_id] if review_id else [],
            )

        if gold_action == 'compare':
            top2 = ranked[:2]
            parts = []
            poi_ids = []
            review_ids = []
            for poi in top2:
                review_text, review_id = self._pick_review(poi)
                parts.append(
                    f"{poi.name} ({poi.stars}/5, {poi.price_range or 'N/A'})"
                    f' — "{review_text}"'
                )
                poi_ids.append(poi.business_id)
                if review_id:
                    review_ids.append(review_id)
            if len(parts) < 2:
                text = f"Here's an option: {parts[0]}." if parts else "I don't have options to compare."
            else:
                text = f"Let me compare two options: {parts[0]}. Alternatively, {parts[1]}."
            return BaselinePrediction(
                response_text=text,
                recommended_poi_ids=poi_ids,
                referenced_review_ids=review_ids,
            )

        if gold_action == 'ask_preference':
            text = f"What kind of {scenario.primary_poi_type} are you looking for? Any preferences on cuisine, price range, or atmosphere?"
            return BaselinePrediction(response_text=text)

        # farewell or other
        text = "I hope that helps! Enjoy your visit."
        return BaselinePrediction(response_text=text)

    @staticmethod
    def _pick_review(poi: YelpPOI) -> tuple:
        """Pick the most useful review. Returns (text_snippet, review_ref_id)."""
        if not poi.reviews:
            return ("Great place!", None)
        # Pick highest-useful review
        best = max(poi.reviews, key=lambda r: (r.useful or 0, len(r.text)))
        best_idx = 0
        for rev_idx, rev in enumerate(poi.reviews):
            if rev is best:
                best_idx = rev_idx
                break
        ref_id = f"{poi.business_id}::R{best_idx + 1}"
        text = best.text[:150]
        return (text, ref_id)


# ---------------------------------------------------------------------------
# 2. TFIDFBaseline (self-implemented, no sklearn)
# ---------------------------------------------------------------------------

# Minimal stop words for TF-IDF
_STOP_WORDS = frozenset({
    'a', 'an', 'the', 'is', 'it', 'in', 'on', 'at', 'to', 'for', 'of',
    'and', 'or', 'but', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
    'might', 'shall', 'can', 'this', 'that', 'these', 'those', 'i', 'me',
    'my', 'we', 'our', 'you', 'your', 'he', 'she', 'they', 'them', 'his',
    'her', 'its', 'with', 'from', 'by', 'as', 'not', 'no', 'so', 'if',
    'are', 'am', 'there', 'here', 'what', 'which', 'who', 'whom', 'when',
    'where', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
    'other', 'some', 'such', 'than', 'too', 'very', 'just', 'about',
})


def _tfidf_tokenize(text: str) -> List[str]:
    tokens = re.findall(r'[a-z]+', text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


class _SparseVec:
    """Dict-based sparse vector with cosine similarity."""
    __slots__ = ('data',)

    def __init__(self, data: Optional[Dict[str, float]] = None):
        self.data = data or {}

    def cosine(self, other: '_SparseVec') -> float:
        dot = sum(self.data.get(k, 0) * v for k, v in other.data.items())
        norm_a = math.sqrt(sum(v * v for v in self.data.values()))
        norm_b = math.sqrt(sum(v * v for v in other.data.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class TFIDFBaseline:
    """TF-IDF retrieval baseline over candidate POI documents.

    Document per POI = name + categories + all review texts.
    Query = concatenation of all user turns in dialogue history.

    Includes index caching by candidate set to avoid redundant rebuilds.
    """
    name = "tfidf"

    def __init__(self):
        self._idf: Dict[str, float] = {}
        self._doc_vecs: Dict[str, _SparseVec] = {}
        self._doc_tokens: Dict[str, List[str]] = {}
        self._index_cache: Dict[frozenset, tuple] = {}
        self._current_key: frozenset = frozenset()

    def build_index(self, pois: List[YelpPOI]) -> None:
        """Build TF-IDF index for a set of candidate POIs.

        Caches the index by the set of business IDs, so repeated calls
        with the same candidate set are free.
        """
        cache_key = frozenset(p.business_id for p in pois)
        if cache_key == self._current_key:
            return
        if cache_key in self._index_cache:
            self._idf, self._doc_vecs, self._doc_tokens = self._index_cache[cache_key]
            self._current_key = cache_key
            return
        # Collect documents
        docs: Dict[str, List[str]] = {}
        for poi in pois:
            text_parts = [poi.name] + poi.categories
            text_parts += [r.text for r in poi.reviews]
            tokens = _tfidf_tokenize(' '.join(text_parts))
            docs[poi.business_id] = tokens

        self._doc_tokens = docs

        # Compute IDF
        n_docs = len(docs)
        df: Counter = Counter()
        for tokens in docs.values():
            df.update(set(tokens))

        self._idf = {
            term: math.log((n_docs + 1) / (count + 1)) + 1
            for term, count in df.items()
        }

        # Compute TF-IDF vectors
        self._doc_vecs = {}
        for bid, tokens in docs.items():
            tf = Counter(tokens)
            vec = {t: (1 + math.log(c)) * self._idf.get(t, 1.0) for t, c in tf.items()}
            self._doc_vecs[bid] = _SparseVec(vec)

        # Cache the index
        self._current_key = cache_key
        self._index_cache[cache_key] = (self._idf, self._doc_vecs, self._doc_tokens)

    def _query_vec(self, text: str) -> _SparseVec:
        tokens = _tfidf_tokenize(text)
        tf = Counter(tokens)
        vec = {t: (1 + math.log(c)) * self._idf.get(t, 1.0) for t, c in tf.items() if c > 0}
        return _SparseVec(vec)

    def _rank_pois(
        self,
        query_text: str,
        candidate_pois: List[YelpPOI],
    ) -> List[YelpPOI]:
        """Rank candidate POIs by TF-IDF cosine similarity to query."""
        if not self._doc_vecs:
            self.build_index(candidate_pois)

        q_vec = self._query_vec(query_text)
        scored = []
        for poi in candidate_pois:
            doc_vec = self._doc_vecs.get(poi.business_id, _SparseVec())
            scored.append((poi, q_vec.cosine(doc_vec)))
        scored.sort(key=lambda x: -x[1])
        return [poi for poi, _ in scored]

    def _pick_best_review(self, poi: YelpPOI, query_text: str) -> tuple:
        """Pick the review with highest term overlap to query."""
        if not poi.reviews:
            return ("Great place!", None)

        q_tokens = set(_tfidf_tokenize(query_text))
        best_rev = None
        best_overlap = -1
        best_idx = 0
        for rev_idx, rev in enumerate(poi.reviews):
            rev_tokens = set(_tfidf_tokenize(rev.text))
            overlap = len(q_tokens & rev_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_rev = rev
                best_idx = rev_idx

        if best_rev is None:
            best_rev = poi.reviews[0]
            best_idx = 0

        ref_id = f"{poi.business_id}::R{best_idx + 1}"
        return (best_rev.text[:150], ref_id)

    def predict_turn(
        self,
        dialogue_history: List[DialogueTurn],
        gold_action: str,
        candidate_pois: List[YelpPOI],
        scenario: ScenarioMeta,
    ) -> BaselinePrediction:
        # Build index if needed
        if not self._doc_vecs:
            self.build_index(candidate_pois)

        # Query = all user turns
        user_texts = [t.text for t in dialogue_history if t.role == 'user']
        query = ' '.join(user_texts) if user_texts else scenario.persona_description

        ranked = self._rank_pois(query, candidate_pois)

        if gold_action == 'recommend':
            poi = ranked[0]
            review_text, review_id = self._pick_best_review(poi, query)
            text = (
                f"Based on what you're looking for, I'd suggest {poi.name}. "
                f"It's rated {poi.stars}/5. "
                f'A reviewer mentioned: "{review_text}"'
            )
            return BaselinePrediction(
                response_text=text,
                recommended_poi_ids=[poi.business_id],
                referenced_review_ids=[review_id] if review_id else [],
            )

        if gold_action == 'compare':
            top2 = ranked[:2]
            parts = []
            poi_ids = []
            review_ids = []
            for poi in top2:
                review_text, review_id = self._pick_best_review(poi, query)
                parts.append(
                    f"{poi.name} ({poi.stars}/5, {poi.price_range or 'N/A'})"
                    f' — "{review_text}"'
                )
                poi_ids.append(poi.business_id)
                if review_id:
                    review_ids.append(review_id)
            if len(parts) < 2:
                text = f"Here's an option: {parts[0]}." if parts else "I don't have options to compare."
            else:
                text = f"Here are two great options: {parts[0]}. On the other hand, {parts[1]}."
            return BaselinePrediction(
                response_text=text,
                recommended_poi_ids=poi_ids,
                referenced_review_ids=review_ids,
            )

        if gold_action == 'ask_preference':
            text = f"Could you tell me more about what you're looking for in a {scenario.primary_poi_type}? Any specific cuisine, budget, or atmosphere preferences?"
            return BaselinePrediction(response_text=text)

        text = "Glad I could help! Have a wonderful time."
        return BaselinePrediction(response_text=text)


# ---------------------------------------------------------------------------
# 3. LLMZeroShotBaseline
# ---------------------------------------------------------------------------

class _LLMTurnOutput(BaseModel):
    """Structured output for LLM zero-shot baseline."""
    # Default to empty string so partial JSON (e.g. claude-haiku occasionally returns
    # only recommended_poi_ids) doesn't trigger Pydantic ValidationError → popularity
    # fallback. Empty text → BLEU/ROUGE 0, which truthfully reflects what happened.
    response_text: str = Field(default="", description="The system's response text")
    recommended_poi_ids: List[str] = Field(
        default_factory=list,
        description="POI business_ids recommended in this response",
    )
    referenced_review_ids: List[str] = Field(
        default_factory=list,
        description="Review reference IDs cited (format: business_id::R{n})",
    )


class _PredictTurnSignature(dspy.Signature):
    """Predict the next system turn in a CRS dialogue."""
    prompt: str = dspy.InputField(desc="Dialogue context and candidate POIs")
    output: _LLMTurnOutput = dspy.OutputField(desc="System response with POI/review references")


class LLMZeroShotBaseline:
    """LLM zero-shot baseline using DSPy + Gemini.

    Uses a minimal prompt (no DIALOGUE_GUIDELINES) — just dialogue history,
    POI cards, and a one-sentence instruction. This is the ceiling baseline.
    """
    name = "llm_zeroshot"

    def __init__(self, model: str = "gemini/gemini-2.5-flash",
                 api_key: Optional[str] = None, api_base: Optional[str] = None):
        self.model = model
        self._lm = _build_lm(model, max_tokens=600, api_key=api_key, api_base=api_base)
        self._predictor = dspy.Predict(_PredictTurnSignature)
        self._fallback = PopularityBaseline()

    def predict_turn(
        self,
        dialogue_history: List[DialogueTurn],
        gold_action: str,
        candidate_pois: List[YelpPOI],
        scenario: ScenarioMeta,
    ) -> BaselinePrediction:
        # Open-set: prefilter large candidate sets via TF-IDF retrieval
        filtered_pois = prefilter_candidates(candidate_pois, dialogue_history)

        # Build prompt
        history_text = _format_history(dialogue_history)
        poi_cards = format_poi_cards(filtered_pois)

        action_instruction = {
            'recommend': "Recommend the most suitable POI(s) from the candidates, quoting relevant reviews verbatim.",
            'compare': "Compare two or more candidate POIs, quoting relevant reviews verbatim.",
            'ask_preference': "Ask the user about their preferences to narrow down recommendations.",
            'farewell': "Wrap up the conversation warmly.",
        }.get(gold_action, f"Respond appropriately (action: {gold_action}).")

        prompt = (
            f"You are a tourism recommendation assistant.\n\n"
            f"## Dialogue so far\n{history_text}\n\n"
            f"## Candidate POIs\n{poi_cards}\n\n"
            f"## Task\n{action_instruction}\n"
            f"When quoting reviews, use exact text from the reviews above. "
            f"Reference reviews as business_id::R{{n}}."
        )

        try:
            with dspy.context(lm=self._lm, adapter=dspy.JSONAdapter()):
                result = self._predictor(prompt=prompt)
            output: Optional[_LLMTurnOutput] = getattr(result, 'output', None)
            if output is None:
                # DSPy returned no parsed output. Treat as a partial success rather than
                # silently fall back to popularity: emit empty text + empty POI list so
                # the metric layer scores zero rather than counting a popularity hit.
                logger.warning(
                    "LLM Zero-Shot returned None output (likely DSPy parse failure); "
                    "recording empty prediction (BLEU/ROUGE/recall = 0)."
                )
                return BaselinePrediction(
                    response_text='',
                    recommended_poi_ids=[],
                    referenced_review_ids=[],
                )
            return BaselinePrediction(
                response_text=output.response_text,
                recommended_poi_ids=output.recommended_poi_ids,
                referenced_review_ids=output.referenced_review_ids,
            )
        except Exception as e:
            logger.warning(f"LLM baseline failed, falling back to popularity: {e}")
            return self._fallback.predict_turn(
                dialogue_history, gold_action, candidate_pois, scenario,
            )


# ---------------------------------------------------------------------------
# Shared utilities for LLM-based baselines
# ---------------------------------------------------------------------------

def _format_history(turns: List[DialogueTurn]) -> str:
    """Format dialogue history as plain text."""
    if not turns:
        return "(No prior turns)"
    lines = []
    for t in turns:
        role = "User" if t.role == 'user' else "System"
        lines.append(f"{role}: {t.text}")
    return '\n'.join(lines)


def _split_review_sentences(pois: List[YelpPOI]) -> List[Dict]:
    """Split all reviews into individual sentences with metadata.

    Returns list of dicts with keys: 'sentence', 'poi_name', 'business_id',
    'review_ref_id', 'stars'.
    """
    sentence_re = re.compile(r'(?<=[.!?])\s+')
    sentences = []
    for poi in pois:
        for rev_idx, rev in enumerate(poi.reviews, 1):
            ref_id = f"{poi.business_id}::R{rev_idx}"
            for sent in sentence_re.split(rev.text.strip()):
                sent = sent.strip()
                if len(sent) >= 15:  # skip very short fragments
                    sentences.append({
                        'sentence': sent,
                        'poi_name': poi.name,
                        'business_id': poi.business_id,
                        'review_ref_id': ref_id,
                        'stars': rev.stars,
                    })
    return sentences


# ---------------------------------------------------------------------------
# 4. DSTBaseline — Schema-Guided Dialogue State Tracking
# ---------------------------------------------------------------------------

class _DSTStateUpdate(dspy.Signature):
    """Extract structured dialogue state changes from the latest user turn."""
    dialogue_context: str = dspy.InputField(
        desc="Full dialogue history so far"
    )
    current_state: str = dspy.InputField(
        desc="Current dialogue state as JSON"
    )
    state_update: TourismDialogueState = dspy.OutputField(
        desc="Incremental state changes extracted from the latest user turn"
    )


class DSTBaseline:
    """Schema-guided DST baseline with constraint-filtered TF-IDF ranking.

    Each turn: (1) DSPy extracts structured state delta from latest user turn,
    (2) merges into cumulative state, (3) state constraints filter candidates,
    (4) TF-IDF ranks remaining, (5) LLM generates grounded response.

    This baseline demonstrates the value of explicit preference tracking.
    """
    name = "dst"

    def __init__(self, model: str = "gemini/gemini-2.5-flash",
                 api_key: Optional[str] = None, api_base: Optional[str] = None):
        self.model = model
        self._lm = _build_lm(model, max_tokens=600, api_key=api_key, api_base=api_base)
        self._dst_extractor = dspy.Predict(_DSTStateUpdate)
        self._response_gen = dspy.Predict(_PredictTurnSignature)
        self._tfidf = TFIDFBaseline()
        self._fallback = PopularityBaseline()
        # Cumulative state, reset per dialogue
        self._state = TourismDialogueState()
        self._last_dialogue_id: str = ""

    def reset_state(self):
        """Reset dialogue state for a new dialogue."""
        self._state = TourismDialogueState()

    def _extract_state_update(
        self,
        dialogue_history: List[DialogueTurn],
    ) -> TourismDialogueState:
        """Use DSPy to extract state changes from dialogue history."""
        history_text = _format_history(dialogue_history)
        current_state_json = self._state.model_dump_json(indent=2)

        try:
            with dspy.context(lm=self._lm, adapter=dspy.JSONAdapter()):
                result = self._dst_extractor(
                    dialogue_context=history_text,
                    current_state=current_state_json,
                )
            delta: TourismDialogueState = result.state_update
            return delta
        except Exception as e:
            logger.warning(f"DST extraction failed: {e}")
            return TourismDialogueState()

    def _filter_by_state(
        self,
        pois: List[YelpPOI],
        state: TourismDialogueState,
    ) -> List[YelpPOI]:
        """Filter candidate POIs based on accumulated dialogue state constraints."""
        filtered = list(pois)

        # Exclude rejected POIs
        if state.rejected_poi_ids:
            rejected = set(state.rejected_poi_ids)
            filtered = [p for p in filtered if p.business_id not in rejected]

        # Price range filter
        if state.price_range:
            n_dollars = len(state.price_range)
            filtered = [
                p for p in filtered
                if not p.price_range or len(p.price_range) <= n_dollars
            ]

        # Cuisine positive filter
        if state.cuisine_prefs:
            prefs_lower = {c.lower() for c in state.cuisine_prefs}
            filtered = [
                p for p in filtered
                if any(c.lower() in prefs_lower for c in p.categories)
                or not p.categories  # keep POIs without category info
            ]

        # Cuisine avoidance filter
        if state.cuisine_avoid:
            avoid_lower = {c.lower() for c in state.cuisine_avoid}
            filtered = [
                p for p in filtered
                if not any(c.lower() in avoid_lower for c in p.categories)
            ]

        # Noise level filter
        if state.noise_level:
            noise_order = {'quiet': 0, 'average': 1, 'loud': 2, 'very_loud': 3}
            target = noise_order.get(state.noise_level, 1)
            kept = []
            for p in filtered:
                poi_noise = str(p.attributes.get('NoiseLevel', '')).strip("'\"u")
                poi_level = noise_order.get(poi_noise, 1)
                # Allow same or quieter than target
                if poi_level <= target:
                    kept.append(p)
                elif 'NoiseLevel' not in p.attributes:
                    kept.append(p)  # keep unknowns
            filtered = kept

        # Required features filter (boolean attributes)
        if state.required_features:
            for feat in state.required_features:
                filtered = [
                    p for p in filtered
                    if str(p.attributes.get(feat, '')).lower() in ('true', "'true'")
                    or feat not in p.attributes  # keep unknowns
                ]

        # If filtering removed everything, fall back to full list minus rejections
        if not filtered:
            rejected = set(state.rejected_poi_ids)
            filtered = [p for p in pois if p.business_id not in rejected]
        if not filtered:
            filtered = list(pois)  # last resort: include everything

        return filtered

    def predict_turn(
        self,
        dialogue_history: List[DialogueTurn],
        gold_action: str,
        candidate_pois: List[YelpPOI],
        scenario: ScenarioMeta,
    ) -> BaselinePrediction:
        # Step 1: Extract state update from latest user turns
        if dialogue_history:
            delta = self._extract_state_update(dialogue_history)
            if delta is not None:
                self._state = self._state.merge_update(delta)

            # Track rejected POIs from reject_and_refine actions via merge_update
            extra_rejected = []
            for t in dialogue_history:
                if t.action == 'reject_and_refine':
                    extra_rejected.extend(t.referenced_poi_ids)
            if extra_rejected:
                reject_delta = TourismDialogueState(rejected_poi_ids=extra_rejected)
                self._state = self._state.merge_update(reject_delta)

        # Step 2: Filter candidates by state constraints
        filtered_pois = self._filter_by_state(candidate_pois, self._state)

        # Step 3: TF-IDF rank filtered candidates
        self._tfidf.build_index(filtered_pois)
        user_texts = [t.text for t in dialogue_history if t.role == 'user']
        query = ' '.join(user_texts) if user_texts else scenario.persona_description
        ranked = self._tfidf._rank_pois(query, filtered_pois)

        # Step 4: LLM generates response with state context
        state_summary = self._format_state_summary()
        history_text = _format_history(dialogue_history)
        poi_cards = format_poi_cards(ranked[:6])  # top-6 after filtering

        action_instruction = {
            'recommend': "Recommend the most suitable POI(s), quoting relevant reviews verbatim.",
            'compare': "Compare two or more POIs, quoting relevant reviews verbatim.",
            'ask_preference': "Ask about preferences to narrow recommendations.",
            'farewell': "Wrap up the conversation warmly.",
        }.get(gold_action, f"Respond appropriately (action: {gold_action}).")

        prompt = (
            f"You are a tourism recommendation assistant.\n\n"
            f"## User Preferences (accumulated)\n{state_summary}\n\n"
            f"## Dialogue so far\n{history_text}\n\n"
            f"## Candidate POIs (filtered by preferences)\n{poi_cards}\n\n"
            f"## Task\n{action_instruction}\n"
            f"Quote reviews verbatim using exact text. "
            f"Reference reviews as business_id::R{{n}}."
        )

        try:
            with dspy.context(lm=self._lm, adapter=dspy.JSONAdapter()):
                result = self._response_gen(prompt=prompt)
            output: _LLMTurnOutput = result.output
            return BaselinePrediction(
                response_text=output.response_text,
                recommended_poi_ids=output.recommended_poi_ids,
                referenced_review_ids=output.referenced_review_ids,
            )
        except Exception as e:
            logger.warning(f"DST baseline LLM failed, falling back: {e}")
            return self._fallback.predict_turn(
                dialogue_history, gold_action, candidate_pois, scenario,
            )

    def _format_state_summary(self) -> str:
        """Format the current dialogue state as a readable summary."""
        s = self._state
        parts = []
        if s.price_range:
            parts.append(f"Budget: {s.price_range}")
        if s.cuisine_prefs:
            parts.append(f"Preferred cuisines: {', '.join(s.cuisine_prefs)}")
        if s.cuisine_avoid:
            parts.append(f"Avoid cuisines: {', '.join(s.cuisine_avoid)}")
        if s.ambiance_prefs:
            parts.append(f"Ambiance: {', '.join(s.ambiance_prefs)}")
        if s.noise_level:
            parts.append(f"Noise level: {s.noise_level}")
        if s.required_features:
            parts.append(f"Required: {', '.join(s.required_features)}")
        if s.location_constraint:
            parts.append(f"Location: {s.location_constraint}")
        if s.rejected_poi_ids:
            parts.append(f"Rejected POIs: {', '.join(s.rejected_poi_ids)}")
        if s.accepted_poi_ids:
            parts.append(f"Accepted POIs: {', '.join(s.accepted_poi_ids)}")
        if s.other_preferences:
            parts.append(f"Other: {', '.join(s.other_preferences)}")
        return '\n'.join(parts) if parts else "(No preferences stated yet)"


# ---------------------------------------------------------------------------
# 5. RAGCitationBaseline — RAG + Citation-Constrained Generation
# ---------------------------------------------------------------------------

class _RAGTurnOutput(BaseModel):
    """Structured output for RAG citation baseline."""
    response_text: str = Field(
        description=(
            "The system's response. MUST include at least one verbatim quote "
            "(10+ words) from the provided review sentences, enclosed in double quotes."
        )
    )
    recommended_poi_ids: List[str] = Field(
        default_factory=list,
        description="POI business_ids recommended in this response",
    )
    referenced_review_ids: List[str] = Field(
        default_factory=list,
        description="Review reference IDs cited (format: business_id::R{n})",
    )


class _RAGSignature(dspy.Signature):
    """Generate a grounded CRS response using retrieved review evidence."""
    prompt: str = dspy.InputField(desc="Dialogue context with retrieved review evidence")
    output: _RAGTurnOutput = dspy.OutputField(desc="Grounded response with verbatim citations")


class RAGCitationBaseline:
    """RAG baseline with citation-constrained generation.

    Two-stage pipeline:
    1. Retrieve: TF-IDF sentence retrieval finds top-k relevant review sentences.
    2. Generate: LLM generates response constrained to include verbatim citations.
       Uses dspy.Assert to enforce that the response contains at least one
       verbatim quote of >=10 tokens from the retrieved evidence.

    This baseline demonstrates the value of retrieval augmentation and
    citation enforcement vs zero-shot LLM.
    """
    name = "rag_citation"

    def __init__(
        self,
        model: str = "gemini/gemini-2.5-flash",
        top_k_sentences: int = 10,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        self.model = model
        self.top_k = top_k_sentences
        self._lm = _build_lm(model, max_tokens=600, api_key=api_key, api_base=api_base)
        self._predictor = dspy.Predict(_RAGSignature)
        self._fallback = PopularityBaseline()
        # Reuse TF-IDF machinery for sentence retrieval
        self._tfidf = TFIDFBaseline()

    def _retrieve_sentences(
        self,
        query: str,
        candidate_pois: List[YelpPOI],
        top_k: int = None,
    ) -> List[Dict]:
        """Retrieve top-k relevant review sentences using TF-IDF.

        Returns list of sentence dicts sorted by relevance.
        """
        top_k = top_k or self.top_k
        all_sentences = _split_review_sentences(candidate_pois)
        if not all_sentences:
            return []

        # Build mini TF-IDF index over sentences
        query_tokens = set(_tfidf_tokenize(query))
        if not query_tokens:
            return all_sentences[:top_k]

        # Score each sentence by token overlap (fast, no full index needed)
        scored = []
        for s in all_sentences:
            sent_tokens = set(_tfidf_tokenize(s['sentence']))
            if not sent_tokens:
                continue
            overlap = len(query_tokens & sent_tokens)
            jaccard = overlap / len(query_tokens | sent_tokens)
            scored.append((s, jaccard))

        scored.sort(key=lambda x: -x[1])
        return [s for s, _ in scored[:top_k]]

    def _check_citation(self, response_text: str, evidence_sentences: List[str]) -> bool:
        """Check if response contains at least one verbatim quote >=10 tokens."""
        quote_pattern = re.compile(r'"([^"]{10,})"')
        quotes = quote_pattern.findall(response_text)

        if not quotes:
            return False

        for quote in quotes:
            quote_lower = quote.lower()
            for sent in evidence_sentences:
                if quote_lower in sent.lower():
                    return True
        return False

    def predict_turn(
        self,
        dialogue_history: List[DialogueTurn],
        gold_action: str,
        candidate_pois: List[YelpPOI],
        scenario: ScenarioMeta,
    ) -> BaselinePrediction:
        # Open-set: prefilter large candidate sets via TF-IDF retrieval
        filtered_pois = prefilter_candidates(candidate_pois, dialogue_history)

        # Build query from all user turns
        user_texts = [t.text for t in dialogue_history if t.role == 'user']
        query = ' '.join(user_texts) if user_texts else scenario.persona_description

        # Stage 1: Retrieve relevant review sentences
        retrieved = self._retrieve_sentences(query, filtered_pois)
        evidence_text = self._format_evidence(retrieved)
        evidence_sentences = [s['sentence'] for s in retrieved]

        # Rank POIs by frequency in retrieved sentences
        poi_freq: Counter = Counter()
        for s in retrieved:
            poi_freq[s['business_id']] += 1
        poi_lookup = {p.business_id: p for p in filtered_pois}
        top_poi_ids = [pid for pid, _ in poi_freq.most_common(3)]

        # Build prompt
        history_text = _format_history(dialogue_history)

        action_instruction = {
            'recommend': (
                "Recommend the most suitable POI from the evidence below. "
                "You MUST include at least one EXACT verbatim quote (10+ words) "
                "from the review evidence, enclosed in double quotes."
            ),
            'compare': (
                "Compare two POIs using the evidence below. "
                "You MUST include verbatim quotes from the reviews for each POI."
            ),
            'ask_preference': "Ask about preferences to narrow recommendations.",
            'farewell': "Wrap up the conversation warmly.",
        }.get(gold_action, f"Respond appropriately (action: {gold_action}).")

        base_prompt = (
            f"You are a tourism recommendation assistant.\n\n"
            f"## Dialogue so far\n{history_text}\n\n"
            f"## Retrieved Review Evidence\n{evidence_text}\n\n"
            f"## Task\n{action_instruction}\n"
            f"CRITICAL: When recommending, you MUST quote EXACT text from the reviews "
            f"above (10+ words, in double quotes). Do NOT paraphrase.\n"
            f"Reference reviews as business_id::R{{n}}."
        )

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                retry_prompt = base_prompt
                if attempt > 0:
                    retry_prompt += (
                        "\n\nYour previous response did NOT contain a verbatim quote. "
                        "Try again, copying EXACT text from the reviews."
                    )
                with dspy.context(lm=self._lm, adapter=dspy.JSONAdapter()):
                    result = self._predictor(prompt=retry_prompt)
                output: _RAGTurnOutput = result.output

                # Citation check: if recommend/compare, verify citation exists
                if gold_action in ('recommend', 'compare'):
                    has_citation = self._check_citation(
                        output.response_text, evidence_sentences,
                    )
                    if not has_citation and attempt < max_retries:
                        logger.debug(
                            f"RAG citation check failed (attempt {attempt + 1}), retrying"
                        )
                        continue

                return BaselinePrediction(
                    response_text=output.response_text,
                    recommended_poi_ids=output.recommended_poi_ids or top_poi_ids[:1],
                    referenced_review_ids=output.referenced_review_ids,
                )

            except Exception as e:
                logger.warning(f"RAG baseline failed (attempt {attempt + 1}): {e}")

        # Final fallback
        return self._fallback.predict_turn(
            dialogue_history, gold_action, candidate_pois, scenario,
        )

    @staticmethod
    def _format_evidence(retrieved: List[Dict]) -> str:
        """Format retrieved sentences as evidence block."""
        if not retrieved:
            return "(No relevant review sentences found)"
        lines = []
        for i, s in enumerate(retrieved, 1):
            lines.append(
                f"[{i}] {s['poi_name']} ({s['review_ref_id']}, "
                f"{'*' * int(s['stars'])}): \"{s['sentence']}\""
            )
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 6. AspectRetrievalBaseline — Aspect-Based Review Decomposition
# ---------------------------------------------------------------------------

# Aspect categories with seed words for keyword-based extraction
_ASPECT_CATEGORIES = {
    'food': {
        'food', 'meal', 'dish', 'menu', 'taste', 'flavor', 'cook', 'chef',
        'ingredient', 'portion', 'appetizer', 'entree', 'dessert', 'breakfast',
        'lunch', 'dinner', 'brunch', 'steak', 'pasta', 'pizza', 'sushi',
        'seafood', 'salad', 'soup', 'sauce', 'spicy', 'fresh', 'delicious',
        'bland', 'overcooked', 'undercooked', 'flavorful', 'tasty',
    },
    'service': {
        'service', 'staff', 'waiter', 'waitress', 'server', 'host', 'hostess',
        'bartender', 'manager', 'attentive', 'friendly', 'rude', 'slow',
        'fast', 'helpful', 'polite', 'professional', 'prompt', 'responsive',
        'courteous', 'accommodating', 'negligent', 'inattentive',
    },
    'ambiance': {
        'ambiance', 'ambience', 'atmosphere', 'decor', 'vibe', 'setting',
        'music', 'noise', 'quiet', 'loud', 'cozy', 'romantic', 'intimate',
        'lively', 'elegant', 'casual', 'modern', 'rustic', 'charming',
        'beautiful', 'clean', 'dirty', 'crowded', 'spacious', 'outdoor',
    },
    'value': {
        'price', 'value', 'worth', 'expensive', 'cheap', 'affordable',
        'overpriced', 'reasonable', 'pricey', 'budget', 'cost', 'bill',
        'tip', 'dollar', 'money', 'bargain', 'deal', 'splurge',
    },
    'location': {
        'location', 'parking', 'walk', 'drive', 'downtown', 'neighborhood',
        'street', 'corner', 'block', 'nearby', 'close', 'far', 'convenient',
        'accessible', 'central', 'quarter', 'district', 'area',
    },
}

# Positive/negative sentiment words
_POSITIVE_WORDS = frozenset({
    'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'perfect',
    'incredible', 'outstanding', 'superb', 'delicious', 'love', 'loved',
    'best', 'favorite', 'recommend', 'enjoy', 'enjoyed', 'impressive',
    'beautiful', 'fresh', 'tasty', 'friendly', 'attentive', 'cozy',
    'pleasant', 'nice', 'good', 'awesome', 'fabulous', 'stellar',
})
_NEGATIVE_WORDS = frozenset({
    'bad', 'terrible', 'awful', 'horrible', 'worst', 'disappointing',
    'mediocre', 'bland', 'rude', 'slow', 'cold', 'overpriced', 'dirty',
    'noisy', 'crowded', 'stale', 'overcooked', 'undercooked', 'soggy',
    'greasy', 'tasteless', 'unfriendly', 'inattentive', 'poor', 'avoid',
})


def _extract_aspect_scores(text: str) -> Dict[str, float]:
    """Extract aspect sentiment scores from review text.

    Returns dict mapping aspect category -> sentiment score in [-1, 1].
    Analyzes sentiment per-sentence to avoid cross-aspect contamination:
    only sentiment words in the same sentence as aspect keywords contribute.
    """
    # Split into sentences for windowed sentiment analysis
    sentences = re.split(r'[.!?]+', text)
    aspect_pos: Dict[str, int] = {}
    aspect_neg: Dict[str, int] = {}

    for sentence in sentences:
        tokens = set(_tfidf_tokenize(sentence))
        if not tokens:
            continue
        sent_pos = len(tokens & _POSITIVE_WORDS)
        sent_neg = len(tokens & _NEGATIVE_WORDS)

        for aspect, keywords in _ASPECT_CATEGORIES.items():
            if tokens & keywords:
                aspect_pos[aspect] = aspect_pos.get(aspect, 0) + sent_pos
                aspect_neg[aspect] = aspect_neg.get(aspect, 0) + sent_neg

    aspect_scores: Dict[str, float] = {}
    for aspect in aspect_pos.keys() | aspect_neg.keys():
        pos = aspect_pos.get(aspect, 0)
        neg = aspect_neg.get(aspect, 0)
        total = pos + neg
        aspect_scores[aspect] = (pos - neg) / total if total > 0 else 0.0

    return aspect_scores


def _build_poi_aspect_profile(poi: YelpPOI) -> Dict[str, float]:
    """Build aspect profile for a POI by aggregating across all reviews.

    Returns dict mapping aspect -> average sentiment score.
    Also incorporates structured attributes from Yelp.
    """
    aspect_accum: Dict[str, List[float]] = {}

    for rev in poi.reviews:
        scores = _extract_aspect_scores(rev.text)
        for asp, score in scores.items():
            aspect_accum.setdefault(asp, []).append(score)

    profile = {}
    for asp, scores in aspect_accum.items():
        profile[asp] = sum(scores) / len(scores)

    # Enrich with structured attributes
    attrs = poi.attributes
    noise = str(attrs.get('NoiseLevel', '')).strip("'\"u")
    if noise in ('quiet', 'average'):
        profile.setdefault('ambiance', 0.0)
        profile['ambiance'] = max(profile.get('ambiance', 0.0), 0.3)
    if str(attrs.get('OutdoorSeating', '')).lower() in ('true', "'true'"):
        profile.setdefault('ambiance', 0.0)
        profile['ambiance'] = max(profile.get('ambiance', 0.0), 0.2)

    return profile


def _extract_query_aspects(query_text: str) -> Dict[str, float]:
    """Extract aspect weights from user query.

    Returns dict mapping aspect -> importance weight (0-1).
    Aspects mentioned more get higher weight.
    """
    tokens = set(_tfidf_tokenize(query_text))
    weights: Dict[str, float] = {}

    for aspect, keywords in _ASPECT_CATEGORIES.items():
        hits = tokens & keywords
        if hits:
            weights[aspect] = min(len(hits) / 3.0, 1.0)  # cap at 1.0

    # If no aspects detected, equal weight on food + service
    if not weights:
        weights = {'food': 0.5, 'service': 0.3, 'ambiance': 0.2}

    # Normalize
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    return weights


class AspectRetrievalBaseline:
    """Aspect-based review retrieval baseline.

    Decomposes reviews into aspect-level sentiment scores, then matches
    POIs to user queries via aspect-weighted similarity.

    This replaces flat TF-IDF with structured aspect understanding:
    "food was amazing but service was terrible" now contributes positively
    to food-seekers and negatively to service-seekers.
    """
    name = "aspect"

    def __init__(self):
        self._poi_profiles: Dict[str, Dict[str, float]] = {}
        self._current_key: frozenset = frozenset()
        self._fallback = PopularityBaseline()

    def _build_profiles(self, pois: List[YelpPOI]) -> None:
        """Build aspect profiles for all candidate POIs (cached)."""
        cache_key = frozenset(p.business_id for p in pois)
        if cache_key == self._current_key:
            return
        self._poi_profiles = {
            poi.business_id: _build_poi_aspect_profile(poi) for poi in pois
        }
        self._current_key = cache_key

    def _rank_pois(
        self,
        query_text: str,
        candidate_pois: List[YelpPOI],
    ) -> List[YelpPOI]:
        """Rank POIs by aspect-weighted similarity to query."""
        self._build_profiles(candidate_pois)
        query_aspects = _extract_query_aspects(query_text)

        scored = []
        for poi in candidate_pois:
            profile = self._poi_profiles.get(poi.business_id, {})
            # Aspect-weighted dot product
            score = sum(
                query_aspects.get(asp, 0) * profile.get(asp, 0)
                for asp in set(query_aspects) | set(profile)
            )
            # Blend with popularity as tiebreaker
            pop_score = (poi.stars / 5.0) * 0.1
            scored.append((poi, score + pop_score))

        scored.sort(key=lambda x: -x[1])
        return [poi for poi, _ in scored]

    def predict_turn(
        self,
        dialogue_history: List[DialogueTurn],
        gold_action: str,
        candidate_pois: List[YelpPOI],
        scenario: ScenarioMeta,
    ) -> BaselinePrediction:
        self._build_profiles(candidate_pois)

        user_texts = [t.text for t in dialogue_history if t.role == 'user']
        query = ' '.join(user_texts) if user_texts else scenario.persona_description
        ranked = self._rank_pois(query, candidate_pois)

        if gold_action == 'recommend':
            poi = ranked[0]
            # Pick review that best matches dominant query aspect
            query_aspects = _extract_query_aspects(query)
            dominant_aspect = max(query_aspects, key=query_aspects.get) if query_aspects else 'food'
            review_text, review_id = self._pick_aspect_review(poi, dominant_aspect)
            text = (
                f"Based on your interest in {dominant_aspect}, I'd recommend {poi.name}. "
                f"It's rated {poi.stars}/5. "
                f'A reviewer noted: "{review_text}"'
            )
            return BaselinePrediction(
                response_text=text,
                recommended_poi_ids=[poi.business_id],
                referenced_review_ids=[review_id] if review_id else [],
            )

        if gold_action == 'compare':
            top2 = ranked[:2]
            parts = []
            poi_ids = []
            review_ids = []
            for poi in top2:
                review_text, review_id = self._pick_aspect_review(poi, 'food')
                parts.append(
                    f"{poi.name} ({poi.stars}/5, {poi.price_range or 'N/A'})"
                    f' — "{review_text}"'
                )
                poi_ids.append(poi.business_id)
                if review_id:
                    review_ids.append(review_id)
            if len(parts) < 2:
                text = f"Here's an option: {parts[0]}." if parts else "I don't have options to compare."
            else:
                text = f"Comparing options: {parts[0]}. Alternatively, {parts[1]}."
            return BaselinePrediction(
                response_text=text,
                recommended_poi_ids=poi_ids,
                referenced_review_ids=review_ids,
            )

        if gold_action == 'ask_preference':
            text = (
                f"What aspects matter most to you in a {scenario.primary_poi_type}? "
                f"For example, food quality, service, ambiance, value, or location?"
            )
            return BaselinePrediction(response_text=text)

        text = "Glad I could help! Have a wonderful time."
        return BaselinePrediction(response_text=text)

    @staticmethod
    def _pick_aspect_review(poi: YelpPOI, aspect: str) -> tuple:
        """Pick review most relevant to a given aspect."""
        if not poi.reviews:
            return ("Great place!", None)

        keywords = _ASPECT_CATEGORIES.get(aspect, set())
        best_rev = None
        best_overlap = -1
        best_idx = 0
        for rev_idx, rev in enumerate(poi.reviews):
            tokens = set(_tfidf_tokenize(rev.text))
            overlap = len(tokens & keywords)
            if overlap > best_overlap:
                best_overlap = overlap
                best_rev = rev
                best_idx = rev_idx

        if best_rev is None:
            best_rev = poi.reviews[0]
            best_idx = 0

        ref_id = f"{poi.business_id}::R{best_idx + 1}"
        return (best_rev.text[:150], ref_id)


# ---------------------------------------------------------------------------
# 7. DenseRetrievalBaseline — Sentence Transformer bi-encoder
# ---------------------------------------------------------------------------

class DenseRetrievalBaseline:
    """Dense retrieval baseline using Sentence Transformer embeddings.

    POI documents are encoded as dense vectors (name + categories + top-3 reviews).
    Query = all user turns concatenated. Rejected POIs are penalized by subtracting
    their embeddings from the query vector (negative preference mechanism).

    Requires: sentence-transformers package.
    """
    name = "dense"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._encoder = SentenceTransformer(model_name)
        self._poi_embeddings: Dict[str, 'numpy.ndarray'] = {}
        self._current_key: frozenset = frozenset()
        self._fallback = PopularityBaseline()

    def _build_index(self, pois: List[YelpPOI]) -> None:
        """Encode all POI documents into dense vectors (cached)."""
        import numpy as np
        cache_key = frozenset(p.business_id for p in pois)
        if cache_key == self._current_key:
            return

        docs = []
        ids = []
        for poi in pois:
            # POI document = name + categories + top-3 review sentences
            parts = [poi.name, ', '.join(poi.categories)]
            for rev in poi.reviews[:3]:
                parts.append(rev.text[:200])
            docs.append(' '.join(parts))
            ids.append(poi.business_id)

        embeddings = self._encoder.encode(docs, convert_to_numpy=True)
        # Normalize for cosine similarity via dot product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms

        self._poi_embeddings = dict(zip(ids, embeddings))
        self._current_key = cache_key

    def _rank_pois(
        self,
        query_text: str,
        candidate_pois: List[YelpPOI],
        rejected_ids: List[str] = None,
    ) -> List[YelpPOI]:
        """Rank POIs by dense cosine similarity with negative preference."""
        import numpy as np
        self._build_index(candidate_pois)

        # Encode query
        q_emb = self._encoder.encode([query_text], convert_to_numpy=True)[0]
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 0:
            q_emb = q_emb / q_norm

        # Negative preference: subtract rejected POI embeddings
        if rejected_ids:
            for rid in rejected_ids:
                if rid in self._poi_embeddings:
                    q_emb = q_emb - 0.5 * self._poi_embeddings[rid]
            q_norm = np.linalg.norm(q_emb)
            if q_norm > 0:
                q_emb = q_emb / q_norm

        rejected_set = set(rejected_ids) if rejected_ids else set()
        scored = []
        for poi in candidate_pois:
            emb = self._poi_embeddings.get(poi.business_id)
            if emb is None:
                scored.append((poi, 0.0))
                continue
            score = float(np.dot(q_emb, emb))
            # Hard penalty for explicitly rejected POIs
            if poi.business_id in rejected_set:
                score -= 1.0
            scored.append((poi, score))

        scored.sort(key=lambda x: -x[1])
        return [poi for poi, _ in scored]

    def predict_turn(
        self,
        dialogue_history: List[DialogueTurn],
        gold_action: str,
        candidate_pois: List[YelpPOI],
        scenario: ScenarioMeta,
    ) -> BaselinePrediction:
        self._build_index(candidate_pois)

        user_texts = [t.text for t in dialogue_history if t.role == 'user']
        query = ' '.join(user_texts) if user_texts else scenario.persona_description

        # Collect rejected POI IDs from dialogue history
        rejected_ids = []
        for t in dialogue_history:
            if t.action == 'reject_and_refine':
                rejected_ids.extend(t.referenced_poi_ids)

        ranked = self._rank_pois(query, candidate_pois, rejected_ids)

        if gold_action == 'recommend':
            poi = ranked[0]
            review_text, review_id = PopularityBaseline._pick_review(poi)
            text = (
                f"Based on semantic similarity to your preferences, "
                f"I'd recommend {poi.name} ({poi.stars}/5). "
                f'A reviewer said: "{review_text}"'
            )
            return BaselinePrediction(
                response_text=text,
                recommended_poi_ids=[poi.business_id],
                referenced_review_ids=[review_id] if review_id else [],
            )

        if gold_action == 'compare':
            top2 = ranked[:2]
            parts = []
            poi_ids = []
            review_ids = []
            for poi in top2:
                review_text, review_id = PopularityBaseline._pick_review(poi)
                parts.append(
                    f"{poi.name} ({poi.stars}/5, {poi.price_range or 'N/A'})"
                    f' — "{review_text}"'
                )
                poi_ids.append(poi.business_id)
                if review_id:
                    review_ids.append(review_id)
            if len(parts) < 2:
                text = f"Here's an option: {parts[0]}." if parts else "I don't have options to compare."
            else:
                text = f"Here are two matches: {parts[0]}. Or consider {parts[1]}."
            return BaselinePrediction(
                response_text=text,
                recommended_poi_ids=poi_ids,
                referenced_review_ids=review_ids,
            )

        if gold_action == 'ask_preference':
            text = (
                f"What kind of {scenario.primary_poi_type} experience are you looking for? "
                f"Describe the vibe, cuisine, or any requirements."
            )
            return BaselinePrediction(response_text=text)

        text = "Hope you enjoy your visit!"
        return BaselinePrediction(response_text=text)


# ---------------------------------------------------------------------------
# 8. SpatialBaseline — Location-Aware Recommendation
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute haversine distance between two lat/lon points in kilometers."""
    R = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


class SpatialBaseline:
    """Location-aware recommendation baseline.

    Combines TF-IDF relevance with spatial proximity scoring.
    When the dialogue state has a location_constraint, attempts to geocode
    it by finding the closest matching POI name/address, then re-ranks
    candidates by proximity.

    Also computes spatial coherence: are recommended POIs walkably close
    to each other (within 2km)?
    """
    name = "spatial"

    def __init__(self):
        self._tfidf = TFIDFBaseline()
        self._fallback = PopularityBaseline()
        self._anchor_lat: float = 0.0
        self._anchor_lon: float = 0.0
        self._has_anchor: bool = False

    def reset_state(self):
        """Reset spatial anchor for new dialogue."""
        self._anchor_lat = 0.0
        self._anchor_lon = 0.0
        self._has_anchor = False

    def _detect_spatial_anchor(
        self,
        dialogue_history: List[DialogueTurn],
        candidate_pois: List[YelpPOI],
    ) -> None:
        """Detect spatial anchor from dialogue history.

        Looks for location mentions in user turns and tries to match
        to a known POI or use city centroid.
        """
        if self._has_anchor:
            return

        location_keywords = {
            'near', 'close', 'nearby', 'around', 'walking', 'distance',
            'next to', 'by the', 'in the', 'downtown', 'quarter', 'district',
            'hotel', 'staying', 'located',
        }

        for turn in dialogue_history:
            if turn.role != 'user':
                continue
            text_lower = turn.text.lower()
            if any(kw in text_lower for kw in location_keywords):
                # Try to find a matching POI name in the text
                for poi in candidate_pois:
                    if poi.name.lower() in text_lower:
                        self._anchor_lat = poi.latitude
                        self._anchor_lon = poi.longitude
                        self._has_anchor = True
                        return

                # Try to match accepted/referenced POIs as anchor
                if turn.referenced_poi_ids:
                    for poi in candidate_pois:
                        if poi.business_id in turn.referenced_poi_ids:
                            self._anchor_lat = poi.latitude
                            self._anchor_lon = poi.longitude
                            self._has_anchor = True
                            return

        # Fallback: use centroid of all candidates as anchor
        if not self._has_anchor and candidate_pois:
            lats = [p.latitude for p in candidate_pois if p.latitude != 0]
            lons = [p.longitude for p in candidate_pois if p.longitude != 0]
            if lats and lons:
                self._anchor_lat = sum(lats) / len(lats)
                self._anchor_lon = sum(lons) / len(lons)
                # Don't set _has_anchor — centroid is weak signal

    def _rank_pois(
        self,
        query_text: str,
        candidate_pois: List[YelpPOI],
        spatial_weight: float = 0.3,
    ) -> List[YelpPOI]:
        """Rank POIs by blended TF-IDF relevance + spatial proximity score."""
        # TF-IDF ranking
        self._tfidf.build_index(candidate_pois)
        q_vec = self._tfidf._query_vec(query_text)
        tfidf_scores = {}
        for poi in candidate_pois:
            doc_vec = self._tfidf._doc_vecs.get(poi.business_id, _SparseVec())
            tfidf_scores[poi.business_id] = q_vec.cosine(doc_vec)

        # Spatial scoring
        spatial_scores = {}
        if self._has_anchor:
            max_dist = 0.0
            dists = {}
            for poi in candidate_pois:
                if poi.latitude == 0 and poi.longitude == 0:
                    dists[poi.business_id] = float('inf')
                    continue
                d = _haversine_km(
                    self._anchor_lat, self._anchor_lon,
                    poi.latitude, poi.longitude,
                )
                dists[poi.business_id] = d
                if d < float('inf'):
                    max_dist = max(max_dist, d)

            for bid, d in dists.items():
                if max_dist > 0 and d < float('inf'):
                    spatial_scores[bid] = 1.0 - (d / max_dist)
                else:
                    spatial_scores[bid] = 0.5  # unknown
        else:
            spatial_weight = 0.0  # no spatial signal available

        # Blend scores
        scored = []
        for poi in candidate_pois:
            tf_score = tfidf_scores.get(poi.business_id, 0.0)
            sp_score = spatial_scores.get(poi.business_id, 0.5)
            blended = (1 - spatial_weight) * tf_score + spatial_weight * sp_score
            scored.append((poi, blended))

        scored.sort(key=lambda x: -x[1])
        return [poi for poi, _ in scored]

    def predict_turn(
        self,
        dialogue_history: List[DialogueTurn],
        gold_action: str,
        candidate_pois: List[YelpPOI],
        scenario: ScenarioMeta,
    ) -> BaselinePrediction:
        self._detect_spatial_anchor(dialogue_history, candidate_pois)

        user_texts = [t.text for t in dialogue_history if t.role == 'user']
        query = ' '.join(user_texts) if user_texts else scenario.persona_description
        ranked = self._rank_pois(query, candidate_pois)

        if gold_action == 'recommend':
            poi = ranked[0]
            review_text, review_id = PopularityBaseline._pick_review(poi)
            dist_info = ""
            if self._has_anchor and poi.latitude != 0 and poi.longitude != 0:
                d = _haversine_km(
                    self._anchor_lat, self._anchor_lon,
                    poi.latitude, poi.longitude,
                )
                if d < 1.0:
                    dist_info = f" It's about {d*1000:.0f}m away."
                else:
                    dist_info = f" It's about {d:.1f}km away."
            text = (
                f"I'd recommend {poi.name} ({poi.stars}/5).{dist_info} "
                f'A reviewer said: "{review_text}"'
            )
            return BaselinePrediction(
                response_text=text,
                recommended_poi_ids=[poi.business_id],
                referenced_review_ids=[review_id] if review_id else [],
            )

        if gold_action == 'compare':
            top2 = ranked[:2]
            parts = []
            poi_ids = []
            review_ids = []
            for poi in top2:
                review_text, review_id = PopularityBaseline._pick_review(poi)
                parts.append(
                    f"{poi.name} ({poi.stars}/5, {poi.price_range or 'N/A'})"
                    f' — "{review_text}"'
                )
                poi_ids.append(poi.business_id)
                if review_id:
                    review_ids.append(review_id)
            if len(parts) < 2:
                text = f"Here's an option: {parts[0]}." if parts else "I don't have options to compare."
            else:
                text = f"Nearby options: {parts[0]}. Or try {parts[1]}."
            return BaselinePrediction(
                response_text=text,
                recommended_poi_ids=poi_ids,
                referenced_review_ids=review_ids,
            )

        if gold_action == 'ask_preference':
            text = (
                f"Are you looking for a {scenario.primary_poi_type} near a specific area? "
                f"Knowing your location helps me find closer options."
            )
            return BaselinePrediction(response_text=text)

        text = "Hope you enjoy your visit!"
        return BaselinePrediction(response_text=text)


# ---------------------------------------------------------------------------
# 9. HybridRRFBaseline — Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _rrf_score(ranks: List[int], k: int = 60) -> float:
    """Compute RRF score: sum of 1/(k + rank_i) across all ranking sources."""
    return sum(1.0 / (k + r) for r in ranks)


class HybridRRFBaseline:
    """Hybrid retrieval via Reciprocal Rank Fusion (RRF).

    Fuses three independent ranking signals:
    1. Sparse (TF-IDF) — good at exact name/term matches
    2. Dense (Sentence Transformer) — good at semantic similarity
    3. Structured attributes — good at hard constraints (price, features)

    RRF formula: score(poi) = Sigma_i 1/(k + rank_i), k=60

    If dense retrieval is unavailable (no sentence-transformers), falls back
    to TF-IDF + Aspect + Structured (three-way without dense).
    """
    name = "hybrid_rrf"

    def __init__(self, dense_model: str = "all-MiniLM-L6-v2"):
        self._tfidf = TFIDFBaseline()
        self._aspect = AspectRetrievalBaseline()
        self._dense = None
        try:
            self._dense = DenseRetrievalBaseline(model_name=dense_model)
        except ImportError:
            logger.warning("sentence-transformers not available, hybrid RRF will skip dense signal")
        self._fallback = PopularityBaseline()

    def _rank_pois(
        self,
        query_text: str,
        candidate_pois: List[YelpPOI],
        rejected_ids: List[str] = None,
    ) -> List[YelpPOI]:
        """Rank POIs via RRF fusion of multiple retrieval signals."""
        poi_ids = [p.business_id for p in candidate_pois]
        poi_lookup = {p.business_id: p for p in candidate_pois}

        # Signal 1: TF-IDF
        self._tfidf.build_index(candidate_pois)
        tfidf_ranked = self._tfidf._rank_pois(query_text, candidate_pois)
        tfidf_rank = {p.business_id: i for i, p in enumerate(tfidf_ranked)}

        # Signal 2: Aspect-based
        self._aspect._build_profiles(candidate_pois)
        aspect_ranked = self._aspect._rank_pois(query_text, candidate_pois)
        aspect_rank = {p.business_id: i for i, p in enumerate(aspect_ranked)}

        # Signal 3: Dense (if available)
        dense_rank = {}
        if self._dense is not None:
            dense_ranked = self._dense._rank_pois(
                query_text, candidate_pois, rejected_ids,
            )
            dense_rank = {p.business_id: i for i, p in enumerate(dense_ranked)}

        # Signal 4: Structured attribute scoring (stars + price match)
        rejected_set = set(rejected_ids) if rejected_ids else set()
        struct_scored = []
        for poi in candidate_pois:
            score = poi.stars / 5.0  # normalized star rating
            # Penalize rejected
            if rejected_set and poi.business_id in rejected_set:
                score -= 0.5
            struct_scored.append((poi.business_id, score))
        struct_scored.sort(key=lambda x: -x[1])
        struct_rank = {bid: i for i, (bid, _) in enumerate(struct_scored)}

        # RRF fusion
        rrf_scores = {}
        for bid in poi_ids:
            ranks = [
                tfidf_rank.get(bid, len(candidate_pois)),
                aspect_rank.get(bid, len(candidate_pois)),
                struct_rank.get(bid, len(candidate_pois)),
            ]
            if dense_rank:
                ranks.append(dense_rank.get(bid, len(candidate_pois)))
            rrf_scores[bid] = _rrf_score(ranks)

        # Sort by RRF score (higher is better)
        sorted_ids = sorted(rrf_scores, key=lambda bid: -rrf_scores[bid])
        return [poi_lookup[bid] for bid in sorted_ids]

    def predict_turn(
        self,
        dialogue_history: List[DialogueTurn],
        gold_action: str,
        candidate_pois: List[YelpPOI],
        scenario: ScenarioMeta,
    ) -> BaselinePrediction:
        user_texts = [t.text for t in dialogue_history if t.role == 'user']
        query = ' '.join(user_texts) if user_texts else scenario.persona_description

        rejected_ids = []
        for t in dialogue_history:
            if t.action == 'reject_and_refine':
                rejected_ids.extend(t.referenced_poi_ids)

        ranked = self._rank_pois(query, candidate_pois, rejected_ids)

        if gold_action == 'recommend':
            poi = ranked[0]
            review_text, review_id = PopularityBaseline._pick_review(poi)
            text = (
                f"After considering multiple factors, I'd recommend {poi.name} "
                f"({poi.stars}/5). "
                f'A reviewer said: "{review_text}"'
            )
            return BaselinePrediction(
                response_text=text,
                recommended_poi_ids=[poi.business_id],
                referenced_review_ids=[review_id] if review_id else [],
            )

        if gold_action == 'compare':
            top2 = ranked[:2]
            parts = []
            poi_ids = []
            review_ids = []
            for poi in top2:
                review_text, review_id = PopularityBaseline._pick_review(poi)
                parts.append(
                    f"{poi.name} ({poi.stars}/5, {poi.price_range or 'N/A'})"
                    f' — "{review_text}"'
                )
                poi_ids.append(poi.business_id)
                if review_id:
                    review_ids.append(review_id)
            if len(parts) < 2:
                text = f"Here's an option: {parts[0]}." if parts else "I don't have options to compare."
            else:
                text = f"Two strong options: {parts[0]}. Or consider {parts[1]}."
            return BaselinePrediction(
                response_text=text,
                recommended_poi_ids=poi_ids,
                referenced_review_ids=review_ids,
            )

        if gold_action == 'ask_preference':
            text = (
                f"Tell me more about what matters most — cuisine type, budget, "
                f"atmosphere, or location?"
            )
            return BaselinePrediction(response_text=text)

        text = "Happy to help! Enjoy your trip."
        return BaselinePrediction(response_text=text)


# ---------------------------------------------------------------------------
# 10. MultiReviewSynthesisBaseline — Multi-Review Aspect Synthesis
# ---------------------------------------------------------------------------

def _cluster_reviews_by_aspect(
    poi: YelpPOI,
) -> Dict[str, List[Dict]]:
    """Cluster a POI's reviews by aspect, with per-review sentiment.

    Returns dict mapping aspect -> list of {review_idx, ref_id, sentiment, sentence}.
    Each review can appear under multiple aspects.
    """
    sentence_re = re.compile(r'(?<=[.!?])\s+')
    clusters: Dict[str, List[Dict]] = {asp: [] for asp in _ASPECT_CATEGORIES}

    for rev_idx, rev in enumerate(poi.reviews, 1):
        ref_id = f"{poi.business_id}::R{rev_idx}"
        for sent in sentence_re.split(rev.text.strip()):
            sent = sent.strip()
            if len(sent) < 15:
                continue
            tokens = set(_tfidf_tokenize(sent))
            pos_hits = len(tokens & _POSITIVE_WORDS)
            neg_hits = len(tokens & _NEGATIVE_WORDS)
            sentiment = 'positive' if pos_hits > neg_hits else (
                'negative' if neg_hits > pos_hits else 'neutral'
            )
            for asp, keywords in _ASPECT_CATEGORIES.items():
                if tokens & keywords:
                    clusters[asp].append({
                        'review_idx': rev_idx,
                        'ref_id': ref_id,
                        'sentiment': sentiment,
                        'sentence': sent,
                        'stars': rev.stars,
                    })

    # Remove empty aspects
    return {asp: items for asp, items in clusters.items() if items}


def _compute_aspect_consensus(
    clusters: Dict[str, List[Dict]],
) -> Dict[str, Dict]:
    """Compute consensus scores per aspect.

    Returns dict mapping aspect -> {positive, negative, neutral, total, consensus}.
    consensus = (n_pos - n_neg) / n_total, in [-1, 1].
    """
    consensus = {}
    for asp, items in clusters.items():
        n_pos = sum(1 for it in items if it['sentiment'] == 'positive')
        n_neg = sum(1 for it in items if it['sentiment'] == 'negative')
        n_neu = sum(1 for it in items if it['sentiment'] == 'neutral')
        total = len(items)
        consensus[asp] = {
            'positive': n_pos,
            'negative': n_neg,
            'neutral': n_neu,
            'total': total,
            'consensus': (n_pos - n_neg) / total if total > 0 else 0.0,
        }
    return consensus


def _format_aspect_evidence(
    poi: YelpPOI,
    clusters: Dict[str, List[Dict]],
    consensus: Dict[str, Dict],
) -> str:
    """Format aspect-clustered review evidence as a structured prompt section."""
    lines = [f"=== {poi.name} (ID: {poi.business_id}) ==="]
    lines.append(f"Rating: {poi.stars}/5 | Price: {poi.price_range or 'N/A'}")
    lines.append("")

    for asp in ('food', 'service', 'ambiance', 'value', 'location'):
        if asp not in clusters:
            continue
        con = consensus[asp]
        label = 'positive' if con['consensus'] > 0.2 else (
            'negative' if con['consensus'] < -0.2 else 'mixed'
        )
        lines.append(
            f"### {asp.upper()} ({label}, "
            f"+{con['positive']}/-{con['negative']}/{con['neutral']}n)"
        )
        # Show up to 4 representative sentences per aspect
        seen_refs = set()
        shown = 0
        for item in clusters[asp]:
            if shown >= 4:
                break
            if item['ref_id'] in seen_refs:
                continue
            seen_refs.add(item['ref_id'])
            lines.append(
                f"  [{item['ref_id']}] ({item['sentiment']}): "
                f"\"{item['sentence']}\""
            )
            shown += 1
        lines.append("")

    return '\n'.join(lines)


class _SynthesisOutput(BaseModel):
    """Structured output for multi-review synthesis."""
    response_text: str = Field(
        description=(
            "Synthesized response citing multiple reviews. For each aspect discussed, "
            "cite the review references in brackets, e.g. 'pasta is consistently praised "
            "([R2], [R5])'. Include both positive and negative aspects when applicable."
        )
    )
    recommended_poi_ids: List[str] = Field(
        default_factory=list,
        description="POI business_ids recommended",
    )
    referenced_review_ids: List[str] = Field(
        default_factory=list,
        description="All review ref IDs cited (format: business_id::R{n})",
    )


class _SynthesisSignature(dspy.Signature):
    """Generate a multi-review synthesis recommendation."""
    prompt: str = dspy.InputField(desc="Dialogue context with aspect-clustered evidence")
    output: _SynthesisOutput = dspy.OutputField(desc="Synthesis response with multi-review citations")


class MultiReviewSynthesisBaseline:
    """Multi-review synthesis baseline with provenance tracking.

    For each top POI:
    1. Clusters reviews by aspect (food/service/ambiance/value/location)
    2. Computes aspect consensus scores
    3. Presents structured aspect evidence to LLM
    4. LLM generates synthesis citing multiple reviews per aspect claim

    Example output: "The pasta is consistently praised ([R2], [R5], [R11]),
    though 2 reviewers noted slow weekend service ([R3], [R8])."
    """
    name = "multi_review_synthesis"

    def __init__(self, model: str = "gemini/gemini-2.5-flash",
                 api_key: Optional[str] = None, api_base: Optional[str] = None):
        self.model = model
        self._lm = _build_lm(model, max_tokens=600, api_key=api_key, api_base=api_base)
        self._predictor = dspy.Predict(_SynthesisSignature)
        self._tfidf = TFIDFBaseline()
        self._fallback = PopularityBaseline()

    def predict_turn(
        self,
        dialogue_history: List[DialogueTurn],
        gold_action: str,
        candidate_pois: List[YelpPOI],
        scenario: ScenarioMeta,
    ) -> BaselinePrediction:
        user_texts = [t.text for t in dialogue_history if t.role == 'user']
        query = ' '.join(user_texts) if user_texts else scenario.persona_description

        # Rank by TF-IDF first
        self._tfidf.build_index(candidate_pois)
        ranked = self._tfidf._rank_pois(query, candidate_pois)

        if gold_action in ('recommend', 'compare'):
            n_pois = 1 if gold_action == 'recommend' else 2
            top_pois = ranked[:n_pois]

            # Build aspect-clustered evidence for each top POI
            evidence_blocks = []
            all_ref_ids = []
            for poi in top_pois:
                clusters = _cluster_reviews_by_aspect(poi)
                consensus = _compute_aspect_consensus(clusters)
                evidence = _format_aspect_evidence(poi, clusters, consensus)
                evidence_blocks.append(evidence)
                # Collect all referenced review IDs
                for items in clusters.values():
                    for item in items:
                        if item['ref_id'] not in all_ref_ids:
                            all_ref_ids.append(item['ref_id'])

            evidence_text = '\n\n'.join(evidence_blocks)
            history_text = _format_history(dialogue_history)

            action_instruction = {
                'recommend': (
                    "Recommend the best POI by SYNTHESIZING multiple reviews. "
                    "For each aspect (food, service, ambiance, etc.), summarize the "
                    "consensus and cite the specific reviews that support your claim "
                    "using [Rn] notation. Include both positives and negatives."
                ),
                'compare': (
                    "Compare the two POIs by SYNTHESIZING multiple reviews for each. "
                    "For each aspect, state the consensus and cite supporting reviews "
                    "using [Rn] notation."
                ),
            }[gold_action]

            prompt = (
                f"You are a tourism recommendation assistant.\n\n"
                f"## Dialogue so far\n{history_text}\n\n"
                f"## Aspect-Clustered Review Evidence\n{evidence_text}\n\n"
                f"## Task\n{action_instruction}\n"
                f"CRITICAL: Cite specific reviews using the reference IDs "
                f"(e.g., business_id::R1). Synthesize across multiple reviews "
                f"per aspect — do NOT just quote one review."
            )

            try:
                with dspy.context(lm=self._lm, adapter=dspy.JSONAdapter()):
                    result = self._predictor(prompt=prompt)
                output: _SynthesisOutput = result.output
                return BaselinePrediction(
                    response_text=output.response_text,
                    recommended_poi_ids=output.recommended_poi_ids
                    or [p.business_id for p in top_pois],
                    referenced_review_ids=output.referenced_review_ids or all_ref_ids[:10],
                )
            except Exception as e:
                logger.warning(f"Multi-review synthesis failed: {e}")
                return self._fallback.predict_turn(
                    dialogue_history, gold_action, candidate_pois, scenario,
                )

        if gold_action == 'ask_preference':
            text = (
                f"What aspects of a {scenario.primary_poi_type} matter most to you? "
                f"For example, food quality, service, ambiance, value, or location?"
            )
            return BaselinePrediction(response_text=text)

        text = "Glad I could help with a comprehensive overview! Enjoy your visit."
        return BaselinePrediction(response_text=text)


# ---------------------------------------------------------------------------
# 11. MemoryAugmentedBaseline — Persistent Cross-Session Preference Memory
# ---------------------------------------------------------------------------

class PreferenceMemory:
    """Lightweight persistent preference memory with semantic retrieval.

    Stores preference signals as embeddings keyed by user/persona ID.
    Supports semantic search with exponential time decay — older preferences
    contribute less to retrieval scores.

    Uses sentence-transformers for embedding (same model as DenseRetrievalBaseline).
    """

    def __init__(self, encoder_model: str = "all-MiniLM-L6-v2", encoder=None):
        import numpy as np
        if encoder is not None:
            self._encoder = encoder
        else:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(encoder_model)
        self._np = np
        # {user_id: [(text, embedding, timestamp)]}
        self._store: Dict[str, List[tuple]] = {}

    def add(
        self,
        user_id: str,
        preference_text: str,
        timestamp: float = None,
    ) -> None:
        """Store a preference signal for a user."""
        emb = self._encoder.encode([preference_text], convert_to_numpy=True)[0]
        norm = self._np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        ts = timestamp or time.time()
        self._store.setdefault(user_id, []).append((preference_text, emb, ts))

    def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        decay_hours: float = 168.0,  # 1 week half-life
    ) -> List[tuple]:
        """Retrieve relevant past preferences with time decay.

        Args:
            user_id: User/persona identifier.
            query: Current context to match against.
            top_k: Max results to return.
            decay_hours: Time decay half-life in hours.

        Returns:
            List of (preference_text, score) tuples, sorted by relevance.
        """
        if user_id not in self._store or not self._store[user_id]:
            return []

        q_emb = self._encoder.encode([query], convert_to_numpy=True)[0]
        q_norm = self._np.linalg.norm(q_emb)
        if q_norm > 0:
            q_emb = q_emb / q_norm

        now = time.time()
        scored = []
        for text, emb, ts in self._store[user_id]:
            sim = float(self._np.dot(q_emb, emb))
            hours_ago = max((now - ts) / 3600.0, 0)
            decay = math.exp(-0.693 * hours_ago / decay_hours)  # exponential decay
            scored.append((text, sim * decay))

        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def get_all(self, user_id: str) -> List[str]:
        """Get all stored preference texts for a user."""
        if user_id not in self._store:
            return []
        return [text for text, _, _ in self._store[user_id]]

    def clear(self, user_id: str = None) -> None:
        """Clear memory for a specific user or all users."""
        if user_id is not None:
            self._store.pop(user_id, None)
        else:
            self._store.clear()

    @property
    def user_ids(self) -> List[str]:
        return list(self._store.keys())


class MemoryAugmentedBaseline:
    """Memory-augmented baseline with cross-session preference learning.

    Stores user preference signals in a persistent memory keyed by persona type.
    On each new dialogue:
    1. Retrieves relevant past preferences from memory (warm-start)
    2. Uses DST to extract current dialogue state
    3. Combines past + current preferences for recommendation

    Memory persists across dialogues — `reset_state()` resets DST state
    but NOT the preference memory. This simulates a user returning for
    multiple sessions where the system "remembers" their tastes.

    Requires: sentence-transformers package.
    """
    name = "memory_augmented"

    def __init__(self, model: str = "gemini/gemini-2.5-flash", shared_encoder=None,
                 api_key: Optional[str] = None, api_base: Optional[str] = None):
        self.model = model
        self._lm = _build_lm(model, max_tokens=600, api_key=api_key, api_base=api_base)
        self._response_gen = dspy.Predict(_PredictTurnSignature)
        self._tfidf = TFIDFBaseline()
        self._fallback = PopularityBaseline()
        self._memory = PreferenceMemory(encoder=shared_encoder)
        self._state = TourismDialogueState()
        self._current_user_id: str = ""

    def reset_state(self):
        """Reset dialogue state for new dialogue. Memory persists."""
        self._state = TourismDialogueState()

    def _extract_preferences(
        self,
        dialogue_history: List[DialogueTurn],
        scenario: ScenarioMeta,
    ) -> List[str]:
        """Extract preference signals from dialogue history for memory storage."""
        prefs = []
        for turn in dialogue_history:
            if turn.role != 'user':
                continue
            text = turn.text.lower()
            # Detect explicit preference statements
            if turn.action in ('state_preference', 'reject_and_refine', 'greet_and_seek'):
                prefs.append(turn.text)
            elif any(kw in text for kw in (
                'like', 'prefer', 'want', 'looking for', 'need',
                'love', 'enjoy', 'hate', 'dislike', 'avoid',
                'budget', 'quiet', 'romantic', 'casual', 'outdoor',
            )):
                prefs.append(turn.text)
        return prefs

    def predict_turn(
        self,
        dialogue_history: List[DialogueTurn],
        gold_action: str,
        candidate_pois: List[YelpPOI],
        scenario: ScenarioMeta,
    ) -> BaselinePrediction:
        # Set user ID from persona type
        self._current_user_id = scenario.persona_type

        # Step 1: Store current preferences in memory (only from latest user turn)
        latest_user_turns = [t for t in dialogue_history if t.role == 'user']
        if latest_user_turns:
            new_prefs = self._extract_preferences([latest_user_turns[-1]], scenario)
            for pref_text in new_prefs:
                self._memory.add(self._current_user_id, pref_text)

        # Step 2: Retrieve past preferences (warm-start)
        user_texts = [t.text for t in dialogue_history if t.role == 'user']
        query = ' '.join(user_texts) if user_texts else scenario.persona_description
        past_prefs = self._memory.search(self._current_user_id, query, top_k=5)
        past_pref_text = '\n'.join(
            f"- {text}" for text, score in past_prefs if score > 0.3
        )

        # Step 3: TF-IDF rank with augmented query
        augmented_query = query
        if past_pref_text:
            augmented_query += ' ' + ' '.join(t for t, _ in past_prefs)

        self._tfidf.build_index(candidate_pois)
        ranked = self._tfidf._rank_pois(augmented_query, candidate_pois)

        # Step 4: LLM generates response with memory context
        history_text = _format_history(dialogue_history)
        poi_cards = format_poi_cards(ranked[:6])

        memory_section = ""
        if past_pref_text:
            memory_section = (
                f"## Known Preferences (from past interactions)\n"
                f"{past_pref_text}\n\n"
            )

        action_instruction = {
            'recommend': (
                "Recommend the most suitable POI. "
                "Consider both current and past preferences. "
                "Quote reviews verbatim."
            ),
            'compare': "Compare two POIs considering past preferences. Quote reviews verbatim.",
            'ask_preference': "Ask about preferences to narrow recommendations.",
            'farewell': "Wrap up warmly.",
        }.get(gold_action, f"Respond appropriately (action: {gold_action}).")

        prompt = (
            f"You are a tourism recommendation assistant with memory of past interactions.\n\n"
            f"{memory_section}"
            f"## Current Dialogue\n{history_text}\n\n"
            f"## Candidate POIs\n{poi_cards}\n\n"
            f"## Task\n{action_instruction}\n"
            f"Reference reviews as business_id::R{{n}}."
        )

        try:
            with dspy.context(lm=self._lm, adapter=dspy.JSONAdapter()):
                result = self._response_gen(prompt=prompt)
            output: _LLMTurnOutput = result.output
            return BaselinePrediction(
                response_text=output.response_text,
                recommended_poi_ids=output.recommended_poi_ids,
                referenced_review_ids=output.referenced_review_ids,
            )
        except Exception as e:
            logger.warning(f"Memory-augmented baseline failed: {e}")
            return self._fallback.predict_turn(
                dialogue_history, gold_action, candidate_pois, scenario,
            )


# ---------------------------------------------------------------------------
# 12. ItineraryBaseline — Cross-Type Itinerary Planning
# ---------------------------------------------------------------------------

def _build_proximity_graph(
    pois: List[YelpPOI],
    max_distance_m: float = 500.0,
) -> Dict[str, List[tuple]]:
    """Build a proximity graph between POIs within walking distance.

    Returns adjacency list: {business_id: [(neighbor_id, distance_m, poi_type)]}.
    Only connects POIs of DIFFERENT types (restaurant<->attraction, etc.).
    """
    graph: Dict[str, List[tuple]] = {p.business_id: [] for p in pois}
    poi_lookup = {p.business_id: p for p in pois}

    for i, p1 in enumerate(pois):
        if p1.latitude == 0 and p1.longitude == 0:
            continue
        for j in range(i + 1, len(pois)):
            p2 = pois[j]
            if p2.latitude == 0 and p2.longitude == 0:
                continue
            # Only connect different types
            if p1.poi_type == p2.poi_type:
                continue
            dist_km = _haversine_km(p1.latitude, p1.longitude,
                                    p2.latitude, p2.longitude)
            dist_m = dist_km * 1000
            if dist_m <= max_distance_m:
                graph[p1.business_id].append(
                    (p2.business_id, dist_m, p2.poi_type)
                )
                graph[p2.business_id].append(
                    (p1.business_id, dist_m, p1.poi_type)
                )

    return graph


def _find_nearby_cross_type(
    poi_id: str,
    graph: Dict[str, List[tuple]],
    poi_lookup: Dict[str, YelpPOI],
    max_results: int = 3,
) -> List[tuple]:
    """Find nearby POIs of different types from the proximity graph.

    Returns list of (YelpPOI, distance_m) sorted by distance.
    """
    neighbors = graph.get(poi_id, [])
    results = []
    for nbr_id, dist_m, _ in neighbors:
        nbr_poi = poi_lookup.get(nbr_id)
        if nbr_poi:
            results.append((nbr_poi, dist_m))

    results.sort(key=lambda x: x[1])
    return results[:max_results]


def _optimize_walk_order(
    poi_ids: List[str],
    poi_lookup: Dict[str, YelpPOI],
) -> List[str]:
    """Simple nearest-neighbor TSP to find a walkable visit order.

    Starts from the first POI with valid coordinates, greedily visits
    the nearest unvisited POI. Returns ordered list of business IDs.
    """
    if len(poi_ids) <= 2:
        return list(poi_ids)

    coords = {}
    for pid in poi_ids:
        poi = poi_lookup.get(pid)
        if poi and poi.latitude != 0 and poi.longitude != 0:
            coords[pid] = (poi.latitude, poi.longitude)

    if len(coords) <= 2:
        return list(poi_ids)

    # Nearest-neighbor greedy — start from first POI with valid coordinates
    remaining = set(coords.keys())
    start_pid = next((pid for pid in poi_ids if pid in coords), poi_ids[0])
    order = [start_pid]
    remaining.discard(start_pid)

    while remaining:
        current = order[-1]
        if current not in coords:
            # No coords, just append remaining
            order.extend(remaining)
            break
        c_lat, c_lon = coords[current]
        nearest = min(
            remaining,
            key=lambda pid: (
                _haversine_km(c_lat, c_lon, coords[pid][0], coords[pid][1])
                if pid in coords else float('inf')
            ),
        )
        order.append(nearest)
        remaining.discard(nearest)

    return order


class ItineraryBaseline:
    """Cross-type itinerary recommendation baseline.

    When recommending a restaurant, also suggests nearby attractions/hotels
    within walking distance. Proposes a mini-itinerary that the user could
    follow on foot.

    Requires a full knowledge base (set via set_knowledge_base) to find
    cross-type POIs that may not be in the dialogue's candidate set.
    """
    name = "itinerary"

    def __init__(self):
        self._tfidf = TFIDFBaseline()
        self._fallback = PopularityBaseline()
        self._full_kb: List[YelpPOI] = []
        self._poi_lookup: Dict[str, YelpPOI] = {}
        self._graph: Dict[str, List[tuple]] = {}
        self._graph_built = False

    def set_knowledge_base(self, pois: List[YelpPOI]) -> None:
        """Set the full POI knowledge base for cross-type discovery."""
        self._full_kb = pois
        self._poi_lookup = {p.business_id: p for p in pois}
        self._graph_built = False

    def _ensure_graph(self) -> None:
        """Build proximity graph lazily."""
        if self._graph_built:
            return
        if self._full_kb:
            self._graph = _build_proximity_graph(self._full_kb, max_distance_m=500.0)
            self._graph_built = True

    def _format_itinerary(
        self,
        primary_poi: YelpPOI,
        nearby: List[tuple],
    ) -> str:
        """Format a mini-itinerary suggestion."""
        lines = [f"Starting point: {primary_poi.name} ({primary_poi.poi_type})"]
        if not nearby:
            return lines[0]

        all_pois = [primary_poi] + [poi for poi, _ in nearby]
        all_ids = [p.business_id for p in all_pois]
        ordered_ids = _optimize_walk_order(all_ids, self._poi_lookup)

        for i, pid in enumerate(ordered_ids, 1):
            poi = self._poi_lookup.get(pid)
            if not poi:
                continue
            if pid == primary_poi.business_id:
                lines.append(f"  {i}. {poi.name} ({poi.poi_type}, {poi.stars}/5) <- your main pick")
            else:
                # Find distance from previous
                dist_m = 0
                if i > 1:
                    prev_pid = ordered_ids[i - 2]
                    prev_poi = self._poi_lookup.get(prev_pid)
                    if prev_poi and poi.latitude != 0 and poi.longitude != 0 and prev_poi.latitude != 0 and prev_poi.longitude != 0:
                        dist_m = _haversine_km(
                            prev_poi.latitude, prev_poi.longitude,
                            poi.latitude, poi.longitude,
                        ) * 1000
                dist_str = f", ~{dist_m:.0f}m walk" if dist_m > 0 else ""
                lines.append(
                    f"  {i}. {poi.name} ({poi.poi_type}, {poi.stars}/5{dist_str})"
                )

        return '\n'.join(lines)

    def predict_turn(
        self,
        dialogue_history: List[DialogueTurn],
        gold_action: str,
        candidate_pois: List[YelpPOI],
        scenario: ScenarioMeta,
    ) -> BaselinePrediction:
        self._ensure_graph()

        # TF-IDF rank primary candidates
        self._tfidf.build_index(candidate_pois)
        user_texts = [t.text for t in dialogue_history if t.role == 'user']
        query = ' '.join(user_texts) if user_texts else scenario.persona_description
        ranked = self._tfidf._rank_pois(query, candidate_pois)

        if gold_action == 'recommend':
            poi = ranked[0]
            review_text, review_id = PopularityBaseline._pick_review(poi)

            # Find nearby cross-type POIs
            nearby = _find_nearby_cross_type(
                poi.business_id, self._graph, self._poi_lookup,
            )
            itinerary = self._format_itinerary(poi, nearby)

            if nearby:
                nearby_names = ', '.join(
                    f"{p.name} ({p.poi_type}, ~{d:.0f}m)"
                    for p, d in nearby[:2]
                )
                text = (
                    f"I'd recommend {poi.name} ({poi.stars}/5). "
                    f'A reviewer said: "{review_text}" '
                    f"Nearby you'll also find: {nearby_names}.\n\n"
                    f"Suggested itinerary:\n{itinerary}"
                )
            else:
                text = (
                    f"I'd recommend {poi.name} ({poi.stars}/5). "
                    f'A reviewer said: "{review_text}"'
                )

            rec_ids = [poi.business_id]
            rec_ids.extend(p.business_id for p, _ in nearby[:2])
            return BaselinePrediction(
                response_text=text,
                recommended_poi_ids=rec_ids,
                referenced_review_ids=[review_id] if review_id else [],
            )

        if gold_action == 'compare':
            top2 = ranked[:2]
            parts = []
            poi_ids = []
            review_ids = []
            for poi in top2:
                review_text, review_id = PopularityBaseline._pick_review(poi)
                nearby = _find_nearby_cross_type(
                    poi.business_id, self._graph, self._poi_lookup, max_results=1,
                )
                nearby_str = ""
                if nearby:
                    n_poi, n_dist = nearby[0]
                    nearby_str = f" (with {n_poi.name} {n_poi.poi_type} ~{n_dist:.0f}m away)"
                parts.append(
                    f"{poi.name} ({poi.stars}/5, {poi.price_range or 'N/A'})"
                    f'{nearby_str} — "{review_text}"'
                )
                poi_ids.append(poi.business_id)
                if review_id:
                    review_ids.append(review_id)
            if len(parts) < 2:
                text = f"Here's an option: {parts[0]}." if parts else "I don't have options to compare."
            else:
                text = f"Two options with things nearby: {parts[0]}. Or try {parts[1]}."
            return BaselinePrediction(
                response_text=text,
                recommended_poi_ids=poi_ids,
                referenced_review_ids=review_ids,
            )

        if gold_action == 'ask_preference':
            text = (
                f"What kind of {scenario.primary_poi_type} are you looking for? "
                f"I can also suggest nearby attractions and activities."
            )
            return BaselinePrediction(response_text=text)

        text = "Have a great time exploring! The places I suggested are all walkable from each other."
        return BaselinePrediction(response_text=text)


# ---------------------------------------------------------------------------
# 13. KnowledgeEnhancedBaseline
# ---------------------------------------------------------------------------

# Attribute constraints map: persona keywords -> POI attribute requirements
_ATTRIBUTE_CONSTRAINTS = {
    'family': {'GoodForKids': 'True'},
    'kids': {'GoodForKids': 'True'},
    'children': {'GoodForKids': 'True'},
    'child': {'GoodForKids': 'True'},
    'outdoor': {'OutdoorSeating': 'True'},
    'patio': {'OutdoorSeating': 'True'},
    'terrace': {'OutdoorSeating': 'True'},
    'quiet': {'NoiseLevel': 'quiet'},
    'romantic': {'NoiseLevel': 'quiet'},
    'intimate': {'NoiseLevel': 'quiet'},
    'delivery': {'RestaurantsDelivery': 'True'},
    'takeout': {'RestaurantsTakeOut': 'True'},
    'take-out': {'RestaurantsTakeOut': 'True'},
    'bar': {'Alcohol': 'full_bar'},
    'drinks': {'Alcohol': 'full_bar'},
    'cocktail': {'Alcohol': 'full_bar'},
    'beer': {'Alcohol': 'full_bar'},
    'wine': {'Alcohol': 'full_bar'},
    'reservation': {'RestaurantsReservations': 'True'},
    'wifi': {'WiFi': 'free'},
    'remote': {'WiFi': 'free'},
    'laptop': {'WiFi': 'free'},
    'parking': {'BikeParking': 'True'},
    'drive': {'BikeParking': 'True'},
}

_TYPE_CATEGORIES = {
    'restaurant': {
        'italian': {'Italian', 'Pizza', 'Pasta'},
        'mexican': {'Mexican', 'Tex-Mex', 'Tacos'},
        'asian': {'Chinese', 'Japanese', 'Thai', 'Vietnamese', 'Korean', 'Sushi', 'Asian Fusion', 'Indian'},
        'american': {'American (Traditional)', 'American (New)', 'Burgers', 'Steakhouses', 'Southern'},
        'seafood': {'Seafood', 'Fish & Chips'},
        'breakfast': {'Breakfast & Brunch', 'Bakeries', 'Cafes', 'Coffee & Tea'},
        'vegetarian': {'Vegetarian', 'Vegan'},
        'fast food': {'Fast Food', 'Sandwiches', 'Hot Dogs'},
    },
    'hotel': {
        'luxury': {'Hotels', 'Resorts'},
        'budget': {'Hotels', 'Hostels', 'Motels'},
        'boutique': {'Hotels', 'Bed & Breakfast'},
    },
    'attraction': {
        'museum': {'Museums', 'Art Galleries'},
        'outdoor': {'Parks', 'Hiking', 'Beaches', 'Gardens'},
        'entertainment': {'Amusement Parks', 'Arcades', 'Bowling', 'Mini Golf'},
        'history': {'Historical Tours', 'Landmarks & Historical Buildings'},
    },
}


def _extract_attribute_constraints(text: str) -> Dict[str, str]:
    """Extract POI attribute constraints from user text."""
    text_lower = text.lower()
    constraints = {}
    for keyword, attrs in _ATTRIBUTE_CONSTRAINTS.items():
        if keyword in text_lower:
            constraints.update(attrs)
    return constraints


def _extract_category_preferences(text: str, poi_type: str) -> set:
    """Extract category preferences from user text."""
    text_lower = text.lower()
    type_cats = _TYPE_CATEGORIES.get(poi_type, {})
    preferred_cats = set()
    for cuisine, yelp_cats in type_cats.items():
        if cuisine in text_lower:
            preferred_cats |= yelp_cats
    return preferred_cats


def _extract_price_constraint(text: str) -> Optional[str]:
    """Extract price range preference from text."""
    text_lower = text.lower()
    if any(w in text_lower for w in ('cheap', 'budget', 'affordable', 'inexpensive')):
        return '$'
    if any(w in text_lower for w in ('moderate', 'mid-range', 'reasonable')):
        return '$$'
    if any(w in text_lower for w in ('upscale', 'fine dining', 'luxury', 'splurge', 'fancy')):
        return '$$$$'
    return None


class KnowledgeEnhancedBaseline:
    """Knowledge-enhanced CRS baseline using structured POI attributes.

    Represents the approach of KG-based CRS methods (KECR, MESE, UniCRS)
    adapted for TRACE's evidence-grounded setting. Instead of a movie KG,
    uses POI metadata (categories, price, attributes like WiFi/outdoor/
    kid-friendly) as structured knowledge for constraint-based filtering,
    then retrieves supporting review evidence.

    Pipeline:
    1. Extract structured constraints from dialogue (category, price, attributes)
    2. Score POIs by attribute match (hard constraints + soft category match)
    3. Retrieve review evidence supporting the matched attributes
    4. Generate template response with verbatim review citations
    """
    name = "knowledge_enhanced"

    def __init__(self):
        self._fallback = PopularityBaseline()

    def reset_state(self) -> None:
        pass

    def set_knowledge_base(self, pois: List[YelpPOI]) -> None:
        pass

    def _score_poi(
        self,
        poi: YelpPOI,
        attr_constraints: Dict[str, str],
        cat_preferences: set,
        price_pref: Optional[str],
    ) -> float:
        """Score a POI based on structured knowledge matching."""
        score = 0.0

        # Attribute constraint matching (hard knowledge)
        for attr_key, expected_val in attr_constraints.items():
            actual = str(poi.attributes.get(attr_key, '')).strip("'\"")
            if actual.lower() == expected_val.lower():
                score += 2.0
            elif actual and actual.lower() not in ('none', 'false', ''):
                score += 0.5

        # Category matching (structured taxonomy)
        poi_cats = set(poi.categories)
        if cat_preferences:
            score += len(poi_cats & cat_preferences) * 1.5

        # Price range matching
        if price_pref and poi.price_range:
            price_diff = abs(len(poi.price_range) - len(price_pref))
            if price_diff == 0:
                score += 1.5
            elif price_diff == 1:
                score += 0.5

        # Popularity tiebreaker
        score += (poi.stars / 5.0) * 0.3
        return score

    def _find_attribute_review(
        self,
        poi: YelpPOI,
        attr_constraints: Dict[str, str],
        cat_preferences: set,
    ) -> tuple:
        """Find review that best supports the matched attributes."""
        if not poi.reviews:
            return ("Highly rated by visitors!", None)

        # Build keyword set from constraints
        match_keywords = set()
        attr_keyword_map = {
            'GoodForKids': {'kid', 'kids', 'children', 'family', 'child'},
            'OutdoorSeating': {'outdoor', 'patio', 'terrace', 'outside'},
            'NoiseLevel': {'quiet', 'loud', 'noisy', 'romantic', 'intimate'},
            'WiFi': {'wifi', 'internet', 'laptop', 'work'},
            'RestaurantsDelivery': {'delivery', 'deliver', 'delivered'},
            'RestaurantsTakeOut': {'takeout', 'take-out', 'pickup', 'to-go'},
            'Alcohol': {'bar', 'drink', 'cocktail', 'beer', 'wine', 'happy hour'},
            'RestaurantsReservations': {'reservation', 'book', 'reserved'},
            'BikeParking': {'parking', 'park', 'lot', 'garage'},
        }
        for attr_key in attr_constraints:
            match_keywords |= attr_keyword_map.get(attr_key, set())

        for cat in cat_preferences:
            match_keywords |= set(_tfidf_tokenize(cat.lower()))

        best_rev = None
        best_score = -1
        best_idx = 0
        for idx, rev in enumerate(poi.reviews):
            tokens = set(_tfidf_tokenize(rev.text))
            overlap = len(tokens & match_keywords)
            pos_count = len(tokens & _POSITIVE_WORDS)
            rev_score = overlap * 2 + pos_count + (rev.stars / 5.0)
            if rev_score > best_score:
                best_score = rev_score
                best_rev = rev
                best_idx = idx

        if best_rev is None:
            best_rev = poi.reviews[0]
            best_idx = 0

        ref_id = f"{poi.business_id}::R{best_idx + 1}"
        return (best_rev.text[:150], ref_id)

    def predict_turn(
        self,
        dialogue_history: List[DialogueTurn],
        gold_action: str,
        candidate_pois: List[YelpPOI],
        scenario: ScenarioMeta,
    ) -> BaselinePrediction:
        user_texts = [t.text for t in dialogue_history if t.role == 'user']
        all_user_text = ' '.join(user_texts) if user_texts else scenario.persona_description
        full_context = scenario.persona_description + ' ' + all_user_text

        attr_constraints = _extract_attribute_constraints(full_context)
        cat_preferences = _extract_category_preferences(full_context, scenario.primary_poi_type)
        price_pref = _extract_price_constraint(full_context)

        scored = [
            (poi, self._score_poi(poi, attr_constraints, cat_preferences, price_pref))
            for poi in candidate_pois
        ]
        scored.sort(key=lambda x: -x[1])
        ranked = [poi for poi, _ in scored]

        if gold_action == 'recommend' and ranked:
            poi = ranked[0]
            review_text, review_id = self._find_attribute_review(poi, attr_constraints, cat_preferences)
            match_reasons = []
            if cat_preferences & set(poi.categories):
                matched = cat_preferences & set(poi.categories)
                match_reasons.append(f"matches your cuisine preference ({', '.join(list(matched)[:2])})")
            if price_pref and poi.price_range and len(poi.price_range) == len(price_pref):
                match_reasons.append(f"fits your budget ({poi.price_range})")
            for attr_key, expected in attr_constraints.items():
                actual = str(poi.attributes.get(attr_key, '')).strip("'\"")
                if actual.lower() == expected.lower():
                    readable = attr_key.replace('Restaurants', '').replace('GoodFor', '').lower()
                    match_reasons.append(f"offers {readable}")
            reason_str = (' — it ' + ', '.join(match_reasons)) if match_reasons else ''
            text = (
                f"I'd recommend {poi.name}{reason_str}. "
                f"It's rated {poi.stars}/5 stars. "
                f'A reviewer shared: "{review_text}"'
            )
            return BaselinePrediction(
                response_text=text,
                recommended_poi_ids=[poi.business_id],
                referenced_review_ids=[review_id] if review_id else [],
            )

        if gold_action == 'compare' and len(ranked) >= 2:
            parts, poi_ids, review_ids = [], [], []
            for poi in ranked[:2]:
                review_text, review_id = self._find_attribute_review(poi, attr_constraints, cat_preferences)
                cats_str = ', '.join(poi.categories[:2]) if poi.categories else poi.poi_type
                parts.append(f"{poi.name} ({poi.stars}/5, {poi.price_range or 'N/A'}, {cats_str})"
                             f' — "{review_text}"')
                poi_ids.append(poi.business_id)
                if review_id:
                    review_ids.append(review_id)
            text = f"Here are two options that match your preferences: {parts[0]}. Alternatively, {parts[1]}."
            return BaselinePrediction(response_text=text, recommended_poi_ids=poi_ids, referenced_review_ids=review_ids)

        if gold_action == 'ask_preference':
            return BaselinePrediction(response_text=(
                f"To find the best {scenario.primary_poi_type} for you, "
                f"could you tell me about any specific requirements? "
                f"For example, cuisine type, price range, or features like outdoor seating or kid-friendliness?"
            ))

        return BaselinePrediction(response_text="I hope those suggestions work well for you. Enjoy your visit!")


# ---------------------------------------------------------------------------
# 14. PersonaGroundedBaseline
# ---------------------------------------------------------------------------

_PERSONA_ASPECT_WEIGHTS = {
    'family': {'service': 1.2, 'value': 1.0, 'ambiance': 0.8, 'food': 1.0, 'location': 0.6},
    'couple': {'ambiance': 1.5, 'food': 1.2, 'service': 1.0, 'value': 0.5, 'location': 0.8},
    'solo': {'value': 1.2, 'food': 1.0, 'location': 1.2, 'ambiance': 0.5, 'service': 0.8},
    'business': {'service': 1.3, 'ambiance': 1.2, 'location': 1.0, 'food': 0.8, 'value': 0.5},
    'friends': {'ambiance': 1.3, 'food': 1.0, 'value': 1.0, 'service': 0.8, 'location': 0.8},
    'budget': {'value': 2.0, 'food': 1.0, 'location': 0.8, 'service': 0.5, 'ambiance': 0.3},
    'luxury': {'ambiance': 1.5, 'food': 1.5, 'service': 1.5, 'location': 0.5, 'value': 0.2},
    'senior': {'service': 1.3, 'ambiance': 1.0, 'value': 1.0, 'food': 1.0, 'location': 1.2},
    'foodie': {'food': 2.0, 'ambiance': 0.8, 'service': 0.8, 'value': 0.5, 'location': 0.3},
}
_DEFAULT_ASPECT_WEIGHTS = {'food': 1.0, 'service': 1.0, 'ambiance': 1.0, 'value': 1.0, 'location': 1.0}


def _detect_persona_type(persona_desc: str, persona_type: str) -> str:
    """Map persona description to a category for aspect weighting."""
    type_lower = persona_type.lower()
    for key in _PERSONA_ASPECT_WEIGHTS:
        if key in type_lower:
            return key

    desc_lower = persona_desc.lower()
    keyword_map = {
        'family': ['family', 'kid', 'children', 'child', 'parent', 'dad', 'mom'],
        'couple': ['couple', 'anniversary', 'honeymoon', 'romantic', 'partner', 'date'],
        'solo': ['solo', 'alone', 'single', 'myself'],
        'business': ['business', 'conference', 'meeting', 'work', 'corporate'],
        'friends': ['friend', 'group', 'buddies', 'crew'],
        'budget': ['budget', 'cheap', 'saving', 'backpack', 'affordable'],
        'luxury': ['luxury', 'upscale', 'splurge', 'premium', 'high-end'],
        'senior': ['senior', 'retire', 'elder', 'grandparent', 'accessibility', 'wheelchair'],
        'foodie': ['foodie', 'culinary', 'gastronomy', 'chef', 'food enthusiast', 'food explorer'],
    }
    for persona_key, keywords in keyword_map.items():
        if any(kw in desc_lower for kw in keywords):
            return persona_key
    return 'solo'


class PersonaGroundedBaseline:
    """Persona-grounded review retrieval baseline.

    Represents the approach of PEARL (persona-knowledge grounded CRS)
    adapted for TRACE. Uses persona information to weight review aspects:
    a family traveler cares more about kid-friendliness and value,
    while a couple cares more about ambiance and food quality.

    Pipeline:
    1. Detect persona type from scenario description
    2. Compute persona-weighted aspect scores for each POI's reviews
    3. Rank POIs by persona-weighted review quality
    4. Select reviews that best match the persona's priorities
    5. Generate response with persona-aware framing and review citations
    """
    name = "persona_grounded"

    def __init__(self):
        self._poi_persona_scores: Dict[str, float] = {}
        self._poi_best_reviews: Dict[str, tuple] = {}
        self._current_persona: str = ''
        self._current_key: frozenset = frozenset()

    def reset_state(self) -> None:
        self._poi_persona_scores = {}
        self._poi_best_reviews = {}
        self._current_persona = ''
        self._current_key = frozenset()

    def set_knowledge_base(self, pois: List[YelpPOI]) -> None:
        pass

    def _build_persona_scores(self, pois: List[YelpPOI], persona_type: str) -> None:
        """Score POIs and select reviews weighted by persona aspect preferences."""
        cache_key = frozenset(p.business_id for p in pois)
        if cache_key == self._current_key and persona_type == self._current_persona:
            return

        weights = _PERSONA_ASPECT_WEIGHTS.get(persona_type, _DEFAULT_ASPECT_WEIGHTS)

        for poi in pois:
            profile = _build_poi_aspect_profile(poi)
            poi_score = sum(weights.get(asp, 0.5) * score for asp, score in profile.items())
            poi_score += (poi.stars / 5.0) * 0.5
            self._poi_persona_scores[poi.business_id] = poi_score

            if not poi.reviews:
                self._poi_best_reviews[poi.business_id] = ("Highly recommended!", None)
                continue

            dominant_aspect = max(weights, key=weights.get)
            dominant_keywords = _ASPECT_CATEGORIES.get(dominant_aspect, set())

            best_rev, best_score, best_idx = None, -999, 0
            for idx, rev in enumerate(poi.reviews):
                rev_aspects = _extract_aspect_scores(rev.text)
                rev_score = sum(weights.get(asp, 0.5) * s for asp, s in rev_aspects.items())
                tokens = set(_tfidf_tokenize(rev.text))
                if tokens & dominant_keywords and tokens & _POSITIVE_WORDS:
                    rev_score += 1.0
                rev_score += (rev.stars / 5.0) * 0.3
                if rev_score > best_score:
                    best_score, best_rev, best_idx = rev_score, rev, idx

            if best_rev is None:
                best_rev, best_idx = poi.reviews[0], 0

            self._poi_best_reviews[poi.business_id] = (
                best_rev.text[:150], f"{poi.business_id}::R{best_idx + 1}"
            )

        self._current_key = cache_key
        self._current_persona = persona_type

    def predict_turn(
        self,
        dialogue_history: List[DialogueTurn],
        gold_action: str,
        candidate_pois: List[YelpPOI],
        scenario: ScenarioMeta,
    ) -> BaselinePrediction:
        persona_type = _detect_persona_type(scenario.persona_description, scenario.persona_type)
        self._build_persona_scores(candidate_pois, persona_type)

        scored = [(poi, self._poi_persona_scores.get(poi.business_id, 0)) for poi in candidate_pois]
        scored.sort(key=lambda x: -x[1])
        ranked = [poi for poi, _ in scored]

        persona_framing = {
            'family': "a great family-friendly option",
            'couple': "a wonderful spot for a special evening",
            'solo': "a solid pick for solo dining",
            'business': "an excellent choice for a professional setting",
            'friends': "a fun place for your group",
            'budget': "a great value option",
            'luxury': "an upscale experience",
            'senior': "a comfortable and accessible choice",
            'foodie': "a culinary gem",
        }
        framing = persona_framing.get(persona_type, "a great option")
        context_str = scenario.travel_context.replace('_', ' ')

        if gold_action == 'recommend' and ranked:
            poi = ranked[0]
            review_text, review_id = self._poi_best_reviews.get(poi.business_id, ("Highly rated!", None))
            text = (
                f"For your {context_str}, I think {poi.name} would be {framing}. "
                f"It's rated {poi.stars}/5 stars"
                f"{' (' + poi.price_range + ')' if poi.price_range else ''}. "
                f'One reviewer noted: "{review_text}"'
            )
            return BaselinePrediction(
                response_text=text,
                recommended_poi_ids=[poi.business_id],
                referenced_review_ids=[review_id] if review_id else [],
            )

        if gold_action == 'compare' and len(ranked) >= 2:
            parts, poi_ids, review_ids = [], [], []
            for poi in ranked[:2]:
                review_text, review_id = self._poi_best_reviews.get(poi.business_id, ("Great place!", None))
                parts.append(f"{poi.name} ({poi.stars}/5, {poi.price_range or 'N/A'})"
                             f' — "{review_text}"')
                poi_ids.append(poi.business_id)
                if review_id:
                    review_ids.append(review_id)
            text = f"For your {context_str}, here are two options: {parts[0]}. Or, {parts[1]}."
            return BaselinePrediction(response_text=text, recommended_poi_ids=poi_ids, referenced_review_ids=review_ids)

        if gold_action == 'ask_preference':
            weights = _PERSONA_ASPECT_WEIGHTS.get(persona_type, _DEFAULT_ASPECT_WEIGHTS)
            top_aspects = sorted(weights, key=weights.get, reverse=True)[:2]
            return BaselinePrediction(response_text=(
                f"Since you're on a {context_str}, "
                f"would you like me to focus on {top_aspects[0]} or {top_aspects[1]} "
                f"when finding the best {scenario.primary_poi_type}?"
            ))

        return BaselinePrediction(
            response_text=f"I hope those recommendations suit your {context_str} perfectly!"
        )


# ---------------------------------------------------------------------------
# 15. ItineraryLLMBaseline — LLM with proximity-graph context (Round 4 R2)
# ---------------------------------------------------------------------------
# Isolates whether LLMs fail at spatial reasoning fundamentally or just
# because they lack a structural representation of the candidate space.
# Uses the same _build_proximity_graph() as ItineraryBaseline, but injects
# the per-POI within-500m neighbor list into a DSPy LLM prompt instead of
# templated text.

class _ItineraryLLMSignature(dspy.Signature):
    """Predict the next system turn using POI cards + spatial neighbor structure."""

    prompt: str = dspy.InputField(desc="Dialogue context, candidate POIs, and spatial neighbor graph")
    output: _LLMTurnOutput = dspy.OutputField(desc="System response with POI/review references")


class ItineraryLLMBaseline:
    """LLM baseline with explicit proximity-graph context.

    Same model + similar minimal prompt as LLMZeroShotBaseline, but the prompt
    includes per-POI within-500m cross-type neighbors (computed via the same
    `_build_proximity_graph` used by ItineraryBaseline). The LLM is asked to
    factor walkability into its recommendation.

    Comparing against:
    - `itinerary` (template) → does LLM beat templated itinerary text given the same graph?
    - `llm_zeroshot` (no graph) → does the graph give the LLM measurable lift?
    """
    name = "itinerary_llm"

    def __init__(self, model: str = "gemini/gemini-2.5-flash",
                 api_key: Optional[str] = None, api_base: Optional[str] = None,
                 max_distance_m: float = 500.0):
        self.model = model
        self.max_distance_m = max_distance_m
        self._lm = _build_lm(model, max_tokens=600, api_key=api_key, api_base=api_base)
        self._predictor = dspy.Predict(_ItineraryLLMSignature)
        self._fallback = PopularityBaseline()
        self._full_kb: List[YelpPOI] = []
        self._poi_lookup: Dict[str, YelpPOI] = {}
        self._graph: Dict[str, List[tuple]] = {}
        self._graph_built = False

    def set_knowledge_base(self, pois: List[YelpPOI]) -> None:
        self._full_kb = pois
        self._poi_lookup = {p.business_id: p for p in pois}
        self._graph_built = False

    def _ensure_graph(self) -> None:
        if self._graph_built:
            return
        if self._full_kb:
            self._graph = _build_proximity_graph(self._full_kb, max_distance_m=self.max_distance_m)
            self._graph_built = True

    def _format_neighbor_block(self, pois: List[YelpPOI], max_per_poi: int = 3) -> str:
        """Build a compact spatial-neighbor block for the prompt."""
        if not self._graph:
            return "(no spatial graph available)"
        lines = []
        for poi in pois:
            neighbors = _find_nearby_cross_type(
                poi.business_id, self._graph, self._poi_lookup,
                max_results=max_per_poi,
            )
            if not neighbors:
                lines.append(f"- {poi.name}: no nearby cross-type POIs within {self.max_distance_m:.0f}m")
                continue
            nbr_strs = [
                f"{nbr.name} ({nbr.poi_type}, ~{dist:.0f}m)"
                for nbr, dist in neighbors
            ]
            lines.append(f"- {poi.name}: {', '.join(nbr_strs)}")
        return '\n'.join(lines)

    def predict_turn(
        self,
        dialogue_history: List[DialogueTurn],
        gold_action: str,
        candidate_pois: List[YelpPOI],
        scenario: ScenarioMeta,
    ) -> BaselinePrediction:
        self._ensure_graph()

        # Open-set: prefilter large candidate sets via retrieval
        filtered_pois = prefilter_candidates(candidate_pois, dialogue_history)

        history_text = _format_history(dialogue_history)
        poi_cards = format_poi_cards(filtered_pois)
        neighbor_block = self._format_neighbor_block(filtered_pois)

        action_instruction = {
            'recommend': "Recommend the most suitable POI(s) from the candidates, quoting relevant reviews verbatim. Factor in walkability when neighbors are available.",
            'compare': "Compare two or more candidate POIs, quoting relevant reviews verbatim. Mention walkable nearby cross-type POIs where it adds value.",
            'ask_preference': "Ask the user about their preferences to narrow down recommendations.",
            'farewell': "Wrap up the conversation warmly.",
        }.get(gold_action, f"Respond appropriately (action: {gold_action}).")

        prompt = (
            f"You are a tourism recommendation assistant with access to a walkability graph.\n\n"
            f"## Dialogue so far\n{history_text}\n\n"
            f"## Candidate POIs\n{poi_cards}\n\n"
            f"## Spatial neighbors (cross-type within {self.max_distance_m:.0f}m walking distance)\n"
            f"{neighbor_block}\n\n"
            f"## Task\n{action_instruction}\n"
            f"When quoting reviews, use exact text. Reference reviews as business_id::R{{n}}. "
            f"Use the spatial neighbor information when it helps the user plan a walkable visit."
        )

        try:
            with dspy.context(lm=self._lm, adapter=dspy.JSONAdapter()):
                result = self._predictor(prompt=prompt)
            output: Optional[_LLMTurnOutput] = getattr(result, 'output', None)
            if output is None:
                logger.warning(
                    "ItineraryLLM returned None output; recording empty prediction."
                )
                return BaselinePrediction(
                    response_text='',
                    recommended_poi_ids=[],
                    referenced_review_ids=[],
                )
            return BaselinePrediction(
                response_text=output.response_text,
                recommended_poi_ids=output.recommended_poi_ids,
                referenced_review_ids=output.referenced_review_ids,
            )
        except Exception as e:
            logger.warning(f"ItineraryLLM failed, falling back to popularity: {e}")
            return self._fallback.predict_turn(
                dialogue_history, gold_action, candidate_pois, scenario,
            )
