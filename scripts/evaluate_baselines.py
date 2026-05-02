#!/usr/bin/env python3
"""Evaluate CRS baselines against the gold-standard dialogue dataset.

Usage:
    # Non-LLM baselines only (fast, no API cost)
    python scripts/eval_crs_baselines.py --baselines popularity tfidf

    # All baselines
    python scripts/eval_crs_baselines.py

    # Custom paths
    python scripts/eval_crs_baselines.py \
        --dataset data/crs/crs_dataset.json \
        --knowledge-base data/yelp/poi_knowledge_base.json \
        --output data/crs/baseline_report.md \
        --model gemini/gemini-2.5-flash
"""

import argparse
import asyncio
import json
import logging
import math
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable

from trace_kit.crs.poi_selector import expand_candidate_pool
from trace_kit.crs.baselines import (
    AspectRetrievalBaseline,
    BaselinePrediction,
    CRSBaseline,
    DenseRetrievalBaseline,
    DSTBaseline,
    HybridRRFBaseline,
    ItineraryBaseline,
    ItineraryLLMBaseline,
    KnowledgeEnhancedBaseline,
    LLMZeroShotBaseline,
    MemoryAugmentedBaseline,
    MultiReviewSynthesisBaseline,
    PersonaGroundedBaseline,
    PopularityBaseline,
    RAGCitationBaseline,
    SpatialBaseline,
    TFIDFBaseline,
)
from trace_kit.crs.metrics import (
    aggregate_dialogue_metrics,
    aggregate_metrics,
    build_review_lookup,
    compute_bleu4,
    compute_citation_density,
    compute_composite_grounding,
    compute_dialogue_metrics,
    compute_grounding_score,
    compute_itinerary_diversity,
    compute_llm_judge_score,
    compute_price_alignment,
    compute_provenance_coverage,
    compute_recommendation_metrics,
    compute_entailment_grounding,
    compute_rouge_l,
    compute_spatial_coherence,
    extract_price_preference,
)
from trace_kit.crs.models import CRSDialogue, DialogueTurn, classify_dialogue_difficulty
from trace_kit.datasets.yelp.models import YelpPOI

logger = logging.getLogger(__name__)

# Actions that involve POI recommendation
RECOMMEND_ACTIONS = {'recommend', 'compare'}


# ---------------------------------------------------------------------------
# Data loading — supports both legacy single-file JSON and the public release
# layout (one .jsonl[.gz] file per split for dialogues, .jsonl[.gz] for KB).
# ---------------------------------------------------------------------------

import gzip


def _open_text(path: str, mode: str = "r"):
    return gzip.open(path, mode + "t", encoding="utf-8") if path.endswith(".gz") else open(path, mode, encoding="utf-8")


def _iter_jsonl(path: str):
    with _open_text(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


_PLACEHOLDER_RE = re.compile(r'\[Q:\d+\]')


def _resolve_split_path(p: Path, split: str) -> Optional[Path]:
    """Prefer hydrated `*_full.jsonl[.gz]` over placeholder `*.jsonl[.gz]`."""
    for stem in (f"{split}_full", split):
        for ext in (".jsonl.gz", ".jsonl"):
            fp = p / f"{stem}{ext}"
            if fp.exists():
                return fp
    return None


def load_dataset(path: str) -> List[CRSDialogue]:
    """Load dialogues. Accepts:
      - .json   single dict with key 'dialogues' (legacy)
      - .jsonl[.gz]  one dialogue per line (release layout)
      - directory path: prefers `<split>_full.jsonl[.gz]` (hydrated), falls back
                        to `<split>.jsonl[.gz]`. Concatenates train+val+test.

    If any dialogue still contains `[Q:N]` placeholders, evaluation is aborted —
    run `scripts/reconstruct_kb.py` first to hydrate dialogue text from your
    own copy of the Yelp Open Dataset.
    """
    p = Path(path)
    raw = []
    try:
        if p.is_dir():
            for split in ("train", "validation", "test"):
                fp = _resolve_split_path(p, split)
                if fp is not None:
                    raw.extend(_iter_jsonl(str(fp)))
        elif path.endswith(".jsonl") or path.endswith(".jsonl.gz"):
            raw = list(_iter_jsonl(path))
        else:
            with open(path, "r") as f:
                data = json.load(f)
            if "dialogues" not in data:
                logger.error(f"Dataset file {path} missing 'dialogues' key. Found keys: {list(data.keys())}")
                sys.exit(1)
            raw = data["dialogues"]
    except (json.JSONDecodeError, FileNotFoundError, OSError) as e:
        logger.error(f"Failed to load dataset from {path}: {e}")
        sys.exit(1)

    # Refuse to evaluate against placeholder-text dialogues. The hydrated
    # versions live next to the masked files as *_full.jsonl[.gz].
    placeholder_dialogues = sum(
        1 for d in raw
        for t in d.get("turns", [])
        if _PLACEHOLDER_RE.search(t.get("text", ""))
    )
    if placeholder_dialogues:
        logger.error(
            f"{placeholder_dialogues} turns in {path} still contain [Q:N] placeholders. "
            "Run `python scripts/reconstruct_kb.py --yelp-reviews <path>` to hydrate "
            "dialogue text from your own copy of the Yelp Open Dataset before evaluating."
        )
        sys.exit(1)

    dialogues = [CRSDialogue.from_dict(d) for d in raw]
    logger.info(f"Loaded {len(dialogues)} dialogues from {path}")
    return dialogues


def load_knowledge_base(path: str) -> List[YelpPOI]:
    """Load knowledge base. Accepts:
      - .json    single dict with key 'pois' (legacy)
      - .jsonl[.gz]  one POI per line — must include populated 'reviews' field
                     (run scripts/reconstruct_kb.py first to hydrate review text)
    """
    try:
        if path.endswith(".jsonl") or path.endswith(".jsonl.gz"):
            raw = list(_iter_jsonl(path))
        else:
            with open(path, "r") as f:
                data = json.load(f)
            if "pois" not in data:
                logger.error(f"Knowledge base file {path} missing 'pois' key. Found keys: {list(data.keys())}")
                sys.exit(1)
            raw = data["pois"]
    except (json.JSONDecodeError, FileNotFoundError, OSError) as e:
        logger.error(f"Failed to load knowledge base from {path}: {e}")
        sys.exit(1)

    # Reject stub-only KB (no review text → most baselines silently degrade)
    if raw and "reviews" not in raw[0] and "review_ids" in raw[0]:
        logger.error(
            f"Knowledge base at {path} is a stub (review_ids only, no review text). "
            "Run scripts/reconstruct_kb.py first to hydrate text from your Yelp Open "
            "Dataset download, then point --knowledge-base at the resulting "
            "pois_full.jsonl.gz file."
        )
        sys.exit(1)

    pois = [YelpPOI.from_dict(p) for p in raw]
    logger.info(f"Loaded {len(pois)} POIs from {path}")
    return pois


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def _evaluate_single_dialogue(
    baseline: CRSBaseline,
    dialogue: CRSDialogue,
    poi_lookup: Dict[str, YelpPOI],
    use_llm_judge: bool = False,
    judge_model: str = "gemini/gemini-2.5-flash",
    use_entailment: bool = False,
    candidate_pool_size: int = 0,
    knowledge_base: Optional[List[YelpPOI]] = None,
    multi_reference: Optional[Dict] = None,
) -> Optional[List[Dict]]:
    """Evaluate a single dialogue with a baseline. Returns per-turn metric dicts.

    Args:
        candidate_pool_size: If >0, expand candidate set to this size (ablation).
        knowledge_base: Full KB needed for pool expansion.
        multi_reference: Dict mapping dialogue_id -> turn_id -> list of alternative gold POI IDs.
    """
    from trace_kit.crs.prompts import format_poi_cards

    # Reset DST state for each new dialogue
    if hasattr(baseline, 'reset_state'):
        baseline.reset_state()

    # Recover candidate POIs from metadata
    candidate_ids = dialogue.metadata.get('candidate_poi_ids', [])
    candidate_pois = [poi_lookup[pid] for pid in candidate_ids if pid in poi_lookup]

    if not candidate_pois:
        logger.warning(f"No candidate POIs found for {dialogue.dialogue_id}, skipping")
        return None

    # Expand candidate pool for ablation (e.g., 8 → 16 or 32)
    if candidate_pool_size > 0 and knowledge_base and len(candidate_pois) < candidate_pool_size:
        first_poi = candidate_pois[0]
        candidate_pois = expand_candidate_pool(
            current_candidates=candidate_pois,
            knowledge_base=knowledge_base,
            target_size=candidate_pool_size,
            city=first_poi.city,
            poi_type=first_poi.poi_type,
        )

    # Build review lookup from ALL candidate POIs (including expanded ones)
    review_lookup = build_review_lookup(candidate_pois)

    # Extract user price preference from dialogue (for price alignment metric)
    user_texts = [t.text for t in dialogue.turns if t.role == 'user']
    user_price_max = extract_price_preference(user_texts)

    # TF-IDF baseline auto-caches index by candidate set
    if isinstance(baseline, TFIDFBaseline):
        baseline.build_index(candidate_pois)

    results = []
    for i, turn in enumerate(dialogue.turns):
        if turn.role == 'user':
            # Include user turns with minimal info so dialogue-level metrics
            # (e.g., rejection recovery) can see the full turn sequence
            results.append({
                'dialogue_id': dialogue.dialogue_id,
                'turn_id': turn.turn_id,
                'action': turn.action,
                'role': 'user',
            })
            continue

        # Build history = all turns before this one
        history = dialogue.turns[:i]

        # Predict
        prediction = baseline.predict_turn(
            dialogue_history=history,
            gold_action=turn.action,
            candidate_pois=candidate_pois,
            scenario=dialogue.scenario,
        )

        # Look up multi-reference alternatives for this turn.
        # Priority: external --multi-reference JSON > built-in field on the turn
        # (v5 datasets carry `acceptable_alternative_poi_ids` per turn).
        alt_pois = None
        if multi_reference and dialogue.dialogue_id in multi_reference:
            turn_refs = multi_reference[dialogue.dialogue_id]
            alt_pois = turn_refs.get(str(turn.turn_id), turn_refs.get(turn.turn_id))
        if not alt_pois:
            built_in = getattr(turn, 'acceptable_alternative_poi_ids', None)
            if built_in:
                alt_pois = list(built_in)

        # Compute metrics
        result = _compute_turn_metrics(
            gold_turn=turn,
            prediction=prediction,
            review_lookup=review_lookup,
            poi_lookup=poi_lookup,
            user_price_max=user_price_max,
            use_entailment=use_entailment,
            alternative_gold_pois=alt_pois,
        )
        result['dialogue_id'] = dialogue.dialogue_id
        result['turn_id'] = turn.turn_id
        result['action'] = turn.action
        result['role'] = 'system'
        result['predicted_poi_ids'] = prediction.recommended_poi_ids

        # LLM-as-Judge scoring (optional, expensive)
        if use_llm_judge and turn.action in RECOMMEND_ACTIONS:
            history_text = '\n'.join(
                f"{'User' if t.role == 'user' else 'System'}: {t.text}"
                for t in history
            )
            poi_context = format_poi_cards(candidate_pois[:4])
            judge_scores = compute_llm_judge_score(
                gold_text=turn.text,
                predicted_text=prediction.response_text,
                dialogue_context=history_text,
                poi_context=poi_context,
                model=judge_model,
            )
            if judge_scores:
                for dim, score in judge_scores.items():
                    result[f'judge_{dim}'] = score

        results.append(result)

    return results


# Counter for progress logging in parallel mode
_progress_lock = threading.Lock()
_progress_count = 0


def evaluate_baseline(
    baseline: CRSBaseline,
    dialogues: List[CRSDialogue],
    poi_lookup: Dict[str, YelpPOI],
    use_llm_judge: bool = False,
    judge_model: str = "gemini/gemini-2.5-flash",
    parallel_workers: int = 0,
    baseline_factory: Optional[Callable] = None,
    kb: Optional[List] = None,
    use_entailment: bool = False,
    candidate_pool_size: int = 0,
    multi_reference: Optional[Dict] = None,
    incremental_path: Optional[Path] = None,
) -> List[Dict]:
    """Run a baseline on all dialogues, compute per-turn metrics.

    Args:
        parallel_workers: Number of parallel threads. 0 = sequential (default).
        baseline_factory: Callable that creates a new baseline instance (needed for
            parallel mode with stateful baselines like DST/MemoryAugmented).
        kb: Full knowledge base list (needed to call set_knowledge_base on new instances).
        use_entailment: Enable entailment-based grounding metric (NLI model).
        incremental_path: Path to JSONL file for per-dialogue incremental save + resume.

    Returns list of per-turn metric dicts.
    """
    global _progress_count

    # --- Incremental resume: load completed dialogue IDs ---
    completed_ids: set = set()
    per_turn_results: list = []
    _jsonl_lock = threading.Lock()

    if incremental_path and incremental_path.exists():
        with open(incremental_path, 'r') as f:
            for line in f:
                row = json.loads(line)
                per_turn_results.append(row)
                completed_ids.add(row.get('dialogue_id'))
        logger.info(f"  Resumed {len(completed_ids)} dialogues from {incremental_path.name}")

    pending = [d for d in dialogues if d.dialogue_id not in completed_ids]
    if not pending:
        logger.info(f"  All {len(dialogues)} dialogues already completed")
        return per_turn_results

    logger.info(f"  {len(completed_ids)} done, {len(pending)} pending")

    def _save_results(dlg_results: List[Dict]):
        """Append results to JSONL file (thread-safe)."""
        if not incremental_path:
            return
        with _jsonl_lock:
            with open(incremental_path, 'a') as f:
                for row in dlg_results:
                    f.write(json.dumps(row) + '\n')

    if parallel_workers <= 0:
        # Sequential mode
        skipped_count = 0
        for dialogue in pending:
            dlg_results = _evaluate_single_dialogue(
                baseline, dialogue, poi_lookup, use_llm_judge, judge_model,
                use_entailment=use_entailment,
                candidate_pool_size=candidate_pool_size,
                knowledge_base=kb,
                multi_reference=multi_reference,
            )
            if dlg_results is None:
                skipped_count += 1
            else:
                per_turn_results.extend(dlg_results)
                _save_results(dlg_results)
        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count}/{len(pending)} dialogues due to missing candidate POIs")
        return per_turn_results

    # Async parallel mode: asyncio.Semaphore controls concurrency,
    # asyncio.to_thread wraps sync baseline calls for true I/O parallelism.
    _thread_local = threading.local()
    _init_lock = threading.Lock()
    _done_count = 0
    _done_lock = threading.Lock()
    total = len(pending)
    skipped_count = 0

    def _sync_eval(dialogue: CRSDialogue) -> Optional[List[Dict]]:
        """Sync wrapper run inside asyncio.to_thread."""
        thread_baseline = getattr(_thread_local, 'baseline', None)
        if thread_baseline is None:
            if baseline_factory:
                with _init_lock:
                    thread_baseline = baseline_factory()
                    if kb and hasattr(thread_baseline, 'set_knowledge_base'):
                        thread_baseline.set_knowledge_base(kb)
            else:
                thread_baseline = baseline
            _thread_local.baseline = thread_baseline
        return _evaluate_single_dialogue(
            thread_baseline, dialogue, poi_lookup, use_llm_judge, judge_model,
            use_entailment=use_entailment,
            candidate_pool_size=candidate_pool_size,
            knowledge_base=kb,
            multi_reference=multi_reference,
        )

    sem = asyncio.Semaphore(parallel_workers)

    async def _async_worker(dialogue: CRSDialogue):
        nonlocal _done_count, skipped_count
        async with sem:
            try:
                dlg_results = await asyncio.to_thread(_sync_eval, dialogue)
            except Exception as e:
                logger.warning(f"Dialogue {dialogue.dialogue_id} failed: {e}")
                with _done_lock:
                    skipped_count += 1
                    _done_count += 1
                return
            with _done_lock:
                _done_count += 1
                if _done_count % 50 == 0 or _done_count == total:
                    logger.info(f"  Progress: {_done_count}/{total} dialogues")
            if dlg_results is None:
                with _done_lock:
                    skipped_count += 1
            else:
                per_turn_results.extend(dlg_results)
                _save_results(dlg_results)

    async def _run_all():
        # Override asyncio's default ThreadPoolExecutor (caps at ~min(32, cpu+4)).
        # Without this, parallel_workers > ~32 has no effect because asyncio.to_thread()
        # serializes through the small default pool even when the semaphore allows more.
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=parallel_workers))
        await asyncio.gather(*[_async_worker(dlg) for dlg in pending])

    asyncio.run(_run_all())

    if skipped_count > 0:
        logger.warning(f"Skipped {skipped_count}/{len(pending)} dialogues due to missing candidate POIs")

    return per_turn_results


def _compute_turn_metrics(
    gold_turn: DialogueTurn,
    prediction: BaselinePrediction,
    review_lookup: Dict[str, str],
    poi_lookup: Dict[str, 'YelpPOI'] = None,
    user_price_max: int = None,
    use_entailment: bool = False,
    alternative_gold_pois: Optional[List[str]] = None,
) -> Dict:
    """Compute all metrics for a single turn prediction vs gold."""
    result = {}

    # Text quality
    result['bleu4'] = compute_bleu4(gold_turn.text, prediction.response_text)
    result['rouge_l'] = compute_rouge_l(gold_turn.text, prediction.response_text)

    # Grounding
    result['grounding'] = compute_grounding_score(
        prediction.response_text,
        prediction.referenced_review_ids,
        review_lookup,
    )

    # Citation density
    result['citation_density'] = compute_citation_density(
        prediction.response_text,
        review_lookup,
    )

    # Provenance coverage
    result['provenance_coverage'] = compute_provenance_coverage(
        prediction.response_text,
    )

    # Composite grounding (penalizes no-citation responses)
    result['composite_grounding'] = compute_composite_grounding(
        result['grounding'],
        result['citation_density'],
        result['provenance_coverage'],
    )

    # Entailment-based grounding (optional, NLI model)
    if use_entailment:
        ent_score = compute_entailment_grounding(
            prediction.response_text,
            prediction.referenced_review_ids,
            review_lookup,
        )
        if ent_score is not None:
            result['entailment_grounding'] = ent_score

    # Recommendation metrics (only for recommend/compare actions)
    if gold_turn.action in RECOMMEND_ACTIONS and gold_turn.referenced_poi_ids:
        rec_metrics = compute_recommendation_metrics(
            gold_poi_ids=gold_turn.referenced_poi_ids,
            predicted_poi_ids=prediction.recommended_poi_ids,
        )
        result.update(rec_metrics)

        # Multi-reference evaluation (if alternatives provided)
        # Counts a hit if predicted POI is in gold OR alternatives.
        # Denominator stays as len(gold) to keep scale comparable.
        if alternative_gold_pois:
            acceptable = set(gold_turn.referenced_poi_ids) | set(alternative_gold_pois)
            pred = prediction.recommended_poi_ids
            def _mr_recall(k):
                hits = acceptable & set(pred[:k])
                return min(1.0, len(hits) / max(1, len(gold_turn.referenced_poi_ids)))
            rrs = []
            for gid in gold_turn.referenced_poi_ids:
                # Accept any acceptable POI in place of this gold
                found = False
                for rank, pid in enumerate(pred, 1):
                    if pid in acceptable:
                        rrs.append(1.0 / rank)
                        found = True
                        break
                if not found:
                    rrs.append(0.0)
            result['multi_ref_recall@1'] = _mr_recall(1)
            result['multi_ref_recall@3'] = _mr_recall(3)
            result['multi_ref_mrr'] = sum(rrs) / len(rrs) if rrs else 0.0

    # Itinerary diversity (if POI lookup available)
    if poi_lookup and prediction.recommended_poi_ids:
        it_div = compute_itinerary_diversity(
            prediction.recommended_poi_ids, poi_lookup,
        )
        if it_div is not None:
            result['itinerary_diversity'] = it_div

    # Spatial coherence (if POI lookup available)
    if poi_lookup and prediction.recommended_poi_ids:
        sp_coh = compute_spatial_coherence(
            prediction.recommended_poi_ids, poi_lookup,
        )
        if sp_coh is not None:
            result['spatial_coherence'] = sp_coh

    # Price alignment (if user expressed price preference and POI lookup available)
    if user_price_max is not None and poi_lookup and prediction.recommended_poi_ids:
        pa = compute_price_alignment(
            prediction.recommended_poi_ids, poi_lookup, user_price_max,
        )
        if pa is not None:
            result['price_alignment'] = pa

    return result


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

def compute_dataset_statistics(dialogues: List[CRSDialogue]) -> Dict:
    """Compute descriptive statistics about the dialogue dataset.

    Returns a dict with stats for reporting.
    """
    import statistics as stats_mod

    n = len(dialogues)
    if n == 0:
        return {}

    # Turn length distribution
    turn_lengths = [d.num_turns for d in dialogues]
    turn_stats = {
        'min': min(turn_lengths),
        'max': max(turn_lengths),
        'mean': stats_mod.mean(turn_lengths),
        'stdev': stats_mod.stdev(turn_lengths) if n > 1 else 0.0,
    }

    # Action distribution
    action_counts: Dict[str, int] = {}
    for d in dialogues:
        for t in d.turns:
            action_counts[t.action] = action_counts.get(t.action, 0) + 1

    # POI type distribution
    poi_type_counts: Dict[str, int] = {}
    for d in dialogues:
        pt = d.scenario.primary_poi_type
        poi_type_counts[pt] = poi_type_counts.get(pt, 0) + 1

    # City distribution
    city_counts: Dict[str, int] = {}
    for d in dialogues:
        city_counts[d.city] = city_counts.get(d.city, 0) + 1

    # Persona distribution
    persona_counts: Dict[str, int] = {}
    for d in dialogues:
        persona_counts[d.scenario.persona_type] = persona_counts.get(d.scenario.persona_type, 0) + 1

    # Rejection rate
    total_user_turns = sum(1 for d in dialogues for t in d.turns if t.role == 'user')
    reject_turns = action_counts.get('reject_and_refine', 0)
    rejection_rate = reject_turns / total_user_turns if total_user_turns > 0 else 0.0

    # Average POIs recommended per dialogue
    pois_per_dialogue = [len(d.recommended_poi_ids) for d in dialogues]
    avg_pois = stats_mod.mean(pois_per_dialogue) if pois_per_dialogue else 0.0

    return {
        'n_dialogues': n,
        'turn_stats': turn_stats,
        'action_counts': action_counts,
        'poi_type_counts': poi_type_counts,
        'city_counts': city_counts,
        'persona_counts': persona_counts,
        'rejection_rate': rejection_rate,
        'reject_turns': reject_turns,
        'total_user_turns': total_user_turns,
        'avg_pois_recommended': avg_pois,
    }


def format_dataset_statistics(ds: Dict) -> str:
    """Format dataset statistics as markdown."""
    if not ds:
        return ""
    lines = [
        "## Dataset Statistics\n",
        f"- Dialogues: {ds['n_dialogues']}",
        f"- Cities: {len(ds['city_counts'])} ({', '.join(sorted(ds['city_counts']))})",
        f"- POI types: {len(ds['poi_type_counts'])} ({', '.join(sorted(ds['poi_type_counts']))})",
        f"- Turn length: min={ds['turn_stats']['min']}, max={ds['turn_stats']['max']}, "
        f"mean={ds['turn_stats']['mean']:.1f}, stdev={ds['turn_stats']['stdev']:.1f}",
        f"- Avg POIs recommended/dialogue: {ds['avg_pois_recommended']:.1f}",
        f"- Rejection rate: {ds['reject_turns']}/{ds['total_user_turns']} "
        f"({ds['rejection_rate'] * 100:.1f}%)",
    ]

    lines.append("\n### POI Type Distribution")
    for pt, count in sorted(ds['poi_type_counts'].items()):
        lines.append(f"- {pt}: {count}")

    lines.append("\n### City Distribution")
    for city, count in sorted(ds['city_counts'].items()):
        lines.append(f"- {city}: {count}")

    lines.append("\n### Persona Distribution")
    for persona, count in sorted(ds['persona_counts'].items()):
        lines.append(f"- {persona}: {count}")

    lines.append("\n### Action Distribution")
    for action, count in sorted(ds['action_counts'].items(), key=lambda x: -x[1]):
        lines.append(f"- {action}: {count}")

    lines.append("")
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _compute_dialogue_level_metrics(
    all_results: Dict[str, List[Dict]],
    dialogues: List[CRSDialogue],
) -> Dict[str, Dict]:
    """Compute dialogue-level metrics for each baseline.

    Returns {baseline_name: aggregated_dialogue_metrics}.
    """
    gold_pois_by_dialogue = {
        d.dialogue_id: d.recommended_poi_ids for d in dialogues
    }

    dialogue_metrics = {}
    for name, turn_results in all_results.items():
        # Group turn results by dialogue
        by_dialogue: Dict[str, List[Dict]] = {}
        for r in turn_results:
            did = r.get('dialogue_id', '')
            by_dialogue.setdefault(did, []).append(r)

        per_dialogue = []
        for did, turns in by_dialogue.items():
            gold_pois = gold_pois_by_dialogue.get(did, [])
            dm = compute_dialogue_metrics(turns, gold_pois)
            per_dialogue.append(dm)

        dialogue_metrics[name] = aggregate_dialogue_metrics(per_dialogue)

    return dialogue_metrics


def generate_report(
    all_results: Dict[str, List[Dict]],
    output_path: str,
    dataset_stats: Dict = None,
    dialogues: List[CRSDialogue] = None,
    difficulty_stratify: bool = False,
) -> str:
    """Generate a markdown report comparing all baselines.

    Args:
        all_results: {baseline_name: [per_turn_results]}
        output_path: path to write the markdown report
        dataset_stats: optional dataset statistics dict from compute_dataset_statistics
        dialogues: optional list of dialogues for dialogue-level metrics
        difficulty_stratify: if True, add difficulty-stratified breakdown table

    Returns:
        The markdown report string.
    """
    lines = []
    lines.append("# CRS Baseline Evaluation Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Dataset statistics section (if available)
    if dataset_stats:
        lines.append(format_dataset_statistics(dataset_stats))

    # Aggregate per baseline (system turns only for per-turn metrics)
    summaries = {}
    for name, results in all_results.items():
        system_results = [r for r in results if r.get('role') != 'user']
        summaries[name] = aggregate_metrics(system_results)

    # --- Table 1: Recommendation Quality ---
    lines.append("## Recommendation Quality (recommend/compare turns)")
    lines.append("")
    lines.append("| Baseline | Recall@1 | Recall@3 | MRR |")
    lines.append("|----------|----------|----------|-----|")
    for name in summaries:
        rec_results = [r for r in all_results[name] if r.get('action') in RECOMMEND_ACTIONS]
        if not rec_results:
            lines.append(f"| {name} | — | — | — |")
            continue
        rec_agg = aggregate_metrics(rec_results)
        o = rec_agg['overall']
        lines.append(
            f"| {name} "
            f"| {o.get('recall@1', 0):.3f} "
            f"| {o.get('recall@3', 0):.3f} "
            f"| {o.get('mrr', 0):.3f} |"
        )
    lines.append("")

    # --- Table 2: Response Quality ---
    lines.append("## Response Quality (all system turns)")
    lines.append("")
    lines.append("| Baseline | BLEU-4 | ROUGE-L | Turns |")
    lines.append("|----------|--------|---------|-------|")
    for name, summary in summaries.items():
        o = summary['overall']
        lines.append(
            f"| {name} "
            f"| {o.get('bleu4', 0):.4f} "
            f"| {o.get('rouge_l', 0):.4f} "
            f"| {o.get('count', 0)} |"
        )
    lines.append("")

    # --- Table 3: Grounding & Citation Quality ---
    lines.append("## Grounding & Citation Quality")
    lines.append("")
    lines.append("| Baseline | Grounding | Citation Density | Provenance Coverage |")
    lines.append("|----------|-----------|------------------|---------------------|")
    for name, summary in summaries.items():
        o = summary['overall']
        lines.append(
            f"| {name} "
            f"| {o.get('grounding', 0):.3f} "
            f"| {o.get('citation_density', 0):.3f} "
            f"| {o.get('provenance_coverage', 0):.3f} |"
        )
    lines.append("")

    # --- Table 3b: Practical Value Metrics (Spatial, Price, Diversity) ---
    lines.append("## Practical Value Metrics (Spatial & Price & Diversity)")
    lines.append("")
    lines.append("| Baseline | Spatial Coherence | Price Alignment | Itinerary Diversity |")
    lines.append("|----------|-------------------|-----------------|---------------------|")
    for name, summary in summaries.items():
        o = summary['overall']
        sc = o.get('spatial_coherence')
        pa = o.get('price_alignment')
        id_val = o.get('itinerary_diversity')
        sc_str = f"{sc:.3f}" if sc is not None else "---"
        pa_str = f"{pa:.3f}" if pa is not None else "---"
        id_str = f"{id_val:.3f}" if id_val is not None else "---"
        lines.append(f"| {name} | {sc_str} | {pa_str} | {id_str} |")
    lines.append("")

    # --- Table 4: Dialogue-Level Metrics (if dialogues available) ---
    # Compute once, reuse for table and raw JSON
    dlg_metrics = _compute_dialogue_level_metrics(all_results, dialogues) if dialogues else None

    if dlg_metrics is not None:
        lines.append("## Dialogue-Level Metrics")
        lines.append("")
        lines.append("| Baseline | Task Success | Turns to 1st Correct | Rejection Recovery |")
        lines.append("|----------|-------------|---------------------|--------------------|")
        for name in all_results:
            dm = dlg_metrics.get(name, {})
            tsr = dm.get('task_success', 0)
            t2r = dm.get('turns_to_first_correct')
            t2r_str = f"{t2r:.2f}" if t2r is not None and not math.isnan(float(t2r)) else "---"
            rrr = dm.get('rejection_recovery_rate')
            rrr_str = f"{rrr:.3f}" if rrr is not None else "---"
            lines.append(
                f"| {name} "
                f"| {tsr:.3f} "
                f"| {t2r_str} "
                f"| {rrr_str} |"
            )
        lines.append("")

    # --- Table 5: LLM Judge Scores (if available) ---
    has_judge = any(
        'judge_overall' in r
        for results in all_results.values()
        for r in results
    )
    if has_judge:
        lines.append("## LLM-as-Judge Scores (recommend/compare turns)")
        lines.append("")
        lines.append(
            "| Baseline | Relevance | Informativeness | Grounding | "
            "Flow | Justification | Overall |"
        )
        lines.append(
            "|----------|-----------|-----------------|-----------|"
            "------|---------------|---------|"
        )
        for name, summary in summaries.items():
            o = summary['overall']
            if 'judge_overall' not in o:
                lines.append(f"| {name} | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {name} "
                f"| {o.get('judge_relevance', 0):.2f} "
                f"| {o.get('judge_informativeness', 0):.2f} "
                f"| {o.get('judge_grounding_quality', 0):.2f} "
                f"| {o.get('judge_natural_flow', 0):.2f} "
                f"| {o.get('judge_recommendation_justification', 0):.2f} "
                f"| {o.get('judge_overall', 0):.2f} |"
            )
        lines.append("")

    # --- Table 6: Difficulty-Stratified Breakdown (if enabled) ---
    if difficulty_stratify and dialogues:
        lines.append("## Difficulty-Stratified Results")
        lines.append("")

        # Classify all dialogues
        difficulties = {
            d.dialogue_id: classify_dialogue_difficulty(d) for d in dialogues
        }

        # Summary of difficulty distribution
        dist = {'easy': 0, 'medium': 0, 'hard': 0}
        for dd in difficulties.values():
            dist[dd.overall.value] += 1
        lines.append(
            f"Dialogue distribution: easy={dist['easy']}, "
            f"medium={dist['medium']}, hard={dist['hard']}"
        )
        lines.append("")

        lines.append(
            "| Baseline | Difficulty | BLEU-4 | ROUGE-L | Recall@1 | MRR | Count |"
        )
        lines.append(
            "|----------|------------|--------|---------|----------|-----|-------|"
        )

        for name, results in all_results.items():
            # Group results by difficulty (system turns only)
            by_diff: Dict[str, List[Dict]] = {'easy': [], 'medium': [], 'hard': []}
            for r in results:
                if r.get('role') == 'user':
                    continue
                did = r.get('dialogue_id', '')
                dd = difficulties.get(did)
                if dd:
                    by_diff[dd.overall.value].append(r)

            for level in ('easy', 'medium', 'hard'):
                level_results = by_diff[level]
                if not level_results:
                    lines.append(f"| {name} | {level} | — | — | — | — | 0 |")
                    continue
                agg = aggregate_metrics(level_results)
                o = agg['overall']
                lines.append(
                    f"| {name} | {level} "
                    f"| {o.get('bleu4', 0):.4f} "
                    f"| {o.get('rouge_l', 0):.4f} "
                    f"| {o.get('recall@1', 0):.3f} "
                    f"| {o.get('mrr', 0):.3f} "
                    f"| {o.get('count', 0)} |"
                )
        lines.append("")

    # --- Per-Action Breakdown ---
    lines.append("## Per-Action Breakdown")
    lines.append("")

    all_actions = sorted({
        r.get('action', '')
        for results in all_results.values()
        for r in results
    })

    for action in all_actions:
        lines.append(f"### {action}")
        lines.append("")
        has_rec = action in RECOMMEND_ACTIONS
        if has_rec:
            lines.append("| Baseline | BLEU-4 | ROUGE-L | Grounding | Citation | Recall@1 | MRR | Count |")
            lines.append("|----------|--------|---------|-----------|----------|----------|-----|-------|")
        else:
            lines.append("| Baseline | BLEU-4 | ROUGE-L | Grounding | Citation | Count |")
            lines.append("|----------|--------|---------|-----------|----------|-------|")

        for name, summary in summaries.items():
            pa = summary['per_action'].get(action)
            if not pa:
                continue
            if has_rec:
                lines.append(
                    f"| {name} "
                    f"| {pa.get('bleu4', 0):.4f} "
                    f"| {pa.get('rouge_l', 0):.4f} "
                    f"| {pa.get('grounding', 0):.3f} "
                    f"| {pa.get('citation_density', 0):.3f} "
                    f"| {pa.get('recall@1', 0):.3f} "
                    f"| {pa.get('mrr', 0):.3f} "
                    f"| {pa.get('count', 0)} |"
                )
            else:
                lines.append(
                    f"| {name} "
                    f"| {pa.get('bleu4', 0):.4f} "
                    f"| {pa.get('rouge_l', 0):.4f} "
                    f"| {pa.get('grounding', 0):.3f} "
                    f"| {pa.get('citation_density', 0):.3f} "
                    f"| {pa.get('count', 0)} |"
                )
        lines.append("")

    report = '\n'.join(lines)

    # Write markdown report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)

    # Write raw JSON alongside
    out_p = Path(output_path)
    raw_path = str(out_p.with_name(out_p.stem + '_raw.json'))
    raw_data = {
        'generated_at': datetime.now().isoformat(),
        'baselines': {
            name: {
                'summary': summaries[name],
                'per_turn': results,
            }
            for name, results in all_results.items()
        },
    }

    # Add dialogue-level metrics to raw data (reuse precomputed)
    if dlg_metrics is not None:
        raw_data['dialogue_metrics'] = dlg_metrics

    # Add difficulty classifications
    if difficulty_stratify and dialogues:
        raw_data['difficulty_classifications'] = {
            d.dialogue_id: classify_dialogue_difficulty(d).to_dict()
            for d in dialogues
        }

    with open(raw_path, 'w') as f:
        json.dump(raw_data, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Report written to {output_path}")
    logger.info(f"Raw data written to {raw_path}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_BASELINE_NAMES = [
    'popularity', 'tfidf', 'llm_zeroshot', 'dst', 'rag_citation',
    'aspect', 'dense', 'spatial', 'hybrid_rrf', 'multi_review_synthesis',
    'memory_augmented', 'itinerary', 'knowledge_enhanced', 'persona_grounded',
    'itinerary_llm',
]

# Per-turn token heuristics: (avg_input_tok, avg_output_tok, calls_per_system_turn)
# Derived from prompt templates + typical dialogue history (~10 turns) + 8 POI cards.
# Non-LLM baselines are 0 API; wall-time only.
LLM_COST_HEURISTICS = {
    'llm_zeroshot':           (1500, 200, 1.0),
    'dst':                    (1800, 250, 2.0),   # DST extract + response gen
    'rag_citation':           (2000, 250, 1.2),   # ~20% retry rate for citation validation
    'multi_review_synthesis': (3000, 300, 1.0),   # aspect clustering = larger prompt
    'memory_augmented':       (2000, 250, 1.0),
}

# USD per 1M tokens (input, output). List prices as of 2026-04. Override via env if needed.
MODEL_PRICING = {
    'gemini/gemini-2.5-flash':            (0.15,  0.60),
    'openai/gemini-2.5-flash':            (0.15,  0.60),
    'openai/gemini-3-flash-preview':      (0.15,  0.60),
    'gpt-4o':                             (2.50, 10.00),
    'openai/gpt-4o':                      (2.50, 10.00),
    'gpt-4.1':                            (2.00,  8.00),
    'openai/gpt-4.1':                     (2.00,  8.00),
    'gpt-4o-mini':                        (0.15,  0.60),
    'openai/gpt-4o-mini':                 (0.15,  0.60),
    'claude-sonnet-4-20250514':           (3.00, 15.00),
    'openai/claude-sonnet-4-20250514':    (3.00, 15.00),
    'anthropic/claude-sonnet-4':          (3.00, 15.00),
}
# Judge call heuristics: (input, output, calls_per_recommend_turn)
JUDGE_HEURISTICS = (800, 50, 1.0)

# Per-turn wall time estimates (seconds), assuming --parallel-workers=32 for LLM.
WALL_TIME_PER_TURN_S = {
    'popularity': 0.01, 'tfidf': 0.02, 'aspect': 0.03, 'dense': 0.05,
    'spatial': 0.03, 'hybrid_rrf': 0.05, 'itinerary': 0.08,
    'knowledge_enhanced': 0.02, 'persona_grounded': 0.03,
    'llm_zeroshot': 0.06, 'dst': 0.10, 'rag_citation': 0.08,
    'multi_review_synthesis': 0.10, 'memory_augmented': 0.08,
    'itinerary_llm': 0.08,
}


def _lookup_price(model: str):
    """Return (input$/M, output$/M) for a model. Falls back to Gemini Flash pricing with a warning."""
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    # normalize: try stripping common prefixes
    for k, v in MODEL_PRICING.items():
        if k.endswith(model) or model.endswith(k.split('/')[-1]):
            return v
    logger.warning(f"No pricing for model '{model}'; using Gemini 2.5 Flash pricing as fallback")
    return (0.15, 0.60)


def _format_cost(usd: float) -> str:
    if usd < 0.01:
        return f"${usd*100:.2f}¢"
    if usd < 10:
        return f"${usd:.3f}"
    return f"${usd:.2f}"


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}min"
    return f"{seconds/3600:.2f}h"


def estimate_cost_and_exit(
    baselines: List[str],
    sys_turns: int,
    rec_turns: int,
    n_dialogues: int,
    model: str,
    use_llm_judge: bool,
    judge_models: Optional[List[str]],
    parallel_workers: int,
):
    """Print cost/time table and exit. Called when --dry-run is set."""
    print()
    print("=" * 84)
    print(f"  DRY RUN — cost & time estimate")
    print("=" * 84)
    print(f"  Dataset:         {n_dialogues} dialogues, {sys_turns} system turns, {rec_turns} recommend turns")
    print(f"  LLM model:       {model}")
    price_in, price_out = _lookup_price(model)
    print(f"  Pricing:         ${price_in}/M input, ${price_out}/M output")
    print(f"  Parallel:        {parallel_workers if parallel_workers > 0 else 'sequential'} workers")
    print()
    print(f"  {'Baseline':<26}{'API calls':>12}{'Input tok':>14}{'Output tok':>14}{'USD':>10}{'Wall time':>12}")
    print(f"  {'-'*26}{'-'*12:>12}{'-'*14:>14}{'-'*14:>14}{'-'*10:>10}{'-'*12:>12}")

    total_calls = 0
    total_in_tok = 0
    total_out_tok = 0
    total_usd = 0.0
    total_seq_seconds = 0.0

    for b in baselines:
        sec_per_turn = WALL_TIME_PER_TURN_S.get(b, 0.05)
        if b in LLM_COST_HEURISTICS:
            ti, to, cpt = LLM_COST_HEURISTICS[b]
            calls = int(sys_turns * cpt)
            in_tok = calls * ti
            out_tok = calls * to
            usd = (in_tok / 1e6) * price_in + (out_tok / 1e6) * price_out
            seq_seconds = sys_turns * sec_per_turn * cpt
            wall = seq_seconds / max(1, parallel_workers) if parallel_workers > 0 else seq_seconds
            print(f"  {b:<26}{calls:>12,}{in_tok:>14,}{out_tok:>14,}{_format_cost(usd):>10}{_format_duration(wall):>12}")
            total_calls += calls
            total_in_tok += in_tok
            total_out_tok += out_tok
            total_usd += usd
            total_seq_seconds += seq_seconds
        else:
            seq_seconds = sys_turns * sec_per_turn
            print(f"  {b:<26}{'—':>12}{'—':>14}{'—':>14}{'$0.00':>10}{_format_duration(seq_seconds):>12}")
            total_seq_seconds += seq_seconds

    print(f"  {'-'*26}{'-'*12:>12}{'-'*14:>14}{'-'*14:>14}{'-'*10:>10}{'-'*12:>12}")
    total_wall = (total_seq_seconds / max(1, parallel_workers)) if parallel_workers > 0 else total_seq_seconds
    print(f"  {'TOTAL (baselines)':<26}{total_calls:>12,}{total_in_tok:>14,}{total_out_tok:>14,}{_format_cost(total_usd):>10}{_format_duration(total_wall):>12}")

    # Judge cost
    if use_llm_judge or judge_models:
        ji, jo, jcpt = JUDGE_HEURISTICS
        judge_list = judge_models or [model]
        print()
        print(f"  Judge evaluation ({len(judge_list)} model{'s' if len(judge_list)>1 else ''}, rec turns only):")
        judge_total_usd = 0.0
        judge_total_seconds = 0.0
        for jm in judge_list:
            jin_price, jout_price = _lookup_price(jm)
            # Judge runs once per baseline per rec turn
            calls = int(rec_turns * jcpt * len(baselines))
            in_tok = calls * ji
            out_tok = calls * jo
            usd = (in_tok / 1e6) * jin_price + (out_tok / 1e6) * jout_price
            seq_seconds = calls * 0.5  # typical judge latency
            wall = seq_seconds / max(1, parallel_workers) if parallel_workers > 0 else seq_seconds
            print(f"  {jm:<26}{calls:>12,}{in_tok:>14,}{out_tok:>14,}{_format_cost(usd):>10}{_format_duration(wall):>12}")
            judge_total_usd += usd
            judge_total_seconds += seq_seconds
        judge_wall = (judge_total_seconds / max(1, parallel_workers)) if parallel_workers > 0 else judge_total_seconds
        print(f"  {'TOTAL (judges)':<26}{'':>12}{'':>14}{'':>14}{_format_cost(judge_total_usd):>10}{_format_duration(judge_wall):>12}")
        total_usd += judge_total_usd
        total_seq_seconds += judge_total_seconds
        total_wall = (total_seq_seconds / max(1, parallel_workers)) if parallel_workers > 0 else total_seq_seconds

    print()
    print(f"  {'GRAND TOTAL':<26}{'':>12}{'':>14}{'':>14}{_format_cost(total_usd):>10}{_format_duration(total_wall):>12}")
    print("=" * 84)
    print("  Estimates are heuristic. Real cost ±30% typical. No API calls were made.")
    print("  Re-run without --dry-run to execute.")
    print("=" * 84)
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CRS baselines against gold-standard dialogues.",
    )
    parser.add_argument(
        '--dataset',
        default='data/crs/crs_dataset.json',
        help='Path to CRS dataset JSON (default: data/crs/crs_dataset.json)',
    )
    parser.add_argument(
        '--knowledge-base',
        default='data/yelp/poi_knowledge_base_v3.json',
        help='Path to POI knowledge base JSON. Dataset v4 requires v3 KB '
             '(v2 only contains 1/8 of v4 candidates — 16.5%% overlap).',
    )
    parser.add_argument(
        '--output',
        default='data/crs/baseline_report.md',
        help='Output path for markdown report',
    )
    parser.add_argument(
        '--baselines',
        nargs='+',
        default=['popularity', 'tfidf'],
        choices=ALL_BASELINE_NAMES,
        help=f'Which baselines to evaluate (choices: {ALL_BASELINE_NAMES})',
    )
    parser.add_argument(
        '--model',
        default='gemini/gemini-2.5-flash',
        help='LLM model for LLM-based baselines (llm_zeroshot, dst, rag_citation)',
    )
    parser.add_argument(
        '--api-key-env',
        default='OPENAI_API_KEY',
        help=(
            'Env var holding the API key for LLM baselines. '
            'Use OPENAI_API_KEY for default 3x group, '
            'CODEX_API_KEY for Codex特价 0.2x (gpt-5.x), '
            'CLAUDE_CODE_API_KEY for ClaudeCode逆向 0.4x (claude-*).'
        ),
    )
    parser.add_argument(
        '--use-llm-judge',
        action='store_true',
        help='Enable LLM-as-Judge scoring (slower, requires API calls per turn)',
    )
    parser.add_argument(
        '--use-entailment',
        action='store_true',
        help='Enable entailment-based grounding metric (requires NLI model, slower)',
    )
    parser.add_argument(
        '--difficulty-stratify',
        action='store_true',
        help='Enable difficulty-stratified reporting',
    )
    parser.add_argument(
        '--parallel-workers',
        type=int,
        default=0,
        help='Number of parallel threads for dialogue evaluation. 0 = sequential (default). '
             'Recommended: 16-64 for LLM baselines to parallelize API calls.',
    )
    parser.add_argument(
        '--prefilter-top-k',
        type=int,
        default=16,
        help='For open-set LLM baselines: prefilter candidates to top-k '
             'before passing to LLM (default: 16). Only applies when candidates > 16.',
    )
    parser.add_argument(
        '--retriever',
        choices=['tfidf', 'dense'],
        default='tfidf',
        help='Retriever used for open-set LLM prefilter (default: tfidf). '
             'Dense uses all-MiniLM-L6-v2 sentence embeddings.',
    )
    parser.add_argument(
        '--candidate-pool-size',
        type=int,
        default=0,
        help='Expand candidate pool to this size (0=use original, e.g., 16 or 32 for ablation)',
    )
    parser.add_argument(
        '--open-set',
        action='store_true',
        help='Open-set evaluation: use all POIs in same city/type as candidates',
    )
    parser.add_argument(
        '--open-set-sample',
        type=int,
        default=500,
        help='Number of dialogues to sample for open-set evaluation (default: 500)',
    )
    parser.add_argument(
        '--multi-reference',
        type=str,
        default=None,
        help='Path to multi-reference JSON for expanded gold evaluation',
    )
    parser.add_argument(
        '--judge-models',
        type=str,
        default=None,
        help='Comma-separated list of judge models for cross-model evaluation '
             '(e.g., "gemini/gemini-2.5-flash,gpt-4o,claude-sonnet-4-6")',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Estimate API calls, token cost, and wall time without running. '
             'Exits after printing the estimate.',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable debug-level logging',
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress info-level logging (show warnings only)',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else (logging.WARNING if args.quiet else logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    # 1. Load data
    dialogues = load_dataset(args.dataset)
    kb = load_knowledge_base(args.knowledge_base)
    poi_lookup = {p.business_id: p for p in kb}

    logger.info(f"Dataset: {len(dialogues)} dialogues, KB: {len(kb)} POIs")

    # Open-set mode: sample dialogues and use full city/type pool
    if args.open_set:
        import random as _rng
        _rng.seed(42)
        if len(dialogues) > args.open_set_sample:
            dialogues = _rng.sample(dialogues, args.open_set_sample)
        logger.info(f"Open-set mode: sampled {len(dialogues)} dialogues, using full city/type POI pools")
        # Set pool size to a large number to trigger full-pool expansion
        args.candidate_pool_size = 9999  # expand_candidate_pool will use all available
        # Configure prefilter for LLM baselines
        from trace_kit.crs import baselines as _bl
        _bl.OPEN_SET_PREFILTER_TOP_K = args.prefilter_top_k
        _bl.OPEN_SET_PREFILTER_RETRIEVER = args.retriever
        logger.info(f"  LLM baselines will prefilter to top-{args.prefilter_top_k} via {args.retriever.upper()}")

    # Load multi-reference data if provided
    multi_reference = None
    if args.multi_reference:
        with open(args.multi_reference, 'r') as f:
            multi_reference = json.load(f)
        logger.info(f"Loaded multi-reference data for {len(multi_reference)} dialogues")

    # Count evaluation points
    sys_turns = sum(1 for d in dialogues for t in d.turns if t.role == 'system')
    rec_turns = sum(
        1 for d in dialogues for t in d.turns
        if t.role == 'system' and t.action in RECOMMEND_ACTIONS
    )
    logger.info(f"Evaluation points: {sys_turns} system turns, {rec_turns} with recommendations")

    # Dry-run: estimate cost/time and exit before loading any LLM or heavy models
    if args.dry_run:
        judge_list = [m.strip() for m in args.judge_models.split(',')] if args.judge_models else None
        estimate_cost_and_exit(
            baselines=args.baselines,
            sys_turns=sys_turns,
            rec_turns=rec_turns,
            n_dialogues=len(dialogues),
            model=args.model,
            use_llm_judge=args.use_llm_judge,
            judge_models=judge_list,
            parallel_workers=args.parallel_workers,
        )
        return

    # 2. Initialize baselines
    # Pre-load shared SentenceTransformer encoder for memory_augmented baseline
    # so parallel workers don't each try to load the model to CUDA (meta tensor crash)
    _shared_encoder = None
    if 'memory_augmented' in args.baselines:
        from sentence_transformers import SentenceTransformer
        _shared_encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # Resolve API key for LLM baselines from --api-key-env (defaults to OPENAI_API_KEY).
    import os
    _llm_api_key = os.environ.get(args.api_key_env)
    _llm_api_base = os.environ.get('OPENAI_BASE_URL')

    baseline_map = {
        'popularity': PopularityBaseline,
        'tfidf': TFIDFBaseline,
        'llm_zeroshot': lambda: LLMZeroShotBaseline(model=args.model, api_key=_llm_api_key, api_base=_llm_api_base),
        'dst': lambda: DSTBaseline(model=args.model, api_key=_llm_api_key, api_base=_llm_api_base),
        'rag_citation': lambda: RAGCitationBaseline(model=args.model, api_key=_llm_api_key, api_base=_llm_api_base),
        'aspect': AspectRetrievalBaseline,
        'dense': DenseRetrievalBaseline,
        'spatial': SpatialBaseline,
        'hybrid_rrf': HybridRRFBaseline,
        'multi_review_synthesis': lambda: MultiReviewSynthesisBaseline(model=args.model, api_key=_llm_api_key, api_base=_llm_api_base),
        'memory_augmented': lambda: MemoryAugmentedBaseline(model=args.model, shared_encoder=_shared_encoder, api_key=_llm_api_key, api_base=_llm_api_base),
        'itinerary': ItineraryBaseline,
        'knowledge_enhanced': KnowledgeEnhancedBaseline,
        'persona_grounded': PersonaGroundedBaseline,
        'itinerary_llm': lambda: ItineraryLLMBaseline(model=args.model, api_key=_llm_api_key, api_base=_llm_api_base),
    }

    baselines: List[CRSBaseline] = []
    for name in args.baselines:
        factory = baseline_map[name]
        b = factory()
        # Pass full KB to itinerary baseline for cross-type discovery
        if hasattr(b, 'set_knowledge_base'):
            b.set_knowledge_base(kb)
        baselines.append(b)

    # Baselines that benefit from parallel API calls.
    # memory_augmented uses a shared pre-loaded SentenceTransformer encoder to avoid
    # per-thread CUDA meta tensor crashes.
    LLM_PARALLEL_SAFE = {'llm_zeroshot', 'dst', 'rag_citation', 'multi_review_synthesis', 'memory_augmented', 'itinerary_llm'}

    # 3. Evaluate each baseline (with per-dialogue incremental save)
    incremental_dir = Path(args.output).parent / f"{Path(args.output).stem}_incremental"
    incremental_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, List[Dict]] = {}

    for baseline in baselines:
        incremental_path = incremental_dir / f"{baseline.name}.jsonl"

        logger.info(f"--- Evaluating: {baseline.name} ---")

        pw = args.parallel_workers if baseline.name in LLM_PARALLEL_SAFE else 0
        factory = None
        if pw > 0:
            factory = baseline_map[baseline.name]

        results = evaluate_baseline(
            baseline, dialogues, poi_lookup,
            use_llm_judge=args.use_llm_judge,
            judge_model=args.model,
            parallel_workers=pw,
            baseline_factory=factory,
            kb=kb,
            use_entailment=args.use_entailment,
            candidate_pool_size=args.candidate_pool_size,
            multi_reference=multi_reference,
            incremental_path=incremental_path,
        )
        all_results[baseline.name] = results

        # Quick summary (system turns only)
        system_only = [r for r in results if r.get('role') != 'user']
        agg = aggregate_metrics(system_only)
        o = agg['overall']
        logger.info(
            f"  {baseline.name}: BLEU-4={o.get('bleu4', 0):.4f} "
            f"ROUGE-L={o.get('rouge_l', 0):.4f} "
            f"Grounding={o.get('grounding', 0):.3f} "
            f"Citation={o.get('citation_density', 0):.3f}"
        )

    # 3b. Cross-model LLM-as-Judge (if multiple judge models specified)
    if args.judge_models:
        judge_model_list = [m.strip() for m in args.judge_models.split(',')]
        logger.info(f"Cross-model judge evaluation with: {judge_model_list}")
        cross_judge_results = {}
        for jm in judge_model_list:
            logger.info(f"--- Judge: {jm} ---")
            jm_results = {}
            for baseline in baselines:
                pw = args.parallel_workers if baseline.name in LLM_PARALLEL_SAFE else 0
                factory = baseline_map[baseline.name] if pw > 0 else None
                results = evaluate_baseline(
                    baseline, dialogues, poi_lookup,
                    use_llm_judge=True,
                    judge_model=jm,
                    parallel_workers=pw,
                    baseline_factory=factory,
                    kb=kb,
                    candidate_pool_size=args.candidate_pool_size,
                    multi_reference=multi_reference,
                )
                jm_results[baseline.name] = results
            cross_judge_results[jm] = jm_results

        # Compute inter-judge agreement
        from trace_kit.crs.metrics import compute_inter_judge_agreement
        agreement = compute_inter_judge_agreement(cross_judge_results)
        # Save cross-judge results
        output_base = Path(args.output).stem
        output_dir = Path(args.output).parent
        cross_judge_path = output_dir / f"{output_base}_cross_judge.json"
        with open(cross_judge_path, 'w') as f:
            json.dump({'agreement': agreement, 'models': judge_model_list}, f, indent=2)
        logger.info(f"Cross-judge results saved to {cross_judge_path}")
        for k, v in agreement.items():
            logger.info(f"  {k}: {v:.4f}")

    # 4. Compute dataset statistics
    dataset_stats = compute_dataset_statistics(dialogues)

    # 5. Generate report (with dialogue-level metrics + optional stratification)
    report = generate_report(
        all_results, args.output,
        dataset_stats=dataset_stats,
        dialogues=dialogues,
        difficulty_stratify=args.difficulty_stratify,
    )
    print(report)

    # Clean up incremental files after successful report generation
    if incremental_dir.exists():
        import shutil
        shutil.rmtree(incremental_dir)
        logger.info(f"Incremental files removed (report generated successfully)")


if __name__ == '__main__':
    main()
