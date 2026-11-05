---
license: cc-by-4.0
license_name: cc-by-4.0-and-yelp-tou
license_link: https://huggingface.co/datasets/taoist/TRACE/blob/main/NOTICE.md
pretty_name: "TRACE: Tourism Recommendation with Accountable Citation Evidence"
language:
  - en
task_categories:
  - text-generation
  - text-retrieval
tags:
  - conversational-recommendation
  - tourism
  - point-of-interest
  - citation-grounding
  - rejection-recovery
  - multi-turn-dialogue
  - benchmark
  - yelp
size_categories:
  - 10K<n<100K
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train.jsonl.gz
      - split: validation
        path: data/validation.jsonl.gz
      - split: test
        path: data/test.jsonl.gz
  - config_name: knowledge_base
    data_files:
      - split: pois
        path: knowledge_base/pois.jsonl.gz
      - split: review_index
        path: knowledge_base/review_index.jsonl.gz
---

# TRACE: Tourism Recommendation with Accountable Citation Evidence

A 10,000-dialogue conversational recommendation benchmark with **review-grounded citations** and **open-set evaluation**, spanning 8 U.S. cities and 2,400 points of interest (restaurants, attractions, hotels) sourced from the Yelp Open Dataset.

> **License notice.** Synthetic content authored by us (dialogues, code, annotations) is released under **CC BY 4.0**. Yelp-derived components (POI metadata, review IDs) are governed by the [Yelp Dataset User Agreement](https://www.yelp.com/dataset) (non-commercial, non-sublicensable). Verbatim Yelp review text is **not** redistributed; users hydrate locally from their own Yelp Open Dataset bundle. See [`NOTICE.md`](./NOTICE.md) for the full clause-by-clause split.

| | |
|---|---|
| **Dialogues** | 10,000 (train 8,001 / validation 999 / test 1,000) |
| **POIs** | 2,400 |
| **Reviews referenced** | 7,683 unique IDs (text hydrated locally — see below) |
| **Cities** | Indianapolis, Nashville, New Orleans, Philadelphia, Reno, Saint Louis, Tampa, Tucson |
| **Avg turns / dialogue** | 10.5 |
| **Difficulty tiers** | easy / medium / hard / expert |
| **Open-set support** | every dialogue ships with a candidate pool |

## Why review text isn't shipped

Review text comes from the Yelp Open Dataset, whose terms of use prohibit redistribution (§4.A) and sublicensing of derivative works (§4.H). We ship:

- **Dialogues** with `[Q:N]` placeholders wherever the LLM quoted a Yelp review verbatim or near-verbatim.
- **POIs** as stubs (no review text, only review IDs).
- Per-quote metadata (`quote_spans`) recording the source `review_id`, character offsets, and — for near-verbatim quotes — a small edit script. The metadata contains **only** review IDs, integer offsets, and whitelisted punctuation; it has no Yelp word content on its own.

You hydrate both the KB and the dialogues locally from your own Yelp Open Dataset download (one-time, ~3 minutes — see *Reconstruct the knowledge base* below). After hydration, dialogue text is byte-identical to what the LLM originally generated. See `NOTICE.md` for the licensing rationale.

## Quick start

```bash
pip install -e .          # installs trace_kit + minimal deps
python examples/load_dataset.py
```

This loads the splits and prints a sample dialogue. No Yelp download required for inspection.

## Layout

```
trace/
├── data/                              ← dialogues (our content, CC BY 4.0)
│   ├── train.jsonl.gz                   8,001 dialogues
│   ├── validation.jsonl.gz                999 dialogues
│   └── test.jsonl.gz                    1,000 dialogues
├── knowledge_base/                    ← POI references (stub form)
│   ├── pois.jsonl.gz                    2,400 POI stubs (no review text)
│   └── review_index.jsonl.gz           34,208 review-ID metadata rows
├── scripts/
│   ├── reconstruct_kb.py              ← run once to hydrate review text
│   └── evaluate_baselines.py          ← reproduce paper baselines
├── trace_kit/                         ← bundled Python package
│   ├── crs/                             baselines, metrics, models, prompts
│   └── datasets/yelp/                   YelpPOI dataclass
├── examples/load_dataset.py
├── dataset_info.json
├── DATASHEET.md
├── CITATION.bib
├── NOTICE.md
└── LICENSE
```

## Reconstruct the knowledge base and dialogues

1. Visit <https://www.yelp.com/dataset>, accept the Terms of Use, and download `yelp_dataset.tar`. Extract `yelp_academic_dataset_review.json`.
2. Run:
   ```bash
   python scripts/reconstruct_kb.py \
       --yelp-reviews /path/to/yelp_academic_dataset_review.json
   ```
   This writes:
   - `knowledge_base/pois_full.jsonl.gz` — POIs with hydrated review text
   - `data/{train,validation,test}_full.jsonl.gz` — dialogues with `[Q:N]` placeholders replaced by their original LLM-generated text (byte-identical to what we generated)

The script reports coverage. If your Yelp dump is current, all 34,208 reviews resolve. If you see <100% coverage, your dump may be a different snapshot — open an issue and we'll publish the snapshot identifier we used.

**Skip dialogue hydration** (KB-only): `--skip-dialogues`. Useful for quick KB inspection.

## Run baselines

After hydrating the KB **and** dialogues (`scripts/reconstruct_kb.py`):

```bash
# Non-LLM baselines (no API key needed)
python scripts/evaluate_baselines.py \
    --dataset data \
    --knowledge-base knowledge_base/pois_full.jsonl.gz \
    --baselines popularity tfidf dense aspect spatial hybrid_rrf \
    --output results/baselines_nonllm.md

# LLM baselines (set OPENAI_API_KEY + OPENAI_BASE_URL first)
python scripts/evaluate_baselines.py \
    --dataset data \
    --knowledge-base knowledge_base/pois_full.jsonl.gz \
    --baselines llm_zeroshot dst rag_citation \
    --model openai/gemini-3-flash-preview \
    --output results/baselines_llm.md

# Open-set evaluation (~50 candidates per recommendation turn)
python scripts/evaluate_baselines.py \
    --dataset data/test.jsonl.gz \
    --knowledge-base knowledge_base/pois_full.jsonl.gz \
    --baselines rag_citation \
    --open-set --open-set-sample 500 \
    --output results/openset.md
```

`--dataset` accepts a single file (`*.jsonl[.gz]` or legacy `*.json`) or a directory containing `train`/`validation`/`test` splits.

## Dialogue schema

Each line in `data/*.jsonl.gz` (placeholder form, before hydration):

```jsonc
{
  "dialogue_id": "crs_yelp_80001",
  "city": "Indianapolis",
  "scenario": {
    "persona_type": "family",
    "persona_description": "...",
    "primary_poi_type": "attraction",
    "travel_context": "..."
  },
  "turns": [
    {
      "turn_id": 3,
      "role": "system",
      "text": "A strong fit is Tarkington Park. One reviewer said [Q:0] and another mentioned [Q:1].",
      "action": "recommend",
      "referenced_poi_ids": ["..."],
      "referenced_review_ids": ["..."],
      "acceptable_alternative_poi_ids": ["..."],
      "quote_spans": [
        {"type": "exact",    "rid": "Qui4...", "span": [142, 198]},
        {"type": "lossless", "rid": "abc...",  "span": [55, 91],
         "edits": [["ins_p", 36, "..."]],     // restore "...." instead of "."
         "score": 96.97}
      ],
      "discovered_review_ids": ["abc...", "Qui4..."]   // rids found via masking
                                                       // (superset of referenced_review_ids)
    },
    ...
  ],
  "referenced_pois": [...],                    // non-stub POI metadata for the gold POIs
  "metadata": {
    "candidate_poi_ids": ["..."],              // closed-set evaluation pool (size 8)
    "open_set_pool_ids": ["..."],              // open-set evaluation pool (~50)
    "difficulty": "easy" | "medium" | "hard" | "expert",
    "model": "...",
    "generated_at": "..."
  }
}
```

After running `scripts/reconstruct_kb.py`, `data/*_full.jsonl.gz` contains the same schema with every `[Q:N]` placeholder substituted back into `text` (byte-identical to the originally-generated dialogue). `quote_spans` and `discovered_review_ids` are preserved for transparency / audit.

### Quote-span format

Each `quote_spans[N]` is one of:

```jsonc
// Exact verbatim substring of a Yelp review
{"type": "exact", "rid": "<review_id>", "span": [start_char, end_char]}

// Near-verbatim (≥90 partial_ratio): apply edits to the review span to recover original
{
  "type": "lossless",
  "rid": "<review_id>",
  "span": [start_char, end_char],
  "edits": [
    ["ins_p", pos, "<whitelisted_punct>"],     // insert punct/whitespace at pos
    ["del",   pos, len],                       // delete len chars at pos
    ["rep_p", pos, len, "<whitelisted_punct>"],// replace
    ["ins_r", pos, "<rid>", c, d],             // insert review[rid][c:d]
    ["rep_r", pos, len, "<rid>", c, d]         // replace with review[rid][c:d]
  ],
  "score": 96.97
}
```

`ins_p` / `rep_p` only carry whitelisted punctuation (no alphanumeric content). `ins_r` / `rep_r` reference positions in any review, so reconstructed text is always taken from the user's local Yelp dump.

## POI schema (after hydration)

```jsonc
{
  "business_id": "...",         // Yelp identifier (factual key)
  "name": "...",
  "city": "...", "state": "...",
  "poi_type": "restaurant" | "attraction" | "hotel",
  "address": "...",
  "stars": 4.0,
  "review_count": 242,
  "categories": [...],
  "latitude": 39.94, "longitude": -75.17,
  "price_range": "$$",
  "attributes": {...},
  "reviews": [                  // present after running reconstruct_kb.py
    {"review_id": "...", "stars": 5, "date": "...", "text": "..."},
    ...
  ]
}
```

## Splits

Stratified by `(city, difficulty)` with seed `42`. Per-bucket ordering is `sha256(dialogue_id|seed)`. Each split has the same difficulty/city distribution as the full dataset (within rounding). The split is fully reproducible from `dialogue_id` alone — see `release/_build/prepare_release.py` in the source repo.

## License

- **Our content** (dialogues, scenarios, personas, annotations, code) — **CC BY 4.0** (`LICENSE`).
- **Yelp-derived content** (POI metadata factual fields, review IDs) — governed by the [Yelp Dataset Terms of Use](https://www.yelp.com/dataset). See `NOTICE.md` for the breakdown.
- **Code in `trace_kit/` and `scripts/`** — also covered by `LICENSE` (CC BY 4.0; you may apply a permissive software license to derivative code distributions if preferred).

## Citation

If you use TRACE, please cite both our paper and the Yelp Open Dataset:

```bibtex
@inproceedings{trace2026,
  title     = {TRACE: Tourism Recommendation with Accountable Citation Evidence},
  author    = {Zhao, Zixu and Wang, Sijin and Hou, Yu and Xu, Yuanyuan and Sheng, Yufan and Xie, Xike and Zhang, Wenjie and Shin, Won-Yong and Cao, Xin},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track},
  year      = {2026},
  note      = {Dataset: \url{https://huggingface.co/datasets/taoist/TRACE}; code: \url{https://github.com/zixuzhaoresearch/TRACE-benchmark}}
}

@misc{yelp_open_dataset,
  title  = {Yelp Open Dataset},
  author = {{Yelp Inc.}},
  url    = {https://www.yelp.com/dataset}
}
```

Full BibTeX in `CITATION.bib`.

## Issues / contact

See the project repository linked in `dataset_info.json`. If you encounter coverage gaps in `reconstruct_kb.py`, please report the missing review IDs.
