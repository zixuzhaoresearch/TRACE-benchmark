# Datasheet for TRACE

Following Gebru et al. (2021), *Datasheets for Datasets*.

## Motivation

**For what purpose was the dataset created?**
TRACE is a benchmark for **conversational recommendation with citation grounding** in the tourism domain. It tests whether systems can (a) recommend appropriate points of interest in a multi-turn dialogue, (b) ground their recommendations in verbatim review evidence, and (c) handle open-set candidate pools where the gold POI is not pre-filtered.

**Who created the dataset and on behalf of whom?**
*Withheld for anonymous review.* Identifying details will be added upon paper acceptance.

**Funding sources**
*Withheld for anonymous review.*

## Composition

**What do instances represent?**
Each instance is a multi-turn dialogue between a user (information seeker) and an assistant (recommender). User turns express preferences and constraints; system turns ask clarifying questions, recommend POIs (with quoted review evidence), or compare options.

**How many instances are there?**
- 10,000 dialogues total
- 8,001 train / 999 validation / 1,000 test (stratified by city × difficulty, seed 42)
- ~104,000 turns total (avg 10.5 per dialogue)
- 2,400 unique POIs / 34,208 review IDs in the knowledge base
- 2,174 POIs and 7,683 review IDs are referenced in dialogues

**Is the dataset a sample of a larger set?**
The KB is sampled from the Yelp Open Dataset: 8 cities × 3 POI types × 100 POIs = 2,400 POIs. Per POI we keep up to 15 reviews (≥5 minimum). KB is derived from snapshot v3 (preserved) plus a v5 top-up.

**Cities**: Indianapolis, Nashville, New Orleans, Philadelphia, Reno, Saint Louis, Tampa, Tucson — chosen for being mid-sized U.S. tourist destinations with strong Yelp coverage.

**What data does each instance consist of?**
See `README.md` *Dialogue schema* and *POI schema*.

**Is there a label or target?**
Yes — every system turn carries a gold action (`recommend`, `compare`, `ask_preference`, ...) and gold POI/review citations. Multi-reference alternatives (`acceptable_alternative_poi_ids`) are included where applicable.

**Are there missing values?**
- Some POIs lack `price_range` or `attributes` (Yelp source has these gaps).
- Some review hydration coverage may be <100% if a user's Yelp dump differs from ours; `reconstruct_kb.py` reports missing IDs.

**Are there errors or noise?**
Dialogues are LLM-generated. We applied automated validators (action transition rules, citation faithfulness, review-text presence in KB). Spot checks on 200 dialogues found no factual hallucinations beyond review-text paraphrasing in <2% of turns.

**Are relationships between instances explicit?**
Dialogues are independent. Within a dialogue, turn order is meaningful and `referenced_poi_ids` / `referenced_review_ids` cross-link to the KB.

**Are there recommended splits?**
Yes — `train` / `validation` / `test` (8001 / 999 / 1000), stratified by `(city, difficulty)` with seed 42.

**Are there any errors, sources of noise, or redundancies?**
Persona / travel-context combinations may repeat across dialogues; `dialogue_id` is unique.

**Self-contained or external dependencies?**
**External**: requires the Yelp Open Dataset for both review text (knowledge base) and dialogue text (LLM quotations of reviews are stored as `[Q:N]` placeholders). One-time hydration via `reconstruct_kb.py`. No other external deps.

**Placeholder design — are dialogues complete on download?**
Dialogue text on download contains `[Q:N]` placeholders wherever the LLM quoted Yelp review content (≈97.5% of detected double-quoted strings). Each placeholder is backed by `quote_spans` metadata pointing to a review ID and character span. After running `reconstruct_kb.py` against your local Yelp Open Dataset copy, dialogue text becomes byte-identical to what the LLM originally generated. The remaining ≈2.5% of LLM-quoted content (substantive paraphrases or non-quote artifacts) is left as plain text. The metadata format is purposefully designed to contain no Yelp word content — see `NOTICE.md` *Quote-masking design* for the full schema.

**Confidential / sensitive content?**
The Yelp dump is a public release; users in reviews may be identifiable by writing style, but Yelp removed direct PII before release. Our generated dialogues do not contain real personal data.

## Collection process

**How was the data acquired?**
Dialogues are generated end-to-end via DSPy-orchestrated LLM calls (Gemini 3 Flash Preview): persona sampling → scenario design → POI candidate selection → multi-turn generation → faithfulness validation. POI metadata and reviews come directly from the Yelp Open Dataset (sub-sampled, deterministically seeded).

**Generation timeline**: Initial generation 2026-04-08 (v4), with an open-set extension and additional difficulty tier added 2026-04-24 (current release).

**Crowdworkers / annotators?**
None — labels are programmatic (LLM action tagging + rule-based validation). A subset of 166 dialogues from a prior version was human-annotated for multi-reference acceptability; that annotation does not transfer to the current ID range.

## Preprocessing

- POIs filtered to ≥5 reviews and ≤15 reviews retained.
- Reviews truncated server-side by Yelp.
- Dialogues that fail action-grammar validation are dropped before merging shards.

## Uses

**Recommended tasks**
- Conversational recommendation with closed-set evaluation (8 candidates).
- Open-set evaluation (~50 candidates from same city × type).
- Citation faithfulness / grounding metrics.
- Dialogue-state tracking and multi-turn reasoning.

**Tasks the dataset should NOT be used for**
- Real recommendation production: review snapshots are static and may be outdated.
- Demographic or sentiment analysis of Yelp users — see Yelp ToS §4.C and §4.G.
- Any commercial product (Yelp ToS §4.B).

## Distribution

**How will the dataset be distributed?**
- Harvard Dataverse (DOI pending)
- HuggingFace Hub (handle pending)

Code is bundled with the release; the full source repo will be linked once de-anonymized.

## Maintenance

**Who is supporting/hosting/maintaining?**
*Withheld for anonymous review.* Errata channel will be added upon paper acceptance.

**Update plans**
- Multi-reference annotations for the current dialogue ID range (in progress).
- KB snapshot identifier publication (to bound coverage drift in `reconstruct_kb.py`).
