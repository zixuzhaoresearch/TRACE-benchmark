#!/usr/bin/env python3
"""Reconstruct the full TRACE knowledge base AND the dialogue text by hydrating
review content from the Yelp Open Dataset.

WHY THIS SCRIPT EXISTS
======================
The TRACE benchmark grounds recommendations in Yelp reviews. The Yelp Dataset
Terms of Use (Feb 2021) prohibit publicly redistributing review text (§4.A) and
derivative works (§4.H, §5). We therefore ship:

  * Knowledge base as POI stubs (review IDs only, no review text).
  * Dialogues with `[Q:N]` placeholders where the LLM quoted Yelp review text.
    Each placeholder's `quote_spans` metadata stores either an exact span
    (rid, start, end) or a lossless edit script that points back to corpus
    positions for any non-typographic characters.

This script does both hydrations in one pass:

  1. Hydrates POI reviews   → `knowledge_base/pois_full.jsonl.gz`
  2. Hydrates dialogue text → `data/{train,validation,test}_full.jsonl.gz`

The hydrated dialogues are byte-identical to what the LLM originally generated.

PREREQUISITES
=============
1. Visit  https://www.yelp.com/dataset  and accept the Terms of Use.
2. Download `yelp_dataset.tar` and extract `yelp_academic_dataset_review.json`.
3. Note the path to that file.

USAGE
=====
    python scripts/reconstruct_kb.py \\
        --yelp-reviews /path/to/yelp_academic_dataset_review.json

Defaults read/write from the standard release layout. Override with
`--pois`, `--review-index`, `--data-dir`, `--out-kb`, `--out-data-dir`.

Runtime: ~3 min on a laptop. Memory: ~3 GB peak (Yelp review file is ~5 GB).
"""
import argparse
import gzip
import json
import re
import sys
from pathlib import Path


PLACEHOLDER_RE = re.compile(r'\[Q:(\d+)\]')


def open_text(path: Path, mode: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def load_jsonl(path: Path):
    with open_text(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def apply_edits(review_span: str, edits: list, reviews: dict) -> str:
    """Reconstruct the LLM quote from a review span + edit script. Mirrors
    release/_build/mask_quotes.py:apply_edits — keep in sync."""
    result = review_span
    for edit in sorted(edits, key=lambda e: e[1], reverse=True):
        op = edit[0]
        pos = edit[1]
        if op == 'del':
            length = edit[2]
            result = result[:pos] + result[pos + length:]
        elif op == 'ins_p':
            text = edit[2]
            result = result[:pos] + text + result[pos:]
        elif op == 'rep_p':
            length, text = edit[2], edit[3]
            result = result[:pos] + text + result[pos + length:]
        elif op == 'ins_r':
            rid, c, d = edit[2], edit[3], edit[4]
            result = result[:pos] + reviews[rid][c:d] + result[pos:]
        elif op == 'rep_r':
            length, rid, c, d = edit[2], edit[3], edit[4], edit[5]
            result = result[:pos] + reviews[rid][c:d] + result[pos + length:]
        else:
            raise ValueError(f"Unknown edit op: {op}")
    return result


def hydrate_dialogue_text(text: str, quote_spans: list, reviews: dict) -> str:
    """Replace every [Q:N] placeholder with the corresponding quote text."""
    if not quote_spans:
        return text

    def repl(match):
        idx = int(match.group(1))
        spec = quote_spans[idx]
        rid = spec["rid"]
        s, e = spec["span"]
        review_text = reviews.get(rid)
        if review_text is None:
            raise KeyError(f"review {rid!r} not in hydrated KB")
        review_span = review_text[s:e]
        if spec["type"] == "exact":
            return review_span
        if spec["type"] == "lossless":
            return apply_edits(review_span, spec["edits"], reviews)
        raise ValueError(f"Unknown quote_span type: {spec['type']}")

    return PLACEHOLDER_RE.sub(repl, text)


def hydrate_kb(args, review_text):
    """Write knowledge_base/pois_full.jsonl.gz with hydrated review text."""
    print(f"\nWriting {args.out_kb} ...", flush=True)
    args.out_kb.parent.mkdir(parents=True, exist_ok=True)
    with open_text(args.out_kb, "w") as out:
        for poi in load_jsonl(args.pois):
            poi["reviews"] = []
            for rid in poi.get("review_ids", []):
                rt = review_text.get(rid)
                if rt is not None:
                    poi["reviews"].append({"review_id": rid, **rt})
            out.write(json.dumps(poi, ensure_ascii=False) + "\n")


def hydrate_dialogues(args, review_text):
    """Write data/{split}_full.jsonl.gz with [Q:N] placeholders replaced."""
    args.out_data_dir.mkdir(parents=True, exist_ok=True)
    # Plain text->text map for hydrate_dialogue_text
    review_text_only = {rid: rec["text"] for rid, rec in review_text.items()}

    for split in ("train", "validation", "test"):
        in_path = args.data_dir / f"{split}.jsonl.gz"
        if not in_path.exists():
            print(f"  skip {split}: {in_path} not found", flush=True)
            continue
        out_path = args.out_data_dir / f"{split}_full.jsonl.gz"
        n_dialogues = n_placeholders = 0
        with open_text(out_path, "w") as out:
            for d in load_jsonl(in_path):
                for t in d["turns"]:
                    qs = t.get("quote_spans")
                    if not qs:
                        continue
                    n_placeholders += len(qs)
                    t["text"] = hydrate_dialogue_text(t["text"], qs, review_text_only)
                    # Drop the masking metadata once hydrated; users see the
                    # original LLM text. quote_spans stays available if needed
                    # via the released (non-_full) files.
                n_dialogues += 1
                out.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"  {split}: hydrated {n_placeholders:,} placeholders across "
              f"{n_dialogues:,} dialogues → {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yelp-reviews", type=Path, required=True,
                    help="Path to yelp_academic_dataset_review.json (from the Yelp Open Dataset)")
    ap.add_argument("--pois", type=Path, default=Path("knowledge_base/pois.jsonl.gz"))
    ap.add_argument("--review-index", type=Path, default=Path("knowledge_base/review_index.jsonl.gz"))
    ap.add_argument("--data-dir", type=Path, default=Path("data"),
                    help="Directory holding train/validation/test.jsonl.gz")
    ap.add_argument("--out-kb", type=Path, default=Path("knowledge_base/pois_full.jsonl.gz"))
    ap.add_argument("--out-data-dir", type=Path, default=Path("data"),
                    help="Where to write {split}_full.jsonl.gz")
    ap.add_argument("--skip-dialogues", action="store_true",
                    help="Hydrate only the KB; leave dialogue placeholders in place")
    args = ap.parse_args()

    if not args.yelp_reviews.exists():
        sys.exit(f"ERROR: {args.yelp_reviews} not found. Download from https://www.yelp.com/dataset")

    needed_ids = {r["review_id"] for r in load_jsonl(args.review_index)}
    print(f"Need {len(needed_ids):,} reviews. Scanning {args.yelp_reviews} ...", flush=True)

    review_text = {}
    found = 0
    with open_text(args.yelp_reviews, "r") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = rec.get("review_id")
            if rid in needed_ids:
                review_text[rid] = {
                    "stars": rec.get("stars"),
                    "date": rec.get("date"),
                    "text": rec.get("text", ""),
                }
                found += 1
                if found % 5000 == 0:
                    print(f"  matched {found:,}/{len(needed_ids):,}", flush=True)

    missing = needed_ids - set(review_text)
    if missing:
        print(f"\nWARNING: {len(missing)} review IDs not found in your Yelp dump.", file=sys.stderr)
        print("  This usually means your Yelp dataset version is older/newer than ours.", file=sys.stderr)
        print("  Sample missing:", sorted(missing)[:5], file=sys.stderr)
        if not args.skip_dialogues:
            print("  Dialogue hydration may fail on placeholders that reference missing reviews.",
                  file=sys.stderr)
    else:
        print(f"All {len(needed_ids):,} reviews matched.")

    hydrate_kb(args, review_text)
    if not args.skip_dialogues:
        print(f"\nHydrating dialogues ({args.data_dir} → {args.out_data_dir}) ...", flush=True)
        hydrate_dialogues(args, review_text)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
