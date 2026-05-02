#!/usr/bin/env python3
"""Quickstart: load the TRACE dataset and inspect a dialogue.

This example needs only `data/`; review text is not required.
"""
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_split(name: str):
    path = ROOT / "data" / f"{name}.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    train = load_split("train")
    val = load_split("validation")
    test = load_split("test")
    print(f"train={len(train)}  validation={len(val)}  test={len(test)}")

    d = test[0]
    print(f"\n=== {d['dialogue_id']} | city={d['city']} | difficulty={d['metadata']['difficulty']} ===")
    print(f"Persona: {d['scenario']['persona_type']} — {d['scenario']['persona_description'][:120]}...")
    print(f"Travel: {d['scenario']['travel_context'][:120]}...")
    print(f"Recommended POIs: {[p['name'] for p in d['referenced_pois']]}")
    print(f"\nFirst 4 turns:")
    for t in d["turns"][:4]:
        marker = "U" if t["role"] == "user" else "S"
        print(f"  [{marker} #{t['turn_id']} | {t.get('action') or '-'}] {t['text'][:140]}")


if __name__ == "__main__":
    main()
