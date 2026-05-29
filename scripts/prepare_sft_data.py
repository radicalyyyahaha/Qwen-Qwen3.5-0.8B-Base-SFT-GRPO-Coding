#!/usr/bin/env python
"""Prepare the SFT dataset (Phase 2).

Downloads Magicoder-OSS-Instruct and converts each example into chat format:

    {"messages": [{"role": "user", "content": <problem>},
                  {"role": "assistant", "content": <solution>}]}

Example:
    python scripts/prepare_sft_data.py --limit 20000 \
        --output data/sft/magicoder_chat.jsonl
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils import write_jsonl  # noqa: E402

# Magicoder uses problem/solution; accept common aliases too.
INSTRUCTION_KEYS = ["problem", "instruction", "prompt", "input", "question"]
RESPONSE_KEYS = ["solution", "response", "output", "completion", "answer"]


def pick(example, keys):
    for key in keys:
        value = example.get(key)
        if value:
            return value
    return None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="ise-uiuc/Magicoder-OSS-Instruct-75K")
    p.add_argument("--split", default="train")
    p.add_argument("--output", default="data/sft/magicoder_chat.jsonl")
    p.add_argument("--limit", type=int, default=20000, help="0 = use all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-shuffle",
        action="store_true",
        help="keep dataset order instead of shuffling before --limit",
    )
    return p.parse_args()


def main():
    args = parse_args()
    from datasets import load_dataset

    print(f"[load] {args.dataset} split={args.split}")
    ds = load_dataset(args.dataset, split=args.split)
    if not args.no_shuffle:
        ds = ds.shuffle(seed=args.seed)
    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"[load] using {len(ds)} examples; columns={ds.column_names}")

    rows = []
    skipped = 0
    for example in ds:
        instruction = pick(example, INSTRUCTION_KEYS)
        response = pick(example, RESPONSE_KEYS)
        if not instruction or not response:
            skipped += 1
            continue
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": response},
                ]
            }
        )

    if not rows:
        raise SystemExit(
            f"No usable rows. Looked for instruction in {INSTRUCTION_KEYS} and "
            f"response in {RESPONSE_KEYS}, but columns are {ds.column_names}."
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, rows)
    print(f"[write] {len(rows)} examples -> {out_path} (skipped {skipped})")


if __name__ == "__main__":
    main()
