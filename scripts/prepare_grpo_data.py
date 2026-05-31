#!/usr/bin/env python
"""Prepare the GRPO/RLVR dataset (Phase 4) from BAAI/TACO.

TACO is a competitive-programming dataset: each problem has a statement and unit
tests (``input_output``, an APPS-style JSON with ``inputs`` / ``outputs`` and an
optional ``fn_name`` for call-based problems). We convert it to GRPO format:

    {"prompt": [{"role": "user", "content": <instruction + problem>}],
     "tests": "<json string: {inputs, outputs, fn_name?}>"}

``prompt`` is conversational so GRPOTrainer applies the chat template; ``tests``
rides along as an extra column and is handed to the reward (src/reward.py).

By default we keep only **stdin/stdout** problems (the clean, reliably-checkable
majority) and filter to EASY difficulty, so a small model solves enough to get a
non-zero reward signal. Use ``--difficulty all`` / ``--include-fn-name`` to widen.

TACO is large (tens of GB with all solutions). Behind the GreatFirewall, set
``HF_ENDPOINT=https://hf-mirror.com`` first, and consider ``--streaming`` so only
the consumed shards are downloaded.

Example:
    python scripts/prepare_grpo_data.py --limit 3000 --difficulty EASY \
        --output data/grpo/taco_grpo.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils import write_jsonl  # noqa: E402

STDIN_INSTRUCTION = (
    "Solve the following competitive programming problem in Python.\n"
    "Read the input from standard input (stdin) and print the answer to "
    "standard output (stdout).\n"
    "Return ONLY the complete program in a single ```python code block, with no "
    "explanation.\n\n"
)
CALL_INSTRUCTION = (
    "Solve the following problem in Python by implementing the required "
    "function.\n"
    "Return ONLY the complete solution in a single ```python code block, with no "
    "explanation.\n\n"
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="BAAI/TACO")
    p.add_argument("--split", default="train")
    p.add_argument("--output", default="data/grpo/taco_grpo.jsonl")
    p.add_argument("--limit", type=int, default=3000, help="0 = use all")
    p.add_argument(
        "--difficulty",
        default="EASY",
        help="comma-separated TACO difficulties to keep, or 'all' "
        "(EASY, MEDIUM, MEDIUM_HARD, HARD, VERY_HARD)",
    )
    p.add_argument("--max-tests", type=int, default=20, help="cap tests per problem")
    p.add_argument(
        "--min-tests", type=int, default=1, help="drop problems with fewer tests"
    )
    p.add_argument(
        "--max-prompt-chars",
        type=int,
        default=6000,
        help="drop problems whose statement is longer than this",
    )
    p.add_argument(
        "--include-fn-name",
        action="store_true",
        help="also keep call-based (fn_name) problems (default: stdin/stdout only)",
    )
    p.add_argument(
        "--streaming",
        action="store_true",
        help="stream the dataset (only download consumed shards; good for TACO)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-shuffle", action="store_true")
    return p.parse_args()


def parse_io(raw):
    """Parse TACO ``input_output`` into (inputs, outputs, fn_name) or None."""
    if not raw:
        return None
    try:
        io = json.loads(raw)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(io, dict):
        return None
    inputs = io.get("inputs") or []
    outputs = io.get("outputs") or []
    if not inputs or not outputs:
        return None
    return inputs, outputs, io.get("fn_name")


def main():
    args = parse_args()
    from datasets import load_dataset

    diffs = None
    if args.difficulty and args.difficulty.lower() != "all":
        diffs = {d.strip().upper() for d in args.difficulty.split(",") if d.strip()}

    print(f"[load] {args.dataset} split={args.split} streaming={args.streaming}")
    try:
        try:
            ds = load_dataset(
                args.dataset,
                split=args.split,
                trust_remote_code=True,
                streaming=args.streaming,
            )
        except TypeError:
            # datasets>=4 dropped the trust_remote_code kwarg
            ds = load_dataset(
                args.dataset, split=args.split, streaming=args.streaming
            )
    except (RuntimeError, ValueError) as exc:
        if "script" in str(exc).lower():
            raise SystemExit(
                f"{exc}\n\n"
                "TACO ships a dataset loading script (TACO.py), and datasets>=4.0 "
                "removed script support. Fixes:\n"
                "  (a) pip install 'datasets<4.0'   # restores script loading, "
                "then re-run this command unchanged\n"
                "  (b) point --dataset at a parquet / no-script copy of TACO"
            )
        raise
    if not args.no_shuffle:
        ds = ds.shuffle(seed=args.seed)

    rows = []
    stats = {"seen": 0, "no_tests": 0, "fn_skipped": 0, "too_long": 0, "diff": 0}
    for ex in ds:
        stats["seen"] += 1
        if diffs is not None and (ex.get("difficulty") or "").upper() not in diffs:
            stats["diff"] += 1
            continue
        question = (ex.get("question") or "").strip()
        if not question or len(question) > args.max_prompt_chars:
            stats["too_long"] += 1
            continue
        parsed = parse_io(ex.get("input_output"))
        if parsed is None:
            stats["no_tests"] += 1
            continue
        inputs, outputs, fn_name = parsed
        if fn_name and not args.include_fn_name:
            stats["fn_skipped"] += 1
            continue
        if min(len(inputs), len(outputs)) < args.min_tests:
            stats["no_tests"] += 1
            continue
        inputs = inputs[: args.max_tests]
        outputs = outputs[: args.max_tests]

        if fn_name:
            content = CALL_INSTRUCTION + question
            starter = (ex.get("starter_code") or "").strip()
            if starter:
                content += "\n\n```python\n" + starter + "\n```"
            tests = {"inputs": inputs, "outputs": outputs, "fn_name": fn_name}
        else:
            content = STDIN_INSTRUCTION + question
            tests = {"inputs": inputs, "outputs": outputs}

        rows.append(
            {
                "prompt": [{"role": "user", "content": content}],
                "tests": json.dumps(tests, ensure_ascii=False),
            }
        )
        if args.limit and args.limit > 0 and len(rows) >= args.limit:
            break

    if not rows:
        raise SystemExit(
            "No usable problems after filtering. Loosen --difficulty / add "
            "--include-fn-name, or raise --max-prompt-chars."
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, rows)
    print(
        f"[write] {len(rows)} problems -> {out_path} | seen={stats['seen']} "
        f"dropped: diff={stats['diff']} no_tests={stats['no_tests']} "
        f"fn={stats['fn_skipped']} too_long={stats['too_long']}"
    )


if __name__ == "__main__":
    main()
