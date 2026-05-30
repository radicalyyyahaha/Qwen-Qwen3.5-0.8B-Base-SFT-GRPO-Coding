"""Test-based (verifiable) reward for GRPO: reward = fraction of unit tests passed.

Built for TRL's ``GRPOTrainer``. The trainer calls the reward with the batch of
``completions`` plus any extra dataset columns as keyword args — here the
``tests`` column (a JSON string per example) produced by
``scripts/prepare_grpo_data.py``. We extract the code from each completion, run
it against its tests in the sandbox, and return one reward per completion.

There is no reward model: the unit tests *are* the reward (RLVR).
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sandbox import check_io  # noqa: E402
from src.utils import extract_code  # noqa: E402


def _completion_text(completion):
    """Normalize a TRL completion (str or list-of-messages) to plain text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):  # conversational: [{"role","content"}, ...]
        for msg in reversed(completion):
            if isinstance(msg, dict) and msg.get("content"):
                return msg["content"]
        return ""
    if isinstance(completion, dict):
        return completion.get("content", "")
    return str(completion)


def _load_tests(t):
    if isinstance(t, dict):
        return t
    if isinstance(t, str):
        try:
            return json.loads(t)
        except Exception:  # noqa: BLE001
            return None
    return None


def make_code_reward(timeout=6.0, mem_mb=1024, max_cases=15, num_workers=8):
    """Build a GRPO reward function (a closure capturing the sandbox settings).

    Use threads (not processes) for the batch: the heavy work happens inside the
    per-candidate subprocess, so threads just wait on I/O — no CUDA-fork issues.
    """

    def code_reward(completions=None, tests=None, **kwargs):
        completions = completions or []
        if tests is None:
            print("[reward][warn] no `tests` column passed; returning zeros")
            return [0.0] * len(completions)
        codes = [extract_code(_completion_text(c)) for c in completions]

        def _score(pair):
            code, raw_tests = pair
            parsed = _load_tests(raw_tests)
            if not code or parsed is None:
                return 0.0
            try:
                return float(
                    check_io(
                        code,
                        parsed,
                        timeout=timeout,
                        mem_mb=mem_mb,
                        max_cases=max_cases,
                    )
                )
            except Exception:  # noqa: BLE001
                return 0.0

        if not codes:
            return []
        workers = max(1, min(num_workers, len(codes)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rewards = list(pool.map(_score, zip(codes, tests)))
        return rewards

    code_reward.__name__ = "code_reward"
    return code_reward
