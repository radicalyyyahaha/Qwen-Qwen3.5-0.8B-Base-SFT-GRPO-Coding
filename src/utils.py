"""Shared helpers used across data prep, eval, and training scripts."""
import json
import re
from datetime import datetime, timezone


def load_yaml(path):
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f)


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text):
    """Pull the first fenced code block from a chat response.

    Falls back to the raw (stripped) text when no fence is present.
    """
    blocks = _CODE_BLOCK_RE.findall(text)
    if blocks:
        return blocks[0].strip()
    return text.strip()


def now_iso():
    return datetime.now(timezone.utc).isoformat()
