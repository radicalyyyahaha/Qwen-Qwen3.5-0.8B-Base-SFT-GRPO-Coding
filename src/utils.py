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


def resolve_attn_implementation(attn):
    """Downgrade flash_attention_2 -> sdpa when flash-attn isn't installed."""
    if attn == "flash_attention_2":
        import importlib.util

        if importlib.util.find_spec("flash_attn") is None:
            print("[warn] flash-attn not installed; falling back to sdpa")
            return "sdpa"
    return attn


def load_lm(
    model_path,
    dtype=None,
    attn_implementation=None,
    device_map=None,
    trust_remote_code=True,
):
    """Load a causal LM for text generation / training.

    Handles multimodal checkpoints transparently. Qwen3.5 checkpoints declare
    `architectures=[Qwen3_5ForConditionalGeneration]` (model_type ``qwen3_5``, a
    text+vision model with a nested ``text_config``/``vision_config``).
    ``AutoModelForCausalLM`` cannot load that config, so we detect it and use
    ``AutoModelForImageTextToText`` instead. For text-only code work the vision
    tower is simply left unused. Plain dense LMs still load via
    ``AutoModelForCausalLM``.

    Returns ``(model, is_multimodal)``.
    """
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
    )

    config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    archs = getattr(config, "architectures", None) or []
    is_mm = hasattr(config, "vision_config") or any(
        ("ConditionalGeneration" in a) or ("ImageTextToText" in a) for a in archs
    )
    cls = AutoModelForImageTextToText if is_mm else AutoModelForCausalLM

    attn = resolve_attn_implementation(attn_implementation)

    def _load(attn_impl):
        kwargs = {"trust_remote_code": trust_remote_code}
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl
        if device_map is not None:
            kwargs["device_map"] = device_map
        return cls.from_pretrained(model_path, **kwargs)

    try:
        return _load(attn), is_mm
    except (ImportError, ValueError, RuntimeError) as exc:  # noqa: BLE001
        if attn and attn != "sdpa":
            print(
                f"[warn] attn_implementation={attn} failed "
                f"({type(exc).__name__}: {exc}); retrying with sdpa"
            )
            return _load("sdpa"), is_mm
        raise
