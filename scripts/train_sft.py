#!/usr/bin/env python
"""Code SFT training (Phase 2).

Reads a YAML config (configs/sft_full.yaml or configs/sft_lora.yaml) and trains
the base model on the chat-format SFT data with TRL's SFTTrainer.

Example:
    python scripts/prepare_sft_data.py --limit 20000
    python scripts/train_sft.py --config configs/sft_full.yaml

Debug (tiny run to check the pipeline end-to-end):
    python scripts/train_sft.py --config configs/sft_full.yaml \
        --limit 100 --max-steps 5
"""
import argparse
import inspect
import sys
from dataclasses import fields as dataclass_fields
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils import load_yaml  # noqa: E402

# Fallback chat template (ChatML) used only if the tokenizer lacks one.
# Qwen tokenizers already ship a chat template + the <|im_*|> tokens, so this
# rarely triggers; it's a safety net for a bare base tokenizer.
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def _filter_kwargs(cls, raw):
    """Keep only kwargs that are valid dataclass fields of `cls`."""
    valid = {f.name for f in dataclass_fields(cls)}
    kept = {k: v for k, v in raw.items() if k in valid}
    dropped = [k for k in raw if k not in valid]
    return kept, dropped


def build_model_and_tokenizer(cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    attn = cfg.get("attn_implementation", "sdpa")
    if attn == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except Exception:  # noqa: BLE001
            print("[warn] flash-attn not available; falling back to sdpa")
            attn = "sdpa"

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        print("[warn] tokenizer has no chat_template; installing ChatML fallback")
        tokenizer.chat_template = CHATML_TEMPLATE

    dtype = torch.bfloat16 if cfg.get("precision", "bf16") == "bf16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        torch_dtype=dtype,
        attn_implementation=attn,
        trust_remote_code=True,
    )
    model.config.use_cache = False  # required with gradient checkpointing
    return model, tokenizer


def build_sft_config(cfg):
    from trl import SFTConfig

    seq_len = cfg.get("seq_len", 4096)
    raw = dict(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=float(cfg["learning_rate"]),
        num_train_epochs=cfg.get("num_train_epochs", 1),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        weight_decay=cfg.get("weight_decay", 0.0),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        bf16=cfg.get("precision", "bf16") == "bf16",
        tf32=cfg.get("tf32", True),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        packing=cfg.get("packing", True),
        # set both names so the TRL max_seq_length->max_length rename is handled
        max_seq_length=seq_len,
        max_length=seq_len,
        optim=cfg.get("optim", "adamw_torch_fused"),
        logging_steps=cfg.get("logging_steps", 10),
        save_strategy=cfg.get("save_strategy", "epoch"),
        save_total_limit=cfg.get("save_total_limit", 2),
        dataloader_num_workers=cfg.get("dataloader_num_workers", 4),
        report_to=cfg.get("report_to", "none"),
        seed=cfg.get("seed", 42),
    )
    kept, dropped = _filter_kwargs(SFTConfig, raw)
    if dropped:
        print(f"[warn] SFTConfig ignored unsupported keys: {dropped}")
    return SFTConfig(**kept)


def build_peft_config(cfg):
    from peft import LoraConfig

    return LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--train-file", default=None, help="override config train_file")
    p.add_argument("--output-dir", default=None, help="override config output_dir")
    p.add_argument("--limit", type=int, default=0, help="debug: first N examples")
    p.add_argument("--max-steps", type=int, default=0, help="debug: cap train steps")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.train_file:
        cfg["train_file"] = args.train_file
    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    from datasets import load_dataset
    from trl import SFTTrainer

    is_lora = cfg.get("training_type", "full") == "lora"
    print(f"[config] {args.config} | training_type={cfg.get('training_type')}")

    model, tokenizer = build_model_and_tokenizer(cfg)

    train_path = cfg["train_file"]
    if not Path(train_path).exists():
        raise SystemExit(
            f"train_file not found: {train_path}. Run prepare_sft_data.py first."
        )
    dataset = load_dataset("json", data_files=train_path, split="train")
    if args.limit and args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    print(f"[data] {len(dataset)} examples from {train_path}")

    sft_config = build_sft_config(cfg)
    if args.max_steps and args.max_steps > 0:
        sft_config.max_steps = args.max_steps

    peft_config = None
    if is_lora:
        peft_config = build_peft_config(cfg)
        # Needed so gradients flow to inputs under gradient checkpointing + PEFT.
        if sft_config.gradient_checkpointing and hasattr(
            model, "enable_input_require_grads"
        ):
            model.enable_input_require_grads()

    # The tokenizer kwarg was renamed tokenizer -> processing_class in TRL.
    tok_key = (
        "processing_class"
        if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters
        else "tokenizer"
    )
    trainer_kwargs = {
        "model": model,
        "args": sft_config,
        "train_dataset": dataset,
        tok_key: tokenizer,
    }
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()

    out_dir = cfg["output_dir"]
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[done] saved model + tokenizer -> {out_dir}")


if __name__ == "__main__":
    main()
