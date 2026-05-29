#!/usr/bin/env python
"""Code SFT training (Phase 2) for Qwen3.5-0.8B-Base.

Qwen3.5-0.8B-Base is a *multimodal hybrid* checkpoint
(architectures=[Qwen3_5ForConditionalGeneration], model_type ``qwen3_5``):

  - text decoder under ``model.language_model.*`` (a 3:1 stack of Gated DeltaNet
    "linear_attention" layers and "full_attention" layers),
  - vision tower under ``model.visual.*``,
  - a multi-token-prediction (MTP) head under ``mtp.*``,
  - tied input/output embeddings (no separate ``lm_head`` weight).

For code SFT we train the text decoder only: load the full model via
``AutoModelForImageTextToText``, freeze the vision tower + MTP head, and
fine-tune ``model.language_model`` on chat-format data with prompt-masked
(completion-only) loss.

Sequence *packing is disabled on purpose*: the DeltaNet layers carry a recurrent
state along the sequence, so packing several documents into one window would leak
state across document boundaries. One example == one (padded) sequence.

Example:
    python scripts/prepare_sft_data.py --limit 20000
    python scripts/train_sft.py --config configs/sft_full.yaml

Debug (tiny run to check the pipeline end-to-end):
    python scripts/train_sft.py --config configs/sft_full.yaml \
        --limit 100 --max-steps 5
"""
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils import load_lm, load_yaml  # noqa: E402

# Fallback chat template (ChatML) used only if the tokenizer lacks one. Qwen
# tokenizers already ship a chat template + the <|im_*|> tokens, so this rarely
# triggers; it's a safety net for a bare base tokenizer.
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def build_tokenizer(cfg):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.chat_template is None:
        print("[warn] tokenizer has no chat_template; installing ChatML fallback")
        tok.chat_template = CHATML_TEMPLATE
    return tok


def make_tokenize_fn(tokenizer, seq_len, mask_prompt):
    """Tokenize one {"messages": [...]} example into input_ids + masked labels.

    With ``mask_prompt`` the user turn (and the assistant header) is set to -100
    so loss is computed only over the assistant's response (completion-only).
    """

    def _tok(example):
        messages = example["messages"]
        # enable_thinking=False: Magicoder has no reasoning traces, so train the
        # plain (non-<think>) chat format. Unknown to non-Qwen templates -> ignored.
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        labels = list(input_ids)
        if mask_prompt and len(messages) >= 2:
            prompt_ids = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            # Mask the longest shared prefix (prompt + any chat scaffolding) and
            # learn from the first token that differs (the assistant response).
            # Robust to <think> scaffolding that may differ between the two renders.
            n = 0
            for a, b in zip(prompt_ids, input_ids):
                if a != b:
                    break
                n += 1
            for i in range(n):
                labels[i] = -100
        input_ids = input_ids[:seq_len]
        labels = labels[:seq_len]
        return {"input_ids": input_ids, "labels": labels, "length": len(input_ids)}

    return _tok


@dataclass
class PadCollator:
    """Right-pad input_ids (pad id), labels (-100), and build attention_mask."""

    pad_token_id: int
    pad_to_multiple_of: int = 8

    def __call__(self, features):
        import torch

        ids = [f["input_ids"] for f in features]
        labs = [f["labels"] for f in features]
        maxlen = max(len(x) for x in ids)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            maxlen = ((maxlen + m - 1) // m) * m
        b_ids, b_lab, b_mask = [], [], []
        for x, y in zip(ids, labs):
            pad = maxlen - len(x)
            b_ids.append(x + [self.pad_token_id] * pad)
            b_lab.append(y + [-100] * pad)
            b_mask.append([1] * len(x) + [0] * pad)
        return {
            "input_ids": torch.tensor(b_ids, dtype=torch.long),
            "labels": torch.tensor(b_lab, dtype=torch.long),
            "attention_mask": torch.tensor(b_mask, dtype=torch.long),
        }


def freeze_non_text(model, freeze_vision, freeze_mtp):
    """Freeze the vision tower (model.visual.*) and MTP head (mtp.*)."""
    trainable = frozen = 0
    for name, param in model.named_parameters():
        block = (freeze_vision and "visual" in name) or (
            freeze_mtp and (name.startswith("mtp.") or ".mtp." in name)
        )
        if block:
            param.requires_grad_(False)
            frozen += param.numel()
        else:
            trainable += param.numel()
    print(f"[freeze] trainable={trainable / 1e6:.1f}M frozen={frozen / 1e6:.1f}M")


def discover_lora_targets(model, scope):
    """Full module names of every nn.Linear under `scope` (the text decoder).

    Returning *full* names (not bare suffixes like ``q_proj``) scopes LoRA to the
    language model, so the frozen vision tower / MTP head are never adapted.
    """
    import torch.nn as nn

    targets = [
        name
        for name, module in model.named_modules()
        if name.startswith(scope) and isinstance(module, nn.Linear)
    ]
    if not targets:
        raise SystemExit(f"no nn.Linear modules found under scope '{scope}'")
    return targets


def build_training_args(cfg, output_dir):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=float(cfg["learning_rate"]),
        num_train_epochs=cfg.get("num_train_epochs", 3),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        weight_decay=cfg.get("weight_decay", 0.0),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        bf16=cfg.get("precision", "bf16") == "bf16",
        tf32=cfg.get("tf32", True),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=cfg.get("optim", "adamw_torch_fused"),
        logging_steps=cfg.get("logging_steps", 10),
        save_strategy=cfg.get("save_strategy", "epoch"),
        save_total_limit=cfg.get("save_total_limit", 2),
        dataloader_num_workers=cfg.get("dataloader_num_workers", 4),
        group_by_length=cfg.get("group_by_length", True),
        length_column_name="length",
        remove_unused_columns=False,
        report_to=cfg.get("report_to", "none"),
        seed=cfg.get("seed", 42),
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

    output_dir = cfg["output_dir"]
    is_lora = cfg.get("training_type", "full") == "lora"
    seq_len = cfg.get("seq_len", 4096)
    mask_prompt = cfg.get("mask_prompt", True)

    import torch
    from datasets import load_dataset
    from transformers import Trainer

    print(
        f"[config] {args.config} | training_type={cfg.get('training_type', 'full')} "
        f"| packing=disabled | mask_prompt={mask_prompt}"
    )

    tokenizer = build_tokenizer(cfg)

    dtype = torch.bfloat16 if cfg.get("precision", "bf16") == "bf16" else torch.float32
    model, is_mm = load_lm(
        cfg["model"],
        dtype=dtype,
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
    )
    model.config.use_cache = False  # required with gradient checkpointing
    print(f"[model] loaded {model.__class__.__name__} (multimodal={is_mm})")

    train_path = cfg["train_file"]
    if not Path(train_path).exists():
        raise SystemExit(
            f"train_file not found: {train_path}. Run prepare_sft_data.py first."
        )
    dataset = load_dataset("json", data_files=train_path, split="train")
    if args.limit and args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    tokenize = make_tokenize_fn(tokenizer, seq_len, mask_prompt)
    dataset = dataset.map(
        tokenize, remove_columns=dataset.column_names, desc="tokenize"
    )
    before = len(dataset)
    dataset = dataset.filter(
        lambda ex: any(label != -100 for label in ex["labels"]),
        desc="drop fully-masked",
    )
    print(
        f"[data] {len(dataset)} examples from {train_path} "
        f"(dropped {before - len(dataset)} with no learnable tokens)"
    )

    scope = cfg.get("language_model_scope", "model.language_model")
    if is_lora:
        from peft import LoraConfig, get_peft_model

        targets = cfg.get("target_modules", "auto")
        if targets in ("auto", ["auto"]):
            targets = discover_lora_targets(model, scope)
            print(
                f"[lora] auto-discovered {len(targets)} Linear modules under "
                f"'{scope}'"
            )
        # Needed so gradients flow to inputs under gradient checkpointing + PEFT.
        if cfg.get("gradient_checkpointing", True) and hasattr(
            model, "enable_input_require_grads"
        ):
            model.enable_input_require_grads()
        lora = LoraConfig(
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg.get("lora_dropout", 0.05),
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()
    elif is_mm:
        freeze_non_text(
            model, cfg.get("freeze_vision", True), cfg.get("freeze_mtp", True)
        )

    training_args = build_training_args(cfg, output_dir)
    if args.max_steps and args.max_steps > 0:
        training_args.max_steps = args.max_steps

    collator = PadCollator(pad_token_id=tokenizer.pad_token_id, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[done] saved model + tokenizer -> {output_dir}")


if __name__ == "__main__":
    main()
