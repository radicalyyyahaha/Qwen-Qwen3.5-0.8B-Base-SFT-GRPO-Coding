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
import json
import math
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

    def render_and_tokenize(messages, add_generation_prompt):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    def _tok(example):
        messages = example["messages"]
        # enable_thinking=False: Magicoder has no reasoning traces, so train the
        # plain (non-<think>) chat format. Unknown to non-Qwen templates -> ignored.
        input_ids = render_and_tokenize(messages, add_generation_prompt=False)
        labels = list(input_ids)
        if mask_prompt and len(messages) >= 2:
            prompt_ids = render_and_tokenize(
                messages[:-1], add_generation_prompt=True
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


def _jsonable(value):
    """Convert Trainer log values to JSON-safe primitives."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            item = value.item()
            if isinstance(item, (str, int, float, bool)) or item is None:
                return item
        except Exception:  # noqa: BLE001
            pass
    return str(value)


def extract_loss_points(log_history):
    points = []
    for row in log_history:
        if "loss" not in row or "step" not in row:
            continue
        try:
            step = int(row["step"])
            loss = float(row["loss"])
        except (TypeError, ValueError):
            continue
        if math.isfinite(loss):
            points.append((step, loss))
    return points


def write_loss_svg(points, path):
    """Write a dependency-free SVG loss curve. Returns True if written."""
    if not points:
        return False

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 900, 520
    left, right, top, bottom = 80, 30, 50, 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    x_min = min(step for step, _ in points)
    x_max = max(step for step, _ in points)
    y_min = min(loss for _, loss in points)
    y_max = max(loss for _, loss in points)
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    if y_min == y_max:
        pad = max(abs(y_min) * 0.05, 0.1)
        y_min -= pad
        y_max += pad
    else:
        pad = (y_max - y_min) * 0.08
        y_min -= pad
        y_max += pad

    def sx(step):
        return left + (step - x_min) / (x_max - x_min) * plot_w

    def sy(loss):
        return top + (y_max - loss) / (y_max - y_min) * plot_h

    polyline = " ".join(f"{sx(step):.2f},{sy(loss):.2f}" for step, loss in points)
    latest_step, latest_loss = points[-1]
    grid = []
    x_labels = []
    y_labels = []
    for i in range(6):
        x = left + plot_w * i / 5
        step = x_min + (x_max - x_min) * i / 5
        grid.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" '
            f'y2="{top + plot_h}" stroke="#e5e7eb"/>'
        )
        x_labels.append(
            f'<text x="{x:.2f}" y="{height - 28}" text-anchor="middle">'
            f"{step:.0f}</text>"
        )
        y = top + plot_h * i / 5
        loss = y_max - (y_max - y_min) * i / 5
        grid.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" '
            f'y2="{y:.2f}" stroke="#e5e7eb"/>'
        )
        y_labels.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end">'
            f"{loss:.4g}</text>"
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<style>
  text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #111827; font-size: 13px; }}
  .muted {{ fill: #6b7280; }}
</style>
<text x="{left}" y="30" font-size="22" font-weight="700">Training Loss</text>
<text x="{width - right}" y="30" text-anchor="end" class="muted">step {latest_step} | loss {latest_loss:.4f}</text>
{''.join(grid)}
<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.5"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.5"/>
<polyline points="{polyline}" fill="none" stroke="#2563eb" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
<circle cx="{sx(latest_step):.2f}" cy="{sy(latest_loss):.2f}" r="4" fill="#2563eb"/>
{''.join(x_labels)}
{''.join(y_labels)}
<text x="{left + plot_w / 2:.2f}" y="{height - 8}" text-anchor="middle" class="muted">global step</text>
<text x="22" y="{top + plot_h / 2:.2f}" transform="rotate(-90 22,{top + plot_h / 2:.2f})" text-anchor="middle" class="muted">loss</text>
</svg>
"""
    path.write_text(svg)
    return True


def write_loss_png(points, path):
    """Write a PNG loss curve when matplotlib is available."""
    if not points:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] matplotlib unavailable; skipped PNG loss plot ({exc})")
        return False

    steps = [step for step, _ in points]
    losses = [loss for _, loss in points]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    ax.plot(steps, losses, color="#2563eb", linewidth=1.8)
    ax.scatter([steps[-1]], [losses[-1]], color="#2563eb", s=18)
    ax.set_title("Training Loss")
    ax.set_xlabel("global step")
    ax.set_ylabel("loss")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def build_training_log_callback(output_dir):
    from transformers import TrainerCallback

    class TrainingLogCallback(TrainerCallback):
        def __init__(self, out_dir):
            self.output_dir = Path(out_dir)
            self.log_path = self.output_dir / "training_log.jsonl"
            self.svg_path = self.output_dir / "loss_curve.svg"
            self.png_path = self.output_dir / "loss_curve.png"

        @staticmethod
        def _is_main_process(state):
            return getattr(state, "is_world_process_zero", True)

        def on_train_begin(self, args, state, control, **kwargs):
            if not self._is_main_process(state):
                return
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("")
            print(f"[logs] training metrics -> {self.log_path}")
            print(f"[logs] loss curve -> {self.svg_path}")

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not self._is_main_process(state) or not logs:
                return
            record = {"step": int(getattr(state, "global_step", 0))}
            record.update({k: _jsonable(v) for k, v in logs.items()})
            with self.log_path.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            write_loss_svg(extract_loss_points(state.log_history), self.svg_path)

        def on_train_end(self, args, state, control, **kwargs):
            if not self._is_main_process(state):
                return
            points = extract_loss_points(state.log_history)
            if not points:
                print("[warn] no loss entries found; skipped loss plot")
                return
            write_loss_svg(points, self.svg_path)
            if write_loss_png(points, self.png_path):
                print(f"[logs] loss PNG -> {self.png_path}")

    return TrainingLogCallback(output_dir)


def build_training_args(cfg, output_dir):
    import inspect

    from transformers import TrainingArguments

    kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": cfg["per_device_train_batch_size"],
        "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
        "learning_rate": float(cfg["learning_rate"]),
        "num_train_epochs": cfg.get("num_train_epochs", 3),
        "lr_scheduler_type": cfg.get("lr_scheduler_type", "cosine"),
        "warmup_ratio": cfg.get("warmup_ratio", 0.03),
        "weight_decay": cfg.get("weight_decay", 0.0),
        "max_grad_norm": cfg.get("max_grad_norm", 1.0),
        "bf16": cfg.get("precision", "bf16") == "bf16",
        "tf32": cfg.get("tf32", True),
        "gradient_checkpointing": cfg.get("gradient_checkpointing", True),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": cfg.get("optim", "adamw_torch_fused"),
        "logging_steps": cfg.get("logging_steps", 10),
        "save_strategy": cfg.get("save_strategy", "epoch"),
        "save_total_limit": cfg.get("save_total_limit", 2),
        "dataloader_num_workers": cfg.get("dataloader_num_workers", 4),
        "group_by_length": cfg.get("group_by_length", True),
        "length_column_name": "length",
        "remove_unused_columns": False,
        "report_to": cfg.get("report_to", "none"),
        "seed": cfg.get("seed", 42),
    }
    params = inspect.signature(TrainingArguments.__init__).parameters
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        unsupported = sorted(set(kwargs) - set(params))
        if unsupported:
            print(
                "[warn] TrainingArguments does not support "
                f"{unsupported}; ignoring them"
            )
            kwargs = {k: v for k, v in kwargs.items() if k in params}
    return TrainingArguments(**kwargs)


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
        callbacks=[build_training_log_callback(output_dir)],
    )
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[done] saved model + tokenizer -> {output_dir}")


if __name__ == "__main__":
    main()
