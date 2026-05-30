#!/usr/bin/env python
"""GRPO / RLVR training (Phase 4) for the SFT'd Qwen3.5-0.8B model.

Starts from the Phase-2 SFT checkpoint and optimizes it with GRPO using a
*verifiable* reward: generated programs are run against TACO unit tests and
rewarded by the fraction of tests they pass (src/reward.py + src/sandbox.py).
No reward model.

Like SFT, this respects the multimodal hybrid architecture: the model is loaded
with ``AutoModelForImageTextToText``; the vision tower + MTP head are frozen
(full FT) or simply left untouched (LoRA scoped to the text decoder).

Generation backend: HF ``generate`` by default (``use_vllm: false``). vLLM is
much faster but its engine-core init has been unreliable for this ``qwen3_5``
hybrid checkpoint, so we don't depend on it; set ``use_vllm: true`` to try it.

Example:
    python scripts/prepare_grpo_data.py --limit 3000 --difficulty EASY
    python scripts/train_grpo.py --config configs/grpo.yaml

Debug (tiny, end-to-end):
    python scripts/train_grpo.py --config configs/grpo.yaml --limit 32 --max-steps 3
"""
import argparse
import inspect
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.reward import make_code_reward  # noqa: E402
from src.utils import load_lm, load_yaml  # noqa: E402

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


def freeze_non_text(model, freeze_vision=True, freeze_mtp=True):
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
    """Full module names of every nn.Linear under `scope` (the text decoder)."""
    import torch.nn as nn

    targets = [
        name
        for name, module in model.named_modules()
        if name.startswith(scope) and isinstance(module, nn.Linear)
    ]
    if not targets:
        raise SystemExit(f"no nn.Linear modules found under scope '{scope}'")
    return targets


def build_grpo_config(cfg, output_dir):
    """Build a GRPOConfig, keeping only keys this TRL version actually accepts.

    TRL's GRPO API churns across releases, so we assemble a candidate dict and
    filter it against ``GRPOConfig.__init__``'s signature instead of hard-coding
    a parameter set that might not exist.
    """
    from trl import GRPOConfig

    candidate = {
        "output_dir": output_dir,
        "per_device_train_batch_size": cfg.get("per_device_train_batch_size", 8),
        "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps", 4),
        "learning_rate": float(cfg.get("learning_rate", 1e-6)),
        "num_train_epochs": cfg.get("num_train_epochs", 1),
        "lr_scheduler_type": cfg.get("lr_scheduler_type", "constant_with_warmup"),
        "warmup_ratio": cfg.get("warmup_ratio", 0.03),
        "weight_decay": cfg.get("weight_decay", 0.0),
        "max_grad_norm": cfg.get("max_grad_norm", 1.0),
        "bf16": cfg.get("precision", "bf16") == "bf16",
        "tf32": cfg.get("tf32", True),
        "gradient_checkpointing": cfg.get("gradient_checkpointing", True),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": cfg.get("optim", "adamw_torch_fused"),
        "logging_steps": cfg.get("logging_steps", 1),
        "save_strategy": cfg.get("save_strategy", "steps"),
        "save_steps": cfg.get("save_steps", 100),
        "save_total_limit": cfg.get("save_total_limit", 2),
        "report_to": cfg.get("report_to", "none"),
        "seed": cfg.get("seed", 42),
        "dataloader_num_workers": cfg.get("dataloader_num_workers", 2),
        # --- GRPO-specific ---
        "num_generations": cfg.get("num_generations", 8),
        "max_prompt_length": cfg.get("max_prompt_length", 1024),
        "max_completion_length": cfg.get("max_completion_length", 1024),
        "temperature": cfg.get("temperature", 1.0),
        "beta": cfg.get("beta", 0.04),
        "use_vllm": cfg.get("use_vllm", False),
        "log_completions": cfg.get("log_completions", True),
    }
    max_steps = cfg.get("max_steps", 0)
    if max_steps and max_steps > 0:
        candidate["max_steps"] = max_steps

    valid = set(inspect.signature(GRPOConfig.__init__).parameters) - {"self"}
    applied = {k: v for k, v in candidate.items() if k in valid and v is not None}
    dropped = sorted(set(candidate) - set(applied))
    if dropped:
        print(f"[grpo] config keys not in this TRL's GRPOConfig, ignored: {dropped}")
    return GRPOConfig(**applied)


# --- training-process logging: full JSONL history + curve plots ----------------
# For GRPO the headline curve is *reward* (it should trend up); loss is often
# near-zero and far less informative. We log everything to training_log.jsonl and
# render a curve per metric. Each entry is (candidate-keys, file-stem, title,
# y-label, color) — the first key present in the log history is used, so this is
# robust to TRL renaming reward keys across versions.
GRPO_METRICS = [
    (["reward", "rewards/code_reward/mean", "rewards/mean"], "reward_curve",
     "GRPO Mean Reward", "reward", "#16a34a"),
    (["loss"], "loss_curve", "GRPO Loss", "loss", "#2563eb"),
    (["kl"], "kl_curve", "GRPO KL", "kl", "#db2777"),
    (["reward_std"], "reward_std_curve", "GRPO Reward Std", "reward std", "#f59e0b"),
]


def _jsonable(value):
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


def extract_points(log_history, keys):
    """Return [(step, value)] for the candidate key best populated in the history."""
    if isinstance(keys, str):
        keys = [keys]
    key = max(keys, key=lambda k: sum(1 for r in log_history if k in r), default=None)
    if key is None or not any(key in r for r in log_history):
        return []
    points = []
    for row in log_history:
        if key not in row:
            continue
        step = row.get("step", row.get("global_step"))
        try:
            step, val = int(step), float(row[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            points.append((step, val))
    return points


def write_curve_svg(points, path, title, ylabel, color="#2563eb"):
    """Write a dependency-free SVG line chart. Returns True if written."""
    if not points:
        return False
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 900, 520
    left, right, top, bottom = 80, 30, 50, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    xs, ys = [s for s, _ in points], [v for _, v in points]
    x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
    if x_min == x_max:
        x_min, x_max = x_min - 1, x_max + 1
    if y_min == y_max:
        pad = max(abs(y_min) * 0.05, 0.1)
    else:
        pad = (y_max - y_min) * 0.08
    y_min, y_max = y_min - pad, y_max + pad

    def sx(s):
        return left + (s - x_min) / (x_max - x_min) * plot_w

    def sy(v):
        return top + (y_max - v) / (y_max - y_min) * plot_h

    polyline = " ".join(f"{sx(s):.2f},{sy(v):.2f}" for s, v in points)
    last_s, last_v = points[-1]
    grid, x_labels, y_labels = [], [], []
    for i in range(6):
        x = left + plot_w * i / 5
        s = x_min + (x_max - x_min) * i / 5
        grid.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" '
            f'y2="{top + plot_h}" stroke="#e5e7eb"/>'
        )
        x_labels.append(
            f'<text x="{x:.2f}" y="{height - 28}" text-anchor="middle">{s:.0f}</text>'
        )
        y = top + plot_h * i / 5
        v = y_max - (y_max - y_min) * i / 5
        grid.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" '
            f'y2="{y:.2f}" stroke="#e5e7eb"/>'
        )
        y_labels.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end">{v:.4g}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<style>
  text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #111827; font-size: 13px; }}
  .muted {{ fill: #6b7280; }}
</style>
<text x="{left}" y="30" font-size="22" font-weight="700">{title}</text>
<text x="{width - right}" y="30" text-anchor="end" class="muted">step {last_s} | {ylabel} {last_v:.4g}</text>
{''.join(grid)}
<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.5"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.5"/>
<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
<circle cx="{sx(last_s):.2f}" cy="{sy(last_v):.2f}" r="4" fill="{color}"/>
{''.join(x_labels)}
{''.join(y_labels)}
<text x="{left + plot_w / 2:.2f}" y="{height - 8}" text-anchor="middle" class="muted">global step</text>
<text x="22" y="{top + plot_h / 2:.2f}" transform="rotate(-90 22,{top + plot_h / 2:.2f})" text-anchor="middle" class="muted">{ylabel}</text>
</svg>
"""
    path.write_text(svg)
    return True


def write_curve_png(points, path, title, ylabel, color="#2563eb"):
    """Write a PNG line chart when matplotlib is available."""
    if not points:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] matplotlib unavailable; skipped PNG plot ({exc})")
        return False
    xs, ys = [s for s, _ in points], [v for _, v in points]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    ax.plot(xs, ys, color=color, linewidth=1.8)
    ax.scatter([xs[-1]], [ys[-1]], color=color, s=18)
    ax.set_title(title)
    ax.set_xlabel("global step")
    ax.set_ylabel(ylabel)
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

        @staticmethod
        def _is_main(state):
            return getattr(state, "is_world_process_zero", True)

        def _render_svgs(self, log_history):
            for keys, stem, title, ylabel, color in GRPO_METRICS:
                pts = extract_points(log_history, keys)
                if pts:
                    write_curve_svg(
                        pts, self.output_dir / f"{stem}.svg", title, ylabel, color
                    )

        def on_train_begin(self, args, state, control, **kwargs):
            if not self._is_main(state):
                return
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("")
            print(f"[logs] training metrics -> {self.log_path}")
            print(f"[logs] curves (reward/loss/kl) -> {self.output_dir}/*_curve.svg")

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not self._is_main(state) or not logs:
                return
            record = {"step": int(getattr(state, "global_step", 0))}
            record.update({k: _jsonable(v) for k, v in logs.items()})
            with self.log_path.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._render_svgs(state.log_history)

        def on_train_end(self, args, state, control, **kwargs):
            if not self._is_main(state):
                return
            wrote_any = False
            for keys, stem, title, ylabel, color in GRPO_METRICS:
                pts = extract_points(state.log_history, keys)
                if not pts:
                    continue
                write_curve_svg(
                    pts, self.output_dir / f"{stem}.svg", title, ylabel, color
                )
                if write_curve_png(
                    pts, self.output_dir / f"{stem}.png", title, ylabel, color
                ):
                    wrote_any = True
            if not wrote_any:
                print("[warn] no metric points logged; skipped curve plots")

    return TrainingLogCallback(output_dir)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--train-file", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--limit", type=int, default=0, help="debug: first N problems")
    p.add_argument("--max-steps", type=int, default=0, help="debug: cap train steps")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.train_file:
        cfg["train_file"] = args.train_file
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.max_steps and args.max_steps > 0:
        cfg["max_steps"] = args.max_steps

    output_dir = cfg["output_dir"]
    is_lora = cfg.get("training_type", "full") == "lora"

    import torch
    from datasets import load_dataset
    from trl import GRPOTrainer

    print(
        f"[config] {args.config} | training_type={cfg.get('training_type', 'full')} "
        f"| use_vllm={cfg.get('use_vllm', False)} "
        f"| num_generations={cfg.get('num_generations', 8)}"
    )

    tokenizer = build_tokenizer(cfg)

    dtype = torch.bfloat16 if cfg.get("precision", "bf16") == "bf16" else torch.float32
    model, is_mm = load_lm(
        cfg["model"],
        dtype=dtype,
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
    )
    # GRPO needs the KV cache for fast rollouts; the SFT checkpoint may have been
    # saved with use_cache=False. TRL re-disables it for the loss/backward pass
    # when gradient checkpointing is on.
    model.config.use_cache = True
    print(f"[model] loaded {model.__class__.__name__} (multimodal={is_mm})")

    train_path = cfg["train_file"]
    if not Path(train_path).exists():
        raise SystemExit(
            f"train_file not found: {train_path}. Run prepare_grpo_data.py first."
        )
    dataset = load_dataset("json", data_files=train_path, split="train")
    if args.limit and args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    print(f"[data] {len(dataset)} problems from {train_path}")

    peft_config = None
    scope = cfg.get("language_model_scope", "model.language_model")
    if is_lora:
        from peft import LoraConfig

        targets = cfg.get("target_modules", "auto")
        if targets in ("auto", ["auto"]):
            targets = discover_lora_targets(model, scope)
            print(
                f"[lora] auto-discovered {len(targets)} Linear modules under "
                f"'{scope}'"
            )
        peft_config = LoraConfig(
            r=cfg.get("lora_r", 16),
            lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif is_mm:
        freeze_non_text(
            model, cfg.get("freeze_vision", True), cfg.get("freeze_mtp", True)
        )

    reward_fn = make_code_reward(
        timeout=float(cfg.get("reward_timeout", 6.0)),
        mem_mb=int(cfg.get("reward_mem_mb", 1024)),
        max_cases=int(cfg.get("reward_max_cases", 15)),
        num_workers=int(cfg.get("reward_num_workers", 8)),
    )

    grpo_config = build_grpo_config(cfg, output_dir)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[build_training_log_callback(output_dir)],
    )
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[done] saved model + tokenizer -> {output_dir}")

    # Multimodal: keep the checkpoint self-contained for downstream loaders
    # (vLLM/AutoProcessor expect preprocessor_config.json). Best-effort.
    if is_mm:
        try:
            from transformers import AutoProcessor

            AutoProcessor.from_pretrained(
                cfg["model"], trust_remote_code=True
            ).save_pretrained(output_dir)
            print(f"[done] saved processor -> {output_dir}")
        except Exception as exc:  # noqa: BLE001
            print(
                f"[warn] could not save processor ({type(exc).__name__}: {exc}); "
                "copy preprocessor_config.json from the base model if needed."
            )


if __name__ == "__main__":
    main()
