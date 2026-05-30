# Qwen3.5-0.8B-Base — Code SFT + GRPO/RLVR

Train and evaluate a small code model through this pipeline:

```
Base → Base Eval → Code SFT → SFT Eval → GRPO/RLVR → Final Eval
```

**Goal:** show that code benchmark pass@1 improves **Base < SFT < SFT+GRPO** on a
single **RTX 5090 (32GB)**. Full design spec lives in [`code_rlvr_plan.md`](code_rlvr_plan.md).

## Status

- [x] **Phase 1 — Base evaluation** (HumanEval+ / MBPP+)
- [x] **Phase 2 — Code SFT** (`ise-uiuc/Magicoder-OSS-Instruct-75K`)
- [x] **Phase 3 — SFT evaluation** (improves over Base on all four metrics)
- [ ] Phase 4 — GRPO / RLVR (`BAAI/TACO`, test-based reward) — *code ready, not yet run*
- [ ] Phase 5 — Final evaluation

## Layout

```
configs/
  eval.yaml              # fixed eval config (keep identical across phases)
  sft_full.yaml          # full SFT (primary, maxed for RTX 5090 32GB)
  sft_lora.yaml          # LoRA SFT (fallback)
  grpo.yaml              # GRPO / RLVR (Phase 4, tuned for A100 40GB)
scripts/
  eval_evalplus.py       # HumanEval+/MBPP+ harness (Phase 1/3/5)
  prepare_sft_data.py    # Magicoder -> chat-format jsonl (Phase 2)
  train_sft.py           # HF Trainer code SFT (Phase 2)
  prepare_grpo_data.py   # TACO -> {prompt, tests} jsonl (Phase 4)
  train_grpo.py          # TRL GRPOTrainer, test-based reward (Phase 4)
src/
  utils.py               # shared helpers (model loader, code extraction)
  reward.py              # GRPO reward: run tests, reward = pass rate (Phase 4)
  sandbox.py             # sandboxed execution of generated code (Phase 4)
data/                    # prepared datasets (gitignored)
requirements.txt
code_rlvr_plan.md        # full plan / spec
```

## Setup

The RTX 5090 is Blackwell (sm_120) and needs a CUDA 12.8+ build of PyTorch:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Qwen3.5 requires **`transformers >= 4.57`** (the `qwen3_5` architecture). For full
DeltaNet speed, also install the optional kernels (commented in
`requirements.txt`): `causal-conv1d` and `flash-linear-attention` (imports as
`fla`). Without them the model still trains/evaluates via a slower pure-PyTorch
path.

If vLLM is hard to install on this GPU, every script supports `--backend hf`
(plain `transformers`) as a fallback.

### Troubleshooting (Blackwell / RTX 5090)

- **`RuntimeError: FlashInfer requires GPUs with sm75 or higher`** — vLLM picked
  the FlashInfer attention backend, but the installed FlashInfer has no sm_120
  kernels (the message is misleading; the 5090 is fine). Fixes, in order:
  1. Force a different backend: add `--attention-backend FLASH_ATTN` (or set
     `attention_backend: FLASH_ATTN` in `configs/eval.yaml`).
  2. If that also fails, run with `--backend hf` to unblock immediately.
  3. For a permanent fix, upgrade to a Blackwell-ready vLLM + FlashInfer built
     against CUDA 12.8.

---

## Model architecture (read before changing model code)

`Qwen/Qwen3.5-0.8B-Base` is **not** a plain dense LLM — it is a natively
multimodal hybrid checkpoint, and the training/eval code is written around that:

- **Class:** `architectures = ["Qwen3_5ForConditionalGeneration"]`,
  `model_type = qwen3_5`. Load it with `AutoModelForImageTextToText`, *not*
  `AutoModelForCausalLM` (the shared helper `src.utils.load_lm` does this
  automatically and falls back to causal-LM loading for dense models).
- **Weights:** text decoder under `model.language_model.*`, vision tower under
  `model.visual.*`, a multi-token-prediction head under `mtp.*`. Embeddings are
  **tied** (no separate `lm_head` weight). Code SFT trains the text decoder only
  and freezes `visual` + `mtp`.
- **Hybrid attention:** 24 decoder layers in a 3:1 pattern — three Gated DeltaNet
  `linear_attention` layers per one `full_attention` layer
  (`full_attention_interval: 4`). `attn_implementation` (flash-attn/sdpa) applies
  to the full-attention layers; the DeltaNet layers use `causal-conv1d` + `fla`
  kernels when installed.
- **No sequence packing.** DeltaNet layers keep a recurrent state across the
  sequence, so packing multiple documents into one window leaks state across
  document boundaries. Both SFT configs set `packing: false`.

---

## Phase 1 — Base evaluation

Evaluates a model on **HumanEval+** and **MBPP+** and reports greedy **pass@1**.

**Design.** Generation runs locally with a *fixed* prompt template so Base, SFT,
and GRPO checkpoints are scored under identical conditions; running the hidden
test suites is delegated to `evalplus.evaluate`. The same script serves all eval
phases via `--mode`:

- `--mode base` — raw completion, for the base model.
- `--mode chat` — applies the tokenizer chat template, for SFT/GRPO checkpoints.

**Run:**

```bash
python scripts/eval_evalplus.py \
  --model Qwen/Qwen3.5-0.8B-Base \
  --mode base --dataset both \
  --output results/base_eval.json
```

**Quick smoke test** (checks the pipeline end-to-end; the reported pass@1 is
*not* meaningful because un-generated tasks count as failures):

```bash
python scripts/eval_evalplus.py --dataset humaneval --limit 5 --backend hf
```

**Outputs:**

- `results/base_eval.json` — summary with base/plus pass@1 per dataset.
- `results/samples/<run>_<dataset>.jsonl` — generated solutions (kept for
  inspection and re-scoring).

**Key flags:** `--backend {vllm,hf}`, `--dataset {humaneval,mbpp,both}`,
`--temperature`, `--max-new-tokens`, `--limit N` (debug), `--no-sanitize`.
Defaults come from [`configs/eval.yaml`](configs/eval.yaml); CLI flags override them.

**Notes:**

- With `--backend hf`, the eval harness loads `Qwen3.5-0.8B-Base` as a
  multimodal checkpoint (`AutoModelForImageTextToText`) and generates from its
  text decoder; with `--backend vllm`, vLLM must support the `qwen3_5`
  architecture (else fall back to `hf`).
- Do not change `configs/eval.yaml` between phases; a fair Base/SFT/GRPO
  comparison depends on an identical eval config.

---

## Phase 2 — Code SFT

Teaches the base model to follow code instructions, using
`ise-uiuc/Magicoder-OSS-Instruct-75K` converted to chat format. Built on the
Hugging Face `Trainer` (not TRL's `SFTTrainer`) so we fully control loading and
tokenization for the multimodal hybrid model (see
[Model architecture](#model-architecture-read-before-changing-model-code)). Two
configs: **full** fine-tuning (primary) and **LoRA** (fallback).

**What the script does.** Loads `Qwen3_5ForConditionalGeneration` via
`AutoModelForImageTextToText`, **freezes** the vision tower (`model.visual.*`)
and the MTP head (`mtp.*`), and fine-tunes the text decoder only. Data is
tokenized with the chat template and **completion-only loss** (`mask_prompt:
true` — the user/prompt tokens are set to `-100`, so loss is computed over the
assistant response). The script renders the chat template to text first, then
tokenizes that text into plain `list[int]` token IDs; this avoids Hugging Face
Datasets trying to write tokenizer-internal `Encoding` objects into Arrow.
**Packing is disabled** (DeltaNet recurrent state).

**Tuning for the RTX 5090 (32GB).** A 0.8B model leaves plenty of VRAM: bf16,
Flash-Attention-2 on the full-attention layers (auto-falls back to SDPA), fused
AdamW, TF32, `group_by_length` to cut padding waste, and an effective batch of 32
(`8 x 4`). With packing off each example is a full padded sequence, so the
per-device batch starts conservative — if `nvidia-smi` shows spare memory
mid-run, raise `per_device_train_batch_size`.

**1. Prepare data:**

```bash
python scripts/prepare_sft_data.py --limit 20000 \
  --output data/sft/magicoder_chat.jsonl
```

`--limit 20000` randomly samples 20k of the ~75k Magicoder examples (shuffled
with `--seed`) for SFT; the plan suggests starting at 10k–20k. Use `--limit 0`
to keep all ~75k. Larger = potentially better but slower.

**2. Train (full SFT):**

```bash
python scripts/train_sft.py --config configs/sft_full.yaml
```

Output model: `outputs/qwen35_0_8b_code_sft/` (full model + tokenizer with chat
template — ready for Phase 3 eval with `--mode chat`).

**Debug run** (checks the whole pipeline in a couple of minutes):

```bash
python scripts/train_sft.py --config configs/sft_full.yaml --limit 100 --max-steps 5
```

**LoRA fallback:** `--config configs/sft_lora.yaml`. With `target_modules: auto`
the script discovers every `nn.Linear` under `model.language_model.*` (DeltaNet
`in_proj_*`/`out_proj`, attention `q/k/v/o_proj`, MLP `gate/up/down_proj`) and
scopes the adapters there, so the frozen vision tower is never touched. This
saves an adapter to `outputs/qwen35_0_8b_code_sft_lora/`; merge it into the base
model before running Phase 3 eval.

**Notes:**

- `--limit` controls how many SFT samples are used (plan suggests 10k–20k).
- If tokenization fails with `OverflowError: There was an overflow with type
  <class 'list'>` plus `Could not convert Encoding(...)`, the real issue is
  usually not a 2GB batch: it means a tokenizer `Encoding` object leaked into
  the dataset cache. `train_sft.py` avoids this by using the two-step
  render-then-tokenize path described above.
- Some older or vendor-patched `transformers` builds may not accept every
  `TrainingArguments` keyword used here (for example `group_by_length`). The
  script filters unsupported keywords at runtime and prints a warning; training
  can continue, but the skipped option's optimization is disabled.
- The same eval script scores this checkpoint in Phase 3 — just point `--model`
  at the output dir and use `--mode chat`.

---

## Phase 3 — SFT evaluation

Scores the SFT checkpoint on the **same** benchmarks with the **same**
`configs/eval.yaml` as Phase 1, so Base vs SFT is apples-to-apples — the only
differences are the model and `--mode chat` (apply the chat template instead of
raw completion).

**Run:**

```bash
python scripts/eval_evalplus.py \
  --model outputs/qwen35_0_8b_code_sft \
  --mode chat --dataset both \
  --output results/sft_eval.json
```

`--mode chat` builds each prompt from the tokenizer chat template with
`enable_thinking=False` — exactly how `train_sft.py` formatted the SFT data, so
the model sees the prompt format it was trained on. The solution is extracted
from the assistant's ```python block (EvalPlus sanitizer as fallback).

**Outputs:**

- `results/sft_eval.json` — base/plus pass@1 per dataset.
- `results/samples/sft_eval_<dataset>.jsonl` — generated solutions.

Then copy the four numbers into the **SFT** row of [Results](#results); the bar
to clear is the Base row (HumanEval+ `0.1830`, MBPP+ `0.2490`).

**Notes:**

- The SFT output dir is a full `qwen3_5` checkpoint (text decoder fine-tuned,
  vision/MTP frozen) bundled with the tokenizer + chat template, so both
  `--backend vllm` and `--backend hf` load it directly. The hf path re-enables
  `use_cache` (training turns it off for gradient checkpointing).
- **LoRA checkpoints** (`outputs/qwen35_0_8b_code_sft_lora/`) are adapters, not a
  full model — merge the adapter into the base model first, then point `--model`
  at the merged dir.
- Keep `configs/eval.yaml` unchanged from Phase 1; the fair comparison depends on
  an identical eval config.

---

## Phase 4 — GRPO / RLVR

Optimizes the **SFT checkpoint** with GRPO and a *verifiable* reward: each
generated program is run against TACO unit tests and rewarded by the fraction of
tests it passes. No reward model — the tests *are* the reward.

**Pieces:**

- `scripts/prepare_grpo_data.py` — `BAAI/TACO` → `{prompt, tests}` jsonl.
- `src/sandbox.py` — runs untrusted, model-generated Python in a subprocess with
  OS resource limits + a reliability guard (handles stdin/stdout and call-based
  tests).
- `src/reward.py` — TRL reward function: extract code, run tests, reward = pass
  rate (threaded across the batch).
- `scripts/train_grpo.py` — TRL `GRPOTrainer`, same multimodal handling as SFT
  (load via `AutoModelForImageTextToText`, freeze vision/MTP, or LoRA on the text
  decoder).

**1. Prepare data:**

```bash
python scripts/prepare_grpo_data.py --limit 3000 --difficulty EASY \
  --output data/grpo/taco_grpo.jsonl
```

Defaults to EASY, stdin/stdout-only problems so the 0.8B model passes enough
tests to get a usable reward signal. `--difficulty all` / `--include-fn-name`
widen the pool; `--streaming` avoids downloading all of TACO (it's large — set
`HF_ENDPOINT` too if downloads are slow).

**2. Train (from the SFT checkpoint):**

```bash
python scripts/train_grpo.py --config configs/grpo.yaml
```

Output: `outputs/qwen35_0_8b_code_grpo/`. Score it just like Phase 3 (`--mode
chat`) and fill the **SFT + GRPO** row of [Results](#results).

**Debug run** (end-to-end in a couple of minutes):

```bash
python scripts/train_grpo.py --config configs/grpo.yaml --limit 32 --max-steps 3
```

**Logging.** Training writes `training_log.jsonl` (full metric history) to the
output dir plus live curve plots — `reward_curve.svg` (**the one to watch**),
`loss_curve.svg`, `kl_curve.svg`, `reward_std_curve.svg` (`.png` too if
matplotlib is installed). If mean reward stays flat near 0, the tasks are too
hard and there's no learning signal — make the data easier (see Phase 4 data
notes). GRPO loss is often near-zero and uninformative; judge progress by reward.

**Tuning for the A100 (40GB).** bf16, gradient checkpointing, fused AdamW. The
memory/speed driver is generation — `num_generations` (8) ×
`max_completion_length` (1024). If you OOM, lower those first, or set
`training_type: lora` (which also drops the KL reference-model copy). Generation
uses HF `generate` (`use_vllm: false`) because vLLM's engine init has been
unreliable for this `qwen3_5` hybrid, so rollouts are the throughput bottleneck.

> **Security:** `src/sandbox.py` executes model-written code. It uses a
> subprocess + OS resource limits + a reliability guard, but that is **not** a
> hard sandbox (no network/namespace isolation). Run GRPO only on a disposable
> box.

---

## Results

Greedy **pass@1** from `scripts/eval_evalplus.py` (EvalPlus). `HumanEval` / `MBPP`
are the original tests; `HumanEval+` / `MBPP+` add EvalPlus's extra tests (the
headline metric for the Base < SFT < SFT+GRPO comparison).

| Model | HumanEval | HumanEval+ | MBPP | MBPP+ | Notes |
|---|---:|---:|---:|---:|---|
| Base | 0.1950 | 0.1830 | 0.3040 | 0.2490 | Qwen3.5-0.8B-Base |
| SFT | 0.2800 | 0.2620 | 0.3440 | 0.2860 | Magicoder SFT (20k, 3 ep) |
| SFT + GRPO | TBD | TBD | TBD | TBD | TACO RLVR |

## References

- Magicoder-OSS-Instruct-75K: https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K
- TACO: https://huggingface.co/datasets/BAAI/TACO
- EvalPlus: https://github.com/evalplus/evalplus
