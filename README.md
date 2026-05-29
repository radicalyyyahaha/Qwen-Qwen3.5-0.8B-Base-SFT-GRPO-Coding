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
- [ ] Phase 3 — SFT evaluation
- [ ] Phase 4 — GRPO / RLVR (`BAAI/TACO`, test-based reward)
- [ ] Phase 5 — Final evaluation

## Layout

```
configs/
  eval.yaml              # fixed eval config (keep identical across phases)
  sft_full.yaml          # full SFT (primary, maxed for RTX 5090 32GB)
  sft_lora.yaml          # LoRA SFT (fallback)
scripts/
  eval_evalplus.py       # HumanEval+/MBPP+ harness (Phase 1/3/5)
  prepare_sft_data.py    # Magicoder -> chat-format jsonl (Phase 2)
  train_sft.py           # HF Trainer code SFT (Phase 2)
src/
  utils.py               # shared helpers
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
assistant response). **Packing is disabled** (DeltaNet recurrent state).

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
- The same eval script scores this checkpoint in Phase 3 — just point `--model`
  at the output dir and use `--mode chat`.

---

## Results

| Model | HumanEval+ pass@1 | MBPP+ pass@1 | Notes |
|---|---:|---:|---|
| Base | TBD | TBD | Qwen3.5-0.8B-Base |
| SFT | TBD | TBD | Magicoder SFT |
| SFT + GRPO | TBD | TBD | TACO RLVR |

## References

- Magicoder-OSS-Instruct-75K: https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K
- TACO: https://huggingface.co/datasets/BAAI/TACO
- EvalPlus: https://github.com/evalplus/evalplus
