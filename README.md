# Qwen3.5-0.8B-Base — Code SFT + GRPO/RLVR

Train and evaluate a small code model through this pipeline:

```
Base → Base Eval → Code SFT → SFT Eval → GRPO/RLVR → Final Eval
```

**Goal:** show that code benchmark pass@1 improves **Base < SFT < SFT+GRPO** on a
single **RTX 5090 (32GB)**. Full design spec lives in [`code_rlvr_plan.md`](code_rlvr_plan.md).

## Status

- [x] **Phase 1 — Base evaluation** (HumanEval+ / MBPP+)
- [ ] Phase 2 — Code SFT (`ise-uiuc/Magicoder-OSS-Instruct-75K`)
- [ ] Phase 3 — SFT evaluation
- [ ] Phase 4 — GRPO / RLVR (`BAAI/TACO`, test-based reward)
- [ ] Phase 5 — Final evaluation

## Layout

```
configs/
  eval.yaml              # fixed eval config (keep identical across phases)
scripts/
  eval_evalplus.py       # HumanEval+/MBPP+ harness (Phase 1/3/5)
src/
  utils.py               # shared helpers
requirements.txt
code_rlvr_plan.md        # full plan / spec
```

## Setup

The RTX 5090 is Blackwell (sm_120) and needs a CUDA 12.8+ build of PyTorch:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

If vLLM is hard to install on this GPU, every script supports `--backend hf`
(plain `transformers`) as a fallback.

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

- The model id `Qwen/Qwen3.5-0.8B-Base` is unverified — swap `--model` if the
  download fails.
- Do not change `configs/eval.yaml` between phases; a fair Base/SFT/GRPO
  comparison depends on an identical eval config.

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
