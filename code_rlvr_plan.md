# Code RLVR Training Plan: Qwen3.5-0.8B-Base

## Goal

Train and evaluate a small code model with this pipeline:

```text
Base Model -> Base Eval -> Code SFT -> SFT Eval -> GRPO/RLVR -> Final Eval
```

Target model:

```text
Qwen/Qwen3.5-0.8B-Base
```

Target GPU:

```text
Single RTX 5090 32GB
```

Main objective:

```text
Check whether code benchmark score improves:
Base < SFT < SFT + GRPO
```

---

## Datasets

### SFT Dataset

Use:

```text
ise-uiuc/Magicoder-OSS-Instruct-75K
```

Purpose:

```text
Teach the base model to follow code instructions and output valid code.
```

Start with:

```text
10k-20k samples
```

Convert to chat-style SFT format:

```json
{
  "messages": [
    {"role": "user", "content": "<instruction>"},
    {"role": "assistant", "content": "<code solution>"}
  ]
}
```

---

### GRPO / RLVR Dataset

Use:

```text
BAAI/TACO
```

Purpose:

```text
Use programming problems with tests as verifiable reward data.
```

Start with:

```text
1000-3000 train problems
```

Preprocessed format:

```json
{
  "prompt": "Problem statement + input/output format + constraints. Return only Python code.",
  "tests": "Hidden unit tests or checker used only by reward function."
}
```

Important:

```text
The model only sees prompt.
The reward function sees tests.
Do not put hidden tests into prompt.
```

---

## Benchmarks

Use for evaluation only:

```text
HumanEval+
MBPP+
```

Optional later:

```text
LiveCodeBench
```

Fixed eval config:

```yaml
temperature: 0
metric: pass@1
same_prompt_template: true
same_max_new_tokens: true
same_timeout: true
```

Do not train on benchmark test data.

---

## Pipeline

### Phase 1: Base Evaluation

Input:

```text
Qwen/Qwen3.5-0.8B-Base
```

Run:

```text
HumanEval+
MBPP+
```

Save:

```text
results/base_eval.json
```

---

### Phase 2: SFT

Input:

```text
Qwen/Qwen3.5-0.8B-Base
Magicoder-OSS-Instruct-75K sample
```

Recommended full-SFT config:

```yaml
training_type: full_sft
precision: bf16
seq_len: 4096
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2e-5
num_train_epochs: 1
gradient_checkpointing: true
output_dir: outputs/qwen35_0_8b_code_sft
```

Fallback LoRA config:

```yaml
training_type: lora_sft
lora_r: 16
lora_alpha: 32
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

---

### Phase 3: SFT Evaluation

Input:

```text
outputs/qwen35_0_8b_code_sft
```

Run the same eval:

```text
HumanEval+
MBPP+
```

Save:

```text
results/sft_eval.json
```

---

### Phase 4: GRPO / RLVR

Input:

```text
outputs/qwen35_0_8b_code_sft
TACO train sample
```

Recommended config:

```yaml
training_type: grpo
precision: bf16
num_generations: 4
max_prompt_length: 1024
max_completion_length: 512
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 1e-6
num_train_epochs: 1
gradient_checkpointing: true
output_dir: outputs/qwen35_0_8b_code_grpo
```

Reward function sketch:

```python
def reward_func(prompts, completions, tests, **kwargs):
    rewards = []
    for code, test in zip(completions, tests):
        result = run_code_safely(code, test, timeout=5)
        rewards.append(1.0 if result.all_passed else 0.0)
    return rewards
```

Safety requirement:

```text
Run generated code in a sandbox.
Use subprocess timeout at minimum.
Do not use raw exec without isolation.
```

---

### Phase 5: Final Evaluation

Input:

```text
outputs/qwen35_0_8b_code_grpo
```

Run:

```text
HumanEval+
MBPP+
Optional: LiveCodeBench
```

Save:

```text
results/grpo_eval.json
```

Final table:

```markdown
| Model | HumanEval+ pass@1 | MBPP+ pass@1 | Notes |
|---|---:|---:|---|
| Base | TBD | TBD | Qwen3.5-0.8B-Base |
| SFT | TBD | TBD | Magicoder SFT |
| SFT + GRPO | TBD | TBD | TACO RLVR |
```

---

## Files to Implement

```text
scripts/
  prepare_sft_data.py
  prepare_grpo_data.py
  train_sft.py
  train_grpo.py
  eval_evalplus.py
  run_all.sh

src/
  reward.py
  sandbox.py
  utils.py

configs/
  sft_full.yaml
  sft_lora.yaml
  grpo.yaml
```

---

## Expected First Run

Debug mode:

```text
SFT: 100 samples
GRPO: 20 problems
Eval: 5 HumanEval problems
```

Real first experiment:

```text
SFT: 10k-20k samples
GRPO: 1k-3k problems
Eval: full HumanEval+ + MBPP+
```

---

## Key Rules

```text
1. Do not train on benchmark test data.
2. Keep prompt template fixed across Base/SFT/GRPO eval.
3. For GRPO, tests are only used inside reward function.
4. Start with num_generations=4, not 8.
5. Keep max_completion_length=512 first.
6. Log compile rate, pass rate, timeout rate, and average completion length.
```

---

## References

- Magicoder-OSS-Instruct-75K: https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K
- TACO: https://huggingface.co/datasets/BAAI/TACO
- EvalPlus: https://github.com/evalplus/evalplus
- LiveCodeBench: https://github.com/livecodebench/livecodebench
