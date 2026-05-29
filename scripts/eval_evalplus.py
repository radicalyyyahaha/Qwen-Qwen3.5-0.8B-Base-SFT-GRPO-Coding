#!/usr/bin/env python
"""Evaluate a model on HumanEval+ and MBPP+ (Phase 1/3/5 of the plan).

Generation is done locally with a fixed, controlled prompt template so that the
Base, SFT, and GRPO checkpoints are scored under identical conditions. Scoring
(running the hidden test suites) is delegated to the `evalplus.evaluate` CLI.

Examples
--------
Base model (raw completion), both benchmarks, vLLM:
    python scripts/eval_evalplus.py \
        --model Qwen/Qwen3.5-0.8B-Base \
        --mode base --output results/base_eval.json

SFT / GRPO checkpoint (uses the chat template):
    python scripts/eval_evalplus.py \
        --model outputs/qwen35_0_8b_code_sft \
        --mode chat --output results/sft_eval.json

Quick pipeline smoke test (NOTE: pass@1 is meaningless with --limit, because
un-generated tasks count as failures in evalplus):
    python scripts/eval_evalplus.py --dataset humaneval --limit 5 --backend hf
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils import extract_code, load_yaml, now_iso, write_jsonl  # noqa: E402

# Stop sequences for base-model completion: cut the model off once it starts a
# new top-level definition / example, so it doesn't ramble into the next task.
# (The evalplus sanitizer also trims trailing junk as a second line of defense.)
BASE_STOP = ["\nclass ", "\ndef ", "\nif __name__", "\nassert ", "\nprint(", "\n```"]

CHAT_INSTRUCTION = (
    "Complete the following Python function. Return only the full function "
    "implementation in a single ```python code block, with no explanation.\n\n"
)

DEFAULTS = {
    "backend": "vllm",
    "temperature": 0.0,
    "top_p": 1.0,
    "max_new_tokens": 512,
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.90,
    "tensor_parallel_size": 1,
    "attention_backend": "",
}


def load_problems(dataset):
    if dataset == "humaneval":
        from evalplus.data import get_human_eval_plus

        return get_human_eval_plus()
    if dataset == "mbpp":
        from evalplus.data import get_mbpp_plus

        return get_mbpp_plus()
    raise ValueError(f"unknown dataset: {dataset}")


def build_chat_prompt(problem, tokenizer):
    user = CHAT_INSTRUCTION + "```python\n" + problem["prompt"].rstrip() + "\n```"
    messages = [{"role": "user", "content": user}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def get_sanitizer():
    try:
        from evalplus.sanitize import sanitize

        return sanitize
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] evalplus.sanitize unavailable ({exc}); using raw output")
        return None


def try_sanitize(sanitizer, code, entry_point):
    if sanitizer is None:
        return code
    try:
        cleaned = sanitizer(code, entry_point)
        return cleaned if cleaned and cleaned.strip() else code
    except Exception:  # noqa: BLE001
        return code


def make_generator(args):
    """Load the backend once and return a `gen(prompts, stop) -> list[str]`."""
    if args.backend == "vllm":
        # Must be set before importing vllm. On Blackwell/RTX 5090 the default
        # FlashInfer backend may lack sm_120 kernels; FLASH_ATTN avoids that.
        if args.attention_backend:
            os.environ["VLLM_ATTENTION_BACKEND"] = args.attention_backend
            print(f"[vllm] VLLM_ATTENTION_BACKEND={args.attention_backend}")
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=args.model,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
        )

        def gen(prompts, stop):
            sampling = SamplingParams(
                n=1,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_new_tokens,
                stop=stop,
            )
            outputs = llm.generate(prompts, sampling)
            return [o.outputs[0].text for o in outputs]

        return gen

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    def gen(prompts, stop):
        out_texts = []
        for i in range(0, len(prompts), args.batch_size):
            batch = prompts[i : i + args.batch_size]
            enc = tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_model_len,
            ).to(model.device)
            gen_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tok.pad_token_id,
            )
            if args.temperature and args.temperature > 0:
                gen_kwargs.update(
                    do_sample=True, temperature=args.temperature, top_p=args.top_p
                )
            else:
                gen_kwargs.update(do_sample=False)
            if stop:
                gen_kwargs.update(stop_strings=stop, tokenizer=tok)
            with torch.no_grad():
                out = model.generate(**enc, **gen_kwargs)
            new_tokens = out[:, enc["input_ids"].shape[1] :]
            out_texts.extend(tok.batch_decode(new_tokens, skip_special_tokens=True))
            print(f"[gen] {min(i + args.batch_size, len(prompts))}/{len(prompts)}")
        return out_texts

    return gen


def run_evalplus_score(dataset, samples_path):
    """Run `evalplus.evaluate` on a samples file; return (base, plus) pass@1."""
    cmd = [
        sys.executable,
        "-m",
        "evalplus.evaluate",
        "--dataset",
        dataset,
        "--samples",
        str(samples_path),
    ]
    print(f"[score] {' '.join(cmd)}")
    captured = []
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        captured.append(line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"evalplus.evaluate failed (exit {proc.returncode})")

    # evalplus prints two pass@1 lines per dataset: base tests, then base+extra.
    nums = re.findall(r"pass@1:\s*([0-9.]+)", "".join(captured))
    base = float(nums[0]) if len(nums) >= 1 else None
    plus = float(nums[1]) if len(nums) >= 2 else None
    return base, plus


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(REPO_ROOT / "configs" / "eval.yaml"))
    known, _ = pre.parse_known_args()

    cfg = dict(DEFAULTS)
    if os.path.exists(known.config):
        loaded = load_yaml(known.config) or {}
        cfg.update({k: v for k, v in loaded.items() if k in cfg})

    p = argparse.ArgumentParser(parents=[pre], description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    p.add_argument(
        "--dataset", choices=["humaneval", "mbpp", "both"], default="both"
    )
    p.add_argument(
        "--mode",
        choices=["base", "chat"],
        default="base",
        help="base = raw completion (base model); chat = apply chat template",
    )
    p.add_argument("--backend", choices=["vllm", "hf"], default=cfg["backend"])
    p.add_argument("--output", default="results/base_eval.json")
    p.add_argument("--temperature", type=float, default=cfg["temperature"])
    p.add_argument("--top-p", type=float, default=cfg["top_p"])
    p.add_argument("--max-new-tokens", type=int, default=cfg["max_new_tokens"])
    p.add_argument("--max-model-len", type=int, default=cfg["max_model_len"])
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=cfg["gpu_memory_utilization"],
    )
    p.add_argument(
        "--tensor-parallel-size", type=int, default=cfg["tensor_parallel_size"]
    )
    p.add_argument(
        "--attention-backend",
        default=cfg["attention_backend"] or None,
        help="vLLM attention backend override (e.g. FLASH_ATTN, XFORMERS). On "
        "Blackwell/RTX 5090, use FLASH_ATTN if the default FlashInfer fails.",
    )
    p.add_argument("--batch-size", type=int, default=16, help="hf backend only")
    p.add_argument("--limit", type=int, default=0, help="debug: first N tasks only")
    p.add_argument("--no-sanitize", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    datasets = ["humaneval", "mbpp"] if args.dataset == "both" else [args.dataset]

    output_path = Path(args.output)
    run_name = output_path.stem
    samples_dir = output_path.parent / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    chat_tokenizer = None
    if args.mode == "chat":
        from transformers import AutoTokenizer

        chat_tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=True
        )

    sanitizer = None if args.no_sanitize else get_sanitizer()
    gen = make_generator(args)

    results = {}
    for ds in datasets:
        print(f"\n=== {ds} ===")
        problems = load_problems(ds)
        task_ids = list(problems)
        if args.limit and args.limit > 0:
            task_ids = task_ids[: args.limit]
            print(
                f"[warn] --limit {args.limit}: scoring {len(task_ids)}/"
                f"{len(problems)} tasks. pass@1 counts the rest as FAIL -- "
                "use only as a pipeline smoke test."
            )

        if args.mode == "chat":
            prompts = [build_chat_prompt(problems[t], chat_tokenizer) for t in task_ids]
            stop = None
        else:
            prompts = [problems[t]["prompt"] for t in task_ids]
            stop = BASE_STOP

        completions = gen(prompts, stop)

        rows = []
        for task_id, completion in zip(task_ids, completions):
            if args.mode == "chat":
                solution = extract_code(completion)
            else:
                solution = problems[task_id]["prompt"] + completion
            solution = try_sanitize(
                sanitizer, solution, problems[task_id]["entry_point"]
            )
            rows.append({"task_id": task_id, "solution": solution})

        samples_path = samples_dir / f"{run_name}_{ds}.jsonl"
        write_jsonl(samples_path, rows)
        print(f"[write] {len(rows)} samples -> {samples_path}")

        base_p, plus_p = run_evalplus_score(ds, samples_path)
        results[ds] = {
            "base_pass@1": base_p,
            "plus_pass@1": plus_p,
            "n_generated": len(rows),
            "n_total": len(problems),
            "partial": bool(args.limit),
            "samples": str(samples_path),
        }

    summary = {
        "model": args.model,
        "mode": args.mode,
        "backend": args.backend,
        "timestamp": now_iso(),
        "config": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "sanitize": not args.no_sanitize,
        },
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    output_path.write_text(json.dumps(summary, indent=2))

    print(f"\n[done] wrote {output_path}")
    print(f"{'dataset':<12}{'base pass@1':>14}{'plus pass@1':>14}")
    for ds, r in results.items():
        b = "n/a" if r["base_pass@1"] is None else f"{r['base_pass@1']:.4f}"
        pl = "n/a" if r["plus_pass@1"] is None else f"{r['plus_pass@1']:.4f}"
        print(f"{ds:<12}{b:>14}{pl:>14}")


if __name__ == "__main__":
    main()
