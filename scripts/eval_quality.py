"""
Quality evaluation: perplexity, MMLU, GSM8K.

Usage:
    # Baseline perplexity
    python scripts/eval_quality.py --model mistralai/Mistral-7B-v0.3 --eval ppl

    # With optimizations
    python scripts/eval_quality.py --model mistralai/Mistral-7B-v0.3 --teal 0.4 --tq-bits 3 --eval ppl mmlu

    # All evaluations
    python scripts/eval_quality.py --config configs/mistral7b.yaml --eval ppl mmlu gsm8k
"""

import argparse
import math
import os
import sys
import torch
import yaml
from torch.nn import CrossEntropyLoss

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leankv.combined import load_model, optimize_model


def evaluate_perplexity(
    model, tokenizer, dataset_name="Salesforce/wikitext", dataset_config="wikitext-2-raw-v1",
    split="test", max_length=2048, stride=512,
):
    """
    Compute perplexity on WikiText-2 using sliding window.

    Returns:
        float: perplexity value (lower = better).
    """
    from datasets import load_dataset

    print(f"[Eval] Computing perplexity on {dataset_name}/{dataset_config}...")
    dataset = load_dataset(dataset_name, dataset_config, split=split)

    # Concatenate all text
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings["input_ids"].to(model.device)

    seq_len = input_ids.shape[1]
    nlls = []
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_len = end - prev_end  # number of new tokens to score

        input_chunk = input_ids[:, begin:end]

        with torch.no_grad():
            outputs = model(input_chunk)
            logits = outputs.logits

        # Only score the new tokens (not the overlap)
        shift_logits = logits[:, -target_len - 1 : -1, :].contiguous()
        shift_labels = input_chunk[:, -target_len:].contiguous()

        loss_fn = CrossEntropyLoss(reduction="none")
        loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        nlls.append(loss.sum().item())

        prev_end = end
        if end >= seq_len:
            break

    total_nll = sum(nlls)
    total_tokens = prev_end - 0  # approximate
    ppl = math.exp(total_nll / total_tokens)
    print(f"[Eval] Perplexity = {ppl:.2f} ({total_tokens} tokens)")
    return ppl


def evaluate_lm_eval(model, tokenizer, tasks, num_fewshot=None):
    """
    Run lm-evaluation-harness benchmarks.

    Args:
        model: HuggingFace model.
        tokenizer: corresponding tokenizer.
        tasks: list of task names (e.g., ["mmlu", "gsm8k"]).
        num_fewshot: override number of few-shot examples.

    Returns:
        dict of {task: accuracy}.
    """
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("[Eval] lm-eval not installed. Run: pip install lm-eval")
        return {}

    print(f"[Eval] Running lm-eval tasks: {tasks}")

    lm = HFLM(pretrained=model, tokenizer=tokenizer)

    fewshot_map = {
        "mmlu": 5,
        "gsm8k": 8,
        "arc_challenge": 25,
        "hellaswag": 10,
        "winogrande": 5,
        "piqa": 0,
        "humaneval": 0,
    }

    results = {}
    for task in tasks:
        n = num_fewshot if num_fewshot is not None else fewshot_map.get(task, 0)
        print(f"  Running {task} ({n}-shot)...")
        try:
            output = lm_eval.simple_evaluate(
                model=lm,
                tasks=[task],
                num_fewshot=n,
                batch_size="auto",
            )
            task_results = output["results"].get(task, {})
            acc = task_results.get("acc,none", task_results.get("acc_norm,none", None))
            if acc is not None:
                results[task] = round(acc * 100, 2)
                print(f"  {task}: {results[task]}%")
            else:
                print(f"  {task}: no accuracy metric found")
                results[task] = None
        except Exception as e:
            print(f"  {task}: failed ({e})")
            results[task] = None

    return results


def main():
    parser = argparse.ArgumentParser(description="LeanKV quality evaluation")
    parser.add_argument("--model", type=str, help="HuggingFace model name")
    parser.add_argument("--config", type=str, help="Config YAML path")
    parser.add_argument("--teal", type=float, default=None, help="TEAL sparsity")
    parser.add_argument("--teal-thresholds", type=str, default=None)
    parser.add_argument("--tq-bits", type=int, default=None, help="TurboQuant bits")
    parser.add_argument(
        "--eval", type=str, nargs="+", default=["ppl"],
        choices=["ppl", "mmlu", "gsm8k", "arc", "hellaswag", "humaneval"],
        help="Evaluations to run",
    )
    args = parser.parse_args()

    # Resolve model
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        model_name = cfg["model_name"]
    elif args.model:
        model_name = args.model
    else:
        parser.error("Either --model or --config is required")

    model, tokenizer = load_model(model_name)
    model, _ = optimize_model(
        model,
        teal_sparsity=args.teal,
        teal_thresholds_path=args.teal_thresholds,
        turboquant_bits=args.tq_bits,
    )

    results = {}

    if "ppl" in args.eval:
        results["perplexity"] = evaluate_perplexity(model, tokenizer)

    lm_eval_tasks = []
    task_map = {
        "mmlu": "mmlu",
        "gsm8k": "gsm8k",
        "arc": "arc_challenge",
        "hellaswag": "hellaswag",
        "humaneval": "humaneval",
    }
    for e in args.eval:
        if e in task_map:
            lm_eval_tasks.append(task_map[e])

    if lm_eval_tasks:
        lm_results = evaluate_lm_eval(model, tokenizer, lm_eval_tasks)
        results.update(lm_results)

    print(f"\n{'='*40}")
    print("Quality Results Summary")
    print(f"{'='*40}")
    for metric, value in results.items():
        if value is not None:
            print(f"  {metric}: {value}")


if __name__ == "__main__":
    main()
