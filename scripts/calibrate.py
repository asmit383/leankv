"""
Run TEAL calibration to compute per-layer sparsity thresholds.

Usage:
    python scripts/calibrate.py --model mistralai/Mistral-7B-v0.3 --sparsity 0.4
    python scripts/calibrate.py --config configs/mistral7b.yaml --sparsity 0.3 0.4 0.5
"""

import argparse
import os
import sys
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leankv.combined import load_model
from leankv.teal.calibration import collect_activation_histograms, load_histograms
from leankv.teal.greedy_opt import greedy_optimize, save_thresholds


def main():
    parser = argparse.ArgumentParser(description="TEAL calibration")
    parser.add_argument("--model", type=str, help="HuggingFace model name")
    parser.add_argument("--config", type=str, help="Config YAML path")
    parser.add_argument(
        "--sparsity", type=float, nargs="+", default=[0.3, 0.4, 0.5],
        help="Target sparsity levels",
    )
    parser.add_argument("--samples", type=int, default=300, help="Calibration samples")
    parser.add_argument("--output-dir", type=str, default="thresholds", help="Output directory")
    parser.add_argument(
        "--skip-histograms", action="store_true",
        help="Skip histogram collection, use existing histograms",
    )
    args = parser.parse_args()

    # Resolve model name
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        model_name = config["model_name"]
        model_short = config["model_short"]
    elif args.model:
        model_name = args.model
        model_short = model_name.split("/")[-1].lower().replace("-", "_")
    else:
        parser.error("Either --model or --config is required")

    hist_dir = os.path.join(args.output_dir, model_short, "histograms")
    thresholds_dir = os.path.join(args.output_dir, model_short)

    # Step 1: Collect histograms
    if not args.skip_histograms:
        model, tokenizer = load_model(model_name)
        num_layers = model.config.num_hidden_layers

        histograms = collect_activation_histograms(
            model, tokenizer,
            num_samples=args.samples,
            save_dir=hist_dir,
        )
        # Free model memory
        del model
        torch.cuda.empty_cache()
    else:
        print(f"[Calibration] Loading existing histograms from {hist_dir}")
        histograms = load_histograms(hist_dir)
        # Infer num_layers from histogram keys
        num_layers = max(k[0] for k in histograms.keys()) + 1

    # Step 2: Run greedy optimization for each sparsity level
    for sparsity in args.sparsity:
        print(f"\n[Calibration] Optimizing for {sparsity:.0%} sparsity...")
        thresholds = greedy_optimize(
            histograms,
            target_sparsity=sparsity,
            num_layers=num_layers,
        )

        path = os.path.join(thresholds_dir, f"thresholds_s{int(sparsity*100)}.json")
        save_thresholds(thresholds, path)

        # Print summary
        total_nonzero = 0
        total_projs = 0
        for layer_thresholds in thresholds.values():
            for t in layer_thresholds.values():
                if t > 0:
                    total_nonzero += 1
                total_projs += 1
        print(
            f"  {total_nonzero}/{total_projs} projections have non-zero thresholds"
        )

    print("\n[Calibration] Done!")


if __name__ == "__main__":
    main()
