"""Train and evaluate accuracy predictors for vit_nas NAS.

Loads a dataset of (config, accuracy) pairs from a JSON file, trains both
an MLP and a GBM predictor, reports ranking metrics, and saves the best model.

Dataset JSON format  (list of objects):
    [
      {"config": {...}, "accuracy": 84.6},
      ...
    ]
    The "accuracy" field should be in percent (0–100).
    Any entry without "accuracy" is skipped.

Usage:
    # collect data first using collect_predictor_data.py, then:
    python train_accuracy_predictor.py \\
        --dataset predictor_data.json \\
        --embed-dim-options 512 \\
        --num-layers-options 2 4 6 \\
        --num-heads-options 2 4 8 \\
        --mlp-dim-options 256 512 1024 \\
        --val-split 0.2 \\
        --save-dir predictor_ckpts/
"""

import argparse
import json
import os
import random

import numpy as np
import torch

from search.search import SearchSpace
from search.accuracy_predictor import (
    ArchEncoder, MLPPredictor, GBMPredictor, evaluate_predictor
)


def parse_args():
    p = argparse.ArgumentParser()

    # dataset
    p.add_argument("--dataset",    type=str, required=True,
                   help="JSON file with list of {config, accuracy} dicts")
    p.add_argument("--val-split",  type=float, default=0.2,
                   help="Fraction of data held out for validation")
    p.add_argument("--seed",       type=int, default=42)

    # search space (must match the one used during data collection)
    p.add_argument("--embed-dim-options",  type=int, nargs="+", default=[512])
    p.add_argument("--num-layers-options", type=int, nargs="+", default=[2, 4, 6])
    p.add_argument("--num-heads-options",  type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--mlp-dim-options",    type=int, nargs="+", default=[256, 512, 1024])

    # MLP hyper-parameters
    p.add_argument("--hidden",     type=int,   default=256)
    p.add_argument("--n-layers",   type=int,   default=3)
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--epochs",     type=int,   default=300)
    p.add_argument("--batch-size", type=int,   default=64)

    # GBM hyper-parameters
    p.add_argument("--gbm-estimators",    type=int,   default=500)
    p.add_argument("--gbm-depth",         type=int,   default=4)
    p.add_argument("--gbm-lr",            type=float, default=0.05)
    p.add_argument("--gbm-subsample",     type=float, default=0.8)

    p.add_argument("--save-dir",   type=str, default="predictor_ckpts",
                   help="Directory to save trained predictor checkpoints")

    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    # --- load dataset ---
    with open(args.dataset) as f:
        raw = json.load(f)

    data = [(r["config"], r["accuracy"]) for r in raw if "accuracy" in r]
    if not data:
        raise ValueError(f"No entries with 'accuracy' found in {args.dataset}")
    print(f"Loaded {len(data)} labelled samples from {args.dataset}")

    # --- train/val split ---
    random.shuffle(data)
    n_val = max(1, int(len(data) * args.val_split))
    val_data   = data[:n_val]
    train_data = data[n_val:]
    train_cfgs, train_accs = zip(*train_data)
    val_cfgs,   val_accs   = zip(*val_data)
    print(f"  train: {len(train_data)}  val: {len(val_data)}")
    print(f"  accuracy range: {min(train_accs):.1f}% – {max(train_accs):.1f}%")

    # --- build search space + encoder ---
    ss = SearchSpace(
        embed_dim_options=args.embed_dim_options,
        num_heads_options=args.num_heads_options,
        mlp_dim_options=args.mlp_dim_options,
        num_layers_options=args.num_layers_options,
    )
    encoder = ArchEncoder(ss)
    print(f"Feature vector dim: {encoder.dim}")

    os.makedirs(args.save_dir, exist_ok=True)

    # --- MLP ---
    print("\n=== Training MLP predictor ===")
    mlp = MLPPredictor(
        encoder,
        hidden=args.hidden, n_layers=args.n_layers,
        dropout=args.dropout, lr=args.lr,
        epochs=args.epochs, batch_size=args.batch_size,
    )
    mlp.fit(list(train_cfgs), list(train_accs),
            val_configs=list(val_cfgs), val_accs=list(val_accs))

    mlp_metrics = evaluate_predictor(mlp, list(val_cfgs), list(val_accs))
    print(f"\nMLP val metrics:")
    print(f"  RMSE        = {mlp_metrics['rmse']:.3f}%")
    print(f"  Kendall-τ   = {mlp_metrics['kendall_tau']:.3f}")
    print(f"  Spearman-ρ  = {mlp_metrics['spearman_rho']:.3f}")
    mlp.save(os.path.join(args.save_dir, "mlp_predictor.pth"))

    # --- GBM ---
    print("\n=== Training GBM predictor ===")
    gbm = GBMPredictor(
        encoder,
        n_estimators=args.gbm_estimators,
        max_depth=args.gbm_depth,
        learning_rate=args.gbm_lr,
        subsample=args.gbm_subsample,
    )
    gbm.fit(list(train_cfgs), list(train_accs),
            val_configs=list(val_cfgs), val_accs=list(val_accs))

    gbm_metrics = evaluate_predictor(gbm, list(val_cfgs), list(val_accs))
    print(f"\nGBM val metrics:")
    print(f"  RMSE        = {gbm_metrics['rmse']:.3f}%")
    print(f"  Kendall-τ   = {gbm_metrics['kendall_tau']:.3f}")
    print(f"  Spearman-ρ  = {gbm_metrics['spearman_rho']:.3f}")
    gbm.save(os.path.join(args.save_dir, "gbm_predictor.pkl"))

    # --- pick best by Kendall-τ ---
    best = "MLP" if mlp_metrics["kendall_tau"] >= gbm_metrics["kendall_tau"] else "GBM"
    print(f"\n✓ Best predictor by Kendall-τ: {best}")


if __name__ == "__main__":
    main()
