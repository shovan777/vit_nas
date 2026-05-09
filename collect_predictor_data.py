"""Collect (config, accuracy) pairs for training the accuracy predictor.

Randomly samples subnets from the search space, evaluates each on the
CIFAR-10 test set using a trained supernet, and saves results to JSON.

Usage:
    python collect_predictor_data.py \\
        --checkpoint final_supernet.pth \\
        --n-samples 200 \\
        --embed-dim-options 512 \\
        --num-layers-options 2 4 6 \\
        --num-heads-options 2 4 8 \\
        --mlp-dim-options 256 512 1024 \\
        --output predictor_data.json

    # Resume / append to an existing file:
    python collect_predictor_data.py ... --output predictor_data.json --resume
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from eval import evaluate
from modules.super_net import SuperNet
from search.search import SearchSpace
from utils.data_handler import build_dataloader


def parse_args():
    p = argparse.ArgumentParser()

    # supernet architecture (must match the checkpoint)
    p.add_argument("--img-size",    type=int, default=32)
    p.add_argument("--patch-size",  type=int, default=4)
    p.add_argument("--embed-dim",   type=int, default=512)
    p.add_argument("--num-layers",  type=int, default=6)
    p.add_argument("--num-heads",   type=int, default=8)
    p.add_argument("--mlp-dim",     type=int, default=1024)
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--dropout",     type=float, default=0.1)

    # search space options
    p.add_argument("--embed-dim-options",  type=int, nargs="+", default=[512])
    p.add_argument("--num-layers-options", type=int, nargs="+", default=[2, 4, 6])
    p.add_argument("--num-heads-options",  type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--mlp-dim-options",    type=int, nargs="+", default=[256, 512, 1024])

    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--n-samples",  type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--output",     type=str, default="predictor_data.json")
    p.add_argument("--resume",     action="store_true",
                   help="Append to existing output file instead of overwriting")
    p.add_argument("--seed",       type=int, default=42)

    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- load supernet ---
    model = SuperNet(
        img_size=args.img_size, patch_size=args.patch_size,
        embed_dim=args.embed_dim, num_layers=args.num_layers,
        num_heads=args.num_heads, mlp_dim=args.mlp_dim,
        num_classes=args.num_classes, dropout=args.dropout,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state.get("model", state))
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # --- search space ---
    ss = SearchSpace(
        embed_dim_options=args.embed_dim_options,
        num_heads_options=args.num_heads_options,
        mlp_dim_options=args.mlp_dim_options,
        num_layers_options=args.num_layers_options,
    )
    print(f"Search space size: {ss.size:.1e}")

    # --- data loader (shared across all evals) ---
    _, test_loader, _ = build_dataloader(batch_size=args.batch_size,
                                         img_size=args.img_size)
    criterion = nn.CrossEntropyLoss()

    # --- resume: load existing records ---
    records = []
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            records = json.load(f)
        print(f"Resuming — loaded {len(records)} existing records from {args.output}")

    already_seen = {json.dumps(r["config"], sort_keys=True) for r in records}

    # --- sample and evaluate ---
    n_to_collect = args.n_samples - len(records)
    if n_to_collect <= 0:
        print(f"Already have {len(records)} samples — nothing to do.")
        return

    print(f"Collecting {n_to_collect} new samples …")
    pbar = tqdm(total=n_to_collect)
    attempts = 0
    while len(records) < args.n_samples:
        attempts += 1
        config = ss.sample_random_config()
        key    = json.dumps(config, sort_keys=True)
        if key in already_seen:
            continue
        already_seen.add(key)

        model.set_active_subnet(config)
        subnet = model.get_active_subnet().to(device)
        _, acc = evaluate(subnet, test_loader, criterion, device)

        records.append({"config": config, "accuracy": round(acc * 100, 4)})
        pbar.update(1)
        pbar.set_postfix(acc=f"{acc*100:.2f}%", attempts=attempts)

        # save incrementally every 10 samples
        if len(records) % 10 == 0:
            with open(args.output, "w") as f:
                json.dump(records, f, indent=2)

    pbar.close()

    with open(args.output, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} records → {args.output}")
    accs = [r["accuracy"] for r in records]
    print(f"Accuracy range: {min(accs):.2f}% – {max(accs):.2f}%  mean={np.mean(accs):.2f}%")


if __name__ == "__main__":
    main()
