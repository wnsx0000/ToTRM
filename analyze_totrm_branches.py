#!/usr/bin/env python3
"""
Simplified ToTRM branch analysis.

Computes:
1. Cosine similarity between 8 z_L branches (8x8 matrix)
2. Cosine similarity between each z_L and final z_H (8 values)

Usage:
    uv run python analyze_totrm_branches.py \
        --checkpoint checkpoints/totrm-sudoku-extreme/totrm_251125/step_78120 \
        --data-path data/sudoku-extreme-1k-aug-1000 \
        --sample-idx 0
"""

import os
import json
import argparse
import yaml

import numpy as np
import torch
import torch.nn.functional as F

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from dataset.common import PuzzleDatasetMetadata
from utils.functions import load_model_class


def load_model_and_data(checkpoint_path, data_path, device):
    """Load model and dataset."""
    # Find checkpoint file
    if os.path.isfile(checkpoint_path):
        checkpoint_file = checkpoint_path
        run_dir = os.path.dirname(checkpoint_path)
    elif os.path.exists(os.path.join(checkpoint_path, "model.pt")):
        checkpoint_file = os.path.join(checkpoint_path, "model.pt")
        run_dir = os.path.dirname(checkpoint_path)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load config
    config_path = os.path.join(run_dir, "all_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load dataset metadata
    with open(os.path.join(data_path, "test", "dataset.json"), 'r') as f:
        metadata = PuzzleDatasetMetadata(**json.load(f))

    # Create model
    arch_config = config['arch']
    model_cfg = {
        **{k: v for k, v in arch_config.items() if k not in ['name', 'loss']},
        'batch_size': 1,
        'vocab_size': metadata.vocab_size,
        'seq_len': metadata.seq_len,
        'num_puzzle_identifiers': metadata.num_puzzle_identifiers,
    }
    model_cls = load_model_class(arch_config['name'])

    with torch.device(device):
        model = model_cls(model_cfg)

    # Load weights
    state_dict = torch.load(checkpoint_file, map_location=device)

    # Remove both '_orig_mod.' and 'model.' prefixes
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove _orig_mod. if present
        k = k.replace('_orig_mod.', '')
        # Remove model. prefix if present (checkpoint has 'model.inner.XXX', we need 'inner.XXX')
        k = k.replace('model.', '')
        new_state_dict[k] = v

    # Verify checkpoint structure
    checkpoint_keys = list(new_state_dict.keys())[:5]
    print(f"Sample checkpoint keys (after processing): {checkpoint_keys}")

    # Load and verify
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"WARNING: Missing keys in checkpoint: {missing[:5]}")
    if unexpected:
        print(f"WARNING: Unexpected keys in checkpoint: {unexpected[:5]}")

    model.to(device).eval()

    # Verify weights were loaded (check if parameters are not all zeros)
    sample_param = next(model.inner.L_level.parameters())
    print(f"Sample inner parameter mean: {sample_param.abs().mean().item():.6f} (should be > 0)")
    print(f"Model type: {type(model).__name__}")
    print(f"Inner model type: {type(model.inner).__name__}")

    # Load dataset
    dataset_config = PuzzleDatasetConfig(
        seed=42,
        dataset_paths=[data_path],
        global_batch_size=1,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1
    )
    dataset = PuzzleDataset(config=dataset_config, split="test")

    return model, dataset


def analyze_single_sample(model, batch, device):
    """
    Analyze one recursion step of ToTRM.

    Returns:
        - z_L_cosine: [8, 8] cosine similarity matrix between z_L branches
        - z_L_z_H_cosine: [8] cosine similarity between each z_L and final z_H
        - branch_embeddings: [8, hidden_size] branch embedding vectors
    """
    model.eval()
    inner_model = model.inner
    batch_size = batch["inputs"].shape[0]

    # Get branch embeddings (learned parameters added during branching)
    max_tree_width = 2 ** inner_model.config.tree_branching_steps
    branch_embeddings = inner_model.branch_embeddings[:max_tree_width].float().cpu().numpy()

    # Prepare inputs
    seq_info = dict(
        cos_sin=inner_model.rotary_emb() if hasattr(inner_model, "rotary_emb") else None,
    )
    input_embeddings = inner_model._input_embeddings(
        batch["inputs"], batch["puzzle_identifiers"]
    )

    # Initialize carry
    carry = inner_model.empty_carry(batch_size)
    z_H = carry.z_H.to(device)
    z_L = carry.z_L.to(device)
    current_tree_width = 1

    # Run L-cycles with branching
    for _L_step in range(inner_model.config.L_cycles):
        expanded_input = inner_model._expand_input_for_tree(input_embeddings, current_tree_width)
        expanded_z_H = inner_model._expand_input_for_tree(z_H, current_tree_width)
        z_L = inner_model.L_level(z_L, expanded_z_H + expanded_input, **seq_info)

        # Branch if needed
        if _L_step < inner_model.config.tree_branching_steps:
            z_L = inner_model._branch_state(z_L, current_tree_width)
            current_tree_width *= 2

    # Now we have z_L with shape [batch_size * 8, seq_len, hidden_size]
    # Reshape to [batch_size, 8, seq_len, hidden_size]
    z_L_branches = z_L.view(batch_size, current_tree_width, *z_L.shape[1:])

    # Flatten to [batch_size, 8, seq_len * hidden_size] for cosine similarity
    z_L_flat = z_L_branches.view(batch_size, current_tree_width, -1)

    # Normalize for cosine similarity
    z_L_norm = F.normalize(z_L_flat, dim=-1)  # [batch_size, 8, seq_len*hidden_size]

    # 1. Compute z_L branch-to-branch cosine similarity [batch_size, 8, 8]
    z_L_cosine = torch.bmm(z_L_norm, z_L_norm.transpose(1, 2))  # [batch_size, 8, 8]

    # Update z_H using L_level with merged z_L (ToTRM uses L_level for both)
    z_L_merged = inner_model._merge_tree(z_L, current_tree_width, batch_size)
    z_H_new = inner_model.L_level(z_H, z_L_merged, **seq_info)

    # Flatten z_H: [batch_size, seq_len, hidden_size] -> [batch_size, 1, seq_len*hidden_size]
    z_H_flat = z_H_new.view(batch_size, 1, -1)
    z_H_norm = F.normalize(z_H_flat, dim=-1)

    # 2. Compute z_L to z_H cosine similarity [batch_size, 8, 1]
    z_L_z_H_cosine = torch.bmm(z_L_norm, z_H_norm.transpose(1, 2)).squeeze(-1)  # [batch_size, 8]

    # Convert to float32 before numpy (bfloat16 not supported by numpy)
    return (z_L_cosine[0].float().cpu().numpy(),
            z_L_z_H_cosine[0].float().cpu().numpy(),
            branch_embeddings)


def main():
    parser = argparse.ArgumentParser(description="Simplified ToTRM branch analysis")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint")
    parser.add_argument("--data-path", type=str, required=True,
                       help="Path to dataset")
    parser.add_argument("--sample-idx", type=int, default=0,
                       help="Sample index to analyze")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")

    args = parser.parse_args()

    print("Loading model and dataset...")
    model, dataset = load_model_and_data(args.checkpoint, args.data_path, args.device)

    print(f"\nModel info:")
    print(f"  Tree branching steps: {model.inner.config.tree_branching_steps}")
    print(f"  Tree width: {2 ** model.inner.config.tree_branching_steps}")
    print(f"  Merge method: {model.inner.config.tree_merge_method}\n")

    # Get the specified sample
    with torch.no_grad():
        for idx, (set_name, batch, global_batch_size) in enumerate(dataset):
            if idx == args.sample_idx:
                batch = {k: v.to(args.device) for k, v in batch.items()}

                print(f"Analyzing sample {args.sample_idx}...")
                z_L_cosine, z_L_z_H_cosine, branch_embs = analyze_single_sample(model, batch, args.device)

                print(f"\n{'='*60}")
                print("1. Branch Embeddings (added during branching)")
                print(f"{'='*60}")
                print(f"Shape: {branch_embs.shape} [num_branches, hidden_size]")
                print(f"\nStatistics per branch:")
                for i in range(branch_embs.shape[0]):
                    emb = branch_embs[i]
                    # print(emb)
                    print(f"  Branch {i}: mean={emb.mean():.6f}, std={emb.std():.6f}, "
                          f"abs_mean={np.abs(emb).mean():.6f}, max={emb.max():.6f}, min={emb.min():.6f}")

                # Compute pairwise L2 distance between branch embeddings
                print(f"\nPairwise L2 distances between branch embeddings:")
                for i in range(min(4, branch_embs.shape[0])):  # Show first 4 pairs
                    for j in range(i+1, min(4, branch_embs.shape[0])):
                        dist = np.linalg.norm(branch_embs[i] - branch_embs[j])
                        print(f"  Branch {i} - Branch {j}: {dist:.6f}")

                print(f"\n{'='*60}")
                print("2. Cosine Similarity between 8 z_L branches (8x8 matrix):")
                print(f"{'='*60}")
                print(z_L_cosine)

                # Get off-diagonal elements (exclude self-similarity)
                mask = ~np.eye(8, dtype=bool)
                off_diag = z_L_cosine[mask]
                print(f"\nMean (off-diagonal): {off_diag.mean():.4f}")
                print(f"Std (off-diagonal): {off_diag.std():.4f}")

                print(f"\n{'='*60}")
                print("3. Cosine Similarity between each z_L and final z_H (8 values):")
                print(f"{'='*60}")
                for i, sim in enumerate(z_L_z_H_cosine):
                    print(f"  Branch {i}: {sim:.4f}")
                print(f"\nMean: {z_L_z_H_cosine.mean():.4f}")
                print(f"Std: {z_L_z_H_cosine.std():.4f}")

                break
            if idx > args.sample_idx:
                print(f"Sample {args.sample_idx} not found!")
                break


if __name__ == "__main__":
    main()
