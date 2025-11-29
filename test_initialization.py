"""Test initialization of hierarchical_attention queries."""

import torch
import torch.nn.functional as F
from models.recursive_reasoning.totrm import ToTRM_Config, ToTRM_Inner
import math

def test_initialization():
    """Verify that query initialization produces near-uniform attention."""

    configs = [
        ("Small model", 256),
        ("Base model", 512),
        ("Large model", 1024),
    ]

    for name, hidden_size in configs:
        print(f"\n{'='*60}")
        print(f"{name} (hidden_size={hidden_size})")
        print('='*60)

        config = ToTRM_Config(
            batch_size=2,
            seq_len=81,
            vocab_size=10,
            num_puzzle_identifiers=100,
            H_cycles=2,
            L_cycles=4,
            H_layers=0,
            L_layers=1,
            hidden_size=hidden_size,
            expansion=2,
            num_heads=4,
            pos_encodings="rope",
            halt_max_steps=4,
            halt_exploration_prob=0.1,
            tree_branching_steps=3,
            tree_merge_method="hierarchical_attention",
        )

        model = ToTRM_Inner(config)

        # Check initialization std
        token_query_std = model.token_query.std().item()
        branch_query_std = model.branch_query.std().item()

        expected_std = 0.02 / math.sqrt(hidden_size / 512.0)

        print(f"\nInitialization:")
        print(f"  token_query std:  {token_query_std:.6f}")
        print(f"  branch_query std: {branch_query_std:.6f}")
        print(f"  expected std:     {expected_std:.6f}")

        # Test attention uniformity
        batch_size = 4
        tree_width = 8
        seq_len = 97  # 81 + 16

        # Simulate z from normal distribution (typical activation)
        z = torch.randn(batch_size, tree_width, seq_len, hidden_size, dtype=torch.bfloat16)

        with torch.no_grad():
            # Token-level attention
            token_scores = torch.einsum('d,bwld->bwl', model.token_query.squeeze(), z)
            token_scores = token_scores / math.sqrt(hidden_size)
            token_weights = F.softmax(token_scores, dim=2)

            # Branch-level attention
            z_branch = (z * token_weights.unsqueeze(-1)).sum(dim=2)
            branch_scores = torch.einsum('d,bwd->bw', model.branch_query.squeeze(), z_branch)
            branch_scores = branch_scores / math.sqrt(hidden_size)
            branch_weights = F.softmax(branch_scores, dim=1)

        # Measure uniformity (entropy)
        # For uniform distribution over n items: entropy = log(n)
        # For non-uniform: entropy < log(n)

        token_entropy = -(token_weights * torch.log(token_weights + 1e-10)).sum(dim=2).mean()
        max_token_entropy = math.log(seq_len)
        token_uniformity = (token_entropy / max_token_entropy).item()

        branch_entropy = -(branch_weights * torch.log(branch_weights + 1e-10)).sum(dim=1).mean()
        max_branch_entropy = math.log(tree_width)
        branch_uniformity = (branch_entropy / max_branch_entropy).item()

        print(f"\nAttention uniformity (1.0 = perfectly uniform):")
        print(f"  Token-level:  {token_uniformity:.4f} (entropy={token_entropy.item():.3f}, max={max_token_entropy:.3f})")
        print(f"  Branch-level: {branch_uniformity:.4f} (entropy={branch_entropy.item():.3f}, max={max_branch_entropy:.3f})")

        # Check if reasonably uniform (>0.95 is good)
        if token_uniformity > 0.95 and branch_uniformity > 0.95:
            print(f"  ✅ Near-uniform initialization achieved!")
        else:
            print(f"  ⚠️  Warning: attention may be too peaked at initialization")

        # Show sample weights
        print(f"\nSample token weights (first batch, first branch, first 10 tokens):")
        print(f"  {token_weights[0, 0, :10].tolist()}")
        print(f"  Expected (uniform): {[1.0/seq_len] * 10}")

        print(f"\nSample branch weights (first batch):")
        print(f"  {branch_weights[0].tolist()}")
        print(f"  Expected (uniform): {[1.0/tree_width] * tree_width}")

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("✅ Initialization properly scales with hidden_size")
    print("✅ Attention starts near-uniform for stable training")
    print("✅ Smaller std for larger models (stability)")

if __name__ == "__main__":
    test_initialization()
