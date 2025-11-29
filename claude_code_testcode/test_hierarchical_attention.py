"""Test script for hierarchical_attention merge method."""

import torch
from models.recursive_reasoning.totrm import ToTRM_Config, ToTRM_Inner

def test_hierarchical_attention_merge():
    """Test that hierarchical_attention merge works correctly."""

    # Create minimal config for testing
    config = ToTRM_Config(
        batch_size=2,
        seq_len=81,
        vocab_size=10,
        num_puzzle_identifiers=100,
        H_cycles=2,
        L_cycles=4,
        H_layers=0,
        L_layers=1,
        hidden_size=128,
        expansion=2,
        num_heads=4,
        pos_encodings="rope",
        halt_max_steps=4,
        halt_exploration_prob=0.1,
        tree_branching_steps=3,  # 2^3 = 8 branches
        tree_merge_method="hierarchical_attention",
    )

    print("Creating ToTRM_Inner model...")
    model = ToTRM_Inner(config)
    model.eval()

    # Test shapes
    batch_size = 2
    tree_width = 2 ** config.tree_branching_steps  # 8
    seq_len = config.seq_len + model.puzzle_emb_len  # 81 + 16 = 97
    hidden_size = config.hidden_size

    print(f"\nTest configuration:")
    print(f"  batch_size: {batch_size}")
    print(f"  tree_width: {tree_width}")
    print(f"  seq_len (with puzzle_emb): {seq_len}")
    print(f"  hidden_size: {hidden_size}")

    # Create test input
    z = torch.randn(batch_size * tree_width, seq_len, hidden_size, dtype=torch.bfloat16)
    print(f"\nInput z shape: {z.shape}")
    print(f"  Expected: [{batch_size * tree_width}, {seq_len}, {hidden_size}]")

    # Test merge
    print("\nTesting _merge_tree with hierarchical_attention...")
    with torch.no_grad():
        z_merged = model._merge_tree(z, tree_width, batch_size)

    print(f"Output z_merged shape: {z_merged.shape}")
    print(f"  Expected: [{batch_size}, {seq_len}, {hidden_size}]")

    # Verify shape
    assert z_merged.shape == (batch_size, seq_len, hidden_size), \
        f"Wrong output shape! Expected {(batch_size, seq_len, hidden_size)}, got {z_merged.shape}"

    # Verify no NaN or Inf
    assert not torch.isnan(z_merged).any(), "Output contains NaN!"
    assert not torch.isinf(z_merged).any(), "Output contains Inf!"

    print("\n✅ All tests passed!")
    print("\nParameter summary:")
    print(f"  token_query: {model.token_query.shape}")
    print(f"  branch_query: {model.branch_query.shape}")

    # Count parameters
    token_params = model.token_query.numel()
    branch_params = model.branch_query.numel()
    total_params = token_params + branch_params

    print(f"\nNew parameters for hierarchical_attention:")
    print(f"  token_query: {token_params:,} parameters")
    print(f"  branch_query: {branch_params:,} parameters")
    print(f"  Total: {total_params:,} parameters")
    print(f"  (Only {total_params / 1e6:.4f}M parameters for content-based merge!)")

if __name__ == "__main__":
    test_hierarchical_attention_merge()
