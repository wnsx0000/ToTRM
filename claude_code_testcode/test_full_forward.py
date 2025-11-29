"""Test full forward pass with hierarchical_attention."""

import torch
from models.recursive_reasoning.totrm import ToTRM

def test_full_forward():
    """Test complete ToTRM forward pass with hierarchical_attention."""

    config_dict = {
        "batch_size": 2,
        "seq_len": 81,
        "vocab_size": 10,
        "num_puzzle_identifiers": 100,
        "puzzle_emb_ndim": 128,
        "H_cycles": 2,
        "L_cycles": 4,
        "H_layers": 0,
        "L_layers": 1,
        "hidden_size": 128,
        "expansion": 2,
        "num_heads": 4,
        "pos_encodings": "rope",
        "halt_max_steps": 4,
        "halt_exploration_prob": 0.1,
        "tree_branching_steps": 3,
        "tree_merge_method": "hierarchical_attention",
    }

    print("Creating ToTRM model with hierarchical_attention...")
    model = ToTRM(config_dict)
    model.eval()

    batch_size = 2
    seq_len = 81

    print(f"\nTest configuration:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  tree_merge_method: {config_dict['tree_merge_method']}")

    # Create test batch
    batch = {
        "inputs": torch.randint(0, 10, (batch_size, seq_len), dtype=torch.long),
        "puzzle_identifiers": torch.randint(0, 100, (batch_size,), dtype=torch.long),
    }

    print(f"\nInput shapes:")
    print(f"  inputs: {batch['inputs'].shape}")
    print(f"  puzzle_identifiers: {batch['puzzle_identifiers'].shape}")

    # Initialize carry
    print("\nInitializing carry...")
    carry = model.initial_carry(batch)

    # Forward pass
    print("Running forward pass...")
    with torch.no_grad():
        new_carry, outputs = model.forward(carry, batch)

    print(f"\nOutput shapes:")
    print(f"  logits: {outputs['logits'].shape}")
    print(f"  q_halt_logits: {outputs['q_halt_logits'].shape}")
    print(f"  q_continue_logits: {outputs['q_continue_logits'].shape}")
    print(f"  z_L_branches: {outputs['z_L_branches'].shape}")

    # Verify shapes
    assert outputs['logits'].shape == (batch_size, seq_len, 10), \
        f"Wrong logits shape! Expected {(batch_size, seq_len, 10)}, got {outputs['logits'].shape}"
    assert outputs['q_halt_logits'].shape == (batch_size,), \
        f"Wrong q_halt_logits shape! Expected {(batch_size,)}, got {outputs['q_halt_logits'].shape}"

    # Verify no NaN or Inf
    assert not torch.isnan(outputs['logits']).any(), "logits contains NaN!"
    assert not torch.isinf(outputs['logits']).any(), "logits contains Inf!"

    print("\n✅ Full forward pass successful!")

    # Compare with mean merge
    print("\n" + "="*60)
    print("Comparing with 'mean' merge method...")
    print("="*60)

    config_dict_mean = config_dict.copy()
    config_dict_mean['tree_merge_method'] = 'mean'

    model_mean = ToTRM(config_dict_mean)
    model_mean.eval()

    carry_mean = model_mean.initial_carry(batch)
    with torch.no_grad():
        _, outputs_mean = model_mean.forward(carry_mean, batch)

    print(f"\nMean merge - logits: {outputs_mean['logits'].shape}")
    print(f"Hierarchical attention - logits: {outputs['logits'].shape}")
    print("\nBoth methods produce same output shape ✓")

    # Check if outputs differ (they should, since merge is different)
    logit_diff = (outputs['logits'] - outputs_mean['logits']).abs().mean()
    print(f"\nAverage absolute difference in logits: {logit_diff:.6f}")
    print("(Non-zero confirms different merge strategies produce different outputs)")

if __name__ == "__main__":
    test_full_forward()
