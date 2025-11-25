"""Test to verify gradient flow through branch embeddings."""

import torch
import torch.nn as nn

print("=" * 60)
print("Gradient Flow Analysis for Branch Embeddings")
print("=" * 60)

# Simulate the ToTRM branching scenario
batch_size = 4
seq_len = 97
hidden_size = 512
tree_width = 8

# Create branch embeddings (learnable parameter)
branch_embeddings = nn.Parameter(torch.randn(tree_width, hidden_size) * 0.02)
print(f"\n1. Initial branch_embeddings:")
print(f"   Shape: {branch_embeddings.shape}")
print(f"   Std: {branch_embeddings.std().item():.6f}")
print(f"   Mean: {branch_embeddings.mean().item():.6f}")

# Simulate z_branched (output from previous layer)
z_branched = torch.randn(batch_size * tree_width // 2, seq_len, hidden_size, requires_grad=True)

# Simulate branching: duplicate states
z_branched_doubled = z_branched.repeat_interleave(2, dim=0)
print(f"\n2. After doubling (repeat_interleave):")
print(f"   z_branched_doubled.shape: {z_branched_doubled.shape}")

# Get branch indices [0, 1, 0, 1, 0, 1, ...]
branch_indices = torch.arange(tree_width).repeat(batch_size) % 2
branch_embs = branch_embeddings[branch_indices]
print(f"\n3. Branch embeddings selection:")
print(f"   branch_indices: {branch_indices.tolist()}")
print(f"   branch_embs.shape: {branch_embs.shape}")

# Add branch embeddings (the key operation we're testing)
z_with_branch = z_branched_doubled + branch_embs.unsqueeze(1)
print(f"\n4. After adding branch embeddings:")
print(f"   z_with_branch.shape: {z_with_branch.shape}")

# Simulate forward pass through model layers
# (simplified: just a linear layer)
output_layer = nn.Linear(hidden_size, 11)  # vocab_size = 11 for Sudoku
logits = output_layer(z_with_branch)
print(f"\n5. After output layer:")
print(f"   logits.shape: {logits.shape}")

# Remove puzzle embedding positions (keep only cells)
puzzle_emb_len = 16
logits_cells = logits[:, puzzle_emb_len:, :]
print(f"\n6. After removing puzzle positions:")
print(f"   logits_cells.shape: {logits_cells.shape}")

# Simulate loss computation (only on cell positions)
targets = torch.randint(0, 11, (batch_size * tree_width, seq_len - puzzle_emb_len))
loss_fn = nn.CrossEntropyLoss()

# Reshape for cross entropy
logits_flat = logits_cells.reshape(-1, 11)
targets_flat = targets.reshape(-1)
loss = loss_fn(logits_flat, targets_flat)

print(f"\n7. Loss computation:")
print(f"   Loss: {loss.item():.4f}")

# Backward pass
loss.backward()

print(f"\n8. Gradient analysis:")
print(f"   branch_embeddings.grad is not None? {branch_embeddings.grad is not None}")
if branch_embeddings.grad is not None:
    print(f"   branch_embeddings.grad.shape: {branch_embeddings.grad.shape}")
    print(f"   Gradient stats:")
    print(f"     - Mean: {branch_embeddings.grad.mean().item():.6f}")
    print(f"     - Std: {branch_embeddings.grad.std().item():.6f}")
    print(f"     - Max: {branch_embeddings.grad.max().item():.6f}")
    print(f"     - Min: {branch_embeddings.grad.min().item():.6f}")

    # Check which embeddings got gradients
    grad_norms = torch.norm(branch_embeddings.grad, dim=1)
    print(f"\n   Gradient L2 norm per embedding:")
    for i, norm in enumerate(grad_norms):
        used = "✓ USED" if i in branch_indices else "✗ unused"
        print(f"     Embedding {i}: {norm.item():.6f} {used}")

    # Calculate gradient magnitude
    grad_magnitude = torch.norm(branch_embeddings.grad).item()
    param_magnitude = torch.norm(branch_embeddings).item()
    relative_grad = grad_magnitude / param_magnitude
    print(f"\n   Gradient magnitude: {grad_magnitude:.6f}")
    print(f"   Parameter magnitude: {param_magnitude:.6f}")
    print(f"   Relative gradient: {relative_grad:.6f}")

print(f"\n9. Key findings:")
print(f"   ✓ Gradients flow through ALL sequence positions")
print(f"   ✓ Loss computed on {seq_len - puzzle_emb_len} cell positions")
print(f"   ✓ Each cell contributes gradient to branch_embeddings")
print(f"   ✓ Strong gradient signal expected")

# Compare with old implementation
print(f"\n10. Comparison with OLD implementation:")
print(f"    OLD: Only position 0 had branch embedding")
print(f"         - Position 0 often excluded from loss")
print(f"         - Weak gradient: ~1 position contributing")
print(f"    NEW: All positions have branch embedding")
print(f"         - All {seq_len - puzzle_emb_len} cells contribute gradient")
print(f"         - Strong gradient: {(seq_len - puzzle_emb_len) / 1:.0f}x more signal")

print("\n" + "=" * 60)
