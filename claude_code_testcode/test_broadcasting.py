"""Test to visualize how unsqueeze and broadcasting works."""

import torch

print("=" * 60)
print("Broadcasting Demonstration")
print("=" * 60)

# Simulate the branching scenario
batch_size = 4
seq_len = 97
hidden_size = 512

# z_branched: [8, 97, 512] - 8 branches (4 examples * 2 branches each)
z_branched = torch.randn(8, seq_len, hidden_size)

# branch_embs: [8, 512] - one embedding per branch
branch_embs = torch.randn(8, hidden_size)

print(f"\n1. Original shapes:")
print(f"   z_branched: {z_branched.shape}")
print(f"   branch_embs: {branch_embs.shape}")

# unsqueeze(1) adds a dimension at position 1
branch_embs_unsqueezed = branch_embs.unsqueeze(1)
print(f"\n2. After unsqueeze(1):")
print(f"   branch_embs.unsqueeze(1): {branch_embs_unsqueezed.shape}")
print(f"   Dimension 1 becomes 1 (singleton dimension)")

# Broadcasting explanation
print(f"\n3. Broadcasting addition:")
print(f"   z_branched:              [8, 97, 512]")
print(f"   branch_embs.unsqueeze(1): [8,  1, 512]")
print(f"                             ↓   ↓   ↓")
print(f"   Result:                  [8, 97, 512]")
print(f"\n   The [8, 1, 512] is broadcast to [8, 97, 512]")
print(f"   by copying the same vector 97 times along dimension 1")

# Perform the addition
result = z_branched + branch_embs_unsqueezed
print(f"\n4. Result shape: {result.shape}")

print(f"z_branched[0, 0, :5]: {z_branched[0, 0, :5].tolist()}...")
print(f"branch_embs[0, :5]: {branch_embs[0, :5].tolist()}...")
print(f"result[0, 0, :5]: {result[0, 0, :5].tolist()}...")

print(f"z_branched[0, 4, :5]: {z_branched[0, 4, :5].tolist()}...")
print(f"branch_embs[0, :5]: {branch_embs[0, :5].tolist()}...")
print(f"result[0, 4, :5]: {result[0, 4, :5].tolist()}...")

# Verify that the same embedding was added to all positions
print(f"\n5. Verification:")
print(f"   For branch 0:")
# Check first two positions for branch 0
diff_pos0 = (result[0, 0, :] - z_branched[0, 0, :])
diff_pos1 = (result[0, 1, :] - z_branched[0, 1, :])
diff_pos96 = (result[0, 96, :] - z_branched[0, 96, :])

print(f"   Added to position 0:  {diff_pos0[:5].tolist()} ...")
print(f"   Added to position 1:  {diff_pos1[:5].tolist()} ...")
print(f"   Added to position 96: {diff_pos96[:5].tolist()} ...")
print(f"   All equal? {torch.allclose(diff_pos0, diff_pos1) and torch.allclose(diff_pos1, diff_pos96)}")

print(f"\n6. Key insight:")
print(f"   ✓ Same branch_emb[i] added to ALL sequence positions for branch i")
print(f"   ✓ Different branches get different embeddings")
print(f"   ✓ Branch 0: one embedding, Branch 1: different embedding, etc.")

# Show difference between branches
print(f"\n7. Branch differentiation:")
emb_diff = branch_embs[0] - branch_embs[1]
print(f"   Difference between branch 0 and 1 embedding:")
print(f"   L2 norm: {torch.norm(emb_diff).item():.4f}")
print(f"   Mean absolute diff: {torch.abs(emb_diff).mean().item():.4f}")

print("\n" + "=" * 60)


