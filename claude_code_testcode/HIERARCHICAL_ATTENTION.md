# Hierarchical Attention Merge for ToTRM

## Overview

`hierarchical_attention` is a new tree merge method that uses **two-level attention** to select the best reasoning branches in a content-aware manner.

## Motivation

Existing merge methods have limitations:

| Method | Limitation |
|--------|------------|
| `mean` | Wrong branches dilute correct ones |
| `max` | Only 1 branch contributes, wastes others |
| `learned_weighted` | **Position-based** (same weights for all batches) |
| `geometric_mean` | Unstable with zero-crossing |
| `gated_product` | Still position-based |

**Key Innovation**: `hierarchical_attention` is **content-based** - it evaluates each branch's actual reasoning quality, not just its position in the tree.

## Architecture

### Two-Level Attention

```
Level 1: Token-Level Attention
  ↓ "Which tokens are important clues?"
  ↓ (Pools each branch into a summary vector)

Level 2: Branch-Level Attention
  ↓ "Which reasoning path is most logical?"
  ↓ (Selects best branches to merge)

Final Merge: Weighted combination
```

### Implementation

```python
# Step 1: Token-level pooling (per branch)
token_scores = token_query • z                    # [B, W, L]
token_weights = softmax(token_scores)             # Focus on important tokens
z_branch = weighted_sum(z, token_weights)         # [B, W, D] - branch summaries

# Step 2: Branch-level selection
branch_scores = branch_query • z_branch           # [B, W]
branch_weights = softmax(branch_scores)           # Select best branches

# Step 3: Final merge
z_merged = weighted_sum(z, branch_weights)        # [B, L, D]
```

### Key Design Decisions

#### No Explicit Masking

The model does **not** explicitly mask empty cells. Instead:

1. **Empty cell embeddings** (token ID 1) are uninformative → naturally get low attention
2. **Filled cells** have meaningful patterns → naturally get high attention
3. **Works across recursive cycles**: As reasoning progresses, attention adapts dynamically

This is better than explicit masking because:
- Compatible with recursive reasoning (attention adapts each H-cycle)
- Simpler implementation
- More flexible (model learns what matters)

#### Parameter Efficiency

Only **2 × hidden_size** parameters:
- `token_query`: [1, 1, D]
- `branch_query`: [1, 1, D]

For `hidden_size=512`: only **1024 parameters** (~0.001M)

Compare to Linear layer alternative: `D × D = 262,144 parameters` (256× larger!)

## Usage

### Training

Edit `config/arch/totrm.yaml`:

```yaml
tree_merge_method: hierarchical_attention
```

Or override from command line:

```bash
uv run python pretrain.py \
  arch=totrm \
  arch.tree_merge_method=hierarchical_attention \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  ...
```

### Evaluation

The merge method is saved in the checkpoint's config, so no changes needed:

```bash
uv run python evaluate.py \
  --data-path data/sudoku4x4/ \
  --config checkpoints/totrm/run-name/all_config.yaml \
  --checkpoint checkpoints/totrm/run-name/final_step_45/model.pt
```

## Expected Benefits

### Sudoku
- Better performance on **hard puzzles** (few clues)
- Branch selection based on logical consistency
- Focus on constraint-satisfying reasoning paths

### ARC-AGI
- Identify important visual patterns
- Select transformations that match task structure

### Rubik's Cube
- Focus on key positions (corners, edges)
- Prefer moves that reduce disorder

## Testing

Verify implementation:

```bash
source /home/wnsx0000/anaconda3/etc/profile.d/conda.sh
conda activate trm
uv run python test_hierarchical_attention.py
uv run python test_full_forward.py
```

Both should output `✅ All tests passed!`

## Comparison with Baselines

Recommended ablation study:

1. **Baseline**: `tree_merge_method=mean`
2. **Position-based learning**: `tree_merge_method=learned_weighted`
3. **Content-based (ours)**: `tree_merge_method=hierarchical_attention`

Expected ranking on hard Sudoku:
```
hierarchical_attention > learned_weighted > mean
```

## Implementation Details

### Files Modified

- `models/recursive_reasoning/totrm.py`:
  - Added `hierarchical_attention` case in `__init__` (lines 197-206)
  - Added `hierarchical_attention` case in `_merge_tree` (lines 405-453)
  - Updated docstrings

- `config/arch/totrm.yaml`:
  - Updated `tree_merge_method` comment with new option

### Code Location

See `models/recursive_reasoning/totrm.py`:
- Initialization: Lines 197-206
- Merge logic: Lines 405-453

### Computational Cost

Compared to `mean` merge:
- **Extra parameters**: 2D (negligible, ~0.001M for D=512)
- **Extra compute per merge**: 2 einsum + 2 softmax (minimal overhead)
- **Memory**: Same (no extra storage needed)

## Future Improvements

Possible extensions (not implemented):

1. **Learnable temperature**: Add `nn.Parameter` to control attention sharpness
2. **Multi-head attention**: Use multiple token/branch queries for richer representations
3. **Cross-attention**: Let branches attend to each other before merging

## Citation

If you use this merge method, please cite the original ToTRM work and mention the hierarchical attention extension:

```
ToTRM with hierarchical attention merge - Content-based branch selection
using two-level attention (token → branch).
```
