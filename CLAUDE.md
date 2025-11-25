# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository implements **ToTRM (Tree-of-Thought Recursive Model)**, an extension of the Tiny Recursion Model (TRM) architecture. It's built on top of the original TRM implementation with simplified setup using `uv` package manager and better checkpoint management. The models are trained to solve complex reasoning tasks like Sudoku, Rubik's Cube, and ARC-AGI challenges using recursive reasoning with small neural networks (~7M parameters).

## Key Architecture Concepts

### TRM (Tiny Recursion Model)
The baseline recursive reasoning model with two levels of latent states:
- **z_H**: High-level reasoning state
- **z_L**: Low-level reasoning state updated recursively

The model performs `H_cycles` high-level updates, with each containing `L_cycles` low-level recursive updates. During training, only the final H-cycle uses gradients to save memory (first H_cycles-1 run with `torch.no_grad()`).

### ToTRM (Tree-of-Thought Recursive Model)
Extends TRM by creating a binary tree of reasoning states during L-cycle updates:
- **Branching**: On the first `tree_branching_steps` L-cycles, each z_L state branches into 2 copies with unique learned embeddings
- **Merging**: After all L-cycles complete, the tree is merged using one of: `mean`, `max`, or `learned_weighted`
- Tree width grows as 2^n where n = number of branching steps
- Enables parallel exploration of multiple reasoning paths

### ACT (Adaptive Computation Time)
Both models use Q-learning based halting:
- **q_halt_logits**: Learns when to stop iterating
- **halt_max_steps**: Maximum inference iterations (16 by default)
- **halt_exploration_prob**: Exploration probability during training (0.1 default)
- During training, models can halt early based on Q-values; during eval, always runs max_steps

## Common Development Commands

### Environment Setup
```bash
# Uses uv for dependency management (Python 3.11+)
uv sync
```

### Dataset Preparation
```bash
# Sudoku 4x4 (simple, for debugging)
uv run python dataset/build_sudoku_4x4_dataset.py

# Sudoku Extreme (9x9)
uv run python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000

# Rubik's Cube 2x2x2
uv run dataset/build_rubik2x2_dataset.py

# ARC-AGI datasets
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation
```

### Training

Training uses Hydra for configuration management. Config files are in `config/`:
- `config/cfg_pretrain.yaml`: Base training config
- `config/arch/*.yaml`: Model architecture configs (trm, totrm, hrm, etc.)

**Basic training pattern:**
```bash
# Single GPU
uv run python pretrain.py arch=<model> data_paths="[<path>]" [additional args...]

# Multi-GPU (distributed)
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
  pretrain.py arch=<model> data_paths="[<path>]" [args...]
```

**Common training scripts:**
```bash
# TRM on Sudoku 4x4 (fast, for testing)
./train_sudoku4x4.sh

# ToTRM on Sudoku Extreme
./train_totrm_sudoku_extreme.sh

# Rubik's Cube 2x2
./train_rubik2x2.sh
```

**Key hyperparameters to override:**
- `arch=trm|totrm|hrm`: Model architecture
- `global_batch_size=<int>`: Total batch size across all GPUs
- `epochs=<int>`: Training epochs
- `eval_interval=<int>`: Evaluate every N epochs
- `lr=<float>`: Learning rate (typically 1e-4)
- `ema=True|False`: Use Exponential Moving Average
- `arch.H_cycles=<int>`: High-level recursion cycles
- `arch.L_cycles=<int>`: Low-level recursion cycles
- `arch.tree_branching_steps=<int>`: ToTRM branching steps (n-1 where n=L_cycles)
- `arch.halt_max_steps=<int>`: Max ACT iterations

### Evaluation
```bash
uv run python evaluate.py \
  --data-path data/sudoku4x4/ \
  --config checkpoints/trm/run-name/all_config.yaml \
  --checkpoint checkpoints/trm/run-name/final_step_45/model.pt
```

### Checkpoint Management

The codebase uses `CheckpointManager` (in `utils/checkpoint_utils.py`) for organized checkpoint saving:
- Checkpoints saved to `checkpoints/<project_name>/<run_name>/`
- Final checkpoint: `final_step_<N>/model.pt`
- Config saved as: `all_config.yaml`
- When using EMA, both EMA and non-EMA weights are saved

## Code Structure

### Model Architecture (`models/`)
- `models/recursive_reasoning/trm.py`: TRM implementation
- `models/recursive_reasoning/totrm.py`: ToTRM with tree-of-thought branching
- `models/recursive_reasoning/hrm.py`: Hierarchical Reasoning Model baseline
- `models/losses.py`: Loss heads including ACT loss
- `models/layers.py`: Building blocks (Attention, SwiGLU, RMSNorm, etc.)
- `models/sparse_embedding.py`: Puzzle-specific embeddings
- `models/ema.py`: Exponential Moving Average helper

### Training Pipeline (`pretrain.py`)
Main training loop handles:
1. **Distributed setup**: Multi-GPU via PyTorch DDP (NCCL backend)
2. **Carry state**: Models maintain recursive state across forward passes
3. **Gradient accumulation**: Loss divided by `global_batch_size` before backward
4. **Optimizer management**: Separate optimizers for model weights vs puzzle embeddings
5. **Cosine LR schedule**: With warmup
6. **Checkpoint saving**: Via `CheckpointManager` at eval intervals and final step

### Dataset (`puzzle_dataset.py`, `dataset/`)
- `PuzzleDataset`: Iterable dataset with memory-mapped numpy arrays
- Each puzzle has: `inputs`, `labels`, `puzzle_identifiers`
- Batching groups examples by puzzle to enable efficient training
- Metadata stored in `dataset.json` (vocab_size, seq_len, etc.)

### Key Training Details
- **Forward dtype**: Uses `bfloat16` for efficiency (configurable via `arch.forward_dtype`)
- **Gradient clipping**: Max norm 1.0
- **Model compilation**: `torch.compile()` enabled by default (disable with `DISABLE_COMPILE=1`)
- **Loss computation**: Cross-entropy with ACT Q-learning loss
- **Carry detachment**: Recursive states detached after each forward pass to prevent gradient accumulation

## Model Loading and Inference

When loading checkpoints:
1. Model may be compiled (`_orig_mod.` prefix in state dict keys)
2. Puzzle embeddings may need resizing if `num_puzzle_identifiers` differs
3. For EMA models, use the EMA checkpoint (not the non-EMA version)

See `evaluate.py` for reference implementation of checkpoint loading.

## Special Configuration Notes

### Hydra Config Overrides
Use `+key=value` to add new keys (e.g., `+run_name=my_experiment`).
Use `key=value` to override existing keys.

### Weights & Biases Integration
Set `use_wandb=true` and configure:
- `+entity=<wandb-entity>`
- `+project_name=<project>`
- `+run_name=<run>` (auto-generated if not specified)

### Device Support
- **CUDA**: Full support with distributed training
- **MPS** (Apple Silicon): Single device only, no distributed
- **CPU**: Fallback, torch.compile disabled

## Debugging Tips

- Use `DISABLE_COMPILE=1` to disable torch.compile for easier debugging
- Reduce model size for quick iterations: small `hidden_size`, fewer `H_cycles`/`L_cycles`
- Check `pretrain.py` lines 340-363 for NaN/Inf detection logic
- The training script logs every 10 steps and creates progress bars
- Eval progress printed every 10 batches with time estimates
