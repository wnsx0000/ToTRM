#!/bin/sh

# Quick test: Run 1 epoch to verify transfer learning setup works
# This will load the TRM checkpoint, freeze weights, and train for 1 epoch only

uv run python pretrain.py \
    arch=totrm \
    data_paths="[data/sudoku-extreme-1k-aug-1000-eval-01pct]" \
    evaluators="[]" \
    \
    +load_checkpoint="shared_checkpoints/TRM_3602/TRM_3602/step_65100" \
    freeze_weights=True \
    \
    epochs=3 \
    eval_interval=1 \
    lr=1e-3 \
    puzzle_emb_lr=1e-4 \
    weight_decay=1.0 \
    puzzle_emb_weight_decay=1.0 \
    \
    arch.H_layers=0 \
    arch.L_layers=2 \
    arch.H_cycles=3 \
    arch.L_cycles=6 \
    \
    ema=True \
    arch.hidden_size=512 \
    arch.num_heads=8 \
    arch.expansion=4 \
    arch.forward_dtype=bfloat16 \
    global_batch_size=2 \
    arch.halt_exploration_prob=0.1 \
    arch.halt_max_steps=16 \
    \
    arch.tree_branching_steps=5 \
    arch.tree_merge_method=mean \
    \
    +project_name="test-transfer" \
    +run_name="test-trm-to-totrm"
