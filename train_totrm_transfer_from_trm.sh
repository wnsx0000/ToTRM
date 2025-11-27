#!/bin/sh

# Transfer Learning: TRM -> ToTRM
# Load pretrained TRM checkpoint and train only ToTRM-specific parameters (branch_embeddings, etc.)

# DISABLE_COMPILE=1 

uv run python pretrain.py \
    arch=totrm \
    data_paths="[data/sudoku-extreme-1k-aug-1000-eval-2pct]" \
    evaluators="[]" \
    \
    +load_checkpoint="shared_checkpoints/TRM_3602/TRM_3602/step_65100" \
    freeze_weights=True \
    \
    epochs=20000 \
    eval_interval=500 \
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
    global_batch_size=128 \
    arch.halt_exploration_prob=0.1 \
    arch.halt_max_steps=16 \
    \
    arch.tree_branching_steps=3 \
    arch.tree_merge_method=max \
    +arch.adaptive_branch_scale=1.0 \
    +arch.diversity_weight=10.0 \
    +arch.diversity_margin=0.85 \
    \
    +entity="jhunforspeed-soongsil-univ" \
    +project_name="totrm-sudoku-extreme" \
    +run_name="trm-to-totrm-normscale0.2-maxmerge-diversityfixed-weight10.0-margin0.85"

# Transfer Learning Configuration:
# --------------------------------
# - load_checkpoint: Load TRM checkpoint (step 65100)
# - freeze_weights=True: Freeze all TRM weights, only train ToTRM-specific params
# - lr=1e-3: Higher learning rate for branch_embeddings (frozen params unaffected)
# - epochs=5000: Shorter training (only ~32K params to train vs 7M)
# - tree_branching_steps=5: Branch on first 5 L-cycles (width = 2^5 = 32)
#   - L_cycles=6, so last L-cycle doesn't branch
#   - This gives a good balance between exploration and efficiency
#
# Expected trainable parameters:
# - branch_embeddings: [32, 512] = 16,384 params (tree_branching_steps=5)
# - puzzle_emb: [num_puzzles, 512]
# Total: ~17K params (0.2% of full model)
#
# Comparison with from-scratch ToTRM:
# - From-scratch: ~7M params, 20k epochs, ~48 hours
# - Transfer: ~17K params, 5k epochs, ~6 hours (estimated)
# - Expected speedup: 8x faster
