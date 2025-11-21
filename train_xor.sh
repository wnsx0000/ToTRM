#!/bin/sh
export WANDB_SYNC_INTERVAL=0.5 
export WANDB_FLUSH_INTERVAL=0.5 

run_name="xor_test"

uv run python pretrain.py \
    arch=trm \
    data_paths="[data/xor-simple]" \
    arch.halt_exploration_prob=0.0 \
    arch.halt_max_steps=2 \
    arch.H_cycles=1 \
    arch.L_cycles=2 \
    arch.H_layers=0 \
    arch.L_layers=1 \
    arch.hidden_size=64 \
    arch.num_heads=2 \
    arch.expansion=2 \
    arch.puzzle_emb_ndim=4 \
    arch.forward_dtype=float32 \
    arch.puzzle_emb_len=4 \
    global_batch_size=8 \
    epochs=2 \
    lr=0.0001 \
    puzzle_emb_lr=0.0001 \
    weight_decay=0.0 \
    puzzle_emb_weight_decay=0.0 \
    lr_warmup_steps=0 \
    eval_interval=1 \
    +run_name=xor_small_stable

