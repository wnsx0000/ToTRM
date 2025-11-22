#!/bin/sh

DISABLE_COMPILE=1 uv run python pretrain.py \
    arch=trm \
    data_paths="[data/sudoku4x4]" \
    arch.halt_exploration_prob=0.0 \
    arch.halt_max_steps=4 \
    arch.H_cycles=2 \
    arch.L_cycles=2 \
    arch.H_layers=0 \
    arch.L_layers=1 \
    arch.hidden_size=64 \
    arch.num_heads=2 \
    arch.expansion=2 \
    arch.puzzle_emb_ndim=4 \
    arch.forward_dtype=float32 \
    arch.puzzle_emb_len=4 \
    global_batch_size=512 \
    epochs=1500 \
    lr=0.001 \
    puzzle_emb_lr=0.01 \
    weight_decay=0.0 \
    puzzle_emb_weight_decay=0.0 \
    lr_warmup_steps=0 \
    eval_interval=1 \
    +entity="jhunforspeed-soongsil-univ" \
    +project_name="totrm-sudoku4x4"
