#!/bin/sh

uv run python pretrain.py \
    arch=totrm \
    data_paths="[data/arc2concept-aug-1000]" \
    arch.H_layers=0 \
    arch.L_layers=2 \
    arch.H_cycles=3 \
    arch.L_cycles=6 \
    ema=True \
    epochs=10000 \
    eval_interval=500 \
    lr=1e-4 \
    arch.hidden_size=512 \
    arch.num_heads=8 \
    arch.expansion=4 \
    arch.forward_dtype=bfloat16 \
    global_batch_size=128 \
    arch.halt_exploration_prob=0.1 \
    arch.halt_max_steps=16 \
    arch.tree_branching_steps=3 \
    arch.tree_merge_method=mean \
    +entity="jhunforspeed-soongsil-univ" \
    +project_name="totrm-arc2" \
