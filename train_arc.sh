#!/bin/sh
run_name="pretrain_att_arc2concept_4"
uv run python pretrain.py arch=trm data_paths="[data/arc2concept-aug-1000]" arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 +run_name=${run_name} ema=True
