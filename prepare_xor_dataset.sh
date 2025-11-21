#!/bin/sh
uv run python dataset/build_xor_dataset.py --output-dir data/xor-simple --num-train-samples 1000 --num-test-samples 200
