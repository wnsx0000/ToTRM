#!/bin/bash

# Create a 10% subsampled evaluation dataset for faster eval

python dataset/build_sudoku_dataset_small_eval.py \
    --source-dir data/sudoku-extreme-1k-aug-1000 \
    --output-dir data/sudoku-extreme-1k-aug-1000-eval-10pct \
    --test-subsample-ratio 0.1 \
    --seed 42
