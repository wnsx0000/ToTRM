#!/bin/sh
uv run python -m dataset.build_arc_dataset --input-file-prefix kaggle/combined/arc-agi --output-dir data/arc2concept-aug-1000 --subsets training2 evaluation2 concept --test-set-name evaluation2
