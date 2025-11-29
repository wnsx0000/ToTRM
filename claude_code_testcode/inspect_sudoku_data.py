#!/usr/bin/env python3
"""
Sudoku 데이터셋의 형식을 확인하는 스크립트
"""
import numpy as np
import json
from pathlib import Path

def inspect_sudoku_dataset(data_dir: str):
    """Sudoku 데이터셋의 구조와 샘플을 출력"""
    data_path = Path(data_dir)

    # 1. dataset.json 메타데이터 읽기
    print("=" * 80)
    print("1. Dataset Metadata (dataset.json)")
    print("=" * 80)
    with open(data_path / "dataset.json", "r") as f:
        metadata = json.load(f)
    for key, value in metadata.items():
        print(f"{key:25s}: {value}")

    # 2. 각 numpy 파일 로드
    print("\n" + "=" * 80)
    print("2. Data Arrays Shape and Dtype")
    print("=" * 80)

    inputs = np.load(data_path / "all__inputs.npy", mmap_mode='r')
    labels = np.load(data_path / "all__labels.npy", mmap_mode='r')
    puzzle_identifiers = np.load(data_path / "all__puzzle_identifiers.npy", mmap_mode='r')
    puzzle_indices = np.load(data_path / "all__puzzle_indices.npy", mmap_mode='r')
    group_indices = np.load(data_path / "all__group_indices.npy", mmap_mode='r')

    print(f"inputs.shape              : {inputs.shape}")
    print(f"inputs.dtype              : {inputs.dtype}")
    print(f"labels.shape              : {labels.shape}")
    print(f"labels.dtype              : {labels.dtype}")
    print(f"puzzle_identifiers.shape  : {puzzle_identifiers.shape}")
    print(f"puzzle_identifiers.dtype  : {puzzle_identifiers.dtype}")
    print(f"puzzle_indices.shape      : {puzzle_indices.shape}")
    print(f"puzzle_indices.dtype      : {puzzle_indices.dtype}")
    print(f"group_indices.shape       : {group_indices.shape}")
    print(f"group_indices.dtype       : {group_indices.dtype}")

    # 3. 첫 번째 샘플 출력
    print("\n" + "=" * 80)
    print("3. First Sample (index 0)")
    print("=" * 80)

    sample_idx = 0
    input_sample = inputs[sample_idx].copy()
    label_sample = labels[sample_idx].copy()
    puzzle_id = puzzle_identifiers[sample_idx].copy()
    puzzle_idx = puzzle_indices[sample_idx]

    print(f"Sample index              : {sample_idx}")
    print(f"Puzzle identifier         : {puzzle_id}")
    print(f"Puzzle index              : {puzzle_idx}")
    print(f"\nInput sequence (length {len(input_sample)}):")
    print(input_sample)
    print(f"\nLabel sequence (length {len(label_sample)}):")
    print(label_sample)

    # 4. Sudoku를 9x9 그리드로 시각화
    print("\n" + "=" * 80)
    print("4. Sudoku Grid Visualization (First Sample)")
    print("=" * 80)

    # Sudoku는 9x9 = 81 셀
    if len(input_sample) == 81:
        print("\nInput Grid (0 = blank):")
        print_sudoku_grid(input_sample)

        print("\nLabel Grid (solution):")
        print_sudoku_grid(label_sample)

    # 5. 값의 분포 확인
    print("\n" + "=" * 80)
    print("5. Value Distribution")
    print("=" * 80)

    unique_inputs, counts_inputs = np.unique(input_sample, return_counts=True)
    unique_labels, counts_labels = np.unique(label_sample, return_counts=True)

    print("Input values and counts:")
    for val, count in zip(unique_inputs, counts_inputs):
        print(f"  {val}: {count} times")

    print("\nLabel values and counts:")
    for val, count in zip(unique_labels, counts_labels):
        print(f"  {val}: {count} times")

    # 6. 여러 샘플 확인
    print("\n" + "=" * 80)
    print("6. Additional Samples")
    print("=" * 80)

    for idx in [1, 2, 100, 500]:
        if idx < len(inputs):
            print(f"\nSample {idx}:")
            print(f"  Puzzle ID: {puzzle_identifiers[idx]}")
            print(f"  Non-zero inputs: {np.count_nonzero(inputs[idx])} / 81")
            print(f"  Input grid:")
            print_sudoku_grid(inputs[idx].copy())


def print_sudoku_grid(values):
    """9x9 Sudoku 그리드를 보기 좋게 출력"""
    grid = values.reshape(9, 9)

    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("  " + "-" * 21)

        row_str = ""
        for j in range(9):
            if j % 3 == 0 and j != 0:
                row_str += "| "

            val = grid[i, j]
            if val == 0:
                row_str += ". "
            else:
                row_str += f"{val} "

        print("  " + row_str)


if __name__ == "__main__":
    data_dir = "/home/wnsx0000/jhun/ToTRM/data/sudoku-extreme-1k-aug-1000-eval-10pct/train"
    inspect_sudoku_dataset(data_dir)
