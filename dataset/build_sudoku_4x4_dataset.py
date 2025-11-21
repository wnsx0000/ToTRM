"""
build_sudoku4x4_dataset.py - Generate 4x4 Sudoku dataset matching reference format
"""

import json
import numpy as np
import os
import random
from typing import List, Tuple
from argdantic import ArgParser
from pydantic import BaseModel

from common import PuzzleDatasetMetadata


cli = ArgParser()


class Sudoku4x4DataProcessConfig(BaseModel):
    output_dir: str = "data/sudoku4x4"
    grid_size: int = 4  # 4x4 Sudoku
    max_grid_size: int = 6  # Pad to 6x6
    num_train_puzzles: int = 2000
    num_test_puzzles: int = 400
    seed: int = 42
    num_examples_per_puzzle: int = 1  # One puzzle per group
    min_givens: int = 6  # Minimum numbers given
    max_givens: int = 10  # Maximum numbers given


def solve_sudoku_backtrack(grid: np.ndarray) -> Tuple[bool, np.ndarray]:
    """Solve a 4x4 Sudoku using backtracking. Returns (success, solution)."""
    solution = grid.copy()
    
    def is_valid(r, c, num):
        # Check row and column
        if num in solution[r, :] or num in solution[:, c]:
            return False
        # Check 2x2 box
        box_r, box_c = 2 * (r // 2), 2 * (c // 2)
        if num in solution[box_r:box_r+2, box_c:box_c+2]:
            return False
        return True
    
    def solve():
        for r in range(4):
            for c in range(4):
                if solution[r, c] == 0:
                    for num in range(1, 5):
                        if is_valid(r, c, num):
                            solution[r, c] = num
                            if solve():
                                return True
                            solution[r, c] = 0
                    return False
        return True
    
    success = solve()
    return success, solution


def generate_sudoku_puzzle(num_givens: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a valid Sudoku puzzle with specified number of givens."""
    # Start with a valid complete grid
    base = np.array([
        [1, 2, 3, 4],
        [3, 4, 1, 2],
        [2, 1, 4, 3],
        [4, 3, 2, 1]
    ], dtype=np.int32)
    
    # Shuffle to create variety
    # Permute numbers
    perm = np.arange(1, 5)
    np.random.shuffle(perm)
    complete = np.zeros_like(base)
    for i in range(4):
        complete[base == i+1] = perm[i]
    
    # Shuffle rows within bands
    if np.random.rand() > 0.5:
        complete[[0, 1]] = complete[[1, 0]]
    if np.random.rand() > 0.5:
        complete[[2, 3]] = complete[[3, 2]]
    
    # Shuffle columns within stacks
    if np.random.rand() > 0.5:
        complete[:, [0, 1]] = complete[:, [1, 0]]
    if np.random.rand() > 0.5:
        complete[:, [2, 3]] = complete[:, [3, 2]]
    
    # Create puzzle by removing cells
    puzzle = complete.copy()
    positions = [(r, c) for r in range(4) for c in range(4)]
    np.random.shuffle(positions)
    
    cells_to_remove = 16 - num_givens
    for i, (r, c) in enumerate(positions[:cells_to_remove]):
        puzzle[r, c] = 0
    
    return puzzle, complete


def pad_and_flatten_with_labels(inp: np.ndarray, out: np.ndarray, max_size: int):
    """Pad grids and create labels array with 0 for padding positions."""
    # For inputs: add 2 offset (0->PAD, 1->EOS, 2->empty, 3-6->values 1-4)
    inp_shifted = inp + 2
    out_shifted = out + 2
    
    # Create padded arrays
    inp_padded = np.zeros((max_size, max_size), dtype=np.int32)
    labels_padded = np.full((max_size, max_size), 0, dtype=np.int32)  # 0 for ignored positions
    
    # Fill in the actual data
    inp_padded[:4, :4] = inp_shifted
    labels_padded[:4, :4] = out_shifted
    
    # Add EOS marker
    if 4 < max_size:
        inp_padded[4, 0] = 1  # EOS
        labels_padded[4, 0] = 1  # EOS in labels too
    
    return inp_padded.flatten(), labels_padded.flatten()


def print_puzzle(puzzle: np.ndarray, solution: np.ndarray = None):
    """Pretty print a Sudoku puzzle and optionally its solution."""
    print("  Puzzle:")
    for i, row in enumerate(puzzle):
        if i == 2:
            print("  ------+------")
        row_str = "  "
        for j, val in enumerate(row):
            if j == 2:
                row_str += "| "
            if val == 0:
                row_str += ". "
            else:
                row_str += f"{val} "
        print(row_str)
    
    if solution is not None:
        print("\n  Solution:")
        for i, row in enumerate(solution):
            if i == 2:
                print("  ------+------")
            row_str = "  "
            for j, val in enumerate(row):
                if j == 2:
                    row_str += "| "
                row_str += f"{val} "
            print(row_str)


def print_encoded_data(encoded_input: np.ndarray, encoded_label: np.ndarray, max_size: int):
    """Print the encoded and padded data."""
    print("\n  Encoded input (flattened, first 24 values):")
    print(f"  {encoded_input[:24]}")
    print(f"  (0=PAD, 1=EOS, 2=empty, 3-6=values 1-4)")
    
    # Reshape to show as grid
    input_grid = encoded_input.reshape(max_size, max_size)
    print("\n  As 6x6 grid:")
    for row in input_grid:
        print(f"  {row}")


def generate_dataset(config: Sudoku4x4DataProcessConfig, split: str, print_samples: bool = False):
    """Generate Sudoku dataset following reference format."""
    np.random.seed(config.seed + (0 if split == "train" else 1000))
    random.seed(config.seed + (0 if split == "train" else 1000))
    
    num_puzzles = config.num_train_puzzles if split == "train" else config.num_test_puzzles
    
    # Dataset arrays
    all_inputs = []
    all_labels = []
    puzzle_identifiers = []
    puzzle_indices = [0]
    group_indices = [0]
    
    example_count = 0
    sample_puzzles = []  # Store a few for printing
    
    for puzzle_id in range(num_puzzles):
        if puzzle_id % 100 == 0:
            print(f"  Generating puzzle {puzzle_id}/{num_puzzles}")
        
        # Generate puzzle with varying difficulty
        num_givens = config.min_givens + (puzzle_id % (config.max_givens - config.min_givens + 1))
        puzzle, solution = generate_sudoku_puzzle(num_givens)
        
        # Store first few puzzles for printing
        if puzzle_id < 3:
            sample_puzzles.append((puzzle, solution, num_givens))
        
        # Convert to dataset format
        inp_flat, labels_flat = pad_and_flatten_with_labels(puzzle, solution, config.max_grid_size)
        
        all_inputs.append(inp_flat)
        all_labels.append(labels_flat)
        example_count += 1
        
        puzzle_indices.append(example_count)
        puzzle_identifiers.append(0)  # All sudoku puzzles have ID 0
        group_indices.append(puzzle_id + 1)
    
    # Print samples if requested
    if print_samples and sample_puzzles:
        print(f"\n{'='*50}")
        print(f"Sample {split.upper()} puzzles:")
        print(f"{'='*50}")
        for i, (puzzle, solution, givens) in enumerate(sample_puzzles[:2]):
            print(f"\nExample {i+1} (with {givens} given numbers):")
            print_puzzle(puzzle, solution)
            
            # Show encoded version for first example
            if i == 0:
                inp_flat, labels_flat = pad_and_flatten_with_labels(puzzle, solution, config.max_grid_size)
                print_encoded_data(inp_flat, labels_flat, config.max_grid_size)
    
    # Convert to numpy arrays
    results_np = {
        "inputs": np.array(all_inputs, dtype=np.int32),
        "labels": np.array(all_labels, dtype=np.int32),
        "puzzle_identifiers": np.array(puzzle_identifiers, dtype=np.int32),
        "puzzle_indices": np.array(puzzle_indices, dtype=np.int32),
        "group_indices": np.array(group_indices, dtype=np.int32)
    }
    
    # Create metadata matching reference format
    metadata = PuzzleDatasetMetadata(
        seq_len=config.max_grid_size * config.max_grid_size,  # 36
        vocab_size=7,  # 0-6: PAD, EOS, empty, 1-4
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=num_puzzles,
        mean_puzzle_examples=1,  # One per puzzle
        total_puzzles=num_puzzles,
        sets=["all"]
    )
    
    # Save dataset
    save_dir = os.path.join(config.output_dir, split)
    os.makedirs(save_dir, exist_ok=True)
    
    for key, value in results_np.items():
        np.save(os.path.join(save_dir, f"all__{key}.npy"), value)
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f, indent=2)
    
    print(f"\nGenerated {split} split:")
    print(f"  Puzzles: {num_puzzles}")
    print(f"  Examples: {example_count}")
    print(f"  Sequence length: {metadata.seq_len}")
    print(f"  Input shape: {results_np['inputs'].shape}")
    print(f"  Labels shape: {results_np['labels'].shape}")


@cli.command(singleton=True)
def main(config: Sudoku4x4DataProcessConfig):
    """Generate Sudoku 4x4 dataset."""
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    print("Generating Sudoku 4x4 dataset (reference format)...")
    print(f"Configuration:")
    print(f"  Grid: {config.grid_size}x{config.grid_size} -> {config.max_grid_size}x{config.max_grid_size}")
    print(f"  Given cells: {config.min_givens}-{config.max_givens}")
    
    # Generate with sample printing
    generate_dataset(config, "train", print_samples=True)
    generate_dataset(config, "test", print_samples=True)
    
    # Save identifiers file
    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>", "sudoku"], f)
    
    print("\n" + "="*50)
    print("Dataset generated successfully!")


if __name__ == "__main__":
    cli()