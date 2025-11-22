"""
Create a smaller evaluation dataset by subsampling test data.
This keeps train data as-is but creates a smaller test set for faster evaluation.

Usage:
    python dataset/build_sudoku_dataset_small_eval.py \\
        --source-dir data/sudoku-extreme-1k-aug-1000 \\
        --output-dir data/sudoku-extreme-1k-aug-1000-eval-10pct \\
        --test-subsample-ratio 0.1
"""

from typing import Optional
import os
import json
import numpy as np
import shutil
from pathlib import Path

from argdantic import ArgParser
from pydantic import BaseModel

from common import PuzzleDatasetMetadata


cli = ArgParser()


class SubsampleConfig(BaseModel):
    source_dir: str = "data/sudoku-extreme-1k-aug-1000"
    output_dir: str = "data/sudoku-extreme-1k-aug-1000-eval-10pct"
    test_subsample_ratio: float = 0.1
    seed: int = 42


def subsample_test_data(config: SubsampleConfig):
    """
    Create a new dataset with subsampled test data.
    Train data is copied as-is.
    """
    np.random.seed(config.seed)
    
    source_path = Path(config.source_dir)
    output_path = Path(config.output_dir)
    
    print(f"{'='*70}")
    print(f"Creating Small Evaluation Dataset")
    print(f"{'='*70}")
    print(f"Source:      {config.source_dir}")
    print(f"Output:      {config.output_dir}")
    print(f"Test ratio:  {config.test_subsample_ratio*100:.1f}%")
    print(f"Seed:        {config.seed}")
    print(f"{'='*70}\n")
    
    # Create output directories
    for split in ['train', 'test']:
        (output_path / split).mkdir(parents=True, exist_ok=True)
    
    # Copy identifiers.json
    identifiers_file = source_path / 'identifiers.json'
    if identifiers_file.exists():
        shutil.copy(identifiers_file, output_path / 'identifiers.json')
        print(f"✓ Copied identifiers.json\n")
    
    # Process each split
    for split in ['train', 'test']:
        print(f"Processing {split} split...")
        
        split_source = source_path / split
        split_output = output_path / split
        
        # Load metadata
        with open(split_source / 'dataset.json', 'r') as f:
            metadata = json.load(f)
        
        original_puzzles = metadata['total_puzzles']
        original_groups = metadata['total_groups']
        
        if split == 'train':
            # Copy train data as-is
            print(f"  Copying train data as-is ({original_puzzles:,} puzzles)...")
            
            for file in split_source.glob('*.npy'):
                shutil.copy(file, split_output / file.name)
                data = np.load(file)
                print(f"    ✓ {file.name}: {data.shape}")
            
            shutil.copy(split_source / 'dataset.json', split_output / 'dataset.json')
            print(f"  ✓ Train: {original_puzzles:,} puzzles (unchanged)\n")
        
        else:  # test
            num_samples = int(original_puzzles * config.test_subsample_ratio)
            print(f"  Subsampling: {original_puzzles:,} → {num_samples:,} puzzles ({config.test_subsample_ratio*100:.1f}%)")
            
            # Load all data
            data_dict = {}
            for file in split_source.glob('all__*.npy'):
                field_name = file.stem.replace('all__', '')
                data_dict[field_name] = np.load(file)
            
            total_count = len(data_dict['inputs'])
            
            # Random sample indices
            sample_indices = np.sort(np.random.choice(total_count, num_samples, replace=False))
            print(f"    Selected {len(sample_indices):,} random indices")
            
            # Subsample main arrays
            sampled_data = {
                'inputs': data_dict['inputs'][sample_indices],
                'labels': data_dict['labels'][sample_indices],
            }
            
            # Handle puzzle_identifiers
            puzzle_ids = []
            for idx in sample_indices:
                # Find which puzzle this sample belongs to
                puzzle_id = np.searchsorted(data_dict['puzzle_indices'], idx, side='right') - 1
                puzzle_ids.append(data_dict['puzzle_identifiers'][puzzle_id])
            sampled_data['puzzle_identifiers'] = np.array(puzzle_ids, dtype=data_dict['puzzle_identifiers'].dtype)
            
            # Reconstruct indices (each sample becomes its own puzzle)
            sampled_data['puzzle_indices'] = np.arange(num_samples + 1, dtype=data_dict['puzzle_indices'].dtype)
            sampled_data['group_indices'] = np.arange(num_samples + 1, dtype=data_dict['group_indices'].dtype)
            
            # Save subsampled data
            for field_name, data in sampled_data.items():
                output_file = split_output / f"all__{field_name}.npy"
                np.save(output_file, data)
                print(f"    ✓ Saved {field_name}: {data.shape}")
            
            # Update metadata
            new_metadata = PuzzleDatasetMetadata(
                seq_len=metadata['seq_len'],
                vocab_size=metadata['vocab_size'],
                pad_id=metadata['pad_id'],
                ignore_label_id=metadata['ignore_label_id'],
                blank_identifier_id=metadata['blank_identifier_id'],
                num_puzzle_identifiers=metadata['num_puzzle_identifiers'],
                total_groups=num_samples,
                mean_puzzle_examples=1.0,
                total_puzzles=num_samples,
                sets=metadata['sets']
            )
            
            with open(split_output / 'dataset.json', 'w') as f:
                json.dump(new_metadata.model_dump(), f, indent=2)
            
            print(f"  ✓ Test: {num_samples:,} puzzles ({config.test_subsample_ratio*100:.1f}% of original)\n")
    
    print(f"{'='*70}")
    print(f"✅ Small evaluation dataset created successfully!")
    print(f"{'='*70}")
    print(f"\nTo use this dataset, update your training script:")
    print(f"  data_paths_test=\"[{config.output_dir}]\"")
    print(f"\nDataset statistics:")
    
    # Print final statistics
    train_meta_path = output_path / 'train' / 'dataset.json'
    test_meta_path = output_path / 'test' / 'dataset.json'
    
    with open(train_meta_path, 'r') as f:
        train_meta = json.load(f)
    with open(test_meta_path, 'r') as f:
        test_meta = json.load(f)
    
    print(f"  Train: {train_meta['total_puzzles']:,} puzzles")
    print(f"  Test:  {test_meta['total_puzzles']:,} puzzles")
    
    # Estimate eval time
    batch_size = 128
    eval_batches = (test_meta['total_puzzles'] + batch_size - 1) // batch_size
    eval_time_min = eval_batches / 1.5 / 60
    
    print(f"\nEstimated eval time (at 1.5 it/s):")
    print(f"  ~{eval_batches:,} batches")
    print(f"  ~{eval_time_min:.1f} minutes")
    print(f"{'='*70}\n")


@cli.command(singleton=True)
def create_small_eval(config: SubsampleConfig):
    """Create a small evaluation dataset by subsampling test data."""
    subsample_test_data(config)


if __name__ == "__main__":
    cli()
