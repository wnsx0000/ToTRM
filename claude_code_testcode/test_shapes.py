"""Test script to print tensor shapes for Sudoku Extreme dataset."""

import torch
from pathlib import Path
from omegaconf import OmegaConf
from models.recursive_reasoning.totrm import ToTRM_Inner, ToTRM_Config
from puzzle_dataset import PuzzleDataset

# Load config
config_path = Path("config/arch/totrm.yaml")
config = OmegaConf.load(config_path)

# Set minimal values for testing
config.batch_size = 4  # Small batch for testing
config.hidden_size = 512
config.L_cycles = 6
config.H_cycles = 3
config.tree_branching_steps = 5
config.L_layers = 2
config.H_layers = 0
config.num_heads = 8
config.expansion = 4
config.forward_dtype = "bfloat16"
config.pos_encodings = "rope"
config.rope_theta = 10000
config.rms_norm_eps = 1e-5
config.mlp_t = False

# Load dataset metadata to get vocab_size and seq_len
import json
dataset_path = Path("data/sudoku-extreme-1k-aug-1000-eval-10pct/train")
with open(dataset_path / "dataset.json", "r") as f:
    dataset_meta = json.load(f)

config.vocab_size = dataset_meta["vocab_size"]
config.seq_len = dataset_meta["seq_len"]
config.num_puzzle_identifiers = dataset_meta["num_puzzle_identifiers"]
config.puzzle_emb_ndim = config.hidden_size
config.puzzle_emb_len = 16

print(f"Dataset info:")
print(f"  vocab_size: {config.vocab_size}")
print(f"  seq_len: {config.seq_len}")
print(f"  num_puzzle_identifiers: {config.num_puzzle_identifiers}")
print()

# Create model
model_config = ToTRM_Config(**config)
model = ToTRM_Inner(model_config)

# Create a dummy batch
batch = {
    "inputs": torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len)),
    "puzzle_identifiers": torch.zeros(config.batch_size, dtype=torch.int32),
    "labels": torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len)),
}

# Initialize carry state
from models.recursive_reasoning.totrm import ToTRM_InnerCarry
carry = model.empty_carry(batch_size=config.batch_size)

# Forward pass (will print debug info)
print("=" * 60)
print("Running forward pass with debug output:")
print("=" * 60)
with torch.no_grad():
    output = model(batch=batch, carry=carry)
print("=" * 60)
