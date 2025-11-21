#!/usr/bin/env python3
"""
Quick test script to verify ToTRM model loads correctly.
"""

import sys
import torch
from utils.functions import load_model_class

def test_totrm_import():
    """Test that ToTRM can be imported and instantiated."""
    print("Testing ToTRM import...")
    
    try:
        # Load the model class
        model_cls = load_model_class("recursive_reasoning.totrm@ToTRM")
        print(f"✓ Successfully loaded model class: {model_cls}")
        
        # Create a minimal config
        config = {
            'batch_size': 4,
            'seq_len': 36,  # 6x6 grid
            'puzzle_emb_ndim': 64,
            'num_puzzle_identifiers': 100,
            'vocab_size': 20,
            'H_cycles': 2,
            'L_cycles': 4,
            'H_layers': 0,
            'L_layers': 1,
            'hidden_size': 64,
            'expansion': 2.0,
            'num_heads': 2,
            'pos_encodings': 'rope',
            'halt_max_steps': 4,
            'halt_exploration_prob': 0.0,
            'forward_dtype': 'float32',
            'mlp_t': False,
            'puzzle_emb_len': 4,
            'no_ACT_continue': True,
            # ToTRM specific
            'tree_branching_steps': 3,
            'tree_merge_method': 'mean',
        }
        
        # Instantiate the model
        model = model_cls(config)
        print(f"✓ Successfully instantiated model")
        print(f"  Model type: {type(model)}")
        print(f"  Tree branching steps: {model.config.tree_branching_steps}")
        print(f"  Tree merge method: {model.config.tree_merge_method}")
        
        # Check model components
        print(f"\n✓ Model components:")
        print(f"  - Inner model: {type(model.inner)}")
        print(f"  - L_level layers: {len(model.inner.L_level.layers)}")
        
        # Create dummy batch
        batch = {
            'inputs': torch.randint(0, config['vocab_size'], (config['batch_size'], config['seq_len'])),
            'puzzle_identifiers': torch.randint(0, config['num_puzzle_identifiers'], (config['batch_size'],)),
            'targets': torch.randint(0, config['vocab_size'], (config['batch_size'], config['seq_len'])),
        }
        
        # Test initial carry
        carry = model.initial_carry(batch)
        print(f"\n✓ Initial carry created:")
        print(f"  - z_H shape: {carry.inner_carry.z_H.shape}")
        print(f"  - z_L shape: {carry.inner_carry.z_L.shape}")
        print(f"  - tree_width: {carry.inner_carry.tree_width}")
        
        # Test forward pass
        print(f"\n✓ Testing forward pass...")
        model.eval()
        with torch.no_grad():
            new_carry, outputs = model(carry, batch)
        
        print(f"✓ Forward pass successful!")
        print(f"  - Output logits shape: {outputs['logits'].shape}")
        print(f"  - New tree width: {new_carry.inner_carry.tree_width}")
        
        # Calculate expected tree width during computation
        expected_max_width = 2 ** config['tree_branching_steps']
        print(f"\n✓ Tree-of-Thought structure:")
        print(f"  - Branching steps: {config['tree_branching_steps']}")
        print(f"  - Max tree width: {expected_max_width}")
        print(f"  - L_cycles: {config['L_cycles']}")
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nToTRM is ready to use!")
        print(f"\nTo train on Sudoku 4x4:")
        print(f"  1. Generate data: uv run python dataset/build_sudoku_4x4_dataset.py")
        print(f"  2. Train model: ./train_totrm_sudoku4x4.sh")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_totrm_import()
    sys.exit(0 if success else 1)
