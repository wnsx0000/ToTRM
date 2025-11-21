#!/usr/bin/env python3
"""
Standalone evaluation script for puzzle model checkpoints.
This script loads a trained checkpoint and runs inference on sample examples.

Example: uv run python evaluate.py --data-path data/sudoku4x4/ --config checkpoints/trm/messy-earwig-of-enthusiasm/all_config.yaml --checkpoint checkpoints/trm/messy-earwig-of-enthusiasm/final_step_45/model.pt

"""

import os
import json
import argparse
import yaml
from typing import Optional, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Import required modules from your training code
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from utils.functions import load_model_class
from models.ema import EMAHelper


def load_config_from_checkpoint(checkpoint_dir: str) -> Dict[str, Any]:
    """Load configuration from checkpoint directory."""
    config_path = os.path.join(checkpoint_dir, "all_config.yaml")
    if os.path.exists(config_path):
        print(f"Loading config from {config_path}")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Config file not found at {config_path}")


def create_model_from_config(config: Dict[str, Any], vocab_size: int, seq_len: int, 
                            num_puzzle_identifiers: int, device: str = "cuda") -> nn.Module:
    """Create model instance from configuration."""
    arch_config = config['arch']
    
    # Model configuration
    model_cfg = dict(
        **{k: v for k, v in arch_config.items() if k not in ['name', 'loss']},
        batch_size=1,  # For evaluation, we'll use batch size 1
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_puzzle_identifiers=num_puzzle_identifiers,
        causal=False  # Non-autoregressive
    )
    
    # Load model and loss head classes
    model_cls = load_model_class(arch_config['name'])
    loss_head_cls = load_model_class(arch_config['loss']['name'])
    
    with torch.device(device):
        model = model_cls(model_cfg)
        # Filter out 'name' from loss config as it's not a constructor parameter
        loss_config = {k: v for k, v in arch_config['loss'].items() if k != 'name'}
        model = loss_head_cls(model, **loss_config)
        
    return model


def load_checkpoint_weights(model: nn.Module, checkpoint_path: str, device: str = "cuda"):
    """Load checkpoint weights into model."""
    print(f"Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Handle compiled model keys (remove '_orig_mod.' prefix if present)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[10:]] = v  # Remove '_orig_mod.' prefix
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    print("Checkpoint loaded successfully!")
    return model


def evaluate_examples(model: nn.Module, dataloader: DataLoader, 
                     num_examples: int = 5, device: str = "cuda", 
                     verbose: bool = True) -> Dict[str, float]:
    """Run evaluation on a few examples and collect metrics."""
    model.eval()
    
    all_metrics = {}
    example_outputs = []
    
    with torch.no_grad():
        for idx, (set_name, batch, global_batch_size) in enumerate(dataloader):
            if idx >= num_examples:
                break
                
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Initialize carry state
            with torch.device(device):
                carry = model.initial_carry(batch)
            
            # Run inference (non-autoregressive may require multiple steps)
            inference_steps = 0
            return_keys = ['logits', 'predictions']  # Adjust based on your model's outputs
            
            while True:
                carry, loss, metrics, preds, all_finish = model(
                    carry=carry, 
                    batch=batch, 
                    return_keys=return_keys
                )
                inference_steps += 1
                
                if all_finish:
                    break
                    
                if inference_steps > 100:  # Safety check
                    print(f"Warning: Inference exceeded 100 steps for example {idx}")
                    break
            
            # Collect metrics
            for key, value in metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value.item() if torch.is_tensor(value) else value)
            
            # Store example details for verbose output
            if verbose:
                example_info = {
                    'set_name': set_name,
                    'example_idx': idx,
                    'inference_steps': inference_steps,
                    'loss': loss.item(),
                    'metrics': {k: v.item() if torch.is_tensor(v) else v for k, v in metrics.items()}
                }
                
                # Add input/output samples if available
                if 'input_ids' in batch:
                    example_info['input_sample'] = batch['input_ids'][0, :20].cpu().numpy()  # First 20 tokens
                
                if 'predictions' in preds:
                    example_info['prediction_sample'] = preds['predictions'][0, :20].cpu().numpy()
                    
                if 'target_ids' in batch:
                    example_info['target_sample'] = batch['target_ids'][0, :20].cpu().numpy()
                    
                example_outputs.append(example_info)
    
    # Compute average metrics
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    
    return avg_metrics, example_outputs


def print_results(avg_metrics: Dict[str, float], example_outputs: list, 
                  vocab_size: Optional[int] = None):
    """Pretty print evaluation results."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Print average metrics
    print("\n📊 Average Metrics:")
    print("-"*40)
    for key, value in avg_metrics.items():
        # Format the metric name nicely
        formatted_key = key.replace('_', ' ').title()
        print(f"  {formatted_key:30s}: {value:.4f}")
    
    # Print per-example details
    if example_outputs:
        print("\n📝 Per-Example Details:")
        print("-"*40)
        
        for i, example in enumerate(example_outputs):
            print(f"\n  Example {i+1} (Set: {example['set_name']}):")
            print(f"    • Inference steps: {example['inference_steps']}")
            print(f"    • Loss: {example['loss']:.4f}")
            
            # Print specific metrics
            if 'exact_acc' in example['metrics']:
                print(f"    • Exact Accuracy: {example['metrics']['exact_acc']:.2%}")
            if 'token_acc' in example['metrics']:
                print(f"    • Token Accuracy: {example['metrics']['token_acc']:.2%}")
            
            # Show sample tokens if available
            if 'input_sample' in example:
                print(f"    • Input tokens (first 20): {example['input_sample'].tolist()}")
            if 'prediction_sample' in example:
                print(f"    • Predicted tokens (first 20): {example['prediction_sample'].tolist()}")
            if 'target_sample' in example:
                print(f"    • Target tokens (first 20): {example['target_sample'].tolist()}")
    
    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description='Evaluate puzzle model checkpoint')
    parser.add_argument('--checkpoint', type=str, 
                       help='Path to checkpoint file (e.g., checkpoints/project/run/step_1000)')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to dataset for evaluation')
    parser.add_argument('--num-examples', type=int, default=5,
                       help='Number of examples to evaluate (default: 5)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for evaluation (default: 1)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run evaluation on (default: cuda if available)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (if not using checkpoint directory)')
    parser.add_argument('--no-ema', action='store_true',
                       help='Load non-EMA weights if available (by default, uses checkpoint as-is)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed per-example outputs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"🔧 Configuration:")
    print(f"  • Checkpoint: {args.checkpoint}")
    print(f"  • Data path: {args.data_path}")
    print(f"  • Device: {args.device}")
    print(f"  • Number of examples: {args.num_examples}")
    print()
    
    try:
        # Load configuration
        if args.config:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Try to load from checkpoint directory
            checkpoint_dir = os.path.dirname(args.checkpoint)
            config = load_config_from_checkpoint(checkpoint_dir)
        
        # Create dataset to get metadata
        dataset_config = PuzzleDatasetConfig(
            seed=args.seed,
            dataset_paths=[args.data_path],
            rank=0,
            num_replicas=1,
            test_set_mode=True,
            epochs_per_iter=1,
            global_batch_size=args.batch_size
        )
        
        dataset = PuzzleDataset(dataset_config, split='test')
        metadata = dataset.metadata
        
        print(f"📚 Dataset Metadata:")
        print(f"  • Vocab size: {metadata.vocab_size}")
        print(f"  • Sequence length: {metadata.seq_len}")
        print(f"  • Number of puzzles: {metadata.num_puzzle_identifiers}")
        print(f"  • Total groups: {metadata.total_groups}")
        print()
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=0,  # Use 0 for debugging
            pin_memory=True
        )
        
        # Create model
        print("🏗️  Creating model...")
        model = create_model_from_config(
            config, 
            vocab_size=metadata.vocab_size,
            seq_len=metadata.seq_len,
            num_puzzle_identifiers=metadata.num_puzzle_identifiers,
            device=args.device
        )
        
        # Load checkpoint
        model = load_checkpoint_weights(model, args.checkpoint, args.device)
        model = model.to(args.device)
        
        # Handle non-EMA weights if requested
        if args.no_ema:
            # Check if this checkpoint contains EMA weights
            metadata_path = Path(args.checkpoint) / 'metadata.json'
            contains_ema = False
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    contains_ema = metadata.get('contains_ema_weights', False)
            
            if contains_ema:
                # Try to load non-EMA version
                checkpoint_dir = Path(args.checkpoint)
                if checkpoint_dir.name.startswith('final'):
                    # For final checkpoints, look for the _no_ema version
                    step_num = checkpoint_dir.name.replace('final_step_', '').replace('final', '')
                    if step_num:
                        no_ema_path = checkpoint_dir.parent / f"final_step_{step_num}_no_ema"
                    else:
                        # Handle 'final' symlink case
                        if checkpoint_dir.is_symlink():
                            real_path = checkpoint_dir.resolve()
                            step_num = real_path.name.replace('final_step_', '')
                            no_ema_path = real_path.parent / f"final_step_{step_num}_no_ema"
                        else:
                            no_ema_path = checkpoint_dir.parent / "final_no_ema"
                    
                    no_ema_model_path = no_ema_path / "model.pt"
                    if no_ema_model_path.exists():
                        print(f"Loading non-EMA weights from {no_ema_model_path}")
                        model = load_checkpoint_weights(model, str(no_ema_model_path), args.device)
                    else:
                        print(f"Warning: Non-EMA weights not found at {no_ema_model_path}")
                        print("Using checkpoint weights (which contain EMA weights)")
                else:
                    print("Note: --no-ema flag used but checkpoint may not contain EMA weights")
            else:
                print("Note: Checkpoint does not contain EMA weights, --no-ema flag has no effect")
        
        # Run evaluation
        print(f"\n🚀 Running evaluation on {args.num_examples} examples...")
        avg_metrics, example_outputs = evaluate_examples(
            model, 
            dataloader,
            num_examples=args.num_examples,
            device=args.device,
            verbose=args.verbose
        )
        
        # Print results
        print_results(avg_metrics, example_outputs if args.verbose else [], 
                     vocab_size=metadata.vocab_size)
        
        # Save results if needed
        output_dir = os.path.dirname(args.checkpoint)
        output_file = os.path.join(output_dir, f"eval_results_{os.path.basename(args.checkpoint)}.yaml")
        
        results = {
            'checkpoint': args.checkpoint,
            'data_path': args.data_path,
            'num_examples': args.num_examples,
            'average_metrics': avg_metrics,
        }
        
        with open(output_file, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        print(f"\n💾 Results saved to {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())