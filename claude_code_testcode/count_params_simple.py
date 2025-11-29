#!/usr/bin/env python3
"""Quick parameter count for any model"""
import sys
import os
import torch
import yaml

# Add to your training script or use interactively:

def count_params(model):
    """Count parameters in a PyTorch model"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total params: {total:,}")
    print(f"Trainable:    {trainable:,}")
    print(f"Non-trainable: {total - trainable:,}")
    
    # Size in MB (assuming float32)
    size_mb = total * 4 / (1024**2)
    print(f"Model size:   {size_mb:.2f} MB")
    
    return total, trainable

def count_params_from_checkpoint(checkpoint_dir):
    """Count parameters from a checkpoint directory"""
    print(f"Loading checkpoint from: {checkpoint_dir}")
    print("="*60)
    
    # Load config
    config_path = os.path.join(checkpoint_dir, "all_config.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(checkpoint_dir), "all_config.yaml")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✓ Config loaded from: {config_path}")
        print(f"  Model: {config['arch']['name']}")
        if 'hidden_size' in config['arch']:
            print(f"  Hidden size: {config['arch']['hidden_size']}")
        if 'L_layers' in config['arch']:
            print(f"  L_layers: {config['arch']['L_layers']}")
        if 'tree_branching_steps' in config['arch']:
            print(f"  Tree branching steps: {config['arch']['tree_branching_steps']}")
    else:
        print(f"⚠ Config not found at: {config_path}")
    
    # Load model checkpoint
    model_path = os.path.join(checkpoint_dir, "model.pt")
    if not os.path.exists(model_path):
        print(f"✗ Model checkpoint not found at: {model_path}")
        return None, None
    
    print(f"✓ Loading model from: {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print(checkpoint)

        # Count parameters from state dict
        total = 0
        trainable = 0
        param_details = {}
        
        for name, param in checkpoint.items():
            num_params = param.numel()
            total += num_params
            trainable += num_params  # Assume all are trainable from checkpoint
            
            # Group by module
            module = name.split('.')[0] if '.' in name else 'root'
            if module.startswith('_orig_mod'):
                module = name.split('.')[1] if len(name.split('.')) > 1 else 'model'
            
            if module not in param_details:
                param_details[module] = 0
            param_details[module] += num_params
        
        print("\n" + "="*60)
        print("PARAMETER COUNT FROM CHECKPOINT")
        print("="*60)
        print(f"Total params: {total:,}")
        print(f"Model size:   {total * 4 / (1024**2):.2f} MB (float32)")
        
        print("\n" + "="*60)
        print("PARAMETERS BY MODULE")
        print("="*60)
        for module, count in sorted(param_details.items(), key=lambda x: -x[1]):
            percentage = 100 * count / total if total > 0 else 0
            print(f"{module:30s} {count:>15,} ({percentage:5.1f}%)")
        
        print("="*60)
        
        return total, trainable
        
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# Example usage in your code:
# from count_params_simple import count_params
# count_params(model)

if __name__ == "__main__":
    checkpoint_dir = "/home/wnsx0000/jhun/TinyRecursiveModels-olivkoch/checkpoints/totrm/totrm_noise_added/final"
    
    if os.path.exists(checkpoint_dir):
        count_params_from_checkpoint(checkpoint_dir)
    else:
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        print("\nYou can also use this as a module:")
        print("  from count_params_simple import count_params")
        print("  count_params(your_model)")
