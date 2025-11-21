"""
Checkpoint utilities for saving and loading complete training states.
"""

import os
import json
import yaml
import torch
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    step: int
    epoch: float
    total_steps: int
    timestamp: str
    checkpoint_path: str
    model_name: str
    dataset_paths: List[str]
    
    # Training metrics
    final_train_loss: Optional[float] = None
    final_eval_loss: Optional[float] = None
    final_exact_acc: Optional[float] = None
    best_exact_acc: Optional[float] = None
    best_exact_acc_step: Optional[int] = None
    
    # Training time
    training_hours: Optional[float] = None
    
    # Hardware info
    device: Optional[str] = None
    world_size: Optional[int] = None
    
    # EMA info
    contains_ema_weights: bool = False
    
    def to_dict(self):
        return asdict(self)


class CheckpointManager:
    """Manages checkpoint saving and loading."""
    
    def __init__(self, base_dir: str, run_name: str):
        self.base_dir = Path(base_dir)
        self.run_name = run_name
        self.checkpoint_dir = self.base_dir / run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track best metrics
        self.best_metrics = {
            'exact_acc': 0.0,
            'exact_acc_step': 0,
            'loss': float('inf'),
            'loss_step': 0
        }
        
        # Track all metrics history
        self.metrics_history = {
            'train': [],
            'eval': []
        }
        
        print(f"📁 Checkpoint directory: {self.checkpoint_dir}")
    
    def save_checkpoint(
        self,
        train_state,  # TrainState object from pretrain.py
        config,  # PretrainConfig
        metrics: Optional[Dict[str, Any]] = None,
        is_final: bool = False,
        save_optimizer: bool = True,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save a complete checkpoint."""
        
        # Determine checkpoint name
        if is_final:
            checkpoint_name = f"final_step_{train_state.step}"
        else:
            checkpoint_name = f"step_{train_state.step}"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Save model weights
        model_path = checkpoint_path / "model.pt"
        torch.save(train_state.model.state_dict(), model_path)
        print(f"  ✓ Model saved to {model_path}")
        
        # 2. Save optimizer states (if requested)
        if save_optimizer and train_state.optimizers:
            optimizers_path = checkpoint_path / "optimizers.pt"
            optimizer_states = []
            for opt in train_state.optimizers:
                optimizer_states.append(opt.state_dict())
            torch.save({
                'optimizers': optimizer_states,
                'optimizer_lrs': train_state.optimizer_lrs,
            }, optimizers_path)
            print(f"  ✓ Optimizers saved to {optimizers_path}")
        
        # 3. Save training state
        training_state_path = checkpoint_path / "training_state.pt"
        torch.save({
            'step': train_state.step,
            'total_steps': train_state.total_steps,
            'carry': train_state.carry if hasattr(train_state, 'carry') else None,
            'best_metrics': self.best_metrics,
        }, training_state_path)
        print(f"  ✓ Training state saved to {training_state_path}")
        
        # 4. Save config
        config_path = checkpoint_path / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config.model_dump(), f, default_flow_style=False)
        print(f"  ✓ Config saved to {config_path}")
        
        # 5. Save metrics
        if metrics:
            self._update_metrics(metrics, train_state.step)
            metrics_path = checkpoint_path / "metrics.json"
            
            # Convert metrics to JSON-serializable format
            json_metrics = self._convert_to_json_serializable({
                'current_metrics': metrics,
                'best_metrics': self.best_metrics,
                'metrics_history': self.metrics_history if is_final else None
            })
            
            with open(metrics_path, 'w') as f:
                json.dump(json_metrics, f, indent=2)
            print(f"  ✓ Metrics saved to {metrics_path}")
        
        # 6. Save metadata
        metadata = CheckpointMetadata(
            step=train_state.step,
            epoch=train_state.step / (train_state.total_steps / config.epochs) if train_state.total_steps > 0 else 0,
            total_steps=train_state.total_steps,
            timestamp=datetime.now().isoformat(),
            checkpoint_path=str(checkpoint_path),
            model_name=config.arch.name,
            dataset_paths=config.data_paths,
            final_train_loss=self._convert_to_json_serializable(metrics.get('train/lm_loss')) if metrics else None,
            final_eval_loss=self._convert_to_json_serializable(self._get_eval_loss(metrics)) if metrics else None,
            final_exact_acc=self._convert_to_json_serializable(self._get_exact_acc(metrics)) if metrics else None,
            best_exact_acc=self._convert_to_json_serializable(self.best_metrics.get('exact_acc')),
            best_exact_acc_step=self.best_metrics.get('exact_acc_step'),
            device=additional_info.get('device') if additional_info else None,
            world_size=additional_info.get('world_size') if additional_info else None,
            training_hours=additional_info.get('training_hours') if additional_info else None,
            contains_ema_weights=additional_info.get('contains_ema_weights', False) if additional_info else False
        )
        
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        print(f"  ✓ Metadata saved to {metadata_path}")
        
        # 7. Create a symlink to latest checkpoint
        if is_final:
            latest_link = self.checkpoint_dir / "final"
            if latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(checkpoint_path.name)
            print(f"  ✓ Created symlink 'final' -> {checkpoint_name}")
        else:
            latest_link = self.checkpoint_dir / "latest"
            if latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(checkpoint_path.name)
            print(f"  ✓ Created symlink 'latest' -> {checkpoint_name}")
        
        # 8. Save a summary file in the root directory
        if is_final:
            self._save_training_summary(metadata, config)
        
        print(f"✅ Checkpoint saved to {checkpoint_path}")
        return str(checkpoint_path)
    
    def _update_metrics(self, metrics: Dict[str, Any], step: int):
        """Update best metrics and history."""
        # Convert metrics to ensure they're JSON-serializable
        metrics = self._convert_to_json_serializable(metrics)
        
        # Update history
        if any(k.startswith('train/') for k in metrics.keys()):
            self.metrics_history['train'].append({
                'step': step,
                **{k: v for k, v in metrics.items() if k.startswith('train/')}
            })
        
        if any(k.startswith('eval/') for k in metrics.keys()):
            self.metrics_history['eval'].append({
                'step': step,
                **{k: v for k, v in metrics.items() if k.startswith('eval/')}
            })
        
        # Update best metrics
        exact_acc = self._get_exact_acc(metrics)
        if exact_acc and exact_acc > self.best_metrics['exact_acc']:
            self.best_metrics['exact_acc'] = exact_acc
            self.best_metrics['exact_acc_step'] = step
        
        loss = self._get_eval_loss(metrics)
        if loss and loss < self.best_metrics['loss']:
            self.best_metrics['loss'] = loss
            self.best_metrics['loss_step'] = step
    
    def _get_exact_acc(self, metrics: Dict[str, Any]) -> Optional[float]:
        """Extract exact accuracy from metrics."""
        for key in ['eval/exact_acc', 'eval/exact_accuracy', 'exact_acc']:
            if key in metrics:
                return metrics[key]
        
        # Try to find it in nested structures
        for key, value in metrics.items():
            if isinstance(value, dict):
                if 'exact_acc' in value:
                    return value['exact_acc']
        return None
    
    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """Convert numpy types and tensors to JSON-serializable Python types."""
        if obj is None:
            return None
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif torch.is_tensor(obj):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    def _get_eval_loss(self, metrics: Dict[str, Any]) -> Optional[float]:
        """Extract evaluation loss from metrics."""
        for key in ['eval/lm_loss', 'eval/loss', 'eval/test_loss']:
            if key in metrics:
                return metrics[key]
        
        # Try to find it in nested structures
        for key, value in metrics.items():
            if isinstance(value, dict):
                if 'lm_loss' in value or 'loss' in value:
                    return value.get('lm_loss', value.get('loss'))
        return None
    
    def _save_training_summary(self, metadata: CheckpointMetadata, config):
        """Save a comprehensive training summary."""
        summary_path = self.checkpoint_dir / "training_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("TRAINING SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Run Name: {self.run_name}\n")
            f.write(f"Model: {config.arch.name}\n")
            f.write(f"Completed: {metadata.timestamp}\n")
            f.write(f"Total Steps: {metadata.step}/{metadata.total_steps}\n")
            f.write(f"Epochs: {metadata.epoch:.2f}/{config.epochs}\n")
            
            if metadata.training_hours:
                f.write(f"Training Time: {metadata.training_hours:.2f} hours\n")
            
            f.write("\n" + "-"*70 + "\n")
            f.write("FINAL METRICS\n")
            f.write("-"*70 + "\n")
            
            if metadata.final_train_loss is not None:
                f.write(f"Final Training Loss: {metadata.final_train_loss:.4f}\n")
            
            if metadata.final_eval_loss is not None:
                f.write(f"Final Eval Loss: {metadata.final_eval_loss:.4f}\n")
            
            if metadata.final_exact_acc is not None:
                f.write(f"Final Exact Accuracy: {metadata.final_exact_acc:.2%}\n")
            
            if metadata.best_exact_acc is not None:
                f.write(f"Best Exact Accuracy: {metadata.best_exact_acc:.2%} (step {metadata.best_exact_acc_step})\n")
            
            f.write("\n" + "-"*70 + "\n")
            f.write("CONFIGURATION\n")
            f.write("-"*70 + "\n")
            
            f.write(f"Learning Rate: {config.lr}\n")
            f.write(f"Batch Size: {config.global_batch_size}\n")
            f.write(f"Weight Decay: {config.weight_decay}\n")
            f.write(f"Warmup Steps: {config.lr_warmup_steps}\n")
            
            if hasattr(config.arch, 'puzzle_emb_ndim'):
                f.write(f"Puzzle Embedding Dim: {config.arch.puzzle_emb_ndim}\n")
                f.write(f"Puzzle Embedding LR: {config.puzzle_emb_lr}\n")
            
            f.write(f"\nDatasets:\n")
            for path in config.data_paths:
                f.write(f"  - {path}\n")
            
            if config.data_paths_test:
                f.write(f"\nTest Datasets:\n")
                for path in config.data_paths_test:
                    f.write(f"  - {path}\n")
            
            f.write("\n" + "-"*70 + "\n")
            f.write("METRICS HISTORY\n")
            f.write("-"*70 + "\n")
            
            # Plot metrics evolution
            if self.metrics_history['eval']:
                f.write("\nEvaluation Metrics Evolution:\n")
                f.write("Step\tLoss\tExact Acc\n")
                
                for entry in self.metrics_history['eval'][-10:]:  # Last 10 evals
                    step = entry['step']
                    loss = self._get_eval_loss(entry)
                    acc = self._get_exact_acc(entry)
                    
                    f.write(f"{step}\t")
                    f.write(f"{loss:.4f}\t" if loss else "N/A\t")
                    f.write(f"{acc:.2%}\n" if acc else "N/A\n")
        
        print(f"  ✓ Training summary saved to {summary_path}")
        
        # Also save metrics history as CSV for analysis
        if self.metrics_history['eval']:
            import pandas as pd
            df = pd.DataFrame(self.metrics_history['eval'])
            csv_path = self.checkpoint_dir / "metrics_history.csv"
            df.to_csv(csv_path, index=False)
            print(f"  ✓ Metrics history saved to {csv_path}")
    
    def load_checkpoint(self, checkpoint_path: str, train_state, config, 
                       load_optimizer: bool = True, strict: bool = True):
        """Load a checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            # Check if it's a relative path in our checkpoint dir
            checkpoint_path = self.checkpoint_dir / checkpoint_path
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"📂 Loading checkpoint from {checkpoint_path}")
        
        # Load model
        model_path = checkpoint_path / "model.pt"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location='cpu')
            train_state.model.load_state_dict(state_dict, strict=strict)
            print(f"  ✓ Model loaded")
        
        # Load optimizers
        if load_optimizer:
            optimizers_path = checkpoint_path / "optimizers.pt"
            if optimizers_path.exists():
                opt_state = torch.load(optimizers_path, map_location='cpu')
                for i, opt in enumerate(train_state.optimizers):
                    if i < len(opt_state['optimizers']):
                        opt.load_state_dict(opt_state['optimizers'][i])
                train_state.optimizer_lrs = opt_state.get('optimizer_lrs', train_state.optimizer_lrs)
                print(f"  ✓ Optimizers loaded")
        
        # Load training state
        training_state_path = checkpoint_path / "training_state.pt"
        if training_state_path.exists():
            state = torch.load(training_state_path, map_location='cpu')
            train_state.step = state['step']
            train_state.total_steps = state['total_steps']
            if 'carry' in state and state['carry'] is not None:
                train_state.carry = state['carry']
            if 'best_metrics' in state:
                self.best_metrics = state['best_metrics']
            print(f"  ✓ Training state loaded (step {train_state.step})")
        
        # Load metrics history
        metrics_path = checkpoint_path / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
                if 'metrics_history' in metrics_data and metrics_data['metrics_history']:
                    self.metrics_history = metrics_data['metrics_history']
            print(f"  ✓ Metrics loaded")
        
        print(f"✅ Checkpoint loaded successfully")
        
        return train_state


def save_code_files(checkpoint_dir: Path, config):
    """Save relevant code files to checkpoint directory."""
    code_dir = checkpoint_dir / "code"
    code_dir.mkdir(exist_ok=True)
    
    # Save the main training script
    if os.path.exists("pretrain.py"):
        shutil.copy("pretrain.py", code_dir / "pretrain.py")
    
    # Save model files
    from utils.functions import get_model_source_path
    
    model_files = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    
    for evaluator in config.evaluators:
        eval_file = get_model_source_path(evaluator.name, "evaluators.")
        if eval_file:
            model_files.append(eval_file)
    
    for file_path in model_files:
        if file_path and os.path.exists(file_path):
            dest_path = code_dir / os.path.basename(file_path)
            shutil.copy(file_path, dest_path)
            print(f"  ✓ Saved {os.path.basename(file_path)}")