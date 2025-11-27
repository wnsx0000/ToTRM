from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )

def log_stablemax(x, dim=-1):
    # Ensure we handle the dtype properly
    original_dtype = x.dtype
    if x.device.type == 'mps' and x.dtype == torch.float64:
        x = x.to(torch.float32)

    s_x = s(x)
    result = torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))

    # Convert back to original dtype if possible
    if x.device.type != 'mps' and original_dtype == torch.float64:
        result = result.to(original_dtype)

    return result

def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    # Detect device type and use appropriate dtype
    if logits.device.type == 'mps':
        # MPS doesn't support float64, use float32
        logprobs = log_stablemax(logits.to(torch.float32), dim=-1)
    else:
        # CUDA and CPU support float64
        logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(
        logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1
    ).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


def diversity_loss(z_branches: torch.Tensor, margin: float = 0.0, reduction: str = "sum") -> torch.Tensor:
    """
    Encourage diversity between tree-of-thought branches using squared hinge loss.

    Uses per-sample reduction (mean over branch pairs within each sample, then sum/mean over batch)
    to ensure each problem in the batch gets appropriate and independent diversity penalties.

    Hinge variant (margin > 0):
        - Only penalizes when similarity exceeds margin
        - Allows model to focus on task performance once diversity is sufficient
        - loss = max(0, cosine_sim - margin)²

    Standard variant (margin = 0):
        - Always penalizes similarity
        - loss = cosine_sim²

    Args:
        z_branches: Tensor of shape [batch_size, tree_width, seq_len, hidden_size]
                   representing the latent states for each branch in the tree.
        margin: Target maximum cosine similarity (e.g., 0.85). When margin > 0, only similarities
                above this threshold are penalized (hinge loss). Default 0.0 (no hinge).
        reduction: "sum" (default, consistent with lm_loss) or "mean"

    Returns:
        diversity_loss: Scalar tensor. Lower values indicate more diverse branches.

        Typical values with margin=0.85:
        - similarity=0.97 → violation=0.12 → loss≈0.014 per sample
        - similarity=0.80 → violation=0.0  → loss=0 (no penalty)

    Example:
        >>> z_branches = torch.randn(8, 4, 32, 512)  # batch=8, 4 branches, seq=32, hidden=512
        >>> loss = diversity_loss(z_branches, margin=0.85)
        >>> # If branches are collapsing (sim>0.85), loss > 0
        >>> # If branches are diverse enough (sim<0.85), loss = 0
    """
    batch_size, tree_width, seq_len, hidden_size = z_branches.shape

    # Flatten seq_len and hidden_size dimensions for each branch
    # Shape: [batch_size, tree_width, seq_len * hidden_size]
    z_flat = z_branches.flatten(2)

    # Normalize each branch to unit vectors for cosine similarity computation
    # Shape: [batch_size, tree_width, seq_len * hidden_size]
    z_norm = F.normalize(z_flat, dim=-1)

    # Compute pairwise cosine similarity matrix via batch matrix multiplication
    # Shape: [batch_size, tree_width, tree_width]
    sim_matrix = torch.bmm(z_norm, z_norm.transpose(1, 2))

    # Create mask to exclude diagonal (self-similarity = 1.0)
    # We only care about similarity between different branches
    mask = ~torch.eye(tree_width, dtype=torch.bool, device=z_branches.device)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

    # Extract off-diagonal elements (inter-branch similarities)
    # Shape: [batch_size, tree_width * (tree_width - 1)]
    off_diag_sim = sim_matrix[mask].view(batch_size, -1)

    # Apply hinge if margin > 0: only penalize violations
    if margin > 0:
        # Hinge: max(0, similarity - margin)
        violations = torch.clamp(off_diag_sim - margin, min=0.0)
        similarities = violations
    else:
        # No hinge: penalize all similarity
        similarities = off_diag_sim

    # Per-sample loss: mean over pairs WITHIN each sample
    # This ensures each problem gets independent diversity penalty
    # Shape: [batch_size]
    per_sample_loss = similarities.pow(2).mean(dim=1)

    # Aggregate over batch (consistent with lm_loss, q_halt_loss)
    if reduction == "sum":
        loss = per_sample_loss.sum()
    elif reduction == "mean":
        loss = per_sample_loss.mean()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    return loss


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            # print(f"{loss_counts=}")
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            # print(f"{valid_metrics.sum()=} {new_carry.halted=}")
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses

        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })
        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()

        # Diversity loss (ToTRM only)
        div_loss = 0
        if "z_L_branches" in outputs:
            # Get diversity hyperparameters from model config
            diversity_weight = getattr(self.model.config, 'diversity_weight', 0.1)
            diversity_margin = getattr(self.model.config, 'diversity_margin', 0.0)

            if diversity_weight > 0:
                # Compute diversity loss with hinge (if margin > 0)
                # Uses per-sample reduction: mean over pairs, sum over batch
                div_loss = diversity_loss(
                    outputs["z_L_branches"],
                    margin=diversity_margin,
                    reduction="sum"
                )
                # Scale by diversity_weight
                div_loss = diversity_weight * div_loss
                metrics["diversity_loss"] = div_loss.detach()

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        # Total loss: lm_loss + q_losses + diversity_loss
        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss) + div_loss

        # print(f"checking {metrics=}")
        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()

