"""
Tree-of-Thought Recursive Model (ToTRM)

Based on TRM but applies Tree-of-Thought at the architecture level.
Each recursion step creates a binary tree of z states, which are merged 
before computing the final output y.
"""

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100


@dataclass
class ToTRM_InnerCarry:
    """Carry state for ToTRM inner model.
    
    In ToTRM, we maintain a tree of z states. For simplicity, we store them
    as a batch dimension where batch_size = original_batch_size * tree_width
    """
    z_H: torch.Tensor  # [batch_size * tree_width, seq_len, hidden_size]
    z_L: torch.Tensor  # [batch_size * tree_width, seq_len, hidden_size]
    tree_width: int    # Number of branches in the tree (2^n for binary tree)


@dataclass
class ToTRM_Carry:
    inner_carry: ToTRM_InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class ToTRM_Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int  # ignored
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # ToTRM specific
    tree_branching_steps: int = 6  # Number of L_cycle steps that branch (n-1 in description)
    tree_merge_method: str = "mean"  # "mean", "max", "learned_weighted"
    mlp_t: bool = False
    puzzle_emb_len: int = 16
    no_ACT_continue: bool = True
    branch_norm_factor: Optional[float] = None  # If None, use adaptive: 0.5 * sqrt(hidden_size/64)
    adaptive_branch_scale: Optional[float] = None  # If set (e.g., 0.2), scale branch_emb as fraction of z norm
    diversity_weight: float = 0.1  # Weight for diversity loss (0 to disable)
    diversity_margin: float = 0.0  # Hinge margin for diversity loss (0 = no hinge, 0.85 = recommended)


class ToTRM_Block(nn.Module):
    """Same as TRM block - reusing the architecture."""
    
    def __init__(self, config: ToTRM_Config) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len,
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1, 2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states


class ToTRM_ReasoningModule(nn.Module):
    def __init__(self, layers: List[ToTRM_Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class ToTRM_Inner(nn.Module):
    """Inner model with Tree-of-Thought reasoning."""
    
    def __init__(self, config: ToTRM_Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers, 
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size, 
                init_std=0, 
                cast_to=self.forward_dtype
            )

        # Position encodings
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len, 
                self.config.hidden_size, 
                init_std=embed_init_std, 
                cast_to=self.forward_dtype
            )

        # Reasoning layers
        self.L_level = ToTRM_ReasoningModule(layers=[ToTRM_Block(self.config) for _i in range(self.config.L_layers)])

        # Tree merge layer (for combining branches)
        if self.config.tree_merge_method == "learned_weighted":
            # Learn weights for each position in the tree
            # Initialize with small random values so all branches start with similar importance
            # but can diverge during training
            max_tree_width = 2 ** self.config.tree_branching_steps
            self.merge_weights = nn.Parameter(
                torch.randn(max_tree_width, dtype=self.forward_dtype) * 0.01
            )
        elif self.config.tree_merge_method == "gated_product":
            # Learnable gate to interpolate between geometric mean (product) and arithmetic mean (sum)
            # Initialize to -2.0 so sigmoid(-2.0) ≈ 0.12 → starts closer to mean for stability
            self.product_gate = nn.Parameter(
                torch.tensor(-2.0, dtype=self.forward_dtype)
            )
        
        # Branch embeddings to differentiate branches at each level
        # Each branch gets a unique embedding based on its position in the tree
        max_tree_width = 2 ** self.config.tree_branching_steps
        self.branch_embeddings = nn.Parameter(
            torch.randn(max_tree_width, self.config.hidden_size, dtype=self.forward_dtype) * 0.02
        )

        # Compute adaptive branch norm factor
        # If not specified, scale with sqrt(hidden_size) to maintain relative impact
        # Base calibration: hidden_size=64 → factor=0.5 (verified on 4x4 Sudoku)
        if self.config.branch_norm_factor is None:
            self.branch_norm_factor = 0.5 * math.sqrt(self.config.hidden_size / 64.0)
        else:
            self.branch_norm_factor = self.config.branch_norm_factor

        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head init
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        """Same as TRM."""
        embedding = self.embed_tokens(input.to(torch.int32))

        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        return self.embed_scale * embedding

    def _branch_state(self, z: torch.Tensor, current_tree_width: int) -> torch.Tensor:
        """Branch each state into 2 copies with distinct embeddings.
        
        Each branch gets a unique learnable embedding to differentiate it from its sibling.
        This allows the model to explore different reasoning paths.
        
        Args:
            z: [batch_size * current_width, seq_len, hidden_size]
            current_tree_width: Current width before branching
        Returns:
            z: [batch_size * current_width * 2, seq_len, hidden_size]
        """
        # Duplicate each state
        z_branched = z.repeat_interleave(2, dim=0)
        # [B*W*2, L, D]
        
        # Add unique branch embeddings to differentiate the branches
        # For each pair of branches, add different embeddings
        batch_size = z.shape[0] // current_tree_width
        new_tree_width = current_tree_width * 2
        
        # Create branch indices: [0, 1, 0, 1, 0, 1, ...] for the entire batch
        branch_pattern = torch.arange(new_tree_width, device=z.device)
        # max_tree_width = 2 ** self.config.tree_branching_steps
        # branch_pattern = torch.randint(0, max_tree_width, (new_tree_width,), device=z.device)
        branch_indices = branch_pattern.repeat(batch_size)

        # Compute branch embedding scale
        if self.config.adaptive_branch_scale is not None:
            # Adaptive scaling: scale as a fraction of current z magnitude
            # This ensures branch embeddings are always relative to z's current scale
            z_magnitude = z_branched.norm(dim=-1, keepdim=True).mean()  # Mean L2 norm across all positions
            branch_scale = self.config.adaptive_branch_scale * z_magnitude
        else:
            # Fixed scaling: use pre-computed factor from __init__
            branch_scale = self.branch_norm_factor

        # Get branch embeddings: [B*W*2, D]
        # Normalize to unit vectors, then scale
        branch_embs = F.normalize(self.branch_embeddings[branch_indices], dim=-1) * branch_scale

        # # DEBUG: Print shapes before and after unsqueeze
        # if current_tree_width == 1:  # Only print on first branching
        #     print(f"[DEBUG BRANCHING]")
        #     print(f"  z_branched.shape (before adding) = {z_branched.shape}")
        #     print(f"  branch_embs.shape = {branch_embs.shape}")
        #     branch_embs_unsqueezed = branch_embs.unsqueeze(1)
        #     print(f"  branch_embs.unsqueeze(1).shape = {branch_embs_unsqueezed.shape}")
        #     print(f"  Broadcasting: [{z_branched.shape[0]}, {z_branched.shape[1]}, {z_branched.shape[2]}]")
        #     print(f"              + [{branch_embs_unsqueezed.shape[0]}, {branch_embs_unsqueezed.shape[1]}, {branch_embs_unsqueezed.shape[2]}]")
        #     print(f"  -> Same branch_emb vector added to ALL {z_branched.shape[1]} positions")

        # # branch embeddings 이전에 먼저 normalize
        # z_branched_norm = F.layer_norm(z_branched, (z_branched.shape[-1],))

        # 더하기
        # z_branched = z_branched_norm + branch_embs.unsqueeze(1)
        z_branched = z_branched + branch_embs.unsqueeze(1)

        # # branch embeddings 더한 후에 normalize
        # z_branched = F.layer_norm(z_branched, (z_branched.shape[-1],))

        return z_branched

    def _merge_tree(self, z: torch.Tensor, tree_width: int, original_batch_size: int) -> torch.Tensor:
        """Merge the tree of states back into a single state per batch item.

        Different merge strategies:
        - mean: Simple average (all branches equally weighted)
        - max: Take max activation (keeps strongest signal)
        - learned_weighted: Learn importance weights for each branch position

        Args:
            z: [batch_size * tree_width, seq_len, hidden_size]
            tree_width: Number of branches (2^n where n = tree_branching_steps)
            original_batch_size: Original batch size before branching
        Returns:
            z: [batch_size, seq_len, hidden_size]
        """
        # Reshape to separate tree dimension
        # [batch_size * tree_width, seq_len, hidden_size] -> [batch_size, tree_width, seq_len, hidden_size]
        z = z.view(original_batch_size, tree_width, *z.shape[1:])

        if self.config.tree_merge_method == "mean":
            # Average across tree branches
            # Pro: Simple, stable
            # Con: Wrong branches can dilute correct ones
            z = z.mean(dim=1)
        elif self.config.tree_merge_method == "max":
            # Max pooling across tree branches
            # Pro: Keeps strongest signal, ignores weak branches
            # Con: Only one branch contributes, others wasted
            z = z.max(dim=1)[0]
        elif self.config.tree_merge_method == "learned_weighted":
            # Weighted sum with learned weights per branch position
            # Pro: Model learns which branch positions are more reliable
            # Con: Same weights used for all batch items (position-based, not content-based)

            # Apply softmax to get normalized weights that sum to 1
            # This ensures merge is a proper weighted average
            weights = F.softmax(self.merge_weights[:tree_width], dim=0)  # [tree_width]

            # Reshape for broadcasting: [tree_width] -> [1, tree_width, 1, 1]
            # This allows element-wise multiplication with z of shape [B, W, L, D]
            weights = weights.view(1, tree_width, 1, 1)

            # Weighted sum: each branch weighted by its learned importance
            # Higher weight = model learned this branch position is more useful
            z = (z * weights).sum(dim=1)  # [B, L, D]
        elif self.config.tree_merge_method == "geometric_mean":
            # Pure geometric mean: (x1 * x2 * ... * xn)^(1/n)
            # Pro: All branches must "agree" for strong signal (consensus)
            # Pro: Less sensitive to outliers than arithmetic mean
            # Con: Can be unstable if values cross zero frequently

            # Use log-space for numerical stability: exp(mean(log(|x|)))
            eps = 1e-8
            z_abs = torch.abs(z) + eps
            log_abs = torch.log(z_abs)
            geom_mean_abs = torch.exp(log_abs.mean(dim=1))  # [B, L, D]

            # Sign handling: use sign of arithmetic sum (simple and stable)
            z_sign = torch.sign(z.sum(dim=1))
            z = geom_mean_abs * z_sign
        elif self.config.tree_merge_method == "gated_product":
            # Interpolate between geometric mean (product) and arithmetic mean (sum)
            # Pro: Combines consensus mechanism (product) with stability (mean)
            # Pro: Model learns optimal balance during training
            # Con: More complex computation

            # Learnable gate: 0 = pure mean, 1 = pure geometric mean
            alpha = torch.sigmoid(self.product_gate)

            # Arithmetic mean (stable baseline)
            arith_mean = z.mean(dim=1)  # [B, L, D]

            # Geometric mean (consensus mechanism)
            # Use log-space for numerical stability
            eps = 1e-8
            z_abs = torch.abs(z) + eps
            log_abs = torch.log(z_abs)
            geom_mean_abs = torch.exp(log_abs.mean(dim=1))  # [B, L, D]

            # Sign handling: use sign of arithmetic sum (simple and stable)
            z_sign = torch.sign(z.sum(dim=1))
            geom_mean = geom_mean_abs * z_sign

            # Interpolate: α * geom_mean + (1-α) * arith_mean
            z = alpha * geom_mean + (1 - alpha) * arith_mean
        else:
            raise ValueError(f"Unknown merge method: {self.config.tree_merge_method}")

        return z

    def _expand_input_for_tree(self, input_embeddings: torch.Tensor, tree_width: int) -> torch.Tensor:
        """Expand input embeddings to match tree width.
        
        Args:
            input_embeddings: [batch_size, seq_len, hidden_size]
            tree_width: Number of tree branches
        Returns:
            expanded: [batch_size * tree_width, seq_len, hidden_size]
        """
        return input_embeddings.repeat_interleave(tree_width, dim=0)

    def empty_carry(self, batch_size: int):
        return ToTRM_InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            tree_width=1
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: ToTRM_InnerCarry):
        return ToTRM_InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
            tree_width=carry.tree_width
        )

    def forward(self, carry: ToTRM_InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[ToTRM_InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        original_batch_size = batch["inputs"].shape[0]

        # # DEBUG: Print batch inputs shape
        # print(f"[DEBUG] batch['inputs'].shape = {batch['inputs'].shape}")
        # print(f"        Meaning: [batch_size, seq_len] where seq_len = num_cells")

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # # DEBUG: Print input embeddings shape
        # print(f"[DEBUG] input_embeddings.shape = {input_embeddings.shape}")
        # print(f"        Meaning: [batch_size, puzzle_emb_len + seq_len, hidden_size]")

        # Initialize tree
        z_H, z_L = carry.z_H, carry.z_L
        current_tree_width = 1
        
        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                # Tree-of-Thought: branch on first n-1 L_cycles, merge on last
                for _L_step in range(self.config.L_cycles):
                    # Expand input and z_H to match current tree width
                    expanded_input = self._expand_input_for_tree(input_embeddings, current_tree_width)
                    expanded_z_H = self._expand_input_for_tree(z_H, current_tree_width)
                    
                    # Update z_L
                    z_L = self.L_level(z_L, expanded_z_H + expanded_input, **seq_info)

                    # # DEBUG: Print z_L shape (only on first iteration to avoid spam)
                    # if _H_step == 0 and _L_step == 0:
                    #     print(f"[DEBUG] z_L.shape = {z_L.shape}")
                    #     print(f"        Meaning: [batch_size * tree_width, puzzle_emb_len + seq_len, hidden_size]")
                    #     print(f"        Current: batch_size={original_batch_size}, tree_width={current_tree_width}")

                    # Branch if this is a branching step
                    if _L_step < self.config.tree_branching_steps:
                        z_L = self._branch_state(z_L, current_tree_width)
                        current_tree_width *= 2

                        # # DEBUG: Print z_L shape after branching (only on first H-cycle, first few L-steps)
                        # if _H_step == 0 and _L_step < 3:
                        #     print(f"[DEBUG] After branching L_step={_L_step}: z_L.shape = {z_L.shape}")
                        #     print(f"        tree_width doubled: {current_tree_width // 2} -> {current_tree_width}")

                # Merge tree before updating z_H
                z_L_merged = self._merge_tree(z_L, current_tree_width, original_batch_size)
                z_H = self.L_level(z_H, z_L_merged, **seq_info)
                
                # Reset tree for next H_cycle
                z_L = z_H.clone()
                current_tree_width = 1
        
        # Final H_cycle with grad
        for _L_step in range(self.config.L_cycles):
            expanded_input = self._expand_input_for_tree(input_embeddings, current_tree_width)
            expanded_z_H = self._expand_input_for_tree(z_H, current_tree_width)
            z_L = self.L_level(z_L, expanded_z_H + expanded_input, **seq_info)
            
            if _L_step < self.config.tree_branching_steps:
                z_L = self._branch_state(z_L, current_tree_width)
                current_tree_width *= 2
        
        # Capture z_L branches for diversity loss (before merging)
        # Reshape from [batch*tree_width, seq_len, hidden] to [batch, tree_width, seq_len, hidden]
        z_L_branches = z_L.view(original_batch_size, current_tree_width, *z_L.shape[1:])

        # Final merge
        z_L_merged = self._merge_tree(z_L, current_tree_width, original_batch_size)
        z_H = self.L_level(z_H, z_L_merged, **seq_info)

        # # DEBUG: Print final z_H shape
        # print(f"[DEBUG] Final z_H.shape = {z_H.shape}")
        # print(f"        Meaning: [batch_size, puzzle_emb_len + seq_len, hidden_size]")

        # LM Outputs
        new_carry = ToTRM_InnerCarry(z_H=z_H.detach(), z_L=z_H.clone().detach(), tree_width=1)

        # Main output: predict next token for each position
        lm_output_full = self.lm_head(z_H)  # [batch, 97, vocab_size]
        output = lm_output_full[:, self.puzzle_emb_len:]  # [batch, 81, vocab_size]

        # # DEBUG: Print output shapes
        # print(f"[DEBUG] self.lm_head(z_H).shape = {lm_output_full.shape}")
        # print(f"        Meaning: [batch_size, puzzle_emb_len + seq_len, vocab_size]")
        # print(f"[DEBUG] output.shape = {output.shape}")
        # print(f"        Meaning: [batch_size, seq_len, vocab_size] (puzzle positions removed)")

        # Q-head: predict halt/continue from position 0
        z_H_pos0 = z_H[:, 0]  # [batch, hidden_size]
        q_logits = self.q_head(z_H_pos0).to(torch.float32)  # [batch, 2]

        # # DEBUG: Print q_logits shape
        # print(f"[DEBUG] z_H[:, 0].shape = {z_H_pos0.shape}")
        # print(f"        Meaning: [batch_size, hidden_size] (position 0 only)")
        # print(f"[DEBUG] q_logits.shape = {q_logits.shape}")
        # print(f"        Meaning: [batch_size, 2] (Q-values: [halt, continue])")

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), z_L_branches


class ToTRM(nn.Module):
    """Tree-of-Thought Recursive Model with ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = ToTRM_Config(**config_dict)
        self.inner = ToTRM_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return ToTRM_Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: ToTRM_Carry, batch: Dict[str, torch.Tensor]) -> Tuple[ToTRM_Carry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits), z_L_branches = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "z_L_branches": z_L_branches  # For diversity loss
        }

        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step

            if self.training and (self.config.halt_max_steps > 1):
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    _, _, (next_q_halt_logits, next_q_continue_logits), _ = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return ToTRM_Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
