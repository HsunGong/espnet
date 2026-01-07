# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""DeepSpeed Expert Parallelism wrapper for Qwen3 MoE layers.

This module provides functionality to convert standard Qwen3 MoE layers
to DeepSpeed expert-parallel versions for distributed training across
multiple GPUs with expert sharding.
"""

import copy
from functools import partial
from typing import Tuple

import torch
import torch.nn.functional as F
from deepspeed import comm as dist
from deepspeed.moe.layer import MoE as DeepSpeed_MoE
from deepspeed.moe.sharded_moe import _AllToAll
from deepspeed.utils import groups
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeSparseMoeBlock,
)


class Qwen3MoeSparseMoeBlock_DeepSpeed_EP(DeepSpeed_MoE):
    """DeepSpeed expert-parallel wrapper for Qwen3 MoE blocks.

    This class wraps a Qwen3MoeSparseMoeBlock to enable expert parallelism
    using DeepSpeed's distributed training infrastructure. Experts are
    sharded across multiple GPUs to reduce memory requirements.

    Args:
        module: Original Qwen3MoeSparseMoeBlock to parallelize
        ep_size: Expert parallelism size (number of processes)

    Attributes:
        num_local_experts: Number of experts on this process
        ep_rank: Rank within the expert parallel group
        ep_group: DeepSpeed expert parallel process group
    """

    def __init__(self, module: torch.nn.Module, ep_size: int) -> None:
        """Initialize expert-parallel MoE layer.

        Args:
            module: Original MoE module to wrap
            ep_size: Number of processes for expert parallelism
        """
        # NOTE: We only initialize as torch.nn.Module, not DeepSpeed_MoE
        # to avoid conflicts with DeepSpeed's initialization
        torch.nn.Module.__init__(self)

        # Internal configuration
        self.enable_expert_tensor_parallelism = False
        self.num_experts = len(module.experts)
        self.ep_size = ep_size
        self.num_local_experts = self.num_experts // self.ep_size
        self.norm_topk_prob = module.norm_topk_prob
        self.top_k = module.top_k

        # Validate configuration
        if self.ep_size <= 1:
            raise ValueError(f"ep_size must be > 1, got {ep_size}")
        if self.num_experts % self.ep_size != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible "
                f"by ep_size ({ep_size})"
            )

        # (3) setup the ep_group
        self.expert_group_name = f"ep_size_{self.ep_size}"
        self.set_deepspeed_parallelism()

        # (4) copy the expert modules from the original module
        self.gate = copy.deepcopy(module.gate)
        start = self.ep_rank * self.num_local_experts
        end = (self.ep_rank + 1) * self.num_local_experts
        self.experts = torch.nn.ModuleList(
            [copy.deepcopy(e) for e in module.experts[start:end]]
        )
        for expert in self.experts:
            for param in expert.parameters():
                param.allreduce = False
                param.group_name = self.expert_group_name

    def set_deepspeed_parallelism(
        self, use_data_before_expert_parallel_: bool = False
    ) -> None:
        """Set up DeepSpeed parallelism groups.

        Args:
            use_data_before_expert_parallel_: Whether to use data parallelism
                before expert parallelism in group creation
        """
        self._create_process_groups(
            use_data_before_expert_parallel_=(use_data_before_expert_parallel_)
        )

    def _create_process_groups(
        self, use_data_before_expert_parallel_: bool = False
    ) -> None:
        # Create process group for a layer if needed
        if self.expert_group_name not in groups._get_expert_parallel_group_dict():
            if (groups.mpu is None) or (not self.enable_expert_tensor_parallelism):
                # Condition 1 - no groups.mpu means no tensor parallelism
                # Condition 2 - disabling expert tensor parallelism on purpose
                groups._create_expert_and_data_parallel(
                    self.ep_size,
                    use_data_before_expert_parallel_=use_data_before_expert_parallel_,
                )
            else:
                # expert tensor parallelism is enabled
                groups._create_expert_data_and_model_parallel(
                    self.ep_size,
                    mpu=groups.mpu,
                    use_data_before_expert_parallel_=use_data_before_expert_parallel_,
                )
        # Set the group handle for the MOELayer (deepspeed_moe) object
        self.ep_group = groups._get_expert_parallel_group(self.expert_group_name)
        self.ep_rank = dist.get_rank(group=self.ep_group)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with memory-efficient sparse dispatch.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]

        Returns:
            Tuple of (output hidden states, router logits)
        """
        batch_size, sequence_length, hidden_dim = hidden_states.size()
        num_tokens = batch_size * sequence_length
        hidden_states = hidden_states.view(num_tokens, hidden_dim)

        # Router and sparse dispatch preparation
        router_logits = self.gate(hidden_states)
        dispatch_idx, topk_indices, topk_weights, positions, capacity = \
            self.prepare_dispatch(router_logits)

        # Gather tokens for each expert using sparse indexing
        # Pad with zero row for empty slots (dispatch_idx == num_tokens)
        hidden_padded = torch.cat([
            hidden_states,
            torch.zeros(1, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)
        ], dim=0)

        expert_input = hidden_padded[dispatch_idx.view(-1)].view(
            self.num_experts, capacity, hidden_dim
        )

        # All-to-all: dispatch tokens to their expert's rank
        expert_input = expert_input.view(
            self.ep_size, self.num_local_experts, capacity, hidden_dim
        )
        expert_input = _AllToAll.apply(self.ep_group, expert_input)

        # Forward through local experts
        expert_input = expert_input.chunk(self.num_local_experts, dim=1)
        expert_output = torch.stack(
            [e(h.squeeze(1)) for e, h in zip(self.experts, expert_input)], dim=1
        )
        expert_output = expert_output.view(self.num_experts, capacity, hidden_dim)

        # All-to-all: return outputs to original ranks
        expert_output = _AllToAll.apply(self.ep_group, expert_output)

        # Combine: weighted sum of expert outputs back to tokens
        output = torch.zeros(
            num_tokens, hidden_dim,
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        expert_output_flat = expert_output.view(-1, hidden_dim)

        for k in range(self.top_k):
            expert_k = topk_indices[:, k]
            pos_k = positions[:, k]
            weight_k = topk_weights[:, k:k+1]

            linear_idx = expert_k * capacity + pos_k
            gathered = expert_output_flat[linear_idx]
            output += weight_k.type_as(output) * gathered

        return output.view(batch_size, sequence_length, hidden_dim), router_logits

    def prepare_dispatch(
        self, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Prepare sparse dispatch tensors for expert routing.

        Args:
            logits: Router logits [num_tokens, num_experts]

        Returns:
            Tuple of:
                - dispatch_idx: [num_experts, capacity] token IDs per slot
                - topk_indices: [num_tokens, top_k] expert IDs per token
                - topk_weights: [num_tokens, top_k] routing weights (with gradients)
                - positions: [num_tokens, top_k] position in expert buffer
                - capacity: max tokens per expert
        """
        num_tokens = logits.size(0)
        device = logits.device

        # Top-k routing - keep gradients for routing weights
        prob = F.softmax(logits, dim=1, dtype=torch.float)
        topk_weights, topk_indices = torch.topk(prob, self.top_k, dim=1)
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)

        # Everything below is index computation - no gradients needed
        with torch.no_grad():
            flat_expert_indices = topk_indices.view(-1)
            expert_counts = torch.bincount(flat_expert_indices, minlength=self.num_experts)
            capacity = expert_counts.max()
            dist.all_reduce(capacity, op=dist.ReduceOp.MAX, group=self.ep_group)
            capacity = capacity.item()

            # Compute positions within each expert's buffer using vectorized ops
            sorted_expert_idx, sort_perm = torch.sort(flat_expert_indices, stable=True)
            arange = torch.arange(1, len(sorted_expert_idx) + 1, device=device, dtype=torch.long)

            # Find segment boundaries and compute positions
            changes = torch.cat([
                torch.ones(1, dtype=torch.bool, device=device),
                sorted_expert_idx[1:] != sorted_expert_idx[:-1]
            ])
            segment_starts = arange * changes
            segment_start_cummax = torch.cummax(segment_starts, dim=0)[0]
            positions_sorted = arange - segment_start_cummax

            # Unsort to original order
            positions = torch.zeros_like(flat_expert_indices)
            positions[sort_perm] = positions_sorted
            positions = positions.view(num_tokens, self.top_k)

            # Build dispatch index (num_tokens as padding for empty slots)
            dispatch_idx = torch.full(
                (self.num_experts, capacity), num_tokens, dtype=torch.long, device=device
            )
            linear_indices = topk_indices * capacity + positions
            token_ids = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, self.top_k)
            dispatch_idx.view(-1).scatter_(0, linear_indices.view(-1), token_ids.reshape(-1))

        return dispatch_idx, topk_indices, topk_weights, positions, capacity


def replace_moe_layer(
    model: torch.nn.Module,
    ep_size: int,
    original_cls: type,
    ep_cls: type,
) -> torch.nn.Module:
    """Replace MoE layers with expert-parallel versions.

    Recursively traverses the model and replaces all instances of
    original_cls with ep_cls initialized with expert parallelism.

    Args:
        model: Model to modify
        ep_size: Expert parallelism size
        original_cls: Original MoE layer class to replace
        ep_cls: Expert-parallel class to use as replacement

    Returns:
        Modified model with expert-parallel MoE layers
    """
    if ep_size <= 1:
        return model

    def recursive_replace(module, parent_name=""):
        """Recursively replace MoE layers in the module tree."""
        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            if isinstance(child, original_cls):
                new_child = ep_cls(child, ep_size)
                setattr(module, name, new_child)
            else:
                recursive_replace(child, full_name)

    recursive_replace(model)
    return model


# Convenience function for replacing Qwen3 MoE layers
replace_qwen3_moe_layer = partial(
    replace_moe_layer,
    original_cls=Qwen3MoeSparseMoeBlock,
    ep_cls=Qwen3MoeSparseMoeBlock_DeepSpeed_EP,
)

__all__ = [
    "Qwen3MoeSparseMoeBlock_DeepSpeed_EP",
    "replace_moe_layer",
    "replace_qwen3_moe_layer",
]
