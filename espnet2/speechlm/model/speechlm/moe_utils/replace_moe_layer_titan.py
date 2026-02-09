# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""TorchTitan Expert Parallelism wrapper for Qwen3 MoE layers.

This module provides functionality to convert standard Qwen3 MoE layers
to TorchTitan expert-parallel versions for distributed training using
FSDP2 and TorchTitan's Expert Parallelism infrastructure.

Key differences from DeepSpeed EP:
- Uses stacked expert weights for torch._grouped_mm efficiency
- Uses DTensor-based all-to-all communication
- Integrates with FSDP2 via separate edp_mesh for expert sharding
"""

from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeSparseMoeBlock,
    load_balancing_loss_func,
)


class GroupedExperts(nn.Module):
    """TorchTitan-compatible grouped experts module.

    This module stores expert weights in stacked format for efficient
    computation using torch._grouped_mm. Compatible with TorchTitan's
    ExpertParallel for DTensor-based sharding and all-to-all communication.

    Attributes:
        num_experts: Total number of experts
        w1: Gate projection weights [num_experts, hidden_dim, input_dim]
        w2: Down projection weights [num_experts, input_dim, hidden_dim]
        w3: Up projection weights [num_experts, hidden_dim, input_dim]
    """

    def __init__(self, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor):
        super().__init__()
        self.num_experts = w1.shape[0]

        # Stacked expert weights
        # w1/w3: (num_experts, hidden_dim, dim) - gate/up projection
        # w2: (num_experts, dim, hidden_dim) - down projection
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.w3 = nn.Parameter(w3)

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through grouped experts using torch._grouped_mm.

        Args:
            x: Routed input tokens [total_tokens, dim]
            num_tokens_per_expert: Number of tokens per expert [num_experts]

        Returns:
            Expert outputs [total_tokens, dim]
        """
        # Handle DTensor parameters
        w1 = self.w1.to_local() if isinstance(self.w1, DTensor) else self.w1
        w2 = self.w2.to_local() if isinstance(self.w2, DTensor) else self.w2
        w3 = self.w3.to_local() if isinstance(self.w3, DTensor) else self.w3

        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

        # grouped_mm requires bfloat16 — cast inputs/weights explicitly
        x_bf16 = x.bfloat16()
        w1_bf16 = w1.bfloat16()
        w2_bf16 = w2.bfloat16()
        w3_bf16 = w3.bfloat16()

        # SwiGLU: down(silu(gate(x)) * up(x))
        h = F.silu(F.grouped_mm(x_bf16, w1_bf16.transpose(-2, -1), offs=offsets))
        h = h * F.grouped_mm(x_bf16, w3_bf16.transpose(-2, -1), offs=offsets)
        out = F.grouped_mm(h, w2_bf16.transpose(-2, -1), offs=offsets)

        return out.type_as(x)


class Qwen3MoeSparseMoeBlock_TorchTitan(Qwen3MoeSparseMoeBlock):
    """TorchTitan-compatible wrapper for Qwen3 MoE blocks.

    This class wraps a Qwen3MoeSparseMoeBlock to enable TorchTitan's
    Expert Parallelism using DTensor-based sharding and FSDP2.

    Converts HuggingFace's nn.ModuleList of experts to stacked weights
    for efficient grouped_mm computation.

    Inherits from Qwen3MoeSparseMoeBlock so HuggingFace's OutputRecorder
    can still detect this class for router_logits collection.

    Args:
        module: Original Qwen3MoeSparseMoeBlock to parallelize

    Attributes:
        moe_enabled: Flag for parallelization (always True)
        num_experts: Number of experts
        top_k: Number of experts selected per token
        experts: GroupedExperts module with stacked weights
        gate: Router gate (replicated across all ranks)
    """

    def __init__(self, module: Qwen3MoeSparseMoeBlock) -> None:
        """Initialize TorchTitan-compatible MoE layer.

        Args:
            module: Original HuggingFace MoE module to wrap
        """
        # Skip Qwen3MoeSparseMoeBlock.__init__ (expects config), call nn.Module directly
        nn.Module.__init__(self)

        # Configuration
        self.moe_enabled = True  # Flag for parallelization detection
        self.num_experts = len(module.experts)
        self.top_k = module.top_k
        self.norm_topk_prob = module.norm_topk_prob

        # Move gate directly (old module will be replaced and GC'd)
        self.gate = module.gate

        # Stack expert weights with torch.stack (single C++ op per weight)
        # instead of allocating empty tensors + Python copy loop over 128 experts
        w1 = torch.stack([e.gate_proj.weight.data for e in module.experts])
        w2 = torch.stack([e.down_proj.weight.data for e in module.experts])
        w3 = torch.stack([e.up_proj.weight.data for e in module.experts])

        self.experts = GroupedExperts(w1, w2, w3)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with sparse routing.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]

        Returns:
            Tuple of (output hidden states, router logits for aux loss)
        """
        batch_size, sequence_length, hidden_dim = hidden_states.size()
        num_tokens = batch_size * sequence_length
        hidden_states = hidden_states.view(num_tokens, hidden_dim)

        # Router: compute routing scores
        router_logits = self.gate(hidden_states)

        # Top-k routing with softmax
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)

        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        topk_weights = topk_weights.to(hidden_states.dtype)

        # Compute number of tokens per expert
        with torch.no_grad():
            flat_indices = topk_indices.view(-1)
            num_tokens_per_expert = torch.bincount(
                flat_indices, minlength=self.num_experts
            ).to(torch.int32)

        # Token reordering for expert computation
        # Sort tokens by expert index
        sorted_indices = torch.argsort(flat_indices, stable=True)
        token_indices = sorted_indices // self.top_k

        # Gather tokens for each expert
        routed_input = hidden_states[token_indices]

        # Apply routing weights before expert computation
        flat_weights = topk_weights.view(-1)
        sorted_weights = flat_weights[sorted_indices]
        routed_input = routed_input * sorted_weights.unsqueeze(-1)

        # Forward through experts
        routed_output = self.experts(routed_input, num_tokens_per_expert)

        # Unsort and combine outputs
        output = torch.zeros(
            num_tokens * self.top_k, hidden_dim,
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        output[sorted_indices] = routed_output

        # Sum over top-k dimension
        output = output.view(num_tokens, self.top_k, hidden_dim).sum(dim=1)

        return output.view(batch_size, sequence_length, hidden_dim), router_logits


def replace_moe_layer_titan(
    model: nn.Module,
    original_cls: type,
    titan_cls: type,
    load_balancing_loss_func: callable = None,
) -> nn.Module:
    """Replace MoE layers with TorchTitan-compatible versions.

    Recursively traverses the model and replaces all instances of
    original_cls with titan_cls.

    Args:
        model: Model to modify
        original_cls: Original MoE layer class to replace
        titan_cls: TorchTitan-compatible class to use as replacement
        load_balancing_loss_func: Function to compute MoE load balancing loss

    Returns:
        Modified model with TorchTitan MoE layers
    """

    def recursive_replace(module, parent_name=""):
        """Recursively replace MoE layers in the module tree."""
        # Collect children first to avoid modifying dict during iteration
        children = list(module.named_children())
        for name, child in children:
            full_name = f"{parent_name}.{name}" if parent_name else name
            if isinstance(child, original_cls):
                new_child = titan_cls(child)
                setattr(module, name, new_child)

                # Also set moe reference on parent for FSDP detection
                parent = module
                if hasattr(parent, 'mlp') and parent.mlp is new_child:
                    parent.moe_enabled = True
                    parent.moe = new_child
                print(f"Replaced MoE layer: {full_name}", flush=True)
            else:
                recursive_replace(child, full_name)

    recursive_replace(model)

    # Set load_balancing_loss_func on model for aux_loss computation
    if load_balancing_loss_func is not None:
        model.load_balancing_loss_func = load_balancing_loss_func

    return model


# Convenience function for replacing Qwen3 MoE layers
replace_qwen3_moe_layer_titan = partial(
    replace_moe_layer_titan,
    original_cls=Qwen3MoeSparseMoeBlock,
    titan_cls=Qwen3MoeSparseMoeBlock_TorchTitan,
    load_balancing_loss_func=load_balancing_loss_func,
)

__all__ = [
    "GroupedExperts",
    "Qwen3MoeSparseMoeBlock_TorchTitan",
    "replace_moe_layer_titan",
    "replace_qwen3_moe_layer_titan",
]
