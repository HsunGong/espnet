# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Expert Parallel MoE block for HuggingFace Qwen3 MoE models.

This module replaces `Qwen3MoeSparseMoeBlock` with an Expert Parallel version
that distributes experts across ranks using all-to-all communication.

Expert weights are stacked into 3D tensors and processed with
`torch._grouped_mm` for fused multi-expert computation (instead of a
Python loop over individual experts).

EP data flow per MoE block:
    1. Route tokens (local gate computation on each rank)
    2. Sort token-expert pairs by global expert index
    3. All-to-all dispatch tokens to expert-owning ranks
    4. Permute received tokens from (rank, expert) to (expert, rank) order
    5. Process all local experts via fused grouped_mm
    6. Unpermute back to (rank, expert) order
    7. All-to-all combine (reverse dispatch)
    8. Apply routing weights, unsort, and accumulate
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._functional_collectives import (
    all_to_all_single,
    all_to_all_single_autograd,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeSparseMoeBlock,
)

try:
    from torchtitan.models.moe.utils import _permute, _unpermute

    _TRITON_PERMUTE_AVAILABLE = True
except ImportError:
    _TRITON_PERMUTE_AVAILABLE = False

logger = logging.getLogger(__name__)


class GroupedExperts(nn.Module):
    """Stacked expert weights with fused grouped_mm computation.

    Stacks individual expert Linear weights into 3D Parameter tensors
    and uses torch._grouped_mm for fused multi-expert matrix multiplication.

    Weight layout (Qwen3 MoE SwiGLU, no biases):
        w1 (gate_proj): (num_experts, intermediate_size, hidden_size)
        w2 (down_proj): (num_experts, hidden_size, intermediate_size)
        w3 (up_proj):   (num_experts, intermediate_size, hidden_size)

    Forward: out = down_proj(silu(gate_proj(x)) * up_proj(x))

    Args:
        experts: List of Qwen3MoeMLP modules to stack.
    """

    def __init__(self, experts: list):
        super().__init__()
        self.num_experts = len(experts)

        # Verify no biases (Qwen3 MoE experts don't have biases)
        assert experts[0].gate_proj.bias is None, (
            "GroupedExperts does not support biases"
        )

        # Stack weights from individual experts into 3D tensors
        self.w1 = nn.Parameter(
            torch.stack([e.gate_proj.weight for e in experts])
        )
        self.w2 = nn.Parameter(
            torch.stack([e.down_proj.weight for e in experts])
        )
        self.w3 = nn.Parameter(
            torch.stack([e.up_proj.weight for e in experts])
        )

        logger.info(
            f"GroupedExperts: stacked {self.num_experts} experts, "
            f"w1={list(self.w1.shape)}, w2={list(self.w2.shape)}, "
            f"w3={list(self.w3.shape)}"
        )

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """Process tokens through stacked experts using grouped_mm.

        Args:
            x: Input tokens sorted by expert, shape (total_tokens, hidden_dim).
            num_tokens_per_expert: Token count per local expert,
                shape (num_experts,).

        Returns:
            Output tensor, shape (total_tokens, hidden_dim).
        """
        # Extract local tensors from DTensor (torch._grouped_mm requires
        # regular tensors, not DTensors)
        if isinstance(self.w1, DTensor):
            w1 = self.w1.to_local()
            w2 = self.w2.to_local()
            w3 = self.w3.to_local()
        else:
            w1 = self.w1
            w2 = self.w2
            w3 = self.w3

        # Cumulative offsets for grouped_mm (marks end of each expert's tokens)
        offsets = torch.cumsum(
            num_tokens_per_expert, dim=0, dtype=torch.int32
        )

        # SwiGLU: out = down_proj(silu(gate_proj(x)) * up_proj(x))
        # torch._grouped_mm requires bfloat16 inputs
        x_bf16 = x if x.dtype == torch.bfloat16 else x.bfloat16()
        w1_t = w1.transpose(-2, -1) if w1.dtype == torch.bfloat16 else w1.bfloat16().transpose(-2, -1)
        w3_t = w3.transpose(-2, -1) if w3.dtype == torch.bfloat16 else w3.bfloat16().transpose(-2, -1)
        w2_t = w2.transpose(-2, -1) if w2.dtype == torch.bfloat16 else w2.bfloat16().transpose(-2, -1)

        h = F.silu(torch._grouped_mm(x_bf16, w1_t, offs=offsets))
        h = h * torch._grouped_mm(x_bf16, w3_t, offs=offsets)
        out = torch._grouped_mm(h, w2_t, offs=offsets).type_as(x)

        return out


class ExpertParallelMoeBlock(Qwen3MoeSparseMoeBlock):
    """MoE block with Expert Parallelism via all-to-all communication.

    Inherits from `Qwen3MoeSparseMoeBlock` so that HF's `OutputRecorder`
    (which uses `isinstance` check) can find this module and capture
    router_logits for the load balancing auxiliary loss.

    Each rank holds a subset of experts (local experts) and tokens are
    dispatched to the expert-owning rank via all-to-all collectives.

    Uses GroupedExperts with torch._grouped_mm for fused multi-expert
    computation instead of a Python loop over individual experts.

    Args:
        original_block: The original Qwen3MoeSparseMoeBlock to replace.
        ep_mesh: DeviceMesh for the EP dimension.
    """

    def __init__(self, original_block: Qwen3MoeSparseMoeBlock, ep_mesh: DeviceMesh):
        # Skip parent __init__ (which would create all experts from config).
        # Directly call nn.Module.__init__ instead.
        nn.Module.__init__(self)

        self.num_experts = original_block.num_experts
        self.top_k = original_block.top_k
        self.norm_topk_prob = original_block.norm_topk_prob

        # Router (replicated on all ranks — same gate weights)
        self.gate = original_block.gate

        # EP mesh info
        self.ep_size = ep_mesh.size()
        self.ep_rank = ep_mesh.get_local_rank()
        self.ep_group = ep_mesh.get_group()

        assert self.num_experts % self.ep_size == 0, (
            f"num_experts ({self.num_experts}) must be divisible by "
            f"ep_size ({self.ep_size})"
        )
        self.experts_per_rank = self.num_experts // self.ep_size

        # Keep only local experts, stacked into GroupedExperts for fused
        # grouped_mm computation
        start = self.ep_rank * self.experts_per_rank
        end = start + self.experts_per_rank
        self.experts = GroupedExperts(list(original_block.experts[start:end]))

        # Register expert weights as DTensors on ep_mesh with Shard(0).
        # This gives params "ep" in device_mesh.mesh_dim_names, which
        # torchtitan's clip_grad_norm_ uses to separate EP vs dense params
        # (they live on different FSDP meshes and can't be stacked).
        for name in ("w1", "w2", "w3"):
            param = getattr(self.experts, name)
            dt = DTensor.from_local(param.data, ep_mesh, [Shard(0)])
            setattr(self.experts, name, nn.Parameter(dt))

        logger.info(
            f"EP rank {self.ep_rank}: keeping experts [{start}:{end}] "
            f"({self.experts_per_rank}/{self.num_experts}), "
            f"using GroupedExperts with torch._grouped_mm"
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple:
        """Forward pass with Expert Parallel all-to-all communication.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_dim).

        Returns:
            Tuple of (output, router_logits):
                - output: shape (batch, seq_len, hidden_dim)
                - router_logits: shape (batch * seq_len, num_experts)
        """
        bsz, seq_len, hidden_dim = hidden_states.shape
        num_tokens = bsz * seq_len
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # --- Step 1: Route (local computation, each rank has different tokens) ---
        router_logits = self.gate(hidden_states_flat)  # (T, E)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )  # (T, K)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states_flat.dtype)

        # --- Step 2: Prepare (token, expert, weight) triples ---
        flat_expert_indices = selected_experts.view(-1)  # (T*K,)
        flat_token_indices = (
            torch.arange(num_tokens, device=hidden_states.device)
            .unsqueeze(1)
            .expand(-1, self.top_k)
            .reshape(-1)
        )  # (T*K,)
        flat_weights = routing_weights.view(-1)  # (T*K,)

        # Sort by expert index for contiguous all-to-all splits
        sort_order = flat_expert_indices.argsort(stable=True)
        sorted_token_indices = flat_token_indices[sort_order]
        sorted_weights = flat_weights[sort_order]
        sorted_tokens = hidden_states_flat[sorted_token_indices]  # (T*K, D)

        # --- Step 3: Count tokens per expert ---
        num_tokens_per_expert = torch.histc(
            flat_expert_indices.float(),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        ).long()

        # --- Step 4: All-to-all dispatch ---
        # Exchange per-expert counts across EP ranks
        with torch.no_grad():
            # all_to_all_single with None splits: equal-split on dim 0
            # Input shape (E,) split into ep_size chunks of (E/ep_size,)
            # Chunk i (counts for experts on rank i) goes to rank i
            recv_counts = all_to_all_single(
                num_tokens_per_expert, None, None, group=self.ep_group
            )
            recv_counts = torch.ops._c10d_functional.wait_tensor(recv_counts)

            # input_splits[i] = total tokens this rank sends to rank i
            input_splits = (
                num_tokens_per_expert.view(self.ep_size, -1)
                .sum(dim=1)
                .to("cpu", non_blocking=True)
            )
            # output_splits[j] = total tokens this rank receives from rank j
            output_splits = (
                recv_counts.view(self.ep_size, -1).sum(dim=1).cpu().tolist()
            )
            # Materialize input_splits after output_splits blocking transfer
            input_splits = input_splits.tolist()

        # Dispatch tokens to expert-owning ranks
        dispatched_tokens = all_to_all_single_autograd(
            sorted_tokens.contiguous(),
            output_splits,
            input_splits,
            self.ep_group,
        )

        # --- Step 5: Permute, process via grouped_mm, unpermute ---
        # Dispatched tokens arrive in (rank, expert) order:
        #   [r0_e0, r0_e1, ..., r0_eK, r1_e0, r1_e1, ..., r1_eK, ...]
        # grouped_mm needs (expert, rank) order:
        #   [e0_r0, e0_r1, ..., e0_rN, e1_r0, e1_r1, ..., e1_rN, ...]
        total_recv = sum(output_splits)
        if total_recv > 0:
            if _TRITON_PERMUTE_AVAILABLE:
                # Triton kernel: permute to (expert, rank) order with
                # alignment padding for better grouped_mm performance
                input_shape, permuted_tokens, permuted_indices, aligned_counts = (
                    _permute(
                        dispatched_tokens,
                        recv_counts,
                        self.ep_size,
                        self.experts_per_rank,
                    )
                )
                processed_tokens = self.experts(
                    permuted_tokens, aligned_counts
                )
                output_tokens = _unpermute(
                    processed_tokens, input_shape, permuted_indices
                )
            else:
                # Fallback: argsort-based permutation
                local_expert_ids = torch.arange(
                    self.experts_per_rank, device=hidden_states.device
                ).repeat(self.ep_size)
                token_expert_ids = local_expert_ids.repeat_interleave(recv_counts)
                permute_order = token_expert_ids.argsort(stable=True)
                permuted_tokens = dispatched_tokens[permute_order]

                num_tokens_per_local_expert = recv_counts.view(
                    self.ep_size, self.experts_per_rank
                ).sum(dim=0)
                processed_tokens = self.experts(
                    permuted_tokens, num_tokens_per_local_expert
                )

                unpermute_order = torch.empty_like(permute_order)
                unpermute_order[permute_order] = torch.arange(
                    total_recv, device=hidden_states.device
                )
                output_tokens = processed_tokens[unpermute_order]
        else:
            output_tokens = dispatched_tokens

        # --- Step 6: All-to-all combine (reverse dispatch) ---
        combined_output = all_to_all_single_autograd(
            output_tokens.contiguous(),
            input_splits,
            output_splits,
            self.ep_group,
        )

        # --- Step 7: Apply weights, unsort, and accumulate ---
        # combined_output is in the same order as sorted_tokens
        weighted_output = combined_output * sorted_weights.unsqueeze(-1)
        final_output = torch.zeros(
            (num_tokens, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        final_output.index_add_(0, sorted_token_indices, weighted_output)

        final_output = final_output.view(bsz, seq_len, hidden_dim)
        return final_output, router_logits
