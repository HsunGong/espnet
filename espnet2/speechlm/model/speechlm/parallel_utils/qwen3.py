# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Parallelization utilities for HuggingFace Qwen3 models.

This module provides Expert Parallelism, activation checkpointing,
torch.compile, and FSDP2 wrapping for HuggingFace Qwen3 (dense and MoE)
models used in the SpeechLM framework. It follows TorchTitan's
parallelization patterns adapted for the HuggingFace model structure.

HuggingFace Qwen3 model structure:
    model.model.embed_tokens  - Token embeddings
    model.model.layers        - List of transformer layers
    model.model.norm          - Final RMSNorm
    model.lm_head             - Output projection

For MoE models (e.g., Qwen3-30B-A3B), some layers have:
    layer.mlp = Qwen3MoeSparseMoeBlock
        .gate: nn.Linear(hidden_size, num_experts)   # Router
        .experts: nn.ModuleList of Qwen3MoeMLP        # Individual experts

Additional multimodal components (added by ParallelHFModel):
    model.multimodal_io_dict  - Dict of multimodal IO handlers
    model.adaptor             - Dict of linear adaptors for continuous modalities
    model.stream_emb          - Stream embeddings
"""

import logging
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torchtitan.distributed import ParallelDims

from espnet2.speechlm.model.speechlm.parallel_utils.expert_parallel import (
    ExpertParallelMoeBlock,
)

logger = logging.getLogger(__name__)


def _is_moe_layer(layer: nn.Module) -> bool:
    """Check if a transformer layer uses MoE (has gate + experts in mlp)."""
    return hasattr(layer.mlp, "gate") and hasattr(layer.mlp, "experts")


def parallelize_qwen3_hf(
    model: nn.Module,
    parallel_dims: ParallelDims,
    titan_config: Dict[str, Any],
) -> nn.Module:
    """Apply parallelization to HuggingFace Qwen3 model.

    Order: EP -> AC -> torch.compile -> FSDP (EP-aware)
    (following TorchTitan's convention)

    Args:
        model: HuggingFace Qwen3 model (possibly wrapped with multimodal components)
        parallel_dims: TorchTitan ParallelDims object with device meshes
        titan_config: Configuration dict containing:
            - activation_checkpoint: AC ratio 0.0-1.0 (default: 0.0).
              1.0 = all layers, 0.5 = every other layer.
            - compile: Whether to enable torch.compile (default: false)
            - compile_mode: Compile mode (default: "default")
            - mixed_precision_param: Parameter dtype (default: "bfloat16")
            - mixed_precision_reduce: Reduce dtype (default: "float32")
            - reshard_after_forward: Whether to reshard params after forward
              (default: true). true saves memory, false is faster.

    Returns:
        Parallelized model
    """
    # 1. Expert Parallelism (must come first — replaces MoE blocks)
    if parallel_dims.ep_enabled:
        model = apply_expert_parallel_qwen3(model, parallel_dims)

    # 2. Activation Checkpointing
    ac_ratio = titan_config.get("activation_checkpoint", 0.0)
    if ac_ratio:
        if ac_ratio is True:
            ac_ratio = 1.0
        model = apply_activation_checkpoint_qwen3(model, ratio=ac_ratio)

    # 3. Torch Compile
    if titan_config.get("compile", False):
        model = apply_torch_compile_qwen3(model, titan_config)

    # 4. FSDP (EP-aware wrapping for MoE layers)
    if parallel_dims.fsdp_enabled:
        model = apply_fsdp_qwen3(model, parallel_dims, titan_config)

    return model


def apply_expert_parallel_qwen3(
    model: nn.Module,
    parallel_dims: ParallelDims,
) -> nn.Module:
    """Replace MoE blocks with Expert Parallel versions.

    Iterates through transformer layers, detects MoE layers, and replaces
    each Qwen3MoeSparseMoeBlock with an ExpertParallelMoeBlock that
    distributes experts across EP ranks via all-to-all communication.

    Must be applied BEFORE activation checkpointing, compile, and FSDP.

    Args:
        model: HuggingFace Qwen3 MoE model
        parallel_dims: TorchTitan ParallelDims with EP mesh

    Returns:
        Model with MoE blocks replaced by EP-aware blocks
    """
    ep_mesh = parallel_dims.get_mesh("ep")
    for layer in model.model.layers:
        if _is_moe_layer(layer):
            layer.mlp = ExpertParallelMoeBlock(layer.mlp, ep_mesh)
    return model


def apply_fsdp_qwen3(
    model: nn.Module,
    parallel_dims: ParallelDims,
    titan_config: Dict[str, Any],
) -> nn.Module:
    """Apply FSDP2 to HuggingFace Qwen3 model structure.

    Strategy:
    - Only wrap modules that have trainable parameters.
    - Wrap individual components first, then the root model.
    - For MoE layers with EP: wrap GroupedExperts on edp_mesh (with
      gradient_divide_factor) BEFORE wrapping the layer on dp_mesh.
    - After wrapping, permanently unshard peripheral components (norm, lm_head,
      stream_emb, multimodal_io_dict, adaptor) to avoid repeated all-gather
      overhead for small or frequently-accessed modules. Only embed_tokens and
      transformer layers stay in the normal FSDP shard/unshard cycle.

    Args:
        model: HuggingFace Qwen3 model to wrap with FSDP
        parallel_dims: TorchTitan ParallelDims with device meshes
        titan_config: Configuration dict

    Returns:
        FSDP-wrapped model
    """
    # (1) Build FSDP config for dense (non-expert) params
    param_dtype = getattr(torch, titan_config.get("mixed_precision_param", "bfloat16"))
    reduce_dtype = getattr(torch, titan_config.get("mixed_precision_reduce", "float32"))
    reshard_after_forward = titan_config.get("reshard_after_forward", True)

    if parallel_dims.dp_replicate_enabled:
        dp_mesh = parallel_dims.get_mesh(["dp_replicate", "fsdp"])
    else:
        dp_mesh = parallel_dims.get_mesh("fsdp")

    fsdp_config = {
        "mesh": dp_mesh,
        "mp_policy": MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype),
        "reshard_after_forward": reshard_after_forward,
    }

    # Build expert FSDP config (only when EP is enabled)
    ep_enabled = parallel_dims.ep_enabled
    if ep_enabled:
        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)
        fsdp_ep_config = {
            "mesh": edp_mesh,
            "mp_policy": MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype
            ),
            "reshard_after_forward": reshard_after_forward,
        }
        gradient_divide_factor = parallel_dims.fsdp_gradient_divide_factor

    def _shard(module: nn.Module):
        if any(p.requires_grad for p in module.parameters()):
            fully_shard(module, **fsdp_config)

    def _unshard(module: nn.Module):
        if hasattr(module, "unshard"):
            module.unshard()

    # (2.1) input embeddings
    _shard(model.model.embed_tokens)

    # (2.2) layers (with EP-aware expert wrapping for MoE layers)
    for layer in model.model.layers:
        if ep_enabled and isinstance(layer.mlp, ExpertParallelMoeBlock):
            # Wrap GroupedExperts (stacked expert weights) on edp_mesh
            # BEFORE wrapping the layer on dp_mesh. FSDP2 hooks fire on
            # GroupedExperts.__call__() which is invoked during forward.
            fully_shard(layer.mlp.experts, **fsdp_ep_config)
            layer.mlp.experts.set_gradient_divide_factor(
                gradient_divide_factor
            )
        _shard(layer)

    # (2.3) norm, lm_head, stream_emb
    _shard(model.model.norm)
    _shard(model.lm_head)
    _shard(model.stream_emb)

    # (2.4) multimodal_io_dict and adaptor
    for module in model.multimodal_io_dict.values():
        if isinstance(module, nn.Module):
            _shard(module)

    for module in model.adaptor.values():
        if isinstance(module, nn.Module):
            _shard(module)

    # (2.5) root
    fully_shard(model, **fsdp_config)

    # (3) unshard peripheral modules
    _unshard(model.model.norm)
    _unshard(model.lm_head)
    _unshard(model.stream_emb)
    for module in model.multimodal_io_dict.values():
        if isinstance(module, nn.Module):
            _unshard(module)
    for module in model.adaptor.values():
        if isinstance(module, nn.Module):
            _unshard(module)

    return model


def apply_activation_checkpoint_qwen3(
    model: nn.Module, ratio: float = 1.0
) -> nn.Module:
    """Apply activation checkpointing to transformer layers.

    Wraps transformer layers with checkpoint_wrapper for memory savings.
    Must be applied before torch.compile and FSDP.

    Args:
        model: HuggingFace Qwen3 model
        ratio: Fraction of layers to checkpoint (0.0-1.0).
            1.0 = all layers, 0.5 = every other layer, etc.

    Returns:
        Model with activation checkpointing applied
    """
    num_layers = len(model.model.layers)
    ac_freq = max(1, round(1.0 / ratio))

    count = 0
    for idx, layer in enumerate(model.model.layers):
        if idx % ac_freq == 0:
            model.model.layers[idx] = checkpoint_wrapper(layer)
            count += 1

    logger.info(
        f"Applied activation checkpointing to {count}/{num_layers} layers "
        f"(ratio={ratio}, freq=every {ac_freq})"
    )
    return model


def apply_torch_compile_qwen3(
    model: nn.Module,
    titan_config: Dict[str, Any],
) -> nn.Module:
    """Apply torch.compile to transformer layers.

    Compiles each transformer layer individually. Must be applied after
    activation checkpointing and before FSDP.

    Args:
        model: HuggingFace Qwen3 model
        titan_config: Configuration dict

    Returns:
        Model with compiled transformer layers
    """
    compile_mode = titan_config.get("compile_mode", "default")

    torch._dynamo.config.capture_scalar_outputs = True

    for idx, layer in enumerate(model.model.layers):
        model.model.layers[idx] = torch.compile(layer, mode=compile_mode)

    logger.info(
        f"Applied torch.compile (mode={compile_mode}) to "
        f"{len(model.model.layers)} layers"
    )
    return model
