# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Parallelization utilities for HuggingFace Qwen3 models (dense and MoE).

This module provides FSDP2, Expert Parallelism, and activation checkpointing
for HuggingFace Qwen3 models used in the SpeechLM framework. It follows
TorchTitan's parallelization patterns adapted for the HuggingFace model structure.

The module auto-detects MoE models and applies Expert Parallelism when enabled.

HuggingFace Qwen3 model structure:
    model.model.embed_tokens  - Token embeddings
    model.model.layers        - List of transformer layers
    model.model.norm          - Final RMSNorm
    model.lm_head             - Output projection

Additional multimodal components (added by ParallelHFModel):
    model.multimodal_io_dict  - Dict of multimodal IO handlers
    model.adaptor             - Dict of linear adaptors for continuous modalities
    model.stream_emb          - Stream embeddings

MoE-specific features (when MoE is detected):
    - Converts HuggingFace MoE blocks to TorchTitan-compatible format
    - Expert Parallelism using TorchTitan's ExpertParallel
    - Separate FSDP mesh for expert parameters (edp_mesh)
"""

import logging
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.tensor.parallel import parallelize_module
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.expert_parallel import ExpertParallel
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

from espnet2.speechlm.model.speechlm.moe_utils.replace_moe_layer_titan import (
    replace_qwen3_moe_layer_titan,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Expert Parallelism
# =============================================================================


def apply_moe_ep_qwen3(model: nn.Module, parallel_dims: ParallelDims) -> nn.Module:
    """Apply Expert Parallelism to MoE layers.

    Uses TorchTitan's ExpertParallel for DTensor-based expert sharding
    and all-to-all token dispatch/combine.

    Args:
        model: Model with TorchTitan MoE layers
        parallel_dims: ParallelDims with EP mesh

    Returns:
        Model with EP applied to MoE experts
    """
    ep_mesh = parallel_dims.get_optional_mesh("ep")
    if ep_mesh is None:
        logger.warning("EP mesh not available, skipping expert parallelism")
        return model

    # Apply EP to each MoE layer
    for layer in model.model.layers:
        if hasattr(layer, "moe_enabled") and layer.moe_enabled:
            parallelize_module(
                module=layer.mlp.experts,
                device_mesh=ep_mesh,
                parallelize_plan=ExpertParallel(),
            )

    logger.info(f"Applied Expert Parallelism (EP={parallel_dims.ep})")
    return model


# =============================================================================
# Main Entry Point
# =============================================================================


def parallelize_qwen3_hf(
    model: nn.Module,
    parallel_dims: ParallelDims,
    titan_config: Dict[str, Any],
) -> nn.Module:
    """Apply parallelization to HuggingFace Qwen3 model (dense or MoE).

    This is the main entry point for parallelizing HuggingFace Qwen3 models.
    Supports FSDP/HSDP, activation checkpointing, and Expert Parallelism for MoE.
    Automatically detects MoE models and applies appropriate parallelization.

    Args:
        model: HuggingFace Qwen3 model (possibly wrapped with multimodal components)
        parallel_dims: TorchTitan ParallelDims object with device meshes
        titan_config: Configuration dict containing:
            - mixed_precision_param: Parameter dtype (default: "bfloat16")
            - mixed_precision_reduce: Reduce dtype (default: "float32")
            - activation_checkpoint: Whether to enable activation checkpointing
            - compile: Whether to enable torch.compile (default: False)
            - compile_mode: Compile mode - "default", "reduce-overhead", or
              "max-autotune" (default: "default")
            - reshard_after_forward: FSDP reshard policy (default: "default")
              "always" = reshard after forward (saves memory)
              "never" = keep params for backward (faster, more memory)
              "default" = auto-decide based on pipeline parallelism

    Returns:
        Parallelized model
    """
    # Check if MoE with EP enabled
    is_moe = isinstance(model, Qwen3MoeForCausalLM) and parallel_dims.ep_enabled

    # Apply torch.compile first (before activation checkpointing and FSDP)
    if titan_config.get("compile", False):
        model = apply_torch_compile_qwen3(model, titan_config)

    # Apply activation checkpointing (before FSDP)
    if titan_config.get("activation_checkpoint", False):
        model = apply_activation_checkpoint_qwen3(model)

    # Apply Expert Parallelism if MoE and EP enabled
    if is_moe:
        model = replace_qwen3_moe_layer_titan(model)
        model.is_moe = True
        model = apply_moe_ep_qwen3(model, parallel_dims)
        logger.info("Applied MoE Expert Parallelism")

    # Apply FSDP2 wrapping
    if parallel_dims.fsdp_enabled or is_moe:
        model = apply_fsdp_qwen3(model, parallel_dims, titan_config, is_moe)

    return model


def apply_fsdp_qwen3(
    model: nn.Module,
    parallel_dims: ParallelDims,
    titan_config: Dict[str, Any],
    is_moe: bool = False,
) -> nn.Module:
    """Apply FSDP2 to HuggingFace Qwen3 model structure.

    Wraps all model components with FSDP for full sharding:
    1. model.embed_tokens - vocabulary embedding
    2. model.layers - each transformer layer individually
       (for MoE: experts sharded on edp_mesh first)
    3. model.norm - final RMSNorm
    4. lm_head - output projection
    5. multimodal_io_dict - multimodal IO handlers (each module in dict)
    6. adaptor - linear adaptors (each module in dict)
    7. stream_emb - stream embeddings
    8. Top-level model wrap

    Args:
        model: HuggingFace Qwen3 model to wrap with FSDP
        parallel_dims: TorchTitan ParallelDims with device meshes
        titan_config: Configuration dict
        is_moe: Whether model has MoE layers (for expert sharding)

    Returns:
        FSDP-wrapped model
    """
    # Setup mixed precision policy
    param_dtype = getattr(
        torch, titan_config.get("mixed_precision_param", "bfloat16")
    )
    reduce_dtype = getattr(
        torch, titan_config.get("mixed_precision_reduce", "float32")
    )
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype, reduce_dtype=reduce_dtype
    )

    # Get FSDP mesh
    if parallel_dims.dp_replicate_enabled:
        dp_mesh = parallel_dims.get_mesh(["dp_replicate", "fsdp"])
    else:
        dp_mesh = parallel_dims.get_mesh("fsdp")

    # Reshard after forward policy:
    # - "always": Always reshard params after forward (saves memory)
    # - "never": Never reshard (faster, more memory)
    # - "default": Auto-decide based on pipeline parallelism
    reshard_policy = titan_config.get("reshard_after_forward", "default")
    if reshard_policy == "always":
        reshard_after_forward = True
    elif reshard_policy == "never":
        reshard_after_forward = False
    elif reshard_policy == "default":
        reshard_after_forward = not parallel_dims.pp_enabled
    else:
        raise ValueError(
            f"Invalid reshard_after_forward: {reshard_policy}. "
            f"Must be 'always', 'never', or 'default'."
        )

    fsdp_config = {
        "mesh": dp_mesh,
        "mp_policy": mp_policy,
        "reshard_after_forward": reshard_after_forward,
    }

    # Get FSDP mesh for MoE experts (efsdp dimension)
    if is_moe and parallel_dims.ep_enabled:
        if parallel_dims.dp_replicate_enabled:
            edp_mesh = parallel_dims.get_optional_mesh(["dp_replicate", "efsdp"])
        else:
            edp_mesh = parallel_dims.get_optional_mesh("efsdp")
        moe_fsdp_config = {
            "mesh": edp_mesh if edp_mesh is not None else dp_mesh,
            "mp_policy": mp_policy,
            "reshard_after_forward": reshard_after_forward,
        }
    else:
        edp_mesh = None
        moe_fsdp_config = None

    # 1. Shard input embeddings
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        fully_shard(model.model.embed_tokens, **fsdp_config)
        logger.info("FSDP wrapped: model.embed_tokens")

    # 2. Shard each transformer layer individually
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            # For MoE layers, first shard experts on edp_mesh
            if is_moe and hasattr(layer, "moe_enabled") and layer.moe_enabled:
                if hasattr(layer.mlp, "experts") and moe_fsdp_config is not None:
                    fully_shard(layer.mlp.experts, **moe_fsdp_config)
            fully_shard(layer, **fsdp_config)
        logger.info(f"FSDP wrapped: {len(model.model.layers)} transformer layers")

    # 3. Shard final norm, lm_head, and stream_emb together
    # Optimization: Don't reshard after forward since they're needed immediately
    # after the last transformer layer (FSDP would prefetch them anyway)
    last_layer_config = fsdp_config.copy()
    last_layer_config["reshard_after_forward"] = (reshard_policy == "always")

    # 4. Collect last layer modules
    last_layer_modules = []
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        last_layer_modules.append(model.model.norm)
    if hasattr(model, "lm_head"):
        last_layer_modules.append(model.lm_head)
    if hasattr(model, "stream_emb") and next(model.stream_emb.parameters(), None) is not None:
        last_layer_modules.append(model.stream_emb)

    fully_shard(last_layer_modules, **last_layer_config)
    module_names = ["model.norm", "lm_head", "stream_emb"][:len(last_layer_modules)]
    logger.info(f"FSDP wrapped: {' + '.join(module_names)} (no reshard after forward)")

    # 5. Shard multimodal_io_dict (each module in the dict)
    if hasattr(model, "multimodal_io_dict") and isinstance(model.multimodal_io_dict, dict):
        for key, io_module in model.multimodal_io_dict.items():
            if isinstance(io_module, nn.Module) and next(io_module.parameters(), None) is not None:
                fully_shard(io_module, **fsdp_config)
                logger.info(f"FSDP wrapped: multimodal_io_dict[{key}]")

    # 6. Shard adaptor (each module in the dict)
    if hasattr(model, "adaptor") and isinstance(model.adaptor, dict):
        for key, adaptor_module in model.adaptor.items():
            if isinstance(adaptor_module, nn.Module) and next(adaptor_module.parameters(), None) is not None:
                fully_shard(adaptor_module, **fsdp_config)
                logger.info(f"FSDP wrapped: adaptor[{key}]")

    # 7. Top-level FSDP wrap to make all remaining params DTensors
    fully_shard(model, **fsdp_config)

    # 9. Set up explicit prefetching to overlap communication with compute
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        _setup_fsdp_prefetching(model, is_moe)

    mode = "HSDP" if parallel_dims.dp_replicate_enabled else "FSDP2"
    if is_moe:
        mode += " with EP"
    logger.info(f"Applied {mode} to Qwen3 model")

    return model


def _setup_fsdp_prefetching(model: nn.Module, is_moe: bool = False) -> None:
    """Set up explicit FSDP prefetching to overlap communication with compute.

    Configures each layer to prefetch the next layer's parameters during forward
    and the previous layer's parameters during backward. This hides FSDP
    all-gather latency behind compute.

    For MoE models, also prefetches expert parameters.

    Args:
        model: FSDP-wrapped model with HuggingFace structure
        is_moe: Whether model has MoE layers (for expert prefetching)
    """
    layers = list(model.model.layers)
    num_layers = len(layers)

    if num_layers == 0:
        return

    def _get_prefetch_modules(layer, include_moe: bool = False):
        """Get modules to prefetch for a layer."""
        modules = [layer]
        if include_moe and hasattr(layer, "moe_enabled") and layer.moe_enabled:
            moe = layer.mlp if hasattr(layer, "mlp") else layer.moe
            if hasattr(moe, "experts"):
                modules.append(moe.experts)
        return modules

    # Forward prefetching: each layer prefetches the next
    # embed_tokens -> first layer
    if hasattr(model.model, "embed_tokens"):
        if hasattr(model.model.embed_tokens, "set_modules_to_forward_prefetch"):
            model.model.embed_tokens.set_modules_to_forward_prefetch(
                _get_prefetch_modules(layers[0], is_moe)
            )

    # layer[i] -> layer[i+1]
    for i in range(num_layers - 1):
        if hasattr(layers[i], "set_modules_to_forward_prefetch"):
            layers[i].set_modules_to_forward_prefetch(
                _get_prefetch_modules(layers[i + 1], is_moe)
            )

    # last layer -> norm + lm_head + stream_emb
    last_modules = []
    if hasattr(model.model, "norm"):
        last_modules.append(model.model.norm)
    if hasattr(model, "lm_head"):
        last_modules.append(model.lm_head)
    if hasattr(model, "stream_emb"):
        last_modules.append(model.stream_emb)
    if last_modules and hasattr(layers[-1], "set_modules_to_forward_prefetch"):
        layers[-1].set_modules_to_forward_prefetch(last_modules)

    # Backward prefetching: each layer prefetches the previous
    # lm_head -> last layer
    if hasattr(model, "lm_head"):
        if hasattr(model.lm_head, "set_modules_to_backward_prefetch"):
            model.lm_head.set_modules_to_backward_prefetch(
                _get_prefetch_modules(layers[-1], is_moe)
            )

    # layer[i] -> layer[i-1]
    for i in range(num_layers - 1, 0, -1):
        if hasattr(layers[i], "set_modules_to_backward_prefetch"):
            layers[i].set_modules_to_backward_prefetch(
                _get_prefetch_modules(layers[i - 1], is_moe)
            )

    # first layer -> embed_tokens
    if hasattr(model.model, "embed_tokens"):
        if hasattr(layers[0], "set_modules_to_backward_prefetch"):
            layers[0].set_modules_to_backward_prefetch([model.model.embed_tokens])

    logger.info("Set up FSDP forward/backward prefetching")


def apply_torch_compile_qwen3(
    model: nn.Module,
    titan_config: Dict[str, Any],
) -> nn.Module:
    """Apply torch.compile to transformer layers for performance optimization.

    Compiles each transformer layer individually using torch.compile.
    This is applied before activation checkpointing and FSDP to avoid
    compatibility issues with wrapped layers.

    Args:
        model: HuggingFace Qwen3 model (already FSDP wrapped)
        titan_config: Configuration dict containing:
            - compile_mode: Compile mode (default: "default")
                Options: "default", "reduce-overhead", "max-autotune"
            - compile_fullgraph: Whether to require full graph compilation
                (default: False, allows graph breaks for compatibility)
            - compile_backend: Backend to use (default: "inductor")

    Returns:
        Model with compiled transformer layers
    """
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        logger.warning(
            "Model does not have expected structure (model.model.layers). "
            "Skipping torch.compile."
        )
        return model

    compile_mode = titan_config.get("compile_mode", "default")
    compile_fullgraph = titan_config.get("compile_fullgraph", False)
    compile_backend = titan_config.get("compile_backend", "inductor")

    for idx, layer in enumerate(model.model.layers):
        model.model.layers[idx] = torch.compile(
            layer,
            mode=compile_mode,
            fullgraph=compile_fullgraph,
            backend=compile_backend,
        )

    logger.info(
        f"Applied torch.compile (mode={compile_mode}, fullgraph={compile_fullgraph}, "
        f"backend={compile_backend}) to {len(model.model.layers)} transformer layers"
    )
    return model


def apply_activation_checkpoint_qwen3(model: nn.Module) -> nn.Module:
    """Apply activation checkpointing to HuggingFace Qwen3 model.

    Activation checkpointing trades compute for memory by recomputing
    activations during the backward pass instead of storing them.
    Each transformer layer is wrapped with checkpoint_wrapper.

    Note: With Flash Attention, the attention memory is already optimized,
    so we checkpoint entire layers rather than selective components.

    Args:
        model: HuggingFace Qwen3 model

    Returns:
        Model with activation checkpointing applied
    """
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        logger.warning(
            "Model does not have expected structure (model.model.layers). "
            "Skipping activation checkpointing."
        )
        return model

    for idx, layer in enumerate(model.model.layers):
        model.model.layers[idx] = checkpoint_wrapper(
            layer,
            checkpoint_impl="no_reentrant",
        )

    logger.info(f"Applied activation checkpointing to {len(model.model.layers)} layers")
    return model
