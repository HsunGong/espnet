# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""MoE utilities for Expert Parallelism support.

This package provides Expert Parallelism wrappers for different training backends:
- DeepSpeed EP: replace_moe_layer.py
- TorchTitan EP: replace_moe_layer_titan.py

Import modules directly to avoid dependency conflicts:
    from espnet2.speechlm.model.speechlm.moe_utils.replace_moe_layer import ...
    from espnet2.speechlm.model.speechlm.moe_utils.replace_moe_layer_titan import ...
"""
