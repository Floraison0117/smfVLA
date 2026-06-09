#!/usr/bin/env python3
"""
Convert Pi0.5 PyTorch (safetensors) checkpoint to JAX Orbax format.

This script converts a PyTorch safetensors checkpoint (from HuggingFace format)
to JAX Orbax format compatible with openpi JAX training/evaluation.

Usage:
    python scripts/convert_pytorch_to_jax.py \
        --safetensors_path /path/to/model.safetensors \
        --output_dir /path/to/output/jax_checkpoint \
        --config_name pi05_libero

Example:
    python scripts/convert_pytorch_to_jax.py \
        --safetensors_path checkpoints/pi05_calvin_ABC_D_SFT/model.safetensors \
        --output_dir checkpoints/pi05_calvin_jax \
        --config_name pi05_libero
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from safetensors import safe_open
import torch

# Import openpi modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "openpi" / "src"))

from openpi.models import pi0_config
from openpi.models import model as openpi_model


def load_safetensors(safetensors_path: str) -> Dict[str, np.ndarray]:
    """Load weights from safetensors file.

    Args:
        safetensors_path: Path to the safetensors file.

    Returns:
        Dictionary mapping parameter names to numpy arrays.
    """
    weights = {}
    with safe_open(safetensors_path, framework='pt', device='cpu') as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            # Convert to float32 first if bfloat16, then to numpy
            if tensor.dtype == torch.bfloat16:
                weights[key] = tensor.to(torch.float32).numpy()
            else:
                weights[key] = tensor.numpy()
    return weights


def convert_paligemma_vision_tower(pt_weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Convert PaliGemma vision tower from PyTorch to JAX format.

    PyTorch key pattern:
        paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight
        ...

    JAX key pattern:
        PaliGemma/img/embedding/kernel
        PaliGemma/img/embedding/bias
        ...
    """
    jax_weights = {}

    # Patch embedding
    pt_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight"
    if pt_key in pt_weights:
        # PyTorch: [out_channels, in_channels, h, w] -> JAX: [h, w, in_channels, out_channels]
        jax_weights["PaliGemma/img/embedding/kernel"] = pt_weights[pt_key].transpose(2, 3, 1, 0)

    pt_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.bias"
    if pt_key in pt_weights:
        jax_weights["PaliGemma/img/embedding/bias"] = pt_weights[pt_key]

    # Positional embedding
    pt_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.position_embedding.weight"
    if pt_key in pt_weights:
        jax_weights["PaliGemma/img/pos_embedding"] = pt_weights[pt_key]

    # Encoder layers (27 layers for gemma_2b)
    # Collect all layer parameters
    num_layers = 27
    layer_norm0_scale = []
    layer_norm0_bias = []
    layer_norm1_scale = []
    layer_norm1_bias = []
    mlp_dense0_kernel = []
    mlp_dense0_bias = []
    mlp_dense1_kernel = []
    mlp_dense1_bias = []
    attn_q_kernel = []
    attn_q_bias = []
    attn_k_kernel = []
    attn_k_bias = []
    attn_v_kernel = []
    attn_v_bias = []
    attn_out_kernel = []
    attn_out_bias = []

    for i in range(num_layers):
        # Layer norm 1
        pt_key = f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.weight"
        if pt_key in pt_weights:
            layer_norm0_scale.append(pt_weights[pt_key])

        pt_key = f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.bias"
        if pt_key in pt_weights:
            layer_norm0_bias.append(pt_weights[pt_key])

        # Layer norm 2
        pt_key = f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.weight"
        if pt_key in pt_weights:
            layer_norm1_scale.append(pt_weights[pt_key])

        pt_key = f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.bias"
        if pt_key in pt_weights:
            layer_norm1_bias.append(pt_weights[pt_key])

        # MLP
        pt_key = f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.weight"
        if pt_key in pt_weights:
            mlp_dense0_kernel.append(pt_weights[pt_key].transpose())  # PyTorch: [out, in] -> JAX: [in, out]

        pt_key = f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.bias"
        if pt_key in pt_weights:
            mlp_dense0_bias.append(pt_weights[pt_key])

        pt_key = f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.weight"
        if pt_key in pt_weights:
            mlp_dense1_kernel.append(pt_weights[pt_key].transpose())

        pt_key = f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.bias"
        if pt_key in pt_weights:
            mlp_dense1_bias.append(pt_weights[pt_key])

        # Attention
        pt_key = f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.weight"
        if pt_key in pt_weights:
            attn_q_kernel.append(pt_weights[pt_key])

        pt_key = f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.bias"
        if pt_key in pt_weights:
            attn_q_bias.append(pt_weights[pt_key])

        pt_key = f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.weight"
        if pt_key in pt_weights:
            attn_k_kernel.append(pt_weights[pt_key])

        pt_key = f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.bias"
        if pt_key in pt_weights:
            attn_k_bias.append(pt_weights[pt_key])

        pt_key = f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.weight"
        if pt_key in pt_weights:
            attn_v_kernel.append(pt_weights[pt_key])

        pt_key = f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.bias"
        if pt_key in pt_weights:
            attn_v_bias.append(pt_weights[pt_key])

        pt_key = f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.weight"
        if pt_key in pt_weights:
            attn_out_kernel.append(pt_weights[pt_key])

        pt_key = f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.bias"
        if pt_key in pt_weights:
            attn_out_bias.append(pt_weights[pt_key])

    # Stack layer parameters
    if layer_norm0_scale:
        jax_weights["PaliGemma/img/Transformer/encoderblock/LayerNorm_0/scale"] = np.stack(layer_norm0_scale, axis=0)
    if layer_norm0_bias:
        jax_weights["PaliGemma/img/Transformer/encoderblock/LayerNorm_0/bias"] = np.stack(layer_norm0_bias, axis=0)
    if layer_norm1_scale:
        jax_weights["PaliGemma/img/Transformer/encoderblock/LayerNorm_1/scale"] = np.stack(layer_norm1_scale, axis=0)
    if layer_norm1_bias:
        jax_weights["PaliGemma/img/Transformer/encoderblock/LayerNorm_1/bias"] = np.stack(layer_norm1_bias, axis=0)

    if mlp_dense0_kernel:
        jax_weights["PaliGemma/img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel"] = np.stack(mlp_dense0_kernel, axis=0)
    if mlp_dense0_bias:
        jax_weights["PaliGemma/img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias"] = np.stack(mlp_dense0_bias, axis=0)
    if mlp_dense1_kernel:
        jax_weights["PaliGemma/img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel"] = np.stack(mlp_dense1_kernel, axis=0)
    if mlp_dense1_bias:
        jax_weights["PaliGemma/img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias"] = np.stack(mlp_dense1_bias, axis=0)

    # Reshape attention kernels for JAX format
    if attn_q_kernel:
        # PyTorch: [num_heads * head_dim, hidden] -> JAX: [num_layers, num_heads, head_dim, hidden]
        num_heads = 16
        head_dim = 72
        hidden = 1152
        attn_q_kernel_stacked = []
        for k in attn_q_kernel:
            attn_q_kernel_stacked.append(k.reshape(num_heads, head_dim, hidden).transpose(1, 2, 0))
        jax_weights["PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel"] = np.stack(attn_q_kernel_stacked, axis=0)

    if attn_q_bias:
        attn_q_bias_stacked = []
        for b in attn_q_bias:
            attn_q_bias_stacked.append(b.reshape(num_heads, head_dim))
        jax_weights["PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias"] = np.stack(attn_q_bias_stacked, axis=0)

    if attn_k_kernel:
        attn_k_kernel_stacked = []
        for k in attn_k_kernel:
            attn_k_kernel_stacked.append(k.reshape(num_heads, head_dim, hidden).transpose(1, 2, 0))
        jax_weights["PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel"] = np.stack(attn_k_kernel_stacked, axis=0)

    if attn_k_bias:
        attn_k_bias_stacked = []
        for b in attn_k_bias:
            attn_k_bias_stacked.append(b.reshape(num_heads, head_dim))
        jax_weights["PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias"] = np.stack(attn_k_bias_stacked, axis=0)

    if attn_v_kernel:
        attn_v_kernel_stacked = []
        for k in attn_v_kernel:
            attn_v_kernel_stacked.append(k.reshape(num_heads, head_dim, hidden).transpose(1, 2, 0))
        jax_weights["PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel"] = np.stack(attn_v_kernel_stacked, axis=0)

    if attn_v_bias:
        attn_v_bias_stacked = []
        for b in attn_v_bias:
            attn_v_bias_stacked.append(b.reshape(num_heads, head_dim))
        jax_weights["PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias"] = np.stack(attn_v_bias_stacked, axis=0)

    if attn_out_kernel:
        attn_out_kernel_stacked = []
        for k in attn_out_kernel:
            attn_out_kernel_stacked.append(k.reshape(hidden, num_heads, head_dim).transpose(1, 2, 0))
        jax_weights["PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel"] = np.stack(attn_out_kernel_stacked, axis=0)

    if attn_out_bias:
        jax_weights["PaliGemma/img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias"] = np.stack(attn_out_bias, axis=0)

    # Post layer norm
    pt_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.weight"
    if pt_key in pt_weights:
        jax_weights["PaliGemma/img/Transformer/encoder_norm/scale"] = pt_weights[pt_key]

    pt_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.bias"
    if pt_key in pt_weights:
        jax_weights["PaliGemma/img/Transformer/encoder_norm/bias"] = pt_weights[pt_key]

    # Multi-modal projector
    pt_key = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight"
    if pt_key in pt_weights:
        jax_weights["PaliGemma/img/head/kernel"] = pt_weights[pt_key].transpose()

    pt_key = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias"
    if pt_key in pt_weights:
        jax_weights["PaliGemma/img/head/bias"] = pt_weights[pt_key]

    return jax_weights


def convert_paligemma_language_model(pt_weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Convert PaliGemma language model (Gemma-2B) from PyTorch to JAX format.

    PyTorch key pattern:
        paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.q_proj.weight
        ...

    JAX key pattern:
        PaliGemma/llm/layers/attn/q_einsum/w
        ...
    """
    jax_weights = {}

    # Input embeddings
    pt_key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    if pt_key in pt_weights:
        jax_weights["PaliGemma/llm/embedder/input_embedding"] = pt_weights[pt_key]

    # Language model layers (18 layers for gemma_2b)
    num_layers = 18
    attn_attn_vec_einsum = []
    attn_kv_einsum = []
    attn_q_einsum = []
    mlp_gating_einsum = []
    mlp_linear = []
    input_layernorm = []
    post_attention_layernorm = []

    for i in range(num_layers):
        # Attention
        pt_key = f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.q_proj.weight"
        if pt_key in pt_weights:
            # PyTorch: [num_heads * head_dim, hidden] -> JAX: [num_heads, head_dim, hidden]
            q_weight = pt_weights[pt_key]
            num_heads = 8
            head_dim = 256
            hidden = 2048
            attn_q_einsum.append(q_weight.reshape(num_heads, head_dim, hidden))

        pt_key = f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.k_proj.weight"
        if pt_key in pt_weights:
            attn_kv_einsum.append([pt_weights[pt_key], np.zeros_like(pt_weights[pt_key])])

        pt_key = f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.v_proj.weight"
        if pt_key in pt_weights:
            # Add to existing kv_einsum entry
            attn_kv_einsum[-1][1] = pt_weights[pt_key]

        pt_key = f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.o_proj.weight"
        if pt_key in pt_weights:
            # PyTorch: [hidden, num_heads * head_dim] -> JAX: [num_heads, head_dim, hidden]
            o_weight = pt_weights[pt_key]
            attn_attn_vec_einsum.append(o_weight.reshape(num_heads, head_dim, hidden).transpose(1, 2, 0))

        # MLP
        pt_key = f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.gate_proj.weight"
        if pt_key in pt_weights:
            mlp_gating_einsum.append([pt_weights[pt_key], np.zeros_like(pt_weights[pt_key])])

        pt_key = f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.up_proj.weight"
        if pt_key in pt_weights:
            mlp_gating_einsum[-1][1] = pt_weights[pt_key]

        pt_key = f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.down_proj.weight"
        if pt_key in pt_weights:
            mlp_linear.append(pt_weights[pt_key])

        # Layer norms
        pt_key = f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.input_layernorm.weight"
        if pt_key in pt_weights:
            input_layernorm.append(pt_weights[pt_key])

        pt_key = f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.post_attention_layernorm.weight"
        if pt_key in pt_weights:
            post_attention_layernorm.append(pt_weights[pt_key])

    # Stack and assign
    if attn_q_einsum:
        jax_weights["PaliGemma/llm/layers/attn/q_einsum/w"] = np.stack(attn_q_einsum, axis=0)

    if attn_kv_einsum:
        jax_weights["PaliGemma/llm/layers/attn/kv_einsum/w"] = np.stack(attn_kv_einsum, axis=0)

    if attn_attn_vec_einsum:
        jax_weights["PaliGemma/llm/layers/attn/attn_vec_einsum/w"] = np.stack(attn_attn_vec_einsum, axis=0)

    if mlp_gating_einsum:
        jax_weights["PaliGemma/llm/layers/mlp/gating_einsum"] = np.stack(mlp_gating_einsum, axis=0)

    if mlp_linear:
        jax_weights["PaliGemma/llm/layers/mlp/linear"] = np.stack(mlp_linear, axis=0)

    if input_layernorm:
        jax_weights["PaliGemma/llm/layers/pre_attention_norm/scale"] = np.stack(input_layernorm, axis=0)

    if post_attention_layernorm:
        jax_weights["PaliGemma/llm/layers/pre_ffw_norm/scale"] = np.stack(post_attention_layernorm, axis=0)

    # Final norm
    pt_key = "paligemma_with_expert.paligemma.model.language_model.norm.weight"
    if pt_key in pt_weights:
        jax_weights["PaliGemma/llm/final_norm/scale"] = pt_weights[pt_key]

    return jax_weights


def convert_gemma_expert(pt_weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Convert Gemma expert (action expert) from PyTorch to JAX format.

    PyTorch key pattern:
        paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.q_proj.weight
        ...

    JAX key pattern:
        PaliGemma/llm/layers/attn_1/q_einsum/w
        ...
    """
    jax_weights = {}

    # Action expert layers (18 layers for gemma_300m)
    # Dimensions based on gemma_300m: hidden=1024, num_heads=8, head_dim=256
    num_layers = 18
    num_heads = 8  # q_proj is 2048 = 8 * 256
    head_dim = 256
    hidden = 1024

    attn_attn_vec_einsum = []
    attn_kv_einsum = []
    attn_q_einsum = []
    mlp_gating_einsum = []
    mlp_linear = []
    input_layernorm_bias = []
    input_layernorm_kernel = []
    post_attention_layernorm_bias = []
    post_attention_layernorm_kernel = []

    for i in range(num_layers):
        # Attention
        pt_key = f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.q_proj.weight"
        if pt_key in pt_weights:
            q_weight = pt_weights[pt_key]
            attn_q_einsum.append(q_weight.reshape(num_heads, head_dim, hidden))

        pt_key = f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.k_proj.weight"
        if pt_key in pt_weights:
            attn_kv_einsum.append([pt_weights[pt_key], np.zeros_like(pt_weights[pt_key])])

        pt_key = f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.v_proj.weight"
        if pt_key in pt_weights:
            attn_kv_einsum[-1][1] = pt_weights[pt_key]

        pt_key = f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.o_proj.weight"
        if pt_key in pt_weights:
            o_weight = pt_weights[pt_key]
            attn_attn_vec_einsum.append(o_weight.reshape(hidden, num_heads, head_dim).transpose(1, 2, 0))

        # MLP
        pt_key = f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.gate_proj.weight"
        if pt_key in pt_weights:
            mlp_gating_einsum.append([pt_weights[pt_key], np.zeros_like(pt_weights[pt_key])])

        pt_key = f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.up_proj.weight"
        if pt_key in pt_weights:
            mlp_gating_einsum[-1][1] = pt_weights[pt_key]

        pt_key = f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.down_proj.weight"
        if pt_key in pt_weights:
            mlp_linear.append(pt_weights[pt_key])

        # Layer norms (Dense layer for pi05 adaptive normalization)
        pt_key = f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.dense.bias"
        if pt_key in pt_weights:
            input_layernorm_bias.append(pt_weights[pt_key])

        pt_key = f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.dense.weight"
        if pt_key in pt_weights:
            input_layernorm_kernel.append(pt_weights[pt_key].transpose())

        pt_key = f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.dense.bias"
        if pt_key in pt_weights:
            post_attention_layernorm_bias.append(pt_weights[pt_key])

        pt_key = f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.dense.weight"
        if pt_key in pt_weights:
            post_attention_layernorm_kernel.append(pt_weights[pt_key].transpose())

    # Stack and assign
    if attn_q_einsum:
        jax_weights["PaliGemma/llm/layers/attn_1/q_einsum/w"] = np.stack(attn_q_einsum, axis=0)

    if attn_kv_einsum:
        jax_weights["PaliGemma/llm/layers/attn_1/kv_einsum/w"] = np.stack(attn_kv_einsum, axis=0)

    if attn_attn_vec_einsum:
        jax_weights["PaliGemma/llm/layers/attn_1/attn_vec_einsum/w"] = np.stack(attn_attn_vec_einsum, axis=0)

    if mlp_gating_einsum:
        jax_weights["PaliGemma/llm/layers/mlp_1/gating_einsum"] = np.stack(mlp_gating_einsum, axis=0)

    if mlp_linear:
        jax_weights["PaliGemma/llm/layers/mlp_1/linear"] = np.stack(mlp_linear, axis=0)

    if input_layernorm_bias:
        jax_weights["PaliGemma/llm/layers/pre_attention_norm_1/Dense_0/bias"] = np.stack(input_layernorm_bias, axis=0)

    if input_layernorm_kernel:
        jax_weights["PaliGemma/llm/layers/pre_attention_norm_1/Dense_0/kernel"] = np.stack(input_layernorm_kernel, axis=0)

    if post_attention_layernorm_bias:
        jax_weights["PaliGemma/llm/layers/pre_ffw_norm_1/Dense_0/bias"] = np.stack(post_attention_layernorm_bias, axis=0)

    if post_attention_layernorm_kernel:
        jax_weights["PaliGemma/llm/layers/pre_ffw_norm_1/Dense_0/kernel"] = np.stack(post_attention_layernorm_kernel, axis=0)

    # Final norm
    pt_key = "paligemma_with_expert.gemma_expert.model.norm.dense.bias"
    if pt_key in pt_weights:
        jax_weights["PaliGemma/llm/final_norm_1/Dense_0/bias"] = pt_weights[pt_key]

    pt_key = "paligemma_with_expert.gemma_expert.model.norm.dense.weight"
    if pt_key in pt_weights:
        jax_weights["PaliGemma/llm/final_norm_1/Dense_0/kernel"] = pt_weights[pt_key].transpose()

    # lm_head (tied with embeddings, but stored separately in some checkpoints)
    pt_key = "paligemma_with_expert.gemma_expert.lm_head.weight"
    if pt_key in pt_weights:
        jax_weights["PaliGemma/llm/final_norm_1/embedding"] = pt_weights[pt_key]

    return jax_weights


def convert_projection_params(pt_weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Convert projection parameters from PyTorch to JAX format.

    PyTorch keys:
        action_in_proj.weight, action_in_proj.bias
        action_out_proj.weight, action_out_proj.bias
        time_mlp_in.weight, time_mlp_in.bias
        time_mlp_out.weight, time_mlp_out.bias

    JAX keys:
        projection_params/action_in_proj/kernel
        projection_params/action_in_proj/bias
        ...
    """
    jax_weights = {}

    projection_keys = ["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]

    for key in projection_keys:
        pt_weight_key = f"{key}.weight"
        pt_bias_key = f"{key}.bias"

        if pt_weight_key in pt_weights:
            # PyTorch: [out, in] -> JAX: [in, out]
            jax_weights[f"projection_params/{key}/kernel"] = pt_weights[pt_weight_key].transpose()

        if pt_bias_key in pt_weights:
            jax_weights[f"projection_params/{key}/bias"] = pt_weights[pt_bias_key]

    return jax_weights


def convert_pytorch_to_jax(
    safetensors_path: str,
    output_dir: str,
    config_name: str = "pi05_libero"
) -> None:
    """Convert PyTorch safetensors checkpoint to JAX Orbax format.

    Args:
        safetensors_path: Path to the PyTorch safetensors file.
        output_dir: Directory to save the JAX checkpoint.
        config_name: Name of the config to use for reference.
    """
    print(f"Loading PyTorch checkpoint from {safetensors_path}...")
    pt_weights = load_safetensors(safetensors_path)
    print(f"Loaded {len(pt_weights)} parameters")

    print("\nConverting to JAX format...")

    # Convert different components
    jax_weights = {}

    print("Converting PaliGemma vision tower...")
    jax_weights.update(convert_paligemma_vision_tower(pt_weights))

    print("Converting PaliGemma language model...")
    jax_weights.update(convert_paligemma_language_model(pt_weights))

    print("Converting Gemma expert...")
    jax_weights.update(convert_gemma_expert(pt_weights))

    print("Converting projection parameters...")
    jax_weights.update(convert_projection_params(pt_weights))

    print(f"Converted {len(jax_weights)} JAX parameters")

    # Create the PyTree structure expected by Orbax
    # The structure should be: {"params": {"PaliGemma": {...}, "projection_params": {...}}}

    # First, create a nested dict from the flattened keys
    nested_params = {}
    for key, value in jax_weights.items():
        parts = key.split("/")
        current = nested_params
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = {"value": value}  # Add "value" suffix for NNX State

    # Structure the params properly
    params = {
        "params": {
            "PaliGemma": nested_params.get("PaliGemma", {}),
            "projection_params": nested_params.get("projection_params", {}),
        }
    }

    # Save with Orbax
    print(f"\nSaving JAX checkpoint to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    params_path = os.path.join(output_dir, "params")

    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(params_path, params)

    print(f"Checkpoint saved successfully to {params_path}")

    # Save metadata
    metadata = {
        "config_name": config_name,
        "conversion": "pytorch_to_jax",
        "num_params": len(jax_weights),
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Pi0.5 PyTorch (safetensors) checkpoint to JAX Orbax format"
    )
    parser.add_argument(
        "--safetensors_path",
        type=str,
        required=True,
        help="Path to the PyTorch safetensors file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the JAX checkpoint"
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="pi05_libero",
        help="Name of the config to use for reference (default: pi05_libero)"
    )

    args = parser.parse_args()

    convert_pytorch_to_jax(
        safetensors_path=args.safetensors_path,
        output_dir=args.output_dir,
        config_name=args.config_name
    )


if __name__ == "__main__":
    main()
