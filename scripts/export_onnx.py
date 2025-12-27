#!/usr/bin/env python3
"""
OssammaNER ONNX Export Script

Converts trained Julia/Lux OssammaNER models to ONNX format for deployment.

This script:
1. Defines the PyTorch equivalent of the OssammaNER architecture
2. Loads weights from Julia .jls checkpoint
3. Exports to ONNX format

Usage:
    python scripts/export_onnx.py checkpoints/ner_110m/checkpoint_best.jls models/ossamma_ner.onnx

Note: CRF layer is NOT included in ONNX export. Use emissions for Viterbi decoding separately.
"""

import argparse
import json
import struct
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =============================================================================
# Julia Checkpoint Loader
# =============================================================================

def load_julia_checkpoint(path: str) -> dict:
    """
    Load a Julia Serialization checkpoint.

    Julia's Serialization.jl format is complex. We use a simplified loader
    that extracts the key-value structure for NamedTuples.
    """
    import subprocess
    import tempfile
    import os

    # Use Julia to convert checkpoint to JSON-compatible format
    julia_script = f'''
    using Serialization
    using JSON3

    function convert_to_dict(nt::NamedTuple)
        Dict(k => convert_to_dict(v) for (k, v) in pairs(nt))
    end
    function convert_to_dict(arr::AbstractArray)
        if eltype(arr) <: Number
            return Array(arr)  # Convert CuArray to Array
        else
            return [convert_to_dict(x) for x in arr]
        end
    end
    convert_to_dict(x) = x

    function extract_params(checkpoint)
        if haskey(checkpoint, :params)
            return convert_to_dict(checkpoint.params)
        elseif haskey(checkpoint, :model_params)
            return convert_to_dict(checkpoint.model_params)
        else
            error("Unknown checkpoint format")
        end
    end

    checkpoint = deserialize("{path}")
    params = extract_params(checkpoint)

    # Also get config if available
    config = haskey(checkpoint, :config) ? checkpoint.config : nothing

    # Save as numpy arrays
    using NPZ

    function flatten_params(params, prefix="")
        result = Dict{{String, Any}}()
        for (k, v) in params
            key = prefix == "" ? string(k) : prefix * "." * string(k)
            if v isa Dict || v isa NamedTuple
                merge!(result, flatten_params(v isa NamedTuple ? Dict(pairs(v)) : v, key))
            elseif v isa AbstractArray
                result[key] = Array(v)
            else
                result[key] = v
            end
        end
        return result
    end

    flat_params = flatten_params(params)
    NPZ.npzwrite("{path}.npz", flat_params)

    # Save config as JSON
    if config !== nothing
        open("{path}.config.json", "w") do f
            JSON3.pretty(f, config)
        end
    end

    println("Converted checkpoint to NPZ format")
    '''

    # Check if NPZ already exists
    npz_path = f"{path}.npz"
    if not os.path.exists(npz_path):
        print(f"Converting Julia checkpoint to NPZ format...")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jl', delete=False) as f:
            f.write(julia_script)
            script_path = f.name

        try:
            result = subprocess.run(
                ['julia', '--project=/root/Ossamma', script_path],
                capture_output=True,
                text=True,
                cwd='/root/Ossamma'
            )
            if result.returncode != 0:
                print(f"Julia error: {result.stderr}")
                raise RuntimeError(f"Failed to convert checkpoint: {result.stderr}")
            print(result.stdout)
        finally:
            os.unlink(script_path)

    # Load NPZ file
    print(f"Loading parameters from {npz_path}")
    params = dict(np.load(npz_path, allow_pickle=True))

    # Load config if available
    config_path = f"{path}.config.json"
    config = None
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    return {'params': params, 'config': config}


# =============================================================================
# Model Components
# =============================================================================

class DLinOSS(nn.Module):
    """Damped Linear Oscillator State Space Model (DLinOSS)"""

    def __init__(self, input_dim: int, state_dim: int, output_dim: int,
                 min_freq: float = 0.1, max_freq: float = 10.0,
                 default_dt: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim

        # Learnable parameters (initialized from Julia)
        self.log_time_step = nn.Parameter(torch.zeros(state_dim))
        self.log_stiffness = nn.Parameter(torch.linspace(
            math.log(min_freq), math.log(max_freq), state_dim
        ))
        self.log_damping = nn.Parameter(torch.full((state_dim,), math.log(0.01)))

        self.input_proj = nn.Linear(input_dim, state_dim, bias=False)
        self.output_proj = nn.Linear(state_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (embedding_dim, seq_len, batch) or (embedding_dim, seq_len)
        Returns:
            output: same shape as input
        """
        # Handle batch dimension
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # Add batch dim
            squeeze_output = True
        else:
            squeeze_output = False

        # x: (D, T, B)
        D, T, B = x.shape

        # Compute physics parameters
        dt = torch.exp(self.log_time_step)  # (N,)
        k = torch.exp(self.log_stiffness)   # (N,)
        c = torch.exp(self.log_damping)     # (N,)

        # Physics operators
        implicit_factor = 1.0 / (1.0 + dt * c)
        vel_retain = implicit_factor
        spring_coupling = -dt * k * implicit_factor
        input_gain = dt * implicit_factor

        # Project input: (N, T, B)
        # Transpose to (B, T, D) for linear, then back
        x_t = x.permute(2, 1, 0)  # (B, T, D)
        proj_input = self.input_proj(x_t)  # (B, T, N)
        proj_input = proj_input.permute(2, 1, 0)  # (N, T, B)

        # Initialize state: velocity and position
        velocity = torch.zeros(self.state_dim, B, device=x.device, dtype=x.dtype)
        position = torch.zeros(self.state_dim, B, device=x.device, dtype=x.dtype)

        # Collect outputs
        outputs = []

        # Sequential scan through time
        for t in range(T):
            u_t = proj_input[:, t, :]  # (N, B)

            # Physics update
            velocity = (vel_retain.unsqueeze(-1) * velocity +
                       spring_coupling.unsqueeze(-1) * position +
                       input_gain.unsqueeze(-1) * u_t)
            position = position + dt.unsqueeze(-1) * velocity

            outputs.append(position)

        # Stack outputs: (N, T, B)
        state_seq = torch.stack(outputs, dim=1)

        # Output projection: (B, T, N) -> (B, T, D_out)
        state_seq_t = state_seq.permute(2, 1, 0)  # (B, T, N)
        output = self.output_proj(state_seq_t)    # (B, T, D_out)
        output = output.permute(2, 1, 0)          # (D_out, T, B)

        if squeeze_output:
            output = output.squeeze(-1)

        return output


class LinearAttentionLayer(nn.Module):
    """Linear Attention with position embeddings and feature maps"""

    def __init__(self, embedding_dim: int, seq_len: int, num_heads: int, time_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)

        # Feature maps
        self.query_feature = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim * 2),
            nn.GELU(),
            nn.Linear(self.head_dim * 2, self.head_dim)
        )
        self.key_feature = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim * 2),
            nn.GELU(),
            nn.Linear(self.head_dim * 2, self.head_dim)
        )

        # Position embeddings
        self.pos_embed_cos = nn.Embedding(seq_len, self.head_dim)
        self.pos_embed_sin = nn.Embedding(seq_len, self.head_dim)

        # Time projection
        self.time_proj = nn.Linear(time_dim, self.head_dim)

        # Feature normalization
        self.feature_norm = nn.LayerNorm(self.head_dim)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (D, T, B) embedding tensor
            time_emb: (time_dim, B) time embedding
        Returns:
            output: (D, T, B)
        """
        D, T, B = x.shape

        # Transpose for linear layers: (B, T, D)
        x_t = x.permute(2, 1, 0)

        # Project Q, K, V
        q = self.query_proj(x_t)  # (B, T, D)
        k = self.key_proj(x_t)
        v = self.value_proj(x_t)

        # Reshape to heads: (B, T, H, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)

        # Time embedding: (B, head_dim)
        time_proj = self.time_proj(time_emb.permute(1, 0))  # (B, head_dim)
        time_proj = time_proj.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, head_dim)

        # Add time to Q and K
        q = q + time_proj
        k = k + time_proj

        # Apply feature maps
        q_feat = self.query_feature(q)
        k_feat = self.key_feature(k)

        # Position embeddings
        pos_idx = torch.arange(T, device=x.device)
        pos_cos = self.pos_embed_cos(pos_idx)  # (T, head_dim)
        pos_sin = self.pos_embed_sin(pos_idx)

        # Apply position (softplus for positivity)
        q_cos = F.softplus(q_feat * pos_cos.unsqueeze(0).unsqueeze(2))
        k_cos = F.softplus(k_feat * pos_cos.unsqueeze(0).unsqueeze(2))
        q_sin = F.softplus(q_feat * pos_sin.unsqueeze(0).unsqueeze(2))
        k_sin = F.softplus(k_feat * pos_sin.unsqueeze(0).unsqueeze(2))

        # Normalize features
        q_cos = self.feature_norm(q_cos)
        k_cos = self.feature_norm(k_cos)
        q_sin = self.feature_norm(q_sin)
        k_sin = self.feature_norm(k_sin)

        # Linear attention: K^T @ V then @ Q
        # Reshape: (B*H, T, head_dim)
        q_cos = q_cos.permute(0, 2, 1, 3).reshape(B * self.num_heads, T, self.head_dim)
        k_cos = k_cos.permute(0, 2, 1, 3).reshape(B * self.num_heads, T, self.head_dim)
        q_sin = q_sin.permute(0, 2, 1, 3).reshape(B * self.num_heads, T, self.head_dim)
        k_sin = k_sin.permute(0, 2, 1, 3).reshape(B * self.num_heads, T, self.head_dim)
        v = v.permute(0, 2, 1, 3).reshape(B * self.num_heads, T, self.head_dim)

        # Compute linear attention
        context_cos = torch.bmm(v.transpose(1, 2), k_cos)  # (B*H, head_dim, head_dim)
        out_cos = torch.bmm(context_cos, q_cos.transpose(1, 2)).transpose(1, 2)

        context_sin = torch.bmm(v.transpose(1, 2), k_sin)
        out_sin = torch.bmm(context_sin, q_sin.transpose(1, 2)).transpose(1, 2)

        # Combine streams
        out = out_cos + out_sin  # (B*H, T, head_dim)

        # Reshape back: (B, T, D)
        out = out.view(B, self.num_heads, T, self.head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, T, D)

        # Output projection
        out = self.output_proj(out)

        # Transpose back: (D, T, B)
        return out.permute(2, 1, 0)


class SWAttention(nn.Module):
    """Sliding Window Attention with SigSoftmax"""

    def __init__(self, seq_len: int, embedding_dim: int, num_heads: int, window_size: int = 256):
        super().__init__()
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.window_size = window_size

        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)

        # Pre-compute mask
        self.register_buffer('window_mask', self._build_mask(seq_len))

    def _build_mask(self, seq_len: int) -> torch.Tensor:
        """Build sliding window mask"""
        idx = torch.arange(seq_len)
        distance = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        return distance > self.window_size

    def _sigsoftmax(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """SigSoftmax: softmax(x + logsigmoid(x))"""
        transformed = x + F.logsigmoid(x)
        return F.softmax(transformed, dim=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (D, T, B) or (D, T)
        Returns:
            output: same shape as input
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            squeeze_output = True
        else:
            squeeze_output = False

        D, T, B = x.shape

        # Transpose: (B, T, D)
        x_t = x.permute(2, 1, 0)

        # Project Q, K, V
        q = self.query_proj(x_t)
        k = self.key_proj(x_t)
        v = self.value_proj(x_t)

        # Reshape to heads: (B, H, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Compute attention scores: (B, H, T, T)
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Apply sliding window mask
        mask = self.window_mask[:T, :T]
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # SigSoftmax
        attn_weights = self._sigsoftmax(scores, dim=-1)

        # Apply to values
        out = torch.matmul(attn_weights, v)  # (B, H, T, head_dim)

        # Reshape back: (B, T, D)
        out = out.permute(0, 2, 1, 3).reshape(B, T, D)

        # Output projection
        out = self.output_proj(out)

        # Transpose: (D, T, B)
        out = out.permute(2, 1, 0)

        if squeeze_output:
            out = out.squeeze(-1)

        return out


class SwiGLU(nn.Module):
    """SwiGLU FFN: Swish-Gated Linear Unit"""

    def __init__(self, dim: int, expansion_factor: float = 1.5):
        super().__init__()
        hidden = int(dim * expansion_factor)
        hidden = hidden + (hidden % 2)  # Ensure even

        self.expand = nn.Linear(dim, hidden)
        self.contract = nn.Linear(hidden // 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (D, T, B) -> transpose for linear
        shape = x.shape
        x_t = x.permute(2, 1, 0) if x.dim() == 3 else x.permute(1, 0)

        expanded = self.expand(x_t)
        half = expanded.shape[-1] // 2
        a, b = expanded[..., :half], expanded[..., half:]

        gated = F.silu(a) * b
        out = self.contract(gated)

        if len(shape) == 3:
            return out.permute(2, 1, 0)
        return out.permute(1, 0)


class TimeConditionedLayerNorm(nn.Module):
    """LayerNorm with time-conditioned scale and shift"""

    def __init__(self, embedding_dim: int, time_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.scale_proj = nn.Linear(time_dim, embedding_dim)
        self.shift_proj = nn.Linear(time_dim, embedding_dim)
        self.alpha_bias_proj = nn.Linear(time_dim, 1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (D, T, B) input tensor
            time_emb: (time_dim, B) time embedding
        Returns:
            normalized: (D, T, B)
            alpha_bias: (1, B)
        """
        D, T, B = x.shape

        # Normalize
        x_t = x.permute(2, 1, 0)  # (B, T, D)
        normalized = self.layer_norm(x_t)

        # Time conditioning
        time_t = time_emb.permute(1, 0)  # (B, time_dim)
        scale = 1.0 + self.scale_proj(time_t)  # (B, D)
        shift = self.shift_proj(time_t)
        alpha_bias = self.alpha_bias_proj(time_t)  # (B, 1)

        # Apply: (B, T, D) * (B, 1, D) + (B, 1, D)
        out = normalized * scale.unsqueeze(1) + shift.unsqueeze(1)

        return out.permute(2, 1, 0), alpha_bias.permute(1, 0)


class OssammaNERBlock(nn.Module):
    """Single OssammaNER block with dual gating"""

    def __init__(self, embedding_dim: int, seq_len: int, num_heads: int, time_dim: int,
                 state_dim: Optional[int] = None, window_size: int = 256,
                 dropout_rate: float = 0.1, use_ffn: bool = True,
                 ffn_expansion: float = 1.5):
        super().__init__()
        self.embedding_dim = embedding_dim
        state_dim = state_dim or embedding_dim

        # Time-conditioned LayerNorm
        self.input_norm = TimeConditionedLayerNorm(embedding_dim, time_dim)

        # GLU branch
        self.glu_proj = nn.Linear(embedding_dim, 2 * embedding_dim)
        self.linear_attn = LinearAttentionLayer(embedding_dim, seq_len, num_heads, time_dim)
        self.oscillator = DLinOSS(embedding_dim, state_dim, embedding_dim)
        self.glu_out_proj = nn.Linear(embedding_dim, embedding_dim)

        # Local branch
        self.sw_attn = SWAttention(seq_len, embedding_dim, num_heads, window_size)

        # Input gate
        self.input_gate = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # Alpha mixing
        self.alpha_proj = nn.Linear(embedding_dim, 1)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # FFN
        self.use_ffn = use_ffn
        if use_ffn:
            self.ffn = SwiGLU(embedding_dim, ffn_expansion)

        # Output LayerNorm
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (D, T, B) input tensor
            time_emb: (time_dim, B) time embedding
        Returns:
            output: (D, T, B)
        """
        D, T, B = x.shape
        residual = x

        # 1. Time-conditioned LayerNorm
        normalized, alpha_bias = self.input_norm(x, time_emb)

        # 2. GLU branch
        norm_t = normalized.permute(2, 1, 0)  # (B, T, D)
        glu_proj = self.glu_proj(norm_t)  # (B, T, 2D)
        path_a = glu_proj[..., :D]  # Content path
        path_b = glu_proj[..., D:]  # Gate path

        # Linear attention on content
        path_a = path_a.permute(2, 1, 0)  # (D, T, B)
        attn_out = self.linear_attn(path_a, time_emb)

        # Oscillator on gate
        path_b = path_b.permute(2, 1, 0)
        osc_out = self.oscillator(path_b)

        # GLU gating
        gated = attn_out * torch.sigmoid(osc_out)

        # Output projection
        gated_t = gated.permute(2, 1, 0)
        glu_out = self.glu_out_proj(gated_t).permute(2, 1, 0)

        # 3. Local branch with input gating
        glu_out_for_gate = glu_out.permute(2, 1, 0)
        input_gate = torch.sigmoid(self.input_gate(glu_out_for_gate))
        gated_x = normalized * input_gate.permute(2, 1, 0)

        local_out = self.sw_attn(gated_x)

        # 4. Adaptive mixing
        # Mean pool normalized for alpha
        input_pooled = normalized.mean(dim=1)  # (D, B)
        input_pooled_t = input_pooled.permute(1, 0)  # (B, D)
        alpha_logits = self.alpha_proj(input_pooled_t)  # (B, 1)
        alpha = torch.sigmoid(alpha_logits + alpha_bias.permute(1, 0))  # (B, 1)
        alpha = alpha.permute(1, 0).unsqueeze(1)  # (1, 1, B)

        mixed = alpha * glu_out + (1.0 - alpha) * local_out

        # 5. Dropout
        mixed_t = mixed.permute(2, 1, 0)  # (B, T, D)
        mixed_t = self.dropout(mixed_t)

        # 6. FFN
        if self.use_ffn:
            mixed = mixed_t.permute(2, 1, 0)
            mixed = self.ffn(mixed)
            mixed_t = mixed.permute(2, 1, 0)

        # 7. Residual + Output LayerNorm
        out_t = residual.permute(2, 1, 0) + mixed_t
        out_t = self.output_norm(out_t)

        return out_t.permute(2, 1, 0)


class FixedTimeEmbedding(nn.Module):
    """Fixed sinusoidal time embedding for NER (no diffusion)"""

    def __init__(self, time_dim: int, fixed_value: float = 0.5):
        super().__init__()
        self.time_dim = time_dim
        half_dim = time_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half_dim) / half_dim)
        args = freqs * fixed_value
        embedding = torch.cat([torch.sin(args), torch.cos(args)])
        self.register_buffer('embedding', embedding)

    def forward(self, batch_size: int) -> torch.Tensor:
        """Returns: (time_dim, B)"""
        return self.embedding.unsqueeze(1).expand(-1, batch_size)


class OssammaNER(nn.Module):
    """Full OssammaNER model for ONNX export"""

    def __init__(self, config: dict):
        super().__init__()

        self.vocab_size = config.get('vocab_size', 32000)
        self.max_seq_len = config.get('max_sequence_length', 512)
        self.embedding_dim = config.get('embedding_dimension', 256)
        self.num_heads = config.get('number_of_heads', 4)
        self.num_layers = config.get('number_of_layers', 4)
        self.num_labels = config.get('num_labels', 17)
        self.time_dim = config.get('time_dimension', 64)
        self.state_dim = config.get('state_dimension', self.embedding_dim)
        self.window_size = config.get('window_size', 256)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        self.use_ffn = config.get('use_ffn', True)
        self.ffn_expansion = config.get('ffn_expansion', 1.5)

        # Embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.position_embedding = nn.Embedding(self.max_seq_len, self.embedding_dim)
        self.time_embedding = FixedTimeEmbedding(self.time_dim)

        # Encoder blocks
        self.blocks = nn.ModuleList([
            OssammaNERBlock(
                self.embedding_dim, self.max_seq_len, self.num_heads, self.time_dim,
                self.state_dim, self.window_size, self.dropout_rate,
                self.use_ffn, self.ffn_expansion
            )
            for _ in range(self.num_layers)
        ])

        # Classification head
        self.dropout = nn.Dropout(self.dropout_rate)
        self.classification_head = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.embedding_dim, self.num_labels)
        )

        # Boundary head (auxiliary)
        self.boundary_head = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, 2)
        )

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            token_ids: (T, B) or (T,) token indices
        Returns:
            emissions: (num_labels, T, B) or (num_labels, T)
            boundary_logits: (2, T, B) or (2, T)
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(-1)
            squeeze_output = True
        else:
            squeeze_output = False

        T, B = token_ids.shape

        # Token embedding
        token_emb = self.token_embedding(token_ids)  # (T, B, D)
        token_emb = token_emb.permute(2, 0, 1)  # (D, T, B)

        # Position embedding
        pos_idx = torch.arange(T, device=token_ids.device)
        pos_emb = self.position_embedding(pos_idx)  # (T, D)
        pos_emb = pos_emb.permute(1, 0).unsqueeze(-1)  # (D, T, 1)

        # Combine
        hidden = token_emb + pos_emb

        # Time embedding
        time_emb = self.time_embedding(B)  # (time_dim, B)

        # Process through blocks
        for block in self.blocks:
            hidden = block(hidden, time_emb)

        # Dropout
        hidden_t = hidden.permute(2, 1, 0)  # (B, T, D)
        hidden_t = self.dropout(hidden_t)

        # Classification head
        emissions = self.classification_head(hidden_t)  # (B, T, num_labels)
        emissions = emissions.permute(2, 1, 0)  # (num_labels, T, B)

        # Boundary head
        boundary_logits = self.boundary_head(hidden_t)  # (B, T, 2)
        boundary_logits = boundary_logits.permute(2, 1, 0)  # (2, T, B)

        if squeeze_output:
            emissions = emissions.squeeze(-1)
            boundary_logits = boundary_logits.squeeze(-1)

        return emissions, boundary_logits


# =============================================================================
# Weight Loading
# =============================================================================

def load_weights_from_julia(model: OssammaNER, params: dict) -> None:
    """Load weights from Julia checkpoint into PyTorch model"""

    def get_param(key: str) -> Optional[np.ndarray]:
        """Get parameter, handling nested keys"""
        # Try direct key
        if key in params:
            return params[key]
        # Try with different separators
        for sep in ['.', '_']:
            alt_key = key.replace('.', sep)
            if alt_key in params:
                return params[alt_key]
        return None

    def set_linear(module: nn.Linear, prefix: str):
        """Load weights for a Linear layer"""
        weight = get_param(f"{prefix}.weight")
        bias = get_param(f"{prefix}.bias")
        if weight is not None:
            # Julia is column-major, PyTorch is row-major
            module.weight.data.copy_(torch.from_numpy(weight.T))
        if bias is not None and module.bias is not None:
            module.bias.data.copy_(torch.from_numpy(bias))

    def set_embedding(module: nn.Embedding, prefix: str):
        """Load weights for an Embedding layer"""
        weight = get_param(f"{prefix}.weight")
        if weight is not None:
            module.weight.data.copy_(torch.from_numpy(weight.T))

    def set_layer_norm(module: nn.LayerNorm, prefix: str):
        """Load weights for a LayerNorm layer"""
        scale = get_param(f"{prefix}.scale")
        bias = get_param(f"{prefix}.bias")
        if scale is not None:
            module.weight.data.copy_(torch.from_numpy(scale))
        if bias is not None:
            module.bias.data.copy_(torch.from_numpy(bias))

    # Load embeddings
    set_embedding(model.token_embedding, "TokenEmbedding")
    set_embedding(model.position_embedding, "PositionEmbedding")

    # Time embedding
    time_emb = get_param("TimeEmbedding.embedding")
    if time_emb is not None:
        model.time_embedding.embedding.copy_(torch.from_numpy(time_emb))

    # Load blocks
    for i, block in enumerate(model.blocks):
        prefix = f"Blocks.Block_{i+1}"

        # Input norm
        set_layer_norm(block.input_norm.layer_norm, f"{prefix}.InputNorm.LayerNorm")
        set_linear(block.input_norm.scale_proj, f"{prefix}.InputNorm.ScaleProjection")
        set_linear(block.input_norm.shift_proj, f"{prefix}.InputNorm.ShiftProjection")
        set_linear(block.input_norm.alpha_bias_proj, f"{prefix}.InputNorm.AlphaBiasProjection")

        # GLU projection
        set_linear(block.glu_proj, f"{prefix}.GluProjection")
        set_linear(block.glu_out_proj, f"{prefix}.GluOutputProjection")

        # Linear attention (complex nested structure)
        la_prefix = f"{prefix}.LinearAttention"
        set_linear(block.linear_attn.query_proj, f"{la_prefix}.QueryProjection")
        set_linear(block.linear_attn.key_proj, f"{la_prefix}.KeyProjection")
        set_linear(block.linear_attn.value_proj, f"{la_prefix}.ValueProjection")
        set_linear(block.linear_attn.output_proj, f"{la_prefix}.OutputProjection")
        set_linear(block.linear_attn.time_proj, f"{la_prefix}.TimeProjection")
        set_embedding(block.linear_attn.pos_embed_cos, f"{la_prefix}.PositionEmbeddingCosine")
        set_embedding(block.linear_attn.pos_embed_sin, f"{la_prefix}.PositionEmbeddingSine")
        set_layer_norm(block.linear_attn.feature_norm, f"{la_prefix}.FeatureNorm")

        # Oscillator
        osc_prefix = f"{prefix}.OscillatorLayer"
        log_dt = get_param(f"{osc_prefix}.log_time_step")
        log_k = get_param(f"{osc_prefix}.log_stiffness_coefficients")
        log_c = get_param(f"{osc_prefix}.log_damping_coefficients")
        input_proj = get_param(f"{osc_prefix}.input_projection")
        output_proj = get_param(f"{osc_prefix}.output_projection")

        if log_dt is not None:
            block.oscillator.log_time_step.data.copy_(torch.from_numpy(log_dt))
        if log_k is not None:
            block.oscillator.log_stiffness.data.copy_(torch.from_numpy(log_k))
        if log_c is not None:
            block.oscillator.log_damping.data.copy_(torch.from_numpy(log_c))
        if input_proj is not None:
            block.oscillator.input_proj.weight.data.copy_(torch.from_numpy(input_proj))
        if output_proj is not None:
            block.oscillator.output_proj.weight.data.copy_(torch.from_numpy(output_proj))

        # Sliding window attention
        sw_prefix = f"{prefix}.SlidingWindowAttention"
        set_linear(block.sw_attn.query_proj, f"{sw_prefix}.QueryProjection")
        set_linear(block.sw_attn.key_proj, f"{sw_prefix}.KeyProjection")
        set_linear(block.sw_attn.value_proj, f"{sw_prefix}.ValueProjection")
        set_linear(block.sw_attn.output_proj, f"{sw_prefix}.OutputProjection")

        # Input gate
        set_linear(block.input_gate, f"{prefix}.InputGate")

        # Alpha projection
        set_linear(block.alpha_proj, f"{prefix}.AlphaProjection")

        # FFN
        if block.use_ffn:
            set_linear(block.ffn.expand, f"{prefix}.FFN.Expand")
            set_linear(block.ffn.contract, f"{prefix}.FFN.Contract")

        # Output norm
        set_layer_norm(block.output_norm, f"{prefix}.OutputNorm")

    # Classification head
    set_layer_norm(model.classification_head[0], "ClassificationHead.layer_1")
    set_linear(model.classification_head[2], "ClassificationHead.layer_3")

    # Boundary head
    set_layer_norm(model.boundary_head[0], "BoundaryHead.layer_1")
    set_linear(model.boundary_head[1], "BoundaryHead.layer_2")

    print("Weights loaded successfully!")


# =============================================================================
# ONNX Export
# =============================================================================

def export_to_onnx(model: OssammaNER, output_path: str, seq_len: int = 128):
    """Export model to ONNX format"""
    model.eval()

    # Create dummy input
    dummy_input = torch.randint(0, model.vocab_size, (seq_len, 1))

    # Export
    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['token_ids'],
        output_names=['emissions', 'boundary_logits'],
        dynamic_axes={
            'token_ids': {0: 'seq_len', 1: 'batch'},
            'emissions': {1: 'seq_len', 2: 'batch'},
            'boundary_logits': {1: 'seq_len', 2: 'batch'},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    print(f"ONNX model saved to: {output_path}")
    print(f"  Input: token_ids (seq_len, batch)")
    print(f"  Output: emissions ({model.num_labels}, seq_len, batch)")
    print(f"  Output: boundary_logits (2, seq_len, batch)")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Export OssammaNER to ONNX")
    parser.add_argument("checkpoint", type=Path, help="Path to Julia .jls checkpoint")
    parser.add_argument("output", type=Path, help="Output ONNX path")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length for export")
    parser.add_argument("--config", type=Path, help="Optional config.toml for model architecture")
    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_julia_checkpoint(str(args.checkpoint))

    # Get config
    if args.config:
        import toml
        config = toml.load(args.config).get('model', {})
    elif checkpoint['config']:
        config = checkpoint['config']
    else:
        # Use defaults
        print("Warning: No config found, using defaults")
        config = {}

    # Create model
    print("Creating PyTorch model...")
    model = OssammaNER(config)

    # Load weights
    print("Loading weights from checkpoint...")
    load_weights_from_julia(model, checkpoint['params'])

    # Export to ONNX
    export_to_onnx(model, str(args.output), args.seq_len)

    # Verify export
    try:
        import onnx
        onnx_model = onnx.load(str(args.output))
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation passed!")
    except ImportError:
        print("Install 'onnx' package to validate exported model")
    except Exception as e:
        print(f"ONNX validation error: {e}")


if __name__ == "__main__":
    main()
