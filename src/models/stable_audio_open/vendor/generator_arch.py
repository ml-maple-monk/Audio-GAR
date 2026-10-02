# vendored: Stability-AI/stable-audio-tools 3241adba4fc2a85cf5b29d9eb68d42f40a28e820 (MIT)
# =============================================================================
# Vendored: Stable Audio Open — latent-diffusion DiT generation model (generator arch).
#
# Copied verbatim from the Stable Audio Open open-source implementation:
#   repo:    https://github.com/Stability-AI/stable-audio-tools
#   commit:  3241adba4fc2a85cf5b29d9eb68d42f40a28e820
#   files:   models/{dit.py, transformer.py, diffusion.py, conditioners.py,
#            adp.py, pretransforms.py, blocks.py, utils.py, lora/utils.py}
#            inference/{sampling.py, generation.py}
# =============================================================================
from __future__ import annotations

import typing as tp
from typing import Callable, Literal, Optional, Union, Dict, List, Tuple
from enum import Enum
from functools import reduce
import math
from math import pi
import os
import copy
import logging
import warnings

import numpy as np
import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from torch.amp import autocast
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from tqdm import trange, tqdm

import k_diffusion as K



# -----------------------------------------------------------------------------
# torch.compile no-op decorator  (models/utils.py)
# -----------------------------------------------------------------------------

enable_torch_compile = os.environ.get("ENABLE_TORCH_COMPILE", "0") == "1"

def compile(function, *args, **kwargs):
    
    if enable_torch_compile:
        try:
            return torch.compile(function, *args, **kwargs)
        except RuntimeError:
            return function

    return function



# -----------------------------------------------------------------------------
# Transformer optional-import guards + module helpers  (models/transformer.py)
# -----------------------------------------------------------------------------

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    flex_attention_available = True
except ImportError:
    flex_attention = None
    create_block_mask = None
    flex_attention_available = False

try:
    from flash_attn import flash_attn_func, flash_attn_kvpacked_func
except ImportError as e:
    print(e)
    print('flash_attn not installed, disabling Flash Attention')
    flash_attn_kvpacked_func = None
    flash_attn_func = None

try:
    from flash_attn import flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input, index_first_axis
except ImportError as e:
    print(e)
    print('flash_attn varlen/bert_padding not available, disabling varlen attention')
    flash_attn_varlen_func = None
    pad_input = None
    unpad_input = None
    index_first_axis = None

def precompute_varlen_metadata(padding_mask: torch.Tensor):
    """
    Precompute varlen attention metadata once to avoid recomputation in every attention layer.

    Args:
        padding_mask: Boolean tensor of shape (batch, seq_len) where True = valid

    Returns:
        Dict with cu_seqlens, max_seqlen, indices, batch_size, seq_len for use in attention
    """
    if padding_mask is None or unpad_input is None:
        return None

    batch_size, seq_len = padding_mask.shape

    # Compute cumulative sequence lengths (same for all of q, k, v)
    seqlens = padding_mask.sum(dim=-1, dtype=torch.int32)
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    max_seqlen = seqlens.max().item()

    # Compute indices for gathering valid tokens
    # indices maps from packed position -> original (batch, seq) position
    indices = torch.nonzero(padding_mask.flatten(), as_tuple=False).flatten()

    return {
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max_seqlen,
        "indices": indices,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }

def _left_pad_to_match(emb, target_len):
    """Left-pad or right-trim emb along seq dim to match target_len.

    Used for local conditioning embeddings that need to align with x
    without affecting prepended tokens (memory tokens, global cond, etc.).
    """
    emb_len = emb.shape[-2]
    if emb_len < target_len:
        return F.pad(emb, (0, 0, target_len - emb_len, 0), value=0.)
    elif emb_len > target_len:
        return emb[:, -target_len:, :]
    return emb

if flex_attention_available:
    try:
        torch._dynamo.config.cache_size_limit = 5000
        flex_attention_compiled = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
    except Exception as e:
        logging.debug(f"Could not compile flex_attention, using uncompiled version: {e}")
        flex_attention_compiled = flex_attention
else:
    flex_attention_compiled = None

# Cache band block_masks for sliding-window attention fallback (flex_attention path).
# Keyed by (seq_q, seq_k, w_left, w_right, device). create_block_mask is expensive
# but the result is reused across all transformer layers and forward passes.
_SLIDING_WINDOW_BLOCK_MASK_CACHE = {}

def _get_sliding_window_block_mask(seq_q, seq_k, w_left, w_right, device):
    key = (seq_q, seq_k, int(w_left), int(w_right), str(device))
    bm = _SLIDING_WINDOW_BLOCK_MASK_CACHE.get(key)
    if bm is None:
        wl, wr = int(w_left), int(w_right)
        def _band_mod(b, h, q_idx, kv_idx):
            delta = kv_idx - q_idx
            return (delta >= -wl) & (delta <= wr)
        bm = create_block_mask(_band_mod, B=None, H=None, Q_LEN=seq_q, KV_LEN=seq_k, device=device)
        _SLIDING_WINDOW_BLOCK_MASK_CACHE[key] = bm
    return bm

def _sliding_window_additive_mask(seq_q, seq_k, w_left, w_right, device, dtype):
    """Build a (seq_q, seq_k) additive mask for masked SDPA fallback.
    0 inside the band [i - w_left, i + w_right], -inf outside.
    """
    ii = torch.arange(seq_q, device=device)
    jj = torch.arange(seq_k, device=device)
    delta = jj[None, :] - ii[:, None]
    in_band = (delta >= -int(w_left)) & (delta <= int(w_right))
    mask = torch.zeros((seq_q, seq_k), dtype=dtype, device=device)
    return mask.masked_fill(~in_band, float('-inf'))

# Chunked-halo SDPA fallback. Math-equivalent to masked SDPA with a band
# mask, but processes queries in non-overlapping chunks with a (w_left,
# w_right) halo of keys/values on each side — every query stays inside its
# chunk's softmax. Avoids materializing the O(N^2) mask.
#
# At realistic SAME-L decoder shapes (N=69632, W=17, packed sequence is
# latent_length * (stride+1)): ~34x faster than full masked SDPA, and
# ~140x less peak mask memory (~1 MB per chunk vs 9.7 GB for one N x N mask).
# Chunk size is a tunable; 1024 is a good default at typical pretransform
# decoder shapes. Larger chunks waste more compute on out-of-band tiles;
# smaller chunks suffer from launch overhead.
_SLIDING_WINDOW_CHUNK_SIZE = 1024

def _sliding_window_chunked_halo_sdpa(q, k, v, w_left, w_right, chunk_size=_SLIDING_WINDOW_CHUNK_SIZE):
    B, H, N, D = q.shape
    outs = []
    for q_start in range(0, N, chunk_size):
        q_end = min(q_start + chunk_size, N)
        k_start = max(0, q_start - int(w_left))
        k_end = min(N, q_end + int(w_right))
        q_c = q[..., q_start:q_end, :]
        k_c = k[..., k_start:k_end, :]
        v_c = v[..., k_start:k_end, :]
        q_idx = torch.arange(q_start, q_end, device=q.device)
        k_idx = torch.arange(k_start, k_end, device=q.device)
        delta = k_idx[None, :] - q_idx[:, None]
        in_band = (delta >= -int(w_left)) & (delta <= int(w_right))
        mask = torch.zeros(delta.shape, dtype=q.dtype, device=q.device).masked_fill(~in_band, float('-inf'))
        outs.append(F.scaled_dot_product_attention(q_c, k_c, v_c, attn_mask=mask, is_causal=False))
    return torch.cat(outs, dim=-2)

def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    # Preserve autocast context during recomputation to avoid dtype mismatches
    if "context_fn" not in kwargs:
        from torch.amp import autocast
        import functools
        # Get current autocast state
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype('cuda')
            def get_contexts():
                return (
                    autocast('cuda', dtype=dtype),
                    autocast('cuda', dtype=dtype),
                )
            kwargs["context_fn"] = get_contexts
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)



# -----------------------------------------------------------------------------
# Timestep Fourier features  (models/blocks.py)
# -----------------------------------------------------------------------------

class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=16.):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)

class ExpoFourierFeatures(nn.Module):
    def __init__(self, dim, min_freq=0.5, max_freq=10000.0):
        super().__init__()
        self.dim = dim
        self.min_freq = min_freq
        self.max_freq = max_freq

    @torch.amp.autocast("cuda",enabled=False)
    def forward(self, t):
        """
        t: [B] tensor.
        """
        in_dtype = t.dtype 
        t = t.float()
        
        if t.dim() == 1:
            t = t.unsqueeze(-1)
            
        half_dim = self.dim // 2
        
        # Calculate frequencies (safely in FP32)
        ramp = torch.linspace(0, 1, half_dim, device=t.device, dtype=torch.float32)
        log_min = math.log(self.min_freq)
        log_max = math.log(self.max_freq)
        
        freqs = torch.exp(ramp * (log_max - log_min) + log_min)
        
        # Calculate arguments (safely in FP32)
        args = t * freqs * 2 * math.pi
        
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        return embedding.to(in_dtype) 



# -----------------------------------------------------------------------------
# Norms  (models/transformer.py)
# -----------------------------------------------------------------------------

# norms
class DynamicTanh(nn.Module):
    def __init__(self, dim, init_alpha=4.0, **kwargs):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = F.tanh(self.alpha * x)
        return self.gamma * x + self.beta

class LayerNorm(nn.Module):
    def __init__(self, dim, bias=False, fix_scale=False, force_fp32=False, eps=1e-5):
        """
        bias-less layernorm has been shown to be more stable. most newer models have moved towards rmsnorm, also bias-less
        """
        super().__init__()

        if fix_scale:
            self.register_buffer("gamma", torch.ones(dim))
        else:
            self.gamma = nn.Parameter(torch.ones(dim))

        if bias:
            self.beta = nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer("beta", torch.zeros(dim))

        self.eps = eps

        self.force_fp32 = force_fp32

    def forward(self, x):
        if not self.force_fp32:
            return F.layer_norm(x, x.shape[-1:], weight=self.gamma, bias=self.beta, eps=self.eps)
        else:
            output = F.layer_norm(x.float(), x.shape[-1:], weight=self.gamma.float(), bias=self.beta.float(), eps=self.eps)
            return output.to(x.dtype)

class RMSNorm(nn.Module):
    def __init__(self, dim, fix_scale=False, force_fp32=False, eps=1e-5):
        super().__init__()

        if fix_scale:
            self.register_buffer("gamma", torch.ones(dim))
        else:
            self.gamma = nn.Parameter(torch.ones(dim))

        self.eps = eps

        self.force_fp32 = force_fp32

    def forward(self, x):
        if not self.force_fp32:
            return F.rms_norm(x, x.shape[-1:], weight=self.gamma, eps=self.eps)
        else:
            output = F.rms_norm(x.float(), x.shape[-1:], weight=self.gamma.float(), eps=self.eps)
            return output.to(x.dtype)

class LayerScale(nn.Module):
    def __init__(self, dim, init_val = 1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.full([dim], init_val))
    def forward(self, x):
        return x * self.scale



# -----------------------------------------------------------------------------
# Rotary position embedding  (models/transformer.py)
# -----------------------------------------------------------------------------

def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

@autocast("cuda", enabled = False)
def apply_rotary_pos_emb(t, freqs, scale = 1):
    out_dtype = t.dtype

    # cast to float32 if necessary for numerical stability
    dtype = reduce(torch.promote_types, (t.dtype, freqs.dtype, torch.float32))
    rot_dim, seq_len = freqs.shape[-1], t.shape[-2]
    freqs, t = freqs.to(dtype), t.to(dtype)
    freqs = freqs[-seq_len:, :]

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, 'b n d -> b 1 n d')

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]

    t = (t * freqs.cos() * scale ) + (rotate_half(t) * freqs.sin() * scale)

    t, t_unrotated = t.to(out_dtype), t_unrotated.to(out_dtype)

    return torch.cat((t, t_unrotated), dim = -1)

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        use_xpos = False,
        scale_base = 512,
        interpolation_factor = 1.,
        base = 10000,
        base_rescale_factor = 1.
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        assert interpolation_factor >= 1.
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer('scale', None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer('scale', scale)

    def forward_from_seq_len(self, seq_len):
        device = self.inv_freq.device

        t = torch.arange(seq_len, device = device)
        return self.forward(t)

    @autocast("cuda", enabled = False)
    def forward(self, t):
        device = self.inv_freq.device

        t = t.to(torch.float32)

        t = t / self.interpolation_factor

        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)

        if self.scale is None:
            return freqs, 1.

        power = (torch.arange(seq_len, device = device) - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale



# -----------------------------------------------------------------------------
# Absolute / sinusoidal position embeddings  (models/transformer.py)
# -----------------------------------------------------------------------------

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.scale = dim ** -0.5
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'

        if pos is None:
            pos = torch.arange(seq_len, device = device)

        if seq_start_pos is not None:
            pos = (pos - seq_start_pos[..., None]).clamp(min = 0)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return pos_emb

class ScaledSinusoidalEmbedding(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        assert (dim % 2) == 0, 'dimension must be divisible by 2'
        self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5)

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta ** -freq_seq
        self.register_buffer('inv_freq', inv_freq, persistent = False)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device

        if pos is None:
            pos = torch.arange(seq_len, device = device)

        if seq_start_pos is not None:
            pos = pos - seq_start_pos[..., None]

        emb = einsum('i, j -> i j', pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
        return emb * self.scale



# -----------------------------------------------------------------------------
# Feedforward (SwiGLU)  (models/transformer.py)
# -----------------------------------------------------------------------------

class GLU(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        activation: Callable,
        use_conv = False,
        conv_kernel_size = 3,
    ):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2) if not use_conv else nn.Conv1d(dim_in, dim_out * 2, conv_kernel_size, padding = (conv_kernel_size // 2))
        self.use_conv = use_conv

    def forward(self, x):
        if self.use_conv:
            x = rearrange(x, 'b n d -> b d n')
            x = self.proj(x)
            x = rearrange(x, 'b d n -> b n d')
        else:
            x = self.proj(x)

        x, gate = x.chunk(2, dim = -1)
        return x * self.act(gate)

class Sin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(3.14159265359 * x)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        mult = 4,
        no_bias = False,
        glu = True,
        use_conv = False,
        conv_kernel_size = 3,
        zero_init_output = True,
        sinusoidal = False
    ):
        super().__init__()
        inner_dim = int(dim * mult)

        # Default to SwiGLU

        activation = nn.SiLU() if not sinusoidal else Sin()

        dim_out = dim if dim_out is None else dim_out

        if glu:
            linear_in = GLU(dim, inner_dim, activation)
        else:
            linear_in = nn.Sequential(
                Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
                nn.Linear(dim, inner_dim, bias = not no_bias) if not use_conv else nn.Conv1d(dim, inner_dim, conv_kernel_size, padding = (conv_kernel_size // 2), bias = not no_bias),
                Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
                activation
            )

        linear_out = nn.Linear(inner_dim, dim_out, bias = not no_bias) if not use_conv else nn.Conv1d(inner_dim, dim_out, conv_kernel_size, padding = (conv_kernel_size // 2), bias = not no_bias)

        # init last linear layer to 0
        if zero_init_output:
            nn.init.zeros_(linear_out.weight)
            if not no_bias:
                nn.init.zeros_(linear_out.bias)


        self.ff = nn.Sequential(
            linear_in,
            Rearrange('b d n -> b n d') if use_conv else nn.Identity(),
            linear_out,
            Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
        )

    #@compile
    def forward(self, x, varlen_metadata=None):
        if varlen_metadata is not None and index_first_axis is not None and pad_input is not None:
            # Pack valid tokens for efficient FFN computation (skip padding tokens)
            # Padding positions become zeros after unpack, which is fine since FFN output
            # is added to residual, preserving values at padding positions
            batch_size = varlen_metadata["batch_size"]
            seq_len = varlen_metadata["seq_len"]
            indices = varlen_metadata["indices"]
            dim = x.shape[-1]

            # Pack to (N_valid, D)
            x_packed = index_first_axis(x.reshape(-1, dim), indices)

            # FFN on packed representation with pseudo-batch dim
            x_packed = self.ff(x_packed.unsqueeze(0)).squeeze(0)

            # Unpack back to (B, T, D)
            return pad_input(x_packed, indices, batch_size, seq_len)
        else:
            return self.ff(x)



# -----------------------------------------------------------------------------
# Attention  (models/transformer.py)
# -----------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_heads = 64,
        dim_context = None,
        causal = False,
        zero_init_output=True,
        qk_norm_eps = 1e-6,
        qk_norm: Literal['l2', 'ln', 'rms', 'dyt', 'none'] = 'none',
        differential = False,
        feat_scale = False
    ):
        super().__init__()
        self.dim = dim
        self.dim_heads = dim_heads

        self.differential = differential

        dim_kv = dim_context if dim_context is not None else dim
        
        self.num_heads = dim // dim_heads
        self.kv_heads = dim_kv // dim_heads

        if dim_context is not None:
            if differential:
                self.to_q = nn.Linear(dim, dim * 2, bias=False)
                self.to_kv = nn.Linear(dim_kv, dim_kv * 3, bias=False)
            else:
                self.to_q = nn.Linear(dim, dim, bias=False)
                self.to_kv = nn.Linear(dim_kv, dim_kv * 2, bias=False)
        else:
            if differential:
                self.to_qkv = nn.Linear(dim, dim * 5, bias=False)
            else:
                self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.to_out = nn.Linear(dim, dim, bias=False)

        if zero_init_output:
            nn.init.zeros_(self.to_out.weight)

        if qk_norm not in ['l2', 'ln', 'rms', 'dyt','none']:
            raise ValueError(f'qk_norm must be one of ["l2", "ln", "rms" ,"dyt", "none"], got {qk_norm}')
            
        self.qk_norm = qk_norm
        self.qk_norm_eps = qk_norm_eps

        if self.qk_norm == "ln":
            self.q_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=qk_norm_eps)
            self.k_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=qk_norm_eps)
        elif self.qk_norm == "rms":
            self.q_norm = RMSNorm(dim_heads, eps=qk_norm_eps)
            self.k_norm = RMSNorm(dim_heads, eps=qk_norm_eps)
        elif self.qk_norm == 'dyt':
            self.q_norm = DynamicTanh(dim_heads)
            self.k_norm = DynamicTanh(dim_heads)

        self.feat_scale = feat_scale

        if self.feat_scale:
            self.lambda_dc = nn.Parameter(torch.zeros(dim))
            self.lambda_hf = nn.Parameter(torch.zeros(dim))

        self.causal = causal
        
    @compile
    def apply_qk_layernorm(self, q, k):
        q_type = q.dtype
        k_type = k.dtype
        q = self.q_norm(q).to(q_type)
        k = self.k_norm(k).to(k_type)
        return q, k


    def apply_attn(self, q, k, v, causal = None, flex_attention_block_mask = None, flex_attention_score_mod = None, flash_attn_sliding_window = None, padding_mask = None, varlen_metadata = None):

        if self.num_heads != self.kv_heads:
             # Repeat interleave kv_heads to match q_heads for grouped query attention
             heads_per_kv_head = self.num_heads // self.kv_heads
             k, v = map(lambda t: t.repeat_interleave(heads_per_kv_head, dim = 1), (k, v))

        flash_attn_available = flash_attn_func is not None
        flash_attn_varlen_available = flash_attn_varlen_func is not None and index_first_axis is not None

        if causal and (flex_attention_block_mask is not None or flex_attention_score_mod is not None):
            flex_attention_block_mask = None
            flex_attention_score_mod = None

        if flex_attention_block_mask is not None or flex_attention_score_mod is not None:
            # Flex attention path - use V-zeroing for padding mask
            if padding_mask is not None:
                mask_expanded = padding_mask.unsqueeze(1).unsqueeze(-1).to(v.dtype)
                v = v * mask_expanded
            out = flex_attention_compiled(q,k,v,
                block_mask = flex_attention_block_mask,
                score_mod = flex_attention_score_mod)
        elif flash_attn_available and varlen_metadata is not None and flash_attn_varlen_available:
            # Flash attention with varlen using precomputed metadata (fast path)
            batch_size = varlen_metadata["batch_size"]
            seq_len = varlen_metadata["seq_len"]
            cu_seqlens = varlen_metadata["cu_seqlens"]
            max_seqlen = varlen_metadata["max_seqlen"]
            indices = varlen_metadata["indices"]

            fa_dtype_in = q.dtype
            # Rearrange to (B, T, H, D) for flash_attn
            q, k, v = map(lambda t: rearrange(t, 'b h n d -> b n h d'), (q, k, v))

            if fa_dtype_in != torch.float16 and fa_dtype_in != torch.bfloat16:
                q, k, v = map(lambda t: t.to(torch.float16), (q, k, v))

            # Pack q, k, v using precomputed indices (much faster than calling unpad_input 3x)
            num_heads, head_dim = q.shape[2], q.shape[3]
            q_unpad = index_first_axis(q.reshape(-1, num_heads, head_dim), indices)
            k_unpad = index_first_axis(k.reshape(-1, num_heads, head_dim), indices)
            v_unpad = index_first_axis(v.reshape(-1, num_heads, head_dim), indices)

            out_unpad = flash_attn_varlen_func(
                q_unpad, k_unpad, v_unpad,
                cu_seqlens, cu_seqlens,
                max_seqlen, max_seqlen,
                causal=causal if causal is not None else False,
                window_size=flash_attn_sliding_window if flash_attn_sliding_window is not None else (-1, -1),
            )

            # Pad output back to original shape
            out = pad_input(out_unpad, indices, batch_size, seq_len)
            out = rearrange(out.to(fa_dtype_in), 'b n h d -> b h n d')
        elif flash_attn_available:
            # Standard flash attention (no padding mask, or varlen imports not available)
            # Apply V-zeroing fallback if padding_mask provided but we couldn't use varlen
            if padding_mask is not None:
                mask_expanded = padding_mask.unsqueeze(1).unsqueeze(-1).to(v.dtype)
                v = v * mask_expanded
            fa_dtype_in = q.dtype
            q, k, v = map(lambda t: rearrange(t, 'b h n d -> b n h d'), (q, k, v))

            if fa_dtype_in != torch.float16 and fa_dtype_in != torch.bfloat16:
                q, k, v = map(lambda t: t.to(torch.float16), (q, k, v))

            out = flash_attn_func(q, k, v, causal = causal, window_size=flash_attn_sliding_window if (flash_attn_sliding_window is not None) else [-1,-1])

            out = rearrange(out.to(fa_dtype_in), 'b n h d -> b h n d')
        else:
            # No flash-attn available. Sliding-window fallback cascade:
            #   Tier 2: flex_attention with band block_mask (best when torch.compile works)
            #   Tier 3: chunked-halo masked SDPA           (math-equivalent, ~30x faster than tier 4)
            #   Tier 4: full masked SDPA (N x N mask)      (last resort; high memory)
            # For the no-sliding-window case, fall through to plain SDPA full attention.
            # All apply V-zeroing for padding masks (cheap and equivalent to masking
            # those positions out of attention output).
            if padding_mask is not None:
                mask_expanded = padding_mask.unsqueeze(1).unsqueeze(-1).to(v.dtype)
                v = v * mask_expanded
            if flash_attn_sliding_window is not None:
                seq_q, seq_k = q.shape[2], k.shape[2]
                wl, wr = flash_attn_sliding_window
                handled = False
                if flex_attention_available and flex_attention_compiled is not None:
                    try:
                        bm = _get_sliding_window_block_mask(seq_q, seq_k, wl, wr, q.device)
                        out = flex_attention_compiled(q, k, v, block_mask=bm)
                        handled = True
                    except Exception as _flex_err:
                        logging.debug(f"flex_attention failed, trying chunked-halo SDPA: {_flex_err}")
                if not handled:
                    try:
                        out = _sliding_window_chunked_halo_sdpa(q, k, v, wl, wr)
                        handled = True
                    except Exception as _chunk_err:
                        logging.debug(f"chunked-halo SDPA failed, falling back to full masked SDPA: {_chunk_err}")
                if not handled:
                    add_mask = _sliding_window_additive_mask(seq_q, seq_k, wl, wr, q.device, q.dtype)
                    out = F.scaled_dot_product_attention(q, k, v, attn_mask=add_mask, is_causal=False)
            else:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=causal if causal is not None else False)
        return out


    #@compile
    def forward(
        self,
        x,
        context = None,
        rotary_pos_emb = None,
        rotary_pos_emb_k = None,
        causal = None,
        flex_attention_block_mask = None,
        flex_attention_score_mod = None,
        flash_attn_sliding_window = None,
        padding_mask = None,
        varlen_metadata = None,
    ):
        h, kv_h, has_context = self.num_heads, self.kv_heads, context is not None

        kv_input = context if has_context else x

        if hasattr(self, 'to_q'):
            # Use separate linear projections for q and k/v
            if self.differential:
                q, q_diff = self.to_q(x).chunk(2, dim=-1)
                q, q_diff = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, q_diff))
                q = torch.stack([q, q_diff], dim = 1)
                k, k_diff, v = self.to_kv(kv_input).chunk(3, dim=-1)
                k, k_diff, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = kv_h), (k, k_diff, v))
                k = torch.stack([k, k_diff], dim = 1)
            else:
                q = self.to_q(x)
                q = rearrange(q, 'b n (h d) -> b h n d', h = h)
                k, v = self.to_kv(kv_input).chunk(2, dim=-1)
                k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = kv_h), (k, v))
        else:
            # Use fused linear projection
            if self.differential:
                q, k, v, q_diff, k_diff = self.to_qkv(x).chunk(5, dim=-1)
                q, k, v, q_diff, k_diff  = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v, q_diff, k_diff))
                q = torch.stack([q, q_diff], dim = 1)
                k = torch.stack([k, k_diff], dim = 1)
            else:
                q, k, v = self.to_qkv(x).chunk(3, dim=-1)
                q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # Normalize q and k for cosine sim attention
        if self.qk_norm == "l2":
            q = F.normalize(q, dim=-1, eps=self.qk_norm_eps)
            k = F.normalize(k, dim=-1, eps=self.qk_norm_eps)
        elif self.qk_norm != "none":
            q, k = self.apply_qk_layernorm(q, k)

        if rotary_pos_emb is not None:
            freqs, _ = rotary_pos_emb
            q_dtype = q.dtype
            k_dtype = k.dtype
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            freqs = freqs.to(torch.float32)
        
            q_freqs = freqs

            if rotary_pos_emb_k is not None:
                k_freqs, _ = rotary_pos_emb_k
                k_freqs = k_freqs.to(torch.float32)
            else:
                k_freqs = q_freqs

                if q.shape[-2] >= k.shape[-2]:
                    ratio = q.shape[-2] / k.shape[-2]
                    q_freqs, k_freqs = freqs, ratio * freqs
                else:
                    ratio = k.shape[-2] / q.shape[-2]
                    q_freqs, k_freqs = ratio * freqs, freqs

            q = apply_rotary_pos_emb(q, q_freqs)
            k = apply_rotary_pos_emb(k, k_freqs)
            q = q.to(v.dtype)
            k = k.to(v.dtype)
        
        n, device = q.shape[-2], q.device

        causal = self.causal if causal is None else causal

        if n == 1 and causal:
            causal = False

        if self.differential:
            q, q_diff = q.unbind(dim = 1)
            k, k_diff = k.unbind(dim = 1)
            out = self.apply_attn(q, k, v,  causal = causal, flex_attention_block_mask = flex_attention_block_mask, flex_attention_score_mod = flex_attention_score_mod, flash_attn_sliding_window = flash_attn_sliding_window, padding_mask = padding_mask, varlen_metadata = varlen_metadata)
            out_diff = self.apply_attn(q_diff, k_diff, v, causal = causal, flex_attention_block_mask = flex_attention_block_mask, flex_attention_score_mod = flex_attention_score_mod, flash_attn_sliding_window = flash_attn_sliding_window, padding_mask = padding_mask, varlen_metadata = varlen_metadata)
            out = out - out_diff
        else:
            out = self.apply_attn(q, k, v, causal = causal, flex_attention_block_mask = flex_attention_block_mask, flex_attention_score_mod = flex_attention_score_mod, flash_attn_sliding_window = flash_attn_sliding_window, padding_mask = padding_mask, varlen_metadata = varlen_metadata)
        # merge heads
        out = rearrange(out, ' b h n d -> b n (h d)')

        # Communicate between heads
        
        # with autocast(enabled = False):
        #     out_dtype = out.dtype
        #     out = out.to(torch.float32)
        #     out = self.to_out(out).to(out_dtype)
        out = self.to_out(out)

        if self.feat_scale:
            if padding_mask is not None:
                mask = padding_mask.unsqueeze(-1).to(out.dtype)  # (b, n, 1)
                out_dc = (out * mask).sum(dim=-2, keepdim=True) / mask.sum(dim=-2, keepdim=True).clamp(min=1)
                out_hf = out - out_dc
                out = out + (self.lambda_dc * out_dc + self.lambda_hf * out_hf) * mask
            else:
                out_dc = out.mean(dim=-2, keepdim=True)
                out_hf = out - out_dc
                out = out + self.lambda_dc * out_dc + self.lambda_hf * out_hf

        return out



# -----------------------------------------------------------------------------
# Conformer / transformer block / backbone  (models/transformer.py)
# -----------------------------------------------------------------------------

class ConformerModule(nn.Module):
    def __init__(
        self,
        dim,
        norm_kwargs = {},
    ):     

        super().__init__()

        self.dim = dim
        
        self.in_norm = LayerNorm(dim, **norm_kwargs)
        self.pointwise_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.glu = GLU(dim, dim, nn.SiLU())
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size=17, groups=dim, padding=8, bias=False)
        self.mid_norm = LayerNorm(dim, **norm_kwargs) # This is a batch norm in the original but I don't like batch norm
        self.swish = nn.SiLU()
        self.pointwise_conv_2 = nn.Conv1d(dim, dim, kernel_size=1, bias=False)

    #@compile
    def forward(self, x):
        x = self.in_norm(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.pointwise_conv(x)
        x = rearrange(x, 'b d n -> b n d')
        x = self.glu(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.depthwise_conv(x)
        x = rearrange(x, 'b d n -> b n d')
        x = self.mid_norm(x)
        x = self.swish(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.pointwise_conv_2(x)
        x = rearrange(x, 'b d n -> b n d')

        return x

class TransformerBlock(nn.Module):
    def __init__(
            self,
            dim,
            dim_heads = 64,
            cross_attend = False,
            dim_context = None,
            global_cond_dim = None,
            local_add_cond_dim = None,
            modular_local_cond_configs = None,
            causal = False,
            zero_init_branch_outputs = True,
            conformer = False,
            layer_ix = -1,
            add_rope = False,
            layer_scale = False,
            norm_type = 'layer_norm',
            attn_kwargs = {},
            ff_kwargs = {},
            norm_kwargs = {}
    ):
        
        super().__init__()
        self.dim = dim
        self.dim_heads = min(dim_heads,dim)
        self.cross_attend = cross_attend
        self.dim_context = dim_context
        self.causal = causal
       
        if layer_scale and zero_init_branch_outputs:
            print('zero_init_branch_outputs is redundant with layer_scale, setting zero_init_branch_outputs to False')
            zero_init_branch_outputs = False
        
        if norm_type not in ['layer_norm', 'rms_norm', 'dyt']:
            raise ValueError(f'norm_type must be one of ["layer_norm", "rms_norm", "dyt"], got {norm_type}')

        norm_layer_map = {
            'layer_norm': LayerNorm,
            'rms_norm': RMSNorm,
            'dyt': DynamicTanh
        }
        norm_layer = norm_layer_map[norm_type]

        self.pre_norm = norm_layer(dim,**norm_kwargs)
        self.add_rope = add_rope

        self.self_attn = Attention(
            dim,
            dim_heads = self.dim_heads,
            causal = causal,
            zero_init_output=zero_init_branch_outputs,
            **attn_kwargs
        )

        self.self_attn_scale = LayerScale(dim) if layer_scale else nn.Identity()

        self.cross_attend = cross_attend
        if cross_attend:
            self.cross_attend_norm = norm_layer(dim, **norm_kwargs)
            self.cross_attn = Attention(
                dim,
                dim_heads = self.dim_heads,
                dim_context=dim_context,
                causal = causal,
                zero_init_output=zero_init_branch_outputs,
                **attn_kwargs
            )
            self.cross_attn_scale = LayerScale(dim) if layer_scale else nn.Identity()
        
        self.ff_norm = norm_layer(dim, **norm_kwargs)
        self.ff = FeedForward(dim, zero_init_output=zero_init_branch_outputs, **ff_kwargs)
        self.ff_scale = LayerScale(dim) if layer_scale else nn.Identity()

        self.layer_ix = layer_ix

        self.conformer = None
        if conformer:
            self.conformer = ConformerModule(dim, norm_kwargs=norm_kwargs)
            self.conformer_scale = LayerScale(dim) if layer_scale else nn.Identity()

        self.global_cond_dim = global_cond_dim

        if global_cond_dim is not None:
            self.to_scale_shift_gate = nn.Parameter(torch.randn(6*dim)/dim**0.5)

        self.local_add_cond_dim = local_add_cond_dim

        if local_add_cond_dim is not None:
            self.to_local_embed = nn.Sequential(
                nn.Linear(local_add_cond_dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim)
            )

            nn.init.zeros_(self.to_local_embed[-1].weight)
            nn.init.zeros_(self.to_local_embed[-1].bias)

        else:
            self.to_local_embed = None

        # Modular local conditioning - independent projections per conditioning ID
        self.modular_local_cond_configs = modular_local_cond_configs or []
        self.modular_local_embeds = nn.ModuleDict()

        for config in self.modular_local_cond_configs:
            cond_id = config["id"]
            cond_dim = config["dim"]
            proj = nn.Sequential(
                nn.Linear(cond_dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim)
            )
            # Zero-init output layer so new conditioning doesn't affect model initially
            nn.init.zeros_(proj[-1].weight)
            nn.init.zeros_(proj[-1].bias)
            self.modular_local_embeds[cond_id] = proj

        self.rope = RotaryEmbedding(self.dim_heads // 2) if add_rope else None

    def _apply_local_conditioning(self, x, local_add_cond, modular_local_cond):
        """Apply local additive and modular local conditioning to x."""
        if local_add_cond is not None and self.to_local_embed is not None:
            local_emb = self.to_local_embed(local_add_cond)
            x = x + _left_pad_to_match(local_emb, x.shape[-2])

        if modular_local_cond is not None and len(self.modular_local_embeds) > 0:
            modular_sum = None
            for cond_id, proj in self.modular_local_embeds.items():
                if cond_id in modular_local_cond:
                    local_emb = proj(modular_local_cond[cond_id])
                    local_emb = _left_pad_to_match(local_emb, x.shape[-2])
                    modular_sum = local_emb if modular_sum is None else modular_sum + local_emb
            if modular_sum is not None:
                x = x + modular_sum

        return x

    @compile
    def forward(
        self,
        x,
        context = None,
        global_cond=None,
        local_add_cond=None,
        modular_local_cond=None,
        rotary_pos_emb = None,
        cross_attn_rotary_pos_emb = None,
        self_attention_block_mask = None,
        self_attention_score_mod = None,
        cross_attention_block_mask = None,
        cross_attention_score_mod = None,
        self_attention_flash_sliding_window = None,
        cross_attention_flash_sliding_window = None,
        padding_mask = None,
        varlen_metadata = None,
    ):
        if rotary_pos_emb is None and self.add_rope:
            rotary_pos_emb = self.rope.forward_from_seq_len(x.shape[-2])

        if self.global_cond_dim is not None and self.global_cond_dim > 0 and global_cond is not None:
            
            scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = (self.to_scale_shift_gate + global_cond).unsqueeze(1).chunk(6, dim=-1)

            # self-attention with adaLN
            residual = x
            x = self.pre_norm(x)
            x = x * (1 + scale_self) + shift_self
            x = self.self_attn(x, rotary_pos_emb = rotary_pos_emb, flex_attention_block_mask = self_attention_block_mask, flex_attention_score_mod = self_attention_score_mod, flash_attn_sliding_window = self_attention_flash_sliding_window, padding_mask = padding_mask, varlen_metadata = varlen_metadata)
            x = x * torch.sigmoid(1 - gate_self)
            x = self.self_attn_scale(x)
            x = x + residual

            if context is not None and self.cross_attend:
                if cross_attn_rotary_pos_emb is not None:
                    x = x + self.cross_attn_scale(self.cross_attn(self.cross_attend_norm(x), rotary_pos_emb = rotary_pos_emb, rotary_pos_emb_k = cross_attn_rotary_pos_emb, context = context, flex_attention_block_mask = cross_attention_block_mask, flex_attention_score_mod = cross_attention_score_mod, flash_attn_sliding_window = cross_attention_flash_sliding_window))
                else:
                    x = x + self.cross_attn_scale(self.cross_attn(self.cross_attend_norm(x), context = context, flex_attention_block_mask = cross_attention_block_mask, flex_attention_score_mod = cross_attention_score_mod, flash_attn_sliding_window = cross_attention_flash_sliding_window))

            if self.conformer is not None:
                x = x + self.conformer_scale(self.conformer(x))

            x = self._apply_local_conditioning(x, local_add_cond, modular_local_cond)

            # feedforward with adaLN
            residual = x
            x = self.ff_norm(x)
            x = x * (1 + scale_ff) + shift_ff
            x = self.ff(x, varlen_metadata=varlen_metadata)
            x = x * torch.sigmoid(1 - gate_ff)
            x = self.ff_scale(x)
            x = x + residual

        else:
            x = x + self.self_attn_scale(self.self_attn(self.pre_norm(x), rotary_pos_emb = rotary_pos_emb, flex_attention_block_mask = self_attention_block_mask, flex_attention_score_mod = self_attention_score_mod, flash_attn_sliding_window = self_attention_flash_sliding_window, padding_mask = padding_mask, varlen_metadata = varlen_metadata))

            if context is not None and self.cross_attend:
                if cross_attn_rotary_pos_emb is not None:
                    x = x + self.cross_attn_scale(self.cross_attn(self.cross_attend_norm(x), rotary_pos_emb = rotary_pos_emb, rotary_pos_emb_k = cross_attn_rotary_pos_emb, context = context, flex_attention_block_mask = cross_attention_block_mask, flex_attention_score_mod = cross_attention_score_mod, flash_attn_sliding_window = cross_attention_flash_sliding_window))
                else:
                    x = x + self.cross_attn_scale(self.cross_attn(self.cross_attend_norm(x), context = context, flex_attention_block_mask = cross_attention_block_mask, flex_attention_score_mod = cross_attention_score_mod, flash_attn_sliding_window = cross_attention_flash_sliding_window))
                    
            if self.conformer is not None:
                x = x + self.conformer_scale(self.conformer(x))

            x = self._apply_local_conditioning(x, local_add_cond, modular_local_cond)

            x = x + self.ff_scale(self.ff(self.ff_norm(x), varlen_metadata=varlen_metadata))
            

        return x

class ContinuousTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        *,
        dim_in = None,
        dim_out = None,
        dim_heads = 64,
        cross_attend=False,
        cond_token_dim=None,
        final_cross_attn_ix=-1,
        global_cond_dim=None,
        local_add_cond_dim=None,
        modular_local_cond_configs=None,
        causal=False,
        rotary_pos_emb=True,
        cross_attn_rotary_pos_emb=False,
        zero_init_branch_outputs=True,
        conformer=False,
        use_sinusoidal_emb=False,
        use_abs_pos_emb=False,
        abs_pos_emb_max_length=10000,
        num_memory_tokens=0,
        sliding_window=None,
        **kwargs
        ):

        super().__init__()

        self.dim = dim
        self.depth = depth
        self.causal = causal
        self.layers = nn.ModuleList([])

        self.project_in = nn.Linear(dim_in, dim, bias=False) if dim_in is not None else nn.Identity()
        self.project_out = nn.Linear(dim, dim_out, bias=False) if dim_out is not None else nn.Identity()

        if rotary_pos_emb:
            self.rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32))
        else:
            self.rotary_pos_emb = None

        if cross_attn_rotary_pos_emb:
            self.cross_attn_rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32))
        else:
            self.cross_attn_rotary_pos_emb = None

        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

        self.use_sinusoidal_emb = use_sinusoidal_emb
        if use_sinusoidal_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(dim)

        self.use_abs_pos_emb = use_abs_pos_emb
        if use_abs_pos_emb:
            self.pos_emb = AbsolutePositionalEmbedding(dim, abs_pos_emb_max_length + self.num_memory_tokens)

        self.global_cond_embedder = None
        if global_cond_dim is not None:
            self.global_cond_embedder = nn.Sequential(
                nn.Linear(global_cond_dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim * 6)
            )

        self.final_cross_attn_ix = final_cross_attn_ix

        self.sliding_window = sliding_window

        for i in range(depth):
            should_cross_attend = cross_attend and (self.final_cross_attn_ix == -1 or i <= (self.final_cross_attn_ix))
            self.layers.append(
                TransformerBlock(
                    dim,
                    dim_heads = dim_heads,
                    cross_attend = should_cross_attend,
                    dim_context = cond_token_dim,
                    global_cond_dim = global_cond_dim,
                    local_add_cond_dim = local_add_cond_dim,
                    modular_local_cond_configs = modular_local_cond_configs,
                    causal = causal,
                    zero_init_branch_outputs = zero_init_branch_outputs,
                    conformer=conformer,
                    layer_ix=i,
                    **kwargs
                )
            )
        
    def forward(
        self,
        x,
        context = None,
        prepend_embeds = None,
        global_cond = None,
        local_add_cond = None,
        modular_local_cond = None,
        return_info = False,
        use_checkpointing = True,
        exit_layer_ix = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        batch, seq, device = *x.shape[:2], x.device

        model_dtype = next(self.parameters()).dtype
        x = x.to(model_dtype)

        info = {
            "hidden_states": [],
        }

        x = self.project_in(x)

        if prepend_embeds is not None:
            prepend_length, prepend_dim = prepend_embeds.shape[1:]

            assert prepend_dim == x.shape[-1], 'prepend dimension must match sequence dimension'

            x = torch.cat((prepend_embeds, x), dim = -2)

        if self.num_memory_tokens > 0:
            memory_tokens = self.memory_tokens.expand(batch, -1, -1)
            x = torch.cat((memory_tokens, x), dim=1)

        if self.rotary_pos_emb is not None:
            rotary_pos_emb = self.rotary_pos_emb.forward_from_seq_len(x.shape[1])
        else:
            rotary_pos_emb = None

        if self.cross_attn_rotary_pos_emb is not None:
            cross_attn_rotary_pos_emb = self.cross_attn_rotary_pos_emb.forward_from_seq_len(context.shape[-1])
        else:
            cross_attn_rotary_pos_emb = None

        if self.use_sinusoidal_emb or self.use_abs_pos_emb:
            x = x + self.pos_emb(x)

        if global_cond is not None and self.global_cond_embedder is not None:
            global_cond = self.global_cond_embedder(global_cond)

        # Extend padding mask for prepended tokens if provided
        extended_padding_mask = None
        varlen_metadata = None
        if padding_mask is not None:
            # Compute total prepend length (memory tokens + prepend_embeds)
            prepend_length = self.num_memory_tokens
            if prepend_embeds is not None:
                prepend_length += prepend_embeds.shape[1]

            # Prepend tokens are always valid for attention
            if prepend_length > 0:
                prepend_valid = torch.ones(batch, prepend_length, device=device, dtype=torch.bool)
                extended_padding_mask = torch.cat([prepend_valid, padding_mask], dim=-1)
            else:
                extended_padding_mask = padding_mask

            # Precompute varlen metadata once for all layers (major performance optimization)
            # Only compute if varlen attention is actually available
            if flash_attn_varlen_func is not None and index_first_axis is not None:
                varlen_metadata = precompute_varlen_metadata(extended_padding_mask)

        # Iterate over the transformer layers
        for layer_ix, layer in enumerate(self.layers):

            layer_kwargs = {
                "context": context,
                "rotary_pos_emb": rotary_pos_emb,
                "cross_attn_rotary_pos_emb": cross_attn_rotary_pos_emb,
                "global_cond": global_cond,
                "local_add_cond": local_add_cond,
                "modular_local_cond": modular_local_cond,
                "self_attention_flash_sliding_window": self.sliding_window,
                "padding_mask": extended_padding_mask,
                "varlen_metadata": varlen_metadata
            }

            if use_checkpointing:
                x = checkpoint(layer, x, **layer_kwargs, **kwargs)
            else:
                x = layer(x, **layer_kwargs, **kwargs)

            if return_info:
                info["hidden_states"].append(x)

            if exit_layer_ix is not None and layer_ix == exit_layer_ix:
                x = x[:, self.num_memory_tokens:, :]

                if return_info:
                    return x, info
                
                return x

        x = x[:, self.num_memory_tokens:, :]

        x = self.project_out(x)

        if return_info:
            return x, info
        
        return x



# -----------------------------------------------------------------------------
# DiT denoiser  (models/dit.py)
# -----------------------------------------------------------------------------

class DiffusionTransformer(nn.Module):
    def __init__(self,
        io_channels=32,
        patch_size=1,
        embed_dim=768,
        cond_token_dim=0,
        project_cond_tokens=True,
        global_cond_dim=0,
        project_global_cond=True,
        input_concat_dim=0,
        prepend_cond_dim=0,
        depth=12,
        num_heads=8,
        transformer_type: tp.Literal["continuous_transformer", "mm_transformer"] = "continuous_transformer",
        global_cond_type: tp.Literal["prepend", "adaLN"] = "prepend",
        timestep_cond_type: tp.Literal["global", "input_concat"] = "global",
        timestep_embed_dim=None,
        diffusion_objective: tp.Literal["v", "rectified_flow", "rf_denoiser"] = "v",
        timestep_features_type: tp.Literal["learned", "expo"] = "learned",
        timestep_features_dim = 256,
        timestep_features_logsnr: bool = False,
        modular_local_cond_configs = None,
        **kwargs):

        super().__init__()

        self.cond_token_dim = cond_token_dim

        # Timestep embeddings
        self.timestep_cond_type = timestep_cond_type
        self.timestep_features_logsnr = timestep_features_logsnr

        timestep_features_dim = timestep_features_dim

        if timestep_features_type == "expo":
            self.timestep_features = ExpoFourierFeatures(timestep_features_dim, 0.5, 10000.0)
        else:
            self.timestep_features = FourierFeatures(1, timestep_features_dim)

        if timestep_cond_type == "global":
            timestep_embed_dim = embed_dim
        elif timestep_cond_type == "input_concat":
            assert timestep_embed_dim is not None, "timestep_embed_dim must be specified if timestep_cond_type is input_concat"
            input_concat_dim += timestep_embed_dim

        self.to_timestep_embed = nn.Sequential(
            nn.Linear(timestep_features_dim, timestep_embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(timestep_embed_dim, timestep_embed_dim, bias=True),
        )
        
        self.diffusion_objective = diffusion_objective

        if cond_token_dim > 0:
            # Conditioning tokens

            cond_embed_dim = cond_token_dim if not project_cond_tokens else embed_dim
            self.to_cond_embed = nn.Sequential(
                nn.Linear(cond_token_dim, cond_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(cond_embed_dim, cond_embed_dim, bias=False)
            )
        else:
            cond_embed_dim = 0

        if global_cond_dim > 0:
            # Global conditioning
            global_embed_dim = global_cond_dim if not project_global_cond else embed_dim
            self.to_global_embed = nn.Sequential(
                nn.Linear(global_cond_dim, global_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(global_embed_dim, global_embed_dim, bias=False)
            )

        if prepend_cond_dim > 0:
            # Prepend conditioning
            self.to_prepend_embed = nn.Sequential(
                nn.Linear(prepend_cond_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False)
            )

        self.input_concat_dim = input_concat_dim

        dim_in = io_channels + self.input_concat_dim

        self.patch_size = patch_size

        # Transformer

        self.transformer_type = transformer_type

        self.global_cond_type = global_cond_type

        transformer_dim_out = io_channels * patch_size

        if self.transformer_type == "continuous_transformer":

            global_dim = None

            if self.global_cond_type == "adaLN":
                # The global conditioning is projected to the embed_dim already at this point
                global_dim = embed_dim

            self.transformer = ContinuousTransformer(
                dim=embed_dim,
                depth=depth,
                dim_heads=embed_dim // num_heads,
                dim_in=dim_in * patch_size,
                dim_out=transformer_dim_out,
                cross_attend = cond_token_dim > 0,
                cond_token_dim = cond_embed_dim,
                global_cond_dim=global_dim,
                modular_local_cond_configs=modular_local_cond_configs,
                **kwargs
            )
      
        else:
            raise ValueError(f"Unknown transformer type: {self.transformer_type}")

        self.preprocess_conv = nn.Conv1d(dim_in, dim_in, 1, bias=False)
        nn.init.zeros_(self.preprocess_conv.weight)
        self.postprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
        nn.init.zeros_(self.postprocess_conv.weight)

    # Fixed logsnr normalization range: maps logsnr to [0, 1] preserving direction (t=0→0, t=1→1)
    _LOGSNR_MIN = -12.0
    _LOGSNR_MAX = 5.0
    _LOGSNR_RANGE = _LOGSNR_MAX - _LOGSNR_MIN

    def _t_to_logsnr_cond(self, t: torch.Tensor) -> torch.Tensor:
        """Convert t to normalized logsnr in [0, 1] for timestep conditioning.

        Maps t through logsnr = log((1-t)/t), clamps to fixed range,
        then normalizes to [0, 1] preserving direction (t=0→0, t=1→1).
        """
        t_clamped = t.float().clamp(1e-7, 1 - 1e-7)
        logsnr = torch.log((1 - t_clamped) / t_clamped)
        logsnr = logsnr.clamp(self._LOGSNR_MIN, self._LOGSNR_MAX)
        return ((self._LOGSNR_MAX - logsnr) / self._LOGSNR_RANGE).to(t.dtype)

    def _call_transformer(self, x, *, prepend_inputs=None, cross_attn_cond=None,
                         mask=None, prepend_mask=None, return_info=False,
                         exit_layer_ix=None, local_add_cond=None,
                         modular_local_cond=None, padding_mask=None,
                         extra_args=None, **kwargs):
        """Helper method to call transformer and handle early exit logic."""
        # FFN input = (B, T, C) =  (B, T_seq, 1536)
        output = self.transformer(x, prepend_embeds=prepend_inputs, context=cross_attn_cond,
                                    return_info=return_info, exit_layer_ix=exit_layer_ix,
                                    local_add_cond=local_add_cond, modular_local_cond=modular_local_cond,
                                    padding_mask=padding_mask,
                                    **(extra_args or {}), **kwargs)

        if return_info:
            output, info = output

        # Avoid postprocessing on early exit
        if exit_layer_ix is not None:
            if return_info:
                return output, info
            else:
                return output

        return (output, info) if return_info and 'info' in locals() else output

    def _forward(
        self,
        x,
        t,
        mask=None,
        cross_attn_cond=None, # prompt + seconds_start/total
        cross_attn_cond_mask=None,
        input_concat_cond=None, # this is the audio conditioning 
        local_add_cond=None,
        modular_local_cond=None,
        global_embed=None, # this is seconds_start/total embedding + timestep_embed 
        prepend_cond=None,
        prepend_cond_mask=None,
        padding_mask=None,
        return_info=False,
        exit_layer_ix=None,
        **kwargs):

        if cross_attn_cond is not None:
            cross_attn_cond = self.to_cond_embed(cross_attn_cond)

        if global_embed is not None:
            # Project the global conditioning to the embedding dimension
            global_embed = self.to_global_embed(global_embed)

        prepend_inputs = None 
        prepend_mask = None
        prepend_length = 0
        if prepend_cond is not None:
            # Project the prepend conditioning to the embedding dimension
            prepend_cond = self.to_prepend_embed(prepend_cond)
            
            prepend_inputs = prepend_cond
            if prepend_cond_mask is not None:
                prepend_mask = prepend_cond_mask

            prepend_length = prepend_cond.shape[1]

        if input_concat_cond is not None:
            # Interpolate input_concat_cond to the same length as x
            if input_concat_cond.shape[2] != x.shape[2]:
                input_concat_cond = F.interpolate(input_concat_cond, (x.shape[2], ), mode='nearest')

            x = torch.cat([x, input_concat_cond], dim=1)

        if local_add_cond is not None:
            local_add_cond = rearrange(local_add_cond, "b c t -> b t c")

        # Rearrange modular_local_cond tensors
        if modular_local_cond is not None:
            modular_local_cond = {
                k: rearrange(v, "b c t -> b t c")
                for k, v in modular_local_cond.items()
            }

        # Get the batch of timestep embeddings
        t_cond = self._t_to_logsnr_cond(t) if self.timestep_features_logsnr else t
        # Convert to model dtype for linear layers (t itself is kept in float32 for precision)
        # x has already been converted to model dtype in the outer forward() method
        t_cond = t_cond.to(x.dtype)
        timestep_embed = self.to_timestep_embed(self.timestep_features(t_cond[:, None])) # (b, embed_dim)

        # Timestep embedding is considered a global embedding. Add to the global conditioning if it exists

        if self.timestep_cond_type == "global":
            if global_embed is not None:
                global_embed = global_embed + timestep_embed
            else:
                global_embed = timestep_embed
        elif self.timestep_cond_type == "input_concat":
            x = torch.cat([x, timestep_embed.unsqueeze(2).expand(-1, -1, x.shape[2])], dim=1)

        # Add the global_embed to the prepend inputs if there is no global conditioning support in the transformer
        if self.global_cond_type == "prepend" and global_embed is not None:
            if prepend_inputs is None:
                # Prepend inputs are just the global embed, and the mask is all ones
                prepend_inputs = global_embed.unsqueeze(1)
                prepend_mask = torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)
            else:
                # Prepend inputs are the prepend conditioning + the global embed
                prepend_inputs = torch.cat([prepend_inputs, global_embed.unsqueeze(1)], dim=1)
                prepend_mask = torch.cat([prepend_mask, torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)], dim=1)

            prepend_length = prepend_inputs.shape[1]
        # C = feature channel = 64 from VAE
        # T is the down sampled 21.5 Hz feature from VAE
        x = self.preprocess_conv(x) + x

        x = rearrange(x, "b c t -> b t c")

        extra_args = {}

        if self.global_cond_type == "adaLN":
            extra_args["global_cond"] = global_embed

        if self.patch_size > 1:
            #  Split the time axis T = t·p, then fold each group of p frames' channels together
            # Shorter sequence for attention via simple rearranging
            # (c p) → c outer, p inner. Flatten = [c0p0, c0p1, c1p0, c1p1, ...]. Index = c·P + p.
            # (p c) → p outer, c inner. Flatten = [p0c0, p0c1, p1c0, p1c1, ...]. Index = p·C + c.
            x = rearrange(x, "b (t p) c -> b t (c p)", p=self.patch_size)
        
        result = self._call_transformer(
            x,
            prepend_inputs=prepend_inputs,
            cross_attn_cond=cross_attn_cond,
            mask=mask,
            prepend_mask=prepend_mask,
            return_info=return_info,
            exit_layer_ix=exit_layer_ix,
            local_add_cond=local_add_cond,
            modular_local_cond=modular_local_cond,
            padding_mask=padding_mask,
            extra_args=extra_args,
            **kwargs,
        )

        # Handle early exit (result contains both output and info)
        if exit_layer_ix is not None:
            return result

        output = result[0] if return_info else result
        if return_info:
            info = result[1]

        output = rearrange(output, "b t c -> b c t")[:,:,prepend_length:]       

        if self.patch_size > 1:
            output = rearrange(output, "b (c p) t -> b c (t p)", p=self.patch_size)

        output = self.postprocess_conv(output) + output

        if return_info:
            return output, info

        return output

    def apg_project(self, v0, v1, padding_mask=None):
        """
        Project v0 into components parallel and orthogonal to v1.

        Args:
            v0: Tensor to project (B, C, T)
            v1: Reference direction (B, C, T)
            padding_mask: Optional mask (B, T) where True = valid, False = padding.
                          If provided, only valid positions contribute to the projection.
        """
        dtype = v0.dtype
        v0, v1 = v0.double(), v1.double()

        if padding_mask is not None:
            # Expand mask to match tensor shape: (B, T) -> (B, 1, T)
            mask = padding_mask.unsqueeze(1).double()
            # Zero out padding positions for projection computation
            v0_masked = v0 * mask
            v1_masked = v1 * mask
            # Normalize only over valid positions
            v1_norm = v1_masked.norm(dim=[-1, -2], keepdim=True).clamp(min=1e-8)
            v1_normalized = v1_masked / v1_norm
            # Compute projection using masked values
            v0_parallel = (v0_masked * v1_normalized).sum(dim=[-1, -2], keepdim=True) * v1_normalized
            # Orthogonal component: subtract parallel from original (not masked) v0
            # but apply mask to ensure padding stays zero
            v0_orthogonal = (v0 - (v0 * v1_normalized).sum(dim=[-1, -2], keepdim=True) * v1_normalized) * mask
        else:
            v1 = torch.nn.functional.normalize(v1, dim=[-1, -2])
            v0_parallel = (v0 * v1).sum(dim=[-1, -2], keepdim=True) * v1
            v0_orthogonal = v0 - v0_parallel

        return v0_parallel.to(dtype), v0_orthogonal.to(dtype)

    def forward(
        self,
        x,
        t,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        negative_cross_attn_cond=None,
        negative_cross_attn_mask=None,
        input_concat_cond=None,
        local_add_cond=None,
        modular_local_cond=None,
        global_embed=None,
        negative_global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        padding_mask=None,
        cfg_scale=1.0,
        cfg_dropout_prob=0.0,
        cfg_interval = (0, 1),
        lora_interval = (0, 1),
        lora_layer_filter = "",
        lora_configs = None,
        causal=False,
        scale_phi=0.0,
        cfg_norm_threshold=0.0,
        apg_scale=1.0,
        mask=None,
        return_info=False,
        exit_layer_ix=None,
        **kwargs):

        assert not causal, "Causal mode is not supported for DiffusionTransformer"

        model_dtype = next(self.parameters()).dtype

        x = x.to(model_dtype)

        # Keep t in float32: the logsnr transform log((1-t)/t) amplifies bf16
        # quantization error ~380x near t=1, causing catastrophic conditioning errors.
        # t is a 1D batch-size tensor so float32 has zero memory impact.
        t = t.float()

        if cross_attn_cond is not None:
            cross_attn_cond = cross_attn_cond.to(model_dtype)

        if negative_cross_attn_cond is not None:
            negative_cross_attn_cond = negative_cross_attn_cond.to(model_dtype)

        if input_concat_cond is not None:
            input_concat_cond = input_concat_cond.to(model_dtype)

        if local_add_cond is not None:
            local_add_cond = local_add_cond.to(model_dtype)

        if modular_local_cond is not None:
            modular_local_cond = {k: v.to(model_dtype) for k, v in modular_local_cond.items()}

        if global_embed is not None:
            global_embed = global_embed.to(model_dtype)

        if negative_global_embed is not None:
            negative_global_embed = negative_global_embed.to(model_dtype)

        if prepend_cond is not None:
            prepend_cond = prepend_cond.to(model_dtype)

        if cross_attn_cond_mask is not None:
            cross_attn_cond_mask = cross_attn_cond_mask.bool()

            # NOTE: the padding masking feature is incompletely implemented.
            # Temporarily disabling conditioning masks due to kernel issue for flash attention.
            # See also: negative prompt masking is not implemented below either
            cross_attn_cond_mask = None

        if prepend_cond_mask is not None:
            prepend_cond_mask = prepend_cond_mask.bool()

        # Early exit bypasses CFG processing
        if exit_layer_ix is not None:
            assert self.transformer_type == "continuous_transformer", "exit_layer_ix is only supported for continuous_transformer"
            return self._forward(
                x,
                t,
                cross_attn_cond=cross_attn_cond,
                cross_attn_cond_mask=cross_attn_cond_mask,
                input_concat_cond=input_concat_cond,
                local_add_cond=local_add_cond,
                modular_local_cond=modular_local_cond,
                global_embed=global_embed,
                prepend_cond=prepend_cond,
                prepend_cond_mask=prepend_cond_mask,
                padding_mask=padding_mask,
                mask=mask,
                return_info=return_info,
                exit_layer_ix=exit_layer_ix,
                **kwargs
            )

        # CFG dropout
        if cfg_dropout_prob > 0.0 and cfg_scale == 1.0:
            if cross_attn_cond is not None:
                null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)
                dropout_mask = torch.bernoulli(torch.full((cross_attn_cond.shape[0], 1, 1), cfg_dropout_prob, device=cross_attn_cond.device)).to(torch.bool)
                cross_attn_cond = torch.where(dropout_mask, null_embed, cross_attn_cond)

            if prepend_cond is not None:
                null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)
                dropout_mask = torch.bernoulli(torch.full((prepend_cond.shape[0], 1, 1), cfg_dropout_prob, device=prepend_cond.device)).to(torch.bool)
                prepend_cond = torch.where(dropout_mask, null_embed, prepend_cond)

        if self.diffusion_objective == "v":
            sigma = torch.sin(t * math.pi / 2)
            alpha = torch.cos(t * math.pi / 2)
        elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
            sigma = t

        # Ensure sigma is indexable (handle scalar tensors from v-diffusion samplers)
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0)

        # LoRA interval
        if has_lora(self):
            if lora_configs is not None:
                # Multi-LoRA: per-LoRA interval and layer filter
                for lora_config in lora_configs:
                    idx = lora_config["lora_index"]
                    interval = lora_config.get("interval", (0, 1))
                    layer_filter = lora_config.get("layer_filter", "")
                    if interval[0] <= sigma[0] <= interval[1]:
                        enable_lora(self, lora_index=idx)
                        filter_lora_layers(self, layer_filter, lora_index=idx)
                    else:
                        disable_lora(self, lora_index=idx)
            else:
                # Legacy single-LoRA path
                if lora_interval[0] <= sigma[0] <= lora_interval[1]:
                    enable_lora(self)
                    filter_lora_layers(self, lora_layer_filter)
                else:
                    disable_lora(self)

        if cfg_scale != 1.0 and (cross_attn_cond is not None or prepend_cond is not None) and (cfg_interval[0] <= sigma[0] <= cfg_interval[1]):

            # Classifier-free guidance
            # Concatenate conditioned and unconditioned inputs on the batch dimension            
            batch_inputs = torch.cat([x, x], dim=0)
            batch_timestep = torch.cat([t, t], dim=0)

            if global_embed is not None:
                batch_global_cond = torch.cat([global_embed, global_embed], dim=0)
            else:
                batch_global_cond = None

            if input_concat_cond is not None:
                batch_input_concat_cond = torch.cat([input_concat_cond, input_concat_cond], dim=0)
            else:
                batch_input_concat_cond = None

            if local_add_cond is not None:
                batch_local_add_cond = torch.cat([local_add_cond, local_add_cond], dim=0)
            else:
                batch_local_add_cond = None

            if modular_local_cond is not None:
                batch_modular_local_cond = {k: torch.cat([v, v], dim=0) for k, v in modular_local_cond.items()}
            else:
                batch_modular_local_cond = None

            batch_cond = None
            batch_cond_masks = None
            
            # Handle CFG for cross-attention conditioning
            if cross_attn_cond is not None:

                null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)

                # For negative cross-attention conditioning, replace the null embed with the negative cross-attention conditioning
                if negative_cross_attn_cond is not None:
                    batch_cond = torch.cat([cross_attn_cond, negative_cross_attn_cond], dim=0)

                else:
                    batch_cond = torch.cat([cross_attn_cond, null_embed], dim=0)

                if cross_attn_cond_mask is not None:
                    batch_cond_masks = torch.cat([cross_attn_cond_mask, cross_attn_cond_mask], dim=0)
               
            batch_prepend_cond = None
            batch_prepend_cond_mask = None

            if prepend_cond is not None:

                null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)

                batch_prepend_cond = torch.cat([prepend_cond, null_embed], dim=0)
                           
                if prepend_cond_mask is not None:
                    batch_prepend_cond_mask = torch.cat([prepend_cond_mask, prepend_cond_mask], dim=0)
         

            if mask is not None:
                batch_masks = torch.cat([mask, mask], dim=0)
            else:
                batch_masks = None

            if padding_mask is not None:
                batch_padding_mask = torch.cat([padding_mask, padding_mask], dim=0)
            else:
                batch_padding_mask = None

            batch_output = self._forward(
                batch_inputs,
                batch_timestep,
                cross_attn_cond=batch_cond,
                cross_attn_cond_mask=batch_cond_masks,
                mask = batch_masks,
                input_concat_cond = batch_input_concat_cond,
                local_add_cond = batch_local_add_cond,
                modular_local_cond=batch_modular_local_cond,
                global_embed = batch_global_cond,
                prepend_cond = batch_prepend_cond,
                prepend_cond_mask = batch_prepend_cond_mask,
                padding_mask = batch_padding_mask,
                return_info = return_info,
                **kwargs)

            if return_info:
                batch_output, info = batch_output

            cond_output, uncond_output = torch.chunk(batch_output, 2, dim=0)

            if self.diffusion_objective == "v":
                cond_denoised = x * alpha[:, None, None] - cond_output * sigma[:, None, None]
                uncond_denoised = x * alpha[:, None, None] - uncond_output * sigma[:, None, None]

            elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
                cond_denoised = x - cond_output * sigma[:, None, None]
                uncond_denoised = x - uncond_output * sigma[:, None, None]

            diff = cond_denoised - uncond_denoised
            
            if cfg_norm_threshold > 0:
                if padding_mask is not None:
                    # Only compute norm over valid positions
                    mask = padding_mask.unsqueeze(1).float()  # (B, 1, T)
                    diff_masked = diff * mask
                    diff_norm = diff_masked.norm(p=2, dim=[-1, -2], keepdim=True)
                else:
                    diff_norm = diff.norm(p=2, dim=[-1, -2], keepdim=True)
                scale_factor = torch.minimum(torch.ones_like(diff), cfg_norm_threshold / diff_norm)
                diff *= scale_factor

            if apg_scale == 0.0:
                # Vanilla CFG: use full diff
                cfg_diff = diff
            elif apg_scale == 1.0:
                # Full APG: use only orthogonal component
                _, diff_orthogonal = self.apg_project(diff, cond_denoised, padding_mask=padding_mask)
                cfg_diff = diff_orthogonal
            else:
                # Blended APG: interpolate between full diff and orthogonal
                diff_parallel, diff_orthogonal = self.apg_project(diff, cond_denoised, padding_mask=padding_mask)
                cfg_diff = apg_scale * diff_orthogonal + (1 - apg_scale) * diff

            cfg_denoised = cond_denoised + (cfg_scale - 1) * cfg_diff
                    
            if self.diffusion_objective == "v":
                output = (x * alpha[:, None, None] - cfg_denoised) / sigma[:, None, None]
            elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
                output = (x - cfg_denoised) / sigma[:, None, None]

            # CFG Rescale
            if scale_phi != 0.0:
                cond_out_std = cond_output.std(dim=1, keepdim=True)
                out_cfg_std = output.std(dim=1, keepdim=True)
                output = scale_phi * (output * (cond_out_std/out_cfg_std)) + (1-scale_phi) * output
           
            if return_info:
                info["uncond_output"] = uncond_output
                return output, info

            return output
            
        else:
            return self._forward(
                x,
                t,
                cross_attn_cond=cross_attn_cond,
                cross_attn_cond_mask=cross_attn_cond_mask,
                input_concat_cond=input_concat_cond,
                local_add_cond=local_add_cond,
                modular_local_cond=modular_local_cond,
                global_embed=global_embed,
                prepend_cond=prepend_cond,
                prepend_cond_mask=prepend_cond_mask,
                padding_mask=padding_mask,
                mask=mask,
                return_info=return_info,
                **kwargs
            )



# -----------------------------------------------------------------------------
# LoRA helpers (dead on the SAO base T2A path; copied verbatim)  (models/lora/utils.py)
# -----------------------------------------------------------------------------

def apply_to_lora(fn):
    """apply a function to LoRAParametrization layers, designed to be used with model.apply"""

    def apply_fn(layer):
        if isinstance(layer, LoRAParametrization):
            fn(layer)

    return apply_fn

def enable_lora(model, lora_index=None):
    """Enable LoRA layers. If lora_index is None, enables all. If specified, enables only that index."""
    def _enable(layer):
        if isinstance(layer, LoRAParametrization):
            if lora_index is None or layer.lora_index == lora_index:
                layer.enable_lora()
    model.apply(_enable)

def disable_lora(model, lora_index=None):
    """Disable LoRA layers. If lora_index is None, disables all. If specified, disables only that index."""
    def _disable(layer):
        if isinstance(layer, LoRAParametrization):
            if lora_index is None or layer.lora_index == lora_index:
                layer.disable_lora()
    model.apply(_disable)

def get_lora_layers(model):
    layers = []
    for name, m in model.named_modules():
        plist = getattr(getattr(m, "parametrizations", None), "weight", None)
        if plist is None:
            continue
        for p in plist:
            if isinstance(p, LoRAParametrization):
                layers.append((name, p))
    return layers

def has_lora(model):
    """Return True if the model has at least one LoRAParametrization on any weight."""
    return len(get_lora_layers(model))>=1

def filter_lora_layers(model, lora_layer_filter, lora_index=None):
    """Enable/disable LoRA layers by name filter. If lora_index is specified, only affects that index."""
    lora_layer_filter = (lora_layer_filter or "").strip()
    # lora_layer_filter: logical OR on comma separated substrings
    # e.g. ".transformer.layers, .to_global_embed."
    def is_filtered(layer_name):
        filters = [x.lower().strip() for x in lora_layer_filter.split(",") if len(x)]
        for f in filters:
            for _f in _expand(f):
                if _f in layer_name:
                    return True
        return False
    for name, p in get_lora_layers(model):
        if lora_index is not None and p.lora_index != lora_index:
            continue
        if is_filtered(name):
            p.disable_lora()
        else:
            p.enable_lora()



# -----------------------------------------------------------------------------
# Distribution-shift schedules  (inference/sampling.py)
# -----------------------------------------------------------------------------

class IdentityDistributionShift:
    """No-op distribution shift — returns timesteps unchanged."""
    def shift(self, t: torch.Tensor, seq_len):
        return t

class FluxDistributionShift:
    """Flux/SD3/Self-Flow timestep shift: t_shifted = alpha * t / (1 + (alpha-1) * t).

    Convention: t=0 is data, t=1 is noise.
    alpha > 1 shifts timesteps toward noise, appropriate for longer sequences
    where the critical structure-from-noise transition happens at higher noise levels.

    Can be used in two ways:
    - Constant alpha: set alpha_min == alpha_max. This is how the Self-Flow paper
      (BFL, 2025) uses it, with alpha chosen per modality/autoencoder.
      Reference values from the paper: audio sampleshift=6.93, trainshift=1.0;
      video sampleshift=15.0, trainshift=2.95; images sampleshift=1.78-6.93.
    - Seq_len-dependent alpha: set different alpha_min/alpha_max. Alpha is
      interpolated log-linearly in seq_len space (power-law), following the
      SD3 derivation where alpha ∝ sqrt(seq_len).

    Args:
        min_length: Minimum sequence length (alpha = alpha_min here)
        max_length: Maximum sequence length (alpha = alpha_max here)
        alpha_min: Shift factor at min_length (1.0 = no shift)
        alpha_max: Shift factor at max_length (1.0 = no shift)
    """
    def __init__(self, min_length=256, max_length=4096,
                 alpha_min=1.0, alpha_max=1.0):
        self.min_length = min_length
        self.max_length = max_length
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        # Precompute for log-linear interpolation
        self.log_alpha_min = math.log(max(alpha_min, 1e-8))
        self.log_alpha_max = math.log(max(alpha_max, 1e-8))
        self.log_min_seq = math.log(min_length)
        self.log_max_seq = math.log(max_length)
        if self.log_max_seq == self.log_min_seq:
            self.log_max_seq += 1e-8  # prevent division by zero for constant alpha

    def get_alpha(self, seq_len: tp.Union[int, torch.Tensor]):
        """Compute alpha via log-linear interpolation in seq_len."""
        if isinstance(seq_len, torch.Tensor):
            seq_len = seq_len.float().clamp(self.min_length, self.max_length)
            log_seq = torch.log(seq_len)
            frac = (log_seq - self.log_min_seq) / (self.log_max_seq - self.log_min_seq)
            log_alpha = self.log_alpha_min + frac * (self.log_alpha_max - self.log_alpha_min)
            return torch.exp(log_alpha)
        else:
            seq_len = max(min(seq_len, self.max_length), self.min_length)
            log_seq = math.log(seq_len)
            frac = (log_seq - self.log_min_seq) / (self.log_max_seq - self.log_min_seq)
            log_alpha = self.log_alpha_min + frac * (self.log_alpha_max - self.log_alpha_min)
            return math.exp(log_alpha)

    def shift(self, t: torch.Tensor, seq_len: tp.Union[int, torch.Tensor]):
        """Shift timesteps based on sequence length.

        Args:
            t: Timesteps tensor of shape (batch_size,) or (steps,)
            seq_len: Either a scalar int (same shift for all elements) or
                     tensor of shape (batch_size,) for per-element shifts
        Returns:
            Shifted timesteps. If seq_len is a tensor and t is 1D with different size,
            returns shape (batch_size, steps) for per-element schedules.
        """
        alpha = self.get_alpha(seq_len)

        if isinstance(seq_len, torch.Tensor):
            alpha = alpha.to(t.device)
            if t.dim() == 1 and alpha.dim() == 1 and t.shape[0] != alpha.shape[0]:
                t = t.unsqueeze(0)
                alpha = alpha.unsqueeze(1)

        return alpha * t / (1 + (alpha - 1.0) * t)

class DistributionShift:
    def __init__(self, base_shift=0.5, max_shift=1.15, max_length=4096, min_length=256, use_sine=False):
        self.base_shift = base_shift
        self.max_shift = max_shift
        self.max_length = max_length
        self.min_length = min_length
        self.use_sine = use_sine

    def shift(self, t: torch.Tensor, seq_len: tp.Union[int, torch.Tensor]):
        """
        Shift timesteps based on sequence length to adjust noise schedule.

        Args:
            t: Timesteps tensor of shape (batch_size,) or (steps,)
            seq_len: Either a scalar int (same shift for all elements) or
                     tensor of shape (batch_size,) for per-element shifts
        Returns:
            Shifted timesteps. If seq_len is a tensor and t is 1D with different size,
            returns shape (batch_size, steps) for per-element schedules.
        """
        if isinstance(seq_len, torch.Tensor):
            # Per-element sequence lengths
            # Ensure seq_len is on the same device as t
            seq_len = seq_len.to(t.device)
            seq_len_clamped = seq_len.float().clamp(self.min_length, self.max_length)
            # Handle broadcasting when t and seq_len have different sizes
            if t.dim() == 1 and seq_len_clamped.dim() == 1 and t.shape[0] != seq_len_clamped.shape[0]:
                # t: (steps,) -> (1, steps), seq_len: (batch,) -> (batch, 1)
                # Result: (batch, steps)
                t = t.unsqueeze(0)
                seq_len_clamped = seq_len_clamped.unsqueeze(1)
            sigma = 1.0
            mu = - (self.base_shift + (self.max_shift - self.base_shift) * (seq_len_clamped - self.min_length) / (self.max_length - self.min_length))
            t_out = 1 - torch.exp(mu) / (torch.exp(mu) + (1 / (1 - t) - 1) ** sigma)
            if self.use_sine:
                t_out = torch.sin(t_out * math.pi / 2)
        else:
            # Scalar path (original behavior)
            seq_len = min(max(seq_len, self.min_length), self.max_length)
            sigma = 1.0
            mu = - (self.base_shift + (self.max_shift - self.base_shift) * (seq_len - self.min_length) / (self.max_length - self.min_length))
            t_out = 1 - math.exp(mu) / (math.exp(mu) + (1 / (1 - t) - 1) ** sigma)

            if self.use_sine:
                t_out = torch.sin(t_out * math.pi / 2)

        return t_out

class LogSNRShift:
    """Adaptive log-SNR distribution shift.

    Maps t∈[0,1] to log-SNR-spaced values while preserving order (0→0, 1→1).
    Equivalent to applying: logsnr = linspace(logsnr_end, logsnr_start, N)
    then t = sigmoid(-logsnr), which spaces steps uniformly in log-SNR.

    logsnr_start (the high-t bound) scales with sequence length following
    the "-1 per doubling" rule:
        logsnr_start = anchor_logsnr - rate * log₂(seq_len / anchor_length)

    This captures the empirical finding that the critical log-SNR point
    (where structure emerges from noise) drops by ~rate for each doubling
    of sequence length. logsnr_end (the low-t bound) is fixed because
    low-t refinement is purely local.
    """

    def __init__(self, anchor_length=2000, anchor_logsnr=-6.2,
                 rate=1.0, logsnr_end=2.0):
        self.anchor_length = anchor_length
        self.anchor_logsnr = anchor_logsnr
        self.rate = rate
        self.logsnr_end = logsnr_end

    def get_logsnr_start(self, seq_len):
        """Compute adaptive logsnr_start: drops by `rate` per doubling of seq_len."""
        if isinstance(seq_len, torch.Tensor):
            log2_ratio = torch.log2(seq_len.float() / self.anchor_length)
            return self.anchor_logsnr - self.rate * log2_ratio
        else:
            log2_ratio = math.log2(seq_len / self.anchor_length)
            return self.anchor_logsnr - self.rate * log2_ratio

    def shift(self, t: torch.Tensor, seq_len: tp.Union[int, torch.Tensor]):
        """Transform t∈[0,1] to log-SNR-spaced t with adaptive bounds.

        Maps through: logsnr = logsnr_end - t * (logsnr_end - logsnr_start)
                      t_out = sigmoid(-logsnr)

        Preserves order: 0→~0, 1→~1, with exact endpoint preservation.

        Args:
            t: Timesteps tensor of shape (batch_size,) or (steps,)
            seq_len: Either a scalar int or tensor of shape (batch_size,)
        Returns:
            Log-SNR-spaced timesteps in [0, 1].
        """
        t_original = t
        logsnr_start = self.get_logsnr_start(seq_len)

        if isinstance(seq_len, torch.Tensor):
            logsnr_start = logsnr_start.to(t.device)
            if t.dim() == 1 and logsnr_start.dim() == 1 and t.shape[0] != logsnr_start.shape[0]:
                t = t.unsqueeze(0)
                logsnr_start = logsnr_start.unsqueeze(1)

        # Map t through log-SNR space (monotonically: low t → high logsnr → low t_out)
        logsnr = self.logsnr_end - t * (self.logsnr_end - logsnr_start)
        t_out = torch.sigmoid(-logsnr)

        # Preserve exact endpoints
        t_out = torch.where(t_original <= 0, torch.zeros_like(t_out), t_out)
        t_out = torch.where(t_original >= 1, torch.ones_like(t_out), t_out)

        return t_out



# -----------------------------------------------------------------------------
# Diffusion model wrappers  (models/diffusion.py)
# -----------------------------------------------------------------------------

class ConditionedDiffusionModel(nn.Module):
    def __init__(self,
                *args,
                supports_cross_attention: bool = False,
                supports_input_concat: bool = False,
                supports_global_cond: bool = False,
                supports_prepend_cond: bool = False,
                **kwargs):
        super().__init__(*args, **kwargs)
        self.supports_cross_attention = supports_cross_attention
        self.supports_input_concat = supports_input_concat
        self.supports_global_cond = supports_global_cond
        self.supports_prepend_cond = supports_prepend_cond

    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                cross_attn_cond: torch.Tensor = None,
                cross_attn_mask: torch.Tensor = None,
                input_concat_cond: torch.Tensor = None,
                local_add_cond: torch.Tensor = None,
                global_embed: torch.Tensor = None,
                prepend_cond: torch.Tensor = None,
                prepend_cond_mask: torch.Tensor = None,
                cfg_scale: float = 1.0,
                cfg_dropout_prob: float = 0.0,
                batch_cfg: bool = False,
                rescale_cfg: bool = False,
                **kwargs):
        raise NotImplementedError()

class ConditionedDiffusionModelWrapper(nn.Module):
    """
    A diffusion model that takes in conditioning
    """
    def __init__(
            self,
            model: ConditionedDiffusionModel,
            conditioner: MultiConditioner,
            io_channels,
            sample_rate,
            min_input_length: int,
            diffusion_objective: tp.Literal["v", "rectified_flow", "rf_denoiser"] = "v",
            distribution_shift_options = None,
            sampling_distribution_shift_options = None,
            mask_padding_attention: bool = False,
            use_effective_length_for_schedule: bool = False,
            pretransform: tp.Optional[Pretransform] = None,
            cross_attn_cond_ids: tp.List[str] = [],
            global_cond_ids: tp.List[str] = [],
            input_concat_ids: tp.List[str] = [],
            local_add_cond_ids: tp.List[str] = [],
            modular_local_cond_ids: tp.List[str] = [],
            prepend_cond_ids: tp.List[str] = [],
            ):
        super().__init__()

        self.model = model
        self.conditioner = conditioner
        self.io_channels = io_channels
        self.sample_rate = sample_rate
        self.diffusion_objective = diffusion_objective
        self.pretransform = pretransform
        self.cross_attn_cond_ids = cross_attn_cond_ids
        self.global_cond_ids = global_cond_ids
        self.input_concat_ids = input_concat_ids
        self.local_add_cond_ids = local_add_cond_ids
        self.modular_local_cond_ids = modular_local_cond_ids
        self.prepend_cond_ids = prepend_cond_ids
        self.min_input_length = min_input_length
        self.mask_padding_attention = mask_padding_attention
        self.use_effective_length_for_schedule = use_effective_length_for_schedule

        self.dist_shift = None
        if distribution_shift_options is not None:
            self.dist_shift = self._create_dist_shift(distribution_shift_options)

        # Sampling dist_shift: separate config for inference-time schedule
        if sampling_distribution_shift_options is not None:
            self.sampling_dist_shift = self._create_dist_shift(sampling_distribution_shift_options)
        else:
            # Default: seq_len-invariant LogSNR shift matching legacy log_snr_sampling=True
            self.sampling_dist_shift = LogSNRShift(rate=0, anchor_logsnr=-6.2, logsnr_end=2.0)

    @staticmethod
    def _create_dist_shift(options: dict):
        '''
        Noise schedule must scale with sequence length. 
        Longer audio = more temporal redundancy, so a given noise level destroys less structure. 
        Fixed schedule tuned on short clips under-noises long clips
        '''
        
        """Create a distribution shift object from config options."""
        dist_shift_type = options.get("type", "full")
        dist_shift_kwargs = {k: v for k, v in options.items() if k != "type"}
        if dist_shift_type == "none":
            return IdentityDistributionShift()
        elif dist_shift_type == "flux":
            return FluxDistributionShift(**dist_shift_kwargs)
        elif dist_shift_type == "full":
            return DistributionShift(**dist_shift_kwargs)
        elif dist_shift_type == "logsnr":
            return LogSNRShift(**dist_shift_kwargs)
        else:
            raise ValueError(f"Unknown distribution shift type: {dist_shift_type}. Expected 'none', 'flux', 'full', or 'logsnr'.")     

    def get_conditioning_inputs(self, conditioning_tensors: tp.Dict[str, tp.Any], negative=False):
        '''
        Cache the text embedding , time start , time total and step embedding 
        for reuse. 
        '''
        
        cross_attention_input = None
        cross_attention_masks = None
        global_cond = None
        input_concat_cond = None
        prepend_cond = None
        prepend_cond_mask = None
        local_add_cond = None
        modular_local_cond = None

        if len(self.cross_attn_cond_ids) > 0:
            # Concatenate all cross-attention inputs over the sequence dimension
            # Assumes that the cross-attention inputs are of shape (batch, seq, channels)
            cross_attention_input = []
            cross_attention_masks = []

            for key in self.cross_attn_cond_ids:
                cross_attn_in, cross_attn_mask = conditioning_tensors[key]

                # Add sequence dimension if it's not there
                if len(cross_attn_in.shape) == 2:
                    cross_attn_in = cross_attn_in.unsqueeze(1)
                    cross_attn_mask = cross_attn_mask.unsqueeze(1)

                cross_attention_input.append(cross_attn_in)
                cross_attention_masks.append(cross_attn_mask)

            cross_attention_input = torch.cat(cross_attention_input, dim=1)
            cross_attention_masks = torch.cat(cross_attention_masks, dim=1)

        if len(self.global_cond_ids) > 0:
            # Concatenate all global conditioning inputs over the channel dimension
            # Assumes that the global conditioning inputs are of shape (batch, channels)
            global_conds = []
            for key in self.global_cond_ids:
                global_cond_input = conditioning_tensors[key][0]

                global_conds.append(global_cond_input)

            # Concatenate over the channel dimension
            global_cond = torch.cat(global_conds, dim=-1)

            if len(global_cond.shape) == 3:
                global_cond = global_cond.squeeze(1)

        if len(self.input_concat_ids) > 0:
            # Concatenate all input concat conditioning inputs over the channel dimension
            # Assumes that the input concat conditioning inputs are of shape (batch, channels, seq)
            input_concat_cond = torch.cat([conditioning_tensors[key][0] for key in self.input_concat_ids], dim=1)

        if len(self.local_add_cond_ids) > 0:
            # Concatenate all local conditioning inputs over the channel dimension
            # Assumes that the local conditioning inputs are of shape (batch, channels, seq)
            local_add_cond = torch.cat([conditioning_tensors[key][0] for key in self.local_add_cond_ids], dim=1)

        if len(self.modular_local_cond_ids) > 0:
            # Keep modular local conditioning as a dict of tensors (not concatenated)
            # Each tensor is of shape (batch, channels, seq)
            modular_local_cond = {}
            for key in self.modular_local_cond_ids:
                if key in conditioning_tensors:
                    modular_local_cond[key] = conditioning_tensors[key][0]
            # Only set if we have any conditioning
            if len(modular_local_cond) == 0:
                modular_local_cond = None

        if len(self.prepend_cond_ids) > 0:
            # Concatenate all prepend conditioning inputs over the sequence dimension
            # Assumes that the prepend conditioning inputs are of shape (batch, seq, channels)
            prepend_conds = []
            prepend_cond_masks = []

            for key in self.prepend_cond_ids:
                prepend_cond_input, prepend_cond_mask = conditioning_tensors[key]
                prepend_conds.append(prepend_cond_input)
                prepend_cond_masks.append(prepend_cond_mask)

            prepend_cond = torch.cat(prepend_conds, dim=1)
            prepend_cond_mask = torch.cat(prepend_cond_masks, dim=1)

        if negative:
            return {
                "negative_cross_attn_cond": cross_attention_input,
                "negative_cross_attn_mask": cross_attention_masks,
                "negative_global_cond": global_cond,
                "negative_input_concat_cond": input_concat_cond
            }
        else:
            return {
                "cross_attn_cond": cross_attention_input,
                "cross_attn_mask": cross_attention_masks,
                "global_cond": global_cond,
                "input_concat_cond": input_concat_cond,
                "local_add_cond": local_add_cond,
                "modular_local_cond": modular_local_cond,
                "prepend_cond": prepend_cond,
                "prepend_cond_mask": prepend_cond_mask
            }

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: tp.Dict[str, tp.Any], **kwargs):
        return self.model(x, t, **self.get_conditioning_inputs(cond), **kwargs)

    def generate(self, *args, **kwargs):
        return generate_diffusion_cond(self, *args, **kwargs)

class DiTWrapper(ConditionedDiffusionModel):
    def __init__(
        self,
        diffusion_objective: str,
        *args,
        **kwargs
    ):
        super().__init__(supports_cross_attention=True, supports_global_cond=False, supports_input_concat=False)

        self.diffusion_objective = diffusion_objective

        self.model = DiffusionTransformer(diffusion_objective=diffusion_objective, *args, **kwargs)

    def forward(self,
                x,
                t,
                cross_attn_cond=None,
                cross_attn_mask=None,
                negative_cross_attn_cond=None,
                negative_cross_attn_mask=None,
                input_concat_cond=None,
                local_add_cond=None,
                negative_input_concat_cond=None,
                global_cond=None,
                negative_global_cond=None,
                prepend_cond=None,
                prepend_cond_mask=None,
                cfg_scale=1.0,
                cfg_dropout_prob: float = 0.0,
                batch_cfg: bool = True,
                rescale_cfg: bool = False,
                scale_phi: float = 0.0,
                **kwargs):

        assert batch_cfg, "batch_cfg must be True for DiTWrapper"
        #assert negative_input_concat_cond is None, "negative_input_concat_cond is not supported for DiTWrapper"

        return self.model(
            x,
            t,
            cross_attn_cond=cross_attn_cond,
            cross_attn_cond_mask=cross_attn_mask,
            negative_cross_attn_cond=negative_cross_attn_cond,
            negative_cross_attn_mask=negative_cross_attn_mask,
            input_concat_cond=input_concat_cond,
            prepend_cond=prepend_cond,
            prepend_cond_mask=prepend_cond_mask,
            cfg_scale=cfg_scale,
            cfg_dropout_prob=cfg_dropout_prob,
            scale_phi=scale_phi,
            global_embed=global_cond,
            local_add_cond=local_add_cond,
            **kwargs)



# -----------------------------------------------------------------------------
# Number embedder for timing conditioning  (models/adp.py)
# -----------------------------------------------------------------------------

class LearnedPositionalEmbedding(nn.Module):
    """Used for continuous time"""

    def __init__(self, dim: int, std=16.0):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim) * std)

    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

def TimePositionalEmbedding(dim: int, out_features: int) -> nn.Module:
    return nn.Sequential(
        LearnedPositionalEmbedding(dim),
        nn.Linear(in_features=dim + 1, out_features=out_features),
    )

class NumberEmbedder(nn.Module):
    def __init__(
        self,
        features: int,
        dim: int = 256,
        fourier_features_type: tp.Literal["learned", "expo"] = "learned"
    ):
        super().__init__()
        self.features = features
        if fourier_features_type == "expo":
            self.embedding = nn.Sequential(ExpoFourierFeatures(dim=dim), nn.Linear(in_features=dim, out_features=features))
        else:
            self.embedding = TimePositionalEmbedding(dim=dim, out_features=features)

    def forward(self, x: Union[List[float], Tensor]) -> Tensor:
        if not torch.is_tensor(x):
            device = next(self.embedding.parameters()).device
            x = torch.tensor(x, device=device)
        assert isinstance(x, Tensor)
        shape = x.shape
        x = rearrange(x, "... -> (...)")
        embedding = self.embedding(x)
        x = embedding.view(*shape, self.features)
        return x  # type: ignore



# -----------------------------------------------------------------------------
# Conditioners  (models/conditioners.py)
# -----------------------------------------------------------------------------

class PaddingMode(str, Enum):
    """Enum for handling padding in text conditioner embeddings."""
    NONE = "none"       # No padding handling (raw embeddings with pad token)
    ZERO = "zero"       # Zero out padding positions (default)
    LEARNED = "learned" # Use learned padding embedding

class Conditioner(nn.Module):
    def __init__(
            self,
            dim: int,
            output_dim: int,
            project_out: bool = False,
            padding_mode: str = "zero"
            ):

        super().__init__()

        self.dim = dim
        self.output_dim = output_dim
        self.padding_mode = padding_mode
        self.proj_out = nn.Linear(dim, output_dim) if (dim != output_dim or project_out) else nn.Identity()

        # Learned padding embedding (only created if needed)
        if padding_mode == "learned" or padding_mode == PaddingMode.LEARNED:
            self.padding_embedding = nn.Parameter(torch.randn(output_dim) * 0.02)

    def apply_padding(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply padding handling based on padding_mode.

        Args:
            embeddings: [batch, seq_len, dim] - the embeddings to process
            attention_mask: [batch, seq_len] bool/int, True/1 = valid token

        Returns:
            embeddings with padding handled according to mode
        """
        mode = self.padding_mode
        if isinstance(mode, str):
            mode = PaddingMode(mode)

        if mode == PaddingMode.NONE:
            return embeddings
        elif mode == PaddingMode.ZERO:
            return embeddings * attention_mask.unsqueeze(-1).float()
        elif mode == PaddingMode.LEARNED:
            mask_expanded = attention_mask.unsqueeze(-1).bool()
            return torch.where(
                mask_expanded,
                embeddings,
                self.padding_embedding.unsqueeze(0).unsqueeze(0).expand_as(embeddings)
            )
        else:
            raise ValueError(f"Unknown padding mode: {mode}")

    def forward(self, x: tp.Any) -> tp.Any:
        raise NotImplementedError()

class NumberConditioner(Conditioner):
    '''
        Conditioner that takes a list of floats, normalizes them for a given range, and returns a list of embeddings
    '''
    def __init__(self,
                output_dim: int,
                min_val: float=0,
                max_val: float=1,
                fourier_features_type : tp.Literal["learned", "expo"] = "learned"
                ):
        super().__init__(output_dim, output_dim)

        self.min_val = min_val
        self.max_val = max_val

        self.embedder = NumberEmbedder(features=output_dim, fourier_features_type=fourier_features_type)

    def forward(self, floats: tp.List[float], device=None) -> tp.Any:
            self.embedder.to(device)
            # Cast the inputs to floats
            floats = [float(x) for x in floats]

            floats = torch.tensor(floats).to(device)

            floats = floats.clamp(self.min_val, self.max_val)

            normalized_floats = (floats - self.min_val) / (self.max_val - self.min_val)

            # Cast floats to same type as embedder
            embedder_dtype = next(self.embedder.parameters()).dtype
            normalized_floats = normalized_floats.to(embedder_dtype)

            float_embeds = self.embedder(normalized_floats).unsqueeze(1)

            return [float_embeds, torch.ones(float_embeds.shape[0], 1).to(device)]

class T5Conditioner(Conditioner):

    T5_MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
              "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
              "google/flan-t5-xl", "google/flan-t5-xxl", "google/t5-v1_1-xl", "google/t5-v1_1-xxl"]

    T5_MODEL_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "google/t5-v1_1-xl": 2048,
        "google/t5-v1_1-xxl": 4096,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
        "google/flan-t5-xl": 2048,
        "google/flan-t5-xxl": 4096,
    }

    def __init__(
            self,
            output_dim: int,
            t5_model_name: str = "t5-base",
            max_length: str = 128,
            enable_grad: bool = False,
            project_out: bool = False,
            padding_mode: str = "zero",
            model_path: str = None,
    ):
        assert t5_model_name in self.T5_MODELS, f"Unknown T5 model name: {t5_model_name}"
        super().__init__(self.T5_MODEL_DIMS[t5_model_name], output_dim, project_out=project_out, padding_mode=padding_mode)

        load_from = model_path or t5_model_name

        self.max_length = max_length
        self.enable_grad = enable_grad

        # Set environment variables to disable progress bars BEFORE importing transformers
        prev_hf_hub = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
        prev_transformers = os.environ.get("TRANSFORMERS_VERBOSITY")
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                from transformers import T5EncoderModel, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(load_from)
                model = T5EncoderModel.from_pretrained(load_from).train(enable_grad).requires_grad_(enable_grad).to(torch.float16)

            finally:
                logging.disable(previous_level)
                # Restore environment variables
                if prev_hf_hub is None:
                    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
                else:
                    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = prev_hf_hub
                if prev_transformers is None:
                    os.environ.pop("TRANSFORMERS_VERBOSITY", None)
                else:
                    os.environ["TRANSFORMERS_VERBOSITY"] = prev_transformers

        if self.enable_grad:
            self.model = model
        else:
            self.__dict__["model"] = model


    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:

        self.model.to(device)
        self.proj_out.to(device)

        if isinstance(texts[0], dict):
            # Pre-tokenized input (e.g. from DataLoader with tokenizers)
            input_ids = torch.stack([x["input_ids"] for x in texts]).to(device, non_blocking=True)
            attention_mask = torch.stack([x["attention_mask"] for x in texts]).to(device, non_blocking=True).to(torch.bool)
        else:
            encoded = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device).to(torch.bool)

        self.model.eval()

        with torch.amp.autocast('cuda', dtype=torch.float16) and torch.set_grad_enabled(self.enable_grad):
            embeddings = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )["last_hidden_state"]

        # Cast embeddings to same type as proj_out, unless proj_out is Identity
        if not isinstance(self.proj_out, nn.Identity):
            proj_out_dtype = next(self.proj_out.parameters()).dtype
            embeddings = embeddings.to(proj_out_dtype)

        embeddings = self.proj_out(embeddings)
        embeddings = self.apply_padding(embeddings, attention_mask)

        return embeddings, attention_mask

class MultiConditioner(nn.Module):
    """
    A module that applies multiple conditioners to an input dictionary based on the keys

    Args:
        conditioners: a dictionary of conditioners with keys corresponding to the keys of the conditioning input dictionary (e.g. "prompt")
        default_keys: a dictionary of default keys to use if the key is not in the input dictionary (e.g. {"prompt_t5": "prompt"})
    """
    def __init__(self, conditioners: tp.Dict[str, Conditioner], default_keys: tp.Dict[str, str] = {}, pre_encoded_keys: tp.List[str] = []):
        super().__init__()

        self.conditioners = nn.ModuleDict(conditioners)
        self.default_keys = default_keys
        self.pre_encoded_keys = pre_encoded_keys

    def forward(self, batch_metadata: tp.List[tp.Dict[str, tp.Any]], device: tp.Union[torch.device, str]) -> tp.Dict[str, tp.Any]:
        output = {}

        for key, conditioner in self.conditioners.items():
            condition_key = key

            conditioner_inputs = []

            for x in batch_metadata:

                if condition_key not in x:
                    if condition_key in self.default_keys:
                        condition_key = self.default_keys[condition_key]
                    else:
                        raise ValueError(f"Conditioner key {condition_key} not found in batch metadata")

                #Unwrap the condition info if it's a single-element list or tuple, this is to support collation functions that wrap everything in a list
                if isinstance(x[condition_key], list) or isinstance(x[condition_key], tuple) and len(x[condition_key]) == 1:
                    conditioner_input = x[condition_key][0]

                else:
                    conditioner_input = x[condition_key]

                conditioner_inputs.append(conditioner_input)

            if key in self.pre_encoded_keys:
                output[key] = [torch.stack(conditioner_inputs, dim=0).to(device), None]
            else:
                output[key] = conditioner(conditioner_inputs, device)

        return output

def create_multi_conditioner_from_conditioning_config(config: tp.Dict[str, tp.Any], pretransform=None) -> MultiConditioner:
    """
    Create a MultiConditioner from a conditioning config dictionary

    Args:
        config: the conditioning config dictionary
        device: the device to put the conditioners on
    """
    conditioners = {}
    cond_dim = config["cond_dim"]

    default_keys = config.get("default_keys", {})

    pre_encoded_keys = config.get("pre_encoded_keys", [])

    for conditioner_info in config["configs"]:
        id = conditioner_info["id"]

        conditioner_type = conditioner_info["type"]

        conditioner_config = {"output_dim": cond_dim}

        conditioner_config.update(conditioner_info["config"])

        if conditioner_type == "t5":
            conditioners[id] = T5Conditioner(**conditioner_config)
        elif conditioner_type == "t5gemma":
            conditioners[id] = T5GemmaConditioner(**conditioner_config)
        elif conditioner_type == "causal_lm":
            conditioners[id] = CausalLMConditioner(**conditioner_config)
        elif conditioner_type == "clap_text":
            conditioners[id] = CLAPTextConditioner(**conditioner_config)
        elif conditioner_type == "clap_audio":
            conditioners[id] = CLAPAudioConditioner(**conditioner_config)
        elif conditioner_type == "int":
            conditioners[id] = IntConditioner(**conditioner_config)
        elif conditioner_type == "number":
            conditioners[id] = NumberConditioner(**conditioner_config)
        elif conditioner_type == "list":
            conditioners[id] = ListConditioner(**conditioner_config)
        elif conditioner_type == "phoneme":
            conditioners[id] = PhonemeConditioner(**conditioner_config)
        elif conditioner_type == "lut":
            conditioners[id] = TokenizerLUTConditioner(**conditioner_config)
        elif conditioner_type == "sat_clap_text":
            from .clap import create_clap_from_config

            use_model_pretransform = conditioner_config.pop("use_model_pretransform", False)

            clap_model = create_clap_from_config(conditioner_config, pretransform=pretransform if use_model_pretransform else None)

            clap_ckpt_path = conditioner_config.get("ckpt_path", None)

            if clap_ckpt_path is not None:
                copy_state_dict(clap_model, load_ckpt_state_dict(clap_ckpt_path))

                # Ensure that loading the checkpoint doesn't overwrite the model's pretransform
                if use_model_pretransform:
                    clap_model.pretransform = pretransform

            conditioners[id] = SATCLAPTextConditioner(clap_model, **conditioner_config)

        elif conditioner_type == "sat_clap_audio":
            from .clap import create_clap_from_config

            sample_rate = conditioner_config.get("sample_rate", None)
            assert sample_rate is not None, "Sample rate must be specified for SAT-CLAP conditioners"

            use_model_pretransform = conditioner_config.pop("use_model_pretransform", False)

            clap_model = create_clap_from_config(conditioner_config, pretransform=pretransform if use_model_pretransform else None)

            clap_ckpt_path = conditioner_config.get("ckpt_path", None)

            if clap_ckpt_path is not None:
                copy_state_dict(clap_model, load_ckpt_state_dict(clap_ckpt_path))

                # Ensure that loading the checkpoint doesn't overwrite the model's pretransform
                if use_model_pretransform:
                    clap_model.pretransform = pretransform

            conditioners[id] = SATCLAPAudioConditioner(clap_model, **conditioner_config)

        elif conditioner_type == "pretransform":
            sample_rate = conditioner_config.pop("sample_rate", None)
            assert sample_rate is not None, "Sample rate must be specified for pretransform conditioners"

            use_model_pretransform = conditioner_config.pop("use_model_pretransform", False)

            if not use_model_pretransform:
                cond_pretransform = create_pretransform_from_config(conditioner_config.pop("pretransform_config"), sample_rate=sample_rate)
            else:
                assert pretransform is not None, "Model pretransform must be specified for pretransform conditioners"
                cond_pretransform = pretransform

            if conditioner_config.get("pretransform_ckpt_path", None) is not None:
                cond_pretransform.load_state_dict(load_ckpt_state_dict(conditioner_config.pop("pretransform_ckpt_path")))

            conditioners[id] = PretransformConditioner(cond_pretransform, **conditioner_config)
        elif conditioner_type == "source_mix":
            sample_rate = conditioner_config.pop("sample_rate", None)
            assert sample_rate is not None, "Sample rate must be specified for source_mix conditioners"

            use_model_pretransform = conditioner_config.pop("use_model_pretransform", False)

            if not use_model_pretransform:
                cond_pretransform = create_pretransform_from_config(conditioner_config.pop("pretransform_config"), sample_rate=sample_rate)
            else:
                assert pretransform is not None, "Model pretransform must be specified for source_mix conditioners if use_model_pretransform is True"
                cond_pretransform = pretransform

            if conditioner_config.get("pretransform_ckpt_path", None) is not None:
                cond_pretransform.load_state_dict(load_ckpt_state_dict(conditioner_config.pop("pretransform_ckpt_path")))

            conditioners[id] = SourceMixConditioner(cond_pretransform, **conditioner_config)
        else:
            raise ValueError(f"Unknown conditioner type: {conditioner_type}")

    return MultiConditioner(conditioners, default_keys=default_keys, pre_encoded_keys=pre_encoded_keys)



# -----------------------------------------------------------------------------
# Pretransform / AutoencoderPretransform moved to tokenizer_arch.py -- the codec
# seam lives beside the AudioAutoencoder it wraps. Import it from there.
# (ConditionedDiffusionModelWrapper's `pretransform: Pretransform` hint above is
#  string-ified by `from __future__ import annotations`, so no import is needed.)
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# Sampler (v-diffusion + DPM-Solver++ via k_diffusion)  (inference/sampling.py)
# -----------------------------------------------------------------------------

# Define the noise schedule and sampling loop
def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

def t_to_alpha_sigma(t):
    """Returns the scaling factors for the clean image and for the noise, given
    a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

def build_schedule(
    steps: int,
    sigma_max: float = 1.0,
    dist_shift = None,
    effective_seq_len: tp.Union[int, torch.Tensor, None] = None,
    fallback_seq_len: tp.Optional[int] = None,
    include_endpoint: bool = True,
    device: tp.Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Build a timestep schedule for diffusion sampling.

    Returns a 1D tensor of shape (N,) where N = steps+1 (if include_endpoint)
    or steps (if not), OR a 2D tensor of shape (batch_size, N) when
    effective_seq_len is a tensor and dist_shift produces per-element schedules.

    Args:
        steps: Number of sampling steps.
        sigma_max: Starting noise level (1.0 for full generation, <1.0 for variations).
        dist_shift: Optional distribution shift object (FluxDistributionShift,
            DistributionShift, LogSNRShift, etc.). Applied to warp the linear schedule.
        effective_seq_len: Sequence length for dist_shift. Scalar int or
            tensor of shape (batch_size,) for per-element schedules.
        fallback_seq_len: Fallback when effective_seq_len is None (typically x.shape[-1]).
        include_endpoint: If True, schedule includes 0 as final value (RF samplers).
            If False, excludes 0 (v-diffusion DDIM).
        device: Device for the output tensor.
    """
    n_points = steps + 1 if include_endpoint else steps

    if include_endpoint:
        t = torch.linspace(sigma_max, 0, n_points, device=device)
    else:
        t = torch.linspace(sigma_max, 0, n_points + 1, device=device)[:-1]

    if dist_shift is not None:
        seq_len = effective_seq_len if effective_seq_len is not None else fallback_seq_len
        if isinstance(seq_len, torch.Tensor):
            # Clamp per-element sequence lengths to avoid zeros causing log/NaN issues
            seq_len = torch.clamp(seq_len, min=1)
        elif seq_len is not None:
            # Clamp scalar sequence length to at least 1
            seq_len = max(int(seq_len), 1)
        t = dist_shift.shift(t, seq_len)

        # Ensure the first timestep remains aligned with sigma_max after shifting.
        # This keeps the schedule consistent with the initialization in sample_diffusion(),
        # which mixes init_data using sigma_max.
        if isinstance(t, torch.Tensor):
            sigma_max_tensor = t.new_tensor(sigma_max)
            if t.ndim == 1:
                t[0] = sigma_max_tensor
            else:
                # For batched/per-element schedules, enforce sigma_max at the first time index.
                t[..., 0] = sigma_max_tensor

    return t

def sample_v(model, x, sigmas, eta=0, callback=None, cfg_pp=False, disable_tqdm=False, **extra_args):
    """Draws samples from a model given starting noise. v-diffusion DDIM.

    Args:
        sigmas: Pre-computed schedule tensor of shape (steps,).
    """
    ts = x.new_ones([x.shape[0]])

    t = sigmas.to(x.device)
    steps = len(t)
    alphas, sigmas = get_alphas_sigmas(t)

    # The sampling loop
    for i in trange(steps, disable=disable_tqdm):

        if cfg_pp:
            # Get the model output (v, the predicted velocity)
            v, info = model(x, ts * t[i], return_info=True, **extra_args)

            if "uncond_output" in info:
                v_eps = info["uncond_output"]
            else:
                v_eps = v
        else:
            v = model(x, ts * t[i], **extra_args)
            v_eps = v

        # Predict the noise and the denoised data
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v_eps * alphas[i]

        if callback is not None:
            callback({'x': x, 't': t[i], 'sigma': sigmas[i], 'i': i, 'denoised': pred})

        # If we are not on the last timestep, compute the noisy data for the
        # next timestep.
        if i < steps - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised data in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    # If we are on the last timestep, output the denoised data
    return pred

def make_cond_model_fn(model, cond_fn):
    def cond_model_fn(x, sigma, **kwargs):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            denoised = model(x, sigma, **kwargs)
            cond_grad = cond_fn(x, sigma, denoised=denoised, **kwargs).detach()
            cond_denoised = denoised.detach() + cond_grad * K.utils.append_dims(sigma**2, x.ndim)
        return cond_denoised
    return cond_model_fn

# Uses k-diffusion from https://github.com/crowsonkb/k-diffusion
# init_data is init_audio as latents (if this is latent diffusion)
# For sampling, init_data to none
# For variations, set init_data
def sample_k(
        model_fn,
        noise,
        init_data=None,
        steps=100,
        sampler_type="dpmpp-2m-sde",
        sigma_min=0.01,
        sigma_max=100,
        rho=1.0,
        device="cuda",
        callback=None,
        cond_fn=None,
        **extra_args
    ):

    is_k_diff = sampler_type in ["k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast", "k-dpm-adaptive", "dpmpp-2m-sde", "dpmpp-3m-sde","dpmpp-2m"]
    is_v_diff = sampler_type in ["v-ddim", "v-ddim-cfgpp"]

    if is_k_diff:

        denoiser = K.external.VDenoiser(model_fn)

        if cond_fn is not None:
            denoiser = make_cond_model_fn(denoiser, cond_fn)

        # Make the list of sigmas. Sigma values are scalars related to the amount of noise each denoising step has
        sigmas = K.sampling.get_sigmas_polyexponential(steps, sigma_min, sigma_max, rho, device=device)
        # Scale the initial noise by sigma
        noise = noise * sigmas[0]

        if init_data is not None:
            # set the initial latent to the init_data, and noise it with initial sigma
            x = init_data + noise
        else:
            # SAMPLING
            # set the initial latent to noise
            x = noise


        if sampler_type == "k-heun":
            return K.sampling.sample_heun(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "k-lms":
            return K.sampling.sample_lms(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "k-dpmpp-2s-ancestral":
            return K.sampling.sample_dpmpp_2s_ancestral(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "k-dpm-2":
            return K.sampling.sample_dpm_2(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "k-dpm-fast":
            return K.sampling.sample_dpm_fast(denoiser, x, sigma_min, sigma_max, steps, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "k-dpm-adaptive":
            return K.sampling.sample_dpm_adaptive(denoiser, x, sigma_min, sigma_max, rtol=0.01, atol=0.01, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "dpmpp-2m":
            return K.sampling.sample_dpmpp_2m(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "dpmpp-2m-sde":
            return K.sampling.sample_dpmpp_2m_sde(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "dpmpp-3m-sde":
            return K.sampling.sample_dpmpp_3m_sde(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
    elif is_v_diff:

        if sigma_max > 1: # sigma_max should be between 0 and 1
            sigma_max = 1

        if cond_fn is not None:
            model_fn = make_cond_model_fn(model_fn, cond_fn)

        alpha, sigma = t_to_alpha_sigma(torch.tensor(sigma_max))

        if init_data is not None:
            x = init_data * alpha + noise * sigma
        else:
            x = noise

        if sampler_type == "v-ddim" or sampler_type == "v-ddim-cfgpp":
            use_cfg_pp = sampler_type == "v-ddim-cfgpp"
            t = build_schedule(steps=steps, sigma_max=sigma_max, include_endpoint=False, device=x.device)
            return sample_v(model_fn, x, sigmas=t, eta=0.0, cfg_pp=use_cfg_pp, callback=callback, **extra_args)
    else:
        raise ValueError(f"Unknown sampler type {sampler_type}")

@torch.no_grad()
def sample_diffusion(
    model,
    noise: torch.Tensor,
    cond_inputs: dict,
    diffusion_objective: str,
    steps: int,
    cfg_scale: float = 1.0,
    # Varlen support
    conditioning: tp.Optional[tp.List[dict]] = None,
    sample_rate: int = 44100,
    pretransform = None,
    mask_padding_attention: bool = False,
    use_effective_length_for_schedule: bool = False,
    headroom_seconds: float = 5.0,
    padding_mask: tp.Optional[torch.Tensor] = None,
    # Timestep schedule
    dist_shift = None,
    # Sampler options
    sampler_type: str = None,
    batch_cfg: bool = True,
    rescale_cfg: bool = False,
    # CFG options
    apg_scale: float = 1.0,
    # Init data (variation / img2img)
    init_data: tp.Optional[torch.Tensor] = None,
    init_noise_level: float = 1.0,
    # Other
    callback = None,
    disable_tqdm: bool = False,
    decode: bool = True,
    **sampler_kwargs
) -> torch.Tensor:
    """
    Unified sampling function for diffusion models. Handles all diffusion objectives,
    varlen support (padding_mask + effective_seq_len), timestep scheduling, and init_data
    for variation/img2img.

    Args:
        model: The diffusion model backbone (model.model, not the wrapper)
        noise: Initial noise tensor of shape (B, C, T)
        cond_inputs: Pre-processed conditioning inputs dict (merged positive + negative)
        diffusion_objective: One of "v", "rectified_flow", "rf_denoiser"
        steps: Number of sampling steps
        cfg_scale: Classifier-free guidance scale
        conditioning: List of conditioning dicts (for computing varlen from seconds_total)
        sample_rate: Audio sample rate
        pretransform: Optional pretransform for decoding latents and computing downsampling_ratio
        mask_padding_attention: Whether to create padding_mask for attention
        use_effective_length_for_schedule: Whether to use effective_seq_len for dist_shift
        padding_mask: Optional pre-computed padding mask (B, T). If provided, skips
            internal mask computation. Use this to ensure consistency with training masks.
        headroom_seconds: Extra seconds beyond seconds_total for valid region
        dist_shift: Distribution shift object for warping the timestep schedule, or None
        sampler_type: Sampler type. For RF: "euler", "rk4", "dpmpp", "pingpong".
            For v-diffusion: "v-ddim", "v-ddim-cfgpp", or k-diffusion types like "dpmpp-2m-sde".
        batch_cfg: Whether to use batched CFG
        rescale_cfg: Whether to use rescaled CFG
        apg_scale: APG (Adaptive Projected Guidance) scale. 1.0 = full APG, 0.0 = vanilla CFG
        init_data: Optional pre-encoded latent tensor for variation/img2img (shape: B, C, T)
        init_noise_level: Noise level (sigma_max) when using init_data. 1.0 = full noise (no variation).
        callback: Optional callback for progress reporting
        disable_tqdm: Whether to disable progress bar
        decode: Whether to decode latents using pretransform
        **sampler_kwargs: Additional kwargs passed to sampler

    Returns:
        Generated samples (decoded audio if decode=True, else latents)
    """
    device = noise.device
    batch_size = noise.shape[0]
    latent_seq_len = noise.shape[-1]

    # Compute downsampling ratio
    downsampling_ratio = pretransform.downsampling_ratio if pretransform is not None else 1

    # Default sampler_type per objective
    if sampler_type is None:
        sampler_type = "pingpong" if diffusion_objective == "rf_denoiser" else "euler"


    # Compute effective_seq_len for dist_shift if enabled
    effective_seq_len = None
    if use_effective_length_for_schedule and conditioning is not None:
        effective_seq_len = compute_effective_seq_len_from_conditioning(
            conditioning, sample_rate, downsampling_ratio, device
        )

    # Create padding_mask for attention if enabled (skip if pre-computed mask provided)
    if padding_mask is None and mask_padding_attention and conditioning is not None:
        raw_effective_len = compute_effective_seq_len_from_conditioning(
            conditioning, sample_rate, downsampling_ratio, device
        )
        if raw_effective_len is not None:
            headroom_tokens = int(headroom_seconds * sample_rate / downsampling_ratio)
            valid_lengths = (raw_effective_len + headroom_tokens).clamp(max=latent_seq_len).long()
            padding_mask = create_padding_mask_from_lengths(valid_lengths, latent_seq_len)

    # Determine sigma_max for schedule
    sigma_max = init_noise_level if init_data is not None else sampler_kwargs.get("sigma_max", 1.0)

    # Mix init_data with noise for variation/img2img
    # For k-diffusion v-diffusion samplers, init_data is passed through to sample_k
    # which handles mixing internally with its own sigma scaling
    k_diff_sampler_types = {"k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2",
                            "k-dpm-fast", "k-dpm-adaptive", "dpmpp-2m-sde", "dpmpp-3m-sde", "dpmpp-2m"}

    if init_data is not None:
        if diffusion_objective == "v" and sampler_type not in k_diff_sampler_types:
            # v-diffusion DDIM: pre-mix noise and init_data
            alpha, sigma = t_to_alpha_sigma(torch.tensor(sigma_max))
            noise = init_data * alpha + noise * sigma
        elif diffusion_objective in ["rectified_flow", "rf_denoiser"]:
            # RF objectives: linear interpolation
            noise = init_data * (1 - sigma_max) + noise * sigma_max

    # Build common sampler kwargs (conditioning + model-level params only).
    # disable_tqdm and callback are passed explicitly to samplers that use them,
    # not included here, to avoid leaking into model forward() calls.
    common_kwargs = {
        **cond_inputs,
        "cfg_scale": cfg_scale,
        "batch_cfg": batch_cfg,
        "rescale_cfg": rescale_cfg,
        "padding_mask": padding_mask,
        "apg_scale": apg_scale,
        **sampler_kwargs
    }

    # Sample based on diffusion objective
    if diffusion_objective == "v":
        if sampler_type in k_diff_sampler_types or sampler_type in ["v-ddim", "v-ddim-cfgpp"]:
            # Route through sample_k which handles k-diffusion and v-ddim samplers
            # sample_k uses its own schedule (polyexponential for k-diff, internal for v-ddim)
            k_init_data = init_data if sampler_type in k_diff_sampler_types else None

            # Determine sigma_max for sample_k:
            # - k-diffusion samplers: default 100 for schedule, or init_noise_level for variations
            # - v-ddim: sigma_max is the noise level (0-1), our sigma_max variable
            if sampler_type in k_diff_sampler_types:
                k_sigma_max = sigma_max if init_data is not None else common_kwargs.pop("sigma_max", 100)
            else:
                k_sigma_max = sigma_max  # v-ddim: already 0-1 range
            # Pop sigma_max from common_kwargs to avoid passing it twice
            common_kwargs.pop("sigma_max", None)

            sampled = sample_k(
                model, noise,
                init_data=k_init_data,
                steps=steps,
                sampler_type=sampler_type,
                sigma_min=common_kwargs.pop("sigma_min", 0.01),
                sigma_max=k_sigma_max,
                rho=common_kwargs.pop("rho", 1.0),
                device=device,
                callback=callback,
                **common_kwargs
            )
        else:
            # DDIM-style sampler with pre-computed schedule
            t = build_schedule(
                steps=steps, sigma_max=sigma_max,
                dist_shift=dist_shift, effective_seq_len=effective_seq_len,
                fallback_seq_len=latent_seq_len, include_endpoint=False, device=device
            )
            sampled = sample_v(model, noise, sigmas=t, callback=callback, disable_tqdm=disable_tqdm, **common_kwargs)

    elif diffusion_objective in ["rectified_flow", "rf_denoiser"]:
        # Remove v-diffusion-specific kwargs that don't apply to RF
        common_kwargs.pop("sigma_min", None)
        common_kwargs.pop("sigma_max", None)
        common_kwargs.pop("rho", None)

        # Build schedule
        sigmas = build_schedule(
            steps=steps, sigma_max=sigma_max,
            dist_shift=dist_shift, effective_seq_len=effective_seq_len,
            fallback_seq_len=latent_seq_len, include_endpoint=True, device=device
        )

        # Route to sampler
        if sampler_type == "euler":
            sampled = sample_discrete_euler(model, noise, sigmas=sigmas, callback=callback, disable_tqdm=disable_tqdm, **common_kwargs)
        elif sampler_type == "rk4":
            sampled = sample_rk4(model, noise, sigmas=sigmas, callback=callback, disable_tqdm=disable_tqdm, **common_kwargs)
        elif sampler_type == "dpmpp":
            sampled = sample_flow_dpmpp(model, noise, sigmas=sigmas, callback=callback, disable_tqdm=disable_tqdm, **common_kwargs)
        elif sampler_type == "pingpong":
            sampled = sample_flow_pingpong(model, noise, sigmas=sigmas, callback=callback, disable_tqdm=disable_tqdm, **common_kwargs)
        else:
            raise ValueError(f"Unknown sampler_type for {diffusion_objective}: {sampler_type}")

    else:
        raise ValueError(f"Unknown diffusion_objective: {diffusion_objective}")

    # Decode if requested
    if decode and pretransform is not None:
        sampled = sampled.to(next(pretransform.parameters()).dtype)
        sampled = pretransform.decode(sampled)

        # Zero out audio beyond valid region (padding positions decode to garbage)
        if padding_mask is not None:
            audio_mask = padding_mask.unsqueeze(1).repeat_interleave(downsampling_ratio, dim=-1)
            # Trim or pad to match sampled length
            if audio_mask.shape[-1] > sampled.shape[-1]:
                audio_mask = audio_mask[..., :sampled.shape[-1]]
            elif audio_mask.shape[-1] < sampled.shape[-1]:
                audio_mask = torch.nn.functional.pad(audio_mask, (0, sampled.shape[-1] - audio_mask.shape[-1]), value=False)
            sampled = sampled * audio_mask.to(sampled.dtype)

    return sampled



# -----------------------------------------------------------------------------
# Generation entry point  (inference/generation.py)
# -----------------------------------------------------------------------------

def generate_diffusion_cond(
        model,
        steps: int = 250,
        cfg_scale=6,
        conditioning: dict = None,
        conditioning_tensors: tp.Optional[dict] = None,
        negative_conditioning: dict = None,
        negative_conditioning_tensors: tp.Optional[dict] = None,
        batch_size: int = 1,
        sample_size: int = 2097152,
        sample_rate: int = 48000,
        seed: int = -1,
        device: str = "cuda",
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        return_latents = False,
        inversion_params: dict = None,
        adapt_duration_to_conditioning: tp.Optional[bool] = None,
        duration_padding_sec: float = 6.0,
        use_effective_length_for_schedule: tp.Optional[bool] = None,
        mask_padding_attention: tp.Optional[bool] = None,
        apg_scale: float = 1.0,
        dist_shift = None,
        **sampler_kwargs
        ) -> torch.Tensor:
    """
    Generate audio from a prompt using a diffusion model.

    Args:
        model: The diffusion model to use for generation.
        steps: The number of diffusion steps to use.
        cfg_scale: Classifier-free guidance scale
        conditioning: A dictionary of conditioning parameters to use for generation.
        conditioning_tensors: A dictionary of precomputed conditioning tensors to use for generation.
        batch_size: The batch size to use for generation.
        sample_size: The length of the audio to generate, in samples.
        sample_rate: The sample rate of the audio to generate (Deprecated, now pulled from the model directly)
        seed: The random seed to use for generation, or -1 to use a random seed.
        device: The device to use for generation.
        init_audio: A tuple of (sample_rate, audio) to use as the initial audio for generation.
        init_noise_level: The noise level to use when generating from an initial audio sample.
        return_latents: Whether to return the latents used for generation instead of the decoded audio.
        adapt_duration_to_conditioning: Adapt sample size based on seconds_total + duration_padding_sec.
            None (default) = auto-detect from model; True/False = explicit override.
        duration_padding_sec: Extra seconds to add when adapting duration (default 6.0).
        use_effective_length_for_schedule: Use effective_seq_len for distribution shift.
            None (default) = auto-detect from model; True/False = explicit override.
        mask_padding_attention: Create padding_mask for attention based on effective_seq_len.
            None (default) = auto-detect from model; True/False = explicit override.
        apg_scale: APG (Adaptive Projected Guidance) scale. 1.0 = full APG, 0.0 = vanilla CFG.
        dist_shift: Optional distribution shift override for sampling. If None, uses model.sampling_dist_shift.
        **sampler_kwargs: Additional keyword arguments to pass to the sampler.
    """

    if mask_padding_attention is None:
        mask_padding_attention = getattr(model, 'mask_padding_attention', False)
    if use_effective_length_for_schedule is None:
        use_effective_length_for_schedule = getattr(model, 'use_effective_length_for_schedule', False)
    if adapt_duration_to_conditioning is None:
        adapt_duration_to_conditioning = mask_padding_attention or use_effective_length_for_schedule

    # The length of the output in audio samples 
    audio_sample_size = sample_size

    # Optionally adapt sample size based on seconds_total conditioning
    if adapt_duration_to_conditioning and conditioning is not None:
        # Find the maximum seconds_total across the batch
        max_seconds = 0.0
        for cond_dict in conditioning:
            if "seconds_total" in cond_dict:
                max_seconds = max(max_seconds, cond_dict["seconds_total"])
        
        if max_seconds > 0:
            # Calculate target audio samples with padding, capped at original sample_size
            target_audio_samples = int((max_seconds + duration_padding_sec) * model.sample_rate)
            
            # Ensure we align to pretransform downsampling ratio if applicable
            if model.pretransform is not None:
                # Round up to nearest multiple of downsampling ratio, and also align to chunk size if using chunked attention
                ds_ratio = model.pretransform.downsampling_ratio
                latent_align = 1
                # Only perform if the encoder has chunked attention layers and not sliding window 
                if (hasattr(model.pretransform, 'model') and
                        hasattr(model.pretransform.model, 'encoder')):
                    encoder = model.pretransform.model.encoder
                    if hasattr(encoder, 'layers'):
                        first_chunked_index = next(
                            (i for i, l in enumerate(encoder.layers)
                             if hasattr(l, 'chunk_size') and getattr(l, 'sliding_window_latents', True) is None), None
                        )
                        if first_chunked_index is not None:
                            first_chunked = encoder.layers[first_chunked_index]
                            stride = getattr(first_chunked, 'stride', None)
                            if stride is None and hasattr(encoder, 'strides') and len(encoder.strides) > first_chunked_index:
                                stride = encoder.strides[first_chunked_index]
                            if stride and stride > 0:
                                latent_align = max(1, first_chunked.chunk_size // stride)
                align = ds_ratio * latent_align  # For chunked attention, align to chunk size after downsampling
                # Round up to nearest multiple
                target_audio_samples = ((target_audio_samples + align - 1) // align) * align
            
            # Cap at original sample_size (don't exceed what was requested)
            audio_sample_size = min(target_audio_samples, sample_size)

    # Use the (potentially adapted) audio_sample_size for latent size calculation
    latent_sample_size = audio_sample_size

    # If this is latent diffusion, change sample_size instead to the downsampled latent size
    if model.pretransform is not None:
        latent_sample_size = audio_sample_size // model.pretransform.downsampling_ratio
        
    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)
    torch.manual_seed(seed)
    # Define the initial noise immediately after setting the seed
    noise = torch.randn([batch_size, model.io_channels, latent_sample_size], device=device)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False

    # Conditioning
    assert conditioning is not None or conditioning_tensors is not None, "Must provide either conditioning or conditioning_tensors"
    if conditioning_tensors is None:
        conditioning_tensors = model.conditioner(conditioning, device)
    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)

    if negative_conditioning is not None or negative_conditioning_tensors is not None:
        
        if negative_conditioning_tensors is None:
            negative_conditioning_tensors = model.conditioner(negative_conditioning, device)
            
        negative_conditioning_tensors = model.get_conditioning_inputs(negative_conditioning_tensors, negative=True)
    else:
        negative_conditioning_tensors = {}

    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)
    conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in conditioning_inputs.items()}

    diff_objective = model.diffusion_objective

    if init_audio is not None:
        # The user supplied some initial audio (for inpainting or variation or inversion). Let us prepare the input audio.
        in_sr, init_audio = init_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)

        init_audio = init_audio.repeat(batch_size, 1, 1)

        if inversion_params is not None:
            # we're doing an inversion task
            assert diff_objective in diff_objective in ["rectified_flow"], "inversion is supported in RF models only"
            # RF-Inversion
            inversion_noise = noise # this is not the inverted latent, this is gamma-parameterized noise we use during inversion to guide it in-distribution 
            # Modify the conditioning for inversion
            inversion_conditioning = copy.deepcopy(conditioning)
            inversion_conditioning_tensors = model.conditioner(inversion_conditioning, device)
            inversion_conditioning_inputs = model.get_conditioning_inputs(inversion_conditioning_tensors)
            inversion_conditioning_inputs =  {k: v.type(model_dtype) if v is not None else v for k, v in inversion_conditioning_inputs.items()}
            if inversion_params["inversion_unconditional"]:
                # Unconditional seems better for prompt re-stylization
                cfg_dropout_prob = 1.0
            else:
                # "" prompt w/ cfg=1 seems better for reconstruction
                cfg_dropout_prob = 0.0
                for x in inversion_conditioning:
                    if "prompt" in x: x["prompt"] = ""
            # invert the audio
            inverted_latents = invert_audio(model.model, init_audio, noise=inversion_noise, \
                inversion_params=inversion_params,
                cfg_dropout_prob=cfg_dropout_prob,
                **inversion_conditioning_inputs)
            noise = inverted_latents # from here on, use this as the seed noise. it represents the inverted audio
            init_audio = None

    # Merge positive and negative conditioning inputs
    cond_inputs = {**conditioning_inputs, **negative_conditioning_tensors}

    # Extract sampler_type from sampler_kwargs if present
    sampler_type = sampler_kwargs.pop("sampler_type", "euler")

    sampled = sample_diffusion(
        model=model.model,
        noise=noise,
        cond_inputs=cond_inputs,
        diffusion_objective=diff_objective,
        steps=steps,
        cfg_scale=cfg_scale,
        # Varlen support
        conditioning=conditioning,
        sample_rate=model.sample_rate,
        pretransform=model.pretransform,
        mask_padding_attention=mask_padding_attention,
        use_effective_length_for_schedule=use_effective_length_for_schedule,
        headroom_seconds=duration_padding_sec,
        # Timestep schedule
        dist_shift=dist_shift if dist_shift is not None else model.sampling_dist_shift,
        # Sampler options
        sampler_type=sampler_type,
        batch_cfg=True,
        rescale_cfg=True,
        apg_scale=apg_scale,
        # Init data
        init_data=init_audio,
        init_noise_level=init_noise_level,
        # Other
        decode=not return_latents,
        **sampler_kwargs
    )

    return sampled

