import torch
import numpy as np


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, pe_interpolation=1.0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32) / pe_interpolation
    grid_w = np.arange(grid_size, dtype=np.float32) / pe_interpolation
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def expand_t(t, x):
    """Function to reshape time t to broadcastable dimension of x
    Args:
        t: [bsz,], time vector
        x: [bsz,...], data point
    """
    dims = [1] * (len(x.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t


def randn_tensor(shape, noise_repeat, device, dtype=torch.float32):
    bsz = shape[0]
    if bsz % noise_repeat != 0:
        raise ValueError(f"Batch size ({bsz}) must be divisible by noise repeat ({noise_repeat})")
    _shape = (noise_repeat,) + shape[1:]
    _tensor = torch.randn(_shape, device=device, dtype=dtype).repeat(bsz // noise_repeat, 1)
    return _tensor


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def identity(input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    return input


def rms_norm(
    input: torch.Tensor,
    normalized_shape: torch.Size,
    eps: float = 1e-6,
    ) -> torch.Tensor:
    dtype = input.dtype
    input = input.to(torch.float32)
    variance = input.flatten(-len(normalized_shape)).pow(2).mean(dim=-1)[(...,) + (None,) * len(normalized_shape)]
    input = input * torch.rsqrt(variance + eps)
    return input.to(dtype)


def layer_norm(
    input: torch.Tensor,
    normalized_shape: torch.Size,
    eps: float = 1e-6,
    ) -> torch.Tensor:
    dtype = input.dtype
    input = input.to(torch.float32)
    mean = input.flatten(-len(normalized_shape)).mean(dim=-1)[(...,) + (None,) * len(normalized_shape)]
    variance = (input - mean).flatten(-len(normalized_shape)).pow(2).mean(dim=-1)[(...,) + (None,) * len(normalized_shape)]
    input = (input - mean) * torch.rsqrt(variance + eps)
    return input.to(dtype)