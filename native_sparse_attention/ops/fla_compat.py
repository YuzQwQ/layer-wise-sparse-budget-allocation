"""Fallback utilities for ``native_sparse_attention.ops.parallel``.

The fixed-length training path only needs differentiable mean pooling plus a
few sequence-offset helpers. These PyTorch implementations avoid requiring the
full flash-linear-attention package on smoke/formal experiment hosts.
"""

from __future__ import annotations

import functools
from typing import Callable, Optional

import torch
import triton


def _identity_decorator(fn):
    return fn


autocast_custom_fwd = _identity_decorator
autocast_custom_bwd = _identity_decorator


def contiguous(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        args = tuple(arg.contiguous() if isinstance(arg, torch.Tensor) else arg for arg in args)
        kwargs = {key: value.contiguous() if isinstance(value, torch.Tensor) else value for key, value in kwargs.items()}
        return fn(*args, **kwargs)

    return wrapper


def prepare_lens(offsets: torch.LongTensor) -> torch.LongTensor:
    return offsets[1:] - offsets[:-1]


def prepare_token_indices(offsets: torch.LongTensor) -> torch.LongTensor:
    rows = []
    for seq_idx, length in enumerate(prepare_lens(offsets).tolist()):
        if length > 0:
            pos = torch.arange(length, device=offsets.device, dtype=offsets.dtype)
            seq = torch.full_like(pos, seq_idx)
            rows.append(torch.stack([seq, pos], dim=1))
    if not rows:
        return offsets.new_empty((0, 2))
    return torch.cat(rows, dim=0)


def prepare_chunk_offsets(offsets: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    chunks = torch.div(prepare_lens(offsets) + chunk_size - 1, chunk_size, rounding_mode="floor")
    return torch.cat([offsets.new_zeros(1), chunks]).cumsum(-1)


def prepare_chunk_indices(offsets: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    rows = []
    for seq_idx, n_chunks in enumerate(torch.div(prepare_lens(offsets) + chunk_size - 1, chunk_size, rounding_mode="floor").tolist()):
        if n_chunks > 0:
            chunk = torch.arange(n_chunks, device=offsets.device, dtype=offsets.dtype)
            seq = torch.full_like(chunk, seq_idx)
            rows.append(torch.stack([seq, chunk], dim=1))
    if not rows:
        return offsets.new_empty((0, 2))
    return torch.cat(rows, dim=0)


def _mean_pool_fixed(x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    chunks = []
    for start in range(0, x.shape[1], chunk_size):
        chunks.append(x[:, start : start + chunk_size].mean(dim=1))
    return torch.stack(chunks, dim=1)


def mean_pooling(
    x: torch.Tensor,
    chunk_size: int,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
) -> torch.Tensor:
    if head_first:
        x = x.transpose(1, 2)

    if cu_seqlens is None:
        out = _mean_pool_fixed(x, chunk_size)
    else:
        if x.shape[0] != 1:
            raise ValueError("cu_seqlens fallback expects flattened variable-length inputs with batch size 1.")
        pooled = []
        flat = x.squeeze(0)
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
            seq = flat[start:end].unsqueeze(0)
            pooled.append(_mean_pool_fixed(seq, chunk_size).squeeze(0))
        max_chunks = max((item.shape[0] for item in pooled), default=0)
        out = x.new_zeros((1, max_chunks, x.shape[2], x.shape[3]))
        cursor = 0
        total_chunks = int(prepare_chunk_offsets(cu_seqlens, chunk_size)[-1].item())
        out = x.new_empty((1, total_chunks, x.shape[2], x.shape[3]))
        for item in pooled:
            out[:, cursor : cursor + item.shape[0]] = item.unsqueeze(0)
            cursor += item.shape[0]

    if head_first:
        out = out.transpose(1, 2)
    return out
