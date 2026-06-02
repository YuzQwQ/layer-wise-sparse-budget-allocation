"""Small PyTorch fallbacks for the FLA modules used by the NSA model.

These classes keep the training entrypoint self-contained on machines where
flash-linear-attention is not installed. They intentionally implement only the
APIs used by ``modeling_nsa.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class Cache:
    is_compileable = True

    def __init__(self, seen_tokens: int = 0) -> None:
        self.states: List[Dict[str, Any]] = []
        self._seen_tokens = seen_tokens

    @classmethod
    def from_legacy_cache(cls, legacy_cache=None) -> "Cache":
        cache = cls()
        if legacy_cache:
            for attn_state in legacy_cache:
                cache.states.append({"attn_state": attn_state})
        return cache

    def __getitem__(self, layer_idx: int) -> Dict[str, Any]:
        return self.states[layer_idx]

    def __iter__(self):
        return iter(self.states)

    def __len__(self) -> int:
        return len(self.states)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.states):
            return 0
        attn_state = self.states[layer_idx].get("attn_state")
        if attn_state is None:
            return 0
        return int(attn_state[0].shape[-2])

    def update(
        self,
        recurrent_state: torch.Tensor = None,
        attn_state: Tuple[torch.Tensor, torch.Tensor] = None,
        conv_state: Tuple[torch.Tensor] = None,
        ffn_state: torch.Tensor = None,
        layer_idx: int = 0,
        offset: Optional[int] = 1,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if layer_idx == 0:
            self._seen_tokens += int(offset or 0)

        while len(self.states) <= layer_idx:
            self.states.append({})

        state = self.states[layer_idx]
        if recurrent_state is not None:
            state["recurrent_state"] = recurrent_state
        if conv_state is not None:
            state["conv_state"] = conv_state
        if ffn_state is not None:
            state["ffn_state"] = ffn_state
        if attn_state is not None:
            window_size = (cache_kwargs or {}).get("window_size")
            if state.get("attn_state") is None:
                k_state, v_state = attn_state
            else:
                k_prev, v_prev = state["attn_state"]
                k_state = torch.cat([k_prev, attn_state[0]], dim=-2)
                v_state = torch.cat([v_prev, attn_state[1]], dim=-2)
            if window_size is not None and window_size > 0:
                k_state = k_state[..., -window_size:, :].contiguous()
                v_state = v_state[..., -window_size:, :].contiguous()
            state["attn_state"] = (k_state, v_state)
        return state


class FusedCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, inplace_backward: bool = False, ignore_index: int = -100, **kwargs: Any) -> None:
        super().__init__(ignore_index=ignore_index, **kwargs)
        self.inplace_backward = inplace_backward


class FusedLinearCrossEntropyLoss(nn.Module):
    ignore_index = -100

    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = F.linear(hidden_states, weight, bias)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=self.ignore_index)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, **_: Any) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        prenorm: bool = False,
        **_: Any,
    ):
        if residual is not None:
            x = x + residual
        residual_out = x
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        out = x * torch.rsqrt(variance + self.eps).to(x.dtype)
        out = out * self.weight.to(dtype=out.dtype)
        return (out, residual_out) if prenorm else out


def _swiglu(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * value


class GatedMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_act: str = "swish",
        fuse_swiglu: bool = True,
    ) -> None:
        super().__init__()
        if hidden_act != "swish":
            raise ValueError(f"Unsupported hidden_act for fallback GatedMLP: {hidden_act}")
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 255) // 256)
        self.hidden_size = hidden_size
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.fuse_swiglu = fuse_swiglu
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.down_proj(_swiglu(self.gate_proj(x), self.up_proj(x)))


def _rotate_half(x: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
    if interleaved:
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool) -> torch.Tensor:
    if interleaved:
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)
    else:
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
    if cos.ndim == 2:
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
    elif cos.ndim == 3:
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
    return (x * cos) + (_rotate_half(x, interleaved) * sin)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        scale_base: Optional[float] = None,
        interleaved: bool = False,
        pos_idx_in_fp32: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if scale_base is not None:
            raise ValueError("Fallback RotaryEmbedding does not implement XPos scaling.")
        self.dim = dim
        self.base = float(base)
        self.scale_base = scale_base
        self.interleaved = interleaved
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def reset_parameters(self) -> None:
        with torch.no_grad():
            inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, device=self.inv_freq.device, dtype=torch.float32) / self.dim)
            )
            self.inv_freq.copy_(inv_freq)

    def _cos_sin(
        self,
        seqlen: int,
        device: torch.device,
        dtype: torch.dtype,
        seqlen_offset: Union[int, torch.Tensor] = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(seqlen_offset, torch.Tensor):
            positions = torch.arange(seqlen, device=device, dtype=torch.float32).unsqueeze(0)
            positions = positions + seqlen_offset.to(device=device, dtype=torch.float32).unsqueeze(1)
            freqs = torch.einsum("bt,d->btd", positions, self.inv_freq.to(device=device))
            return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)
        positions = torch.arange(seqlen, device=device, dtype=torch.float32) + float(seqlen_offset)
        freqs = torch.outer(positions, self.inv_freq.to(device=device))
        return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seqlen_offset: Union[int, torch.Tensor] = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cu_seqlens is not None:
            raise ValueError("Fallback RotaryEmbedding does not support cu_seqlens.")
        seqlen = q.shape[1] if max_seqlen is None else max_seqlen
        cos, sin = self._cos_sin(seqlen, q.device, q.dtype, seqlen_offset)
        if max_seqlen is not None:
            cos = cos[: q.shape[1]]
            sin = sin[: q.shape[1]]
        return _apply_rotary(q, cos, sin, self.interleaved), _apply_rotary(k, cos, sin, self.interleaved)
