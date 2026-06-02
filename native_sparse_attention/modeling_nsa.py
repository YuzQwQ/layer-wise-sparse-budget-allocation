# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

try:
    from fla.models.utils import Cache
    from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss
    from fla.modules import GatedMLP as NSAMLP
    from fla.modules import RMSNorm, RotaryEmbedding
except ImportError:
    from native_sparse_attention.fla_compat import Cache
    from native_sparse_attention.fla_compat import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss
    from native_sparse_attention.fla_compat import GatedMLP as NSAMLP
    from native_sparse_attention.fla_compat import RMSNorm, RotaryEmbedding
from native_sparse_attention.configuration_nsa import NSAConfig
from native_sparse_attention.ops.parallel import (
    mean_pooling,
    parallel_nsa,
    parallel_nsa_compression,
    parallel_nsa_topk,
)

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

logger = logging.get_logger(__name__)


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    """Straight-Through Estimator for rounding.

    Forward : returns x.round() (identical numerical value).
    Backward: passes gradient through as if round() were identity.

    Used for budget scalars so that the continuous budget_signal retains a
    meaningful gradient path while the kernel receives integer block counts.
    """
    return x + (x.round() - x).detach()


class NativeSparseAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 64,
        num_kv_heads: Optional[int] = 4,
        head_dim: int = 64,
        qkv_bias: bool = False,
        block_size: Optional[int] = 64,
        block_counts: Optional[Union[torch.LongTensor, int]] = 16,
        window_size: Optional[int] = 512,
        rope_theta: Optional[float] = 10000.,
        max_position_embeddings: Optional[int] = None,
        layer_idx: int = None,
        # Budget allocation fields ----------------------------------------
        budget_mode: str = "uniform",
        budget_min_k: int = 4,
        budget_max_k: int = 32,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.head_dim = head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias

        self.block_size = block_size
        self.block_counts = block_counts      # int (uniform / layer_aware already resolved)
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx

        self.budget_mode = budget_mode
        self.budget_min_k = budget_min_k
        self.budget_max_k = budget_max_k
        self.budget_shuffle_mode: str = "none"
        self.budget_center_mode: str = "none"
        self.budget_random_mode: str = "none"
        self.budget_clamp_mode: str = "none"
        self.budget_replay_values: Optional[torch.Tensor] = None
        self.budget_replay_probs: Optional[torch.Tensor] = None
        self._budget_tensor: Optional[torch.Tensor] = None
        self._dynamic_k_tensor: Optional[torch.Tensor] = None
        self._debug_record_selection: bool = False
        self._debug_record_selection_tensors: bool = False
        self._debug_selection_state: Optional[Dict[str, object]] = None

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.g_proj = nn.Linear(self.hidden_size, self.num_heads * 3, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)

        # ------------------------------------------------------------------
        # Phase 2A / 2C – independent budget head (validation tool)
        # Projects hidden states to a per-KV-head scalar budget in [0, 1].
        # For query_aware_asymm, only created when budget_min_k < budget_max_k
        # (i.e. the layer is actually dynamic; fixed layers skip this).
        # ------------------------------------------------------------------
        if budget_mode == "query_aware_tool" or (
            budget_mode == "query_aware_asymm" and budget_min_k < budget_max_k
        ):
            self.budget_head = nn.Linear(self.hidden_size, self.num_kv_heads, bias=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_dynamic_block_counts(
        self,
        hidden_states: torch.Tensor,
        g_slc: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> Tuple[torch.LongTensor, Optional[torch.Tensor]]:
        """Return (integer block_counts for kernel, float budget tensor for aux loss).

        Phase 1 / uniform
        -----------------
        Returns the pre-configured self.block_counts (int scalar) directly.

        Phase 2A – query_aware_tool
        ---------------------------
        An independent budget_head maps hidden_states → budget_signal ∈ [0,1].
        STE is applied so that the float budget stays in the autograd graph,
        giving budget_head a gradient path through the task loss even though
        the kernel only sees the detached integer version.

        Phase 2B – query_aware_gate (core contribution)
        ------------------------------------------------
        g_slc (shape: B, T, HQ) is reduced over the GQA group dimension to
        obtain a per-KV-head scalar in [0, 1].  This scalar is mapped to an
        integer block count via the same STE mechanism.  Crucially, g_slc is
        NOT removed from the post-fusion path in parallel_nsa – it continues
        to weight the selection-path output.  This preserves the gradient
        path: the task loss → attention output → g_slc * o_slc → g_slc,
        so g_slc receives gradient signal to learn BOTH roles simultaneously.

        The research question is whether the model can learn a g_slc that
        simultaneously encodes "how much should selection contribute to the
        output" AND "how many blocks should be selected", and whether this
        joint allocation leads to a better accuracy–compute tradeoff than the
        original single-role gate.

        Phase 2C – query_aware_asymm
        ----------------------------
        Uses an independent budget_head (same as 2A) but only for layers where
        budget_min_k < budget_max_k.  Layers where min==max are treated as
        fixed (like layer_aware), effectively pinning sensitive high layers to
        their mono_inc base value while allowing low/mid layers to be dynamic.
        This isolates whether query-aware adjustment in low-sensitivity layers
        alone is beneficial, without risking damage to high-sensitivity layers.
        """
        if self.budget_mode in ("uniform", "layer_aware"):
            return self.block_counts, None

        def sample_replay_budget(reference: torch.Tensor) -> Optional[torch.Tensor]:
            budget_random_mode = getattr(self, "budget_random_mode", "none")
            if budget_random_mode == "none":
                return None
            if budget_random_mode != "layer_histogram_replay":
                raise ValueError(f"Unknown budget_random_mode: {budget_random_mode!r}")
            values = getattr(self, "budget_replay_values", None)
            probs = getattr(self, "budget_replay_probs", None)
            if values is None or probs is None:
                raise RuntimeError(
                    f"Missing replay profile for layer {self.layer_idx}; "
                    "budget_random_mode='layer_histogram_replay' requires per-layer values/probs."
                )
            values = values.to(device=reference.device, dtype=torch.long)
            probs = probs.to(device=reference.device, dtype=torch.float32)
            probs = probs / probs.sum().clamp_min(1e-12)
            sample_idx = torch.multinomial(probs, reference.numel(), replacement=True)
            sampled = values[sample_idx].reshape_as(reference)
            return sampled.clamp(int(self.budget_min_k), int(self.budget_max_k))

        def clamp_budget_sum(dynamic_k: torch.Tensor, budget_cont: torch.Tensor) -> torch.Tensor:
            budget_clamp_mode = getattr(self, "budget_clamp_mode", "none")
            if budget_clamp_mode == "none":
                return dynamic_k
            if budget_clamp_mode != "mono_layer_sum":
                raise ValueError(f"Unknown budget_clamp_mode: {budget_clamp_mode!r}")

            target_sum = int(round(float(self.block_counts) * dynamic_k.numel()))
            flat = dynamic_k.reshape(-1).clone()
            score = (budget_cont.detach().reshape(-1) - budget_cont.detach().reshape(-1).floor()).float()
            diff = target_sum - int(flat.sum().item())
            if diff > 0:
                can_inc = flat < int(self.budget_max_k)
                take = min(diff, int(can_inc.sum().item()))
                if take > 0:
                    idx = torch.nonzero(can_inc, as_tuple=False).flatten()
                    order = torch.argsort(score[idx], descending=True)
                    flat[idx[order[:take]]] += 1
            elif diff < 0:
                can_dec = flat > int(self.budget_min_k)
                take = min(-diff, int(can_dec.sum().item()))
                if take > 0:
                    idx = torch.nonzero(can_dec, as_tuple=False).flatten()
                    order = torch.argsort(score[idx], descending=False)
                    flat[idx[order[:take]]] -= 1
            return flat.reshape_as(dynamic_k)

        def apply_budget_controls(budget_cont: torch.Tensor) -> torch.Tensor:
            budget_center_mode = getattr(self, "budget_center_mode", "none")
            if budget_center_mode == "mono_layer_mean":
                target_mean = budget_cont.new_tensor(float(self.block_counts))
                budget_cont = budget_cont - budget_cont.mean().detach() + target_mean
                budget_cont = budget_cont.clamp(float(self.budget_min_k), float(self.budget_max_k))
            elif budget_center_mode != "none":
                raise ValueError(f"Unknown budget_center_mode: {budget_center_mode!r}")

            budget_shuffle_mode = getattr(self, "budget_shuffle_mode", "none")
            if budget_shuffle_mode == "token":
                flat = budget_cont.reshape(-1)
                perm = torch.randperm(flat.numel(), device=flat.device)
                budget_cont = flat[perm].reshape_as(budget_cont)
            elif budget_shuffle_mode != "none":
                raise ValueError(f"Unknown budget_shuffle_mode: {budget_shuffle_mode!r}")
            return budget_cont

        def finalize_budget(budget_cont: torch.Tensor) -> Tuple[torch.LongTensor, torch.Tensor]:
            replay_budget = sample_replay_budget(budget_cont)
            if replay_budget is not None:
                return replay_budget.long(), replay_budget.float()

            budget_cont = apply_budget_controls(budget_cont)
            budget_rounded = _ste_round(budget_cont)
            dynamic_k = budget_rounded.detach().long().clamp(
                self.budget_min_k, self.budget_max_k
            )
            dynamic_k = clamp_budget_sum(dynamic_k, budget_cont)
            budget_ste = budget_cont + (dynamic_k.float() - budget_cont).detach()
            return dynamic_k.long(), budget_ste

        if self.budget_mode == "query_aware_asymm":
            # Fixed layer (min == max): return base block_counts unchanged.
            if self.budget_min_k == self.budget_max_k:
                return self.block_counts, None
            # Dynamic layer: same logic as query_aware_tool.
            budget_signal = self.budget_head(hidden_states).sigmoid()
            budget_cont = (
                budget_signal * (self.budget_max_k - self.budget_min_k)
                + self.budget_min_k
            )
            dynamic_k, budget_ste = finalize_budget(budget_cont)
            return dynamic_k, budget_ste

        if self.budget_mode == "query_aware_tool":
            # (B, T, H_kv), float, gradient retained
            budget_signal = self.budget_head(hidden_states).sigmoid()
            budget_cont = (
                budget_signal * (self.budget_max_k - self.budget_min_k)
                + self.budget_min_k
            )
            dynamic_k, budget_ste = finalize_budget(budget_cont)
            return dynamic_k, budget_ste

        if self.budget_mode == "query_aware_gate":
            # Reduce g_slc from query heads (HQ) to KV heads (H_kv).
            # g_slc shape: (B, T, HQ); reshape to (B, T, H_kv, G) then mean over G.
            g_slc_kv = g_slc.reshape(
                batch_size, seq_len, self.num_kv_heads, self.num_kv_groups
            ).mean(dim=-1)                                       # (B, T, H_kv)

            budget_cont = (
                g_slc_kv * (self.budget_max_k - self.budget_min_k)
                + self.budget_min_k
            )
            dynamic_k, budget_ste = finalize_budget(budget_cont)
            # g_slc retains its original post-fusion role (not replaced here).
            # See forward() – g_slc is passed unchanged to parallel_nsa, keeping
            # the gradient path: loss → o_slc * g_slc → g_slc.
            return dynamic_k, budget_ste

        raise ValueError(f"Unknown budget_mode: {self.budget_mode!r}. "
                         f"Valid modes: uniform, layer_aware, query_aware_tool, "
                         f"query_aware_gate, query_aware_asymm")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        self._budget_tensor = None
        self._dynamic_k_tensor = None
        self._debug_selection_state = None

        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, seq_len, _ = hidden_states.size()

        q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=3)
        g_cmp, g_slc, g_swa = g.sigmoid().unbind(-1)

        cu_seqlens = kwargs.get('cu_seqlens', None)

        seqlen_offset, max_seqlen = 0, seq_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                seqlen_offset = (seqlen_offset + attention_mask.sum(-1) - attention_mask.shape[-1]).clamp(min=0)
                max_seqlen = q.shape[1] + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)

        if past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            k_cached, v_cached = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=seq_len,
                cache_kwargs=dict(window_size=self.window_size)
            )['attn_state']
            if cache_has_content:
                k, v = k_cached, v_cached
                k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
                v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        # Resolve block budget ------------------------------------------- #
        effective_block_counts, budget_tensor = self._compute_dynamic_block_counts(
            hidden_states=hidden_states,
            g_slc=g_slc,
            batch_size=batch_size,
            seq_len=seq_len,
        )

        debug_block_indices = None
        if getattr(self, "_debug_record_selection", False):
            if g_cmp is None:
                warnings.warn("Selection debug requested, but g_cmp is None; block indices cannot be captured.")
            else:
                debug_scale = q.shape[-1] ** -0.5
                k_cmp = mean_pooling(k, self.block_size, cu_seqlens)
                v_cmp = mean_pooling(v, self.block_size, cu_seqlens)
                _, lse_cmp = parallel_nsa_compression(
                    q=q,
                    k=k_cmp,
                    v=v_cmp,
                    block_size=self.block_size,
                    scale=debug_scale,
                    offsets=cu_seqlens,
                )
                debug_block_indices = parallel_nsa_topk(
                    q=q,
                    k=k_cmp,
                    lse=lse_cmp,
                    block_counts=effective_block_counts,
                    block_size=self.block_size,
                    scale=debug_scale,
                    offsets=cu_seqlens,
                )

        o = parallel_nsa(
            q=q,
            k=k,
            v=v,
            g_cmp=g_cmp,
            g_slc=g_slc,
            g_swa=g_swa,
            block_size=self.block_size,
            block_counts=effective_block_counts,
            window_size=self.window_size,
            cu_seqlens=cu_seqlens,
            head_first=False
        )
        o = o.reshape(batch_size, seq_len, -1)
        o = self.o_proj(o)

        if not output_attentions:
            attentions = None

        # budget_tensor is stored on the module so NSAModel can collect it
        # without breaking the standard 3-tuple attention output interface.
        # It is reset to None at the start of each forward call (see above).
        self._budget_tensor = budget_tensor
        self._dynamic_k_tensor = effective_block_counts if torch.is_tensor(effective_block_counts) else None
        if debug_block_indices is not None:
            self._debug_selection_state = {
                "block_indices": debug_block_indices.detach().cpu(),
                "block_counts": effective_block_counts.detach().cpu() if torch.is_tensor(effective_block_counts) else int(effective_block_counts),
                "block_size": int(self.block_size),
                "seq_len": int(seq_len),
                "layer_idx": int(self.layer_idx) if self.layer_idx is not None else None,
            }
            if getattr(self, "_debug_record_selection_tensors", False):
                self._debug_selection_state.update({
                    "q": q.detach().cpu(),
                    "k": k.detach().cpu(),
                    "scale": float(q.shape[-1] ** -0.5),
                })

        return o, attentions, past_key_values


class NSABlock(nn.Module):
    def __init__(self, config: NSAConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        # ------------------------------------------------------------------ #
        # Phase 1 – layer-aware budget resolution                             #
        # When block_counts is a list, each layer gets its own budget value.  #
        # When it is a scalar, all layers share the same value (uniform).     #
        # ------------------------------------------------------------------ #
        if isinstance(config.block_counts, list):
            layer_block_counts = config.block_counts[layer_idx]
        else:
            layer_block_counts = config.block_counts

        # Resolve per-layer budget bounds (scalar or list).
        layer_budget_min_k = (
            config.budget_min_k[layer_idx]
            if isinstance(config.budget_min_k, list)
            else config.budget_min_k
        )
        layer_budget_max_k = (
            config.budget_max_k[layer_idx]
            if isinstance(config.budget_max_k, list)
            else config.budget_max_k
        )

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.attn = NativeSparseAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            qkv_bias=config.qkv_bias,
            block_size=config.block_size,
            block_counts=layer_block_counts,
            window_size=config.window_size,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            layer_idx=layer_idx,
            budget_mode=config.budget_mode,
            budget_min_k=layer_budget_min_k,
            budget_max_k=layer_budget_max_k,
        )
        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.mlp = NSAMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[Dict]
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        # NativeSparseAttention.forward() stores its budget_tensor as
        # self.attn._budget_tensor so NSAModel can collect it without
        # changing the standard 3-tuple attention output interface.
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs
        )
        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        return hidden_states, attentions, past_key_values


class NSAPreTrainedModel(PreTrainedModel):

    config_class = NSAConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['NSABlock']
    _supports_cache_class = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
        prenorm_residual_strategy: Optional[str] = 'rescale',
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()

        if prenorm_residual_strategy is not None:
            p = None
            if hasattr(module, 'o_proj'):
                p = module.o_proj.weight
            elif hasattr(module, 'down_proj'):
                p = module.down_proj.weight
            if p is not None:
                if prenorm_residual_strategy == 'rescale':
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)
                elif prenorm_residual_strategy == 'zero':
                    nn.init.zeros_(p)
                else:
                    raise ValueError(f"Invalid prenorm_residual_strategy: {prenorm_residual_strategy}")


class NSAModel(NSAPreTrainedModel):

    def __init__(self, config: NSAConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([NSABlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)

        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # noqa
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[Dict]
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if output_attentions:
            warnings.warn("`NSAModel` does not `output_attentions` now, setting it to `False`.")
            output_attentions = False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                hidden_states, attentions, past_key_values = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    use_cache,
                    output_attentions,
                    **kwargs
                )
            else:
                hidden_states, attentions, past_key_values = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    **kwargs
                )

            if output_attentions:
                all_attns += (attentions,)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(i for i in [hidden_states, past_key_values, all_hidden_states, all_attns] if i is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attns,
        )


class NSAForCausalLM(NSAPreTrainedModel, GenerationMixin):

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = NSAModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None

        # ------------------------------------------------------------------ #
        # Auxiliary budget regularisation loss weight.                        #
        # For query_aware_tool / query_aware_gate modes, adding a small       #
        # regularisation loss on the continuous budget signal encourages the  #
        # model to keep budgets diverse and not collapse to a constant k.     #
        # Set to 0.0 to disable (e.g. during uniform / layer_aware phases).  #
        # ------------------------------------------------------------------ #
        self.budget_aux_loss_weight: float = 0.0
        self.budget_aux_mode: str = "variance"
        self.budget_aux_target_min: float = 40.0
        self.budget_aux_target_max: float = 60.0
        self.budget_aux_target_weight: float = 0.0
        self._last_loss_breakdown: Dict[str, float] = {}

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def generate(self, *args, **kwargs):
        try:
            return super().generate(*args, **kwargs)
        except AttributeError as exception:
            if 'past_key_values' in str(exception):
                raise AttributeError(
                    f"You tried to call `generate` with a decoding strategy that manipulates `past_key_values`, "
                    f"which is not supported for {self.__class__.__name__}. "
                    f"Try another generation strategy instead. "
                    f"For the available generation strategies, check this doc: "
                    f"https://huggingface.co/docs/transformers/en/generation_strategies#decoding-strategies"
                )
            else:
                raise exception

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        logits_to_keep: Optional[int] = None,
        **kwargs
    ):
        if past_key_values is not None and len(past_key_values) > 0:
            input_ids = input_ids[:, -1:]
        if inputs_embeds is not None and len(past_key_values) == 0:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids.contiguous()}

        if logits_to_keep is not None:
            model_inputs['logits_to_keep'] = logits_to_keep

        model_inputs.update({
            'past_key_values': past_key_values,
            'use_cache': use_cache,
            'attention_mask': attention_mask,
        })
        return model_inputs

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Optional[int] = 0,
        **kwargs: Unpack[Dict]
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        self._last_loss_breakdown = {}
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        # Collect per-layer budget tensors written by NativeSparseAttention.forward().
        # _budget_tensor is None for uniform/layer_aware modes (no aux loss needed).
        all_budget_tensors = [
            layer.attn._budget_tensor
            for layer in self.model.layers
            if getattr(layer.attn, '_budget_tensor', None) is not None
        ]

        hidden_states = outputs[0]
        fuse_linear_and_cross_entropy = self.config.fuse_cross_entropy and self.training

        loss, logits = None, None
        if not fuse_linear_and_cross_entropy or labels is None:
            logits = self.lm_head(hidden_states if logits_to_keep is None else hidden_states[:, -logits_to_keep:])
        if labels is not None:
            if getattr(self, 'criterion', None) is None:
                if fuse_linear_and_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss()
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion
            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)
            if fuse_linear_and_cross_entropy:
                loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))

            # -------------------------------------------------------------- #
            # Auxiliary budget regularisation loss                            #
            #                                                                 #
            # For query_aware modes the continuous budget tensors (one per    #
            # layer) are stacked and a light variance-encouraging penalty is  #
            # added to prevent the model from collapsing all budgets to a     #
            # constant value.  The weight is set to 0 for Phase 1 runs.      #
            #                                                                 #
            # L_aux = -weight * Var(budget_ste_all_layers)                   #
            # Minimising -Var encourages higher variance → diverse budgets.  #
            # -------------------------------------------------------------- #
            lm_loss = loss
            aux_loss = loss.new_zeros(())
            target_band_loss = loss.new_zeros(())
            budget_var_mean = loss.new_zeros(())
            budget_aux_mode = getattr(self, "budget_aux_mode", "variance")
            if all_budget_tensors:
                # all_budget_tensors: list of (B, T, H_kv) float tensors
                budget_stack = torch.stack(all_budget_tensors, dim=0)   # (L, B, T, H_kv)
                budget_var = budget_stack.var(dim=[1, 2, 3])            # (L,)
                budget_var_mean = budget_var.mean()
                if budget_aux_mode == "variance" and self.budget_aux_loss_weight > 0.0:
                    aux_loss = -self.budget_aux_loss_weight * budget_var_mean
                    loss = loss + aux_loss
                elif budget_aux_mode == "target_band" and self.budget_aux_target_weight > 0.0:
                    target_min = max(float(self.budget_aux_target_min), 1e-6)
                    target_max = max(float(self.budget_aux_target_max), target_min)
                    below = torch.relu(budget_var.new_tensor(target_min) - budget_var) / target_min
                    above = torch.relu(budget_var - budget_var.new_tensor(target_max)) / target_max
                    target_band_loss = (below.square() + above.square()).mean()
                    aux_loss = float(self.budget_aux_target_weight) * target_band_loss
                    loss = loss + aux_loss
            self._last_loss_breakdown = {
                "lm_loss": float(lm_loss.detach().float().item()),
                "aux_loss": float(aux_loss.detach().float().item()),
                "total_loss": float(loss.detach().float().item()),
                "budget_aux_loss_weight": float(self.budget_aux_loss_weight),
                "budget_aux_mode": str(budget_aux_mode),
                "budget_aux_target_min": float(getattr(self, "budget_aux_target_min", 40.0)),
                "budget_aux_target_max": float(getattr(self, "budget_aux_target_max", 60.0)),
                "budget_aux_target_weight": float(getattr(self, "budget_aux_target_weight", 0.0)),
                "target_band_loss": float(target_band_loss.detach().float().item()),
                "budget_var_mean": float(budget_var_mean.detach().float().item()),
            }

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
