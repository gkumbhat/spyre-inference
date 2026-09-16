# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stick-aligned SDPA for vLLM's generic ViT attention path.

``vllm.v1.attention.ops.vit_attn_wrappers.apply_sdpa`` (used by
``MMEncoderAttention``, the default ViT attention for models like CLIP that
don't define a bespoke vision Attention) calls ``F.scaled_dot_product_attention``
directly on whatever sequence length the image produces. torch-spyre compiles
that op internally on every dispatch (regardless of ``--enforce-eager``) and its
BMM-padding pass asserts when the sequence length isn't a multiple of the
64-element fp16 stick (e.g. CLIP ViT-B/32's 50 patches). Padding Q/K/V to the
stick ourselves and masking the padded keys avoids the pass ever needing to act
-- the same fix already proven for Pixtral's vision tower (see
``multimodal/pixtral.py``), ported here for vLLM's shared ViT attention path.
"""

from __future__ import annotations

import einops
import torch
import torch.nn.functional as F
from vllm.logger import init_logger

from spyre_inference.custom_ops.utils import convert

logger = init_logger(__name__)

_STICK = 64


def _align_up(n: int, align: int = _STICK) -> int:
    return (n + align - 1) // align * align


def _padded_apply_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Drop-in replacement for ``vit_attn_wrappers.apply_sdpa``.

    Input/output shape: ``(batch, seq, num_heads, head_size)``.
    """
    seq_q, seq_kv = q.shape[1], k.shape[1]
    head_size = q.shape[-1]
    seq_q_pad, seq_kv_pad, head_pad = (
        _align_up(seq_q),
        _align_up(seq_kv),
        _align_up(head_size),
    )
    # SDPA's default scale (when None) is derived from the *actual* head dim
    # at call time, so padding it would silently change the scale unless the
    # unpadded value is fixed here first.
    if scale is None:
        scale = head_size**-0.5

    q, k, v = (einops.rearrange(x, "b s h d -> b h s d") for x in (q, k, v))

    if (seq_q_pad, seq_kv_pad, head_pad) == (seq_q, seq_kv, head_size):
        # Offset operands read as offset 0 (torch-spyre#3770), so unpadded SDPA
        # would be silently wrong here; the padded branch escapes it only
        # because F.pad below materializes.
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        mask = None
    else:
        q = F.pad(q, (0, head_pad - head_size, 0, seq_q_pad - seq_q))
        k = F.pad(k, (0, head_pad - head_size, 0, seq_kv_pad - seq_kv))
        v = F.pad(v, (0, head_pad - head_size, 0, seq_kv_pad - seq_kv))
        # Padded queries need no mask (their output rows are cropped below);
        # padded keys must never be attended to. Assembled on CPU: a strided
        # slice-assign is not stick-safe on Spyre.
        mask = torch.zeros(1, 1, seq_q_pad, seq_kv_pad, dtype=q.dtype)
        mask[:, :, :, seq_kv:] = torch.finfo(q.dtype).min
        mask = convert(mask, device=q.device)

    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, enable_gqa=enable_gqa
    )
    out = out[:, :, :seq_q, :head_size]
    return einops.rearrange(out, "b h s d -> b s h d")


def register() -> None:
    import vllm.v1.attention.ops.vit_attn_wrappers as vit_attn_wrappers

    if getattr(vit_attn_wrappers.apply_sdpa, "_spyre_patched", False):
        return

    _padded_apply_sdpa._spyre_patched = True  # ty: ignore[unresolved-attribute]
    vit_attn_wrappers.apply_sdpa = _padded_apply_sdpa
    logger.debug_once(
        "Patched vllm.v1.attention.ops.vit_attn_wrappers.apply_sdpa to pad to "
        "the 64-element stick before calling SDPA."
    )
