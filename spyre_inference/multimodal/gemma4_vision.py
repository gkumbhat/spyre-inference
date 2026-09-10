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

"""Gemma 4 full-vision-tower workarounds for Spyre.

``Gemma4Model.vision_tower`` is ``transformers.models.gemma4.modeling_gemma4.
Gemma4VisionModel``, loaded via plain ``AutoModel.from_config`` -- outside vLLM's
layer registries, same category as Pixtral (``multimodal/pixtral.py``). Every fix
here is a guarded, idempotent monkeypatch; ``apply()`` is the only entry point.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

from spyre_inference.custom_ops.utils import convert
from spyre_inference.multimodal.pixtral import padded_sdpa

logger = init_logger(__name__)

# Spyre stick size at fp16 (128 bytes / 2 bytes per element); head_dim must pad to a
# multiple of this, and the two-axis RoPE quarter-interleave (below) additionally
# needs 2*BLOCK_SIZE so each axis's half lands on a whole stick.
BLOCK_SIZE = 64


# ---------------------------------------------------------------------------
# Ported (verbatim tensor math, adapted to operate on the real transformers
# Gemma4VisionModel tree rather than a separate compiled executor) from the
# hardware-validated reference at torch-spyre/hf-adapters#495, branch
# gemma4_vlm_ple_moe, hf_adapters/hf_gemma4_vision.py + hf_adapters/hf_common.py.
# Validated against the real transformers RoPE reference and on real Spyre
# hardware by scripts/probe_gemma4_vision_rope.py and probe_gemma4_vision_attn.py.
# ---------------------------------------------------------------------------


def _as_plain_linear(layer: nn.Module) -> nn.Linear:
    """Bounce a vLLM linear layer into a plain `nn.Linear`.

    `Gemma4ClippableLinear.linear` is a raw `torch.nn.Linear` in the reference HF
    model, but `Gemma4ForConditionalGeneration.__init__` runs `recursive_replace_linear`
    over the whole vision tower, so on Spyre it's actually a `SpyreReplicatedLinear`
    (`custom_ops/linear.py`) whose `.weight` is stored *transposed* -- `[in, out]`,
    for the Spyre-fast `x @ Wᵀ` GEMM (`SpyreTransposedWeightMethod.build_weight_t`) --
    and which exposes `.input_size`/`.output_size`, not `.in_features`/`.out_features`.
    The padding helpers below assume nn.Linear's standard `[out, in]` layout, so
    bounce through this first; the result replaces `.linear` outright (this module's
    patches don't need the Spyre fast-path GEMM to still apply to these four
    projections specifically -- `F.linear` lowers fine here regardless).
    """
    if isinstance(layer, nn.Linear):
        return layer
    weight_t = layer.weight.detach()  # [in, out] once vLLM's Spyre OOT method has run
    in_features, out_features = weight_t.shape
    plain = nn.Linear(in_features, out_features, bias=layer.bias is not None)
    plain.weight = nn.Parameter(weight_t.t().contiguous(), requires_grad=False)
    if layer.bias is not None:
        plain.bias = nn.Parameter(layer.bias.detach().clone(), requires_grad=False)
    return plain


def _pad_qk_linear(proj, num_heads: int, orig_head_dim: int, padded_head_dim: int) -> nn.Linear:
    """Pad and reorder two-axis RoPE channels into one matrix-RoPE layout."""
    linear = proj.linear
    weight = linear.weight.detach().view(num_heads, orig_head_dim, -1)
    new_weight = torch.zeros(num_heads, padded_head_dim, weight.shape[-1], dtype=weight.dtype)
    quarter = orig_head_dim // 4
    padded_half = padded_head_dim // 2
    new_weight[:, :quarter] = weight[:, :quarter]
    new_weight[:, quarter : 2 * quarter] = weight[:, 2 * quarter : 3 * quarter]
    new_weight[:, padded_half : padded_half + quarter] = weight[:, quarter : 2 * quarter]
    new_weight[:, padded_half + quarter : padded_half + 2 * quarter] = weight[:, 3 * quarter :]
    padded = nn.Linear(linear.in_features, num_heads * padded_head_dim, bias=linear.bias is not None)
    padded.weight = nn.Parameter(new_weight.reshape(num_heads * padded_head_dim, -1), requires_grad=False)
    if linear.bias is not None:
        bias = linear.bias.detach().view(num_heads, orig_head_dim)
        new_bias = torch.zeros(num_heads, padded_head_dim, dtype=bias.dtype)
        new_bias[:, :quarter] = bias[:, :quarter]
        new_bias[:, quarter : 2 * quarter] = bias[:, 2 * quarter : 3 * quarter]
        new_bias[:, padded_half : padded_half + quarter] = bias[:, quarter : 2 * quarter]
        new_bias[:, padded_half + quarter : padded_half + 2 * quarter] = bias[:, 3 * quarter :]
        padded.bias = nn.Parameter(new_bias.reshape(-1), requires_grad=False)
    return padded


def _pad_proj_output_simple(proj: nn.Linear, n_heads: int, orig_head_dim: int, padded_head_dim: int) -> nn.Linear:
    """End-pad each head of a [n_heads*head_dim, hidden] output projection (V)."""
    w = proj.weight
    hidden = w.shape[1]
    new_w = torch.zeros(n_heads * padded_head_dim, hidden, dtype=w.dtype)
    for h in range(n_heads):
        s, d = h * orig_head_dim, h * padded_head_dim
        new_w[d : d + orig_head_dim, :] = w[s : s + orig_head_dim, :]
    new_proj = nn.Linear(hidden, n_heads * padded_head_dim, bias=proj.bias is not None)
    new_proj.weight = nn.Parameter(new_w, requires_grad=False)
    if proj.bias is not None:
        new_b = torch.zeros(n_heads * padded_head_dim, dtype=proj.bias.dtype)
        for h in range(n_heads):
            s, d = h * orig_head_dim, h * padded_head_dim
            new_b[d : d + orig_head_dim] = proj.bias[s : s + orig_head_dim]
        new_proj.bias = nn.Parameter(new_b, requires_grad=False)
    return new_proj


def _pad_proj_input_simple(proj: nn.Linear, n_heads: int, orig_head_dim: int, padded_head_dim: int) -> nn.Linear:
    """End-pad each head along the input dim of an O-style projection."""
    w = proj.weight
    hidden = w.shape[0]
    new_w = torch.zeros(hidden, n_heads * padded_head_dim, dtype=w.dtype)
    for h in range(n_heads):
        s, d = h * orig_head_dim, h * padded_head_dim
        new_w[:, d : d + orig_head_dim] = w[:, s : s + orig_head_dim]
    new_proj = nn.Linear(n_heads * padded_head_dim, hidden, bias=proj.bias is not None)
    new_proj.weight = nn.Parameter(new_w, requires_grad=False)
    if proj.bias is not None:
        new_proj.bias = nn.Parameter(proj.bias.detach().clone(), requires_grad=False)
    return new_proj


def _pad_norm_weight(norm, orig_head_dim: int, padded_head_dim: int) -> nn.Parameter:
    weight = norm.weight.detach()
    padded = torch.ones(padded_head_dim, dtype=weight.dtype)
    quarter = orig_head_dim // 4
    padded_half = padded_head_dim // 2
    padded[:quarter] = weight[:quarter]
    padded[quarter : 2 * quarter] = weight[2 * quarter : 3 * quarter]
    padded[padded_half : padded_half + quarter] = weight[quarter : 2 * quarter]
    padded[padded_half + quarter : padded_half + 2 * quarter] = weight[3 * quarter :]
    return nn.Parameter(padded, requires_grad=False)


def _padded_rms_norm(hidden_states: torch.Tensor, weight, eps: float, orig_head_dim: int) -> torch.Tensor:
    """RMSNorm whose denominator is scaled back to the *unpadded* head_dim, so the
    zero-filled padding lanes (which would otherwise deflate the variance) don't
    change the normalization the real channels get.

    No fp32 promotion, unlike the hf-adapters reference the rest of this is ported
    from: torch-spyre does not support it (``custom_ops/rms_norm.py``), and an
    on-device fp16->fp32->fp16 round trip here leaves the result in a stick-tiling
    state a later eager elementwise op cannot broadcast against ("Multi-arg
    pointwise with mixed EA ... no staggered operand is broadcastable either").
    Same trade as ``SpyreRMSNorm``/``SpyreGemmaRMSNorm``: expect small numerical
    differences from upstream.
    """
    dtype = hidden_states.dtype
    variance = (hidden_states * hidden_states).mean(-1, keepdim=True)
    variance = variance * (hidden_states.shape[-1] / orig_head_dim)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    if weight is not None:
        # `Gemma4RMSNorm` weights are fp32 (transformers builds them at the default
        # dtype and nothing downcasts them), so multiplying without this cast would
        # silently promote the whole activation to fp32 -- which is both the
        # unsupported promotion above and a dtype mismatch against the fp16 padded
        # projections. Stock hid this behind a trailing `.type_as(hidden_states)`.
        hidden_states = hidden_states * weight.to(dtype)
    return hidden_states


def _gemma4_rope_cos_sin(
    inv_freq: torch.Tensor,
    position_ids: torch.Tensor,
    padded_head_dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`[bsz, seq, 1, padded_head_dim]` cos and signed-sin tensors for
    `multimodal.pixtral.rope_rotate_matmul`'s `x*cos + (x @ m)*sin` form (the heads
    dim broadcasts, same as Pixtral's own `[1, patches, 1, head_dim]`), adapted to
    Gemma 4 vision's two-axis, `rotate_half`-per-axis convention
    (`transformers.models.gemma4.modeling_gemma4.apply_multidimensional_rope`) and
    the head-dim padding/channel packing `_pad_qk_linear` above already applies:
    padded layout is `[X_quarter, Y_quarter, zeros | X_quarter, Y_quarter, zeros]`
    (two half-width blocks), so a single first-half/second-half swap (`m`,
    `kind="half"`) plays the role of `rotate_half` for *both* axes at once, and each
    axis's angle is duplicated into both blocks with the sign flip `rotate_half`
    needs baked into `sin` rather than into `m`.

    Built on CPU in fp32 and cast on the way out, like Pixtral's own freqs table:
    the trig is exact there and only the fp16 result reaches the device.
    """
    positions = position_ids.to("cpu").clamp(min=0).float()
    angles = positions[..., None] * inv_freq.to("cpu").float()  # [bsz, seq, 2, quarter]
    cos_axis = angles.cos()
    sin_axis = angles.sin()
    bsz, seq_len, _, quarter = cos_axis.shape
    half = padded_head_dim // 2
    cos_half = torch.ones(bsz, seq_len, half)
    sin_half_neg = torch.zeros(bsz, seq_len, half)
    sin_half_pos = torch.zeros(bsz, seq_len, half)
    for axis in range(2):
        start, end = axis * quarter, axis * quarter + quarter
        cos_half[..., start:end] = cos_axis[:, :, axis, :]
        sin_half_neg[..., start:end] = -sin_axis[:, :, axis, :]
        sin_half_pos[..., start:end] = sin_axis[:, :, axis, :]
    cos_full = torch.cat([cos_half, cos_half], dim=-1).unsqueeze(2)  # [bsz, seq, 1, D]
    sin_full = torch.cat([sin_half_neg, sin_half_pos], dim=-1).unsqueeze(2)
    return cos_full.to(dtype), sin_full.to(dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """`x*cos + rotate_half(x)*sin` with the half-swap done by slicing.

    Pixtral needs the matmul form (`multimodal/pixtral.py::rope_rotate_matmul`)
    because its head_dim is 64, so each `rotate_half` half is 32 wide -- narrower
    than the stick, which torch-spyre cannot lay out. Gemma 4 vision pads head_dim
    to 128 (rope needs that padding anyway, for the two-axis channel layout), so
    each half is exactly one 64-element stick and slicing is stick-legal.

    Preferring the slice here is not just simplification: at this tower's real
    shapes the `[N, D] @ [D, D]` rotation reduction fails to tile once it follows
    the RMSNorm reduction that precedes it in the layer ("buf2 (Reduction): no
    mechanism to resolve stick incompatibility"), while the slice form lowers.

    `sin` already carries `rotate_half`'s sign flip as `cat([-sin, +sin])` (see
    `_gemma4_rope_cos_sin`), so the swap here is a plain `cat([x2, x1])`.
    """
    half = x.shape[-1] // 2
    swapped = torch.cat([x[..., half:], x[..., :half]], dim=-1)
    return x * cos + swapped * sin


def _padded_head_dim(orig_head_dim: int) -> int:
    return math.ceil(orig_head_dim / (2 * BLOCK_SIZE)) * (2 * BLOCK_SIZE)


def _pad_mlp(layer, orig_intermediate: int, padded_intermediate: int) -> None:
    """Zero-extend the MLP's intermediate width onto the stick, once per layer.

    26B-A4B's vision tower is 4304 wide, which is not a multiple of 64; E2B's 3072
    already is, so this is a no-op there. Gate/up gain zero output rows and down
    gains matching zero input columns, so the padding contributes nothing to the
    result (`gelu(0) * 0 == 0`, and a zero down-projection column ignores it).
    """
    if padded_intermediate == orig_intermediate:
        return
    if getattr(layer.mlp, "_spyre_padded_intermediate", None) == padded_intermediate:
        return
    mlp = layer.mlp
    device = mlp.gate_proj.linear.weight.device
    for name in ("gate_proj", "up_proj"):
        proj = getattr(mlp, name)
        proj.linear = _pad_proj_output_simple(
            _as_plain_linear(proj.linear), 1, orig_intermediate, padded_intermediate
        ).to(device)
    mlp.down_proj.linear = _pad_proj_input_simple(
        _as_plain_linear(mlp.down_proj.linear), 1, orig_intermediate, padded_intermediate
    ).to(device)
    mlp._spyre_padded_intermediate = padded_intermediate


def _prepare_attention(attn, num_heads: int, orig_head_dim: int, padded_head_dim: int) -> None:
    """Pad one `Gemma4VisionAttention`'s projections/norms to `padded_head_dim`, once."""
    if getattr(attn, "_spyre_padded_head_dim", None) == padded_head_dim:
        return
    if attn.v_norm.with_scale:
        raise NotImplementedError(
            "Scaled Gemma 4 vision V normalization is not supported on Spyre."
        )
    # The padding helpers (ported from hf-adapters, which pads on CPU and moves the
    # whole model to device afterward in one step) build their scratch tensors with
    # torch.zeros/ones -- no device= -- so they land on CPU regardless of the source
    # weight's device. Here the model is already on Spyre, so move every padded
    # result back explicitly.
    device = attn.q_norm.weight.device
    attn.q_proj.linear = _as_plain_linear(attn.q_proj.linear)
    attn.k_proj.linear = _as_plain_linear(attn.k_proj.linear)
    attn.v_proj.linear = _as_plain_linear(attn.v_proj.linear)
    attn.o_proj.linear = _as_plain_linear(attn.o_proj.linear)
    attn.q_proj.linear = _pad_qk_linear(attn.q_proj, num_heads, orig_head_dim, padded_head_dim).to(device)
    attn.k_proj.linear = _pad_qk_linear(attn.k_proj, num_heads, orig_head_dim, padded_head_dim).to(device)
    attn.v_proj.linear = _pad_proj_output_simple(
        attn.v_proj.linear, num_heads, orig_head_dim, padded_head_dim
    ).to(device)
    attn.o_proj.linear = _pad_proj_input_simple(
        attn.o_proj.linear, num_heads, orig_head_dim, padded_head_dim
    ).to(device)
    attn.q_norm.weight = nn.Parameter(
        _pad_norm_weight(attn.q_norm, orig_head_dim, padded_head_dim).to(device), requires_grad=False
    )
    attn.k_norm.weight = nn.Parameter(
        _pad_norm_weight(attn.k_norm, orig_head_dim, padded_head_dim).to(device), requires_grad=False
    )
    attn._spyre_padded_head_dim = padded_head_dim


def _run_attention(
    attn,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attn_mask: torch.Tensor,
    num_heads: int,
    orig_head_dim: int,
    padded_head_dim: int,
) -> torch.Tensor:
    bsz, seq_len, _ = hidden_states.shape

    # Rope at [B, L, H, D], then transpose to [B, H, L, D] for SDPA -- Pixtral's own
    # order (`multimodal/pixtral.py::patch_vision_attention`).
    q = attn.q_proj(hidden_states).view(bsz, seq_len, num_heads, padded_head_dim)
    q = _padded_rms_norm(q, attn.q_norm.weight, attn.q_norm.eps, orig_head_dim)
    q = _apply_rope(q, cos, sin).transpose(1, 2)
    k = attn.k_proj(hidden_states).view(bsz, seq_len, num_heads, padded_head_dim)
    k = _padded_rms_norm(k, attn.k_norm.weight, attn.k_norm.eps, orig_head_dim)
    k = _apply_rope(k, cos, sin).transpose(1, 2)
    v = attn.v_proj(hidden_states).view(bsz, seq_len, num_heads, padded_head_dim)
    v = _padded_rms_norm(v, None, attn.v_norm.eps, orig_head_dim).transpose(1, 2)

    attn_out = padded_sdpa(q, k, v, attn_mask, scale=float(attn.scaling))
    attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
    return attn.o_proj(attn_out)


def _run_layer(
    layer,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attn_mask: torch.Tensor,
    num_heads: int,
    orig_head_dim: int,
    padded_head_dim: int,
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)
    attn_out = _run_attention(
        layer.self_attn,
        hidden_states,
        cos,
        sin,
        attn_mask,
        num_heads,
        orig_head_dim,
        padded_head_dim,
    )
    hidden_states = residual + layer.post_attention_layernorm(attn_out)

    residual = hidden_states
    hidden_states = layer.pre_feedforward_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    hidden_states = layer.post_feedforward_layernorm(hidden_states)
    return residual + hidden_states


def patch_vision_encoder() -> None:
    """Replace `Gemma4VisionEncoder.forward` with a Spyre-safe walk over its layers.

    Three things the stock forward does don't work on Spyre: it builds the mask via
    `masking_utils.create_bidirectional_mask`, it feeds stock `rotate_half`-style
    rope (which slices the head into halves -- 36 wide at Gemma 4 vision's native
    head_dim=72, so not stick-aligned, and 72 is not a 64-multiple to begin with),
    and its attention has no padding for a patch count coprime with the 64 stick.
    So: pad head_dim to 128, rotate via Pixtral's stick-aligned matmul form, and run
    attention through Pixtral's `padded_sdpa`. Both of those are already validated on
    real Spyre hardware (see scripts/probe_gemma4_vision_*.py).
    """
    try:
        from transformers.models.gemma4 import modeling_gemma4
    except ImportError:
        return

    cls = getattr(modeling_gemma4, "Gemma4VisionEncoder", None)
    if cls is None or getattr(cls.forward, "_spyre_patched", False):
        return

    def _forward(self, inputs_embeds, attention_mask, pixel_position_ids=None, **kwargs):
        del kwargs
        config = self.config
        if config.num_key_value_heads != config.num_attention_heads:
            raise NotImplementedError(
                "Gemma 4 vision GQA is not supported on Spyre; num_key_value_heads "
                "must equal num_attention_heads."
            )
        num_heads = config.num_attention_heads
        orig_head_dim = config.head_dim
        padded_head_dim = _padded_head_dim(orig_head_dim)

        device = inputs_embeds.device
        dtype = inputs_embeds.dtype

        cos, sin = _gemma4_rope_cos_sin(
            self.rotary_emb.inv_freq, pixel_position_ids, padded_head_dim, dtype
        )
        cos = convert(cos, device=device)
        sin = convert(sin, device=device)

        # One shared key-validity mask for the whole batch (padded_sdpa's contract):
        # correct for the batch=1 / uniform-padding case this has been validated
        # against; a batch mixing different valid-patch counts per row would need
        # padded_sdpa extended to a per-row mask.
        seq_len = attention_mask.shape[-1]
        key_valid = convert(attention_mask[0], device="cpu").bool()
        attn_mask = key_valid.unsqueeze(0).expand(seq_len, seq_len)

        orig_intermediate = config.intermediate_size
        padded_intermediate = math.ceil(orig_intermediate / BLOCK_SIZE) * BLOCK_SIZE

        hidden_states = inputs_embeds
        for layer in self.layers[: config.num_hidden_layers]:
            _prepare_attention(layer.self_attn, num_heads, orig_head_dim, padded_head_dim)
            _pad_mlp(layer, orig_intermediate, padded_intermediate)
            hidden_states = _run_layer(
                layer,
                hidden_states,
                cos,
                sin,
                attn_mask,
                num_heads,
                orig_head_dim,
                padded_head_dim,
            )

        from transformers.modeling_outputs import BaseModelOutputWithPast

        return BaseModelOutputWithPast(last_hidden_state=hidden_states)

    _forward._spyre_patched = True
    cls.forward = _forward  # ty: ignore[invalid-assignment]
    logger.info_once(
        "Spyre: patched Gemma4VisionEncoder to head_dim-padded rope (Pixtral's "
        "matmul-rotate form) + padded SDPA (pad L/D to 64, mask, crop)."
    )


def patch_rms_norm() -> None:
    """Give ``Gemma4RMSNorm`` the same treatment ``SpyreGemmaRMSNorm`` gives vLLM's:
    no fp32 promotion, and ``torch.rsqrt`` instead of ``torch.pow(x, -0.5)``.

    ``forward`` is what needs patching, not just ``_norm``: stock forward is
    ``self._norm(hidden_states.float())`` with a ``self.weight.float()`` scale, and
    both fp32 casts have to go. torch-spyre does not support dtype promotion
    (``custom_ops/rms_norm.py``), and an on-device fp16->fp32->fp16 round trip also
    leaves the result in a stick-tiling state a later eager elementwise op cannot
    broadcast against (see ``_padded_rms_norm``). ``pow`` additionally has no
    lowering in this fused-kernel context ("Invoked sdsc_fused_pow_0 which contains
    unimplemented operation pow"), and it is only there for JAX/Torch compiler
    parity per transformers' own comment -- ``rsqrt`` is exactly equivalent.

    Same trade the sibling Spyre norms already accept: expect small numerical
    differences from upstream.
    """
    try:
        from transformers.models.gemma4 import modeling_gemma4
    except ImportError:
        return

    cls = getattr(modeling_gemma4, "Gemma4RMSNorm", None)
    if cls is None or getattr(cls.forward, "_spyre_patched", False):
        return

    def forward(self, hidden_states):
        # The trailing cast is stock's `.type_as(hidden_states)`, and it is
        # load-bearing rather than cosmetic: these weights are fp32 (transformers
        # builds them at the default dtype), so an unguarded `* self.weight` would
        # promote the activation to fp32 -- the very thing torch-spyre does not
        # support, and a dtype mismatch against the fp16 projections downstream.
        dtype = hidden_states.dtype
        mean_squared = (hidden_states * hidden_states).mean(-1, keepdim=True) + self.eps
        normed_output = hidden_states * torch.rsqrt(mean_squared)
        if self.with_scale:
            normed_output = normed_output * self.weight.to(dtype)
        return normed_output.to(dtype)

    forward._spyre_patched = True
    cls.forward = forward  # ty: ignore[invalid-assignment]
    logger.info_once(
        "Spyre: Gemma4RMSNorm runs without fp32 promotion and uses torch.rsqrt "
        "instead of torch.pow(x, -0.5); expect small numerical differences."
    )


def patch_pooler() -> None:
    """Run ``Gemma4VisionPooler`` on CPU and hand its result back on device.

    The pooler is integer-geometry work, not arithmetic: a ``masked_fill`` of the
    padding patches (``aten::masked_fill_.Scalar`` has no Spyre kernel at all), a
    ``one_hot`` over floor-divided patch coordinates, and a ``max``/``all`` reduction
    to derive the validity mask. Same doctrine as every other gather-shaped op here
    (Pixtral's ``PatchMerger``, the patch-embed position lookup), and the same split
    hf-adapters#495 documents: "integer position lookup, spatial pooling, and the
    text-space projector stay on CPU".

    It is also cheap to move: the pooler reduces 2520 patches to 280 soft tokens, so
    only the smaller side crosses back.
    """
    try:
        from transformers.models.gemma4 import modeling_gemma4
    except ImportError:
        return

    cls = getattr(modeling_gemma4, "Gemma4VisionPooler", None)
    if cls is None or getattr(cls.forward, "_spyre_patched", False):
        return

    orig_forward = cls.forward

    def forward(self, hidden_states, pixel_position_ids, padding_positions, output_length=None):
        # Both outputs stay on CPU: the caller immediately does `pooled[mask]`
        # (`aten::index.Tensor_out`, no Spyre kernel), then the fp32 `standardize`
        # affine, then `embed_vision` -- `place_vision_tail_on_cpu` keeps that whole
        # tail host-side, so converting back here would just bounce it again.
        return orig_forward(
            self,
            convert(hidden_states, device="cpu"),
            convert(pixel_position_ids, device="cpu"),
            convert(padding_positions, device="cpu"),
            output_length,
        )

    forward._spyre_patched = True
    cls.forward = forward  # ty: ignore[invalid-assignment]
    logger.info_once("Spyre: Gemma4VisionPooler runs on CPU (masked_fill/one_hot geometry).")


def patch_accelerator_memory_info() -> None:
    """Fall back to host RAM for ``torch.accelerator.get_memory_info()``.

    ``Gemma4ForConditionalGeneration._process_image_input`` calls this to size its
    memory-safe encoder-chunking budget. Spyre (a CPU-based platform in vLLM's
    device-config sense) registers no accelerator memory-info hook, so the native
    call raises ``NotImplementedError`` unconditionally -- not Gemma4-vision-specific,
    but this is the first path in this codebase to hit it. ``psutil``'s host RAM
    is the correct substitute here: the dominant transient this budget guards
    against runs on CPU (this codebase's doctrine for the gather/pooling ops in
    Gemma4's vision tower), not on-device.
    """
    orig = torch.accelerator.get_memory_info
    if getattr(orig, "_spyre_patched", False):
        return

    def _get_memory_info(*args, **kwargs):
        try:
            return orig(*args, **kwargs)
        except NotImplementedError:
            import psutil

            vm = psutil.virtual_memory()
            return (vm.available, vm.total)

    _get_memory_info._spyre_patched = True
    torch.accelerator.get_memory_info = _get_memory_info  # ty: ignore[invalid-assignment]
    logger.info_once(
        "Spyre: torch.accelerator.get_memory_info() falls back to host RAM "
        "(psutil) when the native accelerator call is unimplemented."
    )


def patch_patch_embedder() -> None:
    """Run ``Gemma4VisionPatchEmbedder``'s position-embedding gather on CPU.

    ``F.embedding`` needs its index and weight tensors on the same device.
    ``pixel_position_ids`` arrives on CPU (``SpyreModelWrapper.embed_multimodal`` only
    moves floating-point multimodal inputs to Spyre; positions are int64) while
    ``position_embedding_table`` lives on Spyre with the rest of the model. Same
    doctrine as every other integer-gather op in this codebase (Pixtral's position
    lookup, the vision pooler): do the lookup on CPU, move the result.
    """
    try:
        from transformers.models.gemma4 import modeling_gemma4
    except ImportError:
        return

    cls = getattr(modeling_gemma4, "Gemma4VisionPatchEmbedder", None)
    if cls is None or getattr(cls._position_embeddings, "_spyre_patched", False):
        return

    def _position_embeddings(self, pixel_position_ids, padding_positions):
        device = self.position_embedding_table.device
        clamped_positions = convert(pixel_position_ids, device="cpu").clamp(min=0)
        table = convert(self.position_embedding_table, device="cpu")
        x_emb = F.embedding(clamped_positions[..., 0], table[0])
        y_emb = F.embedding(clamped_positions[..., 1], table[1])
        position_embeddings = x_emb + y_emb
        padding_cpu = convert(padding_positions, device="cpu")
        position_embeddings = torch.where(padding_cpu.unsqueeze(-1), 0.0, position_embeddings)
        return convert(position_embeddings, device=device)

    _position_embeddings._spyre_patched = True
    cls._position_embeddings = _position_embeddings  # ty: ignore[invalid-assignment]
    logger.info_once(
        "Spyre: Gemma4VisionPatchEmbedder position-embedding gather runs on CPU."
    )


def place_vision_tail_on_cpu(model: torch.nn.Module) -> None:
    """Keep everything after the encoder host-side: standardize buffers + embed_vision.

    ``_process_image_input`` runs `pooled[valid_mask]` (``aten::index.Tensor_out``,
    which has no Spyre kernel), then the fp32 ``standardize`` affine, then the
    text-space projection -- all on the pooler's output. Since ``patch_pooler``
    already returns CPU, these operands have to follow, or each step trips a
    device mismatch. hf-adapters#495 draws the line in the same place: "integer
    position lookup, spatial pooling, and the text-space projector stay on CPU".

    The projector is also the cheap end of the tower (1152 -> text hidden over ~280
    pooled soft tokens, versus 2520 patches through 27 encoder layers), and its
    output goes straight into the CPU-side image-embedding merge in
    ``SpyreModelWrapper.embed_input_ids`` -- so this avoids a round trip rather than
    adding one.
    """
    tower = getattr(model, "vision_tower", None)
    if tower is not None and getattr(tower.config, "standardize", False):
        for name in ("std_bias", "std_scale"):
            buf = getattr(tower, name, None)
            if buf is not None and buf.device.type != "cpu":
                setattr(tower, name, buf.to("cpu"))
    embed_vision = getattr(model, "embed_vision", None)
    if embed_vision is not None:
        embed_vision.to("cpu")
    logger.info_once(
        "Spyre: Gemma 4 vision tail (standardize buffers + embed_vision) placed on CPU."
    )


def apply(model: torch.nn.Module, device: torch.device) -> None:
    """Install every Gemma 4 vision-tower workaround."""
    del device
    patch_accelerator_memory_info()
    patch_rms_norm()
    patch_patch_embedder()
    patch_pooler()
    patch_vision_encoder()
    place_vision_tail_on_cpu(model)
