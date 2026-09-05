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

"""Encoder-only (bidirectional) self-attention for Spyre without a KV cache.

Selected by ``TorchSpyrePlatform.get_attn_backend_cls`` for ENCODER/ENCODER_ONLY
layers. Operates on direct Q/K/V tensors rather than the paged KV-cache path.

Ragged→dense packing uses host-built indices + ``index_select`` (gather).
Pad slots gather a trailing zero row so the dense batch stays zeros in the
padding region.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionLayer

from spyre_inference import envs
from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionBackend,
    SpyreAttentionImpl,
    SpyreAttentionMetadata,
    SpyrePagedKVCache,
    _maybe_compile,
)
from spyre_inference.v1.pool import select_rows
from spyre_inference.v1.worker.spyre_shape_bucketer import (
    default_encoder_len_buckets,
    next_bucket,
    pick_encoder_attention_shape,
    pooling_warmup_shapes,
)

logger = init_logger(__name__)

# Pad seq length *and* head dim to the Spyre stick (64 fp16 elements).
# L-aligned keeps P·V's K stick-aligned; D-aligned keeps QKᵀ's K stick-aligned
# so Inductor never enters insert_bmm_padding (torch-spyre KeyError: 'val' on
# FX nodes missing meta["val"] when padding MiniLM's head_size=32).
ENCODER_SEQ_ALIGNMENT = 64


def _align_up(n: int, align: int = ENCODER_SEQ_ALIGNMENT) -> int:
    return max(align, (n + align - 1) // align * align)


def host_pack_indices(
    q_starts: list[int],
    lengths: list[int],
    aligned_len: int,
    pad_row: int,
) -> torch.Tensor:
    """Build ``[B, L]`` int64 row indices; pad slots point at ``pad_row``."""
    batch = len(q_starts)
    indices = torch.full((batch, aligned_len), pad_row, dtype=torch.int64)
    for s, (start, length) in enumerate(zip(q_starts, lengths)):
        if length > 0:
            indices[s, :length] = torch.arange(start, start + length, dtype=torch.int64)
    return indices


def host_unpack_indices(
    q_starts: list[int],
    query_lens: list[int],
    aligned_len: int,
    num_tokens: int,
) -> torch.Tensor:
    """Build ``[T]`` int64 indices from flat padded ``[B*L]`` back to tokens.

    ``num_tokens`` may exceed the real count; unfilled entries stay ``0``
    (a safe row to read — nothing downstream reads those output rows).
    """
    indices = torch.zeros(num_tokens, dtype=torch.int64)
    for s, (start, length) in enumerate(zip(q_starts, query_lens)):
        if length > 0:
            base = s * aligned_len
            indices[start : start + length] = torch.arange(base, base + length, dtype=torch.int64)
    return indices


def _build_seq_row_index(start: int, real_len: int, aligned_len: int) -> torch.Tensor:
    """Absolute row indices for one sequence, into the flat batch tensor.

    Positions past ``real_len`` clamp to the last real row rather than a
    sentinel, mirroring the decoder's ``_build_query_row_tables``. A compiled
    kernel reads via ``index_select`` here, never a plain slice, since a
    compiled region reads a view from offset 0 regardless of storage_offset
    (torch-spyre#3770).
    """
    pos = torch.arange(aligned_len, dtype=torch.int64)
    return start + torch.minimum(pos, torch.tensor(real_len - 1, dtype=torch.int64))


def _pad_head_dim_to_stick(flat: torch.Tensor, head_size_padded: int) -> torch.Tensor:
    """Pad last dim to a stick. MiniLM ``[T,H,32]`` cannot ``F.pad`` on Spyre."""
    head_size = flat.shape[-1]
    if head_size == head_size_padded:
        return flat
    device = flat.device
    if device.type == "spyre":
        flat = convert(flat, "cpu")
    flat = F.pad(flat, (0, head_size_padded - head_size))
    if device.type == "spyre":
        flat = convert(flat, device)
    return flat


def gather_pack(
    flat: torch.Tensor,
    pack_indices: torch.Tensor,
    head_size_padded: int,
) -> torch.Tensor:
    """Pack varlen ``[T, H, D]`` → padded ``[B, H, L, Dp]`` via ``index_select``.

    ``pack_indices`` is host ``[B, L]`` int64. Head dim is padded to the stick
    (CPU for MiniLM D=32), then a zero token row is ``F.pad``'d on-device.
    """
    batch, aligned_len = pack_indices.shape
    _t, num_heads, _d = flat.shape
    flat = _pad_head_dim_to_stick(flat, head_size_padded)
    flat_ext = F.pad(flat, (0, 0, 0, 0, 0, 1))
    gathered = select_rows(flat_ext, pack_indices)  # [B*L, H, Dp]
    packed = gathered.view(batch, aligned_len, num_heads, head_size_padded)
    return packed.permute(0, 2, 1, 3).contiguous()


def gather_unpack(
    attn_out: torch.Tensor,
    unpack_indices: torch.Tensor,
    head_size: int,
) -> torch.Tensor:
    """Unpack padded ``[B, H, L, Dp]`` → flat ``[T, H, D]`` via ``index_select``."""
    batch, num_heads, aligned_len, head_size_padded = attn_out.shape
    tokens = attn_out.permute(0, 2, 1, 3).contiguous()
    flat_padded = tokens.reshape(batch * aligned_len, num_heads, head_size_padded)
    gathered = select_rows(flat_padded, unpack_indices)
    if gathered.shape[-1] == head_size:
        return gathered
    # Crop is a slice, not pad. D=32 is half a stick; do it on CPU.
    if gathered.device.type == "spyre":
        gathered = convert(gathered, "cpu")
    return gathered[..., :head_size].contiguous()


def _create_compilable_encoder_attn(
    head_size_padded: int,
    enable_gqa: bool,
):
    """Factory for one pack→SDPA kernel, closed over per-layer constants.

    Mirrors the decoder's ``_create_compilable_page_attn``: everything that
    never varies per call for a given attention layer (head size, GQA) is a
    closure constant, not a runtime argument. Fusing pack and SDPA into one
    function lets ``_maybe_compile`` wrap the whole sequence in a single
    ``torch.compile``, so Dynamo checks guards once per call instead of once
    per inner op — the latter is what makes eager per-op dispatch on Spyre
    recompile on content changes alone (spyre-inference#775 follow-up).

    ``gather_unpack`` stays outside this kernel and runs eager, called
    separately by ``forward()``: compiling it together with SDPA in one graph
    hits a torch-spyre layout-propagation limit ("Incompatible host_size and
    dim_order", `torch_spyre/_inductor/propagate_layouts.py`) regardless of
    mask or GQA — isolated experimentally, not yet root-caused inside
    torch-spyre. Pack+SDPA alone compiles cleanly; that's most of the benefit,
    since it collapses 3 gather_pack calls plus SDPA from 4+ separately
    guard-checked eager dispatches down to 1.
    """

    def encoder_attn(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_pack_idx: torch.Tensor,
        kv_pack_idx: torch.Tensor,
        mask: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        q_batched = gather_pack(query, q_pack_idx, head_size_padded)
        k_batched = gather_pack(key, kv_pack_idx, head_size_padded)
        v_batched = gather_pack(value, kv_pack_idx, head_size_padded)

        sdpa_kwargs: dict = {"is_causal": False, "scale": scale}
        if enable_gqa:
            sdpa_kwargs["enable_gqa"] = True
        return F.scaled_dot_product_attention(
            q_batched, k_batched, v_batched, attn_mask=mask, **sdpa_kwargs
        )

    return encoder_attn


def _create_compilable_encoder_seq_attn(
    head_size_padded: int,
    enable_gqa: bool,
    store: bool = False,
):
    """Factory for one sequence's pack->SDPA kernel, mirroring the decoder's
    ``_create_compilable_page_attn``: per-sequence rather than per-batch, so
    the only shape axis that varies is ``aligned_len`` (the cache key in
    ``_get_encoder_seq_attn_fn``) instead of the dense grid's ``(B, L)`` pair.

    Every tensor here is sized on ``aligned_len`` (a bucket), never on the real
    prompt length: under ``dynamic=False`` a real-length-dependent shape would
    add a Dynamo specialization per distinct prompt length, and Dynamo walks the
    resulting guard chain linearly on every later call, so throughput decays as
    more distinct lengths are seen.

    With ``store``, the write-back is fused here as ``index_copy_`` over the
    full ``aligned_len`` extent, matching the decoder's ``store_mode="index"``.
    That keeps the real length out of the caller's write too, and it must be
    fused rather than eager: eager ``index_copy_`` rejects an int32 index and
    silently falls back to CPU with an int64 one (see
    ``SpyreAttentionImpl._reshape_fn``). Rows past the real length duplicate the
    last real row's index, and the kv-only mask (see
    ``build_seq_attention_mask``) makes their values identical to that row's, so
    ``index_copy_``'s undefined write order for duplicate indices is harmless.
    """

    def encoder_seq_attn(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_row_idx: torch.Tensor,
        kv_row_idx: torch.Tensor,
        mask: torch.Tensor,
        scale: float,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # query/key/value are the whole padded batch tensor; gather (via
        # select_rows, which picks int32+convert on Spyre vs. int64 on CPU)
        # reads this sequence's own rows rather than relying on a slice's
        # storage_offset (torch-spyre#3770).
        q = _pad_head_dim_to_stick(select_rows(query, q_row_idx), head_size_padded)
        k = _pad_head_dim_to_stick(select_rows(key, kv_row_idx), head_size_padded)
        v = _pad_head_dim_to_stick(select_rows(value, kv_row_idx), head_size_padded)
        q = q.unsqueeze(0).transpose(1, 2)  # [1, H, L, Dp]
        k = k.unsqueeze(0).transpose(1, 2)
        v = v.unsqueeze(0).transpose(1, 2)

        sdpa_kwargs: dict = {"is_causal": False, "scale": scale}
        if enable_gqa:
            sdpa_kwargs["enable_gqa"] = True
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, **sdpa_kwargs)
        # [1, H, L, Dp] -> [L, H, Dp]. The trailing .reshape (not .view) forces
        # a contiguous copy after the transpose, mirroring the decoder kernel's
        # identical reshape/transpose/reshape.
        num_heads_out = attn_out.shape[1]
        head_dim_out = attn_out.shape[-1]
        result = attn_out.transpose(1, 2).reshape(-1, num_heads_out, head_dim_out)
        if store:
            assert out is not None
            out.index_copy_(0, q_row_idx, result)
            return out
        return result

    return encoder_seq_attn


def build_attention_mask(
    num_seqs: int,
    aligned_len: int,
    query_lens: list[int],
    kv_lens: list[int],
    dtype: torch.dtype,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Additive mask ``[B, 1, L, L]``: 0 where attend, ``-inf`` elsewhere.

    Built on the host (vectorized ``lt`` + nested ``where``), then ``convert``'d.
    On-device materialization is not stick-safe: Spyre cannot produce bool from
    int32 ``lt``, and cannot broadcast ``where`` of ``[B,1,L,1]`` × ``[B,1,1,L]``
    into ``[B,1,L,L]`` (no stick-scatter).
    """
    if device is None:
        device = torch.device("cpu")
    elif not isinstance(device, torch.device):
        device = torch.device(device)
    if num_seqs != len(query_lens):
        raise ValueError(f"num_seqs={num_seqs} != len(query_lens)={len(query_lens)}")

    q_len = torch.tensor(query_lens, dtype=torch.int32)
    kv_len = torch.tensor(
        [min(q, k) for q, k in zip(query_lens, kv_lens)],
        dtype=torch.int32,
    )
    q_pos = torch.arange(aligned_len, dtype=torch.int32)
    kv_pos = torch.arange(aligned_len, dtype=torch.int32)
    zeros = torch.zeros((), dtype=dtype)
    neg_inf = torch.tensor(torch.finfo(dtype).min, dtype=dtype)

    q_ok = (q_pos.unsqueeze(0) < q_len.unsqueeze(1)).unsqueeze(1).unsqueeze(-1)
    k_ok = (kv_pos.unsqueeze(0) < kv_len.unsqueeze(1)).unsqueeze(1).unsqueeze(2)
    mask = torch.where(q_ok, torch.where(k_ok, zeros, neg_inf), neg_inf)
    if device.type == "spyre":
        return convert(mask, device)
    return mask.to(device)


def build_seq_attention_mask(
    aligned_len: int,
    kv_len: int,
    dtype: torch.dtype,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Single-sequence additive mask ``[1, 1, L, L]``, keyed on KV validity only.

    Deliberately does NOT mask padded *query* rows, unlike
    ``build_attention_mask``. Padded query positions gather the last real query
    row (``_build_seq_row_index`` clamps them), so leaving their row unmasked
    makes their output identical to the last real row's — the invariant the
    fused ``index_copy_`` store relies on, and the same one the decoder's mask
    tiles maintain. Masking them instead would leave them a uniform average of
    V (``finfo.min`` everywhere softmaxes to uniform, not NaN), which the store
    would then scatter over the last real row.

    Real query rows are unaffected: for those, ``build_attention_mask`` already
    reduces to exactly this KV-validity row.

    Materialized as a full ``[1, 1, L, L]`` on the host, not a broadcastable
    ``[1, 1, 1, L]``: Spyre cannot broadcast-scatter a mask across the query
    axis on device (see ``build_attention_mask``).
    """
    if device is None:
        device = torch.device("cpu")
    elif not isinstance(device, torch.device):
        device = torch.device(device)

    kv_pos = torch.arange(aligned_len, dtype=torch.int32)
    k_ok = kv_pos < kv_len
    row = torch.where(
        k_ok,
        torch.zeros((), dtype=dtype),
        torch.tensor(torch.finfo(dtype).min, dtype=dtype),
    )
    mask = row.unsqueeze(0).expand(aligned_len, aligned_len).contiguous()
    mask = mask.unsqueeze(0).unsqueeze(0)
    if device.type == "spyre":
        return convert(mask, device)
    return mask.to(device)


class SpyreEncoderAttentionImpl(SpyreAttentionImpl):
    """Bidirectional encoder self-attention (no KV cache).

    The platform selects this impl for ENCODER/ENCODER_ONLY layers (see
    ``TorchSpyrePlatform.get_attn_backend_cls``), so there is no per-call
    ``attn_type`` branch. Setup is shared with the paged decoder impl; forward
    packs with gather (``index_select``), runs one batched SDPA on Spyre, then
    unpacks with gather.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # get_current_vllm_config() only works at construction time; forward()
        # runs through a custom-op boundary that loses the context.
        cfg = get_current_vllm_config()
        self._cached_max_num_seqs = cfg.scheduler_config.max_num_seqs
        self._cached_max_model_len = cfg.model_config.max_model_len
        self._cached_max_num_batched_tokens = cfg.scheduler_config.max_num_batched_tokens
        self._cached_encoder_shapes = pooling_warmup_shapes(
            max_num_seqs=self._cached_max_num_seqs,
            max_model_len=self._cached_max_model_len,
            max_num_batched_tokens=self._cached_max_num_batched_tokens,
            len_ladder=default_encoder_len_buckets(self._cached_max_model_len),
        )
        # One compiled pack→SDPA kernel per (batch_bucket, aligned_len),
        # mirroring the decoder's self._attn_fns/_decode_fns.
        self._encoder_attn_fns: dict[tuple[int, int], object] = {}
        # Per-sequence loop path (default, see envs.SPYRE_BUCKETED_ENCODE):
        # one compiled kernel per (aligned_len, store), mirroring the decoder's
        # default per-sequence self._attn_fns cache.
        self._encoder_seq_attn_fns: dict[tuple[int, bool], object] = {}

    def _get_encoder_seq_attn_fn(
        self,
        aligned_len: int,
        head_size_padded: int,
        enable_gqa: bool,
        store: bool,
    ):
        key = (aligned_len, store)
        if key not in self._encoder_seq_attn_fns:
            self._encoder_seq_attn_fns[key] = _maybe_compile(
                _create_compilable_encoder_seq_attn(head_size_padded, enable_gqa, store=store),
                self._compile_attn,
            )
        return self._encoder_seq_attn_fns[key]

    def _get_encoder_attn_fn(
        self,
        batch_bucket: int,
        aligned_len: int,
        head_size_padded: int,
        enable_gqa: bool,
    ):
        key = (batch_bucket, aligned_len)
        if key not in self._encoder_attn_fns:
            self._encoder_attn_fns[key] = _maybe_compile(
                _create_compilable_encoder_attn(head_size_padded, enable_gqa),
                self._compile_attn,
            )
        return self._encoder_attn_fns[key]

    def forward(  # ty: ignore[invalid-method-override]
        self,
        layer: AttentionLayer,
        query: torch.Tensor,  # [num_tokens, num_heads, head_size]
        key: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
        value: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
        kv_cache: SpyrePagedKVCache,
        attn_metadata: SpyreAttentionMetadata,
        output: torch.Tensor,  # [num_tokens, num_heads, head_size]
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del layer, kv_cache, output_scale, output_block_scale
        if attn_metadata is None:
            return output

        # query/key/value/output are padded to the runner's warmed body-bucket
        # size, not num_actual_tokens. Keep that shape or index_select
        # recompiles per request.
        n = attn_metadata.num_actual_tokens
        padded_tokens = query.shape[0]

        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        num_seqs = attn_metadata.num_seqs
        scale = self.scale

        qsl = query_start_loc.cpu()
        # query_start_loc is a cumulative offset array of length num_seqs + 1;
        # diff() yields the num_seqs per-sequence query lengths in one pass.
        q_starts = qsl[:-1].tolist()
        query_lens = torch.diff(qsl).tolist()
        kv_lens = seq_lens.cpu().tolist()

        num_heads = query.shape[1]
        num_kv_heads = key.shape[1]
        head_size = query.shape[2]
        # Pad D to the stick when models use a smaller head dim (MiniLM=32).
        # Zero-pad is exact for SDPA: padded Q/K slots add 0 to scores; cropped
        # output drops the zero V channels. Keeps self.scale = 1/sqrt(real D).
        head_size_padded = _align_up(head_size)
        enable_gqa = num_kv_heads != num_heads

        target_device = output.device
        # Keep activations on the SDPA device; pack/unpack via index_select.
        if query.device.type != target_device.type:
            query = convert(query, target_device.type)
            key = convert(key, target_device.type)
            value = convert(value, target_device.type)

        # MiniLM D=32: flatten to [rows, H*D] (384 = 6 sticks) so the write is
        # aligned. Shared by both paths below.
        use_flat_write = target_device.type == "spyre" and head_size % ENCODER_SEQ_ALIGNMENT != 0

        if not envs.SPYRE_BUCKETED_ENCODE:
            # Per-sequence loop (default): mirrors the decoder's default
            # _online_softmax_attention loop instead of the dense (B, L) grid
            # below. Each sequence picks its own smallest sufficient
            # aligned_len, so mixed-length batches don't all pay the batch
            # max's cost, and there is no 2D (B, L) bucket ladder to warm or
            # to have gaps in.
            len_ladder = default_encoder_len_buckets(self._cached_max_model_len)
            # Fuse the write-back into the compiled kernel when the buffer allows
            # it, mirroring the decoder's fused_store_ok gate. This is what keeps
            # the real prompt length out of every device tensor shape; the eager
            # fallback below cannot (eager index_copy_ falls back to CPU), so it
            # reintroduces per-length specializations and is only for buffers the
            # fused path can't accept.
            fused_store_ok = (
                self._compile_attn
                and head_size == head_size_padded
                and output.dtype == query.dtype
                # A compiled kernel reads its arguments from offset 0: torch-spyre#3770.
                and output.storage_offset() == 0
                and output.is_contiguous()
            )
            if self._compile_attn and not fused_store_ok:
                # Only actionable when compiling: in eager mode there are no
                # specializations for a real-length shape to multiply.
                logger.warning_once(
                    "Encoder attention is writing back eagerly despite being "
                    "compiled (head_size=%d/%d, dtype=%s/%s, offset=%d, "
                    "contiguous=%s). The eager write is sized on each request's "
                    "real prompt length, so every distinct length adds a "
                    "torch.compile specialization and throughput decays as more "
                    "lengths are seen.",
                    head_size,
                    head_size_padded,
                    query.dtype,
                    output.dtype,
                    output.storage_offset(),
                    output.is_contiguous(),
                )

            for seq_idx in range(num_seqs):
                q_start = q_starts[seq_idx]
                real_len = query_lens[seq_idx]
                kv_real_len = min(real_len, kv_lens[seq_idx])
                aligned_len = next_bucket(real_len, len_ladder)

                # int32 serves both the gather and the fused index_copy_ store,
                # exactly as the decoder's query_row_tables do.
                q_row_idx = convert(
                    _build_seq_row_index(q_start, real_len, aligned_len).to(torch.int32),
                    target_device,
                )
                # Encoder-only layers have no KV cache: K/V are computed from the
                # same tokens as Q in this same pass, so seq_lens == query_lens and
                # the K/V rows are the Q rows. Reuse the tensor rather than paying a
                # second host build plus H2D transfer per sequence per layer. The
                # guard keeps the general path correct if seq_lens is ever shorter.
                if kv_real_len == real_len:
                    kv_row_idx = q_row_idx
                else:
                    kv_row_idx = convert(
                        _build_seq_row_index(q_start, kv_real_len, aligned_len).to(torch.int32),
                        target_device,
                    )
                mask = build_seq_attention_mask(
                    aligned_len,
                    kv_real_len,
                    dtype=query.dtype,
                    device=target_device,
                )

                encoder_seq_attn_fn = self._get_encoder_seq_attn_fn(
                    aligned_len, head_size_padded, enable_gqa, store=fused_store_ok
                )
                attn_out = encoder_seq_attn_fn(
                    query,
                    key,
                    value,
                    q_row_idx,
                    kv_row_idx,
                    mask,
                    scale,
                    out=output if fused_store_ok else None,
                )
                if fused_store_ok:
                    # The kernel wrote `output` itself; attn_out is that buffer.
                    continue

                if attn_out.shape[-1] != head_size:
                    # Crop is a slice, not pad. D=32 is half a stick; do it on CPU.
                    if attn_out.device.type == "spyre":
                        attn_out = convert(attn_out, "cpu")
                    attn_out = attn_out[..., :head_size].contiguous()
                result = attn_out[:real_len]
                if result.dtype != output.dtype:
                    result = convert(result, dtype=output.dtype)

                if use_flat_write:
                    if result.device.type == "spyre":
                        result = convert(result, "cpu")
                    src = convert(
                        result.reshape(real_len, -1).contiguous(),
                        target_device.type,
                        output.dtype,
                    )
                    output[q_start : q_start + real_len].reshape(real_len, -1).copy_(src)
                else:
                    if result.device.type != output.device.type:
                        result = convert(result, output.device)
                    output[q_start : q_start + real_len] = result

            return output

        # Batched (B, L) dense-grid path (opt-in via SPYRE_BUCKETED_ENCODE=1).
        max_len = max(query_lens, default=0)
        pair = pick_encoder_attention_shape(
            num_seqs,
            max_len,
            self._cached_encoder_shapes,
            self._cached_max_num_seqs,
            self._cached_max_model_len,
            self._cached_max_num_batched_tokens,
        )
        if pair is not None:
            batch_bucket, aligned_len = pair
        else:
            batch_bucket, aligned_len = num_seqs, _align_up(max_len)
            # No warmed (B, L) cell covers this batch — SDPA runs at a shape
            # torch.compile has never seen, which recompiles on this request's
            # critical path. Recurring hits usually mean max_num_seqs and the
            # token budget disagree (spyre-inference#775).
            logger.warning_once(
                "No warmed encoder attention shape covers num_seqs=%d, "
                "max_query_len=%d; falling back to (%d, %d), which triggers a "
                "runtime recompile. Widen --max-num-batched-tokens or lower "
                "--max-num-seqs so every batch bucket has a warmed cell.",
                num_seqs,
                max_len,
                batch_bucket,
                aligned_len,
            )
        orig_q_starts = q_starts
        orig_query_lens = query_lens
        if batch_bucket > num_seqs:
            q_starts = q_starts + [n] * (batch_bucket - num_seqs)
            query_lens = query_lens + [0] * (batch_bucket - num_seqs)
            kv_lens = kv_lens + [0] * (batch_bucket - num_seqs)

        pad_row = padded_tokens  # index of the appended zero row in gather_pack
        q_pack_idx = host_pack_indices(q_starts, query_lens, aligned_len, pad_row)
        # K/V may be shorter than Q when seq_lens < query_lens; still use q_starts.
        kv_pack_lens = [min(q, k) for q, k in zip(query_lens, kv_lens)]
        kv_pack_idx = host_pack_indices(q_starts, kv_pack_lens, aligned_len, pad_row)
        unpack_idx = host_unpack_indices(orig_q_starts, orig_query_lens, aligned_len, padded_tokens)

        # Move index tensors to the SDPA device *before* the compiled kernel
        # runs, matching the decoder's own metadata-mirroring pattern (e.g.
        # _build_query_row_tables). select_rows (inside gather_pack/gather_unpack)
        # does this same int32-then-convert step itself when handed a CPU
        # tensor; converting here makes that inner step a no-op passthrough
        # instead of a second, redundant device transfer inside the compiled
        # region.
        q_pack_idx = convert(q_pack_idx.to(torch.int32), target_device)
        kv_pack_idx = convert(kv_pack_idx.to(torch.int32), target_device)
        unpack_idx = convert(unpack_idx.to(torch.int32), target_device)

        mask = build_attention_mask(
            batch_bucket,
            aligned_len,
            query_lens,
            kv_lens,
            dtype=query.dtype,
            device=target_device,
        )

        # One compiled pack→SDPA kernel per (batch_bucket, aligned_len): fusing
        # pack and SDPA means Dynamo checks guards once per call instead of
        # once per inner op (spyre-inference#775 follow-up). gather_unpack
        # stays eager, outside the compiled region (see
        # _create_compilable_encoder_attn's docstring for why).
        encoder_attn_fn = self._get_encoder_attn_fn(
            batch_bucket,
            aligned_len,
            head_size_padded,
            enable_gqa=enable_gqa,
        )
        attn_out = encoder_attn_fn(query, key, value, q_pack_idx, kv_pack_idx, mask, scale)
        result = gather_unpack(attn_out, unpack_idx, head_size)
        if result.dtype != output.dtype:
            result = convert(result, dtype=output.dtype)

        if use_flat_write:
            if result.device.type == "spyre":
                result = convert(result, "cpu")
            src = convert(
                result.reshape(padded_tokens, -1).contiguous(), target_device.type, output.dtype
            )
            output.reshape(padded_tokens, -1).copy_(src)
        else:
            if result.device.type != output.device.type:
                result = convert(result, output.device)
            output.copy_(result)

        return output


class SpyreEncoderAttentionBackend(SpyreAttentionBackend):
    """Encoder-only (no KV cache) variant of the Spyre backend."""

    # These layers have no KV cache, but vLLM still hands encoder-only specs a
    # zero-filled slot mapping, so upstream must skip `unified_kv_cache_update` entirely.
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_impl_cls() -> type[SpyreEncoderAttentionImpl]:
        return SpyreEncoderAttentionImpl
