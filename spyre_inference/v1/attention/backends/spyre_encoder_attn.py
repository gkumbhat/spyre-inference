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

Dense per-sequence attention over the packed ``[T, H, D]`` list; request
boundaries ride in int32 row-index tables, so a Spyre card never has to do
offset arithmetic on *shapes* -- offsets are data. Unlike the decoder, there is
no paged KV cache forcing a block-wise walk: a sequence's whole K/V already
fits in one gathered tensor, so attention is one dense
``QKᵀ -> +mask -> softmax -> ·V``, not an online-softmax loop. That was blocked
on torch-spyre#4526 (Inductor rewrites a visible ``matmul + mask + softmax +
matmul`` into an ``F.sdpa``-equivalent that drops the additive mask on Spyre);
now that it is fixed, the dense form is safe.

Three separate compiled functions, not one, and that split is the point:

- ``_encoder_gather_kernel`` pulls one sequence's rows out of the step's full
  body buffer via ``index_select``.
- ``_encoder_dense_attn_kernel`` does the actual attention math on those
  already-gathered, sequence-sized tensors.
- ``_encoder_store_kernel`` scatters the result back with ``index_copy_``.

Under ``dynamic=False`` Dynamo guards on every argument's exact shape, not just
the ones the graph's output depends on. A step's Q/K/V/output buffers are sized
to the *body token bucket* (``buffer_rows``), which varies across the whole
compile_sizes ladder -- if the gather, the attention math, and the store were
one function, that function's cache key would include ``buffer_rows``, and the
expensive attention graph would recompile once per ``(buffer_rows, extent)``
pair instead of once per ``extent``. Splitting gather/store (cheap: a handful
of ops, keyed on ``buffer_rows``) away from the attention math (the expensive
part, keyed only on the sequence's own padded length) collapses that back down
to one compile per distinct length -- mirroring the decoder's own
``_index_copy_kernel``, which is "compiled alone" for the same reason.

Two torch-spyre bugs still shape the design. A compiled region reads its
arguments from offset 0 and ignores ``storage_offset`` (#3770), so a sequence
is gathered with ``index_select`` rather than sliced. A gather that selects its
whole source faults the card (#4033), so an identity gather is skipped instead.

No fallbacks: torch-spyre has no on-device ``arange``, ``full``, or similar
construction op, so every index and mask tensor this file touches -- row
tables, mask tiles -- is built on the host and ``convert``'d once, then cached
and reused across sequences, layers, and steps. None of that construction
happens inside a compiled kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from vllm.v1.attention.backend import AttentionLayer

from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionBackend,
    SpyreAttentionImpl,
    SpyreAttentionMetadata,
    SpyrePagedKVCache,
    _call_kernel,
)

# KV block width, in tokens. One Spyre stick of fp16. The only remaining use
# is as the length-bucket granularity -- there is no per-block loop anymore.
ENCODER_BLOCK_SIZE = 64


def _blocks_for(length: int) -> int:
    """Block count covering ``length`` tokens, rounded up to a power of two.

    Encoder self-attention has ``q_len == kv_len``, so this single number fixes
    the sequence's padded extent (``num_blocks * ENCODER_BLOCK_SIZE``) -- the
    attention kernel's cache has one shape axis, not two. Rounding to a power
    of two keeps that axis to a handful of buckets, matching the ladder
    ``_powers_of_two_up_to`` gives decoder attention.
    """
    blocks = max(1, (length + ENCODER_BLOCK_SIZE - 1) // ENCODER_BLOCK_SIZE)
    bucket = 1
    while bucket < blocks:
        bucket *= 2
    return bucket


def _encoder_gather_kernel(query, key, value, row_index):
    """Pull one sequence's rows out of the step's full body buffer.

    Compiled alone, cheap, and keyed on ``(query.shape[0], row_index.shape[0])``
    -- i.e. on ``buffer_rows`` (the body bucket) as well as the sequence's own
    padded length. That is fine: this graph is a handful of ops, so recompiling
    it once per ``(buffer_rows, extent)`` pair costs little. Keeping it
    separate from the attention math is what keeps *that* graph off this
    dependency -- see the module docstring.
    """
    q_rows = query.index_select(0, row_index)
    k_rows = key.index_select(0, row_index)
    v_rows = value.index_select(0, row_index)
    return q_rows, k_rows, v_rows


_encoder_gather_compiled = torch.compile(_encoder_gather_kernel, dynamic=False)


def _encoder_dense_attn_kernel(
    q_rows,
    k_rows,
    v_rows,
    mask,
    scale,
    num_heads,
    num_kv_heads,
    head_size,
):
    """Dense masked self-attention over one already-gathered sequence.

    ``head_size`` need not be stick-aligned: the kernel never slices a matmul
    operand, so Inductor's ``insert_bmm_padding`` can pad the contraction
    dimension itself.

    Expected shapes:
        q_rows: [extent, num_heads, head_size], one sequence, already gathered
            (or passed straight through when the buffer already holds exactly
            this sequence -- see ``needs_gather`` at the call site).
        k_rows/v_rows: [extent, num_kv_heads, head_size], likewise.
        mask: [num_kv_heads, num_queries_per_kv, 1, extent] additive mask. The
            query axis is 1 because an encoder mask depends only on the KV
            column: every query row, real or padding, attends to exactly the
            real keys. That is what makes the padding rows exact duplicates of
            the last real row, which in turn makes the duplicate-index store
            downstream harmless.

    Returns [extent, num_heads, head_size].
    """
    num_queries_per_kv = num_heads // num_kv_heads
    extent = q_rows.shape[0]

    q = (
        (q_rows.unsqueeze(0).transpose(1, 2) * scale)
        .reshape(num_kv_heads, num_queries_per_kv, extent, head_size)
    )
    k = k_rows.unsqueeze(0).transpose(1, 2).reshape(num_kv_heads, 1, extent, head_size)
    v = v_rows.unsqueeze(0).transpose(1, 2).reshape(num_kv_heads, 1, extent, head_size)

    scores = torch.matmul(q, k.transpose(-2, -1)) + mask
    probs = torch.softmax(scores, dim=-1)
    attn = torch.matmul(probs, v)

    attn = attn.reshape(1, num_heads, extent, head_size).transpose(1, 2)
    return attn.reshape(extent, num_heads, head_size)


_encoder_dense_attn_compiled = torch.compile(_encoder_dense_attn_kernel, dynamic=False)


def _encoder_store_kernel(out, row_index, attn):
    """Scatter one sequence's attention output back into the step's output buffer.

    Compiled alone -- mirrors the decoder's ``_index_copy_kernel``: a tiny
    mutation, not fused with the attention math. Keyed on ``out.shape[0]``
    (``buffer_rows``), which is fine because this graph is one op.
    """
    out.index_copy_(0, row_index, attn)
    return out


_encoder_store_compiled = torch.compile(_encoder_store_kernel, dynamic=False)


def _create_dense_attn_kernel(num_heads: int, num_kv_heads: int, head_size: int):
    """Test helper: bind the non-tensor args the way a forward call would."""

    def specialized_dense_attn_kernel(q_rows, k_rows, v_rows, mask, scale):
        return _encoder_dense_attn_kernel(
            q_rows, k_rows, v_rows, mask, scale, num_heads, num_kv_heads, head_size
        )

    return specialized_dense_attn_kernel


@dataclass
class EncoderSeqPlan:
    """Everything the kernels need for one request, built once per step."""

    start: int
    query_len: int
    needs_gather: bool
    row_table: torch.Tensor
    mask: torch.Tensor


def encoder_index_dtype(device: torch.device) -> torch.dtype:
    """int32 on Spyre, which has no int64 and whose compiled ``index_copy_``
    takes it; int64 everywhere else, where eager ``index_copy_`` rejects int32.
    """
    return torch.int32 if device.type == "spyre" else torch.int64


def encoder_row_table(start: int, query_len: int, extent: int, dtype: torch.dtype) -> torch.Tensor:
    """One sequence's absolute rows, pad lanes clamped to its last real row.

    ``extent`` is a multiple of ``ENCODER_BLOCK_SIZE`` and therefore of
    ``INT32_ELEMS_PER_STICK``, so the table is stick-aligned with no extra pad.
    """
    return torch.arange(extent, dtype=dtype).clamp(max=query_len - 1) + start


def _const_tile(
    masked: bool,
    num_kv_heads: int,
    num_queries_per_kv: int,
    dtype: torch.dtype,
    device: torch.device,
    cache: dict | None,
) -> torch.Tensor:
    """Shared all-zero or all-masked 64-wide tile. Read-only: every mask aliases it."""
    cache_key = (masked, num_kv_heads, num_queries_per_kv, dtype, str(device))
    tile = None if cache is None else cache.get(cache_key)
    if tile is None:
        fill = torch.finfo(dtype).min if masked else 0.0
        host = torch.full(
            (num_kv_heads, num_queries_per_kv, 1, ENCODER_BLOCK_SIZE), fill, dtype=dtype
        )
        tile = convert(host, device)
        if cache is not None:
            cache[cache_key] = tile
    return tile


def encoder_mask(
    extent: int,
    kv_len: int,
    num_kv_heads: int,
    num_queries_per_kv: int,
    dtype: torch.dtype,
    device: torch.device,
    cache: dict | None = None,
) -> torch.Tensor:
    """Dense additive mask [num_kv_heads, num_queries_per_kv, 1, extent].

    Built by concatenating 64-wide tiles, not by materialising ``extent``
    fresh elements: interior tiles (entirely real keys) and beyond-the-end
    tiles (entirely padding) are shared constants that ``cache`` hands out by
    reference, so only the one tile straddling ``kv_len`` -- if any -- is
    request-specific. The concatenation itself is a cheap on-device op; this
    is where the mask's ``L``-squared growth disappears.
    """
    num_blocks = extent // ENCODER_BLOCK_SIZE
    tiles: list[torch.Tensor] = []
    for i in range(num_blocks):
        lo = i * ENCODER_BLOCK_SIZE
        if lo + ENCODER_BLOCK_SIZE <= kv_len:
            tiles.append(_const_tile(False, num_kv_heads, num_queries_per_kv, dtype, device, cache))
        elif lo >= kv_len:
            tiles.append(_const_tile(True, num_kv_heads, num_queries_per_kv, dtype, device, cache))
        else:
            host = torch.full(
                (num_kv_heads, num_queries_per_kv, 1, ENCODER_BLOCK_SIZE),
                torch.finfo(dtype).min,
                dtype=dtype,
            )
            host[..., : kv_len - lo] = 0
            tiles.append(convert(host, device))
    return tiles[0] if num_blocks == 1 else torch.cat(tiles, dim=-1)


def dense_sdpa_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_lens: list[int],
    scale: float,
) -> torch.Tensor:
    """Per-sequence eager SDPA on the packed list (probe / unit-test reference)."""
    outs: list[torch.Tensor] = []
    start = 0
    for length in query_lens:
        q = query[start : start + length]
        k = key[start : start + length]
        v = value[start : start + length]
        qh = q.unsqueeze(0).transpose(1, 2)
        kh = k.unsqueeze(0).transpose(1, 2)
        vh = v.unsqueeze(0).transpose(1, 2)
        kwargs: dict = {"is_causal": False, "scale": scale}
        if q.shape[1] != k.shape[1]:
            kwargs["enable_gqa"] = True
        out = F.scaled_dot_product_attention(qh, kh, vh, **kwargs)
        outs.append(out.transpose(1, 2).squeeze(0))
        start += length
    return torch.cat(outs, dim=0)


class SpyreEncoderAttentionImpl(SpyreAttentionImpl):
    """Bidirectional encoder self-attention (no KV cache).

    The platform selects this impl for ENCODER/ENCODER_ONLY layers (see
    ``TorchSpyrePlatform.get_attn_backend_cls``). Forward stays inside the
    opaque ``unified_attention`` op and dispatches gather/attend/store per
    request, with no host-side slicing of activations.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self._compile_attn:
            self._gather_fn = _encoder_gather_compiled
            self._attn_fn = _encoder_dense_attn_compiled
            self._store_fn = _encoder_store_compiled
        else:
            self._gather_fn = _encoder_gather_kernel
            self._attn_fn = _encoder_dense_attn_kernel
            self._store_fn = _encoder_store_kernel
        # Mask tiles are shape-only (depend only on kv_len's block boundary,
        # not on which request/layer/step), so one device copy each serves
        # every sequence, layer and step.
        self._const_tiles: dict[tuple, torch.Tensor] = {}
        self._warmed_buffers: set[int] = set()

    def _run_gather(self, query, key, value, row_index):
        return _call_kernel("encoder_gather", self._gather_fn, query, key, value, row_index)

    def _run_attn(self, q_rows, k_rows, v_rows, mask, num_heads, num_kv_heads, head_size):
        return _call_kernel(
            "encoder_dense_attn",
            self._attn_fn,
            q_rows,
            k_rows,
            v_rows,
            mask,
            self.scale,
            num_heads,
            num_kv_heads,
            head_size,
        )

    def _run_store(self, out, row_index, attn):
        return _call_kernel("encoder_store", self._store_fn, out, row_index, attn)

    def _warm_kernels(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
    ) -> None:
        """Compile every kernel a batch of this token count can ask for.

        A request's padded length follows its own length, not the body bucket,
        so the warmup dummies do not span the ladder on their own -- one long
        request mid-serve would otherwise stall the server compiling. Doing it
        here, on the first forward for each body size, keeps that cost inside
        warmup, where the dummy runs already visit every body size.

        Gather is warmed against the step's own query/key/value, not a fresh
        contiguous scratch tensor: a fused-QKV projection (``qkv.split(...)``)
        hands out a strided view, and under ``dynamic=False`` that stride is
        part of the compiled function's cache key. A contiguous dummy would
        warm a *different* specialization than what a real request's gather
        call actually presents, recompiling the first time one arrives.
        Reusing the real tensors makes the layout match by construction,
        whatever it is. Their values don't matter -- this call's output is
        discarded (or, for store, overwritten by the real per-plan loop right
        after) -- only the shape/stride/dtype reach the cache key.

        Both gather and store are keyed on ``buffer_rows`` (as well as
        ``extent``, for gather), so both are rewarmed every time a new
        ``buffer_rows`` is seen -- unlike attention, whose cache key is
        ``extent`` alone, so calling it again on an already-seen extent is
        just a cache hit, not a second compile.
        """
        buffer_rows = query.shape[0]
        if buffer_rows in self._warmed_buffers:
            return
        self._warmed_buffers.add(buffer_rows)

        dtype, device = query.dtype, query.device
        num_queries_per_kv = num_heads // num_kv_heads
        out = convert(torch.zeros((buffer_rows, num_heads, head_size), dtype=dtype), device)
        index_dtype = encoder_index_dtype(device)

        extent = ENCODER_BLOCK_SIZE
        while extent <= buffer_rows:
            rows = convert(encoder_row_table(0, extent, extent, index_dtype), device)
            q_rows, k_rows, v_rows = self._run_gather(query, key, value, rows)
            mask = encoder_mask(
                extent, extent, num_kv_heads, num_queries_per_kv, dtype, device, self._const_tiles
            )
            attn = self._run_attn(q_rows, k_rows, v_rows, mask, num_heads, num_kv_heads, head_size)
            self._run_store(out, rows, attn)
            extent *= 2

    def _build_plans(
        self,
        attn_metadata: SpyreAttentionMetadata,
        query: torch.Tensor,
        num_kv_heads: int,
        num_queries_per_kv: int,
    ) -> list[EncoderSeqPlan]:
        """Row tables and masks for every request in the step."""
        query_start_loc = attn_metadata.query_start_loc.cpu().tolist()
        seq_lens = attn_metadata.seq_lens.cpu().tolist()
        # The body may 1D-pad past num_actual_tokens; those rows are not a request.
        num_tokens = attn_metadata.num_actual_tokens
        buffer_rows = query.shape[0]
        device = query.device
        index_dtype = encoder_index_dtype(device)

        plans: list[EncoderSeqPlan] = []
        for seq_idx in range(attn_metadata.num_seqs):
            start = int(query_start_loc[seq_idx])
            query_len = int(query_start_loc[seq_idx + 1]) - start
            if start >= num_tokens or query_len <= 0:
                continue
            query_len = min(query_len, num_tokens - start)
            kv_len = min(int(seq_lens[seq_idx]), query_len)

            extent = _blocks_for(max(query_len, kv_len)) * ENCODER_BLOCK_SIZE
            plans.append(
                EncoderSeqPlan(
                    start=start,
                    query_len=query_len,
                    needs_gather=not (start == 0 and query_len == extent and buffer_rows == extent),
                    row_table=convert(
                        encoder_row_table(start, query_len, extent, index_dtype), device
                    ),
                    mask=encoder_mask(
                        extent,
                        kv_len,
                        num_kv_heads,
                        num_queries_per_kv,
                        query.dtype,
                        device,
                        self._const_tiles,
                    ),
                )
            )
        return plans

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

        num_heads = query.shape[1]
        num_kv_heads = key.shape[1]
        head_size = query.shape[2]

        # Everything runs where the result lands. A real step already has all
        # four on the same device; unit tests hand in host activations.
        if query.device != output.device:
            query = convert(query, output.device)
            key = convert(key, output.device)
            value = convert(value, output.device)

        # Folds the per-layer eager store into the attention jobplan. Re-checked
        # per call: vLLM hands out a fresh buffer per layer.
        fused_store_ok = (
            self._compile_attn
            and output.dtype == query.dtype
            # A compiled kernel reads its arguments from offset 0: torch-spyre#3770.
            and output.storage_offset() == 0
            and output.is_contiguous()
        )
        store_mode = "index" if fused_store_ok else "none"

        if store_mode == "index":
            self._warm_kernels(query, key, value, num_heads, num_kv_heads, head_size)

        # Built once per step; the whole encoder stack shares one build.
        if attn_metadata.encoder_seq_plans is None:
            attn_metadata.encoder_seq_plans = self._build_plans(
                attn_metadata, query, num_kv_heads, num_heads // num_kv_heads
            )

        for plan in attn_metadata.encoder_seq_plans:
            if plan.needs_gather:
                q_rows, k_rows, v_rows = self._run_gather(query, key, value, plan.row_table)
            else:
                # index_select always returns a fresh contiguous tensor, so the
                # gather path never hits this -- but a fused QKV projection
                # (qkv.split(...)) hands out strided views, and a compiled
                # region can't resolve that layout for the matmul/reduction
                # that follows. .contiguous() is a no-op when it's already
                # contiguous, so this only ever costs a real copy for fused
                # QKV models. Same fix the old pack path applied for the same
                # reason ("Fused QKV views are strided").
                q_rows, k_rows, v_rows = query.contiguous(), key.contiguous(), value.contiguous()

            attn = self._run_attn(q_rows, k_rows, v_rows, plan.mask, num_heads, num_kv_heads, head_size)

            if store_mode == "index":
                self._run_store(output, plan.row_table, attn)
            else:
                output[plan.start : plan.start + plan.query_len] = attn[: plan.query_len]

        return output


class SpyreEncoderAttentionBackend(SpyreAttentionBackend):
    """Encoder-only (no KV cache) variant of the Spyre backend."""

    # These layers have no KV cache, but vLLM still hands encoder-only specs a
    # zero-filled slot mapping, so upstream must skip `unified_kv_cache_update` entirely.
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_impl_cls() -> type[SpyreEncoderAttentionImpl]:
        return SpyreEncoderAttentionImpl
