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

Attention runs over the packed ``[T, H, D]`` token list; request boundaries ride
in int32 row-index tables, so a card never does offset arithmetic on *shapes* --
offsets are data. Requests sharing a padded length are served by one
gather/attend/store (opt-in, ``SPYRE_ENCODER_BATCHED_ATTN``), since these
kernels are dispatch-bound rather than compute-bound.

Three separately compiled functions, and the split is the point: under
``dynamic=False`` Dynamo keys on every argument's exact shape, so folding the
gather and store into the attention math would make the expensive graph
recompile per ``(buffer_rows, extent)`` instead of per ``extent``. Gather and
store are cheap and keyed on the step's buffer size; the attention math is keyed
only on the sequence's own padded length.

Two torch-spyre bugs shape the design. A compiled region reads its arguments
from offset 0 and ignores ``storage_offset`` (#3770), so a sequence is gathered
with ``index_select`` rather than sliced.

No fallbacks: there is no on-device ``arange`` or ``full``, so every index and
mask tensor here is built on the host, ``convert``'d once, then cached and
reused across sequences, layers and steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from vllm.v1.attention.backend import AttentionLayer

from spyre_inference import envs
from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionBackend,
    SpyreAttentionImpl,
    SpyreAttentionMetadata,
    SpyrePagedKVCache,
    _call_kernel,
    note_unattributed_compiles,
)

# One Spyre stick of fp16, in tokens. A padded length that is a multiple of this
# keeps the row-index table stick-aligned and the matmul's contraction dimension
# aligned, and it is the width of a shared mask tile. Encoder attention has no KV
# cache and so no block walk -- this is alignment, not a block size.
ENCODER_LEN_ALIGNMENT = 64


def _alignment_units_for(length: int) -> int:
    """Stick-aligned units covering ``length``, rounded up to a power of two.

    Encoder self-attention has ``q_len == kv_len``, so this fixes the sequence's
    padded extent (``units * ENCODER_LEN_ALIGNMENT``) -- the attention kernel's
    cache has one shape axis, not two. Rounding to a power of two keeps that axis
    to a handful of buckets, at the cost of padding a request up to the next one:
    a 260-token request attends over 512, not 320.
    """
    units = max(1, (length + ENCODER_LEN_ALIGNMENT - 1) // ENCODER_LEN_ALIGNMENT)
    bucket = 1
    while bucket < units:
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


def _encoder_sdpa_kernel(
    q_rows,
    k_rows,
    v_rows,
    mask,
    scale,
    group,
    num_heads,
    num_kv_heads,
    head_size,
):
    """Masked bidirectional attention over ``group`` sequences of equal padded length.

    ``group == 1`` is the ordinary single-sequence case, so this serves both.

    ``F.scaled_dot_product_attention`` broadcasts the KV heads itself under
    ``enable_gqa``, so operands stay 4-D and no ``num_queries_per_kv`` axis is
    materialised -- which also keeps this clear of torch-spyre's
    ``insert_restickify_padding``, whose assert rejects the interleaved index a
    rank-5 matmul input picks up from ``reshape -> transpose -> reshape``.
    Calling SDPA at all was blocked on torch-spyre#4526, where Inductor rewrote
    a visible ``matmul + mask + softmax + matmul`` into an SDPA-equivalent that
    dropped the additive mask.

    Shapes: q_rows [group * extent, num_heads, head_size]; k_rows/v_rows the
    same with num_kv_heads; mask [group, 1, 1, extent], one row per member. The
    head and query axes broadcast because an encoder mask depends only on the KV
    column -- every query row, real or padding, attends to exactly the real keys.

    Returns [group * extent, num_heads, head_size].
    """
    extent = q_rows.shape[0] // group
    q = q_rows.reshape(group, extent, num_heads, head_size).transpose(1, 2)
    k = k_rows.reshape(group, extent, num_kv_heads, head_size).transpose(1, 2)
    v = v_rows.reshape(group, extent, num_kv_heads, head_size).transpose(1, 2)
    attn = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        scale=scale,
        is_causal=False,
        enable_gqa=(num_heads != num_kv_heads),
    )
    return attn.transpose(1, 2).reshape(group * extent, num_heads, head_size)


_encoder_sdpa_compiled = torch.compile(_encoder_sdpa_kernel, dynamic=False)


def _encoder_store_kernel(out, row_index, attn):
    """Scatter one sequence's attention output back into the step's output buffer.

    Compiled alone -- mirrors the decoder's ``_index_copy_kernel``: a tiny
    mutation, not fused with the attention math. Keyed on ``out.shape[0]``
    (``buffer_rows``), which is fine because this graph is one op.
    """
    out.index_copy_(0, row_index, attn)
    return out


_encoder_store_compiled = torch.compile(_encoder_store_kernel, dynamic=False)


def _encoder_fused_kernel(
    out,
    row_index,
    query,
    key,
    value,
    mask,
    scale,
    group,
    num_heads,
    num_kv_heads,
    head_size,
):
    """Gather, attend and store one group of equal-extent requests, in one graph.

    Every jobplan launch carries its own parameter upload, so launch count is a
    device-path cost, not just host bookkeeping: profiling put attention at 61%
    of the launches in a step. One graph per group per layer instead of three
    removes two thirds of those.

    The copies themselves do not get cheaper -- index_select is charged for the
    source and index_copy_ for the destination, whatever graph they sit in
    (torch-spyre#4239 is fixed only for layout-compliant destinations). This buys
    launch overhead and the intermediate round trips, not copy cost.

    Keyed on ``(out.shape[0], group, extent)``. That carries ``buffer_rows``
    into the expensive attention graph, which the three-way split existed to
    avoid, so it trades warmup for serving throughput.
    """
    q_rows, k_rows, v_rows = _encoder_gather_kernel(query, key, value, row_index)
    attn = _encoder_sdpa_kernel(
        q_rows, k_rows, v_rows, mask, scale, group, num_heads, num_kv_heads, head_size
    )
    out.index_copy_(0, row_index, attn)
    return out


_encoder_fused_compiled = torch.compile(_encoder_fused_kernel, dynamic=False)


def _create_dense_attn_kernel(num_heads: int, num_kv_heads: int, head_size: int):
    """Test helper: bind the non-tensor args the way a forward call would."""

    def specialized_dense_attn_kernel(q_rows, k_rows, v_rows, mask, scale):
        return _encoder_sdpa_kernel(
            q_rows, k_rows, v_rows, mask, scale, 1, num_heads, num_kv_heads, head_size
        )

    return specialized_dense_attn_kernel


@dataclass
class EncoderSeqPlan:
    """What the kernels need for one group of equal-extent requests, per step.

    A group of one is the ordinary single-request case, so this covers both.
    """

    starts: list[int]
    query_lens: list[int]
    extent: int
    row_table: torch.Tensor
    mask: torch.Tensor

    @property
    def group(self) -> int:
        return len(self.starts)


def encoder_index_dtype(device: torch.device) -> torch.dtype:
    """int32 on Spyre, which has no int64 and whose compiled ``index_copy_``
    takes it; int64 everywhere else, where eager ``index_copy_`` rejects int32.
    """
    return torch.int32 if device.type == "spyre" else torch.int64


def encoder_row_table(start: int, query_len: int, extent: int, dtype: torch.dtype) -> torch.Tensor:
    """One sequence's absolute rows, pad lanes clamped to its last real row.

    ``extent`` is a multiple of ``ENCODER_LEN_ALIGNMENT`` and therefore of
    ``INT32_ELEMS_PER_STICK``, so the table is stick-aligned with no extra pad.
    """
    return torch.arange(extent, dtype=dtype).clamp(max=query_len - 1) + start


def encoder_mask(extent: int, kv_len: int, dtype: torch.dtype) -> torch.Tensor:
    """Additive mask ``[1, 1, 1, extent]`` for one sequence, on the host.

    The head and query axes are 1 and left to broadcast: an encoder mask depends
    only on the KV column, since every query row -- real or padding -- attends to
    exactly the real keys.

    Built whole on the host so a group's masks can be concatenated there and reach
    the device in one ``convert``. Assembling it from cached device tiles instead
    left an eager ``cat`` on the device, which torch-spyre compiled once per tile
    pattern -- and warmup could not cover those, because it only ever built masks
    with ``kv_len == extent`` (no partial tile) while real requests always have one.
    """
    mask = torch.full((1, 1, 1, extent), torch.finfo(dtype).min, dtype=dtype)
    mask[..., :kv_len] = 0
    return mask


def _host_pad_head_dim(x: torch.Tensor, padded: int) -> torch.Tensor:
    """Widen the head dim to a whole stick, via the host.

    Below one stick several heads share a stick, and no device op -- compiled or
    eager -- can touch the per-head view that implies ("Unexpected stick
    expression d2 + 32*(Mod(d1, 2))"). ``convert`` is opaque, so the round trip
    is what escapes it; ``F.pad`` and a device ``cat``/``contiguous`` cannot.
    Zeros leave ``QK^T`` unchanged and zero the extra output columns.
    """
    if x.shape[-1] == padded:
        return x
    device = x.device
    on_host = convert(x, "cpu") if device.type == "spyre" else x
    on_host = F.pad(on_host.contiguous(), (0, padded - x.shape[-1]))
    return convert(on_host, device) if device.type == "spyre" else on_host


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
            self._attn_fn = _encoder_sdpa_compiled
            self._store_fn = _encoder_store_compiled
            self._fused_fn = _encoder_fused_compiled
        else:
            self._gather_fn = _encoder_gather_kernel
            self._attn_fn = _encoder_sdpa_kernel
            self._store_fn = _encoder_store_kernel
            self._fused_fn = _encoder_fused_kernel
        # Grouping only pays off through the compiled kernels; in eager mode the
        # per-request loop has no launch overhead to amortise.
        self._batched_attn = self._compile_attn and envs.SPYRE_ENCODER_BATCHED_ATTN
        self._warmed_buffers: set[int] = set()
        # A request's extent cannot exceed its own length, so warming past the
        # model length compiles the most expensive graphs for shapes no request
        # can reach.
        from vllm.config import get_current_vllm_config

        config = get_current_vllm_config()
        self._max_extent = (
            _alignment_units_for(config.model_config.max_model_len) * ENCODER_LEN_ALIGNMENT
        )
        # A group cannot hold more requests than a step can run, so warming past
        # max_num_seqs compiles graphs no plan can dispatch to.
        self._max_group = config.scheduler_config.max_num_seqs

    def _run_gather(self, query, key, value, row_index):
        return _call_kernel("encoder_gather", self._gather_fn, query, key, value, row_index)

    def _run_attn(self, q_rows, k_rows, v_rows, mask, group, num_heads, num_kv_heads, head_size):
        return _call_kernel(
            "encoder_sdpa",
            self._attn_fn,
            q_rows,
            k_rows,
            v_rows,
            mask,
            self.scale,
            group,
            num_heads,
            num_kv_heads,
            head_size,
        )

    def _run_fused(
        self,
        out,
        row_index,
        query,
        key,
        value,
        mask,
        group,
        num_heads,
        num_kv_heads,
        head_size,
    ):
        return _call_kernel(
            "encoder_fused",
            self._fused_fn,
            out,
            row_index,
            query,
            key,
            value,
            mask,
            self.scale,
            group,
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
        output: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
    ) -> None:
        """Compile every kernel a batch of this token count can ask for.

        A request's padded length follows its own length, not the body bucket, so
        the warmup dummies do not span the ladder on their own; doing it here, on
        the first forward per body size, keeps the cost inside warmup.

        Kernels are warmed against the caller's real tensors, never a stand-in: a
        Spyre tensor's device layout is part of the cache key, and a stand-in does
        not reproduce it. Both bit us -- a fused-QKV ``query`` is a strided view,
        and ``output`` arrives as ``torch.empty(rows, H*D).view(-1, H, D)`` whose
        layout differs from a same-shaped ``torch.zeros``. Warming the store
        therefore writes into ``output``, which is safe because the per-plan loop
        right after overwrites every real request's rows and the rest is padding
        the pooler never reads.
        """
        buffer_rows = query.shape[0]
        if buffer_rows in self._warmed_buffers:
            return
        self._warmed_buffers.add(buffer_rows)

        dtype, device = query.dtype, query.device
        index_dtype = encoder_index_dtype(device)

        extent = ENCODER_LEN_ALIGNMENT
        max_extent = min(buffer_rows, self._max_extent)
        while extent <= max_extent:
            rows = convert(encoder_row_table(0, extent, extent, index_dtype), device)
            host_mask = encoder_mask(extent, extent, dtype)
            mask = convert(host_mask, device)
            # The fused graph gathers internally, so warm it on the buffers, not
            # on gathered rows.
            self._run_fused(
                output,
                rows,
                query,
                key,
                value,
                mask,
                1,
                num_heads,
                num_kv_heads,
                head_size,
            )

            if self._batched_attn:
                group = 2
                while group * extent <= buffer_rows and group <= self._max_group:
                    g_rows = convert(
                        torch.cat([encoder_row_table(0, extent, extent, index_dtype)] * group),
                        device,
                    )
                    g_mask = convert(torch.cat([host_mask] * group, dim=0), device)
                    self._run_fused(
                        output,
                        g_rows,
                        query,
                        key,
                        value,
                        g_mask,
                        group,
                        num_heads,
                        num_kv_heads,
                        head_size,
                    )
                    group *= 2
            extent *= 2

    def _build_plans(
        self,
        attn_metadata: SpyreAttentionMetadata,
        query: torch.Tensor,
    ) -> list[EncoderSeqPlan]:
        """Row tables and masks for the step, one plan per group of equal extent."""
        query_start_loc = attn_metadata.query_start_loc.cpu().tolist()
        seq_lens = attn_metadata.seq_lens.cpu().tolist()
        # The body may 1D-pad past num_actual_tokens; those rows are not a request.
        num_tokens = attn_metadata.num_actual_tokens
        buffer_rows = query.shape[0]
        device = query.device
        index_dtype = encoder_index_dtype(device)

        # (start, query_len, kv_len) keyed by extent, in arrival order.
        by_extent: dict[int, list[tuple[int, int, int]]] = {}
        for seq_idx in range(attn_metadata.num_seqs):
            start = int(query_start_loc[seq_idx])
            query_len = int(query_start_loc[seq_idx + 1]) - start
            if start >= num_tokens or query_len <= 0:
                continue
            query_len = min(query_len, num_tokens - start)
            kv_len = min(int(seq_lens[seq_idx]), query_len)
            extent = _alignment_units_for(query_len) * ENCODER_LEN_ALIGNMENT
            by_extent.setdefault(extent, []).append((start, query_len, kv_len))

        def plan(members: list[tuple[int, int, int]], extent: int) -> EncoderSeqPlan:
            return EncoderSeqPlan(
                starts=[m[0] for m in members],
                query_lens=[m[1] for m in members],
                extent=extent,
                row_table=convert(
                    torch.cat(
                        [encoder_row_table(m[0], m[1], extent, index_dtype) for m in members]
                    ),
                    device,
                ),
                # Concatenated, not stacked: member order along dim 0 is what the
                # kernel's batch dim indexes.
                mask=convert(
                    torch.cat([encoder_mask(extent, m[2], query.dtype) for m in members], dim=0),
                    device,
                ),
            )

        plans: list[EncoderSeqPlan] = []
        for extent, members in sorted(by_extent.items()):
            if not self._batched_attn:
                plans.extend(plan([m], extent) for m in members)
                continue
            # Split into descending power-of-two chunks rather than padding up to
            # one: padding 5 members to 8 would attend 3 phantom sequences. Chunks
            # are capped so their row count stays on the ladder warmup covers, which
            # is what lets gather and store reuse their existing graphs.
            cap = 1 << (max(1, buffer_rows // extent).bit_length() - 1)
            offset = 0
            while offset < len(members):
                chunk = min(1 << ((len(members) - offset).bit_length() - 1), cap)
                plans.append(plan(members[offset : offset + chunk], extent))
                offset += chunk
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

        # Anything compiled since the previous kernel call happened in the body
        # (or the pooler, for the first layer of a step) -- attribute it here so
        # it is not silently missed.
        note_unattributed_compiles("model body / pooler")

        num_heads = query.shape[1]
        num_kv_heads = key.shape[1]
        head_size = query.shape[2]

        # Everything runs where the result lands. A real step already has all
        # four on the same device; unit tests hand in host activations.
        if query.device != output.device:
            query = convert(query, output.device)
            key = convert(key, output.device)
            value = convert(value, output.device)

        # Sub-stick head sizes: run the whole attention at a stick-aligned head
        # dim on our own buffers, then narrow on the host and write back through
        # the flattened output view, whose rows are a whole number of sticks.
        head_pad = -head_size % ENCODER_LEN_ALIGNMENT
        narrow_into = None
        if head_pad:
            query = _host_pad_head_dim(query, head_size + head_pad)
            key = _host_pad_head_dim(key, head_size + head_pad)
            value = _host_pad_head_dim(value, head_size + head_pad)
            narrow_into, output = (
                output,
                convert(
                    torch.zeros(
                        (output.shape[0], num_heads, head_size + head_pad), dtype=output.dtype
                    ),
                    output.device,
                ),
            )
            head_size += head_pad

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
            self._warm_kernels(query, key, value, output, num_heads, num_kv_heads, head_size)

        # Built once per step; the whole encoder stack shares one build.
        if attn_metadata.encoder_seq_plans is None:
            attn_metadata.encoder_seq_plans = self._build_plans(attn_metadata, query)

        for plan in attn_metadata.encoder_seq_plans:
            if store_mode == "index":
                # One graph for the whole plan: gather, attend, store.
                self._run_fused(
                    output,
                    plan.row_table,
                    query,
                    key,
                    value,
                    plan.mask,
                    plan.group,
                    num_heads,
                    num_kv_heads,
                    head_size,
                )
                continue

            q_rows, k_rows, v_rows = self._run_gather(query, key, value, plan.row_table)
            attn = self._run_attn(
                q_rows,
                k_rows,
                v_rows,
                plan.mask,
                plan.group,
                num_heads,
                num_kv_heads,
                head_size,
            )
            for i, (start, query_len) in enumerate(zip(plan.starts, plan.query_lens)):
                base = i * plan.extent
                output[start : start + query_len] = attn[base : base + query_len]

        if narrow_into is not None:
            rows = narrow_into.shape[0]
            on_host = convert(output, "cpu")[..., : narrow_into.shape[-1]].contiguous()
            narrow_into.reshape(rows, -1).copy_(
                convert(on_host.reshape(rows, -1), narrow_into.device)
            )
            return narrow_into
        return output


class SpyreEncoderAttentionBackend(SpyreAttentionBackend):
    """Encoder-only (no KV cache) variant of the Spyre backend."""

    # These layers have no KV cache, but vLLM still hands encoder-only specs a
    # zero-filled slot mapping, so upstream must skip `unified_kv_cache_update` entirely.
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_impl_cls() -> type[SpyreEncoderAttentionImpl]:
        return SpyreEncoderAttentionImpl
