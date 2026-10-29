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

"""One kernel for a whole mixed-length ragged-path batch, adapted from the jagged
paged-attention design of spyre-inference#1118 (``jagged_page_attn_kernel``) for
encoder self-attention.

The ragged path's existing kernels (``spyre_encoder_attn.py``) dispatch one fused
gather/attend/scatter per *equal-extent group*, so a batch whose sequences have N
distinct padded lengths pays N kernel launches. This module instead chunks every
sequence's own query and key axes into ``ENCODER_JAGGED_CHUNK``-wide tiles and flattens
every (query chunk, key chunk) pair from every sequence into one "visit" list, then
walks all visits from the whole batch in a single ``for_each_tile`` loop with an
online-softmax carry that resets at each query chunk's first visit. One compiled
kernel call then serves the whole batch regardless of how many distinct sequence
lengths it contains, and regardless of batch composition up to the declared visit
capacity (padded with no-op visits past the real count).

There is no KV cache or page table here: a visit reads its key/value chunk directly
out of the step's own packed body buffer, since encoder attention has the whole
sequence available in one forward pass. Only ``causal=False`` is implemented: encoder
self-attention has no notion of a future token to mask out, so every key chunk within
a sequence's real length is visible to every query chunk of that sequence.

Every visit writes its (possibly still-partial) result straight to its query chunk's
real output rows, unconditionally -- no sink-row redirect for a non-final visit, unlike
``jagged_page_attn_kernel``. Nothing reads the accumulation buffer until the whole scan
finishes, and a query chunk's visits are consecutive in the flattened list (by
construction), so a later key-chunk visit of the same chunk simply overwrites an
earlier one's partial value; only the last visit's write survives. This also sidesteps
a device-layout limit an earlier sink-row design hit (indirect index rows need adjacent
physical sticks across loop trips, which the sink lane's row arithmetic broke).
"""

import torch

from spyre_inference.v1.attention.ops.tile_loop import walk_tiles

# Matches jagged_plan.py's FP16_ELEMS_PER_STICK / INT32_ELEMS_PER_STICK and this
# backend's own ENCODER_LEN_ALIGNMENT: a visit chunk is one stick wide on both axes.
ENCODER_JAGGED_CHUNK = 64


def encoder_jagged_attn_kernel(
    query,
    key,
    value,
    query_row_index,
    kv_row_index,
    valid_keys,
    first_visit,
    scale,
    num_heads,
    num_kv_heads,
    head_size,
    out=None,
):
    """Online-softmax attention over flattened per-sequence chunk visits.

    Expected shapes, all tables built on the host by ``build_encoder_jagged_plan``:
        query, key, value: ``[rows, H, D]`` / ``[rows, KVH, D]``, the step's packed body.
        query_row_index: ``[num_visits, ENCODER_JAGGED_CHUNK]`` int32, this visit's real
            query rows -- gathers Q and scatters the (possibly partial) result every
            visit; see the module docstring for why an unconditional overwrite is safe.
        kv_row_index: ``[num_visits, ENCODER_JAGGED_CHUNK]`` int32, absolute rows this
            visit's keys/values come from. Any in-bounds placeholder past the real
            column count on a padding column; ``valid_keys`` is the real gate.
        valid_keys: ``[num_visits, ENCODER_JAGGED_CHUNK]`` fp16, 1.0 for a real key
            column, 0.0 for padding (including every column of a wholly-padding visit).
        first_visit: ``[num_visits, ENCODER_JAGGED_CHUNK]`` fp16, every column holding
            the same 0/1 flag (broadcast across a whole stick, matching
            ``jagged_page_attn_kernel``'s ``first_page``): 1.0 on a query chunk's first
            key-chunk visit (resets the online-softmax carry), 0.0 otherwise.
        scale: softmax scale.
        out: buffer to store into, or None to return the result instead.

    Returns ``[rows, H, D]`` (or writes into ``out`` and returns it).
    """
    group_size = num_heads // num_kv_heads
    rows = query.shape[0]
    chunk = query_row_index.shape[1]
    # Rank matches jagged_page_attn_kernel's state_shape exactly (no trailing
    # singleton dim on the reduction state): a stick axis of size 1 gave
    # torch-spyre's restickify beam search no feasible layout in an earlier attempt.
    state_shape = (num_kv_heads, group_size, chunk)
    state_kwargs = {"dtype": query.dtype, "device": query.device}
    floor = torch.finfo(query.dtype).min

    def visit_body(carry, tiles):
        old_max, old_sum, old_output, destination = carry
        q_ids, kv_ids, gate, first, query, key, value = tiles
        q = query.index_select(0, q_ids[0]).transpose(0, 1)
        q = q.reshape(num_kv_heads, group_size, chunk, head_size)
        # Key/value have one head axis (num_kv_heads), not num_heads, so there is no
        # group_size dim to reshape into -- unsqueeze a broadcast dim for it instead,
        # as jagged_page_attn_kernel does.
        k = key.index_select(0, kv_ids[0]).transpose(0, 1).unsqueeze(1)
        v = value.index_select(0, kv_ids[0]).transpose(0, 1).unsqueeze(1)

        gate_row = gate[0].reshape(1, 1, 1, chunk) * 1.0
        first_row = first[0, 0:1].reshape(1, 1, 1) * 1.0

        tile_max = torch.where(first_row != 0, floor, old_max)
        tile_sum = torch.where(first_row != 0, 0.0, old_sum)
        tile_output = torch.where(first_row.unsqueeze(-1) != 0, 0.0, old_output)

        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = torch.where(gate_row > 0, scores, floor)
        scores_max = scores.amax(dim=-1)
        rescale = torch.exp(-torch.relu(scores_max - tile_max))
        new_max = torch.maximum(tile_max, scores_max)
        probs = torch.where(gate_row > 0, torch.exp(scores - new_max.unsqueeze(-1)), 0.0)
        new_sum = tile_sum * rescale + probs.sum(dim=-1)
        new_output = tile_output * rescale.unsqueeze(-1) + torch.matmul(probs, v)

        # Unclamped, matching page_attn_kernel: every real query chunk's first visit
        # has at least one real key (its own sequence has length >= 1), so new_sum
        # never stays zero past that visit; a later all-padding placeholder visit's
        # all-zero gate leaves new_sum == old_sum (rescale evaluates to 1 against it),
        # not zero, so the division is always well-defined by the time it is read.
        result = new_output / new_sum.unsqueeze(-1)
        result = result.reshape(num_heads, chunk, head_size).transpose(0, 1)
        destination = destination.index_copy(0, q_ids[0], result)
        return (new_max, new_sum, new_output, destination), None

    (*_, destination), _ = walk_tiles(
        visit_body,
        (query_row_index, kv_row_index, valid_keys, first_visit, query, key, value),
        dims=(0, 0, 0, 0, None, None, None),
        tile_size=1,
        init=(
            torch.full(state_shape, floor, **state_kwargs),
            torch.zeros(state_shape, **state_kwargs),
            torch.zeros((*state_shape, head_size), **state_kwargs),
            torch.zeros((rows, num_heads, head_size), **state_kwargs),
        ),
    )
    if out is not None:
        out.copy_(destination)
        return out
    return destination
