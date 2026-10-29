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

"""Host-side table builder for ``encoder_jagged_attn_kernel``.

Replaces the ragged path's by-extent grouping (``build_encoder_plan`` in
``spyre_encoder_attn.py``, which dispatches one kernel per equal-extent group) with a
single flat visit list covering every sequence in the batch regardless of its length,
so one kernel call serves the whole batch. See ``encoder_jagged_attn.py`` for why this
needs no page table: a visit's keys/values are a stick-aligned chunk of the step's own
packed body buffer, not a cache entry.

Only non-causal (bidirectional) encoder attention is covered -- see that module's
docstring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from vllm.utils.math_utils import cdiv

from spyre_inference.v1.attention.ops.encoder_jagged_attn import ENCODER_JAGGED_CHUNK


@dataclass(frozen=True)
class EncoderJaggedPlan:
    """One flat visit table covering every sequence in the step's batch."""

    query_row_index: torch.Tensor
    kv_row_index: torch.Tensor
    valid_keys: torch.Tensor
    first_visit: torch.Tensor
    query_lens: list[int]
    """Real length per request, same convention as EncoderRectPlan/EncoderGroupPlan --
    read by EncoderDispatchStats.record for the dispatch-ratio log line."""


def build_encoder_jagged_plan(
    starts: list[int],
    query_lens: list[int],
    rows: int,
    *,
    visit_capacity: int | None = None,
    index_dtype: torch.dtype = torch.int32,
) -> EncoderJaggedPlan:
    """Flatten every sequence's (query chunk, key chunk) visits into one table.

    Args:
        starts: each sequence's absolute first row in the packed body buffer.
        query_lens: each sequence's real (unpadded) length; its key length too, since
            encoder self-attention has ``q_len == kv_len``.
        rows: the step's packed body row count -- a visit's placeholder rows must stay
            within it so an out-of-bounds gather never happens even on an all-padding
            visit.
        visit_capacity: visit-table rows to pad to, so the table shape (and so the
            compiled kernel) stays fixed across steps with a different real visit
            count but the same declared body. ``None`` sizes exactly to this call's
            real visit count (for a one-off / warmup build).
        index_dtype: table dtype; ``int32`` on Spyre (no int64, compiled index_copy_
            needs it), ``int64`` elsewhere, mirroring ``encoder_index_dtype``.

    Returns the flat table, host-resident (the caller converts to device).
    """
    chunk = ENCODER_JAGGED_CHUNK
    # (q_chunk_start, last_real_row, kv_chunk_start, real_keys, first)
    visits: list[tuple[int, int, int, int, bool]] = []
    for start, length in zip(starts, query_lens):
        if length <= 0:
            continue
        last_real_row = start + length - 1
        num_chunks = cdiv(length, chunk)
        for qc in range(num_chunks):
            q_chunk_start = start + qc * chunk
            for kc in range(num_chunks):
                kv_chunk_start = start + kc * chunk
                real_keys = min(chunk, length - kc * chunk)
                visits.append((q_chunk_start, last_real_row, kv_chunk_start, real_keys, kc == 0))

    real_visits = len(visits)
    required = max(1, real_visits)
    capacity = visit_capacity if visit_capacity is not None else (1 << (required - 1).bit_length())
    if capacity < required:
        raise ValueError(f"visit_capacity={capacity} cannot hold {real_visits} visits")

    lanes = np.arange(chunk)
    # A fallback query chunk for a padding visit (beyond real_visits, up to capacity):
    # the LAST real visit's own query chunk, clamped to its last real row like any
    # other chunk -- the carry past the last real visit holds that chunk's converged
    # softmax, so a padding visit's all-zero gate (set below) leaves the carry
    # unchanged and its write only repeats what the last real visit already wrote
    # (see encoder_jagged_attn_kernel's docstring for why an unconditional overwrite
    # is safe). Never a sink row, so every visit's index stays a plain
    # stick-aligned run -- a different query chunk here would silently corrupt that
    # chunk's output with a stale (not-yet-converged) carry value.
    fallback_q = np.minimum(visits[-1][0] + lanes, visits[-1][1]) if visits else lanes

    q_ids = np.broadcast_to(fallback_q, (capacity, chunk)).copy()
    kv_ids = np.broadcast_to(fallback_q, (capacity, chunk)).copy()
    valid = np.zeros((capacity, chunk), dtype=np.float32)
    first = np.zeros((capacity, chunk), dtype=np.float32)

    for i, (q_start, last_real_row, kv_start, real_keys, is_first) in enumerate(visits):
        # Padding lanes within a real visit (query past this chunk's real count, or
        # key columns past real_keys) are clamped to the sequence's own last real row
        # rather than left to run into the next sequence's rows in the packed buffer --
        # same convention as encoder_row_table's clamp. The softmax gate (valid_keys)
        # is what makes a padding lane harmless, not its row value, but the gather
        # itself must still land on real data in this sequence.
        q_ids[i] = np.minimum(q_start + lanes, last_real_row)
        kv_ids[i] = np.minimum(kv_start + lanes, last_real_row)
        valid[i, :real_keys] = 1.0
        first[i] = 1.0 if is_first else 0.0

    np_dtype = np.int64 if index_dtype == torch.int64 else np.int32
    return EncoderJaggedPlan(
        query_row_index=torch.from_numpy(q_ids.astype(np_dtype)),
        kv_row_index=torch.from_numpy(kv_ids.astype(np_dtype)),
        valid_keys=torch.from_numpy(valid.astype(np.float16)),
        first_visit=torch.from_numpy(first.astype(np.float16)),
        query_lens=[length for length in query_lens if length > 0],
    )
