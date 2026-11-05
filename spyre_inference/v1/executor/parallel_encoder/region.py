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

# SPDX-License-Identifier: Apache-2.0

"""Shared-memory region sizing and a scheduler-owned block allocator.

The parallel encoder reuses ``ECSharedRegion`` (a ``/dev/shm`` mmap byte grid
where one block holds one token's embedding vector) as the transport for
projected multimodal embeddings.  The scheduler is the single authority for
block-ID allocation; the encoder process and decoder workers only ever
receive explicit block IDs, so there are no cross-process allocation races.
"""

from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.ec_transfer.ec_connector.cpu.ec_shared_region import (
        ECSharedRegion,
    )
    from vllm.multimodal.encoder_budget import MultiModalBudget

    from spyre_inference.config.parallel_encoder import ParallelEncoderConfig

logger = init_logger(__name__)

# Default encoder pipeline depth (in whole mm items) when not configured, so
# the encoder can run ahead of decode instead of stalling on a ~1-2 image
# working set derived from max_num_batched_tokens.
_DEFAULT_INFLIGHT_ITEMS = 16


def resolve_encoder_cache_size(
    mm_budget: "MultiModalBudget",
    pe_config: "ParallelEncoderConfig | None" = None,
) -> int:
    """Return the encoder working-set size (in embeddings).

    This is the shared-region ``num_blocks`` and the scheduler's
    ``EncoderCacheManager.cache_size`` (they must match: one block == one
    embedding slot). All three sites (executor, scheduler, decoder worker)
    call this with the same args so they agree on the region size.

    Sized for the encoder pipeline (how many images can be in-flight +
    pre-encoded), decoupled from the decode-side ``max_num_batched_tokens``.
    This is the analog of a dedicated EPD encoder instance's max_num_seqs.
    """
    if pe_config is not None and pe_config.encoder_cache_tokens:
        return pe_config.encoder_cache_tokens
    per_item = max(mm_budget.mm_max_toks_per_item.values(), default=0)
    return max(mm_budget.encoder_cache_size, _DEFAULT_INFLIGHT_ITEMS * per_item)


def region_engine_id(vllm_config: "VllmConfig") -> str:
    """Region file identity — one region per DP rank (all TP workers share)."""
    dp_rank = vllm_config.parallel_config.data_parallel_rank
    return f"{vllm_config.instance_id}_penc_dp{dp_rank}"


def region_block_size_bytes(vllm_config: "VllmConfig") -> int:
    """Bytes per block == one token embedding (hidden_dim * dtype size).

    Deepstack towers (Qwen3-VL), whose cached rows are wider than the LM's
    inputs, are not supported."""
    import torch

    model_config = vllm_config.model_config
    assert isinstance(model_config.dtype, torch.dtype)
    return model_config.get_inputs_embeds_size() * model_config.dtype.itemsize


def build_region(
    vllm_config: "VllmConfig",
    num_blocks: int,
) -> "ECSharedRegion":
    """Create or attach the parallel-encoder shared region.

    The first process to open the file (``O_CREAT | O_EXCL``) becomes the
    creator/owner-of-unlink; later openers attach. The executor must create
    the region before spawning any worker or the encoder process, so
    attachers never spin waiting for the file to appear.
    """
    from vllm.distributed.ec_transfer.ec_connector.cpu.ec_shared_region import (
        ECSharedRegion,
    )

    return ECSharedRegion(
        engine_id=region_engine_id(vllm_config),
        num_blocks=num_blocks,
        block_size_bytes=region_block_size_bytes(vllm_config),
    )


def is_contiguous(block_ids: list[int]) -> bool:
    """True if ``block_ids`` is an ascending run of consecutive ids.

    Contiguous allocations let the region transport move an image's embedding
    as a single copy instead of one scattered copy per token, which is the
    dominant transport cost. See ``region_io``.
    """
    n = len(block_ids)
    return n > 0 and block_ids[-1] - block_ids[0] + 1 == n


class RegionBlockAllocator:
    """Range allocator over the shared region, keyed by ``mm_hash``.

    Slaved to ``EncoderCacheManager``: allocation happens at dispatch (after
    the cache manager reserves an embedding slot), and freeing is driven by
    ``EncoderCacheManager.get_freed_mm_hashes()`` so the cache manager remains
    the single eviction authority.

    Allocates a **contiguous** block range per ``mm_hash`` whenever a free
    range large enough exists (the common case), so the encoder D2H write and
    decoder H2D read are single copies. If fragmentation prevents a contiguous
    fit, it falls back to a scattered allocation (still correct; the transport
    uses a per-block copy for those) rather than failing.
    """

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        # Disjoint free ranges (start, length), kept sorted by start.
        self._free_ranges: list[tuple[int, int]] = [(0, num_blocks)]
        self._allocated: dict[str, list[int]] = {}

    def num_free(self) -> int:
        return sum(length for _, length in self._free_ranges)

    def is_allocated(self, mm_hash: str) -> bool:
        return mm_hash in self._allocated

    def block_ids(self, mm_hash: str) -> list[int]:
        return self._allocated[mm_hash]

    def alloc(self, mm_hash: str, num_blocks: int) -> list[int]:
        """Allocate ``num_blocks`` blocks for ``mm_hash``.

        Idempotent for an already-allocated hash (returns the existing IDs),
        which lets a repeated identical image reuse the same region bytes.
        Prefers a single contiguous range; falls back to scattered blocks under
        fragmentation. Raises ``MemoryError`` only if total free space is
        insufficient; the caller (scheduler) gates on
        ``EncoderCacheManager.can_allocate`` first so that never trips normally.
        """
        existing = self._allocated.get(mm_hash)
        if existing is not None:
            return existing
        if num_blocks > self.num_free():
            raise MemoryError(
                f"RegionBlockAllocator out of blocks: need {num_blocks}, "
                f"have {self.num_free()} free of {self.num_blocks}"
            )

        # First-fit contiguous range.
        for i, (start, length) in enumerate(self._free_ranges):
            if length >= num_blocks:
                block_ids = list(range(start, start + num_blocks))
                if length == num_blocks:
                    self._free_ranges.pop(i)
                else:
                    self._free_ranges[i] = (start + num_blocks, length - num_blocks)
                self._allocated[mm_hash] = block_ids
                return block_ids

        # Fragmented: gather from multiple ranges (scattered fallback).
        block_ids = []
        need = num_blocks
        new_ranges: list[tuple[int, int]] = []
        for start, length in self._free_ranges:
            if need <= 0:
                new_ranges.append((start, length))
                continue
            take = min(need, length)
            block_ids.extend(range(start, start + take))
            if take < length:
                new_ranges.append((start + take, length - take))
            need -= take
        self._free_ranges = new_ranges
        self._allocated[mm_hash] = block_ids
        return block_ids

    def free(self, mm_hash: str) -> None:
        """Return an mm_hash's blocks to the free pool (idempotent), coalescing
        adjacent free ranges so contiguity is recovered over time."""
        block_ids = self._allocated.pop(mm_hash, None)
        if not block_ids:
            return
        self._insert_free(block_ids)

    def _insert_free(self, block_ids: list[int]) -> None:
        ids = sorted(block_ids)
        # Compress the freed ids into runs.
        freed: list[tuple[int, int]] = []
        run_start = prev = ids[0]
        for b in ids[1:]:
            if b == prev + 1:
                prev = b
            else:
                freed.append((run_start, prev - run_start + 1))
                run_start = prev = b
        freed.append((run_start, prev - run_start + 1))
        # Merge with existing free ranges and coalesce adjacencies.
        merged: list[tuple[int, int]] = []
        for start, length in sorted(self._free_ranges + freed):
            if merged and merged[-1][0] + merged[-1][1] == start:
                merged[-1] = (merged[-1][0], merged[-1][1] + length)
            else:
                merged.append((start, length))
        self._free_ranges = merged

    def reset(self) -> None:
        self._free_ranges = [(0, self.num_blocks)]
        self._allocated.clear()
